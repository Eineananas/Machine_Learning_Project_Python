import pandas as pd
import numpy as np
import re
from scipy.stats import norm
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

stock = "spcx"
SPOT_PRICE = 162.50  # 现货实时价格
RISK_FREE_RATE = 0.045  # 无风险利率设定为 4.5%

files_info = {
    "DTE_0": f"C:/Users/WeiTh/Desktop/option/{stock}_d0.csv",
    "DTE_3": f"C:/Users/WeiTh/Desktop/option/{stock}_d3.csv",
    "DTE_5": f"C:/Users/WeiTh/Desktop/option/{stock}_d5.csv",
    "DTE_7": f"C:/Users/WeiTh/Desktop/option/{stock}_d7.csv"
}
output_file = f"C:/Users/WeiTh/Desktop/option/{stock}_GEX_DEX_Summary_Result.xlsx"


def clean_num(val):
    if pd.isna(val) or val == '' or val == '-' or val == '--':
        return 0.0
    s = str(val).strip().replace(',', '')
    if '万' in s:
        return float(s.replace('万', '')) * 10000
    if '亿' in s:
        return float(s.replace('亿', '')) * 100000000
    if '%' in s:
        try:
            return float(s.replace('%', '')) / 100.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0

def calculate_vanna(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    vanna_raw = -norm.pdf(d1) * (d2 / sigma)
    return vanna_raw * 0.01 * 100 * S  # 1% IV 变动下的 Dollar Vanna

def calculate_charm(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    term = -pdf_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    if option_type == 'call':
        charm_raw = -term
    else:
        charm_raw = -term
        
    # 严格以365天折算每交易日引起的 Dollar Charm 变化
    return charm_raw * (-1.0 / 365.0) * 100 * S


def get_T_year(file_path):
    match = re.search(r"_d(\d+)\.csv$", str(file_path))

    if match is None:
        raise ValueError(
            "文件名必须以 _dN.csv 结尾，例如 options_d3.csv"
        )

    dte_days = int(match.group(1))

    eastern = ZoneInfo("America/New_York")
    t0 = datetime.now(eastern)
    close_today = datetime.combine(
        t0.date(),
        time(16, 0),
        tzinfo=eastern,
    )

    if t0 >= close_today:
        T_rm = 0.0
    else:
        T_rm = (close_today - t0).total_seconds() / 86400.0
    T_years = max(dte_days + T_rm, 0.001) / 365.0

    return T_years

"----------------------------------------------------------------"
"----------------------------------------------------------------"
"----------------------------------------------------------------"
gex_dict = {}
dex_dict = {}
vanna_dict = {}
charm_dict = {}
dte_metrics_list = []

for dte_label, file_path in files_info.items():
    try:
        match = re.search(r'_d(\d+)\.csv$', file_path)
        T_years = get_T_year(file_path)


        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gbk')

        if '行权价' not in df.columns:
            for header_row in range(1, 5):
                try:
                    df_temp = pd.read_csv(file_path, header=header_row, encoding='utf-8-sig')
                except:
                    df_temp = pd.read_csv(file_path, header=header_row, encoding='gbk')
                if '行权价' in df_temp.columns:
                    df = df_temp
                    break

        df.columns = [str(c).strip() for c in df.columns]
        strike = df['行权价'].apply(clean_num)

        strike_col_idx = df.columns.get_loc('行权价')
        df_call = df.iloc[:, :strike_col_idx]
        df_put = df.iloc[:, strike_col_idx + 1:]

        # 精准匹配左右两侧各自的列名
        c_oi = df_call['未平仓数'].apply(clean_num) if '未平仓数' in df_call.columns else pd.Series(0, index=df.index)
        c_vol = df_call['成交量'].apply(clean_num) if '成交量' in df_call.columns else pd.Series(0, index=df.index)
        c_gamma = df_call['Gamma'].apply(clean_num) if 'Gamma' in df_call.columns else pd.Series(0, index=df.index)
        c_delta = df_call['Delta'].apply(clean_num) if 'Delta' in df_call.columns else pd.Series(0, index=df.index)
        c_iv = df_call['隐含波动率'].apply(clean_num) if '隐含波动率' in df_call.columns else pd.Series(0.3, index=df.index)

        p_oi = df_put['未平仓数.1'].apply(clean_num) if '未平仓数.1' in df_put.columns else pd.Series(0, index=df.index)
        p_vol = df_put['成交量.1'].apply(clean_num) if '成交量.1' in df_put.columns else pd.Series(0, index=df.index)
        p_gamma = df_put['Gamma.1'].apply(clean_num) if 'Gamma.1' in df_put.columns else pd.Series(0, index=df.index)
        p_delta = df_put['Delta.1'].apply(clean_num) if 'Delta.1' in df_put.columns else pd.Series(0, index=df.index)
        p_iv = df_put['隐含波动率.1'].apply(clean_num) if '隐含波动率.1' in df_put.columns else pd.Series(0.3, index=df.index)

        # ✨ 核心升级：按每个 DTE 独立计算 Max Pain 和 PCR
        c_vol_sum = c_vol.sum()
        p_vol_sum = p_vol.sum()
        c_oi_sum = c_oi.sum()
        p_oi_sum = p_oi.sum()

        vol_pcr = p_vol_sum / c_vol_sum if c_vol_sum > 0 else 0.0
        oi_pcr = p_oi_sum / c_oi_sum if c_oi_sum > 0 else 0.0

        # 计算当前 DTE 档位的 Max Pain
        pain_results = []
        for test_strk in strike:
            if test_strk <= 0:
                continue
            call_payout = np.maximum(0, test_strk - strike) * c_oi
            put_payout = np.maximum(0, strike - test_strk) * p_oi
            total_pain = (call_payout + put_payout).sum()
            pain_results.append((test_strk, total_pain))

        dte_max_pain = min(pain_results, key=lambda x: x[1])[0] if pain_results else 0.0

        dte_metrics_list.append({
            'DTE': dte_label,
            'Max_Pain_Strike': dte_max_pain, 
            'Call_Volume': c_vol_sum,
            'Put_Volume': p_vol_sum,
            'Volume_PCR': vol_pcr,
            'Call_OI': c_oi_sum,
            'Put_OI': p_oi_sum,
            'OI_PCR': oi_pcr
        })

        # 计算 Vanna 和 Charm
        call_vanna_list, put_vanna_list = [], []
        call_charm_list, put_charm_list = [], []

        for K, c_iv_val, p_iv_val in zip(strike, c_iv, p_iv):
            if K <= 0:
                call_vanna_list.append(0)
                put_vanna_list.append(0)
                call_charm_list.append(0)
                put_charm_list.append(0)
                continue
            
            iv_c = c_iv_val if c_iv_val > 0.05 else 0.30
            iv_p = p_iv_val if p_iv_val > 0.05 else 0.30

            call_vanna_list.append(calculate_vanna(SPOT_PRICE, K, T_years, RISK_FREE_RATE, iv_c))
            put_vanna_list.append(calculate_vanna(SPOT_PRICE, K, T_years, RISK_FREE_RATE, iv_p))
            
            call_charm_list.append(calculate_charm(SPOT_PRICE, K, T_years, RISK_FREE_RATE, iv_c, 'call'))
            put_charm_list.append(calculate_charm(SPOT_PRICE, K, T_years, RISK_FREE_RATE, iv_p, 'put'))

        # 金融 Exposure 计算
        call_gex = c_oi * c_gamma * 100 * SPOT_PRICE
        put_gex = p_oi * p_gamma * 100 * SPOT_PRICE
        net_gex = call_gex - put_gex

        call_dex = c_oi * c_delta * 100 * SPOT_PRICE
        put_dex = p_oi * p_delta * 100 * SPOT_PRICE
        net_dex = call_dex + put_dex

        call_vanna = c_oi * np.array(call_vanna_list)
        put_vanna = p_oi * np.array(put_vanna_list)
        net_vanna = call_vanna - put_vanna

        call_charm = c_oi * np.array(call_charm_list)
        put_charm = p_oi * np.array(put_charm_list)
        net_charm = call_charm - put_charm

        res = pd.DataFrame({
            'Strike': strike,
            'Net_GEX': net_gex,
            'Net_DEX': net_dex,
            'Net_Vanna': net_vanna,
            'Net_Charm': net_charm
        }).groupby('Strike').sum()

        gex_dict[dte_label] = res['Net_GEX']
        dex_dict[dte_label] = res['Net_DEX']
        vanna_dict[dte_label] = res['Net_Vanna']
        charm_dict[dte_label] = res['Net_Charm']

    except Exception as e:
        print(f"✕ 处理 {file_path} 出错: {e}")

# 汇总表格并按 Strike 降序排列
gex_summary = pd.DataFrame(gex_dict).fillna(0).sort_index(ascending=False)
dex_summary = pd.DataFrame(dex_dict).fillna(0).sort_index(ascending=False)
vanna_summary = pd.DataFrame(vanna_dict).fillna(0).sort_index(ascending=False)
charm_summary = pd.DataFrame(charm_dict).fillna(0).sort_index(ascending=False)

gex_summary['Total_GEX'] = gex_summary.sum(axis=1)
dex_summary['Total_DEX'] = dex_summary.sum(axis=1)
vanna_summary['Total_Vanna'] = vanna_summary.sum(axis=1)
charm_summary['Total_Charm'] = charm_summary.sum(axis=1)

# 生成按 DTE 独立拆分的 Max Pain 与 PCR 面板
max_pain_pcr_summary = pd.DataFrame(dte_metrics_list)

# ==========================================
# Excel 导出与渲染引擎
# ==========================================
color_styles = {
    'pos_high':  {'fill': PatternFill(start_color="267326", end_color="267326", fill_type="solid"), 'font': Font(color="FFFFFF", bold=True)},
    'pos_mid':   {'fill': PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid"), 'font': Font(color="000000")},
    'pos_low':   {'fill': PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid"), 'font': Font(color="006100")},
    'neg_high':  {'fill': PatternFill(start_color="B22222", end_color="B22222", fill_type="solid"), 'font': Font(color="FFFFFF", bold=True)},
    'neg_mid':   {'fill': PatternFill(start_color="E66161", end_color="E66161", fill_type="solid"), 'font': Font(color="FFFFFF")},
    'neg_low':   {'fill': PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid"), 'font': Font(color="9C0006")},
    'default':   {'fill': PatternFill(fill_type=None), 'font': Font(color="000000")}
}

def style_excel_sheet(ws, is_matrix=True):
    ws.views.sheetView[0].showGridLines = True
    header_fill = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    if is_matrix:
        for row in ws.iter_rows(min_row=2, min_col=2, max_row=ws.max_row, max_col=ws.max_column):
            for cell in row:
                val = cell.value
                if val is None or pd.isna(val) or val == 0:
                    style = color_styles['default']
                else:
                    abs_val = abs(val)
                    if val > 0:
                        style = color_styles['pos_high'] if abs_val >= 1_000_000 else (color_styles['pos_mid'] if abs_val >= 100_000 else color_styles['pos_low'])
                    else:
                        style = color_styles['neg_high'] if abs_val >= 1_000_000 else (color_styles['neg_mid'] if abs_val >= 100_000 else color_styles['neg_low'])
                
                cell.fill = style['fill']
                cell.font = style['font']
                cell.number_format = '#,##0.00'
                cell.alignment = Alignment(horizontal="right", vertical="center")

        for cell in ws['A'][1:]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")
    else:
        for row in ws.iter_rows(min_row=2, min_col=1, max_row=ws.max_row, max_col=ws.max_column):
            for cell in row:
                cell.alignment = Alignment(horizontal="center" if cell.column == 1 else "right", vertical="center")
                if isinstance(cell.value, (int, float)):
                    cell.number_format = '#,##0.00'

    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = col[0].column_letter
        ws.column_dimensions[col_letter].width = max(max_len + 4, 18)

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    gex_summary.to_excel(writer, sheet_name='GEX_Summary')
    dex_summary.to_excel(writer, sheet_name='DEX_Summary')
    vanna_summary.to_excel(writer, sheet_name='Vanna_Summary')
    charm_summary.to_excel(writer, sheet_name='Charm_Summary')
    max_pain_pcr_summary.to_excel(writer, sheet_name='MaxPain_PCR_Summary', index=False)
    
    style_excel_sheet(writer.sheets['GEX_Summary'], True)
    style_excel_sheet(writer.sheets['DEX_Summary'], True)
    style_excel_sheet(writer.sheets['Vanna_Summary'], True)
    style_excel_sheet(writer.sheets['Charm_Summary'], True)
    style_excel_sheet(writer.sheets['MaxPain_PCR_Summary'], False)

print(f"\n🎉 完美更新！按 DTE 独立拆分的 Max Pain 与 PCR 面板及全套衍生品矩阵已写入：\n{output_file}")
