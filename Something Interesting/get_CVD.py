import time
from futu import *

# 1. 全局变量：用于在内存中实时累加 CVD (成交量 Delta)
cvd_volume = 0 
cvd_amount = 0.0  # 如果你想看资金额度 (Dollar CVD)，也可以同步累加

class CVDTickerTest(TickerHandlerBase):
    def on_recv_rsp(self, rsp_pb):
        global cvd_volume, cvd_amount
        
        ret_code, data = super(CVDTickerTest, self).on_recv_rsp(rsp_pb)
        if ret_code != RET_OK:
            print("TickerTest: error, msg: %s" % data)
            return RET_ERROR, data
        
        # 遍历接收到的每一笔逐笔成交
        for index, row in data.iterrows():
            ticker_time = row['time']
            price = row['price']
            volume = row['volume']
            direction = row['ticker_direction']  # BUY / SELL / NEUTRAL
            
            # 计算方向权值 (Delta)
            if direction == 'BUY':
                delta_vol = volume
                delta_amt = volume * price
            elif direction == 'SELL':
                delta_vol = -volume
                delta_amt = -volume * price
            else:
                delta_vol = 0
                delta_amt = 0.0
            
            # 2. 累加到全局 CVD 中
            cvd_volume += delta_vol
            cvd_amount += delta_amt
            
            # 3. 实时把 CVD 打印出来
            print(f"[{ticker_time}] 逐笔: {direction} {volume}股 @ {price} | "
                  f"单笔Delta: {delta_vol:+d} | "
                  f"🔥 实时累积 CVD (股数): {cvd_volume:+d} | "
                  f"💰 实时累积 CVD (金额): ${cvd_amount:+,.2f}")
            
        return RET_OK, data

if __name__ == '__main__':
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    handler = CVDTickerTest()
    
    # 设置实时逐笔成交回调
    quote_ctx.set_handler(handler)
    
    # 注意：算 CVD 必须订阅 SubType.TICKER (逐笔成交)
    ret, data = quote_ctx.subscribe(['US.AAPL'], [SubType.TICKER])
    
    if ret == RET_OK:
        print("订阅 AAPL 逐笔成交成功！正在实时计算并输出 CVD... (盘中按 Ctrl+C 退出)")
        try:
            # 使用主循环保持连接，盘中只要有成交，就会实时触发回调并打印 CVD
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n手动停止监听...")
    else:
        print('订阅失败:', data) 

    # 安全关闭连接
    quote_ctx.unsubscribe(['US.AAPL'], [SubType.TICKER])
    quote_ctx.close()
    print("连接已安全关闭。")
