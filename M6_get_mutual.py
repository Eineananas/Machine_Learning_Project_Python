import itertools
import random
from collections import Counter
import time

def read_data(filename):
    """读取 / 分隔的数据文件"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = [p.strip() for p in line.split('/')]
            try:
                nums = [int(x) for x in parts]
            except: continue
            if 6 <= len(nums) <= 9 and all(1 <= n <= 49 for n in nums):
                data.append(sorted(set(nums)))  # 去重排序
    return data

def avg_overlap(candidate, data_groups):
    """计算候选组合与所有数据组的平均重合数"""
    cset = set(candidate)
    total = sum(len(cset & set(g)) for g in data_groups)
    return total / len(data_groups)

def generate_optimal_groups(data_groups, target_avg=1.0, num_groups=100, tolerance=0.1, max_iter=100000):
    """
    复杂算法：生成 num_groups 个6数字组合，每个与数据组平均重合≈target_avg
    策略：
    1. 优先从出现频率适中的数字中组合
    2. 智能采样+评分排序
    3. 多轮迭代确保质量
    """
    print("🎯 开始智能生成组合...")
    
    # 1. 分析数字频率，优先选“中等频率”数字（重合1个更容易）
    all_nums = [n for g in data_groups for n in g]
    freq = Counter(all_nums)
    sorted_nums = sorted(freq.items(), key=lambda x: x[1])  # 按频率排序
    
    # 选频率中等的数字作为优先池（频率1-3次最合适）
    mid_freq_nums = [n for n, f in sorted_nums if 1 <= f <= 3]
    low_freq_nums = [n for n, f in sorted_nums if f == 1]
    high_freq_nums = [n for n, f in sorted_nums if f > 3]
    
    print(f"数字池: 中频({len(mid_freq_nums)}), 低频({len(low_freq_nums)}), 高频({len(high_freq_nums)})")
    
    best_groups = []
    attempts = 0
    
    while len(best_groups) < num_groups and attempts < max_iter:
        attempts += 1
        
        # 策略性采样：70%中频 + 20%低频 + 10%高频
        pool = (random.choices(mid_freq_nums, k=4) + 
                random.choices(low_freq_nums + high_freq_nums, k=2))
        candidate = sorted(set(pool + random.sample(range(1, 50), 6 - len(set(pool)))))
        
        # 计算评分
        avg_match = avg_overlap(candidate, data_groups)
        score = abs(avg_match - target_avg)
        
        if score < tolerance:
            best_groups.append((candidate, avg_match))
            print(f"✅ #{len(best_groups)}: {candidate} -> {avg_match:.3f}")
        
        if attempts % 10000 == 0:
            print(f"进度: {attempts}/{max_iter}, 已找到{len(best_groups)}组")
    
    return best_groups

def save_results(groups, filename):
    """保存结果"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# 生成的 {len(groups)} 组组合 (平均重合数≈1.0)\n\n")
        for i, (group, avg) in enumerate(groups, 1):
            f.write(f"Group {i}: {', '.join(map(str, group))} (重合: {avg:.3f})\n")
    print(f"✅ 已保存 {len(groups)} 组到 {filename}")

# 主程序
if __name__ == "__main__":
    filename = "C:/Users/b163450/OneDrive - The Chinese University of Hong Kong/draw16.txt"# 你的数据文件
    output_file = "C:/Users/b163450/OneDrive - The Chinese University of Hong Kong/optimal_groups.txt"
    
    try:
        data_groups = read_data(filename)
        print(f"📖 已读取 {len(data_groups)} 组数据")
        print("前3组示例:", data_groups[:3])
        
        # 生成100组组合，要求平均重合数在1.0±0.1内
        start_time = time.time()
        groups = generate_optimal_groups(
            data_groups, 
            target_avg=1.0, 
            num_groups=100, 
            tolerance=0.1,  # abs(avg-1.0) < 0.1
            max_iter=500000
        )
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ 用时: {elapsed:.1f}秒")
        print(f"📊 最终统计:")
        print(f"  生成组数: {len(groups)}")
        print(f"  平均重合数: {sum(g[1] for g in groups)/len(groups):.3f}")
        print(f"  重合范围: {min(g[1] for g in groups):.3f} ~ {max(g[1] for g in groups):.3f}")
        
        save_results(groups, output_file)
        
    except FileNotFoundError:
        print("❌ 未找到 data.txt")
        print("请创建文件并按格式输入你的30组数据：")
        print("1/4/9/15/37/39")
        print("4/5/11/16/28/47")
