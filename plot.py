import matplotlib
# 强制使用 'Agg' 后端，不依赖图形界面，用于保存文件
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt
import os

# --- (其他辅助函数 parse_your_log_file 和 calculate_metrics 保持不变) ---
# 为了简洁，我只显示修改后的部分

def parse_your_log_file(filename="kv_rank.txt"):
    data_list = []
    print(f"正在读取 {filename} ...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            content = line.replace('[', '').replace(']', '')
            try:
                indices = np.fromstring(content, dtype=int, sep=',')
                if len(indices) > 0:
                    data_list.append(indices)
            except ValueError:
                print(f"警告: 第 {i+1} 行格式解析失败，跳过。内容: {line[:30]}...")
                continue
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filename}。请先运行你的 JAX 推理代码生成日志。")
        return None
    return np.array(data_list)

def calculate_metrics(data):
    if data is None or len(data) < 2:
        print("数据不足，无法计算差异。")
        return [], []
    ious = []
    hit_rates = []
    num_steps, k = data.shape
    print(f"检测到 {num_steps} 个时间步，Top-K = {k}")
    for t in range(1, num_steps):
        prev_idx = set(data[t-1])
        curr_idx = set(data[t])
        intersection = len(prev_idx.intersection(curr_idx))
        union = len(prev_idx.union(curr_idx))
        ious.append(intersection / union if union > 0 else 0)
        hit_rates.append(intersection / k)
    return ious, hit_rates

def plot_results(ious, hit_rates):
    if not ious: return

    steps = range(1, len(ious) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # --- 绘图 ---
    plt.plot(steps, ious, label='Attention IoU (Similarity)', color='blue', alpha=0.6)
    plt.plot(steps, hit_rates, label='Cache Hit Rate (Bandwidth Savings)', color='red', linewidth=2)
    
    plt.title("Diffusion Policy KV Cache Temporal Locality Analysis")
    plt.xlabel("Diffusion Time Step (Iteration)")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 标出平均节省率
    avg_hit = np.mean(hit_rates)
    plt.axhline(y=avg_hit, color='green', linestyle='--', label=f'Avg Hit Rate: {avg_hit:.2%}')
    plt.text(0, avg_hit + 0.02, f" Average Savings: {avg_hit:.1%}", color='green', fontweight='bold')
    
    plt.tight_layout()
    
    # !!! 关键修改：保存为 PNG 文件 !!!
    output_filename = "analysis_result.png"
    plt.savefig(output_filename)
    print(f"图表已成功保存到文件: {output_filename}")
    plt.close() # 关闭图形，释放资源

if __name__ == "__main__":
    data = parse_your_log_file("kv_rank.txt")
    
    ious, hit_rates = calculate_metrics(data)
    
    if ious:
        avg_iou = np.mean(ious)
        avg_hit_rate = np.mean(hit_rates)
        
        print("-" * 40)
        print("✅ 核心数据分析结果:")
        print(f"  总时间步 (Total Steps): {len(data)}")
        print(f"  平均 IoU (Similarity): {avg_iou:.4f}")
        print(f"  平均 Cache Hit Rate (Bandwidth Savings): {avg_hit_rate:.4f} ({avg_hit_rate:.1%})")
        print("-" * 40)
    
    plot_results(ious, hit_rates)