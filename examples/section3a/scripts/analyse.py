import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

# ==========================================
# 1. 配置路径与阈值参数
# ==========================================
# 修改为你的实际结果目录路径（指向跑完的 task_00）
TASK_DIR = pathlib.Path("data/libero/results_real/task_00") 

# 论文中 DyQ-VLA 的报警阈值
THETA_FP = 0.5
# 真实危险降临的阈值 (s_t 突然飙升的水位线，可根据实际 s_t 规模微调)
S_T_DANGER_THRESHOLD = 3.0 

# ==========================================
# 2. 运动学指标计算函数 (复刻论文公式)
# ==========================================
def compute_motion_fineness(actions: np.ndarray) -> np.ndarray:
    if len(actions) == 0: return np.array([])
    xyz = actions[:, :3]
    trans_mag = np.linalg.norm(xyz, axis=1)
    mu_max = np.percentile(trans_mag, 95) if len(trans_mag) > 0 else 1e-6
    mu_max = max(mu_max, 1e-6)
    return 1.0 - trans_mag / mu_max

def compute_angular_jerk(actions: np.ndarray) -> np.ndarray:
    if len(actions) < 2: return np.zeros(len(actions))
    rot = actions[:, 3:6]
    delta_rot = np.diff(rot, axis=0)
    delta_mag = np.linalg.norm(delta_rot, axis=1)
    nu_max = np.percentile(delta_mag, 95) if len(delta_mag) > 0 else 1e-6
    nu_max = max(nu_max, 1e-6)
    j_t = np.zeros(len(actions))
    j_t[1:] = delta_mag / nu_max
    return j_t

# ==========================================
# 3. 主分析逻辑
# ==========================================
def main():
    print(f"Analyzing Task Directory: {TASK_DIR}")
    
    # 读取数据
    s_t_path = TASK_DIR / "s_t.npy"
    actions_path = TASK_DIR / "actions_16b.npy"
    
    if not s_t_path.exists() or not actions_path.exists():
        print("❌ 找不到数据文件，请检查路径是否正确！")
        return

    s_t = np.load(s_t_path)
    actions = np.load(actions_path)
    T = len(s_t)
    
    # 计算运动学指标
    m_t = compute_motion_fineness(actions)
    j_t = compute_angular_jerk(actions)
    
    # 模拟 DyQ-VLA 的窗口平滑与融合逻辑 (W_macro=10, W_micro=5, lambda=0.5)
    proxy_S_t = np.zeros(T)
    for t in range(T):
        w_mac_start = max(0, t - 10 + 1)
        w_mic_start = max(0, t - 5 + 1)
        
        m_t_smooth = np.mean(m_t[w_mac_start:t+1]) if t >=0 else 0
        j_t_smooth = np.mean(j_t[w_mic_start:t+1]) if t >=0 else 0
        
        # 融合敏感度代理指标
        proxy_S_t[t] = max(0, 0.5 * m_t_smooth + 0.5 * j_t_smooth)

    # ==========================================
    # 4. 寻找拐点 (Inflection Points)
    # ==========================================
    # 寻找真实危险降临点 T_true (s_t 首次突破危险阈值)
    T_true_idx = np.where(s_t > S_T_DANGER_THRESHOLD)[0]
    T_true = T_true_idx[0] if len(T_true_idx) > 0 else None

    # 寻找系统滞后报警点 T_dyq (代理指标首次突破 0.5)
    T_dyq_idx = np.where(proxy_S_t > THETA_FP)[0]
    T_dyq = T_dyq_idx[0] if len(T_dyq_idx) > 0 else None

    print("-" * 50)
    if T_true is not None and T_dyq is not None:
        lag = T_dyq - T_true
        print(f"🎯 真实危险降临点 (T_true) : 第 {T_true} 步")
        print(f"🚨 运动学滞后报警点 (T_dyq): 第 {T_dyq} 步")
        print(f"⚠️ 算法级滞后盲区 (Lag Δt) : {lag} 步 !!!")
        if lag > 0:
            print(f"   -> 结论：DyQ-VLA 在危险发生后，盲目飞行了 {lag} 步才反应过来！")
        else:
            print("   -> 结论：在这个任务中，运动学指标凑巧提前或同时报警了。")
    else:
        print("未检测到明显的危险突变或报警，可能是这个任务一直处于粗粒度移动或全程简单。")
    print("-" * 50)

    # ==========================================
    # 5. 绘制绝杀对比图
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 绘制真实的敏感度 s_t (左 Y 轴)
    color = 'tab:red'
    ax1.set_xlabel('Time Step $t$', fontsize=12)
    ax1.set_ylabel('True Sensitivity $s_t$ (Oracle)', color=color, fontsize=12)
    ax1.plot(s_t, color=color, alpha=0.8, linewidth=2, label='True Need ($s_t$)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 绘制 DyQ-VLA 的代理指标 (右 Y 轴)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('DyQ-VLA Proxy Sensitivity $\mathcal{S}_t$', color=color, fontsize=12)
    ax2.plot(proxy_S_t, color=color, linestyle='--', linewidth=2, label='Heuristic Proxy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 标出阈值线
    ax2.axhline(y=THETA_FP, color='gray', linestyle=':', alpha=0.7, label='Alarm Threshold (0.5)')

    # 标出拐点竖线
    if T_true is not None:
        ax1.axvline(x=T_true, color='red', linestyle='-', linewidth=2)
        ax1.text(T_true+1, ax1.get_ylim()[1]*0.8, f'$T_{{true}}={T_true}$', color='red', fontweight='bold')
        
    if T_dyq is not None:
        ax2.axvline(x=T_dyq, color='blue', linestyle='-', linewidth=2)
        ax2.text(T_dyq+1, ax2.get_ylim()[1]*0.5, f'$T_{{dyq}}={T_dyq}$', color='blue', fontweight='bold')

    # 高亮滞后盲区
    if T_true is not None and T_dyq is not None and T_dyq > T_true:
        ax1.axvspan(T_true, T_dyq, color='orange', alpha=0.3, label=f'Lag Blind Spot ($\Delta t={T_dyq - T_true}$)')

    fig.suptitle(f'Oracle vs. Proxy: The Lag Blind Spot in DyQ-VLA (Task 00)', fontsize=14, fontweight='bold')
    
    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    out_fig = TASK_DIR / "lag_analysis.png"
    plt.savefig(out_fig, dpi=300)
    print(f"📈 可视化图表已保存至: {out_fig}")

if __name__ == "__main__":
    main()