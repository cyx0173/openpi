import numpy as np
import pathlib

# 请修改为你的实际路径
TASK_DIR = pathlib.Path("data/libero/results_2/task_00")
BITS = 4  # 如果你跑的是 8-bit，请改成 8

def verify_quantization():
    a16_path = TASK_DIR / "actions_16b.npy"
    aq_path = TASK_DIR / f"actions_{BITS}b.npy"
    
    if not a16_path.exists() or not aq_path.exists():
        print(f"❌ 找不到数据文件，请确认模型前向推理部分是否已经跑完！")
        return
        
    a16 = np.load(a16_path)
    aq = np.load(aq_path)
    
    print("=" * 60)
    print(f"📊 【全新 {BITS}-bit W4A16 量化合理性验证】")
    print("=" * 60)
    
    # 1. 计算总体的欧氏距离误差 e_t
    # 动作空间前7维: [X, Y, Z, Roll, Pitch, Yaw, Gripper]
    e_t = np.linalg.norm(aq[:, :7] - a16[:, :7], axis=1)
    print(f"📈 核心指标 e_t 统计:")
    print(f"  -> 平均 e_t: {np.mean(e_t):.4f}  (期望值: 0.02 ~ 0.15 之间)")
    print(f"  -> 最大 e_t: {np.max(e_t):.4f}  (期望值: 绝对不应超过 0.4)")
    
    print("\n" + "=" * 60)
    print("🔍 各维度最大绝对误差 (Max Absolute Error):")
    dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]
    max_errors = np.max(np.abs(a16[:, :7] - aq[:, :7]), axis=0)
    
    for i in range(7):
        # 如果是 4-bit，单维度误差在 0.1 左右是正常的，超过 0.3 就有翻车风险
        flag = "🔴 依然偏高!" if max_errors[i] > 0.3 else "🟢 完美符合物理规律"
        print(f"  {dim_names[i]:<7}: {max_errors[i]:.4f} {flag}")
        
    print("\n" + "=" * 60)
    print("🔬 抽查第 10 步具体数值 (看看是不是还在瞎抽搐):")
    step = 10
    if step < len(a16):
        for i in range(7):
            diff = abs(a16[step][i] - aq[step][i])
            print(f"  {dim_names[i]:<7} | BF16: {a16[step][i]:>8.4f} | {BITS}-bit: {aq[step][i]:>8.4f} | 差值: {diff:.4f}")

if __name__ == "__main__":
    verify_quantization()