#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pprint

pp = pprint.PrettyPrinter(compact=True, depth=3)

# 脚本所在目录
root = Path(__file__).resolve().parent
records = root.glob("step_*.npy")

for f in sorted(records, key=lambda p: int(p.stem.split('_')[1])):
    data: dict = np.load(f, allow_pickle=True).item()

    state   = data.get("inputs/state")
    actions = data.get("outputs/actions")
    infer_ms = data.get("outputs/policy_timing/infer_ms")

    print(f"\n=== {f.name} ===")
    print("state:", state)
    if actions is not None:
        print("actions shape:", actions.shape)
        # 如需打印动作内容可取消下行注释
        # pp.pprint(actions)
    else:
        print("actions: <None>")

    print("infer time:", infer_ms, "ms")