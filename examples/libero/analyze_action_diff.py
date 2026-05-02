#!/usr/bin/env python3
"""
分析 FP16 vs W4A4 action 差异。
用法: python analyze_action_diff.py [--dir <combined_json_dir>] [--output <output_dir>]

python examples/libero/analyze_action_diff.py \
  --dir /home/chengyuxuan/vla/openpi/data/libero/videos/quant_w4a4 \
  --output /home/chengyuxuan/vla/openpi/data/libero/videos/quant_w4a4/analysis

"""

import argparse
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

ACTION_NAMES = [
    "pos_x", "pos_y", "pos_z",
    "rot_x", "rot_y", "rot_z",
    "gripper"
]
ACTION_COMPONENTS = {
    "position": (0, 3),
    "rotation": (3, 6),
    "gripper": (6, 7),
}


def load_combined(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def per_step_rmse(fp16_action, w4a4_action) -> float:
    diff = np.array(fp16_action) - np.array(w4a4_action)
    return float(np.sqrt(np.mean(diff ** 2)))


def per_dim_errors(fp16_action, w4a4_action) -> np.ndarray:
    diff = np.array(fp16_action) - np.array(w4a4_action)
    return np.abs(diff)


def analyze_trajectory(data: dict) -> dict:
    steps = data["steps"]
    results = {
        "task_description": data["task_description"],
        "task_id": data["task_id"],
        "success": data["success"],
        "total_steps": len(steps),
        "steps_with_w4a4": 0,
    }

    step_rmses = []
    dim_errors = {name: [] for name in ACTION_NAMES}

    for s in steps:
        if s.get("w4a4_action") is None:
            continue
        results["steps_with_w4a4"] += 1
        fp16_a = s["fp16_action"]
        w4a4_a = s["w4a4_action"]

        step_rmses.append(per_step_rmse(fp16_a, w4a4_a))
        errs = per_dim_errors(fp16_a, w4a4_a)
        for i, name in enumerate(ACTION_NAMES):
            dim_errors[name].append(float(errs[i]))

    step_rmses = np.array(step_rmses)

    results["per_step_rmse"] = {
        "mean": float(np.mean(step_rmses)),
        "std": float(np.std(step_rmses)),
        "min": float(np.min(step_rmses)),
        "max": float(np.max(step_rmses)),
        "median": float(np.median(step_rmses)),
        "per_step": [float(x) for x in step_rmses.tolist()],
    }

    results["per_dimension"] = {}
    for name in ACTION_NAMES:
        errs = np.array(dim_errors[name])
        results["per_dimension"][name] = {
            "mean": float(np.mean(errs)),
            "std": float(np.std(errs)),
            "max": float(np.max(errs)),
        }

    results["per_component"] = {}
    for comp_name, (start, end) in ACTION_COMPONENTS.items():
        comp_rmses = []
        for s in steps:
            if s.get("w4a4_action") is None:
                continue
            fp16_a = np.array(s["fp16_action"])[start:end]
            w4a4_a = np.array(s["w4a4_action"])[start:end]
            diff = fp16_a - w4a4_a
            comp_rmses.append(float(np.sqrt(np.mean(diff ** 2))))
        results["per_component"][comp_name] = {
            "mean_rmse": float(np.mean(comp_rmses)),
            "std_rmse": float(np.std(comp_rmses)),
            "max_rmse": float(np.max(comp_rmses)),
        }

    return results


def plot_comparison(data: dict, results: dict, output_dir: pathlib.Path):
    steps_with_w4a4 = [s for s in data["steps"] if s.get("w4a4_action") is not None]
    if not steps_with_w4a4:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    step_indices = list(range(len(steps_with_w4a4)))
    fp16_actions = np.array([s["fp16_action"] for s in steps_with_w4a4])
    w4a4_actions = np.array([s["w4a4_action"] for s in steps_with_w4a4])

    # Plot each dimension
    colors = plt.cm.tab10.colors
    for i, name in enumerate(ACTION_NAMES):
        axes[0].plot(step_indices, fp16_actions[:, i], linestyle="-",  color=colors[i % 10], label=f"FP16 {name}", linewidth=1.5)
        axes[0].plot(step_indices, w4a4_actions[:, i], linestyle="--", color=colors[i % 10], label=f"W4A4 {name}", linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel("Action value")
    axes[0].set_title(f"FP16 vs W4A4 Actions Over Time\n{data['task_description']}")
    axes[0].legend(ncol=2, fontsize=7, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Per-step RMSE
    rmses = results["per_step_rmse"]["per_step"]
    axes[1].plot(step_indices, rmses, color="tab:red", linewidth=1.5)
    axes[1].axhline(results["per_step_rmse"]["mean"], color="tab:red", linestyle="--", label=f"Mean RMSE: {results['per_step_rmse']['mean']:.4f}")
    axes[1].fill_between(step_indices, 0, rmses, alpha=0.2, color="tab:red")
    axes[1].set_ylabel("Per-step RMSE")
    axes[1].set_title("Per-Step RMSE Between FP16 and W4A4 Actions")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Per-dimension absolute error
    for i, name in enumerate(ACTION_NAMES):
        errs = results["per_dimension"][name]["mean"] + np.zeros(len(step_indices))
        # use per-step error instead of mean
    per_step_errs = np.abs(fp16_actions - w4a4_actions)
    for i, name in enumerate(ACTION_NAMES):
        axes[2].plot(step_indices, per_step_errs[:, i], label=name, linewidth=1.5)
    axes[2].set_ylabel("Absolute Error")
    axes[2].set_xlabel("Step")
    axes[2].set_title("Per-Dimension Absolute Error (|FP16 - W4A4|)")
    axes[2].legend(ncol=4, fontsize=8, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    task_seg = data["task_description"].replace(" ", "_")[:50]
    fig_path = output_dir / f"{task_seg}_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {fig_path}")


def plot_summary_bar(all_results: list, output_dir: pathlib.Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    tasks = [r["task_description"][:30] for r in all_results]
    comp_names = list(ACTION_COMPONENTS.keys())

    # Per-component mean RMSE
    comp_data = {c: [] for c in comp_names}
    for r in all_results:
        for c in comp_names:
            comp_data[c].append(r["per_component"][c]["mean_rmse"])
    x = np.arange(len(tasks))
    width = 0.25
    for i, c in enumerate(comp_names):
        axes[0].bar(x + i * width, comp_data[c], width, label=c)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(tasks, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Mean RMSE")
    axes[0].set_title("Per-Component Mean RMSE by Task")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Per-dimension mean absolute error
    dim_names = ACTION_NAMES
    dim_data = {d: [] for d in dim_names}
    for r in all_results:
        for d in dim_names:
            dim_data[d].append(r["per_dimension"][d]["mean"])
    width2 = 0.12
    for i, d in enumerate(dim_names):
        axes[1].bar(x + i * width2, dim_data[d], width2, label=d)
    axes[1].set_xticks(x + 2.5 * width2)
    axes[1].set_xticklabels(tasks, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].set_title("Per-Dimension Mean Abs Error by Task")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Overall per-step RMSE boxplot
    all_rmses = [np.array(r["per_step_rmse"]["per_step"]) for r in all_results]
    axes[2].boxplot(all_rmses, labels=[t[:15] for t in tasks], patch_artist=True)
    axes[2].set_ylabel("Per-step RMSE")
    axes[2].set_title("Per-Step RMSE Distribution by Task")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = output_dir / "summary_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSummary plot saved: {fig_path}")


def print_table(results: dict):
    print(f"\n  Task: {results['task_description']} (id={results['task_id']})")
    print(f"  Success: {results['success']}, Steps: {results['total_steps']}, "
          f"W4A4 steps: {results['steps_with_w4a4']}")
    print(f"  {'Component':<12} {'Mean RMSE':>12} {'Std RMSE':>12} {'Max RMSE':>12}")
    print(f"  {'-'*50}")
    for comp, stats in results["per_component"].items():
        print(f"  {comp:<12} {stats['mean_rmse']:>12.6f} {stats['std_rmse']:>12.6f} {stats['max_rmse']:>12.6f}")
    print(f"  {'-'*50}")
    print(f"  {'Dim':<12} {'Mean AbsErr':>12} {'Std':>12} {'Max':>12}")
    print(f"  {'-'*50}")
    for name in ACTION_NAMES:
        d = results["per_dimension"][name]
        print(f"  {name:<12} {d['mean']:>12.6f} {d['std']:>12.6f} {d['max']:>12.6f}")
    print(f"  Overall per-step RMSE: mean={results['per_step_rmse']['mean']:.6f}  "
          f"std={results['per_step_rmse']['std']:.6f}  "
          f"max={results['per_step_rmse']['max']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze FP16 vs W4A4 action differences")
    parser.add_argument("--dir", type=str,
                       default="/home/chengyuxuan/vla/openpi/data/libero/videos/quant_w4a4",
                       help="Directory containing combined JSON files")
    parser.add_argument("--output", type=str,
                       default="/home/chengyuxuan/vla/openpi/data/libero/videos/quant_w4a4/analysis",
                       help="Output directory for analysis results")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.dir)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(data_dir.glob("*_combined.json"))
    if not json_paths:
        print(f"No combined JSON files found in {data_dir}")
        return

    print(f"Found {len(json_paths)} combined JSON files")
    print(f"Output directory: {output_dir}")

    all_results = []
    for path in json_paths:
        print(f"\nProcessing: {path.name}")
        data = load_combined(path)
        results = analyze_trajectory(data)
        all_results.append(results)
        print_table(results)
        plot_comparison(data, results, output_dir)

    if len(all_results) > 1:
        plot_summary_bar(all_results, output_dir)

    # Save aggregate results
    agg_path = output_dir / "aggregate_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregate results saved: {agg_path}")

    # Print aggregate table
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY (all tasks)")
    print("=" * 80)
    print(f"  {'Task':<35} {'Pos RMSE':>10} {'Rot RMSE':>10} {'Grip RMSE':>10} {'Overall':>10}")
    print(f"  {'-'*77}")
    for r in all_results:
        task = r["task_description"][:33]
        pos_rmse = r["per_component"]["position"]["mean_rmse"]
        rot_rmse = r["per_component"]["rotation"]["mean_rmse"]
        grip_rmse = r["per_component"]["gripper"]["mean_rmse"]
        overall_rmse = r["per_step_rmse"]["mean"]
        print(f"  {task:<35} {pos_rmse:>10.6f} {rot_rmse:>10.6f} {grip_rmse:>10.6f} {overall_rmse:>10.6f}")
    print(f"  {'-'*77}")
    all_pos = np.mean([r["per_component"]["position"]["mean_rmse"] for r in all_results])
    all_rot = np.mean([r["per_component"]["rotation"]["mean_rmse"] for r in all_results])
    all_grip = np.mean([r["per_component"]["gripper"]["mean_rmse"] for r in all_results])
    all_ov = np.mean([r["per_step_rmse"]["mean"] for r in all_results])
    print(f"  {'MEAN ACROSS TASKS':<35} {all_pos:>10.6f} {all_rot:>10.6f} {all_grip:>10.6f} {all_ov:>10.6f}")


if __name__ == "__main__":
    main()

