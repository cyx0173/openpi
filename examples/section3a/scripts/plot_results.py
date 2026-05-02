"""
plot_results.py -- Visualization for Section III-A results.
==========================================================

Generates the key figures from DyQ-VLA Section III-A:
  1. Temporal sensitivity profile s_t per task
  2. Decision error e_t per task
  3. Success rate vs e_t (threshold curve)
  4. Fine-grained vs coarse success rate comparison
  5. Per-task detailed figures
  6. Aggregate summary across all tasks

Usage:
  python examples/section3a/scripts/plot_results.py \
    --results_dir data/libero/results \
    --out_dir data/section3a/results/figures \
    --bits 4
"""

import argparse
import json
import logging
import pathlib
import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("plot")

# Paper-quality color palette
COLORS = {
    "bf16": "#1a1a2e",          # deep navy
    "quant": "#e94560",          # vivid red
    "fine": "#f7c948",           # amber gold
    "coarse": "#0f3460",         # steel blue
    "success": "#06d6a0",         # mint green
    "fail": "#ef476f",           # coral red
    "e_t": "#118ab2",            # ocean blue
    "s_t": "#073b4c",            # deep teal
    "bg": "#f8f9fa",             # off-white
    "grid": "#dee2e6",            # light gray
    "text": "#212529",           # near black
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": COLORS["grid"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


# ===========================================================================
# Data loading helpers
# ===========================================================================

def load_all_tasks(results_dir: pathlib.Path, bits: int):
    """Load all task summaries from the results directory."""
    task_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("task_")
    )
    tasks = []
    for td in task_dirs:
        summary_path = td / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            data = json.load(f)
        e_t = None
        s_t = None
        a16 = td / "actions_16b.npy"
        aq = td / f"actions_{bits}b.npy"
        if a16.exists() and aq.exists():
            a16_data = np.load(a16)
            aq_data = np.load(aq)
            act_dim = 7
            e_t = np.linalg.norm(aq_data[:, :act_dim] - a16_data[:, :act_dim], axis=1)
            T = len(e_t)
            s_t = np.zeros(T, dtype=np.float32)
            for t in range(T):
                window_end = min(t + 1, T)
                window_start = max(0, window_end - 20)
                if window_end - window_start >= 2 and e_t[t] > 1e-9:
                    pos_window = a16_data[window_start:window_end, :3]
                    D_T = sum(
                        np.linalg.norm(pos_window[i] - pos_window[i - 1])
                        for i in range(1, len(pos_window))
                    )
                    s_t[t] = D_T / e_t[t]
        tasks.append({
            "name": td.name,
            "summary": data,
            "e_t": e_t,
            "s_t": s_t,
        })
    logger.info(f"Loaded {len(tasks)} tasks from {results_dir}")
    return tasks


def load_perturb_results(task_dir: pathlib.Path):
    """Load all perturbation results for a task."""
    perturb_files = sorted(task_dir.glob("perturb_t*.json"))
    results = []
    for pf in perturb_files:
        with open(pf) as f:
            results.append(json.load(f))
    return results


# ===========================================================================
# Individual task figure
# ===========================================================================

def plot_task_figure(task_data: dict, perturb_results: list, task_dir: pathlib.Path, bits: int):
    """Generate a detailed 4-panel figure for one task."""
    summary = task_data["summary"]
    task_name = task_data["name"]
    T_summary = summary["T"]

    # 修复维度不匹配：对齐 summary 记录的步数和 action 数组的实际长度
    T = min(T_summary, len(task_data["e_t"]))
    
    e_t = task_data["e_t"][:T]
    s_t = task_data["s_t"][:T]

    is_fine = np.array(summary["is_fine"])[:T] if summary.get("is_fine") else np.zeros(T, dtype=bool)
    m_t = np.array(summary.get("m_t", [0.0] * T_summary))[:T]
    j_t = np.array(summary.get("j_t", [0.0] * T_summary))[:T]
    num_perturb = summary.get("num_perturb_steps", len(perturb_results))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Task: {task_name} | T={T} | {bits}-bit W4A4", fontsize=14, fontweight="bold")

    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

    t_axis = np.arange(T)

    # ---- Panel 1: e_t temporal profile ----
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t_axis, e_t, color=COLORS["e_t"], lw=2, label="$e_t$")
    ax1.axhline(e_t.mean(), color=COLORS["e_t"], lw=1, ls="--", alpha=0.6, label=f"mean={e_t.mean():.4f}")
    fine_mask = is_fine
    coarse_mask = ~is_fine
    if fine_mask.any():
        ax1.scatter(t_axis[fine_mask], e_t[fine_mask], color=COLORS["fine"], s=30, zorder=5,
                    label="Fine-grained")
    if coarse_mask.any():
        ax1.scatter(t_axis[coarse_mask], e_t[coarse_mask], color=COLORS["coarse"], s=30, zorder=5,
                    label="Coarse", alpha=0.7)
    ax1.set_xlabel("Time Step $t$")
    ax1.set_ylabel("Decision Error $e_t$")
    ax1.set_title("Decision Error $e_t = ||a_t^{W4A4} - a_t^{BF16}||_2$")
    ax1.legend(loc="upper right")

    # ---- Panel 2: e_t histogram ----
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(e_t, bins=30, color=COLORS["e_t"], alpha=0.7, edgecolor="white")
    ax2.axvline(e_t.mean(), color="red", lw=1.5, ls="--", label=f"mean={e_t.mean():.4f}")
    ax2.set_xlabel("$e_t$")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Distribution")
    ax2.legend(fontsize=8)

    # ---- Panel 3: s_t temporal profile ----
    ax3 = fig.add_subplot(gs[1, :2])
    nonzero = s_t > 1e-9
    ax3.plot(t_axis, s_t, color=COLORS["s_t"], lw=2, label="$s_t$")
    ax3.scatter(t_axis[nonzero], s_t[nonzero], color=COLORS["s_t"], s=20, zorder=5)
    ax3.set_xlabel("Time Step $t$")
    ax3.set_ylabel("Sensitivity $s_t$")
    ax3.set_title("Sensitivity $s_t = D_T / e_t$ (distance-window / error)")
    ax3.legend(loc="upper right")

    # ---- Panel 4: Motion fineness ----
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(t_axis, m_t, color=COLORS["coarse"], lw=1.5, label="$m_t$")
    ax4.fill_between(t_axis, 0, m_t, alpha=0.3, color=COLORS["coarse"])
    ax4.axhline(np.median(m_t), color="red", lw=1, ls="--", label=f"median={np.median(m_t):.3f}")
    ax4.set_xlabel("Time Step $t$")
    ax4.set_ylabel("Motion Fineness $m_t$")
    ax4.set_title("Motion Fineness Proxy")
    ax4.legend(fontsize=8)

    # ---- Panel 5: Perturbation success map ----
    ax5 = fig.add_subplot(gs[2, :2])
    p_steps = [r["perturb_step"] for r in perturb_results]
    p_e = [r["e_t"] for r in perturb_results]
    p_s = [1.0 if r["success"] else 0.0 for r in perturb_results]
    colors_p = [COLORS["success"] if s > 0.5 else COLORS["fail"] for s in p_s]
    ax5.scatter(p_steps, p_e, c=colors_p, s=60, zorder=5, edgecolors="white", lw=0.5)
    ax5.set_xlabel("Perturbation Step $t$")
    ax5.set_ylabel("Error $e_t$")
    ax5.set_title("Perturbation Result at Each Step (Success=Green, Fail=Red)")
    success_patch = mpatches.Patch(color=COLORS["success"], label="Success")
    fail_patch = mpatches.Patch(color=COLORS["fail"], label="Fail")
    ax5.legend(handles=[success_patch, fail_patch], loc="upper right")

    # ---- Panel 6: Aggregate bar ----
    ax6 = fig.add_subplot(gs[2, 2])
    agg = summary.get("aggregate", {})
    fine_sr = agg.get("fine_sr", 0.0)
    coarse_sr = agg.get("coarse_sr", 0.0)
    bars = ax6.bar(["Fine-grained", "Coarse"], [fine_sr, coarse_sr],
                   color=[COLORS["fine"], COLORS["coarse"]], edgecolor="white", lw=1)
    ax6.set_ylabel("Success Rate")
    ax6.set_title("Success Rate by Phase")
    ax6.set_ylim(0, 1.1)
    ax6.bar_label(bars, fmt="%.1%%", fontsize=9)
    ax6.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)

    out_path = task_dir / f"figure_bits{bits}.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ===========================================================================
# Aggregate figures
# ===========================================================================

def plot_aggregate_figures(tasks: list, out_dir: pathlib.Path, results_dir: pathlib.Path, bits: int):
    """Generate aggregate figures across all tasks."""

    # ---- Figure 1: Mean e_t and s_t across tasks ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Aggregate: {len(tasks)} Tasks | {bits}-bit W4A4", fontsize=14, fontweight="bold")

    min_T = min(len(t["e_t"]) for t in tasks)
    e_t_all = np.stack([t["e_t"][:min_T] for t in tasks])
    s_t_all = np.stack([t["s_t"][:min_T] for t in tasks])
    t_axis = np.arange(min_T)

    # Mean ± std for e_t
    e_mean = e_t_all.mean(axis=0)
    e_std = e_t_all.std(axis=0)
    axes[0].plot(t_axis, e_mean, color=COLORS["e_t"], lw=2)
    axes[0].fill_between(t_axis, e_mean - e_std, e_mean + e_std,
                         color=COLORS["e_t"], alpha=0.15)
    axes[0].set_xlabel("Time Step $t$")
    axes[0].set_ylabel("Decision Error $e_t$")
    axes[0].set_title("Mean Decision Error $\\mathbb{E}[e_t]$ Across Tasks")

    # Mean ± std for s_t
    s_mean = s_t_all.mean(axis=0)
    s_std = s_t_all.std(axis=0)
    nonzero_mask = s_mean > 1e-9
    axes[1].plot(t_axis[nonzero_mask], s_mean[nonzero_mask],
                 color=COLORS["s_t"], lw=2)
    axes[1].fill_between(t_axis[nonzero_mask],
                          s_mean[nonzero_mask] - s_std[nonzero_mask],
                          s_mean[nonzero_mask] + s_std[nonzero_mask],
                          color=COLORS["s_t"], alpha=0.15)
    axes[1].set_xlabel("Time Step $t$")
    axes[1].set_ylabel("Sensitivity $s_t$")
    axes[1].set_title("Mean Sensitivity $\\mathbb{E}[s_t]$ Across Tasks")

    out = out_dir / f"aggregate_e_s_bits{bits}.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"  Saved: {out}")

    # ---- Figure 2: Fine-grained vs Coarse success rate ----
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"Success Rate: Fine-grained vs Coarse | {bits}-bit", fontsize=13, fontweight="bold")

    fine_s = sum(t["summary"]["aggregate"]["fine_success"] for t in tasks)
    fine_t = sum(t["summary"]["aggregate"]["fine_total"] for t in tasks)
    coarse_s = sum(t["summary"]["aggregate"]["coarse_success"] for t in tasks)
    coarse_t = sum(t["summary"]["aggregate"]["coarse_total"] for t in tasks)
    fine_sr = fine_s / fine_t if fine_t else 0.0
    coarse_sr = coarse_s / coarse_t if coarse_t else 0.0

    x = np.arange(2)
    widths = 0.5
    bars = ax.bar(x, [fine_sr, coarse_sr], width=widths,
                  color=[COLORS["fine"], COLORS["coarse"]], edgecolor="white", lw=1.5)
    ax.bar_label(bars, fmt="%.1%%", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fine-grained\n({fine_s}/{fine_t})", f"Coarse\n({coarse_s}/{coarse_t})"])
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)

    delta = coarse_sr - fine_sr
    ax.annotate(
        f"$\\Delta$ = {delta:+.1%}",
        xy=(0.5, max(fine_sr, coarse_sr) + 0.05),
        ha="center", fontsize=12,
        color="red" if delta > 0 else "green",
        fontweight="bold",
    )

    out = out_dir / f"fine_coarse_sr_bits{bits}.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"  Saved: {out}")

    # ---- Figure 3: Success rate vs e_t (threshold curve) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Success Rate vs Decision Error Threshold | {bits}-bit", fontsize=13, fontweight="bold")

    if not tasks:
        plt.close(fig)
        return

    # Collect all (e_t, success) pairs
    all_pairs = []
    for task in tasks:
        task_dir = results_dir / task["name"]
        perturb = load_perturb_results(task_dir)
        for r in perturb:
            all_pairs.append((r["e_t"], r["success"]))

    if not all_pairs:
        plt.close(fig)
        return

    e_vals = np.array([p[0] for p in all_pairs])
    s_vals = np.array([float(p[1]) for p in all_pairs])

    # Sort by e_t and compute cumulative success rate
    sort_idx = np.argsort(e_vals)
    e_sorted = e_vals[sort_idx]
    s_sorted = s_vals[sort_idx]

    thresholds = np.linspace(e_vals.min(), e_vals.max(), 100)
    sr_curve = []
    for thresh in thresholds:
        mask = e_vals <= thresh
        if mask.any():
            sr_curve.append(s_vals[mask].mean())
        else:
            sr_curve.append(np.nan)

    ax.plot(thresholds, sr_curve, color=COLORS["quant"], lw=2.5, label="Success Rate")
    ax.scatter(e_vals[s_vals > 0.5], s_vals[s_vals > 0.5] * 0.95,
               color=COLORS["success"], s=15, alpha=0.4, label="Success")
    ax.scatter(e_vals[s_vals <= 0.5], s_vals[s_vals <= 0.5] * 0.05,
               color=COLORS["fail"], s=15, alpha=0.4, label="Fail")
    ax.set_xlabel("Decision Error Threshold $e_t$")
    ax.set_ylabel("Success Rate")
    ax.legend(loc="lower left")

    out = out_dir / f"sr_vs_et_bits{bits}.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"  Saved: {out}")

    # ---- Figure 4: s_t temporal heatmap across tasks ----
    max_T = max(len(t["s_t"]) for t in tasks)
    s_t_matrix = np.zeros((len(tasks), max_T))
    for i, t in enumerate(tasks):
        s_t_matrix[i, :len(t["s_t"])] = t["s_t"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(tasks) * 0.4)))
    fig.suptitle(f"Sensitivity $s_t$ Heatmap Across Tasks | {bits}-bit", fontsize=13, fontweight="bold")
    im = ax.imshow(s_t_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Time Step $t$")
    ax.set_ylabel("Task")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t["name"] for t in tasks], fontsize=8)
    plt.colorbar(im, ax=ax, label="$s_t$")

    out = out_dir / f"s_t_heatmap_bits{bits}.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot Section III-A results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing task_* subdirectories with summary.json")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for figures")
    parser.add_argument("--bits", type=int, nargs="+", default=[4],
                        help="Bit-widths to plot (default: 4)")
    parser.add_argument("--skip_tasks", action="store_true",
                        help="Skip per-task figures (only plot aggregates)")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bits in args.bits:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Plotting {bits}-bit results")
        logger.info(f"{'='*60}")

        tasks = load_all_tasks(results_dir, bits)

        if not tasks:
            logger.warning(f"No tasks found in {results_dir}")
            continue

        if not args.skip_tasks:
            for task_data in tqdm.tqdm(tasks, desc="Task figures"):
                task_dir = results_dir / task_data["name"]
                perturb = load_perturb_results(task_dir)
                plot_task_figure(task_data, perturb, task_dir, bits)

        plot_aggregate_figures(tasks, out_dir, results_dir, bits)

    logger.info(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
