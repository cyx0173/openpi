# Section III-A: Step-wise Quantization Sensitivity Analysis

**DyQ-VLA (arXiv:2603.07904) — "Step-wise Perturbation Analysis of Quantization Sensitivity"**

This experiment faithfully reproduces Section III-A of the DyQ-VLA paper.

---

## What the experiment measures

For each time step `t` in a successful BF16 trajectory, we measure:

1. **Decision error** `e_t = ||a_t^{W4A4} - a_t^{BF16}||_2` — how much the quantized model deviates from the BF16 baseline at step `t`
2. **Sensitivity** `s_t = D_T / e_t` — the ratio of distance traveled in a local window to the decision error
3. **Success/failure** when the quantized model recovers from perturbation at step `t`

The key finding: coarse movements (reaching) tolerate large `e_t`, while fine-grained manipulations (grasping/precision) have low tolerance and fail when `e_t` exceeds a threshold.

---

## Pipeline

### Stage 1 — Record BF16 trajectories

Collect successful trajectories using the BF16 policy server.

```bash
# Terminal 1: start policy server
python scripts/serve_policy.py --env LIBERO --port 8000

# Terminal 2: collect trajectories
python examples/section3a/scripts/collect_trajectories.py \
  --port 8000 \
  --out_dir data/section3a/trajectories \
  --task_suite libero_spatial \
  --num_episodes 10
```

Each episode is saved as `trajectory.npy`:
- `frames`: (T, 224, 224, 3) uint8
- `states`: (T, 7) float32
- `actions`: (T, 7) float32
- `prompt`: str
- `success`: bool

### Stage 2 — Quantization perturbation experiment

For each task and each perturbation step `t`:
1. Steps 0..t-1: **W4A4 quantized model** controls the environment (real env feedback, quantized inference, quantized action)
2. Step t: inject **BF16 action** instead of the quantized one
3. Steps t+1..T: **W4A4 quantized model** resumes control

```bash
python examples/section3a/scripts/experiment.py \
  --trajectory_dir data/section3a/trajectories \
  --checkpoint /share/chengyuxuan-local/openpi/pi05_libero_pytorch/model.safetensors \
  --out_dir data/section3a/results \
  --perturb_bits 4 \
  --num_samples 20 \
  --device cuda:0
```

Output per task:
- `actions_16b.npy` — BF16 model actions (all T steps)
- `actions_4b.npy` — W4A4 model actions (all T steps)
- `e_t.npy` — decision error at each step
- `s_t.npy` — sensitivity at each step
- `perturb_t{XXX}.json` — result of perturbation at step XXX
- `summary.json` — per-task aggregated results

### Plot results

```bash
python examples/section3a/scripts/plot_results.py \
  --results_dir data/section3a/results \
  --out_dir data/section3a/results/figures \
  --bits 4
```

Generates:
- Per-task 4-panel figures: `e_t`, `s_t`, perturbation success map, fine/coarse SR
- Aggregate figures: mean `e_t`/`s_t`, fine vs coarse SR comparison, SR vs `e_t` threshold curve, `s_t` heatmap

---

## Experiment design (paper-exact)

```
For each task, for each step t:
┌─────────────────────────────────────────────────────────────┐
│  Step 0..t-1:  quantized model → action → env feedback     │
│                (state diverges from recorded BF16 traj)     │
│                                                             │
│  Step t:      inject BF16 action (perturbation/reset)       │
│                                                             │
│  Step t+1..T: quantized model resumes → action → env        │
│                (continues from whatever state t left)       │
└─────────────────────────────────────────────────────────────┘
```

This is different from naive single-step injection because the quantized model's accumulated error propagates through the environment before the BF16 correction at step `t`.

---

## File structure

```
examples/section3a/
├── scripts/
│   ├── collect_trajectories.py   # Stage 1: record BF16 trajectories
│   ├── experiment.py             # Stage 2: perturbation experiment
│   └── plot_results.py          # Plotting
└── README.md                     # This file

data/section3a/
├── trajectories/                  # Stage 1 output
│   └── task_XX/
│       └── trajectory.npy
└── results/                       # Stage 2 output
    ├── task_XX/
    │   ├── actions_16b.npy
    │   ├── actions_4b.npy
    │   ├── e_t.npy
    │   ├── s_t.npy
    │   ├── perturb_t{XXX}.json
    │   ├── summary.json
    │   └── figure_bits4.png
    └── figures/                   # Stage 3 output
        ├── aggregate_e_s_bits4.png
        ├── fine_coarse_sr_bits4.png
        ├── sr_vs_et_bits4.png
        └── s_t_heatmap_bits4.png
```
