"""
dyq_bit.py -- DyQ-VLA (arXiv:2603.07904) dynamic bit-width allocation.

Pipeline:
  actions
    -> M_t (Motion Fineness), J_t (Angular Jerk)          [Section III-B]
    -> S_t = fused sensitivity (kinematic proxy)           [Section IV-A2, Algorithm 1 line 1]
    -> b_hat_t (target bit-width)                          [Algorithm 1 line 2 / Eq. 6]
    -> b_star_t (dispatched bit-width after hysteresis)    [Algorithm 1 lines 3-10]

Usage:
  python examples/libero/dyq_bit.py \
    --trajectory_dir data/recording/libero_10 \
    --out_dir data/analysis/dyq_bits \
    --calibration_dir data/recording/libero_10_cal
"""

import argparse
import json
import pathlib

import numpy as np


# --------------------------------------------------------------------------
# Section III-B: Kinematic Proxy Computation
# --------------------------------------------------------------------------


def compute_motion_fineness(actions: np.ndarray) -> np.ndarray:
    """
    Motion Fineness: M_t = 1 - ||a_t^xyz||_2 / mu_max  (paper Section III-B)

    mu_max = 95th percentile of all ||a_i^xyz||_2.
    Higher M_t = finer (slower) movement = higher sensitivity.
    """
    if len(actions) == 0:
        return np.array([])

    xyz = actions[:, :3]
    trans_mag = np.linalg.norm(xyz, axis=1)
    mu_max = float(np.percentile(trans_mag, 95)) if len(trans_mag) > 0 else 1e-6
    mu_max = max(mu_max, 1e-6)
    return 1.0 - trans_mag / mu_max


def compute_angular_jerk(actions: np.ndarray) -> np.ndarray:
    """
    Angular Jerk: J_t = ||a_t^rot - a_{t-1}^rot||_2 / nu_max  (paper Section III-B)

    nu_max = 95th percentile of all ||a_i^rot - a_{i-1}^rot||_2.
    Higher J_t = rapid direction change = higher sensitivity.
    """
    if len(actions) < 2:
        return np.zeros(len(actions)) if len(actions) > 0 else np.array([])

    rot = actions[:, 3:6]
    delta_mag = np.linalg.norm(np.diff(rot, axis=0), axis=1)
    nu_max = float(np.percentile(delta_mag, 95)) if len(delta_mag) > 0 else 1e-6
    nu_max = max(nu_max, 1e-6)
    j_t = np.zeros(len(actions))
    j_t[1:] = delta_mag / nu_max
    return j_t


# --------------------------------------------------------------------------
# Section IV-A2: Asymmetric Temporal Smoothing
# --------------------------------------------------------------------------


def smooth_trailing(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Trailing (backward-looking) moving average -- paper notation.

    M_tilde = (1/Wmacro) * sum_{i=t-Wmacro+1}^{t} M_i
    J_tilde = (1/Wmicro) * sum_{i=t-Wmicro+1}^{t} J_i

    Args:
      arr:    input array (T,)
      window: window size (causal: uses arr[max(0, t-window+1)..t])

    Returns:
      Smoothed array (T,)
    """
    if window <= 1 or len(arr) == 0:
        return arr.copy()
    result = np.zeros(len(arr), dtype=np.float64)
    for t in range(len(arr)):
        start = max(0, t - window + 1)
        result[t] = np.mean(arr[start:t + 1])
    return result


def compute_fused_sensitivity(
    actions: np.ndarray,
    w_macro: int = 10,
    w_micro: int = 5,
    lambda_param: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute fused sensitivity S_t per DyQ-VLA Algorithm 1, line 1.

    M_t = compute_motion_fineness(actions)     -- captures macro trend
    J_t = compute_angular_jerk(actions)          -- captures micro spikes
    M_tilde = smooth(M_t, Wmacro)
    J_tilde = smooth(J_t, Wmicro)
    S_t = max(0, lambda * M_tilde + (1 - lambda) * J_tilde)

    Returns:
      s_t:  fused sensitivity S_t (T,)
      m_t:  Motion Fineness M_t (T,)
      j_t:  Angular Jerk J_t (T,)
    """
    if len(actions) == 0:
        return np.array([]), np.array([]), np.array([])

    m_t = compute_motion_fineness(actions)
    j_t = compute_angular_jerk(actions)
    s_t = np.maximum(
        0.0,
        lambda_param * smooth_trailing(m_t, w_macro)
        + (1.0 - lambda_param) * smooth_trailing(j_t, w_micro),
    )
    return s_t, m_t, j_t


# --------------------------------------------------------------------------
# Section IV-B1 / Algorithm 1 line 2: Sensitivity-to-Bit Mapping
# --------------------------------------------------------------------------


def sensitivity_to_target_bits(
    s_t: np.ndarray,
    theta_fp: float,
    theta_4_8: float,
    theta_2_4: float,
) -> np.ndarray:
    """
    Piecewise mapping S_t -> b_hat_t (Algorithm 1 line 2 / Eq. 6):

      S_t > theta_fp          -> 16  (BF16 bypass)
      S_t in (theta_4|8, theta_fp]  -> 8
      S_t in (theta_2|4, theta_4|8]  -> 4
      S_t in [0, theta_2|4]          -> 2
    """
    b_hat = np.zeros(len(s_t), dtype=np.int32)
    b_hat[s_t > theta_fp] = 16
    mask_8 = (s_t <= theta_fp) & (s_t > theta_4_8)
    mask_4 = (s_t <= theta_4_8) & (s_t > theta_2_4)
    b_hat[mask_8] = 8
    b_hat[mask_4] = 4
    b_hat[s_t <= theta_2_4] = 2
    return b_hat


# --------------------------------------------------------------------------
# Algorithm 1 lines 3-10: Stateful Hardware Dispatcher
# --------------------------------------------------------------------------


def stateful_dispatcher(
    b_hat: np.ndarray,
    K: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DyQ-VLA Algorithm 1 -- Stateful Hardware Dispatcher.

    Asymmetric hysteresis:
      - Upgrade: immediate (counter resets to 0)
      - Downgrade: requires K consecutive stable steps;
                   counter wraps to 0 on trigger (ct mod K)

    Paper pseudocode:
      if ˆbt >= b*_{t-1}:
          (b*_t, ct, ¯bt) <- (ˆbt, 0, ˆbt)
      else:
          ¯bt <- max(ˆbt, ¯bt_{t-1} · I(ct_{t-1} > 0))
          ct  <- ct_{t-1} · I(¯bt = ¯bt_{t-1}) + 1
          (b*_t, ct) <- (¯bt · I(ct = K) + b*_{t-1} · I(ct < K), ct mod K)

    Args:
      b_hat: target bit-width per step (from sensitivity mapping)
      K:     hysteresis delay window (paper: 5)

    Returns:
      b_star:  final dispatched bit-width per step
      counter: saturating counter state per step
      b_bar:   max candidate bit in window per step (for inspection)
    """
    T = len(b_hat)
    if T == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    b_star = np.zeros(T, dtype=np.int32)
    counter = np.zeros(T, dtype=np.int32)
    b_bar_arr = np.zeros(T, dtype=np.int32)

    c_prev = 0
    b_star_prev = 16      # paper: initial b*_{-1} = 16 (BF16 bypass state)
    b_bar_prev = int(b_hat[0])

    for t in range(T):
        b_t = int(b_hat[t])

        if b_t >= b_star_prev:
            b_star_t = b_t
            c_t = 0
            b_bar_t = b_t
        else:
            b_bar_t = max(b_t, b_bar_prev if c_prev > 0 else b_t)
            c_t = c_prev + 1 if b_bar_t == b_bar_prev else 1
            if c_t >= K:
                b_star_t = b_bar_t
                c_t = c_t % K      # paper: ct mod K  (wraps to 0 when ct==K)
            else:
                b_star_t = b_star_prev

        b_star[t] = b_star_t
        counter[t] = c_t
        b_bar_arr[t] = b_bar_t
        b_star_prev = b_star_t
        b_bar_prev = b_bar_t
        c_prev = c_t

    return b_star, counter, b_bar_arr


# --------------------------------------------------------------------------
# Section IV-B2: Offline Threshold Calibration
# --------------------------------------------------------------------------


def calibrate_thresholds_offline(s_t_cal: np.ndarray) -> dict[str, float]:
    """
    Calibrate sensitivity thresholds via percentile on calibration data.

    From the calibration sensitivity distribution (Section IV-B2):
      theta_fp   = 85th percentile  (top 15%  -> use BF16)
      theta_4|8  = 65th percentile  (next 20% -> use 8-bit)
      theta_2|4  = 40th percentile  (next 25% -> use 4-bit)
      rest                           -> 2-bit

    Note: the paper describes a more sophisticated offline calibration that
    searches for critical sensitivity intersections. This percentile method
    is a practical approximation consistent with the paper's framework.
    """
    if len(s_t_cal) == 0:
        return {"theta_fp": 0.5, "theta_4_8": 0.35, "theta_2_4": 0.2}

    return {
        "theta_fp": float(np.percentile(s_t_cal, 85)),
        "theta_4_8": float(np.percentile(s_t_cal, 65)),
        "theta_2_4": float(np.percentile(s_t_cal, 40)),
    }


# --------------------------------------------------------------------------
# Per-Episode Processing
# --------------------------------------------------------------------------


def process_episode(
    actions: np.ndarray,
    task_id: int,
    ep_idx: int,
    w_macro: int,
    w_micro: int,
    lambda_param: float,
    K: int,
    thresholds: dict[str, float],
) -> dict:
    """
    Kinematic proxy -> fused sensitivity -> bit allocation for one episode.

    Returns:
      Dict with per-step arrays (s_t, m_t, j_t, b_hat_t, b_star_t, counter_t,
      b_bar_t) and aggregate statistics.
    """
    T = len(actions)
    s_t, m_t, j_t = compute_fused_sensitivity(
        actions, w_macro=w_macro, w_micro=w_micro, lambda_param=lambda_param
    )
    b_hat = sensitivity_to_target_bits(
        s_t, thresholds["theta_fp"], thresholds["theta_4_8"], thresholds["theta_2_4"]
    )
    b_star, counter, b_bar = stateful_dispatcher(b_hat, K=K)

    bit_counts = {2: 0, 4: 0, 8: 0, 16: 0}
    for b in b_star:
        bit_counts[int(b)] += 1

    upgrades = sum(1 for t in range(1, T) if b_star[t] > b_star[t - 1])
    downgrades = sum(1 for t in range(1, T) if b_star[t] < b_star[t - 1])

    return {
        "task_id": task_id,
        "episode_idx": ep_idx,
        "num_steps": T,
        "thresholds": thresholds,
        "params": {
            "w_macro": w_macro,
            "w_micro": w_micro,
            "lambda_param": lambda_param,
            "K": K,
        },
        "s_t": s_t,
        "m_t": m_t,
        "j_t": j_t,
        "b_hat_t": b_hat,
        "b_star_t": b_star,
        "counter_t": counter,
        "b_bar_t": b_bar,
        "stats": {
            "mean_sensitivity": float(np.mean(s_t)),
            "std_sensitivity": float(np.std(s_t)),
            "max_sensitivity": float(np.max(s_t)),
            "mean_bits": float(np.mean(b_star)),
            "bit_counts": {str(k): v for k, v in bit_counts.items()},
            "num_upgrades": upgrades,
            "num_downgrades": downgrades,
            "fraction_fp": float(np.sum(b_star == 16) / max(T, 1)),
            "fraction_8bit": float(np.sum(b_star == 8) / max(T, 1)),
            "fraction_4bit": float(np.sum(b_star == 4) / max(T, 1)),
            "fraction_2bit": float(np.sum(b_star == 2) / max(T, 1)),
        },
    }


# --------------------------------------------------------------------------
# Main Pipeline
# --------------------------------------------------------------------------


def load_trajectory(ep_dir: pathlib.Path) -> dict | None:
    traj_path = ep_dir / "trajectory.npy"
    if not traj_path.exists():
        return None
    try:
        return np.load(traj_path, allow_pickle=True).item()
    except Exception:
        return None


def run_pipeline(
    trajectory_dir: str,
    out_dir: str,
    calibration_dir: str | None = None,
    w_macro: int = 10,
    w_micro: int = 5,
    lambda_param: float = 0.5,
    K: int = 5,
) -> None:
    """
    Args:
      trajectory_dir:  directory with task_XX_ep_YY subdirs, each containing trajectory.npy
      out_dir:         output directory
      calibration_dir: separate calibration set. If None, uses first 30%% of trajectory_dir.
      w_macro:         macro window size (paper: 10)
      w_micro:         micro window size (paper: 5)
      lambda_param:    fusion weight (paper: 0.5)
      K:               hysteresis delay window (paper: 5)
    """
    traj_dir = pathlib.Path(trajectory_dir)
    out_dir_path = pathlib.Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(
        d for d in traj_dir.iterdir() if d.is_dir() and d.name.startswith("task_")
    )
    print(f"Found {len(episode_dirs)} task directories in {traj_dir}")
    if not episode_dirs:
        print("No episode directories found.")
        return

    # Phase 1: Threshold calibration (Section IV-B2)
    if calibration_dir is not None:
        cal_dirs = sorted(
            d for d in pathlib.Path(calibration_dir).iterdir()
            if d.is_dir() and d.name.startswith("task_")
        )
    else:
        n_cal = max(1, int(len(episode_dirs) * 0.3))
        cal_dirs = episode_dirs[:n_cal]

    cal_s_t_list = []
    for ep_dir in cal_dirs:
        data = load_trajectory(ep_dir)
        if data is None:
            continue
        actions = np.array(data["actions"], dtype=np.float64)
        if len(actions) == 0:
            continue
        s_t, _, _ = compute_fused_sensitivity(actions, w_macro, w_micro, lambda_param)
        cal_s_t_list.append(s_t)

    if not cal_s_t_list:
        print("ERROR: No valid calibration episodes found.")
        return

    calibration_s_t = np.concatenate(cal_s_t_list)
    thresholds = calibrate_thresholds_offline(calibration_s_t)
    print(f"Calibration: {len(calibration_s_t)} steps from {len(cal_dirs)} episodes")
    print(f"Thresholds: {thresholds}")

    # Phase 2: Process all episodes
    all_results = []
    for ep_dir in episode_dirs:
        data = load_trajectory(ep_dir)
        if data is None:
            continue
        actions = np.array(data["actions"], dtype=np.float64)
        if len(actions) == 0:
            continue

        parts = ep_dir.name.split("_")
        task_id = int(parts[1]) if len(parts) > 1 else 0
        ep_idx = 0

        result = process_episode(
            actions=actions,
            task_id=task_id,
            ep_idx=ep_idx,
            w_macro=w_macro,
            w_micro=w_micro,
            lambda_param=lambda_param,
            K=K,
            thresholds=thresholds,
        )
        result["task_description"] = data.get("prompt", "")
        result["success"] = data.get("success", False)
        all_results.append(result)

        ep_out = out_dir_path / ep_dir.name
        ep_out.mkdir(parents=True, exist_ok=True)

        np.save(ep_out / "bits_dyqvla.npy", result["b_star_t"])
        np.save(ep_out / "s_t.npy", result["s_t"])
        np.save(ep_out / "m_t.npy", result["m_t"])
        np.save(ep_out / "j_t.npy", result["j_t"])
        np.save(ep_out / "b_hat_t.npy", result["b_hat_t"])
        np.save(ep_out / "counter_t.npy", result["counter_t"])
        np.save(ep_out / "b_bar_t.npy", result["b_bar_t"])

        stats = result["stats"]
        print(
            f"  {ep_dir.name}: T={result['num_steps']} | "
            f"S={stats['mean_sensitivity']:.3f} | bits={stats['mean_bits']:.2f} | "
            f"FP={stats['fraction_fp']:.1%}  8b={stats['fraction_8bit']:.1%}  "
            f"4b={stats['fraction_4bit']:.1%}  2b={stats['fraction_2bit']:.1%}"
        )

    if not all_results:
        print("No valid episodes processed.")
        return

    # Phase 3: Aggregate summary
    summary = {
        "num_episodes": len(all_results),
        "num_successful": sum(1 for r in all_results if r.get("success")),
        "total_steps": sum(r["num_steps"] for r in all_results),
        "thresholds": thresholds,
        "params": {"w_macro": w_macro, "w_micro": w_micro,
                   "lambda_param": lambda_param, "K": K},
        "mean_bits": float(np.mean([r["stats"]["mean_bits"] for r in all_results])),
        "std_bits": float(np.std([r["stats"]["mean_bits"] for r in all_results])),
        "mean_sensitivity": float(np.mean([r["stats"]["mean_sensitivity"] for r in all_results])),
        "mean_upgrades": float(np.mean([r["stats"]["num_upgrades"] for r in all_results])),
        "mean_downgrades": float(np.mean([r["stats"]["num_downgrades"] for r in all_results])),
        "mean_fraction_fp": float(np.mean([r["stats"]["fraction_fp"] for r in all_results])),
        "mean_fraction_8bit": float(np.mean([r["stats"]["fraction_8bit"] for r in all_results])),
        "mean_fraction_4bit": float(np.mean([r["stats"]["fraction_4bit"] for r in all_results])),
        "mean_fraction_2bit": float(np.mean([r["stats"]["fraction_2bit"] for r in all_results])),
    }

    summary_path = out_dir_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")

    results_path = out_dir_path / "all_episodes_results.json"
    for r in all_results:
        for key in ["s_t", "m_t", "j_t", "b_hat_t", "b_star_t",
                    "counter_t", "b_bar_t"]:
            r.pop(key, None)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results: {results_path}")


def main():
    p = argparse.ArgumentParser(
        description="DyQ-VLA (arXiv:2603.07904) dynamic bit-width allocation."
    )
    p.add_argument("--trajectory_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--calibration_dir", type=str, default=None)
    p.add_argument("--w_macro", type=int, default=10,
                   help="Macro window size (paper: 10)")
    p.add_argument("--w_micro", type=int, default=5,
                   help="Micro window size (paper: 5)")
    p.add_argument("--lambda_param", type=float, default=0.5,
                   help="Fusion weight lambda (paper: 0.5)")
    p.add_argument("--hysteresis_K", dest="K", type=int, default=5,
                   help="Hysteresis delay window (paper: 5)")

    args = p.parse_args()
    run_pipeline(
        trajectory_dir=args.trajectory_dir,
        out_dir=args.out_dir,
        calibration_dir=args.calibration_dir,
        w_macro=args.w_macro,
        w_micro=args.w_micro,
        lambda_param=args.lambda_param,
        K=args.K,
    )


if __name__ == "__main__":
    main()
