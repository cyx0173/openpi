"""
Mode 6: Active perturbation data collection for adaptive quantization.
Collects (observation, required_precision) pairs to train an Adaptive Quantization Selector
that predicts which precision level (w4a4/w4a8/w4a16/fp16) is needed at each step.
"""
import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/libero"))

import imageio.v2 as imageio
import numpy as np
import torch
import tqdm
import tyro

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# PyTorch legacy load hack
_original_torch_load = torch.load
def _safe_legacy_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_legacy_load

# robosuite robot name patch
from robosuite.models.robots.robot_model import create_robot as _create_robot_orig
def _safe_create_robot(robot_name, *args, **kwargs):
    try:
        return _create_robot_orig(robot_name, *args, **kwargs)
    except KeyError:
        import robosuite.models.robots.panda_model as pm
        return pm.Panda(idn=kwargs.get("idn", 0))
import robosuite.models.robots.robot_model
robosuite.models.robots.robot_model.create_robot = _safe_create_robot
for mod_name, mod in list(sys.modules.items()):
    if mod is not None and hasattr(mod, "create_robot"):
        mod.create_robot = _safe_create_robot


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8001
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    seed: int = 7
    resize_size: int = 224
    replan_steps: int = 5
    combined_dir: str = "/home/chengyuxuan/vla/openpi/examples/quant_experiment/data/quant/quant_w4a4"
    output_dir: str = "/home/chengyuxuan/vla/openpi/examples/active_experiment/data"
    video_dir: str = "/home/chengyuxuan/vla/openpi/examples/active_experiment/data/videos"
    trajectory_list: str = ""
    skip_existing: bool = True
    checkpoint_every: int = 100
    worker_id: int = 0


_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_obs_element(obs, task_description, resize_size):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )),
        "prompt": str(task_description),
    }


def _task_segment(task_description):
    return task_description.replace(" ", "_").replace("/", "_").replace("'", "")[:80]


def _get_eef_pos(env, obs):
    if "robot0_eef_pos" in obs and obs["robot0_eef_pos"] is not None:
        return np.array(obs["robot0_eef_pos"])
    if hasattr(env, "sim") and hasattr(env.sim, "data"):
        return env.sim.data.body_xpos[env.robot0_gripper_body_id].copy()
    if obs is not None:
        for key in ("eef_pos", "robot0_eef_quat", "state"):
            if key in obs and obs[key] is not None:
                arr = np.array(obs[key])
                if arr.shape[-1] == 3:
                    return arr
    return None


def _save_video(video_dir, traj_name, tag, images, success):
    if not images:
        return
    suffix = "success" if success else "failure"
    out_path = pathlib.Path(video_dir) / f"{traj_name}_{tag}_{suffix}.mp4"
    imageio.mimwrite(out_path, images, fps=10)
    logging.info(f"    [Video] {out_path}")


def _restore_env_to_state(env, initial_state, chunk_states, num_steps_wait,
                           fp16_chunks, chunk_perturb, step_perturb):
    """
    Restore environment to state at (chunk_perturb, step_perturb).
    Falls back to dummy actions or replay if teleportation fails.
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    has_restored = False
    if chunk_perturb < len(chunk_states) and chunk_states[chunk_perturb] is not None:
        cs = chunk_states[chunk_perturb]
        mj = env.sim.get_state()
        mj.time = cs["time"]
        mj.qpos[:] = cs["qpos"]
        mj.qvel[:] = cs["qvel"]
        env.sim.set_state(mj)
        env.sim.forward()
        obs = env.get_observation()
        has_restored = True

    if not has_restored:
        for _ in range(num_steps_wait):
            try:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            except ValueError:
                done = True
            if done:
                return obs

        for i in range(chunk_perturb):
            for a in fp16_chunks[i]:
                try:
                    obs, _, done, _ = env.step(a)
                except ValueError:
                    done = True
                if done:
                    return obs

    for a in fp16_chunks[chunk_perturb][:step_perturb]:
        try:
            obs, _, done, _ = env.step(a)
        except ValueError:
            done = True
        if done:
            return obs

    return obs


def _run_fp16_recovery(env, fp16_client, task_description, args, images=None, max_steps=600):
    """Let FP16 online inference recover after injecting a candidate action."""
    obs = env.get_observation()
    steps_taken = 0
    done = False

    while not done and steps_taken < max_steps:
        try:
            element = _get_obs_element(obs, task_description, args.resize_size)
            action_chunk = fp16_client.infer(element)["actions"]
            actions = action_chunk[: args.replan_steps].tolist()
        except Exception:
            break

        for a in actions:
            try:
                obs, _, done, _ = env.step(a)
            except ValueError:
                return False
            if images is not None:
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                images.append(img)
            if done:
                return True
            steps_taken += 1
            if steps_taken >= max_steps:
                return False

    return bool(done)


def _try_single_action(env, action):
    """
    Advance environment by one step.
    Returns (done, step_crash):
        (True, False)  = task completed
        (False, False)  = not done; FP16 recovery possible
        (False, True)   = physics crash; FP16 recovery useless
    """
    try:
        _, _, done, _ = env.step(action)
        return (done, False)
    except ValueError:
        return (False, True)


def _run_active_step_perturbation(env, initial_state,
                                  fp16_chunks, w4a4_chunks, w4a8_chunks, w4a16_chunks,
                                  chunk_states, num_steps_wait,
                                  chunk_perturb, step_perturb,
                                  task_description, fp16_client, args,
                                  traj_name="", video_dir=None):
    """
    For a single (chunk_k, step_s) moment, try precision candidates in order:
    w4a4 -> w4a8 -> w4a16 -> fp16 -> failed
    Returns data_point dict.
    """
    action_fp16  = fp16_chunks[chunk_perturb][step_perturb]
    action_w4a4  = w4a4_chunks[chunk_perturb][step_perturb]
    action_w4a8  = w4a8_chunks[chunk_perturb][step_perturb]
    action_w4a16 = w4a16_chunks[chunk_perturb][step_perturb]

    def _diff(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    # Try w4a4
    obs_at_k_s = _restore_env_to_state(
        env, initial_state, chunk_states, num_steps_wait,
        fp16_chunks, chunk_perturb, step_perturb
    )#恢复状态
    eef_before = _get_eef_pos(env, obs_at_k_s)
    images_w4a4 = [np.ascontiguousarray(obs_at_k_s["agentview_image"][::-1, ::-1])]
    done, step_crash = _try_single_action(env, action_w4a4)#执行 w4a4 action
    if step_crash:
        done_w4a4 = False
        logging.info(f"  chunk={chunk_perturb} step={step_perturb} w4a4: crash")
    elif done:
        done_w4a4 = True
    else:
        done_w4a4 = _run_fp16_recovery(env, fp16_client, task_description, args, images=images_w4a4)
        eef_after = _get_eef_pos(env, env.get_observation())
        eef_delta = float(np.linalg.norm(eef_after - eef_before)) if (eef_before is not None and eef_after is not None) else None
        ad = _diff(action_w4a4, action_fp16)
        eef_s = f"{eef_delta:.4f}" if eef_delta is not None else "N/A"
        logging.info(f"  chunk={chunk_perturb} step={step_perturb} w4a4: success={done_w4a4}, eef_delta={eef_s}, action_diff={ad:.4f}")
    if video_dir is not None:
        _save_video(video_dir, traj_name, f"chunk{chunk_perturb}_step{step_perturb}_w4a4", images_w4a4, done_w4a4)
    if done_w4a4:
        return _build_data_point(
            obs_at_k_s, action_fp16, action_w4a4, action_w4a8, action_w4a16,
            "w4a4", chunk_perturb, step_perturb, task_description
        )

    # Try w4a8
    obs_at_k_s = _restore_env_to_state(
        env, initial_state, chunk_states, num_steps_wait,
        fp16_chunks, chunk_perturb, step_perturb
    )
    eef_before = _get_eef_pos(env, obs_at_k_s)
    images_w4a8 = [np.ascontiguousarray(obs_at_k_s["agentview_image"][::-1, ::-1])]
    done, step_crash = _try_single_action(env, action_w4a8)
    if step_crash:
        done_w4a8 = False
        logging.info(f"  chunk={chunk_perturb} step={step_perturb} w4a8: crash")
    elif done:
        done_w4a8 = True
    else:
        done_w4a8 = _run_fp16_recovery(env, fp16_client, task_description, args, images=images_w4a8)
        eef_after = _get_eef_pos(env, env.get_observation())
        eef_delta = float(np.linalg.norm(eef_after - eef_before)) if (eef_before is not None and eef_after is not None) else None
        ad = _diff(action_w4a8, action_fp16)
        eef_s = f"{eef_delta:.4f}" if eef_delta is not None else "N/A"
        logging.info(f"  chunk={chunk_perturb} step={step_perturb} w4a8: success={done_w4a8}, eef_delta={eef_s}, action_diff={ad:.4f}")
    if video_dir is not None:
        _save_video(video_dir, traj_name, f"chunk{chunk_perturb}_step{step_perturb}_w4a8", images_w4a8, done_w4a8)
    if done_w4a8:
        return _build_data_point(
            obs_at_k_s, action_fp16, action_w4a4, action_w4a8, action_w4a16,
            "w4a8", chunk_perturb, step_perturb, task_description
        )

    # Try w4a16
    obs_at_k_s = _restore_env_to_state(
        env, initial_state, chunk_states, num_steps_wait,
        fp16_chunks, chunk_perturb, step_perturb
    )
    eef_before = _get_eef_pos(env, obs_at_k_s)
    images_w4a16 = [np.ascontiguousarray(obs_at_k_s["agentview_image"][::-1, ::-1])]
    done, step_crash = _try_single_action(env, action_w4a16)
    if step_crash:
        done_w4a16 = False
        logging.info(f"  chunk={chunk_perturb} step={step_perturb} w4a16: crash")
    elif done:
        done_w4a16 = True
    else:
        done_w4a16 = _run_fp16_recovery(env, fp16_client, task_description, args, images=images_w4a16)
        eef_after = _get_eef_pos(env, env.get_observation())
        eef_delta = float(np.linalg.norm(eef_after - eef_before)) if (eef_before is not None and eef_after is not None) else None
        ad = _diff(action_w4a16, action_fp16)
        eef_s = f"{eef_delta:.4f}" if eef_delta is not None else "N/A"
        logging.info(f"  chunk={chunk_perturb} step={step_perturb} w4a16: success={done_w4a16}, eef_delta={eef_s}, action_diff={ad:.4f}")
    if video_dir is not None:
        _save_video(video_dir, traj_name, f"chunk{chunk_perturb}_step{step_perturb}_w4a16", images_w4a16, done_w4a16)
    if done_w4a16:
        return _build_data_point(
            obs_at_k_s, action_fp16, action_w4a4, action_w4a8, action_w4a16,
            "w4a16", chunk_perturb, step_perturb, task_description
        )

    # Try fp16
    obs_at_k_s = _restore_env_to_state(
        env, initial_state, chunk_states, num_steps_wait,
        fp16_chunks, chunk_perturb, step_perturb
    )
    eef_before = _get_eef_pos(env, obs_at_k_s)
    images_fp16 = [np.ascontiguousarray(obs_at_k_s["agentview_image"][::-1, ::-1])]
    done, step_crash = _try_single_action(env, action_fp16)
    if step_crash:
        done_fp16 = False
        logging.info(f"  chunk={chunk_perturb} step={step_perturb} fp16: crash")
    elif done:
        done_fp16 = True
    else:
        done_fp16 = _run_fp16_recovery(env, fp16_client, task_description, args, images=images_fp16)
        eef_after = _get_eef_pos(env, env.get_observation())
        eef_delta = float(np.linalg.norm(eef_after - eef_before)) if (eef_before is not None and eef_after is not None) else None
        eef_s = f"{eef_delta:.4f}" if eef_delta is not None else "N/A"
        logging.info(f"  chunk={chunk_perturb} step={step_perturb} fp16: success={done_fp16}, eef_delta={eef_s}")
    if video_dir is not None:
        _save_video(video_dir, traj_name, f"chunk{chunk_perturb}_step{step_perturb}_fp16", images_fp16, done_fp16)
    if done_fp16:
        return _build_data_point(
            obs_at_k_s, action_fp16, action_w4a4, action_w4a8, action_w4a16,
            "fp16", chunk_perturb, step_perturb, task_description
        )

    # Failed
    logging.info(f"  chunk={chunk_perturb} step={step_perturb} failed: no precision succeeded")
    return _build_data_point(
        obs_at_k_s, action_fp16, action_w4a4, action_w4a8, action_w4a16,
        "failed", chunk_perturb, step_perturb, task_description
    )


def _build_data_point(obs, action_fp16, action_w4a4, action_w4a8, action_w4a16,
                       label, chunk_idx, step_idx, task_description):
    """Package observation + 4 actions + label into a data point dict."""
    obs_element = _get_obs_element(obs, task_description, 224)
    meta = {
        "chunk_idx": chunk_idx,
        "step_idx": step_idx,
        "task_description": task_description,
    }

    return {
        "observation": obs_element,
        "gt_action_fp16": list(action_fp16),
        "gt_action_w4a4": list(action_w4a4),
        "gt_action_w4a8": list(action_w4a8),
        "gt_action_w4a16": list(action_w4a16),
        "required_precision": label,
        "meta": meta,
    }


def _run_active_perturbation(args: Args, fp16_client) -> None:
    """Main loop: iterate over all trajectories and (chunk, step) moments, collect data points."""
    np.random.seed(args.seed)
    logging.info("Active perturbation data collection (Mode 6)")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = pathlib.Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    if args.trajectory_list.strip():
        paths = [pathlib.Path(p.strip()) for p in args.trajectory_list.split(",") if p.strip()]
        combined_paths = [p for p in paths if p.exists()]
        logging.info(f"Using {len(combined_paths)} trajectories from --trajectory-list")
    else:
        combined_dir = pathlib.Path(args.combined_dir)
        combined_paths = sorted(combined_dir.glob("*_combined.json"))
        logging.info(f"Found {len(combined_paths)} trajectories in {combined_dir}")

    if not combined_paths:
        logging.error("No trajectory files found.")
        return

    completed, all_data, checkpoint_entries = _load_checkpoint(output_dir, args.worker_id)
    all_results = []
    checkpoint_new_entries = []
    logging.info(f"  Resuming: {len(all_data)} data points already collected, "
                 f"{len(completed)} (traj,k,s) positions completed")

    for traj_path in tqdm.tqdm(combined_paths):
        traj_name = traj_path.stem.replace("_combined", "")
        out_result_path = output_dir / f"{traj_name}_active.json"

        if out_result_path.exists() and args.skip_existing:
            logging.info(f"  [Skip] {traj_name} already fully processed")
            continue

        logging.info(f"  Processing: {traj_name}")

        with open(traj_path) as f:
            traj_data = json.load(f)

        task_id = traj_data["task_id"]
        task_description = traj_data["task_description"]
        chunks = traj_data["chunks"]

        if not chunks:
            logging.warning(f"  No chunks in {traj_path.name}, skipping")
            continue

        fp16_chunks  = [c["fp16_actions"]  for c in chunks]
        w4a4_chunks  = [c["w4a4_actions"]  for c in chunks]
        w4a8_chunks  = [c["w4a8_actions"]  for c in chunks]
        w4a16_chunks = [c["w4a16_actions"] for c in chunks]
        chunk_states = [c.get("mujoco_state") for c in chunks]
        num_chunks = len(fp16_chunks)

        task = task_suite.get_task(task_id)
        init_qpos = task_suite.get_task_init_states(task_id)[0]
        if isinstance(init_qpos, np.ndarray):
            init_qpos = init_qpos.tolist()
        initial_state = np.array(init_qpos)

        traj_results = [
            {
                "chunk_idx": e["chunk_idx"],
                "step_idx": e["step_idx"],
                "required_precision": e["data_point"]["required_precision"],
                "meta": e["data_point"]["meta"],
            }
            for e in checkpoint_entries
            if e["traj_name"] == traj_name
        ]

        env_traj, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:
            for k in range(num_chunks):
                num_steps = len(fp16_chunks[k])
                for s in range(num_steps):
                    if (traj_name, k, s) in completed:
                        continue

                    data_point = _run_active_step_perturbation(
                        env_traj, initial_state,
                        fp16_chunks, w4a4_chunks, w4a8_chunks, w4a16_chunks,
                        chunk_states, args.num_steps_wait,
                        k, s,
                        task_description, fp16_client, args,
                        traj_name=traj_name, video_dir=video_dir
                    )
                    all_data.append(data_point)
                    traj_results.append({
                        "chunk_idx": k,
                        "step_idx": s,
                        "required_precision": data_point["required_precision"],
                        "meta": data_point["meta"],
                    })

                    entry = {
                        "traj_name": traj_name,
                        "chunk_idx": k,
                        "step_idx": s,
                        "data_point": data_point,
                    }
                    checkpoint_new_entries.append(entry)
                    if len(checkpoint_new_entries) >= args.checkpoint_every:
                        _save_checkpoint(output_dir, checkpoint_new_entries, args.worker_id)
                        checkpoint_new_entries = []
                        logging.info(f"    [Checkpoint] saved {len(all_data)} data points")
        finally:
            env_traj.close()

        with open(out_result_path, "w") as f:
            json.dump({
                "task_id": task_id,
                "task_description": task_description,
                "num_chunks": num_chunks,
                "results": traj_results,
            }, f, indent=2, cls=_NumpyEncoder)

    if checkpoint_new_entries:
        _save_checkpoint(output_dir, checkpoint_new_entries, args.worker_id)
        logging.info(f"  [Final checkpoint] flushed {len(checkpoint_new_entries)} remaining entries")

    _save_dataset(all_data, output_dir, args.worker_id)
    _save_summary(all_data, output_dir, args.worker_id)
    logging.info("Done.")


def _load_checkpoint(output_dir, worker_id: int):
    """Parse checkpoint_w{worker_id}.jsonl and return completed set, all_data, entries."""
    checkpoint_path = pathlib.Path(output_dir) / f"checkpoint_w{worker_id}.jsonl"
    completed = set()
    all_data = []
    entries = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = (entry["traj_name"], entry["chunk_idx"], entry["step_idx"])
                completed.add(key)
                all_data.append(entry["data_point"])
                entries.append(entry)
    return completed, all_data, entries


def _save_checkpoint(output_dir, new_entries, worker_id: int):
    """Append entries to checkpoint_w{worker_id}.jsonl."""
    checkpoint_path = pathlib.Path(output_dir) / f"checkpoint_w{worker_id}.jsonl"
    with open(checkpoint_path, "a") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, cls=_NumpyEncoder) + "\n")


def _save_dataset(all_data, output_dir, worker_id: int):
    """Save all data points to active_dataset_w{worker_id}.jsonl."""
    dataset_path = pathlib.Path(output_dir) / f"active_dataset_w{worker_id}.jsonl"
    with open(dataset_path, "w") as f:
        for dp in all_data:
            f.write(json.dumps(dp, cls=_NumpyEncoder) + "\n")
    logging.info(f"  [Dataset] saved {len(all_data)} data points to {dataset_path}")


def _save_summary(all_data, output_dir, worker_id: int):
    """Count label frequencies and save to summary.json."""
    from collections import Counter
    label_counts = Counter(dp["required_precision"] for dp in all_data)
    total = len(all_data)

    summary = {
        "total_data_points": total,
        "label_counts": dict(label_counts),
        "label_rates": {
            label: count / total if total > 0 else 0.0
            for label, count in label_counts.items()
        }
    }

    summary_path = pathlib.Path(output_dir) / f"active_summary_w{worker_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"  [Summary] saved to {summary_path}")
    logging.info(f"  Labels: {dict(label_counts)}")


def main(args: Args) -> None:
    fp16_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    _run_active_perturbation(args, fp16_client)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(main)
