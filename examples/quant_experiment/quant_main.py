"""
quant_main.py - Unified trajectory recording + perturbation sensitivity evaluation.

Modes:
  Mode 1 (basic):       Record FP16 baseline trajectories. Outputs _trajectory.json.
                         Server: --port (FP16).
  Mode 2 (combined):    Record FP16 + W4A4 trajectories + mujoco_state.
                         Outputs _combined.json (required by Mode 3/4).
                         Servers: --port (FP16) and --w4a4_port (W4A4).
  Mode 3 (chunk):       Chunk-level perturbation: inject W4A4 chunk k, then FP16 inference.
                         Server: --port (FP16 inference). Needs _combined.json from Mode 2.
  Mode 4 (step):        Step-level perturbation: inject W4A4 action at step (k,s), replay FP16.
                         Server: --port (FP16 inference). Needs _combined.json from Mode 2.
  Mode 5 (quad):        Record FP16 + W4A4 + W4A8 + W4A16 trajectories + mujoco_state.
                         Outputs _combined.json.
                         Servers: --port (FP16), --w4a4_port (W4A4), --w4a8_port (W4A8), --w4a16_port (W4A16).

Usage:
  # Mode 1 - Record FP16 baseline trajectories (single server on port 8000)
  uv run python examples/libero/quant_main.py \
      --mode 1 --task_suite_name libero_spatial \
      --output_dir data/libero/videos/quant \
      --port 8000

  # Mode 2 - Record FP16 + W4A4 trajectories (dual servers)
  #   Terminal 1 (FP16):  uv run python scripts/serve_policy.py --env LIBERO --port 8001
  #   Terminal 2 (W4A4):  uv run python scripts/serve_policy.py --env LIBERO --port 8000 --quantize --quantize-bits-w 4 --quantize-bits-a 4
  #   This script:
  uv run python examples/libero/quant_main.py \
      --mode 2 --task_suite_name libero_spatial \
      --port 8001 --w4a4_port 8000 \
      --output_dir data/libero/videos/quant_w4a4

  # Mode 5 - Record FP16 + W4A4 + W4A8 + W4A16 trajectories (quad servers)
  #   Terminal 1 (FP16):   uv run python scripts/serve_policy.py --env LIBERO --port 8001
  #   Terminal 2 (W4A4):   uv run python scripts/serve_policy.py --env LIBERO --port 8000 --quantize --quantize-bits-w 4 --quantize-bits-a 4
  #   Terminal 3 (W4A8):   uv run python scripts/serve_policy.py --env LIBERO --port 8002 --quantize --quantize-bits-w 4 --quantize-bits-a 8
  #   Terminal 4 (W4A16):  uv run python scripts/serve_policy.py --env LIBERO --port 8003 --quantize --quantize-bits-w 4 --quantize-bits-a 16
  #   This script:
  uv run python examples/libero/quant_main.py \
      --mode 5 --task_suite_name libero_spatial \
      --port 8001 --w4a4_port 8000 --w4a8_port 8002 --w4a16_port 8003 \
      --output_dir data/libero/videos/quant_all

  # Mode 3 - Chunk-level perturbation (needs _combined.json from Mode 2)
  uv run python examples/libero/quant_main.py \
      --mode 3 --task_suite_name libero_spatial \
      --port 8001 \
      --combined_dir data/libero/videos/quant_w4a4 \
      --output_dir data/quant/spatial_chunk_results

  # Mode 4 - Step-level perturbation (needs _combined.json from Mode 2)
  uv run python examples/libero/quant_main.py \
      --mode 4 --task_suite_name libero_spatial \
      --port 8001 \
      --combined_dir data/libero/videos/quant_w4a4 \
      --output_dir data/quant/spatial_step_results
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
import sys
import os

# Custom JSON encoder that handles numpy types
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/libero"))

import imageio
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

# ── PyTorch legacy load hack ──────────────────────────────────────────────────
_original_torch_load = torch.load
def _safe_legacy_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_legacy_load

# ── robosuite robot name patch ───────────────────────────────────────────────
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


# ── Args ─────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Args:
    # ── Server ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    # FP16 server port (used for Mode 1 recording and Mode 3/4 perturbation inference)
    port: int = 8001
    # W4A4 server port (only for Mode 2 dual-server recording)
    w4a4_port: int = 8000
    # W4A8 server port (only for Mode 5 quad-server recording)
    w4a8_port: int = 8002
    # W4A16 server port (only for Mode 5 quad-server recording)
    w4a16_port: int = 8003

    # ── Task ─────────────────────────────────────────────────────────────────
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    seed: int = 7
    resize_size: int = 224
    replan_steps: int = 5

    # ── Mode ─────────────────────────────────────────────────────────────────
    # 1 = record basic (FP16), 2 = record combined (FP16 + W4A4 dual server),
    # 3 = chunk-level perturbation, 4 = step-level perturbation,
    # 5 = record combined (FP16 + W4A4 + W4A8 + W4A16 quad server),
    # 6 = active data collection (needs Mode 5 _combined.json, separate file active_main.py)
    mode: int = 5

    # ── Recording (Mode 1 / 2) ──────────────────────────────────────────────
    output_dir: str = "/home/chengyuxuan/vla/openpi/examples/quant_experiment/data_new/quant_10/quant_w4a4"
    video_dir: str = "/home/chengyuxuan/vla/openpi/examples/quant_experiment/data_new/quant_10/videos"
    task_name: str = ""
    skip_existing: bool = True

    # ── Perturbation (Mode 3 / 4) ────────────────────────────────────────────
    combined_dir: str = "/home/chengyuxuan/vla/openpi/examples/quant_experiment/data_new/quant_10/combined"
    trajectory_list: str = ""


# ── Max steps per task suite ──────────────────────────────────────────────────
_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
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
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }


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


def _task_segment(task_description):
    return task_description.replace(" ", "_").replace("/", "_").replace("'", "")[:80]


def _save_video(video_dir, traj_name, tag, images, success):
    if not images:
        return
    suffix = "success" if success else "failure"
    out_path = pathlib.Path(video_dir) / f"{traj_name}_{tag}_{suffix}.mp4"
    imageio.mimwrite(out_path, images, fps=10)
    logging.info(f"    [Video] {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1: Basic W4A4 trajectory recording
# ═══════════════════════════════════════════════════════════════════════════════
def _record_mode1(env, task, task_description, initial_state, client, args):
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    t = 0
    max_steps = _MAX_STEPS.get(args.task_suite_name, 520)
    action_plan = collections.deque()
    replay_images = []
    steps = []

    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            replay_images.append(img)

            if not action_plan:
                element = _get_obs_element(obs, task_description, args.resize_size)
                action_chunk = client.infer(element)["actions"]
                action_plan.extend(action_chunk[: args.replan_steps].tolist())

            action = action_plan.popleft()
            obs, reward, done, info = env.step(action)
            steps.append({
                "step": t - args.num_steps_wait,
                "action": action,
                "eef_pos": _get_eef_pos(env, obs).tolist(),
                "is_success": done,
                "task_desc": task_description,
            })
            t += 1
            if done:
                return {"steps": steps, "success": done, "replay_images": replay_images}
        except Exception as e:
            logging.error(f"  Exception at step {t}: {e}")
            break

    return {"steps": steps, "success": done, "replay_images": replay_images}


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2: Combined FP16 + W4A4 dual-server recording
# ═══════════════════════════════════════════════════════════════════════════════
def _record_mode2(env, task, task_description, initial_state, w4a4_client, fp16_client, args):
    """
    Record trajectory with both W4A4 and FP16 actions by querying two servers simultaneously.
    Each replan window = one chunk. Stores fp16_actions, w4a4_actions, mujoco_state per chunk.
    """
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    t = 0
    max_steps = _MAX_STEPS.get(args.task_suite_name, 520)
    w4a4_plan = collections.deque()
    fp16_plan = collections.deque()
    replay_images = []
    chunks = []
    current_fp16 = []
    current_w4a4 = []
    current_w4a8 = []
    current_w4a16 = []
    chunk_step_start = args.num_steps_wait

    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            replay_images.append(img)

            if not w4a4_plan:
                element = _get_obs_element(obs, task_description, args.resize_size)
                w4a4_result = w4a4_client.infer(element)
                fp16_result = fp16_client.infer(element)

                mj = env.sim.get_state()
                mujoco_state = {
                    "time": float(mj.time),
                    "qpos": [float(x) for x in mj.qpos],
                    "qvel": [float(x) for x in mj.qvel],
                }

                if current_fp16:
                    chunks.append({
                        "chunk_idx": len(chunks),
                        "step_start": chunk_step_start,
                        "fp16_actions": current_fp16,
                        "w4a4_actions": current_w4a4,
                        "mujoco_state": mujoco_state,
                    })

                current_fp16 = []
                current_w4a4 = []
                chunk_step_start = t

                w4a4_plan.extend(w4a4_result["actions"][: args.replan_steps].tolist())
                fp16_plan.extend(fp16_result["actions"][: args.replan_steps].tolist())

            w4a4_action = w4a4_plan.popleft()
            fp16_action = fp16_plan.popleft()

            obs, reward, done, info = env.step(fp16_action)

            current_fp16.append(fp16_action)
            current_w4a4.append(w4a4_action)
            t += 1

            if done:
                if current_fp16:
                    mj = env.sim.get_state()
                    chunks.append({
                        "chunk_idx": len(chunks),
                        "step_start": chunk_step_start,
                        "fp16_actions": current_fp16,
                        "w4a4_actions": current_w4a4,
                        "mujoco_state": {
                            "time": float(mj.time),
                            "qpos": [float(x) for x in mj.qpos],
                            "qvel": [float(x) for x in mj.qvel],
                        },
                    })
                return {"chunks": chunks, "success": done, "replay_images": replay_images}

        except Exception as e:
            logging.error(f"  Exception at step {t}: {e}")
            if current_fp16:
                mj = env.sim.get_state()
                chunks.append({
                    "chunk_idx": len(chunks),
                    "step_start": chunk_step_start,
                    "fp16_actions": current_fp16,
                    "w4a4_actions": current_w4a4,
                    "mujoco_state": {
                        "time": float(mj.time),
                        "qpos": [float(x) for x in mj.qpos],
                        "qvel": [float(x) for x in mj.qvel],
                    },
                })
            break

    if current_fp16:
        mj = env.sim.get_state()
        chunks.append({
            "chunk_idx": len(chunks),
            "step_start": chunk_step_start,
            "fp16_actions": current_fp16,
            "w4a4_actions": current_w4a4,
            "mujoco_state": {
                "time": float(mj.time),
                "qpos": [float(x) for x in mj.qpos],
                "qvel": [float(x) for x in mj.qvel],
            },
        })

    return {"chunks": chunks, "success": done, "replay_images": replay_images}


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 5: Combined FP16 + W4A4 + W4A8 + W4A16 quad-server recording
# ═══════════════════════════════════════════════════════════════════════════════
def _record_mode5(env, task, task_description, initial_state,
                   fp16_client, w4a4_client, w4a8_client, w4a16_client, args):
    """
    Record trajectory with FP16, W4A4, W4A8, and W4A16 actions by querying four servers simultaneously.
    Each replan window = one chunk. Stores fp16_actions, w4a4_actions, w4a8_actions, w4a16_actions, mujoco_state per chunk.
    """
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    t = 0
    max_steps = _MAX_STEPS.get(args.task_suite_name, 520)
    fp16_plan = collections.deque()
    w4a4_plan = collections.deque()
    w4a8_plan = collections.deque()
    w4a16_plan = collections.deque()
    replay_images = []
    chunks = []
    current_fp16 = []
    current_w4a4 = []
    current_w4a8 = []
    current_w4a16 = []
    chunk_step_start = args.num_steps_wait

    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            replay_images.append(img)

            if not fp16_plan:
                element = _get_obs_element(obs, task_description, args.resize_size)
                fp16_result = fp16_client.infer(element)
                w4a4_result = w4a4_client.infer(element)
                w4a8_result = w4a8_client.infer(element)
                w4a16_result = w4a16_client.infer(element)

                mj = env.sim.get_state()
                mujoco_state = {
                    "time": float(mj.time),
                    "qpos": [float(x) for x in mj.qpos],
                    "qvel": [float(x) for x in mj.qvel],
                }

                if current_fp16:
                    chunks.append({
                        "chunk_idx": len(chunks),
                        "step_start": chunk_step_start,
                        "fp16_actions": current_fp16,
                        "w4a4_actions": current_w4a4,
                        "w4a8_actions": current_w4a8,
                        "w4a16_actions": current_w4a16,
                        "mujoco_state": current_chunk_state,
                    })

                current_chunk_state = mujoco_state
                current_fp16 = []
                current_w4a4 = []
                current_w4a8 = []
                current_w4a16 = []
                chunk_step_start = t

                fp16_plan.extend(fp16_result["actions"][: args.replan_steps].tolist())
                w4a4_plan.extend(w4a4_result["actions"][: args.replan_steps].tolist())
                w4a8_plan.extend(w4a8_result["actions"][: args.replan_steps].tolist())
                w4a16_plan.extend(w4a16_result["actions"][: args.replan_steps].tolist())

            fp16_action = fp16_plan.popleft()
            w4a4_action = w4a4_plan.popleft()
            w4a8_action = w4a8_plan.popleft()
            w4a16_action = w4a16_plan.popleft()

            obs, reward, done, info = env.step(fp16_action)

            current_fp16.append(fp16_action)
            current_w4a4.append(w4a4_action)
            current_w4a8.append(w4a8_action)
            current_w4a16.append(w4a16_action)
            t += 1

            if done:
                if current_fp16:
                    mj = env.sim.get_state()
                    chunks.append({
                        "chunk_idx": len(chunks),
                        "step_start": chunk_step_start,
                        "fp16_actions": current_fp16,
                        "w4a4_actions": current_w4a4,
                        "w4a8_actions": current_w4a8,
                        "w4a16_actions": current_w4a16,
                        "mujoco_state": {
                            "time": float(mj.time),
                            "qpos": [float(x) for x in mj.qpos],
                            "qvel": [float(x) for x in mj.qvel],
                        },
                    })
                return {"chunks": chunks, "success": done, "replay_images": replay_images}

        except Exception as e:
            logging.error(f"  Exception at step {t}: {e}")
            if current_fp16:
                mj = env.sim.get_state()
                chunks.append({
                    "chunk_idx": len(chunks),
                    "step_start": chunk_step_start,
                    "fp16_actions": current_fp16,
                    "w4a4_actions": current_w4a4,
                    "w4a8_actions": current_w4a8,
                    "w4a16_actions": current_w4a16,
                    "mujoco_state": {
                        "time": float(mj.time),
                        "qpos": [float(x) for x in mj.qpos],
                        "qvel": [float(x) for x in mj.qvel],
                    },
                })
            break

    if current_fp16:
        mj = env.sim.get_state()
        chunks.append({
            "chunk_idx": len(chunks),
            "step_start": chunk_step_start,
            "fp16_actions": current_fp16,
            "w4a4_actions": current_w4a4,
            "w4a8_actions": current_w4a8,
            "w4a16_actions": current_w4a16,
            "mujoco_state": {
                "time": float(mj.time),
                "qpos": [float(x) for x in mj.qpos],
                "qvel": [float(x) for x in mj.qvel],
            },
        })

    return {"chunks": chunks, "success": done, "replay_images": replay_images}


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 & 4: Perturbation helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _run_baseline_replay(env, initial_state, fp16_chunks, chunk_states, num_steps_wait):
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    replay_images = []

    for _ in range(num_steps_wait):
        try:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
        except ValueError:
            done = True
        if done:
            return _get_eef_pos(env, obs), True, replay_images

    for chunk_idx, chunk in enumerate(fp16_chunks):
        if chunk_idx < len(chunk_states):
            cs = chunk_states[chunk_idx]
            mj = env.sim.get_state()
            mj.time = cs["time"]
            mj.qpos[:] = cs["qpos"]
            mj.qvel[:] = cs["qvel"]
            env.sim.set_state(mj)
            env.sim.forward()
            obs = env.get_observation()
        for a in chunk:
            try:
                obs, reward, done, info = env.step(a)
            except ValueError:
                done = True
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            replay_images.append(img)
            if done:
                return _get_eef_pos(env, obs), True, replay_images
    return _get_eef_pos(env, obs), bool(done), replay_images


def _run_with_chunk_perturbation(env, initial_state, w4a4_chunks,
                                  chunk_states, num_steps_wait, chunk_perturb,
                                  task_description, fp16_client, args):
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    replay_images = []

    for _ in range(num_steps_wait):
        try:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
        except ValueError:
            done = True
        if done:
            break

    if chunk_perturb < len(chunk_states):
        cs = chunk_states[chunk_perturb]
        mj = env.sim.get_state()
        mj.time = cs["time"]
        mj.qpos[:] = cs["qpos"]
        mj.qvel[:] = cs["qvel"]
        env.sim.set_state(mj)
        env.sim.forward()
        obs = env.get_observation()

    if chunk_perturb < len(w4a4_chunks):
        for a in w4a4_chunks[chunk_perturb]:
            try:
                obs, reward, done, info = env.step(a)
            except ValueError:
                done = True
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            replay_images.append(img)
            if done:
                return _get_eef_pos(env, obs), True, replay_images

    prev_eef = _get_eef_pos(env, obs)
    MAX_STEPS = 600
    steps_taken = 0
    while not done and steps_taken < MAX_STEPS:
        try:
            element = _get_obs_element(obs, task_description, args.resize_size)
            action_chunk = fp16_client.infer(element)["actions"]
            actions = action_chunk[: args.replan_steps].tolist()
        except Exception:
            break
        for a in actions:
            try:
                obs, reward, done, info = env.step(a)
            except ValueError:
                done = True
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            replay_images.append(img)
            if done:
                break
            steps_taken += 1
        if done:
            break
        prev_eef = _get_eef_pos(env, obs)

    eef_pos = _get_eef_pos(env, obs)
    if eef_pos is None:
        eef_pos = prev_eef if prev_eef is not None else np.zeros(3)
    return eef_pos, bool(done), replay_images


def _run_with_step_perturbation(env, initial_state, fp16_chunks, w4a4_chunks,
                                  chunk_states, num_steps_wait, chunk_perturb,
                                  step_perturb, task_description,
                                  fp16_client, args):
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    replay_images = []

    for _ in range(num_steps_wait):
        try:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
        except ValueError:
            done = True
        if done:
            break

    if chunk_perturb < len(chunk_states):
        cs = chunk_states[chunk_perturb]
        mj = env.sim.get_state()
        mj.time = cs["time"]
        mj.qpos[:] = cs["qpos"]
        mj.qvel[:] = cs["qvel"]
        env.sim.set_state(mj)
        env.sim.forward()
        obs = env.get_observation()

    if chunk_perturb < len(fp16_chunks):
        for a in fp16_chunks[chunk_perturb][:step_perturb]:
            try:
                obs, reward, done, info = env.step(a)
            except ValueError:
                done = True
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            replay_images.append(img)
            if done:
                return _get_eef_pos(env, obs), True, replay_images

    w4a4_action = w4a4_chunks[chunk_perturb][step_perturb]
    try:
        obs, reward, done, info = env.step(w4a4_action)
    except ValueError:
        done = True
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    replay_images.append(img)
    if done:
        return _get_eef_pos(env, obs), True, replay_images

    prev_eef = _get_eef_pos(env, obs)
    MAX_STEPS = 600
    steps_taken = 0
    while not done and steps_taken < MAX_STEPS:
        try:
            element = _get_obs_element(obs, task_description, args.resize_size)
            action_chunk = fp16_client.infer(element)["actions"]
            actions = action_chunk[: args.replan_steps].tolist()
        except Exception:
            break
        for a in actions:
            try:
                obs, reward, done, info = env.step(a)
            except ValueError:
                done = True
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            replay_images.append(img)
            if done:
                break
            steps_taken += 1
        if done:
            break
        prev_eef = _get_eef_pos(env, obs)

    eef_pos = _get_eef_pos(env, obs)
    if eef_pos is None:
        eef_pos = prev_eef if prev_eef is not None else np.zeros(3)
    return eef_pos, bool(done), replay_images


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1, 2 & 5: Recording dispatcher
# ═══════════════════════════════════════════════════════════════════════════════
def _run_recording(args: Args, fp16_client, w4a4_client=None, w4a8_client=None, w4a16_client=None) -> None:
    np.random.seed(args.seed)
    mode_name = {1: "Basic (FP16)", 2: "Combined (FP16 + W4A4)", 5: "Combined (FP16 + W4A4 + W4A8 + W4A16)"}[args.mode]
    logging.info(f"Trajectory recording ({mode_name}, Mode {args.mode})")
    logging.info(f"  Task suite: {args.task_suite_name}")
    logging.info(f"  FP16 server: {args.host}:{args.port}")
    if args.mode == 2:
        logging.info(f"  W4A4 server:  {args.host}:{args.w4a4_port}")
    if args.mode == 5:
        logging.info(f"  W4A4 server:  {args.host}:{args.w4a4_port}")
        logging.info(f"  W4A8 server:  {args.host}:{args.w4a8_port}")
        logging.info(f"  W4A16 server: {args.host}:{args.w4a16_port}")
    logging.info(f"  Output dir: {args.output_dir}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logging.info(f"  Total tasks in suite: {num_tasks}")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = pathlib.Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    suffix = "trajectory" if args.mode == 1 else "combined"

    for task_id in tqdm.tqdm(range(num_tasks)):
        task = task_suite.get_task(task_id)
        task_description = task.language

        if args.task_name and args.task_name.lower() not in task_description.lower():
            continue

        init_states = task_suite.get_task_init_states(task_id)
        init_arr = init_states[0]
        initial_state = np.array(init_arr.tolist() if hasattr(init_arr, "tolist") else init_arr)

        seg = _task_segment(task_description)
        out_path = output_dir / f"rollout_{seg}_success_{suffix}.json"

        if args.skip_existing and out_path.exists():
            logging.info(f"  [Skip] {task_description[:60]}... (exists)")
            continue

        logging.info(f"  Recording: {task_description}")

        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:
            if args.mode == 1:
                result = _record_mode1(env, task, task_description, initial_state, fp16_client, args)
                record = {
                    "task_description": task_description,
                    "task_id": task_id,
                    "episode_idx": 0,
                    "success": bool(result["success"]),
                    "steps": result["steps"],
                }
            elif args.mode == 2:
                result = _record_mode2(env, task, task_description, initial_state, w4a4_client, fp16_client, args)
                record = {
                    "task_description": task_description,
                    "task_id": task_id,
                    "episode_idx": 0,
                    "success": bool(result["success"]),
                    "replan_steps": args.replan_steps,
                    "num_chunks": len(result["chunks"]),
                    "chunks": result["chunks"],
                }
            elif args.mode == 5:
                result = _record_mode5(env, task, task_description, initial_state,
                                       fp16_client, w4a4_client, w4a8_client, w4a16_client, args)
                record = {
                    "task_description": task_description,
                    "task_id": task_id,
                    "episode_idx": 0,
                    "success": bool(result["success"]),
                    "replan_steps": args.replan_steps,
                    "num_chunks": len(result["chunks"]),
                    "chunks": result["chunks"],
                }

            with open(out_path, "w") as f:
                json.dump(record, f, indent=2, cls=_NumpyEncoder)
            logging.info(f"    Saved: {out_path}")

            vid_suffix = f"{suffix}_success" if result["success"] else f"{suffix}_failure"
            _save_video(video_dir, f"rollout_{seg}", vid_suffix, result["replay_images"], result["success"])

        finally:
            env.close()

    logging.info("Done.")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 & 4: Perturbation dispatcher
# ═══════════════════════════════════════════════════════════════════════════════
def _run_perturbation(args: Args, fp16_client) -> None:
    np.random.seed(args.seed)
    mode_name = {3: "Chunk-level", 4: "Step-level"}[args.mode]
    logging.info(f"{mode_name} perturbation sensitivity evaluation (Mode {args.mode})")
    logging.info(f"  Server: {args.host}:{args.port}")
    logging.info(f"  Combined dir: {args.combined_dir}")
    logging.info(f"  Output dir: {args.output_dir}")

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

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = pathlib.Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    for traj_path in tqdm.tqdm(combined_paths):
        traj_name = traj_path.stem.replace("_combined", "")
        suffix = "step_perturb" if args.mode == 4 else "chunk_perturb"
        out_path = output_dir / f"{traj_name}_{suffix}.json"

        if out_path.exists():
            logging.info(f"  [Skip] {traj_name} already processed")
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

        fp16_chunks = [c["fp16_actions"] for c in chunks]
        w4a4_chunks = [c["w4a4_actions"] for c in chunks]
        chunk_states = [c.get("mujoco_state") for c in chunks]

        task = task_suite.get_task(task_id)
        init_qpos = task_suite.get_task_init_states(task_id)[0]
        if isinstance(init_qpos, np.ndarray):
            init_qpos = init_qpos.tolist()
        initial_state = np.array(init_qpos)

        env0, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        baseline_eef, baseline_success, baseline_images = _run_baseline_replay(
            env0, initial_state, fp16_chunks, chunk_states, args.num_steps_wait
        )
        env0.close()
        logging.info(f"    Baseline: success={baseline_success}, eef={baseline_eef.round(4)}")
        _save_video(video_dir, traj_name, "baseline", baseline_images, baseline_success)

        num_chunks = len(fp16_chunks)

        if args.mode == 3:
            chunk_results = []
            for k in range(num_chunks):
                env_k, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                try:
                    perturb_eef, perturb_success, perturb_images = _run_with_chunk_perturbation(
                        env_k, initial_state, w4a4_chunks,
                        chunk_states, args.num_steps_wait, k,
                        task_description, fp16_client, args
                    )
                finally:
                    env_k.close()

                _save_video(video_dir, traj_name, f"chunk{k}", perturb_images, perturb_success)

                fp16_chunk = fp16_chunks[k]
                w4a4_chunk = w4a4_chunks[k]
                chunk_action_diffs = [
                    float(np.linalg.norm(np.array(w) - np.array(f)))
                    for w, f in zip(w4a4_chunk, fp16_chunk)
                ]
                chunk_action_diff = sum(chunk_action_diffs) / len(chunk_action_diffs)
                eef_delta = float(np.linalg.norm(perturb_eef - baseline_eef))
                chunk_results.append({
                    "chunk_idx": k,
                    "success": perturb_success,
                    "eef": perturb_eef.tolist(),
                    "eef_delta": eef_delta,
                    "eef_delta_normalized": eef_delta / max(float(np.linalg.norm(baseline_eef)), 1e-6),
                    "chunk_action_diff_avg": chunk_action_diff,
                })
                logging.info(
                    f"    chunk={k}: success={perturb_success}, "
                    f"eef_delta={eef_delta:.4f}, chunk_action_diff={chunk_action_diff:.4f}"
                )

            result = {
                "mode": 3,
                "task_description": task_description,
                "baseline_success": baseline_success,
                "baseline_eef": baseline_eef.tolist(),
                "num_chunks": num_chunks,
                "chunk_results": chunk_results,
            }

        elif args.mode == 4:
            step_results = []
            for k in range(num_chunks):
                num_steps = len(fp16_chunks[k])
                for s in range(num_steps):
                    env_ks, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                    try:
                        perturb_eef, perturb_success, perturb_images = _run_with_step_perturbation(
                            env_ks, initial_state, fp16_chunks, w4a4_chunks,
                            chunk_states, args.num_steps_wait, k, s,
                            task_description, fp16_client, args
                        )
                    finally:
                        env_ks.close()

                    _save_video(video_dir, traj_name, f"chunk{k}_step{s}", perturb_images, perturb_success)

                    eef_delta = float(np.linalg.norm(perturb_eef - baseline_eef))
                    fp16_action = fp16_chunks[k][s]
                    w4a4_action = w4a4_chunks[k][s]
                    action_diff = float(np.linalg.norm(np.array(w4a4_action) - np.array(fp16_action)))
                    action_diff_norm = action_diff / max(float(np.linalg.norm(fp16_action)), 1e-6)
                    step_results.append({
                        "chunk_idx": k,
                        "step_idx": s,
                        "success": perturb_success,
                        "eef": perturb_eef.tolist(),
                        "eef_delta": eef_delta,
                        "eef_delta_normalized": eef_delta / max(float(np.linalg.norm(baseline_eef)), 1e-6),
                        "action_diff": action_diff,
                        "action_diff_normalized": action_diff_norm,
                    })
                    logging.info(
                        f"    chunk={k}, step={s}: success={perturb_success}, "
                        f"eef_delta={eef_delta:.4f}, action_diff={action_diff:.4f}"
                    )

            result = {
                "mode": 4,
                "task_description": task_description,
                "baseline_success": baseline_success,
                "baseline_eef": baseline_eef.tolist(),
                "num_chunks": num_chunks,
                "step_results": step_results,
            }

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, cls=_NumpyEncoder)

    logging.info("Done.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main(args: Args) -> None:
    if args.mode == 1:
        fp16_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        _run_recording(args, fp16_client)

    elif args.mode == 2:
        fp16_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        w4a4_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.w4a4_port)
        _run_recording(args, fp16_client, w4a4_client)

    elif args.mode == 5:
        fp16_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        w4a4_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.w4a4_port)
        w4a8_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.w4a8_port)
        w4a16_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.w4a16_port)
        _run_recording(args, fp16_client, w4a4_client, w4a8_client, w4a16_client)

    elif args.mode in (3, 4):
        fp16_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        _run_perturbation(args, fp16_client)

    else:
        logging.error(f"Unknown mode: {args.mode}. Use 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(main)
