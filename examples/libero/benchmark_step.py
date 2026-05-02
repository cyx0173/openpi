#!/usr/bin/env python3
"""Benchmark: measure per-step overhead breakdown for LIBERO simulation.

Scenarios:
  1. step() only                    -- pure physics
  2. step() + render               -- physics + GPU rendering
  3. step() + render + preprocess -- physics + rendering + CPU image ops
  4. step() + render + preprocess + real inference  -- full pipeline

Usage:
  uv run python examples/libero/benchmark_step.py --task_suite_name libero_10 --num_steps 100
"""
import collections
import dataclasses
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/libero"))

import numpy as np
import torch

_original_torch_load = torch.load
def _safe_legacy_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_legacy_load

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

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_10"
    num_steps: int = 100
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    from libero.libero import get_libero_path
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _get_init_state(task_suite, task_id, episode_idx=0):
    initial_states = task_suite.get_task_init_states(task_id)
    return initial_states[episode_idx]


def _preprocess(obs, resize_size):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    return img, wrist_img


def run_benchmark(args):
    np.random.seed(42)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(0)
    initial_state = _get_init_state(task_suite, 0)

    print(f"\n{'='*70}")
    print(f"Benchmark: LIBERO overhead breakdown")
    print(f"Task suite: {args.task_suite_name}")
    print(f"Task: {task.language[:70]}")
    print(f"Resolution: {args.resize_size}x{args.resize_size}, replan every {args.replan_steps} steps")
    print(f"{'='*70}\n")

    # ── Scenario 1: step() only ────────────────────────────────────────────
    print("Scenario 1: step() only (no rendering, no inference)")
    print("-" * 60)
    env1, _ = _get_libero_env(task, 256, 42)
    env1.reset()
    env1.set_init_state(initial_state)

    for _ in range(10):
        env1.step([0.0]*6 + [-1.0])

    times = []
    for _ in range(args.num_steps):
        t = time.perf_counter()
        _, reward, done, info = env1.step([0.0]*6 + [-1.0])
        times.append(time.perf_counter() - t)
        if done:
            env1.reset()
            env1.set_init_state(initial_state)

    step_only = np.mean(times) * 1000
    print(f"  step() only:  {step_only:.2f} ms/step  (std={np.std(times)*1000:.2f} ms)")
    env1.close()

    # ── Scenario 2: step() + render ─────────────────────────────────────────
    print(f"\nScenario 2: step() + camera render (256x256, 2 cameras)")
    print("-" * 60)
    env2, _ = _get_libero_env(task, 256, 42)
    env2.reset()
    env2.set_init_state(initial_state)

    for _ in range(10):
        env2.step([0.0]*6 + [-1.0])

    times = []
    for _ in range(args.num_steps):
        t = time.perf_counter()
        obs, reward, done, info = env2.step([0.0]*6 + [-1.0])
        _ = obs["agentview_image"]
        _ = obs["robot0_eye_in_hand_image"]
        times.append(time.perf_counter() - t)
        if done:
            env2.reset()
            env2.set_init_state(initial_state)

    step_render = np.mean(times) * 1000
    print(f"  step() + render:  {step_render:.2f} ms/step")
    print(f"  Render overhead:  {step_render - step_only:.2f} ms/step")
    env2.close()

    # ── Scenario 3: step() + render + preprocess ─────────────────────────────
    print(f"\nScenario 3: step() + render + image preprocess")
    print("-" * 60)
    env3, _ = _get_libero_env(task, args.resize_size, 42)
    env3.reset()
    env3.set_init_state(initial_state)

    for _ in range(10):
        env3.step([0.0]*6 + [-1.0])

    step_times, prep_times = [], []
    for _ in range(args.num_steps):
        t0 = time.perf_counter()
        obs, reward, done, info = env3.step([0.0]*6 + [-1.0])
        t1 = time.perf_counter()
        img, wrist_img = _preprocess(obs, args.resize_size)
        t2 = time.perf_counter()
        step_times.append(t1 - t0)
        prep_times.append(t2 - t1)
        if done:
            env3.reset()
            env3.set_init_state(initial_state)

    step_p3 = np.mean(step_times) * 1000
    prep_ms = np.mean(prep_times) * 1000
    step_render_prep = step_p3 + prep_ms
    print(f"  step():          {step_p3:.2f} ms/step")
    print(f"  preprocess():    {prep_ms:.2f} ms/step")
    print(f"  Total:           {step_render_prep:.2f} ms/step")
    env3.close()

    # ── Scenario 4: full pipeline with real inference ─────────────────────────
    print(f"\nScenario 4: full pipeline (step + render + preprocess + VLM inference)")
    print("-" * 60)
    print(f"Connecting to ws://{args.host}:{args.port}...")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    env4, task_desc = _get_libero_env(task, args.resize_size, 42)
    env4.reset()
    env4.set_init_state(initial_state)

    # Warmup: get real observation from env
    obs_warm = env4.step([0.0]*6 + [-1.0])[0]
    img_w, wrist_w = _preprocess(obs_warm, args.resize_size)
    warm_element = {
        "observation/image": img_w,
        "observation/wrist_image": wrist_w,
        "observation/state": np.concatenate((
            obs_warm["robot0_eef_pos"],
            _quat2axisangle(obs_warm["robot0_eef_quat"]),
            obs_warm["robot0_gripper_qpos"],
        )),
        "prompt": str(task_desc),
    }
    client.infer(warm_element)

    # Benchmark: measure per-step breakdown
    action_plan = collections.deque()
    step_t, prep_t, inf_t = [], [], []
    inf_call_times = []  # individual inference call durations
    t = 0
    max_steps = args.num_steps
    dummy_action = [0.0]*6 + [-1.0]

    for _ in range(max_steps):
        t0 = time.perf_counter()
        obs4, reward, done, info = env4.step(dummy_action)
        t1 = time.perf_counter()
        img, wrist_img = _preprocess(obs4, args.resize_size)
        t2 = time.perf_counter()

        inf_start = time.perf_counter()
        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate((
                    obs4["robot0_eef_pos"],
                    _quat2axisangle(obs4["robot0_eef_quat"]),
                    obs4["robot0_gripper_qpos"],
                )),
                "prompt": str(task_desc),
            }
            result = client.infer(element)
            action_chunk = result["actions"]
            action_plan.extend(action_chunk[:args.replan_steps])
            inf_call_times.append(time.perf_counter() - inf_start)

        _ = action_plan.popleft()
        t3 = time.perf_counter()

        step_t.append(t1 - t0)
        prep_t.append(t2 - t1)
        inf_t.append(t3 - t2)

        if done:
            env4.reset()
            env4.set_init_state(initial_state)
        t += 1

    avg_step = np.mean(step_t) * 1000
    avg_prep = np.mean(prep_t) * 1000
    # Only count actual inference calls
    avg_infer_per_call = np.mean(inf_call_times) * 1000 if inf_call_times else 0
    # amortized: inference only happens every replan_steps
    avg_infer = avg_infer_per_call / args.replan_steps
    avg_total = avg_step + avg_prep + avg_infer

    print(f"  step() (sim + render):  {avg_step:.2f} ms/step")
    print(f"  preprocess():            {avg_prep:.2f} ms/step")
    print(f"  VLM inference:           {avg_infer_per_call:.1f} ms/call  ({avg_infer:.1f} ms/step amortized)")
    print(f"  Total per action step:  {avg_total:.1f} ms/step")
    print(f"  # inference calls: {len(inf_call_times)}, calls per replan: {args.replan_steps}")

    env4.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY  (all values in ms/step)")
    print(f"{'='*70}")
    print(f"  step() only:                   {step_only:.2f}")
    print(f"  step() + render:              {step_render:.2f}  (+ {step_render-step_only:.2f} render)")
    print(f"  step() + render + preprocess:  {step_render_prep:.2f}  (+ {prep_ms:.2f} preprocess)")
    print(f"  FULL PIPELINE:                 {avg_total:.1f}  (inference={avg_infer:.1f} amortized)")
    print(f"\n  Bottleneck analysis:")
    if avg_infer_per_call > avg_step * 2:
        print(f"  >>> VLM INFERENCE is the bottleneck ({avg_infer/avg_total*100:.0f}% of total time)")
    elif step_render - step_only > avg_step * 0.3:
        print(f"  >>> RENDERING is a significant bottleneck")
    else:
        print(f"  >>> Time is spread across all stages")
    print(f"  Render overhead vs step-only: {(step_render-step_only)/step_only*100:.0f}%")
    print(f"  Preprocess overhead: {prep_ms/avg_total*100:.0f}% of full pipeline")
    print(f"  Inference overhead:  {avg_infer/avg_total*100:.0f}% of full pipeline (amortized)")
    print(f"  Raw inference call:  {avg_infer_per_call:.0f} ms/call ({avg_infer_per_call/avg_step:.1f}x step time)")


def _quat2axisangle(quat):
    import math
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_benchmark(args)
