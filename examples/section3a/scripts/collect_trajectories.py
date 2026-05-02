"""
Stage 1: collect_trajectories.py
================================
Record BF16 (full-precision) trajectories for LIBERO tasks.

This script is derived from examples/libero/main.py with trajectory-saving added.
Only one episode per task is collected.

Usage:
  # Terminal 1: start policy server
  python scripts/serve_policy.py --env LIBERO --port 8000

  # Terminal 2: collect trajectories
  python examples/section3a/scripts/collect_trajectories.py \
    --port 8000 \
    --out_dir data/section3a/trajectories \
    --task_suite libero_spatial
"""

import collections
import dataclasses
import logging
import math
import pathlib
import sys

import numpy as np
import torch
import imageio
import tqdm
import tyro

import os
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")


def setup_env():
    import torch
    import sys as _sys

    _orig = torch.load
    def _safe(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return _orig(*a, **kw)
    torch.load = _safe
    print("[patch] torch.load: weights_only=False")

    import robosuite.models.bases.robot_base_factory as rbf_mod
    def _patched_robot_base(name, idn=0):
        if name in rbf_mod.BASE_MAPPING:
            return rbf_mod.BASE_MAPPING[name](idn=idn)
        return rbf_mod.BASE_MAPPING["NullBase"](idn=idn)
    rbf_mod.robot_base_factory = _patched_robot_base
    import robosuite.models.bases
    robosuite.models.bases.robot_base_factory = _patched_robot_base
    import robosuite.robots.robot
    robosuite.robots.robot.robot_base_factory = _patched_robot_base
    for _mod_name, _mod in list(_sys.modules.items()):
        if _mod is not None:
            setattr(_mod, "robot_base_factory", _patched_robot_base)
    print("[patch] robot_base_factory: fallback to NullBase")

    import robosuite.models.robots.robot_model as rm_mod
    _create_robot_orig = rm_mod.create_robot
    def _safe_create_robot(robot_name, *args, **kwargs):
        try:
            return _create_robot_orig(robot_name, *args, **kwargs)
        except KeyError:
            import robosuite.models.robots.panda_model as pm
            return pm.Panda(idn=kwargs.get("idn", 0))
    rm_mod.create_robot = _safe_create_robot
    import robosuite.models.robots
    robosuite.models.robots.robot_model.create_robot = _safe_create_robot
    for _mod_name, _mod in list(_sys.modules.items()):
        if _mod is not None and hasattr(_mod, "create_robot"):
            _mod.create_robot = _safe_create_robot
        if _mod is not None and hasattr(_mod, "robot_base_factory"):
            _mod.robot_base_factory = _patched_robot_base
    print("[patch] create_robot: fallback to Panda for unknown robot types")


setup_env()


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "libero"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "openpi-client" / "src"))

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_ACTION_DIM = 7

@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    video_out_path: str = "data/libero/videos"
    out_dir: str = "data/section3a/trajectories"
    seed: int = 7
    task_ids: str = ""


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


@tyro.cli
def collect_trajectories(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks_in_suite} tasks)")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    else:
        task_ids = list(range(num_tasks_in_suite))

    task_max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = task_max_steps.get(args.task_suite_name, 400)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"Connected to policy server at {args.host}:{args.port}")

    total_episodes, total_successes = 0, 0
    successes = []

    pbar = tqdm.tqdm(total=len(task_ids), desc="Tasks", unit="task")
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_name = f"task_{task_id:02d}"
        pbar.set_postfix_str(f"{task_description[:50]}...")
        print(f"[{task_name}] {task_description[:80]}")

        ep_dir = out_dir / task_name
        ep_dir.mkdir(parents=True, exist_ok=True)

        env.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(initial_states[0])

        t = 0
        done = False
        info = {}

        frames = []
        states = []
        actions = []

        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                frames.append(img)

                state = np.concatenate([
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]).astype(np.float32)
                states.append(state)

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                actions.append(np.array(action[:LIBERO_ACTION_DIM], dtype=np.float32))

                obs, reward, done, info = env.step(action.tolist())

                if done:
                    total_successes += 1
                    break
                t += 1

            except Exception as e:
                logging.error(f"Caught exception at step {t}: {e}")
                pbar.update(1)
                env.close()
                pbar.close()
                sr = total_successes / total_episodes if total_episodes else 0.0
                print(f"=== Summary ===")
                print(f"  Tasks: {total_episodes}")
                print(f"  Success: {total_successes}/{total_episodes} ({sr:.1%})")
                print(f"  Output: {out_dir}")
                return

        traj = {
            "frames": np.array(frames, dtype=np.uint8),
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "prompt": task_description,
            "success": bool(info.get("success", False)),
            "final_step": t,
        }
        np.save(ep_dir / "trajectory.npy", traj)

        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
        suffix = "success" if done else "failure"
        task_segment = str(task_description).replace(" ", "_").replace("/", "_")
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
            [np.asarray(x) for x in frames],
            fps=10,
        )

        print(
            f"  [{task_name}] {'SUCCESS' if done else 'FAIL'} | "
            f"steps={len(actions)}/{max_steps + args.num_steps_wait}"
        )
        successes.append(done)
        total_episodes += 1
        pbar.update(1)

        env.close()

    pbar.close()
    sr = total_successes / total_episodes if total_episodes else 0.0
    print(f"=== Summary ===")
    print(f"  Tasks: {total_episodes}")
    print(f"  Success: {total_successes}/{total_episodes} ({sr:.1%})")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = tyro.cli(Args)
    collect_trajectories(args)
