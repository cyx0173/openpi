"""
Stage 2: experiment.py
=====================
Section III-A of DyQ-VLA (arXiv:2603.07904):
"Step-wise Perturbation Analysis of Quantization Sensitivity"

This script faithfully reproduces the experiment described in Section III-A.
The key design (from the paper):

  For each time step t in a successful BF16 trajectory:
    1. Run W4A4 quantized model for steps 0..t-1 (full env control)
       -> accumulated quantization error at step t
    2. At step t: inject the BF16 (ground-truth) action instead
    3. Steps t+1..T: resume W4A4 model control
    4. Record:
         e_t  = ||action_W4A4[t] - action_BF16[t]||_2   (decision error at t)
         s_t  = D_T / e_t                               (sensitivity)
         success / failure of the episode

  The result is a fine-grained map of quantization sensitivity across
  the full episode timeline, enabling:
    - Success rate vs e_t curve
    - s_t temporal profile
    - Fine-grained vs coarse movement sensitivity comparison

Pipeline (mirrors the paper exactly):
  Stage 1 (collect_trajectories.py):
    Record BF16 baseline trajectories: obs + actions from server.
  Stage 2 (this file):
    For each (task, perturb_step):
      - W4A4 model: steps 0..t-1  (quantized, env-controlled)
      - BF16 inject: step t
      - W4A4 model: steps t+1..T  (quantized, env-controlled)
      -> success / e_t / s_t

Usage:
  python examples/section3a/scripts/experiment.py \
    --trajectory_dir data/section3a/trajectories \
    --checkpoint /share/chengyuxuan-local/openpi/pi05_libero_pytorch/model.safetensors \
    --out_dir data/section3a/results \
    --perturb_bits 4 \
    --num_samples 20 \
    --device cuda:0

Output:
  data/section3a/results/
    task_XX/
      actions_16b.npy       # BF16 model actions (all T steps)
      actions_4b.npy       # W4A4 model actions (all T steps)
      e_t.npy               # decision error at each step
      s_t.npy               # sensitivity at each step
      perturb_tXX.json      # perturbation result at step XX
      summary.json          # per-task aggregated results
    summary_bits4.json       # aggregate results across all tasks
"""

import os
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")

import argparse
import json
import logging
import math
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn
import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent / "third_party" / "libero"))

from openpi.models_pytorch import pi0_pytorch
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer

# Monkey-patch: replace the unreliable is_sparse check in preprocess_observation_pytorch
# with unconditional to_dense() call (to_dense is a no-op on dense tensors).
# This fixes RuntimeError: permute(sparse_coo) when sparse COO tensors reach the permute.
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
_orig_preprocess = _preprocessing.preprocess_observation_pytorch

def _patched_preprocess_observation_pytorch(observation, *, train=False,
        image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        image_resolution=(224, 224)):
    """Patched version: unconditionally call to_dense() on each image tensor,
    and fix resize_with_pad_torch dropping the batch dimension."""
    import openpi.shared.image_tools as _img_tools
    import torch.nn.functional as F
    batch_shape = observation.state.shape[:-1]
    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        try:
            image = image.to_dense()
        except RuntimeError:
            pass

        if image.ndim == 3:
            if image.shape[0] == 3:
                image = image.unsqueeze(0)
            elif image.shape[2] == 3:
                image = image.unsqueeze(0)
            else:
                image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)

        is_channels_first = image.ndim == 4 and image.shape[1] == 3
        if is_channels_first:
            image = image.permute(0, 2, 3, 1)
        elif image.ndim != 4:
            raise ValueError(f"Image {key} has unexpected shape {image.shape}, expected 4D")

        if image.shape[1:3] != image_resolution:
            image = _img_tools.resize_with_pad_torch(image, *image_resolution)
            # Workaround: resize_with_pad_torch incorrectly squeezes the batch
            # dimension when channels_last (shape[-1] <= 4). Restore it.
            if image.ndim == 3:
                image = image.unsqueeze(0)

        if train:
            image = image / 2.0 + 0.5
            if "wrist" not in key:
                h, w = image.shape[1:3]
                ch = int(h * 0.95); cw = int(w * 0.95)
                mh = h - ch; mw = w - cw
                if mh > 0 and mw > 0:
                    sh = torch.randint(0, mh + 1, (1,), device=image.device)
                    sw = torch.randint(0, mw + 1, (1,), device=image.device)
                    image = image[:, sh:sh+ch, sw:sw+cw, :]
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                ang = torch.rand(1, device=image.device) * 10 - 5
                if torch.abs(ang) > 0.1:
                    rad = ang * math.pi / 180.0
                    ca, sa = torch.cos(rad), torch.sin(rad)
                    gx = torch.linspace(-1, 1, w, device=image.device)
                    gy = torch.linspace(-1, 1, h, device=image.device)
                    gy, gx = torch.meshgrid(gy, gx, indexing="ij")
                    gx = gx.unsqueeze(0).expand(image.shape[0], -1, -1)
                    gy = gy.unsqueeze(0).expand(image.shape[0], -1, -1)
                    gxr = gx * ca - gy * sa
                    gyr = gx * sa + gy * ca
                    grid = torch.stack([gxr, gyr], dim=-1)
                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros", align_corners=False
                    ).permute(0, 2, 3, 1)
            bf = 0.7 + torch.rand(1, device=image.device) * 0.6
            image = image * bf
            cf = 0.6 + torch.rand(1, device=image.device) * 0.8
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * cf + mean
            sf = 0.5 + torch.rand(1, device=image.device) * 1.0
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * sf
            image = torch.clamp(image, 0, 1)
            image = image * 2.0 - 1.0

        if is_channels_first:
            image = image.permute(0, 3, 1, 2)
        out_images[key] = image

    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    class _SimpleObs:
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)

    return _SimpleObs(
        images=out_images, image_masks=out_masks,
        state=observation.state, tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask, token_loss_mask=observation.token_loss_mask,
    )

_preprocessing.preprocess_observation_pytorch = _patched_preprocess_observation_pytorch
from openpi.training import config as _config

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("section3a")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ACTION_DIM = 7
ENV_RESOLUTION = 256
MODEL_RESOLUTION = 224
NUM_STEPS_DENOISE = 10
MAX_DISTurb_STEPS = 20      # D_T: window for s_t computation (paper: max distance traveled in a window)


# ===========================================================================
# Environment setup
# ===========================================================================

def setup_env():
    import torch
    _orig = torch.load
    def _safe(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return _orig(*a, **kw)
    torch.load = _safe
    print("[patch] torch.load: weights_only=False")


# ===========================================================================
# FakeQuant (consistent with fake_quant_inference.py)
# ===========================================================================

def compute_scale_per_channel(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """
    逐通道 (Per-Channel) 对称比例尺。
    weight shape: (out_features, in_features)
    返回 shape: (out_features, 1)
    """
    levels = (2 ** (bits - 1)) - 1
    # 沿 in_features 维度寻找最大值，保留 out_features 维度的独立性
    max_val = torch.max(torch.abs(weight), dim=1, keepdim=True)[0]
    # 避免除以 0 的情况
    scale = torch.where(max_val < 1e-9, torch.ones_like(max_val), max_val / levels)
    return scale

class FakeQuantLinear(nn.Module):
    """
    业界标准的 LLM 伪量化算子：
    - 权重 (Weight): Per-Channel 逐通道量化
    - 激活 (Activation): Per-Token 逐词元量化 (如果开启)
    """
    def __init__(self, linear: nn.Linear, bits: int, quantize_activations: bool = False):
        super().__init__()
        self._linear = linear
        self._bits = bits
        self._quantize_activations = quantize_activations
        # 预计算并注册权重的 per-channel scale
        self.register_buffer("_w_scale", compute_scale_per_channel(linear.weight.data, bits))

    @property
    def in_features(self): return self._linear.in_features

    @property
    def out_features(self): return self._linear.out_features

    @property
    def bias(self): return self._linear.bias

    @property
    def weight(self): return self._linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._bits >= 16:
            return self._linear(x)

        levels = (2 ** (self._bits - 1)) - 1
        q_min = -levels - 1  # 例如 4-bit 时为 -8
        q_max = levels       # 例如 4-bit 时为 7

        # --- 1. 权重 Per-Channel 量化 ---
        w = self._linear.weight
        w_q = torch.round(w / self._w_scale)
        w_q = torch.clamp(w_q, q_min, q_max)
        w_deq = w_q * self._w_scale

        # --- 2. 激活 Per-Token 量化 (如果开启) ---
        if self._quantize_activations:
            # x shape: (..., in_features)
            # 沿特征维度寻找最大值，让每个 token (图像patch/文本词) 拥有独立的 scale
            x_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
            x_scale = torch.where(x_max < 1e-9, torch.ones_like(x_max), x_max / levels)
            x_q = torch.round(x / x_scale)
            x_q = torch.clamp(x_q, q_min, q_max)
            x_deq = x_q * x_scale
        else:
            x_deq = x

        # --- 3. 执行线性变换 ---
        return torch.nn.functional.linear(x_deq, w_deq, self._linear.bias)

def apply_fake_quant(model: nn.Module, bits: int, quantize_activations: bool = False):
    """递归替换 Linear 层。注意：学术界推荐 VLA 模型默认使用 W4A16 (关闭激活量化)"""
    def _rec(module, prefix=""):
        for name in list(module._modules.keys()):
            child = module._modules[name]
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                module._modules[name] = FakeQuantLinear(child, bits, quantize_activations)
            elif hasattr(child, "_modules") and len(child._modules) > 0:
                _rec(child, full)
    _rec(model)

def remove_fake_quant(model: nn.Module):
    """恢复原始 nn.Linear 层"""
    def _rec(module, prefix=""):
        for name in list(module._modules.keys()):
            child = module._modules[name]
            if isinstance(child, FakeQuantLinear):
                module._modules[name] = child._linear
            elif hasattr(child, "_modules") and len(child._modules) > 0:
                _rec(child, prefix)
    _rec(model)
# ===========================================================================
# Observation builder (matches collect_trajectories.py + main.py format)
# ===========================================================================

def _quat2axisangle(quat):
    quat = np.array(quat)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    angle = 2 * np.arccos(np.clip(quat[3], -1, 1))
    s = np.sqrt(max(0, 1 - quat[3] ** 2))
    if s < 1e-6:
        return np.zeros(3)
    return (quat[:3] / s) * angle


def build_observation(
    frame: np.ndarray,
    state: np.ndarray,
    prompt: str,
    tokenizer: _tokenizer.PaligemmaTokenizer,
    model_config,
    device: torch.device,
) -> _model.Observation:
    """
    Build model.Observation from a single-step frame and state.
    Frame: uint8 [H, W, C], preprocessed by collect_trajectories.py.
    State: float32 [7] from collect_trajectories.py.
    """
    img = np.asarray(frame)
    if img.dtype == np.uint8:
        img_f = img.astype(np.float32) / 255.0 * 2.0 - 1.0
    else:
        img_f = img.astype(np.float32)
    img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0).to(device)

    state_vec = np.asarray(state, dtype=np.float32)
    if len(state_vec) == 7:
        state_vec = np.concatenate([state_vec, [0.0]])

    state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(device)

    images = {
        "base_0_rgb": img_t,
        "left_wrist_0_rgb": img_t.clone(),
        "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device),
    }
    image_masks = {
        "base_0_rgb": torch.tensor([True], device=device),
        "left_wrist_0_rgb": torch.tensor([True], device=device),
        "right_wrist_0_rgb": torch.tensor([False], device=device),
    }

    if prompt:
        state_for_tok = state_vec[:7]
        tokens, mask = tokenizer.tokenize(prompt, state_for_tok)
    else:
        tokens = np.zeros(model_config.max_token_len, dtype=np.int64)
        mask = np.zeros(model_config.max_token_len, dtype=np.bool_)

    tokenized_prompt = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
    tokenized_prompt_mask = torch.from_numpy(mask).bool().unsqueeze(0).to(device)

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state_t,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )


# ===========================================================================
# Inference utilities
# ===========================================================================

@torch.no_grad()
def run_inference_batch(
    model,
    frames: np.ndarray,
    states: np.ndarray,
    prompt: str,
    tokenizer: _tokenizer.PaligemmaTokenizer,
    model_config,
    device: torch.device,
    num_steps: int = NUM_STEPS_DENOISE,
    seed: int = 42,
    label: str = "Inference",
) -> np.ndarray:
    """
    Run model inference on all T steps of a recorded episode.
    Returns: actions [T, action_dim], first action of each denoising output.
    """
    torch.manual_seed(seed)
    T = len(frames)
    actions_list = []

    pbar = tqdm.tqdm(total=T, desc=label, leave=False)
    for t in range(T):
        obs = build_observation(
            frame=frames[t],
            state=states[t],
            prompt=prompt,
            tokenizer=tokenizer,
            model_config=model_config,
            device=device,
        )
        action_seq = model.sample_actions(device, obs, num_steps=num_steps)
        actions_list.append(action_seq[0, 0].cpu().numpy())
        pbar.update(1)
    pbar.close()

    return np.array(actions_list, dtype=np.float32)


# ===========================================================================
# Environment helpers
# ===========================================================================

def get_env(task, resolution=ENV_RESOLUTION, seed=42):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _env_reset(env, seed):
    """Reset env with warm-up steps (consistent with collect_trajectories.py)."""
    np.random.seed(seed)
    env.reset()
    for _ in range(10):
        env.step(LIBERO_DUMMY_ACTION)


def _preprocess_image(img):
    """Preprocess image exactly as in collect_trajectories.py."""
    img = np.ascontiguousarray(img[::-1, ::-1])
    return img


def _build_obs_from_env(env, prompt: str):
    """Build model observation from live env observation."""
    obs = env.get_observation()
    img = _preprocess_image(obs["agentview_image"])
    wrist_img = _preprocess_image(obs["robot0_eye_in_hand_image"])

    state = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": prompt,
    }


# ===========================================================================
# Paper-exact perturbation replay
# ===========================================================================

def replay_with_bf16_control_and_quantized_injection(
    env,
    task_description: str,
    model,
    tokenizer,
    model_config,
    device: torch.device,
    actions_quant: np.ndarray,  # 预计算的 W4A4 动作
    perturb_step: int,
    max_steps: int,
    init_state,                 # 必须传入正确的初始状态
    num_steps: int = NUM_STEPS_DENOISE,
) -> dict:
    # 1. 严格对齐初始状态
    env.reset()
    env.set_init_state(init_state)
    for _ in range(10):  # 等待物体下落
        env.step(LIBERO_DUMMY_ACTION)

    done = False
    step = 0
    
    while not done and step < max_steps:
        # --- 步骤 t: 注入预先计算好的 4-bit 动作 ---
        if step == perturb_step and step < len(actions_quant):
            action = actions_quant[step][:LIBERO_ACTION_DIM].tolist()
            
        # --- 步骤 0..t-1 及 t+1..T: 实时 BF16 控制 ---
        else:
            live_obs = _build_obs_from_env(env, task_description)
            model_obs = _build_model_obs_from_dict(
                live_obs, tokenizer, model_config, device
            )
            # 注意：传入的 model 必须是未经过 FakeQuant 的纯 BF16 模型
            action_seq = model.sample_actions(
                device, model_obs, num_steps=num_steps
            )
            action = action_seq[0, 0].cpu().numpy()
            action = action[:LIBERO_ACTION_DIM].tolist()

        obs, reward, done, info = env.step(action)
        step += 1

    # 记录终端位置，用于后续计算 D_T
    final_pos = obs["robot0_eef_pos"].copy()
    
    return {
        "success": bool(info.get("success", False)),
        "final_step": step,
        "final_pos": final_pos,
        "perturb_step": perturb_step,
    }


def _build_model_obs_from_dict(obs_dict, tokenizer, model_config, device):
    """Convert server-style observation dict to model.Observation."""

    def _make_image_tensor(img_np):
        """Convert image numpy array to CHW float32 tensor on device."""
        img = np.asarray(img_np)
        if img.dtype == np.uint8:
            img_f = img.astype(np.float32) / 255.0 * 2.0 - 1.0
        else:
            img_f = img.astype(np.float32)
        if img_f.shape[2] == 3:  # HWC -> CHW
            img_f = img_f.transpose(2, 0, 1)
        img_t = torch.from_numpy(img_f).unsqueeze(0).to(device)
        return img_t

    img_t = _make_image_tensor(obs_dict["observation/image"])
    wrist_t = _make_image_tensor(obs_dict["observation/wrist_image"])

    state_vec = np.asarray(obs_dict["observation/state"], dtype=np.float32)
    if len(state_vec) == 7:
        state_vec = np.concatenate([state_vec, [0.0]])
    state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(device)

    images = {
        "base_0_rgb": img_t,
        "left_wrist_0_rgb": wrist_t,
        "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device),
    }
    image_masks = {
        "base_0_rgb": torch.tensor([True], device=device),
        "left_wrist_0_rgb": torch.tensor([True], device=device),
        "right_wrist_0_rgb": torch.tensor([False], device=device),
    }

    prompt = obs_dict.get("prompt", "")
    if prompt:
        state_for_tok = state_vec[:7]
        tokens, mask = tokenizer.tokenize(prompt, state_for_tok)
    else:
        tokens = np.zeros(model_config.max_token_len, dtype=np.int64)
        mask = np.zeros(model_config.max_token_len, dtype=np.bool_)

    tokenized_prompt = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
    tokenized_prompt_mask = torch.from_numpy(mask).bool().unsqueeze(0).to(device)

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state_t,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )


# ===========================================================================
# Full experiment for one task
# ===========================================================================
def run_task_experiment(
    task_id: int,
    task_suite,
    traj_dir: pathlib.Path,
    out_dir: pathlib.Path,
    model,
    tokenizer,
    model_config,
    device: torch.device,
    perturb_bits: int,
    num_samples: int,
    max_steps: int,
    seed: int,
    resume: bool = False,
) -> dict | None:
    
    task_name = f"task_{task_id:02d}"
    ep_dir = traj_dir / task_name
    traj_path = ep_dir / "trajectory.npy"

    if not traj_path.exists():
        logger.warning(f"  [{task_name}] No trajectory at {traj_path}, skipping")
        return None

    traj = np.load(traj_path, allow_pickle=True).item()
    T = len(traj.get("actions", []))

    if T == 0:
        logger.warning(f"  [{task_name}] Empty trajectory, skipping")
        return None

    logger.info(f"  [{task_name}] T={T}, bits={perturb_bits}")

    task_out = out_dir / task_name
    task_out.mkdir(parents=True, exist_ok=True)

    frames = np.array(traj["frames"], dtype=np.uint8)
    states = np.array(traj["states"], dtype=np.float32)
    prompt = traj.get("prompt", "") or traj.get("task_description", "")

    # --- BF16 model inference ---
    actions_16b_path = task_out / "actions_16b.npy"
    t0 = time.time()
    if resume and actions_16b_path.exists():
        actions_bf16 = np.load(actions_16b_path)
        logger.info(f"    [CACHE] Loaded BF16 actions from {actions_16b_path.name}")
    else:
        remove_fake_quant(model) # 确保是 BF16
        actions_bf16 = run_inference_batch(
            model, frames, states, prompt,
            tokenizer, model_config, device,
            num_steps=NUM_STEPS_DENOISE, seed=seed,
            label=f"[task_{task_id:02d}] BF16",
        )
        np.save(actions_16b_path, actions_bf16)
    fp16_time = time.time() - t0
    logger.info(f"    BF16 inference: {fp16_time:.1f}s ({T} steps)")

    # --- W4A4 quantized model inference ---
    quant_path = task_out / f"actions_{perturb_bits}b.npy"
    t0 = time.time()
    if resume and quant_path.exists():
        actions_quant = np.load(quant_path)
        logger.info(f"    [CACHE] Loaded {perturb_bits}b actions from {quant_path.name}")
    else:
        apply_fake_quant(model, perturb_bits, quantize_activations=True)
        actions_quant = run_inference_batch(
            model, frames, states, prompt,
            tokenizer, model_config, device,
            num_steps=NUM_STEPS_DENOISE, seed=seed,
            label=f"[task_{task_id:02d}] W{perturb_bits}A{perturb_bits}",
        )
        remove_fake_quant(model) # 跑完立刻恢复，保证下面物理仿真是 BF16
        np.save(quant_path, actions_quant)
    quant_time = time.time() - t0
    logger.info(f"    W4A4 inference: {quant_time:.1f}s ({T} steps)")

    # --- 计算单步动作误差 e_t ---
    e_t = np.linalg.norm(
        actions_quant[:, :LIBERO_ACTION_DIM] - actions_bf16[:, :LIBERO_ACTION_DIM], axis=1
    )
    np.save(task_out / "e_t.npy", e_t)

    # --- Determine perturbation steps ---
    if T <= num_samples:
        perturb_steps = list(range(T))
    else:
        perturb_steps = sorted(set(int(x) for x in np.linspace(0, T - 1, num_samples)))

    # --- Load task + create env ---
    task = task_suite.get_task(task_id)
    env, _ = get_env(task, seed=seed + task_id)
    
    # 获取正确的初始状态和基线最终位置
    initial_states = task_suite.get_task_init_states(task_id)
    init_state = initial_states[0] # 假定轨迹对应第0个初始状态
    baseline_final_pos = states[-1, :3]

    # --- Determine which perturb_steps are already done ---
    done_steps = set()
    if resume:
        for p in perturb_steps:
            pfile = task_out / f"perturb_t{p:03d}.json"
            if pfile.exists():
                done_steps.add(p)
        if done_steps:
            remaining_steps = [p for p in perturb_steps if p not in done_steps]
            logger.info(f"    [RESUME] {len(done_steps)} perturb steps already done, {len(remaining_steps)} remaining")
        else:
            logger.info(f"    [RESUME] No cached perturb results found, running all")
            remaining_steps = perturb_steps
    else:
        remaining_steps = perturb_steps

    perturb_results = []
    logger.info(f"    Perturbation sweep: {len(remaining_steps)} steps...")
    pbar = tqdm.tqdm(remaining_steps, desc=f"[task_{task_id:02d}] Perturb", leave=False)
    
    for p_step in pbar:
        step_seed = seed + task_id * 10000 + p_step
        remove_fake_quant(model) # 严格确保每次扰动基线都是 BF16

        result = replay_with_bf16_control_and_quantized_injection(
            env=env,
            task_description=prompt,
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            device=device,
            actions_quant=actions_quant,
            perturb_step=p_step,
            max_steps=max_steps,
            init_state=init_state,
            num_steps=NUM_STEPS_DENOISE,
        )

        # --- 正确计算 D_T (终端偏差) 和 s_t (敏感度) ---
        D_T = float(np.linalg.norm(result["final_pos"] - baseline_final_pos))
        current_e_t = float(e_t[p_step])
        current_s_t = D_T / current_e_t if current_e_t > 1e-9 else 0.0

        result["e_t"] = current_e_t
        result["s_t"] = current_s_t
        result["D_T"] = D_T
        del result["final_pos"] # 移除 numpy 对象以支持 JSON 序列化

        perturb_results.append(result)

        sr_tag = "SUCCESS" if result["success"] else "FAIL"
        logger.info(
            f"      t={p_step:3d}/{T-1}: {sr_tag} | "
            f"e_t={result['e_t']:.6f} | s_t={result['s_t']:.4f}"
        )
        pbar.set_postfix_str(f"t={p_step} {sr_tag} e={result['e_t']:.4f}")
    pbar.close()
    env.close()

    # --- Save per-perturbation results ---
    for i, p_step in enumerate(remaining_steps):
        res = perturb_results[i]
        with open(task_out / f"perturb_t{p_step:03d}.json", "w") as f:
            json.dump(res, f, indent=2)

    # --- Merge new results with any cached ones ---
    all_perturb_results = {}
    for p in done_steps:
        with open(task_out / f"perturb_t{p:03d}.json") as f:
            all_perturb_results[p] = json.load(f)
    for i, p in enumerate(remaining_steps):
        all_perturb_results[p] = perturb_results[i]

    # --- 收集并保存全局 s_t 数组 ---
    s_t_array = np.zeros(T, dtype=np.float32)
    for p_step, r in all_perturb_results.items():
        s_t_array[int(p_step)] = r["s_t"]
    np.save(task_out / "s_t.npy", s_t_array)

    # --- Classify fine-grained vs coarse movements ---
    is_fine = classify_fine_vs_coarse(actions_bf16)
    m_t = compute_motion_fineness(actions_bf16).tolist()
    j_t = compute_angular_jerk(actions_bf16).tolist()

    # Aggregate fine vs coarse
    fine_success = fine_total = coarse_success = coarse_total = 0

    for p_step in perturb_steps:
        phase = "fine" if (p_step < len(is_fine) and is_fine[p_step]) else "coarse"
        res = all_perturb_results.get(p_step)
        if res is None:
            continue
        if phase == "fine":
            fine_total += 1
            if res["success"]: fine_success += 1
        else:
            coarse_total += 1
            if res["success"]: coarse_success += 1

    fine_sr = fine_success / fine_total if fine_total > 0 else 0.0
    coarse_sr = coarse_success / coarse_total if coarse_total > 0 else 0.0

    summary = {
        "task_name": task_name,
        "task_id": task_id,
        "T": T,
        "perturb_bits": perturb_bits,
        "num_perturb_steps": len(perturb_steps),
        "is_fine": is_fine.tolist(),
        "m_t": m_t,
        "j_t": j_t,
        "perturb_results": [
            {
                "perturb_step": r["perturb_step"],
                "success": r["success"],
                "final_step": r["final_step"],
                "e_t": r["e_t"],
                "s_t": r["s_t"],
                "D_T": r["D_T"],
            }
            for r in sorted(all_perturb_results.values(), key=lambda x: x["perturb_step"])
        ],
        "aggregate": {
            "fine_sr": float(fine_sr),
            "fine_success": fine_success,
            "fine_total": fine_total,
            "coarse_sr": float(coarse_sr),
            "coarse_success": coarse_success,
            "coarse_total": coarse_total,
            "e_mean": float(e_t.mean()),
            "s_mean": float(s_t_array[s_t_array > 0].mean()) if (s_t_array > 0).any() else 0.0,
            "s_max": float(s_t_array.max()),
        },
    }

    with open(task_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"    Fine-grained: {fine_success}/{fine_total} ({fine_sr:.1%}) | "
        f"Coarse: {coarse_success}/{coarse_total} ({coarse_sr:.1%})"
    )

    return summary


# ===========================================================================
# Kinematic analysis helpers
# ===========================================================================

def compute_motion_fineness(actions: np.ndarray) -> np.ndarray:
    """
    Compute motion fineness proxy m_t = 1 - ||pos_t|| / ||pos||_max.
    High m_t (close to 1): fine-grained movement (near origin = grasping/precision)
    Low m_t (close to 0):   coarse movement (large displacement = reaching)
    """
    if len(actions) == 0:
        return np.array([])
    xyz = actions[:, :3]
    trans_mag = np.linalg.norm(xyz, axis=1)
    mu_max = np.percentile(trans_mag, 95) if len(trans_mag) > 0 else 1e-6
    mu_max = max(mu_max, 1e-6)
    return 1.0 - trans_mag / mu_max


def compute_angular_jerk(actions: np.ndarray) -> np.ndarray:
    """Angular jerk proxy (rate of rotation change)."""
    if len(actions) < 2:
        return np.zeros(len(actions)) if len(actions) > 0 else np.array([])
    rot = actions[:, 3:6]
    delta_rot = np.diff(rot, axis=0)
    delta_mag = np.linalg.norm(delta_rot, axis=1)
    nu_max = np.percentile(delta_mag, 95) if len(delta_mag) > 0 else 1e-6
    nu_max = max(nu_max, 1e-6)
    j_t = np.zeros(len(actions))
    j_t[1:] = delta_mag / nu_max
    return j_t


def classify_fine_vs_coarse(actions: np.ndarray) -> np.ndarray:
    """
    Bool array: True = fine-grained (high quantization sensitivity).
    Uses median of motion_fineness as threshold (paper convention).
    """
    if len(actions) == 0:
        return np.array([], dtype=bool)
    m_t = compute_motion_fineness(actions)
    if len(m_t) == 0:
        return np.zeros(len(actions), dtype=bool)
    threshold = float(np.median(m_t))
    return m_t > threshold


# ===========================================================================
# Aggregate results across tasks
# ===========================================================================

def _aggregate(bits_list, all_results, out_dir):
    for bits in bits_list:
        results = [r for r in all_results if r["perturb_bits"] == bits]
        if not results:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  AGGREGATE: {bits}-bit ({len(results)} tasks)")
        logger.info(f"{'='*60}")

        # Overall baseline success rate
        baseline_count = sum(1 for r in results if r.get("aggregate", {}).get("e_mean", 0) >= 0)
        logger.info(f"  Tasks: {len(results)}")

        # Fine vs coarse success rates
        fine_s = sum(r["aggregate"]["fine_success"] for r in results)
        fine_t = sum(r["aggregate"]["fine_total"] for r in results)
        coarse_s = sum(r["aggregate"]["coarse_success"] for r in results)
        coarse_t = sum(r["aggregate"]["coarse_total"] for r in results)
        fine_sr = fine_s / fine_t if fine_t else 0.0
        coarse_sr = coarse_s / coarse_t if coarse_t else 0.0

        # Error statistics
        agg_e_mean = np.mean([r["aggregate"]["e_mean"] for r in results])
        agg_e_max = np.max([r["aggregate"]["e_max"] for r in results])
        agg_s_mean = np.mean([r["aggregate"]["s_mean"] for r in results])
        agg_s_max = np.max([r["aggregate"]["s_max"] for r in results])

        logger.info(f"  Fine-grained: {fine_s}/{fine_t} ({fine_sr:.1%})")
        logger.info(f"  Coarse:       {coarse_s}/{coarse_t} ({coarse_sr:.1%})")
        logger.info(f"  e_t mean: {agg_e_mean:.6f}, max: {agg_e_max:.6f}")
        logger.info(f"  s_t mean: {agg_s_mean:.6f}, max: {agg_s_max:.6f}")

        summary = {
            "bits": bits,
            "num_tasks": len(results),
            "fine": {
                "sr": float(fine_sr),
                "success": fine_s,
                "total": fine_t,
            },
            "coarse": {
                "sr": float(coarse_sr),
                "success": coarse_s,
                "total": coarse_t,
            },
            "e_t": {
                "mean": float(agg_e_mean),
                "max": float(agg_e_max),
            },
            "s_t": {
                "mean": float(agg_s_mean),
                "max": float(agg_s_max),
            },
        }

        with open(out_dir / f"summary_bits{bits}.json", "w") as f:
            json.dump(summary, f, indent=2)

        key_obs = (
            "Fine-grained SR < Coarse SR: confirms DyQ-VLA temporal sensitivity"
            if fine_sr < coarse_sr else
            f"Fine={fine_sr:.1%}, Coarse={coarse_sr:.1%}"
        )
        logger.info(f"  Key: {key_obs}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Section III-A: Step-wise Quantization Sensitivity Analysis"
    )
    parser.add_argument(
        "--trajectory_dir", type=str, required=True,
        help="Directory with recorded BF16 trajectories"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to PyTorch safetensors checkpoint"
    )
    parser.add_argument("--config", type=str, default="pi05_libero")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--perturb_bits", type=int, nargs="+", default=[4],
        help="Bit widths for W4A4 quantization (default: 4)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=20,
        help="Perturbation steps to sample per episode (default: 20)"
    )
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--task_ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted run: skip tasks/perturb_steps with existing output files"
    )
    args = parser.parse_args()

    setup_env()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = pathlib.Path(args.trajectory_dir)

    # Infer task IDs from trajectory directory
    if args.task_ids is not None:
        task_ids = args.task_ids
    else:
        task_ids = sorted([
            int(d.name.split("_")[1])
            for d in traj_dir.iterdir()
            if d.is_dir() and d.name.startswith("task_")
        ])
    logger.info(f"Auto-detected {len(task_ids)} tasks: {task_ids}")

    # Load model
    logger.info(f"Loading model config: {args.config}")
    train_config = _config.get_config(args.config)
    model_config = train_config.model

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = pi0_pytorch.PI0Pytorch(config=model_config)
    import safetensors.torch
    safetensors.torch.load_model(model, args.checkpoint)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    logger.info(f"Model on {device}")

    # Load tokenizer
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=model_config.max_token_len)

    # Task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    all_results = []

    for bits in args.perturb_bits:
        logger.info(f"\n{'='*60}")
        logger.info(f"  BIT-WIDTH: {bits}-bit W4A4")
        logger.info(f"{'='*60}")

        for task_id in tqdm.tqdm(task_ids, desc=f"[{bits}-bit] Tasks"):
            ep_out = out_dir / f"task_{task_id:02d}"
            ep_out.mkdir(parents=True, exist_ok=True)

            if args.resume and (ep_out / "summary.json").exists():
                logger.info(f"  [task_{task_id:02d}] summary.json exists, skipping (use --resume to auto-skip)")
                # Load existing result so aggregate still sees it
                with open(ep_out / "summary.json") as f:
                    result = json.load(f)
                all_results.append(result)
                continue

            result = run_task_experiment(
                task_id=task_id,
                task_suite=task_suite,
                traj_dir=traj_dir,
                out_dir=out_dir,
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                device=device,
                perturb_bits=bits,
                num_samples=args.num_samples,
                max_steps=args.max_steps,
                seed=args.seed,
                resume=args.resume,
            )

            if result is not None:
                all_results.append(result)

        _aggregate(args.perturb_bits, all_results, out_dir)

    logger.info(f"\nDone. Results: {out_dir}")


if __name__ == "__main__":
    main()
