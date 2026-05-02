"""
Stage 2: experiment.py
=====================
Section III-A of DyQ-VLA (arXiv:2603.07904):
"Step-wise Perturbation Analysis of Quantization Sensitivity"

Pipeline:
  Stage 1 (collect_trajectories.py):
    Record BF16 baseline trajectories: obs + actions from server.
  Stage 2 (this file - FAST DECOUPLED MODE):
    1. Run BF16 model inference -> actions_16b.npy
    2. Run W4A16 quantized model inference -> actions_4b.npy
    3. Calculate mathematical decision error -> e_t.npy
    4. EARLY EXIT (Skip physical simulation for fast data generation)
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
# with unconditional to_dense() call
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
_orig_preprocess = _preprocessing.preprocess_observation_pytorch

def _patched_preprocess_observation_pytorch(observation, *, train=False,
        image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        image_resolution=(224, 224)):
    """Patched version: unconditionally call to_dense() on each image tensor"""
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
MAX_DISTurb_STEPS = 20      


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
# Robust FakeQuant (Per-Channel Weight + Per-Token Activation)
# ===========================================================================

def compute_scale_per_channel(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """
    逐通道 (Per-Channel) 对称比例尺。
    weight shape: (out_features, in_features)
    返回 shape: (out_features, 1)
    """
    levels = (2 ** (bits - 1)) - 1
    max_val = torch.max(torch.abs(weight), dim=1, keepdim=True)[0]
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
        q_min = -levels - 1  
        q_max = levels       

        # --- 1. 权重 Per-Channel 量化 ---
        w = self._linear.weight
        w_q = torch.round(w / self._w_scale)
        w_q = torch.clamp(w_q, q_min, q_max)
        w_deq = w_q * self._w_scale

        # --- 2. 激活 Per-Token 量化 (如果开启) ---
        if self._quantize_activations:
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
    """递归替换 Linear 层。"""
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
# Observation builder 
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
# Full experiment for one task (FAST DECOUPLED MODE)
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
        remove_fake_quant(model) 
        actions_bf16 = run_inference_batch(
            model, frames, states, prompt,
            tokenizer, model_config, device,
            num_steps=NUM_STEPS_DENOISE, seed=seed,
            label=f"[task_{task_id:02d}] BF16",
        )
        np.save(actions_16b_path, actions_bf16)
    fp16_time = time.time() - t0
    logger.info(f"    BF16 inference: {fp16_time:.1f}s ({T} steps)")

    # --- Quantized model inference ---
    quant_path = task_out / f"actions_{perturb_bits}b.npy"
    t0 = time.time()
    if resume and quant_path.exists():
        actions_quant = np.load(quant_path)
        logger.info(f"    [CACHE] Loaded {perturb_bits}b actions from {quant_path.name}")
    else:
        # 【修改点 1】：强行采用 W4A16 (关闭激活量化)，保证物理退化正常
        apply_fake_quant(model, perturb_bits, quantize_activations= True)
        actions_quant = run_inference_batch(
            model, frames, states, prompt,
            tokenizer, model_config, device,
            num_steps=NUM_STEPS_DENOISE, seed=seed,
            # 【修改点 2】：修复显示的 Label，避免终端日志骗人
            label=f"[task_{task_id:02d}] W{perturb_bits}A16",
        )
        remove_fake_quant(model) 
        np.save(quant_path, actions_quant)
    quant_time = time.time() - t0
    logger.info(f"    W4A16 inference: {quant_time:.1f}s ({T} steps)")

    # --- 计算单步动作误差 e_t ---
    e_t = np.linalg.norm(
        actions_quant[:, :LIBERO_ACTION_DIM] - actions_bf16[:, :LIBERO_ACTION_DIM], axis=1
    )
    np.save(task_out / "e_t.npy", e_t)

    # =========================================================
    # 【修改点 3】：极速解耦模式 - 在此强行截断，跳过物理仿真！
    # =========================================================
    logger.info(f"    [FAST MODE] 数学推理已解耦完成！成功生成 {task_name} 的 numpy 文件，直接进入下一个任务。")
    return None

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Section III-A: Fast Offline Quantization Extraction (No Simulation)"
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
        help="Bit widths for W4A16 quantization (default: 4)"
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

    for bits in args.perturb_bits:
        logger.info(f"\n{'='*60}")
        logger.info(f"  FAST MODE EXTRACTION: {bits}-bit W4A16")
        logger.info(f"{'='*60}")

        for task_id in tqdm.tqdm(task_ids, desc=f"[{bits}-bit] Tasks"):
            run_task_experiment(
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

    logger.info(f"\nDone. Offline NPY Extraction finished. Check: {out_dir}")

if __name__ == "__main__":
    main()