import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig, #从前面的config.py中拿到的训练配置
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    # --- W4A4 Quantization args ---
    quantize: bool = False,
    quantize_bits: int = 4,
    quantize_group_size: int = 256,
    # --- Activation calibration (only used when quantize=True) ---
    calibration_steps: int = 0,
    calibration_data_dir: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.

    Quantization:
        When quantize=True, applies DyQVLA-style W4A4 fake quantization to all nn.Linear
        layers in the PyTorch model. Only supported for PyTorch models (is_pytorch=True).
        - quantize_bits: bit-width (default 4, also supports 2/8)
        - quantize_group_size: weight group size (default 256, tested optimal from benchmarks)
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    print(f"Loading checkpoint from {checkpoint_dir}...")
    #这里的 checkpoint_dir 是前面配置好的路径 "gs://openpi-assets/checkpoints/pi05_droid"
    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)
    print(f"Is_pytoch", is_pytorch)
    #OpenPI 的权重命名约定
    #PyTorch：使用 model.safetensors
    #Flax/JAX：使用目录 params/ 保存分片权重
    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
        #用 Flax 权重文件反序列化并构建模型,用读出来的参数，构建出 JAX 的模型结构
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    #：创建数据处理的配置对象,它决定了图片要怎么裁剪、分辨率要是多少，是根据模型的需求动态生成的。
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # ============================================================
    # DyQVLA-style W4A4 FakeQuant (only for PyTorch models)
    # ============================================================
    if is_pytorch and quantize:
        import sys as _sys
        _fakequant_dir = pathlib.Path(__file__).parent.parent.parent / "examples" / "libero"
        if _fakequant_dir.exists():
            _sys.path.insert(0, str(_fakequant_dir))
            from fakequant import CalibrationRunner, apply_fake_quant, set_global_bits

            # Layers that operate on the 7-dim action space stay FP16
            _exclude = [
                "action_in_proj",
                "action_out_proj",
                "time_mlp_in",
                "time_mlp_out",
                "state_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
            ]

            n_layers = apply_fake_quant(
                model,
                bits=quantize_bits,
                group_size=quantize_group_size,
                percentile=99.9,
                quantize_activations=True,
                exclude_names=_exclude,
                exclude_bits=[2],  # don't pre-compute W2 scales
            )
            set_global_bits(quantize_bits)
            print(f"✅ W{quantize_bits}A{quantize_bits} FakeQuant applied to {n_layers} Linear layers "
                  f"(group_size={quantize_group_size})")

            # ---- Activation calibration ----
            if calibration_steps > 0:
                import json as _json
                import numpy as _np
                import torch as _torch
                import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

                calib_dir = calibration_data_dir or str(pathlib.Path(__file__).parent.parent.parent / "data" / "libero" / "videos" / "quant")
                calib_path = pathlib.Path(calib_dir)
                if not calib_path.exists():
                    print(f"⚠️  Calibration dir not found: {calib_dir}, skipping calibration")
                else:
                    calib_files = sorted(calib_path.glob("*.json"))
                    if not calib_files:
                        print(f"⚠️  No JSON files in calibration dir: {calib_dir}, skipping calibration")
                    else:
                        print(f"🔧 Running activation calibration: up to {calibration_steps} steps from {len(calib_files)} trajectories...")

                        # Register all FakeQuantLinear layers for calibration
                        from fakequant import get_all_fakequant_layers
                        calib_layers = get_all_fakequant_layers(model)
                        for layer in calib_layers:
                            layer.start_calibration()
                        print(f"  Registered {len(calib_layers)} FakeQuantLinear layers for calibration")

                        # Build a helper to create Observation objects from raw dicts
                        def _make_obs(state_np, images_dict, lang_prompt, device):
                            """Build a model.Observation from raw numpy dicts."""
                            obs_raw = {
                                "state": _np.asarray(state_np, dtype=_np.float32),
                                "image": {k: _np.asarray(v, dtype=_np.uint8) for k, v in images_dict.items()},
                                "image_mask": {
                                    "base_0_rgb": True,
                                    "left_wrist_0_rgb": True,
                                    "right_wrist_0_rgb": False,
                                },
                                "prompt": lang_prompt,
                            }

                            class _DictObs:
                                def __init__(self, d):
                                    for k, v in d.items():
                                        setattr(self, k, v)

                            obs_raw = _DictObs(obs_raw)
                            obs = _preprocessing.preprocess_observation_pytorch(obs_raw, train=False)
                            return _model.Observation(
                                state=obs.state.to(device),
                                images={k: v.to(device) for k, v in obs.images.items()},
                                image_masks={k: v.to(device) for k, v in obs.image_masks.items()},
                                tokenized_prompt=obs.tokenized_prompt.to(device),
                                tokenized_prompt_mask=obs.tokenized_prompt_mask.to(device),
                            )

                        steps_done = 0
                        with _torch.no_grad():
                            for calib_file in calib_files:
                                if steps_done >= calibration_steps:
                                    break
                                try:
                                    with open(calib_file) as f:
                                        traj = _json.load(f)
                                    steps = traj.get("steps", [])
                                    if not steps:
                                        continue

                                    # Sample steps uniformly across the trajectory
                                    num_trajectories = min(calibration_steps, len(steps))
                                    indices = _np.linspace(0, len(steps) - 1, num_trajectories, dtype=int)

                                    for idx in indices:
                                        if steps_done >= calibration_steps:
                                            break
                                        step_record = steps[idx]

                                        # Build observation: use zero images (vision encoder path won't exercise activation quant)
                                        # but real state from trajectory
                                        eef_pos = step_record.get("eef_pos", [0.0] * 3)
                                        state_np = _np.array(eef_pos + [0.0] * 5, dtype=_np.float32)
                                        zero_img = _np.zeros((224, 224, 3), dtype=_np.uint8)
                                        images_dict = {
                                            "base_0_rgb": zero_img,
                                            "left_wrist_0_rgb": zero_img,
                                            "right_wrist_0_rgb": zero_img,
                                        }

                                        try:
                                            obs = _make_obs(
                                                state_np,
                                                images_dict,
                                                traj.get("task_description", "do something"),
                                                pytorch_device,
                                            )
                                            action_t = _torch.randn(
                                                1, train_config.model.action_horizon, train_config.model.action_dim,
                                                device=pytorch_device, dtype=_torch.float32,
                                            )
                                            timestep_t = _torch.rand(1, device=pytorch_device, dtype=_torch.float32)

                                            # Full forward pass through both prefix (vision+lang) and suffix (action expert)
                                            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                                                obs.images, obs.image_masks, obs.tokenized_prompt, obs.tokenized_prompt_mask
                                            )
                                            # Build dummy past_kv from prefix
                                            att_2d_masks = model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                                            att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)
                                            position_ids = _torch.cumsum(prefix_pad_masks, dim=1) - 1
                                            model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
                                            _, past_kv = model.paligemma_with_expert.forward(
                                                attention_mask=att_2d_masks_4d,
                                                position_ids=position_ids,
                                                past_key_values=None,
                                                inputs_embeds=[prefix_embs, None],
                                                use_cache=True,
                                            )
                                            _ = model.denoise_step(
                                                obs.state.unsqueeze(0),
                                                prefix_pad_masks,
                                                past_kv,
                                                action_t,
                                                timestep_t,
                                            )
                                            steps_done += 1
                                        except Exception as e:
                                            logging.warning(f"Calibration step failed: {e}")
                                            continue

                                except Exception as e:
                                    logging.warning(f"Failed to load calibration file {calib_file}: {e}")
                                    continue

                        # Finalize calibration
                        for layer in calib_layers:
                            layer.finish_calibration(percentile=99.9)
                        print(f"✅ Activation calibration complete: {steps_done} steps processed across {len(calib_layers)} layers")
        else:
            print(f"⚠️  fakequant.py not found at {_fakequant_dir}, skipping quantization")

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
