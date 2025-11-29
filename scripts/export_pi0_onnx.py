#!/usr/bin/env python3
"""
Load a JAX model and print all parameter keys, with optional conversion to PyTorch.
"""

from __future__ import annotations

import argparse
import os
import math # 确保 math 被导入
from types import SimpleNamespace
from typing import Literal # 确保 Literal 被导入

import torch
from safetensors.torch import load_model

from openpi.models_pytorch import pi0_pytorch
from openpi.training import config as train_config


class Pi0OnnxWrapper(torch.nn.Module):
    """Wrap PI0 PyTorch model with tensor-only inputs for ONNX export."""

    def __init__(self, base_model: pi0_pytorch.PI0Pytorch, num_steps: int):
        super().__init__()
        self.model = base_model
        self.num_steps = num_steps

    def forward(  # noqa: PLR0913
        self,
        base_img: torch.Tensor,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        base_mask: torch.Tensor,
        left_mask: torch.Tensor,
        right_mask: torch.Tensor,
        tokenized_prompt: torch.Tensor,
        tokenized_prompt_mask: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        
        # --- 修正 1: 确保所有浮点数输入张量使用与模型相同的 DType ---
        current_dtype = self.model.time_mlp_in.weight.dtype # 获取模型权重当前的 DType
        
        # 1. 重构 Observation (与原脚本相同)
        observation = SimpleNamespace(
            images={
                "base_0_rgb": base_img,
                "left_wrist_0_rgb": left_img,
                "right_wrist_0_rgb": right_img,
            },
            image_masks={
                "base_0_rgb": base_mask.bool(),
                "left_wrist_0_rgb": left_mask.bool(),
                "right_wrist_0_rgb": right_mask.bool(),
            },
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=None,
            token_loss_mask=None,
        )

        # 2. Preprocess inputs (与原脚本相同)
        images, img_masks, lang_tokens, lang_masks, state_tensor = self.model._preprocess_observation(
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        # 禁用 eager (这一行是兼容性修正，保留)
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager" 

        # 3. KV Cache Pre-fill (与原脚本相同)
        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        
        # 4. Diffusion Loop Setup
        bsize = state_tensor.shape[0]
        dt_value = -1.0 / float(self.num_steps)
        
        # 修正 2: 确保 dt 和 time 标量张量使用正确的 DType
        dt = torch.tensor(dt_value, dtype=current_dtype, device=state_tensor.device) # <--- 修正 DType
        x_t = noise # 噪声 noise 已经具有正确的 DType
        time = torch.tensor(1.0, dtype=current_dtype, device=state_tensor.device) # <--- 修正 DType

        # 5. Diffusion Loop (此部分不能被 ONNX 导出，但 torch.onnx.export 忽略它)
        # 注意：导出的 ONNX 模型将是一个巨大的 Graph，它包含了 num_steps 次 unrolled 的计算。
        for _ in range(self.num_steps):
            expanded_time = time.expand(bsize)
            v_t = self.model.denoise_step(
                state_tensor,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t


def build_dummy_inputs(config, batch_size: int, device: torch.device, dtype: torch.dtype):
    height = width = 224
    action_dim = config.model.action_dim
    action_horizon = config.model.action_horizon
    seq_len = config.model.max_token_len

    def image_tensor():
        return torch.zeros((batch_size, 3, height, width), dtype=dtype, device=device)

    base_img = image_tensor()
    left_img = image_tensor()
    right_img = image_tensor()

    mask = torch.ones((batch_size,), dtype=torch.bool, device=device)
    
    # 修正 3: tokenized_prompt 必须是整数类型 (int32/int64)，不能是 dtype (可能是 bfloat16)
    tokenized_prompt = torch.zeros((batch_size, seq_len), dtype=torch.int32, device=device) 
    tokenized_prompt_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    
    state = torch.zeros((batch_size, action_dim), dtype=dtype, device=device)
    noise = torch.zeros((batch_size, action_horizon, action_dim), dtype=dtype, device=device)

    return (
        base_img,
        left_img,
        right_img,
        mask,
        mask,
        mask,
        tokenized_prompt,
        tokenized_prompt_mask,
        state,
        noise,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export PI0/PI0.5 PyTorch checkpoint to ONNX.")
    parser.add_argument("--config_name", required=True, help="Training config name, e.g. pi05_droid")
    parser.add_argument("--weights_dir", required=True, help="Directory containing model.safetensors")
    parser.add_argument("--output", required=True, help="Path to write the ONNX file")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of denoising steps for sampling")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch_size", type=int, default=1, help="Dummy batch size for tracing")
    # 新增参数，方便在导出时降级到 float32
    parser.add_argument("--export_dtype", type=str, default="auto", choices=["auto", "float32"], help="Force export dtype")
    return parser.parse_args()


def main():
    args = parse_args()

    config = train_config.get_config(args.config_name)
    model_cfg = config.model

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    base_model = pi0_pytorch.PI0Pytorch(model_cfg)
    weight_path = os.path.join(args.weights_dir, "model.safetensors")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Expected weights at {weight_path}")
    load_model(base_model, weight_path)

    # Disable torch.compile wrapper for ONNX tracing.
    base_model.sample_actions = pi0_pytorch.PI0Pytorch.sample_actions.__get__(
        base_model, pi0_pytorch.PI0Pytorch
    )
    
    # 决定模型权重和浮点输入张量的 DType
    model_dtype = torch.bfloat16
    if device.type == "cpu" or args.export_dtype == "float32":
         model_dtype = torch.float32

    # 将模型权重转移到正确的设备和 DType
    base_model = base_model.to(dtype=model_dtype, device=device)
    base_model.eval()

    # 构造 ONNX Wrapper 和输入
    wrapper = Pi0OnnxWrapper(base_model, num_steps=args.num_steps)
    dummy_inputs = build_dummy_inputs(config, args.batch_size, device, model_dtype) # 使用修正后的 model_dtype

    input_names = [
        "base_img",
        "left_img",
        "right_img",
        "base_mask",
        "left_mask",
        "right_mask",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "state",
        "noise",
    ]
    output_names = ["actions"]
    # 保持 dynamic_axes (动态形状)
    dynamic_axes = {
        "base_img": {0: "batch"},
        "left_img": {0: "batch"},
        "right_img": {0: "batch"},
        "base_mask": {0: "batch"},
        "left_mask": {0: "batch"},
        "right_mask": {0: "batch"},
        "tokenized_prompt": {0: "batch", 1: "seq_len"},
        "tokenized_prompt_mask": {0: "batch", 1: "seq_len"},
        "state": {0: "batch"},
        "noise": {0: "batch"},
        "actions": {0: "batch"},
    }

    print(f"\n--- Starting ONNX Export (Target DType: {model_dtype}) ---")
    
    # 导出核心代码
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        args.output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"\nExported ONNX model successfully to {args.output}")
    print("WARNING: This ONNX model contains the full diffusion loop unrolled num_steps times.")
    print("If you encounter size or performance issues, you must manually export the denoise_step function.")


if __name__ == "__main__":
    main()