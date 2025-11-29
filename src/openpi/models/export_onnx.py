import logging
import os
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import einops
import tensorflow as tf
from jax.experimental import jax2tf
import tf2onnx

# --- 1. 导入 openpi 模块 ---
# 确保你的 PYTHONPATH 包含了 openpi/src
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.models import model as _model
from openpi.shared import array_typing as at

logger = logging.getLogger("onnx_export")

# --- 2. 辅助函数 (从源码复制，确保独立性) ---
def posemb_sincos(pos, embedding_dim, min_period, max_period):
    """Computes sine-cosine positional embedding vectors."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij", pos, 1.0 / period * 2 * jnp.pi, precision=jax.lax.Precision.HIGHEST
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)

def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)

# --- 3. 核心 Wrapper 类：将 Step 逻辑封装为纯函数 ---
class ActionDenoiseStep(nnx.Module):
    def __init__(self, model: Pi0):
        self.model = model

    def __call__(self, 
                 kv_cache,           # [B, S_prefix, Layers, 2, Heads, Dim] 或 简化版
                 prefix_mask,        # [B, S_prefix]
                 x_t,                # [B, Horizon, ActionDim]
                 time):              # [B]
        
        # -------------------------------------------------------
        # Part A: 复刻 embed_suffix 的逻辑 (转为纯 Tensor 操作)
        # -------------------------------------------------------
        batch_size = x_t.shape[0]
        
        # 1. Action Embedding
        action_tokens = self.model.action_in_proj(x_t)
        
        # 2. Time Embedding (Sine-Cosine)
        # 注意：action_in_proj.out_features 是内部属性，这里直接获取
        embed_dim = action_tokens.shape[-1] 
        time_emb = posemb_sincos(time, embed_dim, min_period=4e-3, max_period=4.0)
        
        # 3. Time MLP & Combination (针对 pi0.5 逻辑)
        if self.model.pi05:
            # Pi0.5 使用 AdaRMS，Time Embedding 经过 MLP 后作为条件
            time_emb = self.model.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.model.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            
            suffix_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # 非 Pi0.5 (Pi0 original)，Time 与 Action 拼接
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.model.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.model.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.model.action_time_mlp_out(action_time_tokens)
            
            suffix_tokens = action_time_tokens
            adarms_cond = None

        # 构造 Suffix Mask (Action 自回归掩码)
        # action tokens attend to each other (Full Causal or Block Causal)
        # 这里简化为 Full Attention 或者是源码中的 Causal
        # 源码中 ar_mask += [True] + ([False] * (horizon - 1))，这意味着第一个token看所有，后面看前面？
        # 为了简化导出，这里假设 Suffix 内部是全可见的 (根据 Pi0 代码推断通常是 Causal)
        # 我们构建一个标准的 Causal Mask
        suffix_len = suffix_tokens.shape[1]
        suffix_mask = jnp.ones((batch_size, suffix_len), dtype=jnp.bool_)
        
        # -------------------------------------------------------
        # Part B: 构建 Attention Mask 和 Position
        # -------------------------------------------------------
        # Prefix Mask 扩展: [B, S_prefix] -> [B, S_suffix, S_prefix]
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
        
        # Suffix Mask: [B, S_suffix, S_suffix] (Causal)
        suffix_attn_mask = jnp.tril(jnp.ones((batch_size, suffix_len, suffix_len), dtype=jnp.bool_))
        
        # Combined Mask: [B, S_suffix, S_total]
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        
        # Positions: Suffix tokens 的位置索引
        # prefix_mask 求和得到 prefix 长度，加上 suffix 的累积索引
        prefix_len = jnp.sum(prefix_mask.astype(jnp.int32), axis=-1)[:, None]
        suffix_pos = jnp.arange(suffix_len)[None, :] # [1, S_suffix]
        positions = prefix_len + suffix_pos # [B, S_suffix]

        # -------------------------------------------------------
        # Part C: 核心 Transformer 推理
        # -------------------------------------------------------
        # 调用 PaliGemma LLM
        # 注意：这里只传入 suffix_tokens，让它利用 kv_cache
        (prefix_out, suffix_out), _ = self.model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,               # <--- 关键：传入外部的大 Cache
            adarms_cond=[None, adarms_cond], # Pi0.5 需要这个
        )
        
        # -------------------------------------------------------
        # Part D: 输出投影
        # -------------------------------------------------------
        v_t = self.model.action_out_proj(suffix_out[:, -self.model.action_horizon :])
        return v_t

# --- 4. 准备工作：Monkeypatch 和 初始化 ---

# 4.1 定义具体的 Observation 生成函数 (修复 TypeError)
def concrete_fake_obs(batch_size=1):
    """生成具体形状的假数据，用于模型初始化"""
    # 根据你的实际配置修改 H, W
    H, W = 224, 224 
    return _model.Observation(
        images={
            "cam_high": jnp.zeros((batch_size, H, W, 3), dtype=jnp.float32)
        },
        image_masks={
            # FIX: 形状从 (B, H, W) 改为 (B,)
            "cam_high": jnp.ones((batch_size,), dtype=jnp.bool_)
        },
        state=jnp.zeros((batch_size, 14), dtype=jnp.float32), # 假设动作维度14
        tokenized_prompt=jnp.zeros((batch_size, 32), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, 32), dtype=jnp.bool_)
    )

# 4.2 加载配置
# 注意：根据你的实际模型 variant 修改 (gemma_2b, etc.)
print("Loading Config...")
config = Pi0Config(
    action_dim=14,           # 修改：匹配你的机器人
    action_horizon=10,       # 修改：匹配你的 Horizon
    max_token_len=1024,
    pi05=True,               # 使用 Pi0.5
    paligemma_variant="gemma_2b", # 示例
    action_expert_variant="gemma_300m" # 示例
)

# 4.3 应用 Monkeypatch
object.__setattr__(config, 'fake_obs', lambda: concrete_fake_obs(batch_size=1))

# 4.4 初始化模型
print("Initializing Pi0 Model...")
model = Pi0(config, rngs=nnx.Rngs(0))
print("Model Initialized.")

# --- 5. 导出流程 ---

# 5.1 实例化 Wrapper
step_model = ActionDenoiseStep(model)

# 5.2 定义 Dummy Inputs (用于 Trace)
B = 1
H_action = config.action_horizon
D_action = config.action_dim

# 获取实际的 Embedding Dimension
embed_dim = config.action_expert_variant.width 

# [关键] KV Cache 形状
# PaliGemma 的 KV Cache 通常是 PyTree。
# 为了导出方便，我们这里需要知道它是怎么存的。
# 如果是标准 Flax Gemma，它可能是 tuple of layers。
# **为了让脚本能跑通，我们需要先运行一次 forward pass 拿到真实的 kv_cache 结构**
print("Tracing specific shapes...")
obs = concrete_fake_obs(B)
# 运行一次 prefix embed 来获取真实的 kv_cache 结构
prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
pos = jnp.cumsum(prefix_mask, axis=1) - 1
_, real_kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=pos)

# 使用真实的 kv_cache 结构作为 dummy
dummy_kv_cache = real_kv_cache
dummy_prefix_mask = prefix_mask
dummy_x_t = jnp.zeros((B, H_action, D_action), dtype=jnp.float32)
dummy_time = jnp.array([1.0], dtype=jnp.float32) # Batch size 1

# 5.3 JAX to TF 转换
print("Converting JAX to TensorFlow SavedModel...")

# 分离 Graph 和 Weights
graphdef, params = nnx.split(step_model)

def forward_fn(params, kv, p_mask, x, t):
    model_fn = nnx.merge(graphdef, params)
    return model_fn(kv, p_mask, x, t)

# 使用 Polymorphic shapes 或者 具体 shapes
# 既然是导出给 NPU，建议使用具体 shapes 固定下来
tf_func = jax2tf.convert(
    lambda kv, pm, x, t: forward_fn(params, kv, pm, x, t),
    with_gradient=False,
    polymorphic_shapes=None # 使用具体形状
)

# 5.4 定义 Input Signatures
# 注意：TF 需要知道 input 的 spec
# real_kv_cache 是一个复杂的 PyTree，我们需要用 jax2tf.input_signature 来自动生成
# 或者手动展平。tf2onnx 对嵌套输入的处理可能比较麻烦。
# **最稳妥的方法**：利用 tf.function 的 input_signature 自动推导
input_signature = [
    tf.TensorSpec.from_tensor(tf.constant(jax.tree_util.tree_map(lambda x: np.array(x), x)))
    for x in [dummy_kv_cache, dummy_prefix_mask, dummy_x_t, dummy_time]
]

# 5.5 TF to ONNX
print("Converting TensorFlow to ONNX...")
import tf2onnx

# 由于 kv_cache 是 PyTree，我们需要把它展平成列表传给 tf2onnx
# 但 jax2tf 生成的函数期望的是 PyTree 结构。
# 这里的 trick 是：让 tf2onnx 处理 tf_func，tf_func 内部处理结构。
# 但是 tf2onnx.convert.from_function 要求 input_signature 是 flat list of TensorSpecs.

# --- 解决方案：再包一层，把所有输入展平 ---
flat_args, structure = jax.tree_util.tree_flatten((dummy_kv_cache, dummy_prefix_mask, dummy_x_t, dummy_time))

def flat_forward_fn(*flat_args_in):
    # 重组结构
    args_in = jax.tree_util.tree_unflatten(structure, flat_args_in)
    kv, pm, x, t = args_in
    return forward_fn(params, kv, pm, x, t)

tf_flat_func = jax2tf.convert(flat_forward_fn, with_gradient=False)

# 生成扁平化的签名
flat_input_signature = [
    tf.TensorSpec(shape=leaf.shape, dtype=tf.float32 if leaf.dtype==jnp.float32 else tf.bool if leaf.dtype==jnp.bool_ else tf.float32, name=f"input_{i}")
    for i, leaf in enumerate(flat_args)
]

model_proto, _ = tf2onnx.convert.from_function(
    tf_flat_func,
    input_signature=flat_input_signature,
    output_path="pi0_action_step.onnx",
    opset=17,
    large_model=True # 防止 protobuf 溢出（如果模型很大）
)

print("Done! Saved to pi0_action_step.onnx")
print("INFO: The input of this ONNX model is flattened.")
print("When running on Ascend, you need to flatten your kv_cache PyTree into a list of tensors.")