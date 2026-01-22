"""Micro-benchmark for Pi0/PaliGemma + KV cache on DROID.

This script loads the pi05_droid model directly and runs a few
`sample_actions` calls on synthetic observations generated on the GPU.

Usage (plain run,看数值):
    uv run kvcache_bench.py

Usage (配合 nsys 做 GPU profiling):
    cd openpi
    nsys profile \
      --trace=cuda \
      --output=profiles/kvcache_macro.nsys-rep \
      --force-overwrite=true \
      uv run kvcache_bench.py

这样得到的 nsys trace 里只有:
  - checkpoint 加载
  - pi05_droid 模型本身 (PaliGemma + KV cache)
没有 websocket/simple_client 这些系统噪音，更适合看模型内部的 kernel 分布。
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.training import config as _config


logger = logging.getLogger(__name__)


def load_pi05_droid_model() -> _model.BaseModel:
    """创建一个随机初始化的 pi05_droid 模型，用于 profiling（不加载 50G 权重）."""
    # 1. 和 serve_policy 一样拿到训练配置
    train_cfg = _config.get_config("pi05_droid")
    model_cfg = train_cfg.model  # 其实就是 Pi0Config

    logger.info("Creating random-initialized pi05_droid model ...")
    logger.info(
        "  action_dim=%s, action_horizon=%s, max_token_len=%s",
        model_cfg.action_dim, model_cfg.action_horizon, model_cfg.max_token_len,
    )

    # 2. 直接用 config.create 随机初始化一个 Pi0 模型，不 restore checkpoint
    rng = jax.random.key(0)
    model = model_cfg.create(rng)
    logger.info("Random-initialized model created.")
    return model

def make_fake_observation(model_cfg: _model.BaseModelConfig, batch_size: int = 1) -> _model.Observation:
    """Create a synthetic Observation on the device using the model config."""
    # `fake_obs` returns a tree of jax.Array filled with ones, shapes/dtypes
    # exactly match what the model expects. 这样可以避免走 simple_client /
    # websocket 等路径，只测模型本身。
    obs = model_cfg.fake_obs(batch_size=batch_size)
    return obs


def main() -> None:
    print(">>> kvcache_bench start <<<")
    logging.basicConfig(level=logging.INFO)

    # Make sure we use GPU if available
    logger.info("JAX devices: %s", jax.devices())

    # Load model & create fake obs
    train_cfg = _config.get_config("pi05_droid")
    model_cfg = train_cfg.model
    model = load_pi05_droid_model()

    batch_size = 1
    obs = make_fake_observation(model_cfg, batch_size=batch_size)

    rng = jax.random.key(0)

    # Warmup: JIT compile once
    logger.info("Warmup run (JIT compilation)...")
    rng, subkey = jax.random.split(rng)
    actions = model.sample_actions(subkey, obs)  # use default num_steps
    jax.block_until_ready(actions)
    logger.info("Warmup done.")

    # Benchmark multiple runs to get a stable number
    num_runs = 10
    logger.info("Benchmarking %d runs of model.sample_actions (batch_size=%d)...", num_runs, batch_size)

    times_ms: list[float] = []
    for i in range(num_runs):
        rng, subkey = jax.random.split(rng)
        t0 = time.perf_counter()
        actions = model.sample_actions(subkey, obs)
        jax.block_until_ready(actions)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        times_ms.append(elapsed_ms)
        logger.info("Run %d/%d: %.2f ms", i + 1, num_runs, elapsed_ms)

    # Simple statistics
    import numpy as np

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    logger.info("Summary over %d runs: mean=%.2f ms, std=%.2f ms", num_runs, mean_ms, std_ms)


if __name__ == "__main__":
    print(">>> kvcache_bench start <<<")
    main()


