import dataclasses
import enum
import logging
import os
import socket
import torch

import tyro
from openpi.models.gemma import set_attention_log_file
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
import jax.profiler

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # --- DyQVLA-style WxAy Quantization (PyTorch models only) ---
    # If True, apply fake quantization to all nn.Linear layers.
    quantize: bool = False
    # Bit-width for weight quantization. Options: 4, 8, 16 (16 = full precision bypass).
    quantize_bits_w: int = 4
    # Bit-width for activation quantization. Options: 4, 8, 16 (16 = full precision bypass).
    quantize_bits_a: int = 4
    # Weight group size for per-group quantization. 256 is recommended (tested optimal).
    quantize_group_size: int = 256
    # Number of calibration steps (batches) to run before inference.
    # Each step uses a real observation from the calibration data directory.
    # 0 = no calibration (online scale estimation). Recommended: 32-128.
    calibration_steps: int = 0

    # Random seed for reproducible inference.
    # - JAX models: used to seed jax.random.key() internally via policy.reset_rng()
    # - PyTorch models: sets torch.manual_seed() after policy creation
    # All servers and clients using the same seed will produce identical results.
    seed: int = 42


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        print(f"【Checkpoint Dir】: {checkpoint.dir}")  # <--- 这里就是你要确认的路径！
        print(f"【checkpoint.config】: {checkpoint.config}")
        print("="*52 + "\n")
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")
#【Checkpoint Dir】: gs://openpi-assets/checkpoints/pi05_droid
#【checkpoint.config】: pi05_droid

def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            config_data = _config.get_config(args.policy.config)
            weight_path = os.path.join(args.policy.dir, "model.safetensors")
            is_pytorch = os.path.exists(weight_path)
            
            print("\n" + "="*52 + " DEBUG INFO " + "="*52)
            print(f"【Config Name】: {args.policy.config}")
            print(f"【Checkpoint Dir】: {args.policy.dir}")
            print(f"【Is PyTorch Model】: {is_pytorch}  {'(model.safetensors found)' if is_pytorch else '(no model.safetensors)'}")
            print(f"【Quantize】: {args.quantize}")
            print(f"【Quantize Bits W】: {args.quantize_bits_w}")
            print(f"【Quantize Bits A】: {args.quantize_bits_a}")
            print(f"【Quantize Group Size】: {args.quantize_group_size}")
            print(f"【Calibration Steps】: {args.calibration_steps}")
            print(f"【Default Prompt】: {args.default_prompt}")
            print("="*120 + "\n")

            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                quantize=args.quantize,
                quantize_bits_w=args.quantize_bits_w,
                quantize_bits_a=args.quantize_bits_a,
                quantize_group_size=args.quantize_group_size,
                calibration_steps=args.calibration_steps,
            )

        case Default():
            print("="*52 + "\n")
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _seed_pytorch(seed: int) -> None:
    """Seed all PyTorch RNG sources for deterministic inference."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disable benchmark mode for fully deterministic CUDA kernels
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args: Args) -> None:
    os.environ["OPENPI_DATA_HOME"] = "/share/chengyuxuan-local/openpi"
    policy = create_policy(args)#创建对应的policy

    # Seed RNGs for reproducibility
    _seed_pytorch(args.seed)
    policy.reset_rng(args.seed)
    logging.info(f"Set random seed to {args.seed} for both PyTorch and JAX RNGs")

    policy_metadata = policy.metadata
    set_attention_log_file()
    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )#实例化的是WebsocketPolicyServer类 然后下面调用对应的serve_forever方法
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
"""
指定cuda_device CUDA_VISIBLE_DEVICES=0 
# FP16 baseline（和以前一样）
CUDA_VISIBLE_DEVICES=1   python scripts/serve_policy.py     --port 8001  policy:checkpoint     --policy.config pi05_libero     --policy.dir /share/chengyuxuan-local/openpi/openpi-assets/checkpoints/pi05_libero_pytorch    
# W4A4 量化（你要跑的）
CUDA_VISIBLE_DEVICES=0   python scripts/serve_policy.py     --port 8000     --quantize --quantize-bits-w 4 --quantize-bits-a 4 --calibration-steps 64     policy:checkpoint     --policy.config pi05_libero     --policy.dir /share/chengyuxuan-local/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 
 
# W4A8（权重4bit，激活8bit）
python scripts/serve_policy.py \
    --policy.checkpoint.config pi05_libero \
    --policy.checkpoint.dir gs://path/to/checkpoint \
    --quantize \
    --quantize-bits-w 4 \
    --quantize-bits-a 8 \
    --quantize-group-size 256

# W8A8（备选，更高精度）
python scripts/serve_policy.py \
    --policy.checkpoint.config pi05_libero \
    --policy.checkpoint.dir gs://path/to/checkpoint \
    --quantize \
    --quantize-bits-w 8 \
    --quantize-bits-a 8 \
    --quantize-group-size 256

# W2A2（最低精度，最大压缩）
python scripts/serve_policy.py \
    --policy.checkpoint.config pi05_libero \
    --policy.checkpoint.dir gs://path/to/checkpoint \
    --quantize \
    --quantize-bits-w 2 \
    --quantize-bits-a 2 \
    --quantize-group-size 256
    """