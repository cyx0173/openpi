import dataclasses
import enum
import logging
import os
import pathlib
import random
import sys
import time
from typing import Callable

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from datasets import load_dataset
import polars as pl
import rich
import tqdm
import tyro

logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "0.0.0.0"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8000
    # API key to use for the server.
    api_key: str | None = None
    # Number of steps to run the policy for.
    num_steps: int = 20
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Environment to run the policy in.
    env: EnvMode = EnvMode.ALOHA_SIM
    # LeRobot dataset repo_id to load real data from. If None, uses random data.
    # Example: "your_hf_username/my_droid_dataset" (from convert_droid_data_to_lerobot.py)
    dataset_repo_id: str | None = None


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def main(args: Args) -> None:
    # Load real data if dataset_repo_id is provided, otherwise use random data
    if args.dataset_repo_id is not None:
        logger.info(f"Loading real data from LeRobot dataset: {args.dataset_repo_id}")
        obs_fn = _create_real_data_loader(args.env, args.dataset_repo_id)
    else:
        logger.info("Using random data (no dataset_repo_id provided)")
        obs_fn = {
            EnvMode.ALOHA: _random_observation_aloha,
            EnvMode.ALOHA_SIM: _random_observation_aloha,
            EnvMode.DROID: _random_observation_droid,
            EnvMode.LIBERO: _random_observation_libero,
        }[args.env]

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    # Send a few observations to make sure the model is loaded.
    for _ in range(2):
        policy.infer(obs_fn())

    timing_recorder = TimingRecorder()

    for _ in tqdm.trange(args.num_steps, desc="Running policy"):
        inference_start = time.time()
        action = policy.infer(obs_fn())
        timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
        for key, value in action.get("server_timing", {}).items():
            timing_recorder.record(f"server_{key}", value)
        for key, value in action.get("policy_timing", {}).items():
            timing_recorder.record(f"policy_{key}", value)

    timing_recorder.print_all_stats()

    if args.timing_file is not None:
        timing_recorder.write_parquet(args.timing_file)


def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _create_real_data_loader(env: EnvMode, repo_id: str) -> Callable[[], dict]:
    """Create a function that loads real observations from a LeRobot dataset."""
    hf_dataset = None
    dataset = None
    
    # Try to load the dataset directly via HuggingFace to avoid LeRobotDataset timestamp issues
    # LeRobot datasets are stored in $HF_LEROBOT_HOME or can be loaded from HuggingFace Hub
    hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
    local_path = pathlib.Path(hf_lerobot_home) / repo_id
    
    logger.info(f"Looking for dataset at local path: {local_path}")
    logger.info(f"HF_LEROBOT_HOME: {hf_lerobot_home}")
    
    # First, try loading from local path (priority)
    if local_path.exists():
        logger.info(f"Local path exists, attempting to load from: {local_path}")
        
        # Try loading from parquet files first
        parquet_files = list(local_path.glob("*.parquet"))
        if parquet_files:
            logger.info(f"Found {len(parquet_files)} parquet file(s), loading from parquet files")
            try:
                hf_dataset = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")
                logger.info("Successfully loaded dataset from parquet files")
            except Exception as e:
                logger.warning(f"Failed to load from parquet files: {e}")
                hf_dataset = None
        
        # If parquet loading failed, try loading the directory directly
        if hf_dataset is None:
            logger.info(f"Trying to load dataset directory directly: {local_path}")
            try:
                hf_dataset = load_dataset(str(local_path), split="train")
                logger.info("Successfully loaded dataset from directory")
            except Exception as e:
                logger.warning(f"Failed to load from directory: {e}")
                hf_dataset = None
    else:
        logger.info(f"Local path does not exist: {local_path}")
    
    # Only try HuggingFace Hub if local loading failed
    if hf_dataset is None:
        logger.info(f"Local loading failed, trying HuggingFace Hub: {repo_id}")
        try:
            hf_dataset = load_dataset(repo_id, split="train")
            logger.info("Successfully loaded dataset from HuggingFace Hub")
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace Hub: {e}")
            hf_dataset = None
    
    # Fallback: try using LeRobotDataset with minimal delta_timestamps
    if hf_dataset is None:
        logger.info("Direct HuggingFace load failed, trying LeRobotDataset as fallback")
        try:
            # Determine action key based on environment
            if env == EnvMode.DROID:
                action_key = "actions"
            else:
                action_key = "action"
            
            # Use minimal delta_timestamps to avoid timestamp processing issues
            delta_timestamps = {action_key: [0.0]}
            dataset = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
            logger.info("Successfully loaded dataset using LeRobotDataset")
        except Exception as e2:
            raise ValueError(
                f"Failed to load LeRobot dataset '{repo_id}'. "
                f"Make sure you have converted your data using convert_droid_data_to_lerobot.py. "
                f"The dataset should be saved in $HF_LEROBOT_HOME/{repo_id}. "
                f"Checked local path: {local_path} (exists: {local_path.exists()}). "
                f"Error: {e2}"
            ) from e2

    # Pre-load all observations into memory for faster access
    logger.info(f"Loading observations from dataset (this may take a moment)...")
    observations = []
    
    if hf_dataset is not None:
        # Use HuggingFace dataset directly
        dataset_len = len(hf_dataset)
        logger.info(f"Dataset contains {dataset_len} samples")
        
        pbar = tqdm.tqdm(
            total=dataset_len,
            desc="Loading dataset",
            ncols=100,
            mininterval=0.1,
            file=sys.stdout,
            dynamic_ncols=False,
            leave=False
        )
        for idx in range(dataset_len):
            try:
                sample = hf_dataset[idx]
                obs = _convert_lerobot_sample_to_observation(env, sample)
                if obs is not None:
                    observations.append(obs)
            except Exception as e:
                logger.warning(f"Failed to load sample {idx}: {e}")
            finally:
                pbar.update(1)
        pbar.close()
    else:
        # Use LeRobotDataset
        dataset_len = len(dataset)
        logger.info(f"Dataset contains {dataset_len} samples")
        
        pbar = tqdm.tqdm(
            total=dataset_len,
            desc="Loading dataset",
            ncols=100,
            mininterval=0.1,
            file=sys.stdout,
            dynamic_ncols=False,
            leave=False
        )
        for idx in range(dataset_len):
            try:
                sample = dataset[idx]
                obs = _convert_lerobot_sample_to_observation(env, sample)
                if obs is not None:
                    observations.append(obs)
            except Exception as e:
                logger.warning(f"Failed to load sample {idx}: {e}")
            finally:
                pbar.update(1)
        pbar.close()
    
    if not observations:
        raise ValueError(f"No valid observations found in dataset '{repo_id}'")
    
    logger.info(f"Successfully loaded {len(observations)} observations from dataset")
    
    # Create a function that returns random observations from the loaded data
    def get_observation() -> dict:
        return random.choice(observations)
    
    return get_observation


def _convert_lerobot_sample_to_observation(env: EnvMode, sample: dict) -> dict | None:
    """Convert a LeRobot dataset sample to the observation format expected by the policy."""
    try:
        if env == EnvMode.DROID:
            # LeRobot format uses keys like "exterior_image_1_left", "wrist_image_left", etc.
            # We need to convert to "observation/exterior_image_1_left" format
            obs = {}
            
            # Handle images - LeRobot stores as (H, W, C) uint8
            if "exterior_image_1_left" in sample:
                img = np.asarray(sample["exterior_image_1_left"])
                # Resize to 224x224 if needed (DROID policy expects 224x224)
                if img.shape[:2] != (224, 224):
                    from PIL import Image
                    img = np.array(Image.fromarray(img).resize((224, 224), Image.BICUBIC))
                obs["observation/exterior_image_1_left"] = img
            
            if "wrist_image_left" in sample:
                img = np.asarray(sample["wrist_image_left"])
                if img.shape[:2] != (224, 224):
                    from PIL import Image
                    img = np.array(Image.fromarray(img).resize((224, 224), Image.BICUBIC))
                obs["observation/wrist_image_left"] = img
            
            # Handle joint and gripper positions
            if "joint_position" in sample:
                obs["observation/joint_position"] = np.asarray(sample["joint_position"], dtype=np.float32)
            
            if "gripper_position" in sample:
                gripper = np.asarray(sample["gripper_position"], dtype=np.float32)
                # Ensure it's 1D array
                if gripper.ndim == 0:
                    gripper = gripper[np.newaxis]
                obs["observation/gripper_position"] = gripper
            
            # Handle prompt/task
            if "task" in sample:
                task = sample["task"]
                if isinstance(task, bytes):
                    task = task.decode("utf-8")
                obs["prompt"] = task
            else:
                obs["prompt"] = "do something"
            
            return obs
        
        elif env == EnvMode.LIBERO:
            obs = {}
            
            if "state" in sample:
                obs["observation/state"] = np.asarray(sample["state"], dtype=np.float32)
            
            if "image" in sample:
                img = np.asarray(sample["image"])
                if img.shape[:2] != (224, 224):
                    from PIL import Image
                    img = np.array(Image.fromarray(img).resize((224, 224), Image.BICUBIC))
                obs["observation/image"] = img
            
            if "wrist_image" in sample:
                img = np.asarray(sample["wrist_image"])
                if img.shape[:2] != (224, 224):
                    from PIL import Image
                    img = np.array(Image.fromarray(img).resize((224, 224), Image.BICUBIC))
                obs["observation/wrist_image"] = img
            
            if "task" in sample:
                task = sample["task"]
                if isinstance(task, bytes):
                    task = task.decode("utf-8")
                obs["prompt"] = task
            else:
                obs["prompt"] = "do something"
            
            return obs
        
        elif env in (EnvMode.ALOHA, EnvMode.ALOHA_SIM):
            # ALOHA format conversion would go here if needed
            logger.warning(f"Real data loading for {env} is not yet implemented, using random data")
            return None
        
        else:
            logger.warning(f"Unsupported environment for real data: {env}")
            return None
    
    except Exception as e:
        logger.warning(f"Failed to convert sample: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
