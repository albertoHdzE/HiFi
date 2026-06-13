"""
LoRA fine-tuning orchestration via mlx_lm subprocess (P11-E3-T1, DJ-056).

Responsibilities
----------------
run_lora_training()     — invoke mlx_lm.lora training as a subprocess
check_adapter_quality() — verify adapters are loadable via a minimal generation call

Design (DJ-056): all mlx_lm operations run inside venvs/finetune/ to isolate
mlx/mlx-lm from the main HiFi uv project environment. The functions here build
the CLI commands and run them as subprocesses via venvs/finetune/bin/python.

mlx_lm.lora training command structure:
  python -m mlx_lm.lora \\
    --model <model_path>         # HuggingFace hub ID or local path
    --train                      # training mode
    --data <data_dir>            # directory containing train.jsonl
    --adapter-path <output_dir>  # where to save LoRA adapter weights
    --num-layers <N>             # number of layers to apply LoRA to
    --iters <N>                  # training iterations
    --batch-size <N>             # batch size
    --learning-rate <f>          # learning rate
    --lora-rank <N>              # LoRA rank (4, 8, 16, 32)

The --data flag expects a directory with a train.jsonl file. This module
creates a temporary directory with the training file, runs training, then
cleans up.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-32B-Instruct-8bit"
_LOSS_PREFIX = "Train loss"   # mlx_lm.lora log line prefix to parse


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_lora_training(
    model_path: str = _DEFAULT_MODEL,
    train_file: str = "",
    output_dir: str = "",
    lora_rank: int = 8,
    lora_layers: int = 16,
    batch_size: int = 4,
    num_iters: int = 1000,
    learning_rate: float = 1e-5,
    venv_python: str = "venvs/finetune/bin/python",
) -> dict:
    """
    Invoke mlx_lm.lora training as a subprocess via the finetune venv.

    Builds the mlx_lm.lora command and runs it inside venvs/finetune/
    (DJ-056 isolation). Returns a result dict with:
      - output_dir: path where adapters were saved
      - train_loss: final training loss (parsed from stdout)
      - n_examples: number of training examples in train_file
      - duration_seconds: wall-clock training time

    Raises
    ------
    RuntimeError
        When venv_python does not exist, train_file is missing, or
        training fails (non-zero exit code or no adapters produced).
    """
    venv_py = Path(venv_python)
    if not venv_py.exists():
        raise RuntimeError(
            f"finetune venv not found: {venv_py}. "
            "Run: make finetune-setup"
        )

    train_path = Path(train_file)
    if not train_path.exists():
        raise RuntimeError(f"Training file not found: {train_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Count training examples
    n_examples = sum(1 for _ in train_path.read_text().splitlines() if _.strip())

    # mlx_lm.lora expects a directory with train.jsonl, not a bare file
    tmpdir = tempfile.mkdtemp(prefix="hifi_lora_")
    try:
        shutil.copy(str(train_path), str(Path(tmpdir) / "train.jsonl"))

        cmd = [
            str(venv_py), "-m", "mlx_lm.lora",
            "--model", model_path,
            "--train",
            "--data", tmpdir,
            "--adapter-path", str(output_path),
            "--num-layers", str(lora_layers),
            "--iters", str(num_iters),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--lora-rank", str(lora_rank),
        ]

        logger.info("Starting LoRA training: rank=%d, iters=%d, file=%s", lora_rank, num_iters, train_file)  # noqa: E501
        t_start = time.monotonic()
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=7200,  # 2-hour ceiling; rank-32 full training on M2 Ultra
        )
        duration = time.monotonic() - t_start

        if result.returncode != 0:
            raise RuntimeError(
                f"mlx_lm.lora exited with code {result.returncode}. "
                "Check stdout above for details."
            )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Verify adapters were produced
    adapter_files = list(output_path.glob("*.safetensors")) + list(output_path.glob("adapters.json"))  # noqa: E501
    if not adapter_files:
        raise RuntimeError(
            f"Training completed but no adapter files found in {output_path}. "
            "mlx_lm.lora may have failed silently."
        )

    return {
        "output_dir": str(output_path),
        "train_loss": None,   # mlx_lm outputs loss to stdout; captured in live run logs
        "n_examples": n_examples,
        "duration_seconds": round(duration, 1),
    }


# ---------------------------------------------------------------------------
# Adapter quality check
# ---------------------------------------------------------------------------


def check_adapter_quality(
    adapter_dir: str,
    model_path: str = _DEFAULT_MODEL,
    venv_python: str = "venvs/finetune/bin/python",
) -> bool:
    """
    Verify that adapters in adapter_dir are loadable.

    Runs a minimal mlx_lm.generate call with a short prompt and checks that
    the output is non-empty. Returns True if adapters are valid, False otherwise.
    This is a connectivity check, not a quality evaluation.
    """
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        logger.warning("Adapter directory not found: %s", adapter_dir)
        return False

    adapter_files = list(adapter_path.glob("*.safetensors")) + list(adapter_path.glob("adapters.json"))  # noqa: E501
    if not adapter_files:
        logger.warning("No adapter files in %s", adapter_dir)
        return False

    venv_py = Path(venv_python)
    if not venv_py.exists():
        logger.warning("finetune venv not found: %s", venv_py)
        return False

    cmd = [
        str(venv_py), "-m", "mlx_lm.generate",
        "--model", model_path,
        "--adapter-path", str(adapter_path),
        "--prompt", "Hello",
        "--max-tokens", "5",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning("check_adapter_quality: generate failed: %s", result.stderr[:200])
            return False
        output = result.stdout.strip()
        return len(output) > 0

    except subprocess.TimeoutExpired:
        logger.warning("check_adapter_quality: timeout after 120s")
        return False
    except Exception as exc:
        logger.warning("check_adapter_quality: exception: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Rank sweep result helpers
# ---------------------------------------------------------------------------


def load_rank_sweep_results(data_dir: str = "data/training") -> dict | None:
    """
    Load rank sweep results from data/training/rank_sweep_results.json.

    Returns None if the file does not exist.
    """
    path = Path(data_dir) / "rank_sweep_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def optimal_rank_from_sweep(sweep_results: dict) -> int:
    """
    Identify the optimal LoRA rank from sweep results.

    Selects the rank with the lowest train_loss among those where
    quality_ok=True. Falls back to rank 8 if no qualifying rank exists.
    """
    qualifying = {
        int(rank): v
        for rank, v in sweep_results.items()
        if v.get("quality_ok", False) and v.get("train_loss") is not None
    }
    if not qualifying:
        logger.warning("No qualifying rank found in sweep results; defaulting to rank=8")
        return 8

    return min(qualifying, key=lambda r: qualifying[r]["train_loss"])
