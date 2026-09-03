"""
run_phase11_finetune.py -- Main LoRA fine-tuning pipeline (P11-E3-T2, DJ-056).

Orchestrates training for Technical and Fundamental agents. Supports:
  --rank-sweep: rank sweep (4, 8, 16, 32) on Technical Agent to find optimal rank
  --agent: train a specific agent (technical|fundamental|both)
  --rank: explicit rank (overrides sweep result)

Output:
  data/adapters/technical_v1/         -- Technical Agent LoRA adapters
  data/adapters/fundamental_v1/       -- Fundamental Agent LoRA adapters
  data/training/rank_sweep_results.json
  data/training/optimal_rank.json

Usage:
    uv run python scripts/run_phase11_finetune.py [--agent technical|fundamental|both]
                                                   [--rank N]
                                                   [--rank-sweep]
                                                   [--data-dir DIR]
                                                   [--iters N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from hifi.models.fine_tune import (  # noqa: E402
    check_adapter_quality,
    load_rank_sweep_results,
    optimal_rank_from_sweep,
    run_lora_training,
)

# Use local LM Studio copy to avoid re-downloading 32GB model (DJ-056)
_MODEL_BASE = str(Path.home() / ".lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-8bit")  # noqa: E501
_VENV_PYTHON = str(_ROOT / "venvs" / "finetune" / "bin" / "python")
_SWEEP_RANKS = [4, 8, 16, 32]


def _training_file(data_dir: str, agent: str, horizon: int) -> str:
    if agent == "technical":
        return str(Path(data_dir) / "training" / f"technical_max_return_{horizon}d.jsonl")
    return str(Path(data_dir) / "training" / f"fundamental_risk_adjusted_{horizon}d.jsonl")


def run_rank_sweep(data_dir: str, horizon: int, num_iters: int) -> dict:
    """Run LoRA rank sweep (4, 8, 16, 32) on Technical Agent."""
    train_file = _training_file(data_dir, "technical", horizon)
    sweep_results: dict[str, dict] = {}

    print(f"\nLoRA rank sweep -- Technical Agent ({len(_SWEEP_RANKS)} ranks)")
    print(f"Training file: {train_file}")
    print(f"Iterations per rank: {num_iters}")

    for rank in _SWEEP_RANKS:
        adapter_dir = str(Path(data_dir) / "adapters" / f"technical_rank{rank}")
        print(f"\n  Rank {rank}: training...", flush=True)
        try:
            result = run_lora_training(
                model_path=_MODEL_BASE,
                train_file=train_file,
                output_dir=adapter_dir,
                lora_rank=rank,
                num_iters=num_iters,
                venv_python=_VENV_PYTHON,
            )
            quality_ok = check_adapter_quality(adapter_dir, _MODEL_BASE, _VENV_PYTHON)
            sweep_results[str(rank)] = {
                "train_loss": result["train_loss"],
                "duration_seconds": result["duration_seconds"],
                "quality_ok": quality_ok,
                "n_examples": result["n_examples"],
            }
            print(f"  Rank {rank}: loss={result['train_loss']}, duration={result['duration_seconds']}s, quality_ok={quality_ok}")  # noqa: E501
        except Exception as exc:
            logger.warning("Rank %d sweep failed: %s", rank, exc)
            sweep_results[str(rank)] = {
                "train_loss": None,
                "duration_seconds": None,
                "quality_ok": False,
                "error": str(exc),
            }

    # Save sweep results
    sweep_path = Path(data_dir) / "training" / "rank_sweep_results.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text(json.dumps(sweep_results, indent=2))
    print(f"\nRank sweep results saved to: {sweep_path}")

    # Determine optimal rank
    optimal = optimal_rank_from_sweep(sweep_results)
    optimal_path = Path(data_dir) / "training" / "optimal_rank.json"
    optimal_path.write_text(json.dumps({
        "optimal_rank": optimal,
        "justification": "Lowest train_loss among quality_ok=True ranks in sweep.",
        "sweep_summary": sweep_results,
    }, indent=2))
    print(f"Optimal rank: {optimal}  (saved to {optimal_path})")

    return sweep_results


def run_agent_training(
    agent: str,
    rank: int,
    data_dir: str,
    horizon: int,
    num_iters: int,
    combine_compliance: bool = True,
    adapter_name: str | None = None,
) -> None:
    """Train one agent. Optionally combines main JSONL with compliance examples.

    Parameters
    ----------
    adapter_name : str | None
        Override the output adapter directory name (default: ``{agent}_v1``).
        Use ``technical_v2`` for the Phase 12 compliance-fix re-train (DJ-061).
    """
    import tempfile

    train_file = _training_file(data_dir, agent, horizon)
    compliance_file = str(Path(data_dir) / "training" / f"{agent}_compliance.jsonl")
    _adapter_name = adapter_name if adapter_name else f"{agent}_v1"
    adapter_dir = str(Path(data_dir) / "adapters" / _adapter_name)

    if combine_compliance and Path(compliance_file).exists():
        # Merge main + compliance JSONL into a temp file
        # delete=False: the trainer reads this path after the handle closes.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
            prefix=f"hifi_{agent}_combined_",
        ) as tmpfile:
            for src in (train_file, compliance_file):
                with open(src) as f:
                    tmpfile.write(f.read())
            effective_train_file = tmpfile.name
        logger.info("Combined training file: %s + %s", train_file, compliance_file)
    else:
        effective_train_file = train_file

    print(f"\nTraining {agent} agent: rank={rank}, iters={num_iters}")
    print(f"  Training data: {effective_train_file}")
    print(f"  Output: {adapter_dir}")

    try:
        result = run_lora_training(
            model_path=_MODEL_BASE,
            train_file=effective_train_file,
            output_dir=adapter_dir,
            lora_rank=rank,
            num_iters=num_iters,
            venv_python=_VENV_PYTHON,
        )
        print(f"  Completed: {result['n_examples']} examples, {result['duration_seconds']}s")
        quality_ok = check_adapter_quality(adapter_dir, _MODEL_BASE, _VENV_PYTHON)
        print(f"  Adapter quality check: {'PASS' if quality_ok else 'FAIL'}")
        if not quality_ok:
            logger.warning("Adapter quality check failed for %s. Adapters may be unusable.", agent)
    finally:
        if effective_train_file != train_file:
            Path(effective_train_file).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 11 LoRA fine-tuning pipeline.")
    parser.add_argument("--agent", choices=["technical", "fundamental", "both"], default="both")
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank (overrides sweep result)")
    parser.add_argument("--rank-sweep", action="store_true", help="Run rank sweep before training")
    parser.add_argument("--data-dir", default=str(_ROOT / "data"))
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--sweep-iters", type=int, default=300, help="Iterations per rank in sweep")
    parser.add_argument(
        "--adapter-name", default=None,
        help=(
            "Override output adapter directory name (default: {agent}_v1). "
            "Use 'technical_v2' for the Phase 12 compliance-fix re-train (DJ-061): "
            "uv run python scripts/run_phase11_finetune.py --agent technical "
            "--rank 8 --iters 500 --adapter-name technical_v2"
        ),
    )
    args = parser.parse_args()

    # Determine rank to use
    selected_rank = args.rank
    if args.rank_sweep:
        sweep_results = run_rank_sweep(args.data_dir, args.horizon, args.sweep_iters)
        if selected_rank is None:
            selected_rank = optimal_rank_from_sweep(sweep_results)
    elif selected_rank is None:
        # Load from previous sweep if available
        existing_sweep = load_rank_sweep_results(str(Path(args.data_dir) / "training"))
        if existing_sweep:
            selected_rank = optimal_rank_from_sweep(existing_sweep)
            logger.info("Using optimal rank from previous sweep: %d", selected_rank)
        else:
            selected_rank = 8  # default per DJ-056
            logger.info("No sweep results found; using default rank=%d", selected_rank)

    # Train agents
    if args.agent in ("technical", "both"):
        run_agent_training(
            "technical", selected_rank, args.data_dir, args.horizon, args.iters,
            adapter_name=args.adapter_name,
        )

    if args.agent in ("fundamental", "both"):
        # --adapter-name only applies to a single-agent run; ignore for "both"
        fund_adapter = args.adapter_name if args.agent == "fundamental" else None
        run_agent_training(
            "fundamental", selected_rank, args.data_dir, args.horizon, args.iters,
            adapter_name=fund_adapter,
        )

    print(f"\nFine-tuning complete. Adapters in: {args.data_dir}/adapters/")


if __name__ == "__main__":
    main()
