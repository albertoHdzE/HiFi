"""
analyze_rank_sweep.py -- LoRA rank sweep analysis (P11-E4-T3).

Reads data/training/rank_sweep_results.json and prints a formatted table
with the recommended rank.

Usage:
    uv run python scripts/analyze_rank_sweep.py [--data-dir DIR]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    import argparse
    sys.path.insert(0, str(_ROOT / "src"))

    parser = argparse.ArgumentParser(description="Analyze LoRA rank sweep results.")
    parser.add_argument("--data-dir", default=str(_ROOT / "data"))
    args = parser.parse_args()

    sweep_path = Path(args.data_dir) / "training" / "rank_sweep_results.json"
    if not sweep_path.exists():
        print(f"No rank sweep results found at {sweep_path}")
        print("Run first: uv run python scripts/run_phase11_finetune.py --rank-sweep")
        sys.exit(1)

    results = json.loads(sweep_path.read_text())

    print("\nLoRA rank sweep results (Technical Agent)")
    print(f"{'rank':>6}  {'train_loss':>12}  {'duration_s':>12}  {'quality_ok':>12}")
    print("-" * 50)

    for rank_str in sorted(results, key=int):
        v = results[rank_str]
        loss_str = f"{v['train_loss']:.4f}" if v.get("train_loss") is not None else "N/A"
        dur_str = f"{v['duration_seconds']:.0f}" if v.get("duration_seconds") is not None else "N/A"
        quality_str = str(v.get("quality_ok", False))
        print(f"{rank_str:>6}  {loss_str:>12}  {dur_str:>12}  {quality_str:>12}")

    # Read optimal rank recommendation
    optimal_path = Path(args.data_dir) / "training" / "optimal_rank.json"
    if optimal_path.exists():
        opt_data = json.loads(optimal_path.read_text())
        optimal = opt_data.get("optimal_rank", "N/A")
        justification = opt_data.get("justification", "")
        print(f"\nRecommended rank: {optimal}  ({justification})")
    else:
        from hifi.models.fine_tune import optimal_rank_from_sweep
        optimal = optimal_rank_from_sweep(results)
        print(f"\nRecommended rank: {optimal}  (lowest loss with quality_ok=True)")


if __name__ == "__main__":
    main()
