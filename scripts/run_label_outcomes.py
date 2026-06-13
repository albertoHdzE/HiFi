"""
run_label_outcomes.py -- Incremental weight update hook (P11-E5-T1, DJ-060).

Labels any unlabeled records in data/agent_performance_history.json where
60 trading days of forward data exist. Updates agent accuracy weights accordingly.

No LM Studio required. Pure Parquet + pandas computation.

Usage:
    uv run python scripts/run_label_outcomes.py [--data-dir DIR] [--horizon 60] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hifi.collective.labeler import compute_forward_return  # noqa: E402
from hifi.collective.performance_store import (  # noqa: E402
    compute_weights,
    load_history,
    save_history,
)
from hifi.collective.schemas import AgentPerformanceHistory  # noqa: E402

_LABEL_THRESHOLD = 0.02  # DJ-042: ±2% band


def _apply_label(decision: str, forward_return: float) -> bool:
    """Apply DJ-042 labeling rules."""
    if decision == "Buy":
        return forward_return > _LABEL_THRESHOLD
    if decision == "Sell":
        return forward_return < -_LABEL_THRESHOLD
    return abs(forward_return) <= _LABEL_THRESHOLD


def label_unlabeled_records(
    data_dir: str,
    horizon_days: int = 60,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Label unlabeled records in agent_performance_history.json.

    For each record where outcome_correct is None, computes the forward return
    using the record's analysis_date + horizon_days. If sufficient forward data
    exists (forward price is available in the OHLCV Parquet), labels the record
    with DJ-042 rules and updates the timestamp. Saves atomically with FileLock.

    Returns (n_newly_labeled, n_still_unlabeled).
    n_still_unlabeled means forward data is not yet available (date in the future
    or beyond available Parquet range).
    """
    from filelock import FileLock

    data_path = Path(data_dir)
    history_path = data_path / "agent_performance_history.json"

    if not history_path.exists():
        print(f"[label-outcomes] No history file at {history_path} -- nothing to label.")
        return 0, 0

    lock_path = history_path.with_suffix(".json.lock")
    now_iso = datetime.now(tz=UTC).isoformat()

    with FileLock(str(lock_path)):
        history = load_history(data_dir)

        n_newly_labeled = 0
        n_still_unlabeled = 0
        new_records = []

        for record in history.records:
            if record.outcome_correct is not None:
                new_records.append(record)
                continue

            h = record.horizon_days if record.horizon_days else horizon_days
            fwd = compute_forward_return(record.ticker, record.analysis_date, data_dir, h)

            if fwd is None:
                new_records.append(record)
                n_still_unlabeled += 1
                continue

            labeled = record.model_copy(update={
                "forward_return": fwd,
                "outcome_correct": _apply_label(record.decision, fwd),
                "outcome_labeled_at": now_iso,
            })
            new_records.append(labeled)
            n_newly_labeled += 1

        if dry_run:
            print(f"[label-outcomes] DRY RUN: would label {n_newly_labeled} records.")
            print(f"[label-outcomes] {n_still_unlabeled} records still need forward data.")
            return n_newly_labeled, n_still_unlabeled

        updated_weights = compute_weights(new_records)
        updated = AgentPerformanceHistory(
            records=new_records,
            weights=updated_weights,
            last_updated=now_iso,
            n_labeled=0,  # auto-computed by model_validator
        )
        save_history(updated, data_dir)

    return n_newly_labeled, n_still_unlabeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label unlabeled performance records where forward data is available."
    )
    parser.add_argument(
        "--data-dir",
        default=str(_ROOT / "data"),
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
        help="Forward return horizon in trading days (default: 60)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be labeled without writing changes.",
    )
    args = parser.parse_args()

    n_labeled, n_unlabeled = label_unlabeled_records(args.data_dir, args.horizon, args.dry_run)
    print(f"[label-outcomes] Newly labeled: {n_labeled}")
    print(f"[label-outcomes] Still unlabeled (forward data pending): {n_unlabeled}")


if __name__ == "__main__":
    main()
