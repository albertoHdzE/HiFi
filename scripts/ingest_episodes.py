"""
Episode ingestion stub with temporal filtering (E6-T3, DJ-093).

Provides --through-date DATE filtering for walk-forward evaluation isolation
(Phase 15).  Populates a target LanceDB namespace with episodes up to DATE.

Usage:
    uv run python scripts/ingest_episodes.py [--namespace NS] [--through-date DATE]

Temporal discipline:
    When --through-date is provided, only episodes with decision_date <= DATE
    are ingested into the target namespace.  This ensures that Phase 15 evaluation
    windows cannot see future data.

See Makefile target: eval-ingest-through
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filtering helper (importable for unit testing, E6-T3)
# ---------------------------------------------------------------------------


def filter_by_through_date(
    records: list[dict[str, Any]],
    through_date: str | None,
    date_field: str = "decision_date",
) -> list[dict[str, Any]]:
    """
    Filter records to only those whose ``date_field`` is on or before ``through_date``.

    Parameters
    ----------
    records : list[dict]
        Records to filter.  Each must have a ``date_field`` key with an ISO 8601
        date string (e.g. "2023-03-31").
    through_date : str | None
        Cutoff date (ISO 8601, inclusive).  If None, all records are returned.
    date_field : str
        Key to use for date comparison (default "decision_date").

    Returns
    -------
    list[dict]
        Subset of records with date_field <= through_date, or all records
        when through_date is None.
    """
    if through_date is None:
        return list(records)
    return [r for r in records if str(r.get(date_field, "")) <= through_date]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest episodes into a LanceDB namespace")
    p.add_argument("--namespace", default="hifi-dev", help="Target LanceDB namespace")
    p.add_argument(
        "--through-date",
        default=None,
        help="Only ingest episodes with decision_date <= DATE (ISO 8601)",
    )
    p.add_argument("--data-dir", default=None, help="Data directory path")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    if args.through_date:
        logger.info(
            "ingest_episodes: namespace=%s through-date=%s",
            args.namespace, args.through_date,
        )
    else:
        logger.info("ingest_episodes: namespace=%s (all dates)", args.namespace)

    # Stub: actual ingestion logic will be implemented in Phase 15 when the
    # walk-forward evaluation pipeline is built.  The through_date filtering
    # utility (filter_by_through_date) is already testable.
    print(
        f"ingest_episodes: stub — namespace={args.namespace} "
        f"through_date={args.through_date or 'all'}"
    )


if __name__ == "__main__":
    main()
