"""
Namespace management for Phase 14+ LanceDB stores (E6-T2, DJ-093).

Supports reset / list / status operations on a named namespace.

Usage:
    uv run python scripts/manage_namespaces.py --action reset   --namespace hifi-eval
    uv run python scripts/manage_namespaces.py --action list    --namespace hifi-eval
    uv run python scripts/manage_namespaces.py --action status  --namespace hifi-eval

See Makefile targets: eval-reset, live-reset.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _connect(db_path: str) -> object:
    import lancedb
    return lancedb.connect(db_path)


def _all_tables(db: object) -> list[str]:
    result = db.list_tables()
    if hasattr(result, "tables"):
        return list(result.tables)
    return list(result)


def action_reset(db_path: str, namespace: str) -> int:
    """
    Drop all tables belonging to ``namespace``.

    Parameters
    ----------
    db_path : str
        LanceDB directory path.
    namespace : str
        Namespace prefix (e.g. "hifi-eval").

    Returns
    -------
    int
        Number of tables dropped.
    """
    db = _connect(db_path)
    prefix = f"{namespace}-"
    all_tables = _all_tables(db)
    to_drop = [t for t in all_tables if t.startswith(prefix)]
    for table in to_drop:
        try:
            db.drop_table(table)
            logger.info("Dropped table: %s", table)
        except Exception as exc:
            logger.warning("Failed to drop %s: %s", table, exc)
    return len(to_drop)


def action_list(db_path: str, namespace: str) -> list[str]:
    """Return logical table names (prefix stripped) in the given namespace."""
    db = _connect(db_path)
    prefix = f"{namespace}-"
    all_tables = _all_tables(db)
    return [t[len(prefix):] for t in all_tables if t.startswith(prefix)]


def action_status(db_path: str, namespace: str) -> dict[str, int]:
    """
    Return record count per table in the given namespace.

    Returns
    -------
    dict[str, int]
        Logical table name → row count.
    """
    db = _connect(db_path)
    prefix = f"{namespace}-"
    all_tables = _all_tables(db)
    status: dict[str, int] = {}
    for physical in all_tables:
        if not physical.startswith(prefix):
            continue
        logical = physical[len(prefix):]
        try:
            tbl = db.open_table(physical)
            count = len(tbl.to_pandas())
        except Exception as exc:
            logger.warning("Could not count rows in %s: %s", physical, exc)
            count = -1
        status[logical] = count
    return status


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Manage LanceDB namespace tables for HiFi Phase 14+"
    )
    p.add_argument(
        "--action",
        choices=["reset", "list", "status"],
        required=True,
        help="Action to perform on the namespace",
    )
    p.add_argument(
        "--namespace",
        required=True,
        help="Namespace prefix (e.g. hifi-eval, hifi-live, hifi-dev)",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Path to data directory (default: HIFI_DATA_DIR env var or 'data')",
    )
    args = p.parse_args()

    data_dir = args.data_dir or os.environ.get("HIFI_DATA_DIR", "data")
    db_path = str(Path(data_dir) / "knowledge.lance")

    if args.action == "reset":
        n = action_reset(db_path, args.namespace)
        print(f"Reset {args.namespace}: dropped {n} table(s).")

    elif args.action == "list":
        tables = action_list(db_path, args.namespace)
        if tables:
            print(f"Tables in namespace '{args.namespace}':")
            for t in tables:
                print(f"  {t}")
        else:
            print(f"No tables found in namespace '{args.namespace}'.")

    elif args.action == "status":
        status = action_status(db_path, args.namespace)
        if status:
            print(f"Status for namespace '{args.namespace}':")
            for table, count in sorted(status.items()):
                print(f"  {table}: {count} rows")
        else:
            print(f"No tables found in namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
