#!/usr/bin/env python
"""Refresh the FRED macro series in place, merging rather than overwriting.

The macro agent reads ``data/macro/<SERIES_ID>.parquet`` (columns: date, value).
``acquire_macro_phase14.py`` targets a combined ``macro.parquet`` that does not
exist in the live layout, so these per-series files had no refresh path at all
and drifted: VIXCLS, a *daily* series, was 17 days stale on 2026-08-30 while the
macro agent quoted it as current.

Monthly and quarterly series (CPI, FEDFUNDS, GS10, GS2, UNRATE, GDP) publish
with a genuine lag and being "behind" is correct for them; only the gap against
their own publication schedule matters. The daily series is the one that must
track the calendar.

Merge semantics match scripts/refresh_fundamentals.py: union of dates, fresh
values win on overlap so revisions propagate, existing history never dropped.

Usage
-----
    uv run python scripts/refresh_macro.py [--quiet]

Requires FRED_API_KEY (loaded from .env if present).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("refresh_macro")

#: FRED series the agents actually read, with their native publication cadence.
SERIES = {
    "VIXCLS": "daily",
    "GS10": "monthly",
    "GS2": "monthly",
    "FEDFUNDS": "monthly",
    "CPIAUCSL": "monthly",
    "UNRATE": "monthly",
    "A191RL1Q225SBEA": "quarterly",
}


def _load_env(root: Path) -> None:
    """Read FRED_API_KEY from .env when it is not already in the environment."""
    if os.environ.get("FRED_API_KEY"):
        return
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if line.startswith("FRED_API_KEY="):
            os.environ["FRED_API_KEY"] = line.split("=", 1)[1].strip()
            return


def refresh_series(series_id: str, data_dir: Path, fred, quiet: bool = False) -> dict:
    import pandas as pd

    path = data_dir / "macro" / f"{series_id}.parquet"
    existing = None
    if path.exists():
        try:
            existing = pd.read_parquet(path)
        except Exception as exc:
            logger.error("%s: existing parquet unreadable (%s); refusing to clobber",
                         series_id, exc)
            return {"series": series_id, "status": "unreadable"}

    try:
        raw = fred.get_series(series_id)
    except Exception as exc:
        logger.error("%s: FRED fetch failed: %s", series_id, exc)
        return {"series": series_id, "status": "fetch_failed"}

    fresh = (
        raw.rename("value").rename_axis("date").reset_index().dropna(subset=["value"])
    )
    fresh["date"] = pd.to_datetime(fresh["date"])

    if existing is None or existing.empty:
        merged = fresh
    else:
        existing["date"] = pd.to_datetime(existing["date"])
        # Fresh last so it wins on duplicate dates: FRED revises published values.
        merged = (
            pd.concat([existing, fresh], ignore_index=True)
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

    before = 0 if existing is None else len(existing)
    prev_latest = None if existing is None or existing.empty else existing["date"].max()

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)

    report = {
        "series": series_id,
        "status": "ok",
        "rows_before": before,
        "rows_after": len(merged),
        "was": None if prev_latest is None else str(prev_latest.date()),
        "now": str(merged["date"].max().date()),
    }
    if not quiet:
        logger.info(
            "%-18s %5d -> %5d rows, latest %s -> %s",
            series_id, before, len(merged), report["was"], report["now"],
        )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    root = Path(__file__).resolve().parent.parent
    _load_env(root)
    if not os.environ.get("FRED_API_KEY"):
        logger.error("FRED_API_KEY not set and not found in .env")
        return 1

    import fredapi

    fred = fredapi.Fred(api_key=os.environ["FRED_API_KEY"])
    data_dir = Path(args.data_dir)

    reports = [refresh_series(s, data_dir, fred, quiet=args.quiet) for s in SERIES]
    failed = [r for r in reports if r["status"] != "ok"]

    print(f"\nrefreshed {len(reports) - len(failed)}/{len(reports)} series")
    for r in reports:
        if r["status"] == "ok" and r["was"] != r["now"]:
            print(f"  {r['series']:18} {r['was']} -> {r['now']}")
    if failed:
        print(f"  FAILED: {[(r['series'], r['status']) for r in failed]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
