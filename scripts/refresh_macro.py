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

MUST use hifi.data.storage.write_macro, never df.to_parquet
------------------------------------------------------------
``write_macro`` embeds series_id, name, frequency, unit and provenance in the
Parquet *schema metadata*, and ``read_macro`` raises without it. The first
version of this script wrote the files with a plain ``df.to_parquet()``, which
silently dropped that metadata and changed ``date`` from date32 to timestamp.
All seven series became unreadable; ``financial_server._load_all_macro``
swallows the per-file exception with a warning and returns ``{}``; and the
macro agent then reported NO_MACRO_DATA and voted Hold on 193 of 194 passes on
2026-08-31, up from 54% Hold on 2026-08-24.

That is the DJ-120 / DJ-133a pattern reproduced by the very script written to
prevent staleness: an agent blinded by a data-path change, rendering the
blindness as a confident decision. Round-tripping through read_macro is
verified below before anything is written.

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
from datetime import UTC, datetime
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

    from hifi.data.macro import SERIES_METADATA
    from hifi.data.schemas import MacroDataset, MacroIndicator, ProvenanceRecord
    from hifi.data.storage import read_macro, write_macro

    path = data_dir / "macro" / f"{series_id}.parquet"
    existing = None
    if path.exists():
        try:
            existing = pd.read_parquet(path)[["date", "value"]]
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

    meta = SERIES_METADATA.get(series_id, {})
    fetched_at = datetime.now(UTC)
    merged["date"] = pd.to_datetime(merged["date"])
    dataset = MacroDataset(
        series_id=series_id,
        name=meta.get("name", series_id),
        frequency=meta.get("frequency", "unknown"),
        unit=meta.get("unit", "unknown"),
        observations=[
            MacroIndicator(series_id=series_id, date=r.date.date(), value=float(r.value))
            for r in merged.itertuples()
        ],
        source="FRED",
        fetched_at=fetched_at,
        date_from=merged["date"].min().date(),
        date_to=merged["date"].max().date(),
        provenance=ProvenanceRecord(
            source="FRED",
            fetched_at=fetched_at,
            parameters={"series_id": series_id},
        ),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    write_macro(dataset, path)

    # Round-trip before declaring success. The defect this guards against was
    # invisible at write time and only surfaced as an agent voting Hold on
    # everything three days later.
    try:
        back = read_macro(path)
        if back.series_id != series_id or len(back.observations) != len(merged):
            raise ValueError(
                f"round trip mismatch: {back.series_id} "
                f"{len(back.observations)} vs {len(merged)}"
            )
    except Exception as exc:
        logger.error("%s: WROTE AN UNREADABLE FILE (%s)", series_id, exc)
        return {"series": series_id, "status": "unreadable_after_write"}

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
