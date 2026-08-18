"""Repopulate data/macro/ with per-series FRED parquets (DJ-120).

Why this exists
---------------
``financial_server._load_all_macro`` globs ``data/macro/*.parquet`` and calls
``read_macro`` on each, which requires the ``hifi_dataset_metadata`` schema
block and the long (date, value) single-series layout written by
``write_macro``. The store instead held one *wide* frame
(``macro.parquet``: treasury_10y / treasury_2y / spread_10y2y) produced by an
earlier regime-detection phase. ``read_macro`` raised on it, ``_load_all_macro``
returned ``{}``, and every ``get_macro_snapshot`` call answered
``NO_MACRO_DATA`` — so the macro agent voted a near-constant Hold on 100% of
passes across Phase 15 and the Phase 16 live run.

This script fetches the six series ``compute_macro_snapshot`` actually reads
and writes each one in the format the reader expects. The legacy wide frame is
moved to ``macro_wide_legacy.parquet.bak`` so the glob no longer trips on it
while the original data is preserved.

Usage
-----
    uv run python scripts/refresh_macro_store.py [--start 2004-01-01] [--quiet]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logger = logging.getLogger("refresh_macro")

# Exactly the series consumed by hifi.engines.macro.compute_macro_snapshot
# (_DIRECT_FIELDS plus the CPI series used for the YoY calculation).
SERIES = [
    "FEDFUNDS", "CPIAUCSL", "UNRATE", "GS10", "GS2", "VIXCLS",
    "A191RL1Q225SBEA",  # real GDP growth, quarterly -> MacroSnapshotResult.gdp_growth
]

_LEGACY_WIDE = "macro.parquet"
_LEGACY_BACKUP = "macro_wide_legacy.parquet.bak"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2004-01-01", help="earliest observation date")
    ap.add_argument("--end", default=None, help="latest observation date (default: today)")
    ap.add_argument("--data-dir", default=os.environ.get("HIFI_DATA_DIR", "data"))
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    from hifi.data.macro import MacroDataFetcher
    from hifi.data.storage import write_macro

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    macro_dir = Path(args.data_dir) / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)

    # Retire the wide legacy frame: read_macro cannot parse it and the glob in
    # _load_all_macro would log a warning on every snapshot call.
    legacy = macro_dir / _LEGACY_WIDE
    if legacy.exists():
        legacy.rename(macro_dir / _LEGACY_BACKUP)
        logger.info("Moved legacy wide frame -> %s", _LEGACY_BACKUP)

    fetcher = MacroDataFetcher()
    rows = []
    failures = 0
    for series_id in SERIES:
        try:
            ds = fetcher.fetch_series(series_id, start=start, end=end)
            write_macro(ds, macro_dir / f"{series_id}.parquet")
            obs = [o for o in ds.observations if o.value is not None]
            last = max((o.date for o in obs), default=None)
            rows.append((series_id, len(ds.observations), str(last), ds.frequency))
            logger.info("%-16s %5d obs  last=%s", series_id, len(ds.observations), last)
        except Exception as exc:
            failures += 1
            rows.append((series_id, 0, "FAILED", str(exc)[:40]))
            logger.error("%-16s FAILED: %s", series_id, exc)

    print(f"\n{'series':<18}{'obs':>7}  {'last':<12}{'frequency'}")
    print("-" * 60)
    for sid, n, last, freq in rows:
        print(f"{sid:<18}{n:>7}  {last:<12}{freq}")
    print(f"\n{len(SERIES) - failures}/{len(SERIES)} series written to {macro_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
