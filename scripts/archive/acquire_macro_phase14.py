"""
Phase 14 FRED macro indicator extension (E2-T4, DJ-090).

Extends Phase 1 FRED acquisition with:
  - 10Y Treasury yield (GS10)
  - 10Y-2Y spread (GS10 - GS2)
Period: 2004-01-01 to 2025-12-31.

Appends new series to data/macro/macro.parquet.

Usage:
    uv run python scripts/acquire_macro_phase14.py [--data-dir DIR]

Make target: acquire-macro-phase14  (internet required, ~5 min)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

NEW_SERIES = {
    "GS10": "treasury_10y",
    "GS2":  "treasury_2y",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    import fredapi
    import pandas as pd

    fred_key = os.environ.get("FRED_API_KEY", "")
    fred = fredapi.Fred(api_key=fred_key)
    data_dir = Path(args.data_dir or os.environ.get("HIFI_DATA_DIR", "data"))
    macro_path = data_dir / "macro" / "macro.parquet"

    existing = pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()

    frames = [existing] if not existing.empty else []
    for series_id, col_name in NEW_SERIES.items():
        if col_name in existing.columns:
            logger.info("Series %s already present; skipping", col_name)
            continue
        try:
            s = fred.get_series(series_id, observation_start="2004-01-01",
                                observation_end="2025-12-31")
            s.name = col_name
            frames.append(s.to_frame())
            logger.info("Fetched %s (%d obs)", series_id, len(s))
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", series_id, exc)

    if frames:
        merged = pd.concat(frames, axis=1).sort_index()
        # Compute 10Y-2Y spread if both present
        if "treasury_10y" in merged.columns and "treasury_2y" in merged.columns:
            merged["spread_10y2y"] = merged["treasury_10y"] - merged["treasury_2y"]
        macro_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(macro_path)
        logger.info("Saved macro.parquet: %d rows, %d columns", len(merged), len(merged.columns))
    else:
        logger.warning("No new macro series added")


if __name__ == "__main__":
    main()
