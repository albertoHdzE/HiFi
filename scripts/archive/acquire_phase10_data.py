"""
Phase 10 market data acquisition — 12 new tickers (P10-E2-T1, DJ-046).

Downloads OHLCV price history for the 12 tickers added in Phase 10 to expand
the ensemble universe from 3 to 15 tickers (sector-diverse, DJ-046). Phase 1
tickers (AAPL, JPM, XOM, SPY) are untouched.

New tickers: MSFT, NVDA, GOOGL, BAC, GS, CVX, JNJ, UNH, AMZN, WMT, CAT, NEE
Date range:  2016-01-01 to 2023-06-30 (matches Phase 1 range)

Idempotent: skips tickers whose Parquet files already exist.
Usage: uv run python scripts/acquire_phase10_data.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os  # noqa: E402

from hifi.data.market import MarketDataFetcher  # noqa: E402, download from Yahoo Finance
from hifi.data.storage import write_ohlcv  # noqa: E402, save to a Parqu

_NEW_TICKERS = [
    "MSFT", "NVDA", "GOOGL",          # Technology
    "BAC", "GS",                       # Finance
    "CVX",                             # Energy
    "JNJ", "UNH",                      # Healthcare
    "AMZN", "WMT",                     # Consumer
    "CAT",                             # Industrial
    "NEE",                             # Utilities
]
_DATE_FROM = date(2016, 1, 1)
_DATE_TO = date(2023, 6, 30)


def _market_exists(ticker: str, data_dir: Path) -> bool:
    pattern = str(data_dir / "market" / f"{ticker}_*.parquet")
    return bool(_glob.glob(pattern))


def acquire_phase10_market(data_dir: Path) -> None:
    fetcher = MarketDataFetcher()
    market_dir = data_dir / "market"
    market_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPhase 10 Market Data  ({_DATE_FROM} to {_DATE_TO})")
    print(f"  New tickers:  {', '.join(_NEW_TICKERS)}")
    print(f"  Destination: {market_dir}")

    for ticker in _NEW_TICKERS:
        if _market_exists(ticker, data_dir):
            print(f"  {ticker:<6}  SKIP (already present)")
            continue
        print(f"  {ticker:<6}  downloading ... ", end="", flush=True)
        try:
            dataset = fetcher.fetch_ohlcv(ticker, _DATE_FROM, _DATE_TO)
            filename = f"{ticker}_{_DATE_FROM}_{_DATE_TO}.parquet"
            write_ohlcv(dataset, market_dir / filename)
            print(f"OK  ({len(dataset.bars)} bars)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            logger.exception("Failed to acquire %s", ticker)

    print("\nDone. Run 'make bootstrap' to re-seed the 15-ticker performance history.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire Phase 10 market Parquet files for 12 new tickers. Idempotent."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", str(_ROOT / "data")),
        help="Destination root directory (default: data/).",
    )
    args = parser.parse_args()
    print("Phase 10 Data Acquisition")
    print("=" * 60)
    print(f"Data dir: {args.data_dir}")
    acquire_phase10_market(Path(args.data_dir))


if __name__ == "__main__":
    main()
