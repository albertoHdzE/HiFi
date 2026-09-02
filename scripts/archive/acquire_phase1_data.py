"""
Phase 1 market and macro data acquisition.

Downloads OHLCV price history and macroeconomic time series needed by all
downstream baseline scripts and MCP tool calls. Idempotent: skips any ticker
or macro series whose Parquet file already exists in DATA_DIR.

Market data (yfinance)
----------------------
Tickers:  AAPL, JPM, XOM (ensemble universe) + SPY (beta benchmark)
Range:    2016-01-01 to 2023-06-30
Covers:   bootstrap 2018-Q1 through 2022-Q4 (252-day rolling window needs
          ~1 year of history before first quarter-end) plus 60-day forward
          returns through 2023-06-30, and the 2023-03-31 live baseline.

Macro data (FRED via fredapi)
-----------------------------
Series:   FEDFUNDS, CPIAUCSL, UNRATE, GS10, GS2, VIXCLS, A191RL1Q225SBEA
Range:    2016-01-01 to 2023-06-30
Requires: FRED_API_KEY environment variable. Skipped with a warning if absent.

Idempotency
-----------
A Parquet file is considered present when at least one file matching
  DATA_DIR/market/{TICKER}_*.parquet
exists. Delete the file and re-run to force a fresh download.

Usage
-----
    uv run python scripts/acquire_phase1_data.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import os
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from hifi.data.market import MarketDataFetcher  # noqa: E402
from hifi.data.storage import write_ohlcv  # noqa: E402

_MARKET_TICKERS = ["AAPL", "JPM", "XOM", "SPY"]
_DATE_FROM = date(2016, 1, 1)
_DATE_TO = date(2023, 6, 30)

_MACRO_SERIES = [
    "FEDFUNDS", "CPIAUCSL", "UNRATE",
    "GS10", "GS2", "VIXCLS", "A191RL1Q225SBEA",
]


def _market_exists(ticker: str, data_dir: Path) -> bool:
    """Return True if at least one Parquet file exists for this ticker."""
    pattern = str(data_dir / "market" / f"{ticker}_*.parquet")
    return bool(_glob.glob(pattern))


def _macro_exists(series_id: str, data_dir: Path) -> bool:
    """Return True if at least one Parquet file exists for this macro series."""
    pattern = str(data_dir / "macro" / f"{series_id}_*.parquet")
    return bool(_glob.glob(pattern))


def acquire_market(data_dir: Path) -> None:
    fetcher = MarketDataFetcher()
    market_dir = data_dir / "market"
    market_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMarket data  ({_DATE_FROM} to {_DATE_TO})")
    print(f"  Destination: {market_dir}")

    for ticker in _MARKET_TICKERS:
        if _market_exists(ticker, data_dir):
            print(f"  {ticker:<6}  SKIP (already present)")
            continue

        print(f"  {ticker:<6}  downloading ... ", end="", flush=True)
        try:
            dataset = fetcher.fetch_ohlcv(ticker, _DATE_FROM, _DATE_TO)
            filename = f"{ticker}_{_DATE_FROM}_{_DATE_TO}.parquet"
            path = market_dir / filename
            write_ohlcv(dataset, path)
            n_bars = len(dataset.bars)
            print(f"OK  ({n_bars} bars -> {path.name})")
        except Exception as exc:
            print(f"FAILED: {exc}")
            logger.exception("Failed to acquire %s", ticker)


def acquire_macro(data_dir: Path) -> None:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print(
            "\nMacro data   SKIPPED"
            "\n  FRED_API_KEY is not set. Export it to download macro series:"
            "\n    export FRED_API_KEY=your_key_here"
            "\n  Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return

    from hifi.data.macro import MacroDataFetcher  # noqa: PLC0415
    from hifi.data.storage import write_macro  # noqa: PLC0415

    macro_dir = data_dir / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMacro data   ({_DATE_FROM} to {_DATE_TO})")
    print(f"  Destination: {macro_dir}")

    fetcher = MacroDataFetcher(api_key=api_key)

    for series_id in _MACRO_SERIES:
        if _macro_exists(series_id, data_dir):
            print(f"  {series_id:<24}  SKIP (already present)")
            continue

        print(f"  {series_id:<24}  downloading ... ", end="", flush=True)
        try:
            dataset = fetcher.fetch_series(series_id, _DATE_FROM, _DATE_TO)
            filename = f"{series_id}_{_DATE_FROM}_{_DATE_TO}.parquet"
            path = macro_dir / filename
            write_macro(dataset, path)
            n_obs = len(dataset.indicators)
            print(f"OK  ({n_obs} observations -> {path.name})")
        except Exception as exc:
            print(f"FAILED: {exc}")
            logger.exception("Failed to acquire %s", series_id)


def run_acquisition(data_dir: str) -> None:
    root = Path(data_dir)
    print("Phase 1 Data Acquisition")
    print("=" * 60)
    print(f"Data dir: {root}")

    acquire_market(root)
    acquire_macro(root)

    print("\nDone. Run 'make bootstrap-phase9' to seed performance history.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire Phase 1 market and macro Parquet files. "
            "Idempotent: skips tickers/series whose files already exist."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", str(_ROOT / "data")),
        help="Destination root directory (default: data/).",
    )
    args = parser.parse_args()
    run_acquisition(args.data_dir)


if __name__ == "__main__":
    main()
