"""
generate_reference_strategies.py -- Dataset Family C generation (P11-E1-T2, DJ-054).
Full code functionality explanation:
This script creates historical trading strategy labels for training AI trading
agents. It processes 15 stock tickers between 2016-2022 to generate two types of
reference strategy label, saved as Parquet files:
1. Max-return labels: for training a Technical Agent that prioritizes pure
   profit maximization
2. Risk-adjusted Sharpe ratio labels: for training a Fundamental Agent that
   balances returns against risk

Core workflow:
- Loads historical OHLCV price data for each ticker
- For every date in the 2016-2022 window, calculates future performance over the
  next 60 trading days
- Assigns a categorical "best action" label (buy/hold/sell) that would have
  delivered the optimal outcome
- Saves these label files for model fine-tuning, skipping any ticker that
  already has a valid label file

Idempotent: skips tickers whose output Parquet already exists with the correct
row count (>= 400 rows, sufficient for fine-tuning per DJ-054).

Output Parquets:
  data/reference_strategies/max_return/{ticker}_60d.parquet
  data/reference_strategies/risk_adjusted/{ticker}_60d.parquet

Usage:
    uv run python scripts/generate_reference_strategies.py [--data-dir DIR]
                                                           [--tickers AAPL,JPM,...]
                                                           [--horizon 60]
                                                           [--output-dir data/reference_strategies]
                                                           [--start-year 2016]
                                                           [--end-year 2022]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from hifi.models.training_data import (  # noqa: E402
    generate_max_return_labels,
    generate_risk_adjusted_labels,
)

# 15 trade tickers: original Phase 1 tickers (minus SPY benchmark) + Phase 10 tickers
_DEFAULT_TICKERS = [
    "AAPL", "JPM", "XOM",
    "MSFT", "NVDA", "GOOGL", "BAC", "GS", "CVX",
    "JNJ", "UNH", "AMZN", "WMT", "CAT", "NEE",
]

_MIN_ROWS = 400  # DJ-054: lower bound for LoRA literature


def _should_skip(path: Path, min_rows: int) -> bool:
    """Return True if output Parquet already exists with sufficient rows."""
    if not path.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return len(df) >= min_rows
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Dataset Family C reference strategy Parquets.")  # noqa: E501
    parser.add_argument("--data-dir", default=str(_ROOT / "data"), help="Root data directory")
    parser.add_argument("--tickers", default=",".join(_DEFAULT_TICKERS), help="Comma-separated tickers")  # noqa: E501
    parser.add_argument("--horizon", type=int, default=60, help="Evaluation horizon in trading days")  # noqa: E501
    parser.add_argument("--output-dir", default=str(_ROOT / "data" / "reference_strategies"))  # noqa: E501
    parser.add_argument("--start-year", type=int, default=2016, help="First year of training period")  # noqa: E501
    parser.add_argument("--end-year", type=int, default=2022, help="Last year of training period (inclusive)")  # noqa: E501
    args = parser.parse_args()


    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    output_dir = Path(args.output_dir)
    max_return_dir = output_dir / "max_return"
    risk_adj_dir = output_dir / "risk_adjusted"
    max_return_dir.mkdir(parents=True, exist_ok=True)
    risk_adj_dir.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_skipped = 0
    n_failed = 0

    for ticker in tickers:
        mr_path = max_return_dir / f"{ticker}_{args.horizon}d.parquet"
        ra_path = risk_adj_dir / f"{ticker}_{args.horizon}d.parquet"

        # --- Max-return labels ---
        if _should_skip(mr_path, _MIN_ROWS):
            logger.info("%s max_return: SKIPPED (already %d+ rows)", ticker, _MIN_ROWS)
            n_skipped += 1
        else:
            try:
                # generate_max_return_labels creates optimal action labels that maximize raw returns
                # It looks at the next `horizon` trading days of price data to pick the best action
                df = generate_max_return_labels(ticker, args.data_dir, args.horizon)
                if not df.empty:
                    # Filter to the training date window the caller asked for,
                    # so the labels align with the model training requirements
                    df = df[(df["date"].astype(str) >= f"{args.start_year}-01-01") &
                            (df["date"].astype(str) <= f"{args.end_year}-12-31")]
                n_rows = len(df)
                if n_rows == 0:
                    logger.warning("%s max_return: 0 rows (check OHLCV Parquet exists)", ticker)
                    n_failed += 1
                else:
                    df.to_parquet(mr_path, index=False)
                    counts = df["label"].value_counts().to_dict()
                    logger.info("%s max_return: %d rows %s", ticker, n_rows, counts)
                    n_success += 1
            except Exception as exc:
                logger.warning("%s max_return: FAILED: %s", ticker, exc)
                n_failed += 1

        # --- Risk-adjusted labels ---
        if _should_skip(ra_path, _MIN_ROWS):
            logger.info("%s risk_adjusted: SKIPPED (already %d+ rows)", ticker, _MIN_ROWS)
            n_skipped += 1
        else:
            try:
                # generate_risk_adjusted_labels picks the action that maximizes
                # the Sharpe ratio, accounting for both return and volatility,
                # so the labels are more risk-balanced than max_return's
                df = generate_risk_adjusted_labels(ticker, args.data_dir, args.horizon)
                if not df.empty:
                    df = df[(df["date"].astype(str) >= f"{args.start_year}-01-01") &
                            (df["date"].astype(str) <= f"{args.end_year}-12-31")]
                n_rows = len(df)
                if n_rows == 0:
                    logger.warning("%s risk_adjusted: 0 rows", ticker)
                    n_failed += 1
                else:
                    df.to_parquet(ra_path, index=False)
                    counts = df["label"].value_counts().to_dict()
                    logger.info("%s risk_adjusted: %d rows %s", ticker, n_rows, counts)
                    n_success += 1
            except Exception as exc:
                logger.warning("%s risk_adjusted: FAILED: %s", ticker, exc)
                n_failed += 1

    print("\nReference strategies generated:")
    print(f"  Success: {n_success}, Skipped: {n_skipped}, Failed: {n_failed}")
    print(f"  Output: {output_dir}")

    # Validate: at least 12 Parquets per strategy
    mr_files = list(max_return_dir.glob("*.parquet"))
    ra_files = list(risk_adj_dir.glob("*.parquet"))
    if len(mr_files) < 12 or len(ra_files) < 12:
        print("WARNING: Expected >= 12 Parquets per strategy.")
        print(f"  max_return: {len(mr_files)}, risk_adjusted: {len(ra_files)}")
        print("  Run `make acquire-data-phase10` if OHLCV data is missing.")
        sys.exit(1)

    print(f"  Validation: max_return={len(mr_files)} files, risk_adjusted={len(ra_files)} files OK")


if __name__ == "__main__":
    main()
