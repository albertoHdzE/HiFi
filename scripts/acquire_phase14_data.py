"""
Phase 14 bulk OHLCV + fundamentals acquisition (E2-T2, DJ-090).

Downloads daily OHLCV and quarterly fundamentals for all 100 tickers in
PHASE14_UNIVERSE from yfinance.  Stores as Parquet in:
  data/market/{ticker}/ohlcv.parquet
  data/fundamentals/{ticker}/quarterly.parquet

Provenance metadata is written alongside each file.

Usage:
    uv run python scripts/acquire_phase14_data.py [--data-dir DIR] [--quiet]

Make target: acquire-data-phase14  (internet required, ~30-60 min)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

START_DATE = "2004-01-01"
END_DATE   = "2025-12-31"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Acquire Phase 14 market data")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def acquire_ohlcv(ticker: str, data_dir: Path, quiet: bool = False) -> bool:
    """Download OHLCV for one ticker; return True on success."""
    try:
        import yfinance as yf

        out_dir = data_dir / "market" / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ohlcv.parquet"

        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=True)
        if df.empty:
            logger.warning("%s: empty OHLCV", ticker)
            return False

        # Flatten MultiIndex if present
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] for c in df.columns]

        df.to_parquet(out_path)

        prov = {
            "ticker": ticker, "source": "yfinance",
            "start": START_DATE, "end": END_DATE,
            "fetched_at": datetime.now(UTC).isoformat(),
            "rows": len(df),
        }
        (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
        if not quiet:
            logger.info("%s: %d rows → %s", ticker, len(df), out_path)
        return True
    except Exception as exc:
        logger.error("%s: OHLCV download failed: %s", ticker, exc)
        return False


def acquire_fundamentals(ticker: str, data_dir: Path, quiet: bool = False) -> bool:
    """Download quarterly fundamentals; return True on success."""
    try:
        import pandas as pd
        import yfinance as yf

        out_dir = data_dir / "fundamentals" / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        t = yf.Ticker(ticker)
        frames = []
        for attr in ("quarterly_income_stmt", "quarterly_balance_sheet", "quarterly_cashflow"):
            df = getattr(t, attr, None)
            if df is not None and not df.empty:
                frames.append(df.T)

        if not frames:
            return False

        combined = pd.concat(frames, axis=1)
        combined.index = pd.to_datetime(combined.index)
        combined.to_parquet(out_dir / "quarterly.parquet")
        if not quiet:
            logger.info("%s: fundamentals → %d quarters", ticker, len(combined))
        return True
    except Exception as exc:
        logger.error("%s: fundamentals download failed: %s", ticker, exc)
        return False


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from hifi.data.universe import PHASE14_UNIVERSE
    data_dir = Path(args.data_dir or os.environ.get("HIFI_DATA_DIR", "data"))

    tickers = [e["ticker"] for e in PHASE14_UNIVERSE]
    ok_ohlcv = ok_fund = 0
    for i, ticker in enumerate(tickers, 1):
        if not args.quiet:
            print(f"[{i}/{len(tickers)}] {ticker}", flush=True)
        if acquire_ohlcv(ticker, data_dir, args.quiet):
            ok_ohlcv += 1
        if acquire_fundamentals(ticker, data_dir, args.quiet):
            ok_fund += 1

    print(f"\nacquire_phase14_data: OHLCV={ok_ohlcv}/{len(tickers)} "
          f"fundamentals={ok_fund}/{len(tickers)}")


if __name__ == "__main__":
    main()
