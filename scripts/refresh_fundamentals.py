#!/usr/bin/env python
"""Refresh quarterly fundamentals, merging rather than overwriting.

Why not just re-run acquire_phase14_data.py
-------------------------------------------
``acquire_fundamentals`` writes ``combined.to_parquet(...)``, replacing the file
with whatever yfinance returns today. yfinance serves only the most recent five
to seven quarters, so a plain re-run buys the newest quarter at the cost of
deleting the oldest ones. That is silent history loss in the exact data the TTM
ratios and any walk-forward depend on.

This script unions the periods instead: existing rows are kept, new rows are
added, and where both sources carry a period the fresh value wins (restatements
are real and should propagate). It reports what changed per ticker so the
refresh is auditable rather than assumed.

Context (DJ-133a): 91 of 97 tickers were missing a quarter that had already
been filed with the SEC -- everyone sat at 2026-03-31 while the June quarter
had been public since late July. The point-in-time gate meant this was safe
rather than lookahead, but it made every agent read a quarter-old book.

Usage
-----
    uv run python scripts/refresh_fundamentals.py [--quiet] [--tickers AAPL,MSFT]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("refresh_fundamentals")

_STATEMENTS = ("quarterly_income_stmt", "quarterly_balance_sheet", "quarterly_cashflow")


def _fetch(ticker: str):
    """Fresh quarterly statements from yfinance, period-indexed, or None."""
    import pandas as pd
    import yfinance as yf

    t = yf.Ticker(ticker)
    frames = []
    for attr in _STATEMENTS:
        df = getattr(t, attr, None)
        if df is not None and not df.empty:
            frames.append(df.T)
    if not frames:
        return None
    # sort=False is explicit: the frames share a period index and we sort once
    # below, so letting pandas sort per-concat only costs work and a warning.
    combined = pd.concat(frames, axis=1, sort=False)
    combined.index = pd.to_datetime(combined.index)
    # Duplicate column labels appear when two statements share a line item;
    # keeping the first occurrence avoids a reindex explosion on merge.
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined.sort_index()


def refresh_ticker(ticker: str, data_dir: Path, quiet: bool = False) -> dict:
    """Merge fresh statements into the cached parquet. Returns a change report."""
    import pandas as pd

    out_dir = data_dir / "fundamentals" / ticker
    path = out_dir / "quarterly.parquet"

    existing = None
    if path.exists():
        try:
            existing = pd.read_parquet(path)
        except Exception as exc:
            logger.error("%s: existing parquet unreadable (%s); refusing to clobber", ticker, exc)
            return {"ticker": ticker, "status": "unreadable"}

    try:
        fresh = _fetch(ticker)
    except Exception as exc:
        logger.error("%s: fetch failed: %s", ticker, exc)
        return {"ticker": ticker, "status": "fetch_failed"}

    if fresh is None or fresh.empty:
        logger.warning("%s: yfinance returned no statements", ticker)
        return {"ticker": ticker, "status": "empty_response"}

    if existing is None or existing.empty:
        merged, added, kept = fresh, list(fresh.index), []
    else:
        added = [p for p in fresh.index if p not in set(existing.index)]
        kept = [p for p in existing.index if p not in set(fresh.index)]
        # Fresh wins on overlap so restatements propagate; union of columns so a
        # newly reported line item is not dropped.
        merged = fresh.combine_first(existing)
        merged = merged.reindex(sorted(set(existing.index) | set(fresh.index)))

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.sort_index().to_parquet(path)

    report = {
        "ticker": ticker,
        "status": "ok",
        "periods_before": 0 if existing is None else len(existing),
        "periods_after": len(merged),
        "added": [str(p.date()) for p in added],
        "preserved_only_locally": [str(p.date()) for p in kept],
        "latest": str(merged.index.max().date()),
    }
    if not quiet:
        logger.info(
            "%-6s %d -> %d quarters, latest %s, added %s%s",
            ticker, report["periods_before"], report["periods_after"],
            report["latest"], report["added"] or "none",
            f", kept {len(kept)} local-only" if kept else "",
        )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tickers", help="Comma-separated subset; default is the live universe")
    ap.add_argument("--quiet", action="store_true", help="Summary only, no per-ticker lines")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        from hifi.data.universe import PHASE14_UNIVERSE

        tickers = [e["ticker"] for e in PHASE14_UNIVERSE]

    data_dir = Path(args.data_dir)
    reports = [refresh_ticker(t, data_dir, quiet=args.quiet) for t in tickers]

    ok = [r for r in reports if r["status"] == "ok"]
    failed = [r for r in reports if r["status"] != "ok"]
    gained = [r for r in ok if r["added"]]
    lost_risk = [r for r in ok if r["preserved_only_locally"]]

    print(f"\nrefreshed {len(ok)}/{len(tickers)} tickers")
    print(f"  gained at least one new quarter: {len(gained)}")
    print(f"  had local-only history preserved: {len(lost_risk)}")
    if failed:
        print(f"  FAILED: {[(r['ticker'], r['status']) for r in failed]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
