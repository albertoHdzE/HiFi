#!/usr/bin/env python
"""Bring the local data stores up to date. All logic lives in ``hifi.data.refresh``.

Replaces scripts/refresh_fundamentals.py and scripts/refresh_macro.py, which
were the same script twice with different sources. Everything merges rather than
overwrites: union the periods, fresh values win on overlap, existing history is
never dropped.

    uv run python scripts/refresh_data.py --all
    uv run python scripts/refresh_data.py --fundamentals --tickers AAPL,MSFT
    uv run python scripts/refresh_data.py --macro --quiet
    uv run python scripts/refresh_data.py --check          # quality only, no fetch

--ohlcv is not here on purpose: the nightly cycle refreshes bars itself
(`make live-update-data`), and a second writer for the same store is how two
sources of truth start. Use --check to score what the cycle wrote.

Requires FRED_API_KEY (read from .env if present) for --macro.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hifi.data.refresh import (  # noqa: E402
    SERIES,
    _load_env,
    check_ohlcv_quality,
    refresh_series,
    refresh_ticker,
)

logger = logging.getLogger("refresh_data")


def _tickers(arg: str | None) -> list[str]:
    if arg:
        return [t.strip().upper() for t in arg.split(",") if t.strip()]
    from hifi.data.universe import PHASE14_UNIVERSE

    return [e["ticker"] for e in PHASE14_UNIVERSE]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fundamentals", action="store_true",
                    help="Quarterly statements from yfinance")
    ap.add_argument("--macro", action="store_true", help="FRED series")
    ap.add_argument("--check", action="store_true",
                    help="Score OHLCV completeness; fetches nothing")
    ap.add_argument("--all", action="store_true",
                    help="Equivalent to --fundamentals --macro --check")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tickers", help="Comma-separated subset; default is the live universe")
    ap.add_argument("--quiet", action="store_true", help="Summary only, no per-item lines")
    args = ap.parse_args()

    if args.all:
        args.fundamentals = args.macro = args.check = True
    if not (args.fundamentals or args.macro or args.check):
        ap.error("pick at least one of --fundamentals, --macro, --check, --all")

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(message)s")
    data_dir = Path(args.data_dir)
    failed = False

    if args.fundamentals:
        tickers = _tickers(args.tickers)
        reports = [refresh_ticker(t, data_dir, quiet=args.quiet) for t in tickers]
        ok = [r for r in reports if r["status"] == "ok"]
        bad = [r for r in reports if r["status"] != "ok"]
        print(f"\nfundamentals: refreshed {len(ok)}/{len(tickers)} tickers")
        print(f"  gained at least one new quarter: {sum(1 for r in ok if r['added'])}")
        print(f"  had local-only history preserved: "
              f"{sum(1 for r in ok if r['preserved_only_locally'])}")
        if bad:
            print(f"  FAILED: {[(r['ticker'], r['status']) for r in bad]}")
            failed = True

    if args.macro:
        _load_env(_ROOT)
        if not os.environ.get("FRED_API_KEY"):
            logger.error("FRED_API_KEY not set and not found in .env")
            return 1
        import fredapi

        fred = fredapi.Fred(api_key=os.environ["FRED_API_KEY"])
        reports = [refresh_series(s, data_dir, fred, quiet=args.quiet) for s in SERIES]
        bad = [r for r in reports if r["status"] != "ok"]
        print(f"\nmacro: refreshed {len(reports) - len(bad)}/{len(reports)} series")
        for r in reports:
            if r["status"] == "ok" and r["was"] != r["now"]:
                print(f"  {r['series']:18} {r['was']} -> {r['now']}")
        if bad:
            print(f"  FAILED: {[(r['series'], r['status']) for r in bad]}")
            failed = True

    if args.check:
        tickers = _tickers(args.tickers)
        poor = check_ohlcv_quality(tickers, data_dir, quiet=args.quiet)
        print(f"\nOHLCV quality: {len(tickers) - len(poor)}/{len(tickers)} tickers clean")
        for r in poor:
            c = r.get("completeness")
            print(f"  {r['ticker']:6} completeness="
                  f"{'n/a' if c is None else f'{c:.1%}'} {r.get('error', '')}")
        if poor:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
