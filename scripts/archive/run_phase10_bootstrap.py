"""
Phase 10 historical bootstrap — 15 tickers (P10-E2-T2, DJ-051).

Extends the Phase 9 bootstrap from 3 to 15 tickers using the same RSI/Sharpe
proxy rules (DJ-041). Labels records at both 60-day and 20-day horizons (DJ-052).

Default behaviour (--reset): replaces the existing agent_performance_history.json
entirely. The Phase 9 3-ticker bootstrap is superseded by this 15-ticker version.
Use --extend to append to the existing history instead.

Usage: uv run python scripts/run_phase10_bootstrap.py [--data-dir DIR] [--reset|--extend]
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from hifi.agents.mcp_client import call_tool  # noqa: E402
from hifi.collective.performance_store import (  # noqa: E402
    compute_weights,
    load_history,
    save_history,
)
from hifi.collective.schemas import AgentPerformanceHistory, DecisionRecord  # noqa: E402

_TICKERS = [
    "AAPL", "JPM", "XOM",             # Phase 1 (existing)
    "MSFT", "NVDA", "GOOGL",          # Technology
    "BAC", "GS",                       # Finance
    "CVX",                             # Energy
    "JNJ", "UNH",                      # Healthcare
    "AMZN", "WMT",                     # Consumer
    "CAT",                             # Industrial
    "NEE",                             # Utilities
]

_QUARTER_ENDS = [
    "2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31",
    "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
    "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
    "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
    "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
]

_CONF_DIRECTIONAL = 0.65
_CONF_NEUTRAL = 0.50
_LABEL_THRESHOLD = 0.02


def _load_prices(ticker: str, data_dir: str) -> dict:
    import pandas as pd
    pattern = str(Path(data_dir) / "market" / f"{ticker}_*.parquet")
    files = sorted(_glob.glob(pattern))
    if not files:
        return {}
    df = pd.read_parquet(files[-1])
    if "date" in df.columns:
        df = df.set_index("date")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    col = next(
        (c for c in ("adjusted_close", "Adj Close", "close", "Close") if c in df.columns),
        None,
    )
    if col is None:
        return {}
    prices = df[col].dropna()
    return {ts.date(): float(v) for ts, v in prices.items()}


def _forward_return(prices: dict, analysis_date: str, horizon: int) -> float | None:
    from datetime import date as _date
    t0 = _date.fromisoformat(analysis_date)
    sorted_dates = sorted(prices)
    starts = [d for d in sorted_dates if d >= t0]
    if not starts:
        return None
    t0_actual = starts[0]
    idx = sorted_dates.index(t0_actual)
    end_idx = idx + horizon
    if end_idx >= len(sorted_dates):
        return None
    p0 = prices[t0_actual]
    if p0 == 0:
        return None
    return (prices[sorted_dates[end_idx]] - p0) / p0


def _label(decision: str, fwd: float | None) -> bool | None:
    if fwd is None:
        return None
    if decision == "Buy":
        return fwd > _LABEL_THRESHOLD
    if decision == "Sell":
        return fwd < -_LABEL_THRESHOLD
    return abs(fwd) <= _LABEL_THRESHOLD


def _proxy_technical(rsi: float | None) -> tuple[str, float]:
    if rsi is None:
        return "Hold", _CONF_NEUTRAL
    if rsi < 40:
        return "Buy", _CONF_DIRECTIONAL
    if rsi > 60:
        return "Sell", _CONF_DIRECTIONAL
    return "Hold", _CONF_NEUTRAL


def _proxy_risk(sharpe: float | None) -> tuple[str, float]:
    if sharpe is None:
        return "Hold", _CONF_NEUTRAL
    if sharpe > 0.8:
        return "Buy", _CONF_DIRECTIONAL
    if sharpe < 0.3:
        return "Sell", _CONF_DIRECTIONAL
    return "Hold", _CONF_NEUTRAL


def run_bootstrap(data_dir: str, reset: bool = True) -> None:
    print("Phase 10 Bootstrap: 15-Ticker Performance History")
    print("=" * 60)
    print(f"Tickers:  {', '.join(_TICKERS)}")
    print(f"Quarters: {_QUARTER_ENDS[0]} through {_QUARTER_ENDS[-1]}")
    print(f"Mode:     {'RESET (replaces existing history)' if reset else 'EXTEND (append)'}")
    print("Horizons: 60d (primary) + 20d (secondary, DJ-052)")
    print()

    records: list[DecisionRecord] = []
    n_mcp_errors = 0

    for ticker in _TICKERS:
        print(f"\n{ticker}")
        prices = _load_prices(ticker, data_dir)
        if not prices:
            print(f"  WARNING: No price data — skipping {ticker}")
            continue

        for qe in _QUARTER_ENDS:
            tech_result = call_tool(
                "get_technical_indicators", {"ticker": ticker, "date": qe}, data_dir
            )
            rsi = None if "error" in tech_result else tech_result.get("rsi")
            if "error" in tech_result:
                n_mcp_errors += 1

            risk_result = call_tool("get_risk_metrics", {"ticker": ticker, "date": qe}, data_dir)
            sharpe = None if "error" in risk_result else risk_result.get("sharpe_252d")
            if "error" in risk_result:
                n_mcp_errors += 1

            tech_dec, tech_conf = _proxy_technical(rsi)
            risk_dec, risk_conf = _proxy_risk(sharpe)

            for horizon in (60, 20):
                fwd = _forward_return(prices, qe, horizon)
                records.extend([
                    DecisionRecord(ticker=ticker, analysis_date=qe, agent_type="fundamental",
                                   decision="Hold", confidence=_CONF_NEUTRAL,
                                   outcome_correct=_label("Hold", fwd), forward_return=fwd,
                                   horizon_days=horizon),
                    DecisionRecord(ticker=ticker, analysis_date=qe, agent_type="macro",
                                   decision="Hold", confidence=_CONF_NEUTRAL,
                                   outcome_correct=_label("Hold", fwd), forward_return=fwd,
                                   horizon_days=horizon),
                    DecisionRecord(ticker=ticker, analysis_date=qe, agent_type="technical",
                                   decision=tech_dec, confidence=tech_conf,
                                   outcome_correct=_label(tech_dec, fwd), forward_return=fwd,
                                   horizon_days=horizon),
                    DecisionRecord(ticker=ticker, analysis_date=qe, agent_type="risk",
                                   decision=risk_dec, confidence=risk_conf,
                                   outcome_correct=_label(risk_dec, fwd), forward_return=fwd,
                                   horizon_days=horizon),
                ])

    from datetime import UTC, datetime
    if reset:
        base_records = records
    else:
        existing = load_history(data_dir)
        base_records = existing.records + records

    weights = compute_weights(base_records)
    history = AgentPerformanceHistory(
        records=base_records,
        weights=weights,
        last_updated=datetime.now(tz=UTC).isoformat(),
        n_labeled=0,
    )
    save_history(history, data_dir)

    labeled = sum(1 for r in base_records if r.outcome_correct is not None)
    print(f"\n{'=' * 60}")
    print(f"Total records:  {len(base_records)}")
    print(f"Labeled:        {labeled}")
    if n_mcp_errors:
        print(f"MCP errors:     {n_mcp_errors}")
    print("\nWeights after bootstrap:")
    for at, w in sorted(weights.items()):
        print(f"  {at:<15} {w:.4f}")
    print(f"\nSaved: {Path(data_dir) / 'agent_performance_history.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10 bootstrap — 15 tickers.")
    parser.add_argument("--data-dir", default=os.environ.get("HIFI_DATA_DIR", str(_ROOT / "data")))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reset", action="store_true", default=True,
                       help="Replace existing history (default).")
    group.add_argument("--extend", action="store_true", default=False,
                       help="Append to existing history.")
    args = parser.parse_args()
    run_bootstrap(args.data_dir, reset=not args.extend)


if __name__ == "__main__":
    main()
