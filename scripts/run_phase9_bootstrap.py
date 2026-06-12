"""
Phase 9 historical bootstrap (P9-E5).

Deterministic bootstrap of agent_performance_history.json from 20 quarter-ends
(2018-Q1 through 2022-Q4) across 3 tickers (AAPL, JPM, XOM). No LLM required.

Proxy signal rules — bootstrap heuristics, not LLM output (DJ-041):
  Technical:   RSI < 40  → Buy  (confidence 0.65)
               RSI > 60  → Sell (confidence 0.65)
               else      → Hold (confidence 0.50)
  Risk:        Sharpe_252d > 0.8 → Buy  (confidence 0.65)
               Sharpe_252d < 0.3 → Sell (confidence 0.65)
               else              → Hold (confidence 0.50)
  Fundamental: Hold / confidence=0.50  (uniform prior; LLM not run)
  Macro:       Hold / confidence=0.50  (uniform prior; LLM not run)
  Sentiment:   skipped  (fail-open; no historical RAG corpus)
  Contrarian:  skipped  (second-pass; requires other agent outputs)

Forward-return labeling (DJ-042):
  Load close prices from Phase 1 Parquet files.
  BUY correct  if 60-trading-day forward return > +0.02
  SELL correct if 60-trading-day forward return < -0.02
  HOLD correct within +-0.02
  Records where forward data is unavailable are persisted with
  outcome_correct=None (unlabeled; Phase 10 will back-fill).

Output
------
data/agent_performance_history.json

Usage
-----
    uv run python scripts/run_phase9_bootstrap.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import os
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from hifi.agents.mcp_client import call_tool  # noqa: E402
from hifi.collective.performance_store import update_and_save  # noqa: E402
from hifi.collective.schemas import DecisionRecord  # noqa: E402

_TICKERS = ["AAPL", "JPM", "XOM"]

# 20 quarter-ends: 2018-Q1 through 2022-Q4 (fully covered by Phase 1 Parquet files)
_QUARTER_ENDS = [
    "2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31",
    "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
    "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
    "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
    "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
]

# Bootstrap confidence constants (DJ-041)
_CONF_DIRECTIONAL = 0.65  # Buy or Sell from threshold rule
_CONF_NEUTRAL = 0.50      # Hold or uniform prior


def _proxy_technical(rsi: float | None) -> tuple[str, float]:
    """RSI threshold rule → (decision, confidence)."""
    if rsi is None:
        return "Hold", _CONF_NEUTRAL
    if rsi < 40:
        return "Buy", _CONF_DIRECTIONAL
    if rsi > 60:
        return "Sell", _CONF_DIRECTIONAL
    return "Hold", _CONF_NEUTRAL


def _proxy_risk(sharpe: float | None) -> tuple[str, float]:
    """Sharpe_252d threshold rule → (decision, confidence)."""
    if sharpe is None:
        return "Hold", _CONF_NEUTRAL
    if sharpe > 0.8:
        return "Buy", _CONF_DIRECTIONAL
    if sharpe < 0.3:
        return "Sell", _CONF_DIRECTIONAL
    return "Hold", _CONF_NEUTRAL


def _load_close_prices(ticker: str, data_dir: str) -> dict[date, float]:
    """
    Load adjusted-close (or close) prices from the most-recent OHLCV Parquet.

    Returns {trading_date: price}. Empty dict when no file is found.
    """
    import pandas as pd

    pattern = str(Path(data_dir) / "market" / f"{ticker}_*.parquet")
    files = sorted(_glob.glob(pattern))
    if not files:
        logger.warning("No OHLCV parquet for %s in %s", ticker, data_dir)
        return {}

    df = pd.read_parquet(files[-1])

    # Normalise index: HiFi write_ohlcv uses lowercase "date" column;
    # raw yfinance parquet uses "Date". Handle both.
    if "date" in df.columns:
        df = df.set_index("date")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    # HiFi format: "adjusted_close" / "close"; raw yfinance: "Adj Close" / "Close"
    col = next(
        (c for c in ("adjusted_close", "Adj Close", "close", "Close") if c in df.columns),
        None,
    )
    if col is None:
        logger.warning("No close price column found for %s in %s", ticker, files[-1])
        return {}
    prices = df[col].dropna()
    return {ts.date(): float(v) for ts, v in prices.items()}


def _forward_return(
    prices: dict[date, float],
    analysis_date: str,
    horizon_days: int = 60,
) -> float | None:
    """
    Compute the horizon_days-trading-day forward return from analysis_date.

    Finds the first available trading day on/after analysis_date (t0) and the
    trading day horizon_days sessions later. Returns None when data is
    insufficient (e.g., analysis_date is near the end of the price series).
    """
    t0 = date.fromisoformat(analysis_date)
    sorted_dates = sorted(prices)

    start_candidates = [d for d in sorted_dates if d >= t0]
    if not start_candidates:
        return None
    start_date = start_candidates[0]

    idx = sorted_dates.index(start_date)
    end_idx = idx + horizon_days
    if end_idx >= len(sorted_dates):
        return None  # insufficient forward data

    p0 = prices[start_date]
    p1 = prices[sorted_dates[end_idx]]
    if p0 == 0:
        return None
    return (p1 - p0) / p0


def _label(decision: str, forward_return: float | None) -> bool | None:
    """
    Label a decision as correct/incorrect from the 60-day forward return.

    Returns None when forward_return is None (record will be unlabeled).
    """
    if forward_return is None:
        return None
    if decision == "Buy":
        return forward_return > 0.02
    if decision == "Sell":
        return forward_return < -0.02
    # Hold: correct within +-2%
    return abs(forward_return) <= 0.02


def run_bootstrap(data_dir: str) -> None:
    print("Phase 9 Bootstrap: Historical Performance Seeding")
    print("=" * 60)
    print(f"Quarters: {_QUARTER_ENDS[0]} through {_QUARTER_ENDS[-1]}")
    print(f"Tickers:  {', '.join(_TICKERS)}")
    print(f"Data dir: {data_dir}")
    print()

    records: list[DecisionRecord] = []
    n_mcp_errors = 0

    for ticker in _TICKERS:
        print(f"\n{ticker}")
        prices = _load_close_prices(ticker, data_dir)
        if not prices:
            print(f"  WARNING: No price data — skipping {ticker}")
            continue

        for qe in _QUARTER_ENDS:
            # Technical indicators via MCP
            tech_result = call_tool(
                tool_name="get_technical_indicators",
                params={"ticker": ticker, "date": qe},
                data_dir=data_dir,
            )
            if "error" in tech_result:
                logger.warning(
                    "get_technical_indicators error for %s/%s: %s",
                    ticker, qe, tech_result,
                )
                n_mcp_errors += 1
                rsi = None
            else:
                rsi = tech_result.get("rsi")

            # Risk metrics via MCP
            risk_result = call_tool(
                tool_name="get_risk_metrics",
                params={"ticker": ticker, "date": qe},
                data_dir=data_dir,
            )
            if "error" in risk_result:
                logger.warning(
                    "get_risk_metrics error for %s/%s: %s",
                    ticker, qe, risk_result,
                )
                n_mcp_errors += 1
                sharpe = None
            else:
                sharpe = risk_result.get("sharpe_252d")

            fwd = _forward_return(prices, qe)
            tech_decision, tech_conf = _proxy_technical(rsi)
            risk_decision, risk_conf = _proxy_risk(sharpe)

            records.extend([
                DecisionRecord(
                    ticker=ticker,
                    analysis_date=qe,
                    agent_type="fundamental",
                    decision="Hold",
                    confidence=_CONF_NEUTRAL,
                    outcome_correct=_label("Hold", fwd),
                    forward_return=fwd,
                ),
                DecisionRecord(
                    ticker=ticker,
                    analysis_date=qe,
                    agent_type="macro",
                    decision="Hold",
                    confidence=_CONF_NEUTRAL,
                    outcome_correct=_label("Hold", fwd),
                    forward_return=fwd,
                ),
                DecisionRecord(
                    ticker=ticker,
                    analysis_date=qe,
                    agent_type="technical",
                    decision=tech_decision,
                    confidence=tech_conf,
                    outcome_correct=_label(tech_decision, fwd),
                    forward_return=fwd,
                ),
                DecisionRecord(
                    ticker=ticker,
                    analysis_date=qe,
                    agent_type="risk",
                    decision=risk_decision,
                    confidence=risk_conf,
                    outcome_correct=_label(risk_decision, fwd),
                    forward_return=fwd,
                ),
            ])

            rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
            sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
            fwd_str = f"{fwd:+.3f}" if fwd is not None else "N/A "
            print(
                f"  {qe}  RSI={rsi_str:<6} Sharpe={sharpe_str:<7} "
                f"fwd={fwd_str}  tech={tech_decision}  risk={risk_decision}"
            )

    # Persist to data/agent_performance_history.json
    print(f"\n{'=' * 60}")
    print(f"Total records:  {len(records)}")
    labeled = sum(1 for r in records if r.outcome_correct is not None)
    print(f"Labeled:        {labeled}")
    unlabeled = len(records) - labeled
    print(f"Unlabeled:      {unlabeled}")
    if n_mcp_errors:
        print(f"MCP errors:     {n_mcp_errors}  (see WARNING lines above)")

    history = update_and_save(records, data_dir)

    print("\nWeights after bootstrap:")
    for agent_type, weight in sorted(history.weights.items()):
        print(f"  {agent_type:<15} {weight:.4f}")

    output_path = Path(data_dir) / "agent_performance_history.json"
    print(f"\nSaved to {output_path}")
    print(
        "\nNote: Fundamental and Macro weights reflect only the Hold heuristic. "
        "Phase 10 will replace these with true LLM agent accuracy labels."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 9 historical performance bootstrap. "
            "No LLM required. Reads Phase 1 Parquet files from DATA_DIR."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", str(_ROOT / "data")),
        help="Path to the market/macro data root directory (default: data/).",
    )
    args = parser.parse_args()
    run_bootstrap(args.data_dir)


if __name__ == "__main__":
    main()
