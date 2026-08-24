"""Automatic financial-performance capture for Phase 16 (DJ-114).

Every strategy account's performance evolution must be reconstructable after
the fact with scientific rigor — returns, Sharpe, drawdown (QuantStats-style
tearsheets) AND holdings evolution for attribution. Two durable records, both
written automatically each nightly cycle, no human intervention:

1. equity.jsonl   — one appended row per capture: our point-in-time equity,
   cash, and every position (qty, market_value, avg_entry, unrealized_pnl).
   The holdings time-series for attribution / turnover analysis.

2. portfolio_history.json — Alpaca's authoritative, close-marked, gap-free
   daily equity curve (server-computed; correct even on days we don't run).
   This is the QuantStats-ready series: equity -> daily returns.

Design note: the Alpaca history is the source of truth for the equity curve
(it never has gaps and is marked at close); equity.jsonl adds the per-position
detail Alpaca history omits. Analysis modules consume portfolio_history.json
for performance metrics and equity.jsonl for holdings/attribution.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _account_dir(data_dir: str, account: str) -> Path:
    return Path(data_dir) / "live" / account


def snapshot_account(executor, account: str, data_dir: str,
                     decision_date: str | None = None) -> dict:
    """Append a point-in-time equity + positions row to equity.jsonl.

    Returns the snapshot dict. Never raises on capture-side issues — a failed
    capture must never abort a trading cycle, but is logged loudly.
    """
    try:
        acct = executor.get_account_snapshot()
        positions = executor.get_positions()
    except Exception as exc:  # pragma: no cover - network/broker guard
        logger.error("[%s] portfolio snapshot failed: %s", account, exc)
        return {}

    snap = {
        "timestamp": datetime.now().isoformat(),
        "decision_date": decision_date or datetime.now().strftime("%Y-%m-%d"),
        "account": account,
        "equity": acct["equity"],
        "last_equity": acct["last_equity"],
        "cash": acct["cash"],
        "buying_power": acct["buying_power"],
        "n_positions": len(positions),
        "positions": [
            {
                "ticker": p.ticker,
                "qty": p.qty,
                "market_value": p.market_value,
                "avg_entry_price": p.avg_entry_price,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in sorted(positions.values(), key=lambda x: x.ticker)
        ],
    }

    d = _account_dir(data_dir, account)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "equity.jsonl", "a") as f:
        f.write(json.dumps(snap) + "\n")
    logger.info("[%s] snapshot: equity=$%.2f cash=$%.2f positions=%d",
                account, snap["equity"], snap["cash"], snap["n_positions"])
    return snap


def save_portfolio_history(executor, account: str, data_dir: str) -> int:
    """Overwrite portfolio_history.json with Alpaca's full authoritative curve.

    Returns the number of daily points saved (0 on failure).
    """
    try:
        hist = executor.get_portfolio_history(period="all", timeframe="1D")
    except Exception as exc:  # pragma: no cover - network/broker guard
        logger.error("[%s] portfolio history fetch failed: %s", account, exc)
        return 0

    d = _account_dir(data_dir, account)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "account": account,
        "fetched_at": datetime.now().isoformat(),
        "timeframe": "1D",
        **hist,
    }
    with open(d / "portfolio_history.json", "w") as f:
        json.dump(payload, f)
    n = len(hist.get("equity", []))
    logger.info("[%s] portfolio history saved: %d daily points", account, n)
    return n


def record_account(executor, account: str, data_dir: str,
                   decision_date: str | None = None) -> None:
    """Capture both records for one account (called automatically per cycle)."""
    snapshot_account(executor, account, data_dir, decision_date)
    save_portfolio_history(executor, account, data_dir)


# ---------------------------------------------------------------------------
# Analysis-ready loaders (stable format contract for future analysis module)
# ---------------------------------------------------------------------------


def load_equity_curve(account: str, data_dir: str):
    """Alpaca's authoritative daily equity as a pandas Series indexed by date.

    This is the QuantStats-ready input: ``qs.reports.html(curve.pct_change())``.
    """
    import pandas as pd  # noqa: PLC0415

    path = _account_dir(data_dir, account) / "portfolio_history.json"
    if not path.exists():
        return pd.Series(dtype=float, name="equity")
    h = json.loads(path.read_text())
    idx = pd.to_datetime(h.get("timestamp", []), unit="s")
    return pd.Series(h.get("equity", []), index=idx, name="equity").dropna()


def load_returns(account: str, data_dir: str):
    """Daily returns Series for ``account`` — direct input to QuantStats."""
    return load_equity_curve(account, data_dir).pct_change().dropna().rename("returns")
