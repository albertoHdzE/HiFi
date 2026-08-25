"""Portfolio context for situated agents (Phase 20, DJ-130).

Agents previously answered a decontextualized question — "is this name
attractive?" — with no knowledge of the book they manage: 80 Holds from an
all-cash account is an unanswered entry question, not caution. This module
gives eligible agents the standing situation of the arm they serve:

  - equity / cash / invested share of the book
  - holdings with weight and unrealized P&L
  - days since genesis and phase (DEPLOYMENT vs STEADY)
  - the arm's own return since genesis vs the equal-weight control

Design constraints carried over from the 2026-08-23 audit:

  - **technical is excluded**: its schema promises price-derived information
    only; injecting book state would deepen the context-contamination class
    already found with shared GraphRAG.
  - The block travels through the existing ``extra_memory_prefix`` channel,
    so no agent graph is touched; a native per-agent context field is the
    clean long-term shape and is deliberately deferred.
  - Everything here is descriptive state as-of now; nothing forward-looking
    enters the prompt (point-in-time discipline).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CONTEXT_ELIGIBLE_AGENTS = frozenset(
    {"fundamental", "risk", "macro", "sentiment", "contrarian"}
)

GENESIS_MARKER = "genesis_date.txt"
DEPLOYMENT_MAX_AGE_SESSIONS = 10
DEPLOYMENT_MAX_EXPOSURE = 0.25
_TOP_HOLDINGS_SHOWN = 8


def genesis_date(data_dir: str) -> str | None:
    """Genesis marker for this deployment, 'YYYY-MM-DD', or None."""
    path = Path(data_dir) / "live" / GENESIS_MARKER
    try:
        raw = path.read_text().strip()
        date.fromisoformat(raw)
        return raw
    except (OSError, ValueError):
        return None


def write_book_state(executor, account: str, data_dir: str) -> dict | None:
    """Snapshot the arm's book into data/live/<acct>/book_state.json.

    Called at cycle start so signal generation sees pre-trade state. Best-
    effort: returns None on broker failure — absence simply means agents run
    decontextualized tonight, exactly as before DJ-130.
    """
    try:
        equity = executor.get_portfolio_value()
        cash = executor.get_account_cash()
        positions = executor.get_positions()
    except Exception as exc:
        logger.warning("[%s] book-state capture failed (%s); agents run "
                       "without portfolio context tonight", account, exc)
        return None

    invested = sum(p.market_value for p in positions.values())
    book = {
        "updated": datetime.now().isoformat(),
        "account": account,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "invested": round(invested, 2),
        "exposure": round(invested / equity, 4) if equity > 0 else 0.0,
        "n_positions": len(positions),
        "positions": [
            {
                "ticker": p.ticker,
                "weight": round(p.market_value / equity, 4) if equity > 0 else 0.0,
                "unrealized_pnl_pct": round(
                    p.unrealized_pnl / (p.avg_entry_price * p.qty), 4
                ) if (p.avg_entry_price > 0 and p.qty > 0) else 0.0,
            }
            for p in sorted(positions.values(),
                            key=lambda x: x.market_value, reverse=True)
        ],
    }

    out = Path(data_dir) / "live" / account / "book_state.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(book, indent=2))
    tmp.rename(out)
    return book


def load_book_state(account: str, data_dir: str) -> dict | None:
    path = Path(data_dir) / "live" / account / "book_state.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _days_since_genesis(data_dir: str) -> int | None:
    g = genesis_date(data_dir)
    if g is None:
        return None
    return (date.today() - date.fromisoformat(g)).days


def _return_since_genesis(account: str, data_dir: str) -> float | None:
    """Arm's cumulative return since the first recorded post-genesis row.

    Reads only rows at or after the genesis marker — point-in-time safe by
    construction (rows are appended nightly; nothing future exists yet).
    """
    from hifi.analytics.live_report import _read_jsonl

    g = genesis_date(data_dir)
    if g is None:
        return None
    rows = _read_jsonl(Path(data_dir) / "live" / account / "equity.jsonl")
    rows = [r for r in rows if r.get("decision_date", "") >= g]
    if len(rows) < 1:
        return None
    first, last = rows[0], rows[-1]
    base = float(first.get("equity") or 0)
    cur = float(last.get("equity") or 0)
    if base <= 0:
        return None
    return round(cur / base - 1.0, 4)


def build_portfolio_context(book: dict, account: str, data_dir: str) -> str:
    """Render the book-state block injected into eligible agents' prompts.

    Pure function of the snapshot + on-disk markers; deterministic text so
    runs stay comparable night over night.
    """
    days = _days_since_genesis(data_dir)
    exposure = float(book.get("exposure", 0.0))
    age_ok = days is not None and days <= DEPLOYMENT_MAX_AGE_SESSIONS
    phase = "DEPLOYMENT" if (exposure < DEPLOYMENT_MAX_EXPOSURE and age_ok) else "STEADY"

    lines = [
        "PORTFOLIO CONTEXT (standing situation of the account you advise):",
        f"- Account {book.get('account')}: equity ${book.get('equity', 0):,.2f}, "
        f"cash ${book.get('cash', 0):,.2f} "
        f"({(1 - exposure) * 100:.1f}% undeployed), "
        f"{book.get('n_positions', 0)} open positions.",
    ]
    pos = book.get("positions", [])[:_TOP_HOLDINGS_SHOWN]
    if pos:
        held = ", ".join(
            f"{p['ticker']} {p['weight'] * 100:.1f}% ({p['unrealized_pnl_pct'] * 100:+.1f})"
            for p in pos
        )
        lines.append(f"- Largest holdings: {held}.")
    if days is not None:
        lines.append(f"- Days since genesis: {days}. Phase: {phase}.")
        if phase == "DEPLOYMENT":
            lines.append(
                "- COLD START: most capital is undeployed by protocol design. "
                "A 'Hold' on an unowned name is an entry abstention, not risk "
                "management; judge entries on conviction, not on preservation of "
                "a book that does not exist yet."
            )
        else:
            lines.append(
                "- STEADY STATE: you manage an existing book; weigh exits and "
                "concentration alongside new entries."
            )
    ret = _return_since_genesis(account, data_dir)
    if ret is not None:
        ctrl = _return_since_genesis("C", data_dir)
        line = f"- Arm return since genesis: {ret * 100:+.2f}%."
        if ctrl is not None:
            delta = (ret - ctrl) * 100
            line += (f" Equal-weight control C: {ctrl * 100:+.2f}%"
                     f" (delta {delta:+.2f} pp).")
        lines.append(line)
    else:
        lines.append("- Arm track record since genesis: insufficient data (early record).")
    lines.append(
        "- Interpret your role within THIS situation; do not assume a blank book "
        "or a full one without checking the facts above."
    )
    return "\n".join(lines)


def build_market_block(ticker: str, as_of_date: str, data_dir: str) -> str:
    """Engine-computed market one-liners for THIS ticker (DJ-002 discipline).

    Every number comes from ``hifi.engines.market_summary`` — the LLM never
    computes, and every input's freshness is printed so staleness is visible
    rather than hidden. Absent inputs are reported as unavailable; nothing is
    guessed.
    """
    from hifi.engines import market_summary as ms

    lines = [f"MARKET CONTEXT (engine-computed, point-in-time {as_of_date}):"]

    regime = ms.regime_snapshot(as_of_date, data_dir)
    if regime and regime.get("label"):
        stamps = regime.get("inputs") or {}
        fresh = ", ".join(
            f"{k} through {v}" for k, v in stamps.items() if v
        ) or "input dates unknown"
        lines.append(f"- Market regime: {regime['label']} ({fresh}).")
    else:
        reason = (regime or {}).get("reason") or "classifier unavailable"
        lines.append(f"- Market regime: unavailable ({reason}); do not assume neutral.")

    rel = ms.relative_strength(ticker, None, as_of_date, data_dir)
    if rel:
        lines.append(
            f"- {ticker} {rel['window_sessions']}-session return "
            f"{rel['ticker_return'] * 100:+.1f}% vs median sector peer "
            f"{rel['peer_median'] * 100:+.1f}% "
            f"(relative strength {rel['delta_pp']:+.1f} pp over {rel['n_peers']} peers)."
        )
    else:
        lines.append(f"- Relative strength vs sector peers: unavailable for {ticker}.")

    account = os.environ.get("HIFI_ACTIVE_ACCOUNT")
    book = load_book_state(account, data_dir) if account else None
    if book and book.get("positions"):
        weights = {p["ticker"]: float(p["weight"]) for p in book["positions"]}
        var = ms.book_var_95(weights, as_of_date, data_dir)
        if var and var.get("var_95_1d") is not None:
            lines.append(
                f"- Book risk: 1-day VaR(95%) = {var['var_95_1d'] * 100:.1f}% of equity "
                f"(historical simulation over {var['aligned_sessions']} aligned sessions; "
                f"one-day figure, horizon is NOT {var['window_sessions']} days)."
            )
        else:
            reason = (var or {}).get("reason") or "insufficient holdings history"
            lines.append(f"- Book risk VaR(95%): unavailable ({reason}).")
    return "\n".join(lines)
