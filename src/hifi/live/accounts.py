"""The four experimental arms: identity, credentials, and per-arm state.

An "account" here is one arm of the ablation, so the mapping from arm letter to
condition is the experiment's design and lives in exactly one place. The
credential suffixes are the opposite kind of fact — a deployment accident of the
order in which the four Alpaca paper accounts were provisioned — and are tried
in sequence for that reason.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from hifi.live import paths

logger = logging.getLogger(__name__)


# DJ-111: three-account live ablation. suffixes -> env var suffixes tried in order.
_ACCOUNTS: dict[str, dict] = {
    "A": {"condition": "parallel", "suffixes": ["_FIRST", "_A", ""],
          "label": "parallel ensemble (champion)"},
    "B": {"condition": "full", "suffixes": ["_SECOND", "_B"],
          "label": "full sequential ensemble (herding contrast)"},
    "C": {"condition": "control", "suffixes": ["_THIRD", "_C"],
          "label": "equal-weight buy-and-hold (null model)"},
    # DJ-113: external deterministic quant strategy (riskbudget calm_exposure).
    "D": {"condition": "riskbudget", "suffixes": ["_FOURTH", "_D"],
          "label": "riskbudget calm_exposure (deterministic quant)"},
}


def _client_order_id(account: str, date: str, ticker: str, side: str) -> str:
    """Deterministic idempotency key for one intended order (DJ-129a).

    A crash at any point of the submit loop leaves orders at the broker with
    these ids; a same-evening re-run recomputes identical keys, and the broker
    refuses the duplicate instead of filling it twice at the open. Charset and
    length are within Alpaca's client_order_id rules.
    """
    return f"hifi{account}-{date}-{side.lower()}-{ticker}"[:48]


def _seed_hwm_from_history(account: str) -> float:
    """Highest equity ever recorded for this account (0.0 if no history).

    Seeds the high-water mark from the existing record so activating the
    drawdown breaker does not silently reset its baseline on the first run
    after the fix: an account that once sat at $110k must carry that peak even
    if tonight's equity is lower.
    """
    import pandas as pd

    path = Path(paths._DATA_DIR) / "live" / account / "equity.jsonl"
    if not path.exists():
        return 0.0
    try:
        df = pd.read_json(path, lines=True)
        if "equity" not in df.columns or df.empty:
            return 0.0
        return float(pd.to_numeric(df["equity"], errors="coerce").max() or 0.0)
    except Exception as exc:
        logger.warning("[%s] Could not seed HWM from history (%s); using 0.0",
                       account, exc)
        return 0.0


def update_hwm(account: str, current_equity: float) -> float:
    """Ratchet the account's high-water mark and persist it atomically (DJ-129b).

    The drawdown breaker compares equity to this mark. Before DJ-129b the
    caller passed today's equity as its own HWM, so (hwm - pv) / hwm was
    identically zero and the pre-registered -15% control could never fire.
    The mark ratchets up only; a falling market must never lower it.
    """
    stored = 0.0
    path = paths._hwm_path(account)
    if path.exists():
        try:
            stored = float(json.loads(path.read_text()).get("hwm", 0.0))
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("[%s] Could not read HWM file (%s); rebuilding from history",
                           account, exc)
    hwm = max(stored, _seed_hwm_from_history(account), current_equity)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"hwm": hwm, "updated": datetime.now().isoformat(),
         "equity_now": current_equity}) + "\n")
    tmp.rename(path)
    return hwm


def already_decided(account: str, date: str) -> bool:
    """True if this account already logged an episode for `date` (DJ-119).

    One decision cycle per account per day is the protocol. A second run on the
    same date re-reads the cached ensemble (agents all skip), so the LLM arms
    reproduce their signals — but the deterministic arms re-derive against the
    *updated* portfolio state and trade again. That is what happened on
    2026-07-28: account D placed 2 orders in the morning and 4 more that
    evening. The check is per-account so a run that died partway can still be
    resumed for the accounts that never completed.
    """
    log_path = paths._decisions_log(account)
    if not log_path.exists():
        return False
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line and json.loads(line).get("decision_date") == date:
                    return True
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[%s] Could not read decision log (%s) — proceeding", account, exc)
    return False


def get_executor(account: str):
    """Build an AlpacaExecutor for the given account (A/B/C).

    Tries each env suffix in order (e.g. ALPACA_API_KEY_FIRST, ALPACA_API_KEY_A,
    ALPACA_API_KEY). Returns None if no credentials found (not yet provisioned).
    """
    from hifi.execution.alpaca_executor import AlpacaExecutor

    api_key = secret = None
    for suffix in _ACCOUNTS[account]["suffixes"]:
        api_key = os.environ.get(f"ALPACA_API_KEY{suffix}")
        secret = os.environ.get(f"ALPACA_SECRET{suffix}")
        if api_key and secret:
            break

    if not api_key or not secret:
        logger.warning("Account %s: no credentials found — skipping", account)
        return None

    ex = AlpacaExecutor(api_key=api_key, secret_key=secret, paper=True)
    ex.connect()
    return ex


def show_status(account: str, executor) -> None:
    acct = executor.client.get_account()
    cfg = _ACCOUNTS[account]
    print(f"\n{'='*64}")
    print(f"Account {account} — {cfg['label']}")
    print(f"{'='*64}")
    print(f"  Equity:       ${float(acct.equity):>14,.2f}")
    print(f"  Cash:         ${float(acct.cash):>14,.2f}")
    print(f"  Daily P&L:    ${float(acct.equity) - float(acct.last_equity):>14,.2f}")

    positions = executor.get_positions()
    if positions:
        print(f"\n  {'Ticker':<8} {'Qty':>6} {'Value':>12} {'P&L':>10}")
        print(f"  {'-'*40}")
        for sym, p in sorted(positions.items()):
            print(f"  {sym:<8} {p.qty:>6.0f} ${p.market_value:>11,.2f} ${p.unrealized_pnl:>9,.2f}")
    else:
        print("  No open positions")

    log_path = paths._decisions_log(account)
    if log_path.exists():
        with open(log_path) as f:
            lines = f.readlines()
        print(f"\n  Decisions logged: {len(lines)}")
        for line in lines[-3:]:
            ep = json.loads(line)
            print(
                f"    {ep['decision_date']}: {ep['n_orders']} orders,"
                f" ${ep['portfolio_value']:,.2f}"
            )
    print()
