"""Phase 16 Live Paper Trading Orchestrator (DJ-098, DJ-099, DJ-111).

Three-account live ablation on Alpaca paper trading (DJ-111):
  Account A: parallel ensemble  (Phase 15 champion, IC=+0.0642, herding=0)
  Account B: full sequential ensemble (herding contrast, IC=+0.0232, herding=0.361)
  Account C: equal-weight buy-and-hold control (no LLM, null model)

All accounts trade the same 98-ticker PHASE14_UNIVERSE, one decision
cycle per day (evening after close; orders fill at next open).

Daily batch pipeline per account:
  1. Update OHLCV data through today (Alpaca bars API)
  2. Circuit breaker check (daily loss 2%, position loss 10%)
  3. Generate signals (ensemble agents for A/B; equal-weight rule for C)
  4. Run MCP pipeline (compose -> risk -> allocate)  [A/B only]
  5. Execute orders on the account's Alpaca paper account
  6. Log episode for future outcome labeling

Usage:
  uv run python scripts/run_phase16_live.py --status                    # all accounts
  uv run python scripts/run_phase16_live.py --update-data               # refresh OHLCV
  uv run python scripts/run_phase16_live.py --account A --dry-run       # signals only
  uv run python scripts/run_phase16_live.py --account all --execute     # full nightly batch
  uv run python scripts/run_phase16_live.py --account A --execute --smoke  # 22-ticker test

Env (.env):
  Account A: ALPACA_API_KEY_FIRST / ALPACA_SECRET_FIRST
  Account B: ALPACA_API_KEY_SECOND / ALPACA_SECRET_SECOND
  Account C: ALPACA_API_KEY_THIRD / ALPACA_SECRET_THIRD
  (also accepts _A/_B/_C suffixes; account A falls back to unsuffixed keys)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# Thread watchdog (DJ-112 backstop). The real fix is the memoised tracer in
# tracing.py; this is defence-in-depth. macOS panics the kernel when a process
# nears ~2048 pthreads (three crashes cost full nights). If thread count ever
# climbs abnormally, abort THIS process cleanly long before it can take down
# the machine — a lost run is recoverable via checkpoint-resume; a kernel
# panic is not.
_THREAD_ABORT = 600   # healthy live run sits under ~30; 600 = clearly leaking


def _start_thread_watchdog(interval_s: int = 30) -> None:
    def _watch() -> None:
        peak = 0
        while True:
            n = threading.active_count()
            peak = max(peak, n)
            if n >= _THREAD_ABORT:
                logger.critical(
                    "THREAD WATCHDOG: %d active threads >= %d limit — aborting "
                    "to prevent kernel panic. Re-run to resume via checkpoint.",
                    n, _THREAD_ABORT,
                )
                os._exit(75)  # EX_TEMPFAIL: transient, safe to retry
            if n > 100:
                logger.warning("THREAD WATCHDOG: elevated thread count %d (peak %d)", n, peak)
            time.sleep(interval_s)

    t = threading.Thread(target=_watch, name="thread-watchdog", daemon=True)
    t.start()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_DIR = str(_ROOT / "data")
_OUTPUT_DIR = str(_ROOT / "data" / "live")
_DB_PATH = str(_ROOT / "data" / "knowledge.lance")

_DAILY_LOSS_LIMIT = 0.02       # 2% portfolio loss -> halt
_POSITION_LOSS_LIMIT = 0.10    # 10% single position -> FLAG only (never halts)
_POSITION_IMPACT_LIMIT = 0.02  # single position costing >2% of equity -> halt

_EDGAR_NAMESPACE = "hifi-dev-sec"
_CONTEXT_NAMESPACE = "hifi-live-context"

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


def _get_tickers(smoke: bool = False) -> list[str]:
    if smoke:
        from hifi.data.smoke_universe import SMOKE_UNIVERSE
        return [row["ticker"] for row in SMOKE_UNIVERSE]
    from hifi.data.universe import PHASE14_UNIVERSE
    return [row["ticker"] for row in PHASE14_UNIVERSE]


def _get_sectors() -> dict[str, str]:
    from hifi.data.universe import PHASE14_UNIVERSE
    return {row["ticker"]: row["sector"] for row in PHASE14_UNIVERSE}


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _account_dir(account: str) -> Path:
    return Path(_OUTPUT_DIR) / account


def _decisions_log(account: str) -> Path:
    return _account_dir(account) / "decisions.jsonl"


def _breaker_log(account: str) -> Path:
    return _account_dir(account) / "circuit_breakers.jsonl"


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
    log_path = _decisions_log(account)
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


# ---------------------------------------------------------------------------
# Broker connection per account
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Step 1: Update market data
# ---------------------------------------------------------------------------


def update_data(tickers: list[str]) -> dict[str, int]:
    from hifi.execution.market_data import update_local_ohlcv

    result = update_local_ohlcv(tickers, market_dir=os.path.join(_DATA_DIR, "market"))
    total_new = sum(result.values())
    logger.info("OHLCV update: %d new bars across %d tickers", total_new, len(tickers))
    return result


def _latest_prices(tickers: list[str]) -> dict[str, float]:
    import pandas as pd

    prices: dict[str, float] = {}
    for ticker in tickers:
        pq = Path(_DATA_DIR) / "market" / ticker / "ohlcv.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
            df.columns = df.columns.str.lower()
            prices[ticker] = float(df["close"].iloc[-1])
    return prices


# ---------------------------------------------------------------------------
# Steps 2-3 (A/B): ensemble agents + aggregate
# ---------------------------------------------------------------------------


def run_ensemble(
    tickers: list[str], date: str, condition: str, account: str, dry_run: bool
) -> None:
    """Run all 6 agents for the given condition, then aggregate."""
    from run_phase15_orchestrator import (
        CANONICAL_ORDER,
        run_agent_mode,
        run_aggregate_mode,
    )

    # Date-partition the ensemble output (HiFi issue #2). The walk-forward
    # _ensemble_path is MONTH-keyed and the aggregate step skips-if-exists
    # (checkpoint-resume); in live trading multiple decision dates share a
    # month, so without this every run after the first-of-month reused the
    # first run's stale ensemble. One dir per date isolates each decision.
    output_dir = str(_account_dir(account) / "walkforward" / date)

    for agent_type in CANONICAL_ORDER:
        run_agent_mode(
            agent_type=agent_type,
            condition=condition,
            dates=[date],
            tickers=tickers,
            data_dir=_DATA_DIR,
            db_path=_DB_PATH,
            dry_run=dry_run,
            quiet=False,
        )

    if not dry_run:
        run_aggregate_mode(
            condition=condition,
            dates=[date],
            tickers=tickers,
            data_dir=_DATA_DIR,
            db_path=_DB_PATH,
            output_dir=output_dir,
            dry_run=False,
            quiet=False,
        )


def load_ensemble_signals(
    tickers: list[str], date: str, condition: str, account: str
) -> list[dict]:
    sectors = _get_sectors()
    year, month, _ = date.split("-")
    signals = []
    for ticker in tickers:
        ens_path = (
            _account_dir(account) / "walkforward" / date
            / condition / year / month / f"{ticker}.json"
        )
        if not ens_path.exists():
            continue
        with open(ens_path) as f:
            ens = json.load(f)
        ed = ens.get("ensemble_decision") or {}
        signals.append({
            "ticker": ticker,
            "decision": ed.get("collective_decision", "Hold"),
            "confidence": ed.get("collective_confidence", 0.5),
            "sector": sectors.get(ticker, "Unknown"),
        })
    return signals


# ---------------------------------------------------------------------------
# Step 4 (A/B): MCP pipeline
# ---------------------------------------------------------------------------


def run_mcp_pipeline(signals: list[dict], tickers: list[str], executor):
    """Run compose -> risk -> allocate and return the portfolio snapshot."""
    import pandas as pd

    from hifi.simulation.pipeline import run_pipeline

    sectors = _get_sectors()
    positions = executor.get_positions()
    portfolio_value = executor.get_portfolio_value()
    prices = _latest_prices(tickers)
    # Allocator contract: current_capital = value of EXISTING holdings
    # (0.0 = all cash / fresh). Passing cash here made every fresh Buy read
    # current_weight=0, trip the 5% rebalance-skip, and drop the order — the
    # bug that kept A/B in cash (HiFi issue #1).
    invested_value = sum(p.market_value for p in positions.values())

    portfolio_state = {
        "portfolio": {
            sym: {
                "weight": p.market_value / portfolio_value if portfolio_value else 0,
                "sector": sectors.get(sym, "Unknown"),
            }
            for sym, p in positions.items()
        },
        "portfolio_value": portfolio_value,
        "hwm_value": portfolio_value,
        "holdings": {sym: int(p.qty) for sym, p in positions.items()},
        "prices": prices,
    }

    constraints = {
        "max_single_stock": 0.05,
        "max_sector": 0.20,
        "min_position": 0.01,
        "capital": portfolio_value,
        "current_capital": invested_value,
    }

    ohlcv = {}
    for ticker in tickers:
        pq = Path(_DATA_DIR) / "market" / ticker / "ohlcv.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
            if df.index.name and df.index.name.lower() == "date":
                df = df.reset_index()
            df.columns = df.columns.str.lower()
            df["date"] = df["date"].astype(str)
            ohlcv[ticker] = df.to_dict("records")

    return run_pipeline(signals, ohlcv, portfolio_state, constraints)


# ---------------------------------------------------------------------------
# Account C: equal-weight buy-and-hold control (no LLM)
# ---------------------------------------------------------------------------


def run_control_strategy(tickers: list[str], executor, dry_run: bool) -> list[dict]:
    """Null model (DJ-111): buy each ticker at 1/N equity once, then hold.

    Only emits Buy orders for tickers with no existing position. No
    rebalancing, no selling — pure buy-and-hold market exposure.
    """
    positions = executor.get_positions()
    equity = executor.get_portfolio_value()
    cash = executor.get_account_cash()
    prices = _latest_prices(tickers)

    # 1% cash buffer: prices move between close (sizing) and open (fill),
    # and rounding accumulates across ~98 orders.
    target_value = equity * 0.99 / len(tickers)
    orders = []
    spend = 0.0

    for ticker in tickers:
        if ticker in positions:
            continue
        price = prices.get(ticker)
        if not price or price <= 0:
            logger.warning("Control: no price for %s, skipping", ticker)
            continue
        # Fractional shares so expensive tickers (LLY, GS, BLK, EQIX > slice)
        # still get their equal weight — but only where Alpaca allows it
        # (e.g. HON is not fractionable: whole shares, skip if price > slice).
        fractionable = dry_run or executor.is_fractionable(ticker)
        if fractionable:
            qty = round(target_value / price, 3)
        else:
            qty = float(int(target_value / price))
            if qty < 1:
                logger.warning("Control: %s not fractionable and price $%.2f > slice, skipping",
                               ticker, price)
                continue
        if qty <= 0:
            continue
        if spend + qty * price > cash:
            logger.warning("Control: out of cash at %s (spent $%.2f of $%.2f)",
                           ticker, spend, cash)
            break

        if dry_run:
            logger.info("[DRY-RUN] Control would buy %s x%d (~$%.2f)", ticker, qty, qty * price)
            orders.append({"ticker": ticker, "side": "buy", "qty": qty, "status": "dry_run"})
        else:
            result = executor.place_market_order(ticker, qty, "buy")
            orders.append({
                "ticker": ticker, "side": "buy", "qty": qty,
                "status": result.status, "order_id": result.order_id,
            })
        spend += qty * price

    logger.info("Control strategy: %d orders, ~$%.2f notional", len(orders), spend)
    return orders


# ---------------------------------------------------------------------------
# Step 5 (A/B): execute pipeline orders
# ---------------------------------------------------------------------------


def execute_orders(snapshot, executor, dry_run: bool) -> list[dict]:
    orders_src = getattr(snapshot, "orders", None) or []
    if not orders_src:
        logger.info("No orders to execute")
        return []

    results = []
    for order in orders_src:
        ticker = order.get("ticker", order.get("symbol"))
        side = order.get("action", order.get("side", "buy")).lower()
        # allocate_capital emits "quantity"; accept "shares"/"qty" too (HiFi issue #1).
        qty = order.get("quantity", order.get("shares", order.get("qty", 0)))
        if qty <= 0:
            continue

        if dry_run:
            logger.info("[DRY-RUN] Would %s %s x%d", side, ticker, qty)
            results.append({"ticker": ticker, "side": side, "qty": qty, "status": "dry_run"})
        else:
            result = executor.place_market_order(ticker, qty, side)
            results.append({
                "ticker": ticker, "side": side, "qty": qty,
                "status": result.status, "order_id": result.order_id,
                "filled_price": result.filled_avg_price,
            })
    return results


# ---------------------------------------------------------------------------
# Step 6: episode logging
# ---------------------------------------------------------------------------


def log_episode(account: str, date: str, condition: str,
                signals: list[dict], orders: list[dict], portfolio_value: float,
                strategy_meta: dict | None = None) -> None:
    log_path = _decisions_log(account)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    episode = {
        "timestamp": datetime.now().isoformat(),
        "decision_date": date,
        "account": account,
        "condition": condition,
        "signals": signals,
        "n_orders": len(orders),
        "orders": orders,
        "portfolio_value": portfolio_value,
    }
    # Provider/version attribution for external strategies (DJ-113) — required
    # for scientific traceability (which version produced which orders).
    if strategy_meta is not None:
        episode["strategy_meta"] = strategy_meta
    with open(log_path, "a") as f:
        f.write(json.dumps(episode) + "\n")
    logger.info("[%s] Episode logged: %d signals, %d orders, $%.2f portfolio",
                account, len(signals), len(orders), portfolio_value)


# ---------------------------------------------------------------------------
# Circuit breakers
# ---------------------------------------------------------------------------


def check_circuit_breakers(account: str, executor) -> bool:
    """Returns True if trading should HALT for this account.

    Position-loss scaling (DJ-119). A raw "any position down >10% halts the
    account" rule does not survive contact with a wide book: with N names the
    probability that at least one is down >10% on a given night approaches 1,
    so it halted the equal-weight control (C) on every run from 2026-07-22 to
    2026-07-28 while the concentrated arms kept trading — a silent confound in
    the ablation, not a risk control.

    The halt criterion is therefore the position's *impact on the portfolio*,
    which self-scales with book width:

        impact = |pnl_pct| * (market_value / equity)

    and requires BOTH a materially adverse move (>_POSITION_LOSS_LIMIT) and a
    material cost to the book (>_POSITION_IMPACT_LIMIT). For an equal-weight
    N-name book the effective per-position threshold is _POSITION_IMPACT_LIMIT
    * N, so it relaxes as the book widens; for a fully concentrated book the
    weight is 1 and it collapses to the portfolio daily limit. The 10% breach
    is still recorded as action="flag" — the observation is kept for the
    science record, it just no longer stops the arm.
    """
    try:
        acct = executor.client.get_account()
        equity = float(acct.equity)
        last_equity = float(acct.last_equity)

        if last_equity > 0:
            daily_change = (equity - last_equity) / last_equity
            if daily_change < -_DAILY_LOSS_LIMIT:
                _log_circuit_breaker(account, "daily_loss", daily_change, equity,
                                     action="halt")
                logger.error("[%s] CIRCUIT BREAKER: daily loss %.2f%% exceeds %.0f%% limit",
                             account, daily_change * 100, _DAILY_LOSS_LIMIT * 100)
                return True

        halt = False
        for sym, pos in sorted(executor.get_positions().items()):
            if pos.avg_entry_price <= 0 or pos.qty <= 0:
                continue
            pnl_pct = pos.unrealized_pnl / (pos.avg_entry_price * pos.qty)
            if pnl_pct >= -_POSITION_LOSS_LIMIT:
                continue

            weight = (pos.market_value / equity) if equity > 0 else 0.0
            impact = abs(pnl_pct) * weight
            if impact > _POSITION_IMPACT_LIMIT:
                _log_circuit_breaker(account, "position_loss", pnl_pct, equity,
                                     ticker=sym, action="halt",
                                     weight=weight, impact=impact)
                logger.error("[%s] CIRCUIT BREAKER: %s loss %.2f%% at weight %.2f%% "
                             "costs %.2f%% of equity (limit %.0f%%)",
                             account, sym, pnl_pct * 100, weight * 100,
                             impact * 100, _POSITION_IMPACT_LIMIT * 100)
                halt = True
            else:
                _log_circuit_breaker(account, "position_loss", pnl_pct, equity,
                                     ticker=sym, action="flag",
                                     weight=weight, impact=impact)
                logger.warning("[%s] FLAG: %s loss %.2f%% at weight %.2f%% "
                               "costs %.2f%% of equity — below halt limit, trading continues",
                               account, sym, pnl_pct * 100, weight * 100, impact * 100)
        return halt
    except Exception as exc:
        logger.error("[%s] Circuit breaker check failed: %s", account, exc)

    return False


def _log_circuit_breaker(account: str, trigger: str, value: float,
                         equity: float, ticker: str = "", action: str = "halt",
                         weight: float | None = None,
                         impact: float | None = None) -> None:
    log_path = _breaker_log(account)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "account": account,
        "trigger": trigger,
        "value": round(value, 6),
        "equity": round(equity, 2),
        "ticker": ticker,
        # Rows written before DJ-119 have no "action" key; they were all halts.
        "action": action,
    }
    if weight is not None:
        entry["weight"] = round(weight, 6)
    if impact is not None:
        entry["impact"] = round(impact, 6)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# --status mode
# ---------------------------------------------------------------------------


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

    log_path = _decisions_log(account)
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


# ---------------------------------------------------------------------------
# Per-account daily cycle
# ---------------------------------------------------------------------------


def run_account_cycle(account: str, tickers: list[str], date: str,
                      dry_run: bool, execute: bool, force: bool = False) -> None:
    condition = _ACCOUNTS[account]["condition"]
    is_dry = dry_run or not execute

    if not is_dry and already_decided(account, date):
        if force:
            logger.warning("[%s] Already decided for %s — --force given, running again. "
                           "Annotate this date as a protocol deviation.", account, date)
        else:
            logger.info("[%s] Already decided for %s — skipping (use --force to override)",
                        account, date)
            return

    executor = get_executor(account)
    if executor is None:
        return

    logger.info("[%s] Daily cycle: condition=%s date=%s tickers=%d dry_run=%s",
                account, condition, date, len(tickers), is_dry)

    if check_circuit_breakers(account, executor):
        logger.error("[%s] HALTED: circuit breaker triggered. No orders.", account)
        return

    strategy_meta: dict | None = None

    if condition == "control":
        orders = run_control_strategy(tickers, executor, dry_run=is_dry)
        signals = []
    elif condition == "riskbudget":
        # External deterministic quant provider (DJ-113). as_of_date is the
        # last completed trading day; the store already holds it.
        from hifi.execution.riskbudget_strategy import get_riskbudget_signals
        payload = get_riskbudget_signals(tickers, date, _DATA_DIR, sectors=_get_sectors())
        signals = payload.get("signals", [])
        strategy_meta = {
            "provider": "riskbudget",
            "strategy": payload.get("strategy"),
            "strategy_version": payload.get("strategy_version"),
            "call_id": payload.get("call_id"),
            "skipped": payload.get("skipped", []),
        }
        if not signals:
            logger.warning("[%s] riskbudget returned no signals for %s — skipping", account, date)
            return
        if dry_run:
            from collections import Counter
            c = Counter(s["decision"] for s in signals)
            logger.info("[%s] Dry-run: riskbudget %s -> %s, %d skipped",
                        account, dict(c), payload.get("call_id"), len(strategy_meta["skipped"]))
            return
        snapshot = run_mcp_pipeline(signals, tickers, executor)
        orders = execute_orders(snapshot, executor, dry_run=is_dry)
    else:
        run_ensemble(tickers, date, condition, account, dry_run=dry_run)
        if dry_run:
            logger.info("[%s] Dry-run: schedule printed, stopping before pipeline", account)
            return
        signals = load_ensemble_signals(tickers, date, condition, account)
        if not signals:
            logger.warning("[%s] No ensemble signals for %s — skipping pipeline", account, date)
            return
        snapshot = run_mcp_pipeline(signals, tickers, executor)
        orders = execute_orders(snapshot, executor, dry_run=is_dry)

    log_episode(account, date, condition, signals, orders, executor.get_portfolio_value(),
                strategy_meta=strategy_meta)
    # Automatic financial-performance capture (DJ-114): equity + positions row
    # and Alpaca's authoritative equity curve. No human intervention.
    from hifi.execution.portfolio_recorder import record_account
    record_account(executor, account, _DATA_DIR, decision_date=date)
    show_status(account, executor)
    executor.disconnect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 16 Live Paper Trading (3-account ablation, DJ-111)"
    )
    parser.add_argument("--account", type=str, default="A",
                        choices=["A", "B", "C", "D", "all"], help="Which account(s) to run")
    parser.add_argument("--execute", action="store_true", help="Place real paper orders")
    parser.add_argument("--dry-run", action="store_true", help="Print orders without executing")
    parser.add_argument("--status", action="store_true",
                        help="Show portfolio status for all provisioned accounts")
    parser.add_argument("--update-data", action="store_true", help="Only refresh OHLCV data")
    parser.add_argument("--snapshot", action="store_true",
                        help="Capture equity/positions/history for all accounts (no trading)")
    parser.add_argument("--smoke", action="store_true",
                        help="Use 22-ticker smoke universe instead of 98")
    parser.add_argument("--date", type=str, default=None,
                        help="Override decision date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true",
                        help="Run even if this account already decided for the date "
                             "(protocol deviation — annotate the date)")
    args = parser.parse_args()

    # Only guard long inference runs; --status/--update-data are short.
    if args.execute or args.dry_run:
        _start_thread_watchdog()

    tickers = _get_tickers(smoke=args.smoke)
    date = args.date or _today_str()

    if args.status:
        for account in _ACCOUNTS:
            executor = get_executor(account)
            if executor is not None:
                show_status(account, executor)
                executor.disconnect()
        return

    if args.snapshot:
        from hifi.execution.portfolio_recorder import record_account
        for account in _ACCOUNTS:
            executor = get_executor(account)
            if executor is not None:
                record_account(executor, account, _DATA_DIR, decision_date=date)
                executor.disconnect()
        return

    if args.update_data:
        update_data(tickers)
        return

    logger.info("Step 1: Updating OHLCV data for %d tickers...", len(tickers))
    update_data(tickers)

    accounts = list(_ACCOUNTS) if args.account == "all" else [args.account]
    failed = []
    for account in accounts:
        # Isolate each account: a network failure (or any error) on one arm must
        # not abort the others (DJ-117). Retries live in the executor; this is the
        # last-resort guard so a partial outage still trades the reachable arms.
        try:
            run_account_cycle(account, tickers, date, dry_run=args.dry_run,
                              execute=args.execute, force=args.force)
        except Exception as exc:
            failed.append(account)
            logger.error("[%s] cycle FAILED (%s); continuing with remaining accounts", account, exc)

    if failed:
        logger.warning("Nightly batch done with FAILURES: failed=%s ok=%s date=%s",
                       ",".join(failed), ",".join(a for a in accounts if a not in failed), date)
    else:
        logger.info("Nightly batch complete: accounts=%s date=%s", ",".join(accounts), date)


if __name__ == "__main__":
    main()
