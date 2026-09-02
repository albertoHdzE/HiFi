"""One arm, one evening: signals -> portfolio -> orders -> record.

``run_account_cycle`` is the spine of the experiment. Its ordering is not
incidental and several steps sit where they do because of a specific failure:

* the high-water mark is ratcheted *before* anything trades (DJ-129b), or the
  drawdown breaker compares equity against itself and can never fire;
* the breakers run again immediately before submission (DJ-129c), because hours
  of LLM inference separate the first check from the first order;
* the episode is logged in a ``try`` whose failure re-raises loudly (DJ-123),
  because an arm that traded without leaving a decision record moves the equity
  curve with nothing to attribute it to — worse than a clean failure.
"""

from __future__ import annotations

import collections
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from hifi.live import accounts, ensemble, guards, market, paths, strategies

logger = logging.getLogger(__name__)


def run_mcp_pipeline(signals: list[dict], tickers: list[str], executor,
                     hwm_value: float | None = None):
    """Run compose -> risk -> allocate and return the portfolio snapshot.

    ``hwm_value`` is the persisted high-water mark (DJ-129b). Passing it makes
    the -15% drawdown breaker live: with the historical bug of passing today's
    equity, ``(hwm - pv) / hwm`` was identically zero and the control could
    never trip. ``None`` (dry-run) preserves the old fallback behavior.
    """
    import pandas as pd

    from hifi.simulation.pipeline import run_pipeline

    sectors = paths._get_sectors()
    positions = executor.get_positions()
    portfolio_value = executor.get_portfolio_value()
    prices = market._latest_prices(tickers)
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
        "hwm_value": hwm_value if (hwm_value and hwm_value > 0) else portfolio_value,
        # Fractional holdings preserved (DJ-121): int() truncation made a
        # 3.9-share position read as 3 and manufactured phantom rebalances.
        "holdings": {sym: float(p.qty) for sym, p in positions.items()},
        "prices": prices,
    }

    # compose_portfolio spreads ~100% of capital across the Buy names alone,
    # while Hold positions stay on the book — so targets can sum past 100% and
    # exceed buying power. Passing cash lets the allocator scale buys to fit
    # rather than emitting orders the broker will reject (DJ-121).
    try:
        available_cash = executor.get_account_cash()
    except Exception as exc:
        logger.warning("Could not read account cash (%s); falling back to equity - invested", exc)
        available_cash = max(0.0, portfolio_value - invested_value)

    # Single control point (DJ-122): limits are derived from how many names are
    # actually actionable today, not restated as absolutes at each call site.
    # Hardcoded 5%/20%/1% caps stranded capital on a narrow book (8 Buys in one
    # sector invested 20% of the account) while being inert across 98 names,
    # and they taxed a diversified arm more than a concentrated one — the same
    # invariance failure as the old circuit breaker (DJ-119).
    from hifi.portfolio import PortfolioPolicy

    buys = [s for s in signals if s.get("decision") == "Buy"]
    sector_counts = collections.Counter(
        sectors.get(s["ticker"], "Unknown") for s in buys
    )
    policy = PortfolioPolicy(n_candidates=len(buys))
    constraints = policy.as_constraints(
        capital=portfolio_value,
        current_capital=invested_value,
        available_cash=available_cash,
        n_in_largest_sector=max(sector_counts.values()) if sector_counts else None,
    )
    logger.info("Allocation %s max_sector=%.1f%%",
                policy.describe(), constraints["max_sector"] * 100)

    ohlcv = {}
    for ticker in tickers:
        pq = Path(paths._DATA_DIR) / "market" / ticker / "ohlcv.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
            if df.index.name and df.index.name.lower() == "date":
                df = df.reset_index()
            df.columns = df.columns.str.lower()
            df["date"] = df["date"].astype(str)
            ohlcv[ticker] = df.to_dict("records")

    return run_pipeline(signals, ohlcv, portfolio_state, constraints)


def execute_orders(snapshot, executor, dry_run: bool,
                   account: str = "A", date: str | None = None) -> list[dict]:
    orders_src = getattr(snapshot, "orders", None) or []
    if not orders_src:
        logger.info("No orders to execute")
        return []

    # DJ-129a: ids already committed by a previous (crashed) attempt tonight.
    # Prefetch failure aborts the arm BEFORE any submit: an idempotency gate
    # that cannot verify must not wave orders through.
    existing_ids: set[str] = set()
    if not dry_run and date:
        existing_ids = executor.get_client_order_ids()

    results = []
    for order in orders_src:
        ticker = order.get("ticker", order.get("symbol"))
        side = order.get("action", order.get("side", "buy")).lower()
        # allocate_capital emits "quantity"; accept "shares"/"qty" too (HiFi issue #1).
        qty = order.get("quantity", order.get("shares", order.get("qty", 0)))
        if qty <= 0:
            continue

        coid = accounts._client_order_id(account, date, ticker, side) if date else None
        if coid and coid in existing_ids:
            logger.warning("[%s] %s %s already has an order for %s (crashed "
                           "attempt?) — skipping duplicate", account, side, ticker, date)
            results.append({
                "ticker": ticker, "side": side, "qty": qty,
                "notional": None,
                "client_order_id": coid,
                "status": "skipped_duplicate",
            })
            continue

        if dry_run:
            logger.info("[DRY-RUN] Would %s %s x%s", side, ticker, qty)
            results.append({"ticker": ticker, "side": side, "qty": qty, "status": "dry_run"})
        else:
            # Isolate each order (DJ-123). A single unroutable symbol used to
            # abort the whole cycle: on 2026-08-17 EQR was removed from
            # Alpaca's asset universe, and the resulting 404 killed arm A after
            # 37 of 39 orders had already filled — taking EXC, a perfectly
            # valid ticker later in the list, down with it, and leaving no
            # decision record for a day the arm had actually traded.
            #
            # A rejected order is data, not a crash: it is recorded with its
            # error so the funnel in the report shows conviction that never
            # reached the broker.
            # Size BUYs in dollars, not shares (DJ-126). Orders are sized after
            # the close and fill at the next open; a share-count order spends
            # whatever the overnight gap decides, which put all three pipeline
            # arms on margin on 2026-08-18. Notional spends exactly the budget.
            #
            # SELLs stay share-based: a notional sell could exceed the shares
            # held if the price gapped down, and the book is long-only.
            notional = None
            if side == "buy":
                want = order.get("estimated_value")
                if want and executor.is_fractionable(ticker):
                    notional = float(want)
                elif want:
                    logger.info("[order] %s not fractionable; sizing in shares", ticker)

            try:
                result = executor.place_market_order(
                    ticker, qty, side, notional=notional,
                    **({"client_order_id": coid} if coid else {}))
                if coid:
                    existing_ids.add(coid)
                results.append({
                    "ticker": ticker, "side": side, "qty": qty,
                    "notional": round(notional, 2) if notional is not None else None,
                    "client_order_id": coid,
                    "status": result.status, "order_id": result.order_id,
                    "filled_price": result.filled_avg_price,
                })
            except Exception as exc:
                logger.error("[order] %s %s %s REJECTED: %s", side, ticker,
                             f"${notional:,.2f}" if notional else f"x{qty}", exc)
                results.append({
                    "ticker": ticker, "side": side, "qty": qty,
                    "notional": round(notional, 2) if notional is not None else None,
                    "client_order_id": coid,
                    "status": "rejected", "error": str(exc)[:200],
                })

    rejected = [r for r in results if r.get("status") == "rejected"]
    if rejected:
        logger.warning("%d of %d orders rejected: %s", len(rejected), len(results),
                       ",".join(r["ticker"] for r in rejected))
    return results


def log_episode(account: str, date: str, condition: str,
                signals: list[dict], orders: list[dict], portfolio_value: float,
                strategy_meta: dict | None = None) -> None:
    log_path = paths._decisions_log(account)
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


def run_account_cycle(account: str, tickers: list[str], date: str,
                      dry_run: bool, execute: bool, force: bool = False) -> None:
    condition = accounts._ACCOUNTS[account]["condition"]
    is_dry = dry_run or not execute

    if not is_dry and accounts.already_decided(account, date):
        if force:
            logger.warning("[%s] Already decided for %s — --force given, running again. "
                           "Annotate this date as a protocol deviation.", account, date)
        else:
            logger.info("[%s] Already decided for %s — skipping (use --force to override)",
                        account, date)
            return

    executor = accounts.get_executor(account)
    if executor is None:
        return

    logger.info("[%s] Daily cycle: condition=%s date=%s tickers=%d dry_run=%s",
                account, condition, date, len(tickers), is_dry)

    if guards.check_circuit_breakers(account, executor):
        logger.error("[%s] HALTED: circuit breaker triggered. No orders.", account)
        # A halt must suppress ORDERS, not OBSERVATION (DJ-119). This return
        # used to precede record_account(), so a halted arm silently stopped
        # capturing its equity curve: account C's portfolio_history.json froze
        # at 2026-07-17 while A/B/D ran to 07-27. Halted days are still days
        # the book was marked to market, and a gap in the benchmark's curve is
        # worse than the halt it recorded.
        if not is_dry:
            from hifi.execution.portfolio_recorder import record_account
            record_account(executor, account, paths._DATA_DIR, decision_date=date)
        executor.disconnect()
        return

    strategy_meta: dict | None = None
    # Bound before the branches so the recovery block below can tell whether
    # this cycle reached the broker (DJ-123).
    orders: list[dict] = []
    signals: list[dict] = []

    # DJ-129b: ratchet and persist the high-water mark BEFORE anything trades,
    # so every pipeline decision tonight sees a real HWM instead of today's
    # equity (which made the -15% drawdown breaker unfireable). Persistence
    # failure raises: an unverifiable risk control must fail the arm, not
    # silently revert it to the dead-breaker state.
    hwm_value: float | None = None
    if not is_dry:
        hwm_value = accounts.update_hwm(account, executor.get_portfolio_value())
        logger.info("[%s] High-water mark: $%.2f", account, hwm_value)

    # DJ-130: tag the arm whose signals are being generated this cycle and
    # snapshot its book so eligible agents receive standing-situation context.
    # Agents read data/live/<acct>/book_state.json via the orchestrator; the
    # env tag keeps injection live-only (eval harnesses never set it).
    if not is_dry:
        os.environ["HIFI_ACTIVE_ACCOUNT"] = account
        from hifi.agents.context import write_book_state

        if write_book_state(executor, account, paths._DATA_DIR):
            logger.info("[%s] Book state written for agent context (DJ-130)", account)

    if condition == "control":
        if guards._halt_before_submit(account, executor, is_dry, date):
            return
        orders = strategies.run_control_strategy(tickers, executor, dry_run=is_dry,
                                      account=account, date=None if is_dry else date)
        signals = []
    elif condition == "riskbudget":
        # External deterministic quant provider (DJ-113). as_of_date is the
        # last completed trading day; the store already holds it.
        from hifi.execution.riskbudget_strategy import get_riskbudget_signals
        payload = get_riskbudget_signals(tickers, date, paths._DATA_DIR,
                                         sectors=paths._get_sectors())
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
        if guards._halt_before_submit(account, executor, is_dry, date):
            return
        snapshot = run_mcp_pipeline(signals, tickers, executor, hwm_value=hwm_value)
        orders = execute_orders(snapshot, executor, dry_run=is_dry,
                                account=account, date=date)
    else:
        ensemble.run_ensemble(tickers, date, condition, account, dry_run=dry_run)
        if dry_run:
            logger.info("[%s] Dry-run: schedule printed, stopping before pipeline", account)
            return
        signals = ensemble.load_ensemble_signals(tickers, date, condition, account)
        if not signals:
            logger.warning("[%s] No ensemble signals for %s — skipping pipeline", account, date)
            return
        if guards._halt_before_submit(account, executor, is_dry, date):
            return
        snapshot = run_mcp_pipeline(signals, tickers, executor, hwm_value=hwm_value)
        orders = execute_orders(snapshot, executor, dry_run=is_dry,
                                account=account, date=date)

    # The broker and the experimental record must not diverge (DJ-123). On
    # 2026-08-17 arm A filled 37 orders and then raised, so log_episode never
    # ran: the account had traded but the record showed nothing for that date,
    # which is worse than a clean failure — the equity curve moves with no
    # decision to attribute it to. Recording is therefore best-effort and
    # independent of whatever else fails afterwards.
    try:
        log_episode(account, date, condition, signals, orders,
                    executor.get_portfolio_value(), strategy_meta=strategy_meta)
    except Exception:
        logger.exception("[%s] FAILED to log episode for %s — %d order(s) may be "
                         "at the broker with no decision record", account, date, len(orders))
        raise
    # Automatic financial-performance capture (DJ-114): equity + positions row
    # and Alpaca's authoritative equity curve. No human intervention.
    from hifi.execution.portfolio_recorder import record_account
    record_account(executor, account, paths._DATA_DIR, decision_date=date)
    accounts.show_status(account, executor)
    executor.disconnect()


def run_batch(tickers: list[str], date: str, arms: list[str],
              dry_run: bool, execute: bool, force: bool = False,
              resolve_session: bool = True) -> list[str]:
    """Run one evening for every requested arm. Returns the arms that failed.

    ``resolve_session`` is False only when the caller pinned a date explicitly:
    an operator asking for a specific date means that date, not the session the
    store happens to end on.
    """
    logger.info("Step 1: Updating OHLCV data for %d tickers...", len(tickers))
    market.update_data(tickers)

    # Date the decision by the session the agents actually see (DJ-121). On a
    # weekday evening this is today and nothing changes; on a weekend or the
    # evening of a market holiday it resolves back to the last real session, so
    # the run is that session's cycle executed late rather than a phantom
    # decision on a day with no close.
    if resolve_session:
        session = market._last_completed_session(tickers)
        if session and session != date:
            logger.info("Decision date -> %s (last completed session; wall-clock date is %s)",
                        session, date)
            date = session
        elif not session:
            logger.warning("Could not determine last completed session; using wall-clock %s", date)

    # Data coverage gate (DJ-120). Runs before anything writes a decision
    # record. --dry-run is exempt so the pipeline stays inspectable while the
    # store is being repaired.
    if not dry_run and not guards.check_data_coverage(tickers):
        raise SystemExit(2)

    # Broker-side universe check (DJ-123). Reports, never blocks.
    if not dry_run:
        guards.check_tradability(tickers, arms[0])

    if len(arms) > 1:
        guards.log_arm_invariance(arms)

    failed = []
    for account in arms:
        # Isolate each account: a network failure (or any error) on one arm must
        # not abort the others (DJ-117). Retries live in the executor; this is the
        # last-resort guard so a partial outage still trades the reachable arms.
        try:
            run_account_cycle(account, tickers, date, dry_run=dry_run,
                              execute=execute, force=force)
        except Exception as exc:
            failed.append(account)
            logger.error("[%s] cycle FAILED (%s); continuing with remaining arms", account, exc)

    if failed:
        logger.warning("Nightly batch done with FAILURES: failed=%s ok=%s date=%s",
                       ",".join(failed), ",".join(a for a in arms if a not in failed), date)
    else:
        logger.info("Nightly batch complete: arms=%s date=%s", ",".join(arms), date)
    return failed
