"""Everything that can stop or qualify a cycle.

Grouped deliberately. These are not interchangeable safety checks: they differ
in whether they block, and the difference is a scientific judgement recorded in
each docstring. ``check_data_coverage`` blocks, because a starved run produces
decision records indistinguishable from opinions. ``check_tradability`` and
``log_arm_invariance`` only report, because a delisting and a concentrated book
are facts about the world rather than faults in the apparatus. The circuit
breakers halt an arm but never abort the batch.

``log_arm_invariance`` is the one to read first: an experiment comparing
ensemble architectures is valid only if every rule that is *not* the treatment
applies equally to every arm, and DJ-119 was a case where a risk control failed
that test and silently taxed the widest book.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from hifi.live import accounts, paths

logger = logging.getLogger(__name__)

_DAILY_LOSS_LIMIT = 0.02       # 2% portfolio loss -> halt
_POSITION_LOSS_LIMIT = 0.10    # 10% single position -> FLAG only (never halts)
_POSITION_IMPACT_LIMIT = 0.02  # single position costing >2% of equity -> halt
_VANISH_LOOKBACK_SNAPSHOTS = 5  # snapshots scanned for broker-removed positions (DJ-123)


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

            # A position removed by a corporate action is not a trading loss
            # (DJ-123). On 2026-08-17 EQR was deleted from Alpaca's asset
            # universe; arm D's holding vanished from equity without being
            # credited to cash, and the resulting -3.72% "daily loss" halted a
            # healthy arm for a day. The breaker must measure P&L, not
            # bookkeeping, or delistings silently cost trading days.
            vanished = _vanished_position_value(account, executor)
            if vanished > 0:
                adjusted = (equity + vanished - last_equity) / last_equity
                logger.warning(
                    "[%s] $%.2f of equity vanished with no matching cash credit "
                    "(position removed by the broker, not sold). Daily change "
                    "%.2f%% -> %.2f%% after excluding it.",
                    account, vanished, daily_change * 100, adjusted * 100)
                _log_circuit_breaker(account, "position_removed", vanished, equity,
                                     action="flag")
                daily_change = adjusted

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


def effective_halt_threshold(n_positions: int, exposure: float = 1.0) -> float:
    """Per-position loss that would halt an equal-weight book of this width.

    Exposed so the invariance check can state, in one number per arm, how
    unequally the apparatus constrains the arms.
    """
    if n_positions <= 0 or exposure <= 0:
        return float("inf")
    weight = exposure / n_positions
    return max(_POSITION_LOSS_LIMIT, _POSITION_IMPACT_LIMIT / weight)


def _halt_before_submit(account: str, executor, is_dry: bool,
                        date: str | None) -> bool:
    """Re-run the breakers immediately before order submission (DJ-129c).

    The main check runs hours earlier, before the LLM passes; between that
    check and the first submit lies the entire cycle. A book can cross a limit
    in that window only through external marks — but the broker/API itself can
    also have degraded, which is exactly when the old single-check design
    submitted blind. Returns True if the arm must stop; records and disconnects
    on trip so a halted arm still lands in the equity curve (DJ-119).
    """
    if is_dry:
        return False
    if not check_circuit_breakers(account, executor):
        return False
    logger.error("[%s] PRE-SUBMIT HALT: breakers tripped after signal "
                 "generation — no orders submitted", account)
    from hifi.execution.portfolio_recorder import record_account
    record_account(executor, account, paths._DATA_DIR, decision_date=date)
    executor.disconnect()
    return True


def check_data_coverage(tickers: list[str], min_fraction: float = 0.99) -> bool:
    """Pre-flight: refuse to run agents over a starved universe (DJ-120).

    Unlike the invariance probe this one BLOCKS, because the failure mode it
    guards against is not a visible outage but a plausible-looking result. When
    the MCP tools answered TICKER_NOT_FOUND the agents did not error — they
    reasoned over the absence and returned "no data available -- Sell" at
    confidence 1.0. Eighty-three of ninety-eight tickers were in that state for
    a month of live trading and the only symptom was an unusually bearish arm.

    A cycle that cannot see its universe produces decision records that are
    worse than none: they are indistinguishable from opinions and they enter
    the permanent experimental record. Better to fail loudly and lose a night.

    Also warns on stale bars, which is the same defect's quieter half: the 15
    tickers that *did* resolve were being analysed on 2023 prices.
    """
    from hifi.data.market_store import coverage_report

    cov = coverage_report(tickers, paths._DATA_DIR)
    missing = sorted(t for t, r in cov.items() if not r["found"])
    legacy = sorted(t for t, r in cov.items() if r["layout"] == "flat-legacy")
    found = len(tickers) - len(missing)
    fraction = found / len(tickers) if tickers else 0.0

    logger.info("Data coverage pre-flight (DJ-120): %d/%d tickers resolved (%.1f%%)",
                found, len(tickers), fraction * 100)

    if legacy:
        logger.warning(
            "%d ticker(s) resolved to the STALE legacy flat layout (bars end 2023-06-30): %s. "
            "The canonical nested store data/market/<TICKER>/ohlcv.parquet is missing for these.",
            len(legacy), ",".join(legacy[:10]) + ("..." if len(legacy) > 10 else ""))

    last_dates = {r["last_date"] for r in cov.values() if r["found"] and r["last_date"]}
    if last_dates:
        logger.info("  bar coverage: last_date range %s .. %s",
                    min(last_dates), max(last_dates))

    if fraction < min_fraction:
        logger.error(
            "ABORT: only %d/%d tickers (%.1f%%) have OHLCV data; threshold is %.0f%%. "
            "Missing: %s. Agents would report the gap as bearish conviction rather "
            "than as an error (DJ-120) — refusing to write decision records. "
            "Run `make live-update-data` or check data/market/ layout.",
            found, len(tickers), fraction * 100, min_fraction * 100,
            ",".join(missing[:15]) + ("..." if len(missing) > 15 else ""))
        return False
    return True


def _vanished_position_value(account: str, executor) -> float:
    """Value that left the book without becoming cash (DJ-123).

    Compares the last recorded snapshot with the live account. A symbol held
    then and absent now is either (a) sold, in which case its value reappears
    as cash and equity is unaffected, or (b) removed by the broker through a
    delisting or corporate action, in which case the value simply disappears.

    Only the unexplained remainder is returned, so a legitimate sale is never
    mistaken for a vanished position — which would turn a flat day into an
    apparent gain and blind the breaker in the opposite direction.
    """
    from hifi.analytics.live_report import _read_jsonl  # noqa: PLC0415

    rows = _read_jsonl(Path(paths._DATA_DIR) / "live" / account / "equity.jsonl")
    if not rows:
        return 0.0

    try:
        current = set(executor.get_positions())
    except Exception as exc:
        logger.warning("[%s] could not read positions for vanish check: %s", account, exc)
        return 0.0

    # Identify the event directly rather than inferring it from snapshot dates.
    # Date alignment is unreliable: Alpaca's `last_equity` is struck at a
    # session close we cannot observe, and our own snapshot may already have
    # been rewritten post-removal (which is exactly the state arm D was left in
    # after 2026-08-17). A symbol we recorded holding, which is absent from the
    # account AND which the broker no longer recognises as an asset at all, is
    # unambiguously a delisting — never a sale.
    recent = rows[-_VANISH_LOOKBACK_SNAPSHOTS:]
    believed: dict[str, float] = {}
    for row in recent:
        for p in row.get("positions", []):
            sym = p.get("ticker")
            if sym and sym not in current:
                believed[sym] = float(p.get("market_value", 0.0))
    if not believed:
        return 0.0

    total = 0.0
    for sym, value in believed.items():
        try:
            executor.client.get_asset(sym)
        except Exception:
            # Asset no longer exists at the broker: its value did not become
            # cash, it simply left the book.
            logger.warning("[%s] %s is no longer a broker asset; $%.2f left the book "
                           "without a matching cash credit", account, sym, value)
            total += value
    return total


def check_tradability(tickers: list[str], account: str = "A") -> list[str]:
    """Pre-flight: which universe tickers the broker will no longer trade (DJ-123).

    The data-coverage gate checks our parquet store; this checks the broker.
    The two can disagree, and on 2026-08-17 they did: EQR had been removed
    from Alpaca's asset universe entirely while our store still held bars
    through 2026-08-13, so coverage passed 98/98 and the failure only surfaced
    four hours later as a 404 in the middle of order placement.

    Reports rather than blocks. A delisting is a fact about the world, not a
    fault in the run, and the remaining 97 names are still a valid experiment
    — but it must be visible before the agents spend a night analysing a
    security that cannot be bought, and before it silently changes the
    universe out from under the ablation.
    """
    executor = accounts.get_executor(account)
    if executor is None:
        logger.warning("Tradability pre-flight skipped: no executor for %s", account)
        return []
    untradable = []
    try:
        for ticker in tickers:
            try:
                asset = executor.client.get_asset(ticker)
                if not getattr(asset, "tradable", True):
                    untradable.append(ticker)
            except Exception:
                untradable.append(ticker)
    finally:
        executor.disconnect()

    if untradable:
        logger.warning(
            "Tradability pre-flight (DJ-123): %d/%d universe ticker(s) NOT tradable "
            "at the broker: %s. Agents will still analyse them and any resulting "
            "order will be rejected and recorded. Consider retiring them from the "
            "universe and noting the change in the experiment record.",
            len(untradable), len(tickers), ",".join(untradable))
    else:
        logger.info("Tradability pre-flight (DJ-123): %d/%d tickers tradable",
                    len(tickers), len(tickers))
    return untradable


def log_arm_invariance(arms: list[str]) -> None:
    """Pre-flight: the apparatus must not constrain arms differently (DJ-119).

    An experiment comparing ensemble architectures is only valid if every rule
    that is NOT the treatment applies equally to every arm. The old per-position
    breaker violated this: halt probability was 1-(1-p)^N in book width N, and N
    is downstream of the treatment (a diverse ensemble spreads, a herding one
    concentrates), so the "risk control" was a trading tax monotone in exactly
    the quantity under test. It silenced C for five runs before anyone noticed.

    This logs each arm's book width, exposure and effective halt threshold so
    that divergence is visible on every run instead of being inferred from a
    week of missing orders. It only reports — it never blocks a cycle — because
    a genuinely concentrated arm is a legitimate outcome, not a fault; what must
    not happen silently is the apparatus *causing* the concentration.
    """
    rows = []
    for account in arms:
        executor = accounts.get_executor(account)
        if executor is None:
            continue
        try:
            equity = executor.get_portfolio_value()
            positions = executor.get_positions()
            invested = sum(p.market_value for p in positions.values())
            exposure = invested / equity if equity > 0 else 0.0
            rows.append((account, accounts._ACCOUNTS[account]["condition"], len(positions),
                         exposure, effective_halt_threshold(len(positions), exposure)))
        except Exception as exc:
            logger.warning("[%s] invariance probe failed: %s", account, exc)
        finally:
            executor.disconnect()

    if not rows:
        return

    logger.info("Arm invariance probe (DJ-119):")
    for account, condition, n_pos, exposure, thresh in rows:
        logger.info("  [%s] %-11s n_positions=%3d exposure=%5.1f%% halt_at=%.0f%% loss",
                    account, condition, n_pos, exposure * 100, thresh * 100)

    exposures = [r[3] for r in rows]
    spread = max(exposures) - min(exposures)
    if spread > 0.5:
        logger.warning(
            "Arms differ in capital deployment by %.0f pp (%.0f%% vs %.0f%%). Raw return, "
            "Sharpe and drawdown are NOT comparable across arms at this spread — they "
            "measure exposure, not signal. Use IC/herding, or exposure-adjust.",
            spread * 100, max(exposures) * 100, min(exposures) * 100)


def _log_circuit_breaker(account: str, trigger: str, value: float,
                         equity: float, ticker: str = "", action: str = "halt",
                         weight: float | None = None,
                         impact: float | None = None) -> None:
    log_path = paths._breaker_log(account)
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
