"""Arm C: the equal-weight buy-and-hold null model.

The control exists to answer "did the ensemble do anything, or did the market?".
It therefore has to be as close to the other arms as possible in every respect
except the signal — which is why it sizes in notional dollars like the pipeline
arms rather than in shares, and carries the same idempotency keys.
"""

from __future__ import annotations

import logging

from hifi.live import accounts, market

logger = logging.getLogger(__name__)


def run_control_strategy(tickers: list[str], executor, dry_run: bool,
                         account: str = "C", date: str | None = None) -> list[dict]:
    """Null model (DJ-111): buy each ticker at 1/N equity once, then hold.

    Only emits Buy orders for tickers with no existing position. No
    rebalancing, no selling — pure buy-and-hold market exposure.

    With ``date`` set (live path), submission is idempotent (DJ-129a): each
    order carries a deterministic client_order_id and ids already at the broker
    from a crashed earlier attempt are skipped, never duplicated.
    """
    positions = executor.get_positions()
    equity = executor.get_portfolio_value()
    cash = executor.get_account_cash()
    prices = market._latest_prices(tickers)

    # DJ-129a: ids already committed by a previous (crashed) attempt tonight.
    # A prefetch failure aborts before any submit — "cannot verify" must not
    # become "submit anyway".
    existing_ids: set[str] = set()
    if not dry_run and date:
        existing_ids = executor.get_client_order_ids()

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

        coid = accounts._client_order_id(account, date, ticker, "buy") if date else None
        if coid and coid in existing_ids:
            logger.warning("[C] %s already has an open/closed order for %s "
                           "(crashed attempt?) — skipping duplicate", ticker, date)
            orders.append({
                "ticker": ticker, "side": "buy", "qty": qty,
                "notional": round(target_value, 2) if fractionable else None,
                "status": "skipped_duplicate", "client_order_id": coid,
            })
            # Do NOT add to local `spend`: the duplicate's cash was already
            # reserved at the broker and is reflected in the cash read above.
            continue

        if dry_run:
            logger.info("[DRY-RUN] Control would buy %s x%d (~$%.2f)", ticker, qty, qty * price)
            orders.append({"ticker": ticker, "side": "buy", "qty": qty,
                           "client_order_id": coid, "status": "dry_run"})
        else:
            # Same per-order isolation as the pipeline arms (DJ-123): a symbol
            # delisted since the universe was fixed must not stop the control
            # from completing its book.
            #
            # Notional sizing where the asset allows it (DJ-126), for the same
            # reason as the pipeline arms AND to keep the arms comparable: if
            # C sized in shares while A/B/D sized in dollars, the control alone
            # would absorb the overnight gap and its deployed exposure would
            # drift from theirs for a purely mechanical reason.
            notional = round(target_value, 2) if fractionable else None
            try:
                result = executor.place_market_order(
                    ticker, qty, "buy", notional=notional,
                    **({"client_order_id": coid} if coid else {}))
                if coid:
                    existing_ids.add(coid)
                orders.append({
                    "ticker": ticker, "side": "buy", "qty": qty,
                    "notional": notional,
                    "client_order_id": coid,
                    "status": result.status, "order_id": result.order_id,
                })
            except Exception as exc:
                logger.error("[C] buy %s %s REJECTED: %s", ticker,
                             f"${notional:,.2f}" if notional else f"x{qty}", exc)
                orders.append({
                    "ticker": ticker, "side": "buy", "qty": qty,
                    "notional": notional,
                    "client_order_id": coid,
                    "status": "rejected", "error": str(exc)[:200],
                })
                continue
        # Notional spends exactly target_value; share orders spend qty*price.
        spend += target_value if (not dry_run and fractionable) else qty * price

    logger.info("Control strategy: %d orders, ~$%.2f notional", len(orders), spend)
    return orders
