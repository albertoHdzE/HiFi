"""
HiFi Capital Allocator MCP Server (E4-T3, DJ-091).

Exposes one MCP tool: ``allocate_capital``.  Pure deterministic math,
no LLMs, no external data fetches.

Algorithm (applied in order):
  1. Target shares: floor(target_weight * capital / price).
  2. Kelly cap: max 25% of capital per single position.
  3. Rebalancing threshold: skip order if |current_weight - target_weight| <= 5%.
  4. IBKR tiered commission:
       qty <= 300 shares: max($0.0035 * qty, $0.35)
       qty  > 300 shares: $0.002 * qty
  5. Emit MARKET orders only (limit orders deferred to Phase 17+, DJ-091).

Transport: stdio MCP (DJ-009).
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("hifi-capital-allocator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KELLY_CAP = 0.25               # max 25% of capital per position

# Rebalance band, expressed as a FRACTION OF THE TARGET WEIGHT (DJ-121).
#
# This was an absolute 0.05 (5 percentage points), which silently froze every
# arm. The skip test for a fresh position is |target - 0| = target, and
# max_single_stock is also 0.05 — so no new position could ever exceed the
# band. Once an arm held anything (current_capital > 0) it could not open
# another position again: A and B each opened exactly one holding on day one
# while still 100% cash, then sat at ~95% cash for a month.
#
# A relative band scales with book width instead: a fresh position is always
# 100% adrift and trades, while a held position near its target does not.
_REBAL_DRIFT_FRAC = 0.20

# Orders below this notional are suppressed. Integer share rounding on a
# single name produced ±1-share BUY/SELL flip-flops on consecutive days
# (arm D churned BAC daily); a deadband makes rounding noise untradeable.
_MIN_ORDER_NOTIONAL = 50.0
_IBKR_PER_SHARE_SMALL = 0.0035  # $/share for qty <= 300
_IBKR_MIN_COMMISSION = 0.35     # $ minimum commission (small lots)
_IBKR_PER_SHARE_LARGE = 0.002   # $/share for qty > 300


# ---------------------------------------------------------------------------
# Commission helper (importable for unit testing)
# ---------------------------------------------------------------------------


def ibkr_commission(qty: int) -> float:
    """
    IBKR tiered per-share commission.

    Parameters
    ----------
    qty : int
        Number of shares (must be positive).

    Returns
    -------
    float
        Estimated commission in USD.
    """
    if qty <= 0:
        return 0.0
    if qty <= 300:
        return max(_IBKR_PER_SHARE_SMALL * qty, _IBKR_MIN_COMMISSION)
    return _IBKR_PER_SHARE_LARGE * qty


# ---------------------------------------------------------------------------
# Order generation (importable for unit testing)
# ---------------------------------------------------------------------------


def generate_orders(
    weights: dict[str, float],
    prices: dict[str, float],
    holdings: dict[str, float],
    capital: float,
    current_capital: float = 0.0,
    fractional: bool = True,
    min_notional: float = _MIN_ORDER_NOTIONAL,
    drift_frac: float = _REBAL_DRIFT_FRAC,
    available_cash: float | None = None,
    cash_buffer: float = 0.01,
) -> list[dict[str, Any]]:
    """
    Generate MARKET orders to rebalance toward target weights.

    Both the target and the current weight are measured against ``capital``
    (total portfolio value). They previously used different denominators —
    target on ``capital``, current on ``current_capital`` (invested value) —
    which inflated the apparent weight of held names whenever an arm carried
    cash. At 61% cash that made a 5.0% position read as 12.8% (DJ-121).

    Parameters
    ----------
    weights : dict[str, float]
        Target portfolio weight per ticker (values should sum ≤ 1.0).
    prices : dict[str, float]
        Current price per share per ticker.
    holdings : dict[str, float]
        Current share holdings per ticker. Fractional holdings are supported.
    capital : float
        Total portfolio value — the denominator for every weight.
    current_capital : float
        Value of existing holdings. Used only as an "is there a book yet?"
        flag: the rebalance band is not applied to a portfolio starting from
        cash, so the opening allocation is never suppressed.
    fractional : bool
        Emit fractional share quantities (Alpaca supports these, and arm C
        already uses them). When False, quantities are floored to whole shares.
    min_notional : float
        Suppress orders below this dollar value.
    drift_frac : float
        Skip a rebalance when |target - current| is within this fraction of
        the target weight.
    available_cash : float | None
        Buying power. When given, BUY quantities are scaled down
        proportionally so their total notional fits. Necessary because
        ``compose_portfolio`` allocates ~100% of capital across the Buy names
        alone while Hold positions stay on the book — targets can therefore
        sum past 100% and exceed cash. None disables the check.
    cash_buffer : float
        Fraction of available cash held back from the scaling, mirroring the
        1% buffer arm C uses to absorb intraday price moves between decision
        and fill.

    Returns
    -------
    list[dict]
        MARKET order dicts with fields:
        ticker, side, quantity, order_type, estimated_cost, estimated_value.
    """
    orders: list[dict[str, Any]] = []

    for ticker, target_weight in weights.items():
        price = float(prices.get(ticker, 0.0))
        if price <= 0:
            logger.warning("No valid price for %s; skipping", ticker)
            continue

        # Kelly cap
        effective_weight = min(float(target_weight), _KELLY_CAP)
        target_value = effective_weight * capital

        current_shares = float(holdings.get(ticker, 0) or 0.0)
        current_value = current_shares * price
        current_weight = (current_value / capital) if capital > 0 else 0.0

        # Rebalance band, relative to the target. Only applies once a book
        # exists — the first allocation out of cash must never be suppressed.
        drift = abs(effective_weight - current_weight)
        if (current_capital > 0 and effective_weight > 0
                and drift / effective_weight <= drift_frac):
            logger.debug(
                "%s drift=%.4f (%.1f%% of target %.4f) within band; skipping",
                ticker, drift, 100 * drift / effective_weight, effective_weight,
            )
            continue

        delta_value = target_value - current_value
        if abs(delta_value) < min_notional:
            logger.debug("%s delta $%.2f below $%.2f deadband; skipping",
                         ticker, abs(delta_value), min_notional)
            continue

        qty = abs(delta_value) / price
        if not fractional:
            qty = float(math.floor(qty))
        qty = round(qty, 3)
        if qty <= 0:
            continue

        side = "BUY" if delta_value > 0 else "SELL"
        # Never sell more than is held — the book is long-only.
        if side == "SELL":
            qty = round(min(qty, current_shares), 3)
            if qty <= 0:
                continue

        commission = ibkr_commission(math.ceil(qty))

        orders.append({
            "ticker": ticker,
            "side": side,
            "quantity": qty,
            "order_type": "MARKET",
            "estimated_cost": round(commission, 4),
            "estimated_value": round(qty * price, 4),
        })

    return _fit_to_cash(orders, prices, available_cash, cash_buffer, fractional, min_notional)


def _fit_to_cash(
    orders: list[dict[str, Any]],
    prices: dict[str, float],
    available_cash: float | None,
    cash_buffer: float,
    fractional: bool,
    min_notional: float,
) -> list[dict[str, Any]]:
    """Scale BUY orders down proportionally to fit available buying power.

    Proportional scaling preserves the *relative* weighting the ensemble
    asked for, which matters here: the experiment measures how signals map to
    a portfolio, so truncating the tail of the buy list (the obvious
    alternative) would silently bias the book toward whichever names happened
    to be enumerated first.

    SELLs are never scaled — they raise cash and reduce risk.
    """
    if available_cash is None:
        return orders

    buys = [o for o in orders if o["side"] == "BUY"]
    if not buys:
        return orders

    budget = max(0.0, available_cash * (1.0 - cash_buffer))
    demand = sum(o["estimated_value"] for o in buys)
    if demand <= budget:
        return orders

    scale = budget / demand if demand > 0 else 0.0
    logger.warning(
        "BUY demand $%.2f exceeds buying power $%.2f; scaling all buys by %.3f",
        demand, budget, scale,
    )

    out = []
    for o in orders:
        if o["side"] != "BUY":
            out.append(o)
            continue
        price = float(prices.get(o["ticker"], 0.0))
        qty = o["quantity"] * scale
        if not fractional:
            qty = float(math.floor(qty))
        qty = round(qty, 3)
        if qty <= 0 or qty * price < min_notional:
            continue
        out.append({
            **o,
            "quantity": qty,
            "estimated_cost": round(ibkr_commission(math.ceil(qty)), 4),
            "estimated_value": round(qty * price, 4),
        })
    return out


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------


@mcp.tool()
def allocate_capital(
    weights_json: str,
    prices_json: str,
    holdings_json: str,
    capital: float,
    current_capital: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Generate MARKET orders to rebalance toward target portfolio weights.

    Parameters
    ----------
    weights_json : str
        JSON object: target portfolio weights per ticker.
        Schema: ``{"AAPL": 0.05, "MSFT": 0.04, ...}``
    prices_json : str
        JSON object: current price per share.
        Schema: ``{"AAPL": 175.50, "MSFT": 310.20, ...}``
    holdings_json : str
        JSON object: current share holdings.
        Schema: ``{"AAPL": 100, "MSFT": 0, "NVDA": 3.063, ...}``
        Fractional holdings are accepted.
    capital : float
        Total capital available for investment ($).
    current_capital : float
        Current portfolio value for weight-drift computation (0 = all cash).

    Returns
    -------
    list[dict]
        MARKET order dicts.  Each has: ticker, side (BUY|SELL), quantity,
        order_type, estimated_cost ($), estimated_value ($).
        On parse error: ``[{"error": ..., "detail": ...}]``.
    """
    try:
        weights = json.loads(weights_json)
        if not isinstance(weights, dict):
            raise ValueError("weights_json must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return [{"error": "INVALID_WEIGHTS_JSON", "detail": str(exc)}]

    try:
        prices = json.loads(prices_json)
        if not isinstance(prices, dict):
            raise ValueError("prices_json must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return [{"error": "INVALID_PRICES_JSON", "detail": str(exc)}]

    try:
        holdings = json.loads(holdings_json)
        if not isinstance(holdings, dict):
            raise ValueError("holdings_json must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return [{"error": "INVALID_HOLDINGS_JSON", "detail": str(exc)}]

    return generate_orders(
        # Holdings are float: fractional shares are supported end to end.
        weights={str(k): float(v) for k, v in weights.items()},
        prices={str(k): float(v) for k, v in prices.items()},
        holdings={str(k): float(v) for k, v in holdings.items()},
        capital=float(capital),
        current_capital=float(current_capital),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
