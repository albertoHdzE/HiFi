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
_REBAL_THRESHOLD = 0.05         # rebalance only when drift exceeds 5%
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
    holdings: dict[str, int],
    capital: float,
    current_capital: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Generate MARKET orders to rebalance toward target weights.

    Parameters
    ----------
    weights : dict[str, float]
        Target portfolio weight per ticker (values should sum ≤ 1.0).
    prices : dict[str, float]
        Current price per share per ticker.
    holdings : dict[str, int]
        Current share holdings per ticker.
    capital : float
        Total capital available for investment.
    current_capital : float
        Total current portfolio value (for current-weight computation).
        Pass 0.0 when starting from cash.

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
        target_shares = int(math.floor(target_value / price))

        current_shares = int(holdings.get(ticker, 0))
        current_value = current_shares * price
        current_weight = (current_value / current_capital) if current_capital > 0 else 0.0

        # Rebalancing threshold — only applies when there IS an existing portfolio
        if current_capital > 0 and abs(effective_weight - current_weight) <= _REBAL_THRESHOLD:
            logger.debug(
                "%s drift=%.4f <= %.2f threshold; skipping",
                ticker, abs(effective_weight - current_weight), _REBAL_THRESHOLD,
            )
            continue

        delta = target_shares - current_shares
        if delta == 0:
            continue

        side = "BUY" if delta > 0 else "SELL"
        qty = abs(delta)
        commission = ibkr_commission(qty)

        orders.append({
            "ticker": ticker,
            "side": side,
            "quantity": qty,
            "order_type": "MARKET",
            "estimated_cost": round(commission, 4),
            "estimated_value": round(qty * price, 4),
        })

    return orders


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
        Schema: ``{"AAPL": 100, "MSFT": 0, ...}``
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
        weights={str(k): float(v) for k, v in weights.items()},
        prices={str(k): float(v) for k, v in prices.items()},
        holdings={str(k): int(v) for k, v in holdings.items()},
        capital=float(capital),
        current_capital=float(current_capital),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
