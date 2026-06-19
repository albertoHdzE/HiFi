"""
Unit tests for allocate_capital (E4-T3, DJ-091).

Tests:
- Correct share quantities for $100k capital with known weights and prices.
- IBKR commission within expected schedule (small/large lot boundary).
- Rebalancing threshold suppresses trivial rebalances.
- Kelly cap (25%) enforced on oversized weight targets.
- Empty weights produces no orders.
- SELL orders generated for over-allocated positions.
- MCP tool round-trip (JSON parse).
"""

from __future__ import annotations

import json
import math

import pytest

from hifi.mcp.capital_allocator import (
    _IBKR_MIN_COMMISSION,
    _KELLY_CAP,
    generate_orders,
    ibkr_commission,
)

# ---------------------------------------------------------------------------
# ibkr_commission
# ---------------------------------------------------------------------------


def test_commission_small_lot_above_minimum():
    # 100 shares × $0.0035 = $0.35 → hits minimum
    c = ibkr_commission(100)
    assert c == pytest.approx(max(0.0035 * 100, 0.35))


def test_commission_small_lot_below_minimum():
    # 1 share × $0.0035 = $0.0035 < $0.35 → uses minimum
    c = ibkr_commission(1)
    assert c == pytest.approx(_IBKR_MIN_COMMISSION)


def test_commission_exactly_300_shares():
    c = ibkr_commission(300)
    assert c == pytest.approx(max(0.0035 * 300, 0.35))


def test_commission_above_300_large_lot():
    # 301 shares × $0.002 = $0.602
    c = ibkr_commission(301)
    assert c == pytest.approx(0.002 * 301)


def test_commission_large_lot_1000_shares():
    c = ibkr_commission(1000)
    assert c == pytest.approx(0.002 * 1000)


def test_commission_zero_shares():
    assert ibkr_commission(0) == 0.0


# ---------------------------------------------------------------------------
# generate_orders — basic share quantities
# ---------------------------------------------------------------------------


def test_correct_share_quantities():
    """$100k, target weight 0.05 AAPL @ $200 → 25 shares."""
    orders = generate_orders(
        weights={"AAPL": 0.05},
        prices={"AAPL": 200.0},
        holdings={"AAPL": 0},
        capital=100_000.0,
        current_capital=0.0,
    )
    assert len(orders) == 1
    assert orders[0]["ticker"] == "AAPL"
    assert orders[0]["side"] == "BUY"
    # floor(0.05 * 100_000 / 200) = floor(25.0) = 25
    assert orders[0]["quantity"] == 25


def test_share_quantity_floor():
    """floor() applied: 0.05 * 100k / 333 = 15.01 → 15 shares."""
    orders = generate_orders(
        weights={"JPM": 0.05},
        prices={"JPM": 333.0},
        holdings={"JPM": 0},
        capital=100_000.0,
    )
    assert orders[0]["quantity"] == math.floor(0.05 * 100_000 / 333)


def test_multiple_tickers():
    weights = {"AAPL": 0.05, "MSFT": 0.04, "GOOGL": 0.03}
    prices = {"AAPL": 200.0, "MSFT": 400.0, "GOOGL": 150.0}
    orders = generate_orders(
        weights=weights,
        prices=prices,
        holdings={},
        capital=100_000.0,
    )
    assert len(orders) == 3
    tickers = {o["ticker"] for o in orders}
    assert tickers == {"AAPL", "MSFT", "GOOGL"}


# ---------------------------------------------------------------------------
# Kelly cap
# ---------------------------------------------------------------------------


def test_kelly_cap_enforced():
    """Weight of 0.40 → capped at 0.25; $250k × 0.25 / $100 = 625 shares."""
    orders = generate_orders(
        weights={"AAPL": 0.40},
        prices={"AAPL": 100.0},
        holdings={"AAPL": 0},
        capital=250_000.0,
    )
    assert len(orders) == 1
    expected = math.floor(_KELLY_CAP * 250_000.0 / 100.0)
    assert orders[0]["quantity"] == expected


def test_kelly_cap_weight_below_cap_unchanged():
    """Weight of 0.10 < 0.25 cap → no cap applied."""
    orders = generate_orders(
        weights={"AAPL": 0.10},
        prices={"AAPL": 100.0},
        holdings={},
        capital=100_000.0,
    )
    expected = math.floor(0.10 * 100_000.0 / 100.0)
    assert orders[0]["quantity"] == expected


# ---------------------------------------------------------------------------
# Rebalancing threshold
# ---------------------------------------------------------------------------


def test_rebalancing_threshold_suppresses_trivial_order():
    """Current weight == target weight → no order generated."""
    # $100k capital, 50 shares @ $200 = $10k = 10% weight
    # Target weight 0.10, current weight 10000/100000 = 0.10 → drift = 0
    orders = generate_orders(
        weights={"AAPL": 0.10},
        prices={"AAPL": 200.0},
        holdings={"AAPL": 50},
        capital=100_000.0,
        current_capital=100_000.0,
    )
    assert orders == []


def test_rebalancing_threshold_drift_within_band():
    """Drift of exactly 5% (boundary) → no order generated (existing portfolio)."""
    # current weight = 0.05, target = 0.10, drift = 0.05 → at threshold, skip
    orders = generate_orders(
        weights={"AAPL": 0.10},
        prices={"AAPL": 200.0},
        holdings={"AAPL": 25},   # 25 * 200 / 100k = 5%
        capital=100_000.0,
        current_capital=100_000.0,  # existing portfolio → threshold applies
    )
    assert orders == []


def test_rebalancing_threshold_drift_above_band():
    """Drift > 5% → order generated."""
    # current weight = 0.0 (no holding), target = 0.10, drift = 0.10 > 0.05
    orders = generate_orders(
        weights={"AAPL": 0.10},
        prices={"AAPL": 200.0},
        holdings={"AAPL": 0},
        capital=100_000.0,
        current_capital=100_000.0,
    )
    assert len(orders) == 1


# ---------------------------------------------------------------------------
# SELL orders
# ---------------------------------------------------------------------------


def test_sell_order_when_overweight():
    """Holdings exceed target → SELL order generated."""
    # Holdings: 100 shares @ $200 = $20k = 20% of $100k
    # Target weight: 0.05 = 5% → target shares = floor(0.05 * 100k / 200) = 25
    # delta = 25 - 100 = -75 → SELL 75
    orders = generate_orders(
        weights={"AAPL": 0.05},
        prices={"AAPL": 200.0},
        holdings={"AAPL": 100},
        capital=100_000.0,
        current_capital=100_000.0,
    )
    assert len(orders) == 1
    assert orders[0]["side"] == "SELL"
    assert orders[0]["quantity"] == 75


# ---------------------------------------------------------------------------
# Commission in orders
# ---------------------------------------------------------------------------


def test_commission_is_positive():
    orders = generate_orders(
        weights={"AAPL": 0.10},
        prices={"AAPL": 100.0},
        holdings={},
        capital=100_000.0,
    )
    assert orders[0]["estimated_cost"] > 0


def test_commission_uses_correct_schedule():
    """500 shares → large lot: $0.002 × 500 = $1.00"""
    orders = generate_orders(
        weights={"AAPL": 0.50},
        prices={"AAPL": 100.0},  # 0.25 cap → $25k / $100 = 250 shares (small lot)
        holdings={},
        capital=100_000.0,
    )
    qty = orders[0]["quantity"]
    expected = max(0.0035 * qty, 0.35) if qty <= 300 else 0.002 * qty
    assert orders[0]["estimated_cost"] == pytest.approx(expected, abs=0.001)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_weights():
    orders = generate_orders(weights={}, prices={}, holdings={}, capital=100_000.0)
    assert orders == []


def test_missing_price_skipped():
    orders = generate_orders(
        weights={"AAPL": 0.05, "MSFT": 0.05},
        prices={"AAPL": 200.0},  # MSFT missing
        holdings={},
        capital=100_000.0,
    )
    tickers = {o["ticker"] for o in orders}
    assert "MSFT" not in tickers


def test_zero_price_skipped():
    orders = generate_orders(
        weights={"AAPL": 0.05},
        prices={"AAPL": 0.0},
        holdings={},
        capital=100_000.0,
    )
    assert orders == []


# ---------------------------------------------------------------------------
# MCP tool round-trip
# ---------------------------------------------------------------------------


def test_mcp_tool_valid_json():
    from hifi.mcp.capital_allocator import allocate_capital

    weights = {"AAPL": 0.05, "MSFT": 0.04}
    prices = {"AAPL": 200.0, "MSFT": 350.0}
    holdings = {"AAPL": 0, "MSFT": 0}

    orders = allocate_capital(
        weights_json=json.dumps(weights),
        prices_json=json.dumps(prices),
        holdings_json=json.dumps(holdings),
        capital=100_000.0,
    )
    assert isinstance(orders, list)
    assert all("ticker" in o for o in orders)


def test_mcp_tool_invalid_weights_json():
    from hifi.mcp.capital_allocator import allocate_capital

    result = allocate_capital(
        weights_json="bad json",
        prices_json="{}",
        holdings_json="{}",
        capital=100_000.0,
    )
    assert result[0].get("error") == "INVALID_WEIGHTS_JSON"


# ---------------------------------------------------------------------------
# $250k integration scenario (10 tickers, spec)
# ---------------------------------------------------------------------------


def test_250k_capital_ten_tickers():
    """Broad scenario: 10 tickers, no existing holdings, all orders are BUY."""
    weights = {f"T{i}": 0.05 for i in range(10)}  # 10 × 5% = 50% invested
    prices = {f"T{i}": float(100 + i * 10) for i in range(10)}
    orders = generate_orders(
        weights=weights,
        prices=prices,
        holdings={},
        capital=250_000.0,
    )
    assert len(orders) == 10
    assert all(o["side"] == "BUY" for o in orders)
    total_notional = sum(o["estimated_value"] for o in orders)
    assert total_notional <= 250_000.0 * 1.01  # small floor() rounding tolerance
    total_commission = sum(o["estimated_cost"] for o in orders)
    assert total_commission < total_notional * 0.005  # commission < 0.5% of notional
