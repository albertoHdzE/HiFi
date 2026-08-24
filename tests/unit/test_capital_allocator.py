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


def test_fractional_quantity_hits_target_exactly():
    """Fractional sizing (default, DJ-121): 0.05 * 100k / 333 = 15.015 shares.

    Whole-share flooring left a residue of cash on every position and, on a
    name already near target, produced ±1-share BUY/SELL flip-flops on
    consecutive days. Alpaca supports fractional quantities and arm C already
    uses them.
    """
    orders = generate_orders(
        weights={"JPM": 0.05},
        prices={"JPM": 333.0},
        holdings={"JPM": 0},
        capital=100_000.0,
    )
    assert orders[0]["quantity"] == pytest.approx(0.05 * 100_000 / 333, abs=1e-3)


def test_whole_share_mode_still_floors():
    """fractional=False preserves the old integer behaviour for brokers
    that do not support fractional shares (IBKR is next in the queue)."""
    orders = generate_orders(
        weights={"JPM": 0.05},
        prices={"JPM": 333.0},
        holdings={"JPM": 0},
        capital=100_000.0,
        fractional=False,
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


def test_rebalancing_band_is_relative_to_target():
    """The band is a fraction of the target, not absolute percentage points.

    An absolute 5pp band was >= max_single_stock (5%), so a fresh position's
    drift (|target - 0| = target) could never exceed it and no new position
    could ever be opened once a book existed. That froze arms A, B and D at
    one position each for a month (DJ-121).
    """
    # Held 4.8% against a 5% target → 4% of target adrift → inside the band.
    inside = generate_orders(
        weights={"AAPL": 0.05},
        prices={"AAPL": 200.0},
        holdings={"AAPL": 24},   # 24 * 200 / 100k = 4.8%
        capital=100_000.0,
        current_capital=100_000.0,
    )
    assert inside == []

    # Held 2.5% against a 5% target → 50% of target adrift → rebalance.
    outside = generate_orders(
        weights={"AAPL": 0.05},
        prices={"AAPL": 200.0},
        holdings={"AAPL": 12.5},
        capital=100_000.0,
        current_capital=100_000.0,
    )
    assert len(outside) == 1
    assert outside[0]["side"] == "BUY"


def test_fresh_position_always_opens_despite_existing_book():
    """The exact regression: 30 Buy signals produced 0 orders because every
    target weight was below the absolute band."""
    weights = {f"T{i}": 0.033 for i in range(30)}
    prices = {f"T{i}": 100.0 for i in range(30)}
    orders = generate_orders(
        weights=weights,
        prices=prices,
        holdings={},              # all fresh
        capital=100_438.67,
        current_capital=4_974.20,  # a book exists (one legacy holding)
    )
    assert len(orders) == 30
    assert all(o["side"] == "BUY" for o in orders)


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


# ---------------------------------------------------------------------------
# Cash fitting and churn suppression (DJ-121)
# ---------------------------------------------------------------------------


def test_buys_scaled_to_fit_available_cash():
    """compose_portfolio spreads ~100% of capital across Buy names alone while
    Hold positions stay on the book, so demand can exceed buying power. Arm A
    asked for $100,196 of buys against $95,464 cash."""
    weights = {f"T{i}": 0.033 for i in range(30)}
    prices = {f"T{i}": 100.0 for i in range(30)}
    orders = generate_orders(
        weights=weights, prices=prices, holdings={},
        capital=100_438.67, current_capital=4_974.20,
        available_cash=95_464.47,
    )
    notional = sum(o["estimated_value"] for o in orders)
    assert notional <= 95_464.47
    # 1% buffer retained for intraday drift between decision and fill.
    assert notional == pytest.approx(95_464.47 * 0.99, rel=0.01)
    # Proportional scaling, not truncation: every name survives.
    assert len(orders) == 30


def test_relative_weighting_preserved_when_scaling():
    """Scaling must not bias the book toward names enumerated first."""
    weights = {"A": 0.10, "B": 0.05}
    prices = {"A": 100.0, "B": 100.0}
    orders = generate_orders(
        weights=weights, prices=prices, holdings={},
        capital=100_000.0, current_capital=1_000.0,
        available_cash=7_500.0,
    )
    by = {o["ticker"]: o["quantity"] for o in orders}
    assert by["A"] == pytest.approx(2 * by["B"], rel=1e-3)


def test_sells_are_never_scaled():
    """SELLs raise cash and reduce risk; the cash guard must not touch them."""
    orders = generate_orders(
        weights={"AAPL": 0.01}, prices={"AAPL": 200.0}, holdings={"AAPL": 100},
        capital=100_000.0, current_capital=100_000.0, available_cash=0.0,
    )
    assert len(orders) == 1
    assert orders[0]["side"] == "SELL"


def test_never_sells_more_than_held():
    """Long-only book: a zero target on a small holding must not short."""
    orders = generate_orders(
        weights={"AAPL": 0.0}, prices={"AAPL": 200.0}, holdings={"AAPL": 3.5},
        capital=100_000.0, current_capital=700.0,
    )
    assert orders[0]["quantity"] <= 3.5


def test_min_notional_suppresses_rounding_churn():
    """The BAC flip-flop: a ~$64 one-share delta traded every day, alternating
    side as the price wiggled. A notional deadband makes it untradeable."""
    orders = generate_orders(
        weights={"BAC": 0.05}, prices={"BAC": 64.49}, holdings={"BAC": 77},
        capital=99_037.05, current_capital=38_490.74,
    )
    assert orders == []


def test_fractional_holdings_not_truncated():
    """int() on a 3.9-share position read it as 3 and manufactured a phantom
    rebalance every night."""
    orders = generate_orders(
        weights={"AAPL": 0.0078}, prices={"AAPL": 200.0}, holdings={"AAPL": 3.9},
        capital=100_000.0, current_capital=780.0,
    )
    assert orders == []


# ---------------------------------------------------------------------------
# Exits on Sell signals (DJ-127)
# ---------------------------------------------------------------------------


class TestExits:
    """A target weight of 0 on a held name is an exit, not a rebalance.

    compose_portfolio is long-only, so it never mentions names outside the buy
    list and generate_orders never touched them. Across the whole live record
    4,206 Sell signals produced 0 sell orders on a Sell-signalled ticker: half
    of each arm's conviction never reached the portfolio while IC scored it as
    though it had.
    """

    def test_zero_target_exits_the_full_position(self):
        o = generate_orders({"AAPL": 0.0}, {"AAPL": 200.0}, {"AAPL": 50.0},
                            capital=100_000.0, current_capital=10_000.0)
        assert len(o) == 1
        assert o[0]["side"] == "SELL"
        assert o[0]["quantity"] == pytest.approx(50.0)

    def test_exit_bypasses_the_dust_deadband(self):
        """A position too small to trade must not become one that can never be
        left, or the book accumulates residue from every name it rejected."""
        o = generate_orders({"X": 0.0}, {"X": 2.0}, {"X": 10.0},
                            capital=100_000.0, current_capital=20.0)
        assert len(o) == 1 and o[0]["estimated_value"] == pytest.approx(20.0)

    def test_exit_ignores_the_rebalance_band(self):
        """Drift is measured relative to the target; with a target of 0 that
        ratio is undefined, so the band must not be consulted at all."""
        o = generate_orders({"X": 0.0}, {"X": 100.0}, {"X": 1.0},
                            capital=100_000.0, current_capital=100_000.0)
        assert o and o[0]["side"] == "SELL"

    def test_exit_never_oversells(self):
        o = generate_orders({"X": 0.0}, {"X": 100.0}, {"X": 3.5},
                            capital=100_000.0, current_capital=350.0)
        assert o[0]["quantity"] <= 3.5

    def test_zero_target_on_unheld_name_is_a_noop(self):
        assert generate_orders({"X": 0.0}, {"X": 100.0}, {},
                               capital=100_000.0, current_capital=1_000.0) == []

    def test_exits_are_not_cash_scaled(self):
        """Selling raises cash; scaling an exit by available cash would make
        an arm unable to leave a position precisely when it is short of cash."""
        o = generate_orders({"AAPL": 0.0}, {"AAPL": 200.0}, {"AAPL": 50.0},
                            capital=100_000.0, current_capital=10_000.0,
                            available_cash=0.0)
        assert len(o) == 1 and o[0]["quantity"] == pytest.approx(50.0)
