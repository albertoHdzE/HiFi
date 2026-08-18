"""
Integration tests for hifi.simulation.pipeline.run_pipeline (E1-T2, DJ-108).

Scenario: 11-sector synthetic signals (2 per sector = 22 tickers matching
SMOKE_UNIVERSE layout) with $500k capital.

Verifications:
- Sector cap respected: no sector exceeds max_sector in approved weights
- Stock cap respected: no ticker exceeds max_single_stock
- Risk-approved tickers have orders generated
- Blocked tickers do not appear in orders
- Order quantities are positive integers
- Commission cost < 0.5% of total notional
- PortfolioSnapshot is JSON-serializable and round-trips correctly
"""

from __future__ import annotations

import json

import pytest

from hifi.simulation.pipeline import PortfolioSnapshot, run_pipeline

# ---------------------------------------------------------------------------
# Scenario setup: 22 synthetic tickers (2 per sector)
# ---------------------------------------------------------------------------

_SECTORS = [
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Materials",
    "Real Estate",
    "Utilities",
]

# 2 tickers per sector, alternating Buy / Hold
_SIGNALS = []
_PRICES: dict[str, float] = {}
for i, sector in enumerate(_SECTORS):
    t1 = f"T{i * 2:02d}A"
    t2 = f"T{i * 2:02d}B"
    _SIGNALS.append({"ticker": t1, "decision": "Buy",  "confidence": 0.75, "sector": sector})
    _SIGNALS.append({"ticker": t2, "decision": "Hold", "confidence": 0.55, "sector": sector})
    _PRICES[t1] = 100.0 + i * 10
    _PRICES[t2] = 100.0 + i * 10

# OHLCV: 60 rows of flat prices (enough for correlation; VaR returns 0 on flat)
_OHLCV: dict[str, list[dict]] = {}
for ticker, price in _PRICES.items():
    _OHLCV[ticker] = [{"date": f"2022-{(d // 20) + 1:02d}-{(d % 20) + 1:02d}", "close": price}
                      for d in range(60)]

_CAPITAL = 500_000.0
_PORTFOLIO_STATE = {
    "portfolio": {},
    "portfolio_value": _CAPITAL,
    "hwm_value": _CAPITAL,
    "holdings": {},
    "prices": _PRICES,
}
_CONSTRAINTS = {
    "max_single_stock": 0.05,
    "max_sector": 0.20,
    "min_position": 0.005,
    "capital": _CAPITAL,
    "current_capital": 0.0,
}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_snapshot() -> PortfolioSnapshot:
    return run_pipeline(_SIGNALS, _OHLCV, _PORTFOLIO_STATE, _CONSTRAINTS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_returns_snapshot(pipeline_snapshot):
    assert isinstance(pipeline_snapshot, PortfolioSnapshot)


def test_pipeline_buy_count(pipeline_snapshot):
    # 11 Buy signals (one per sector)
    assert pipeline_snapshot.n_buy == 11


def test_pipeline_hold_count(pipeline_snapshot):
    assert pipeline_snapshot.n_hold == 11


def test_pipeline_sell_count(pipeline_snapshot):
    assert pipeline_snapshot.n_sell == 0


def test_pipeline_no_stock_cap_violations(pipeline_snapshot):
    max_w = _CONSTRAINTS["max_single_stock"]
    for ticker, w in pipeline_snapshot.weights.items():
        assert w <= max_w + 1e-6, (
            f"{ticker} weight {w:.4f} exceeds max_single_stock {max_w}"
        )


def test_pipeline_no_sector_cap_violations(pipeline_snapshot):
    max_s = _CONSTRAINTS["max_sector"]
    for sector, exposure in pipeline_snapshot.sector_exposure.items():
        assert exposure <= max_s + 1e-6, (
            f"Sector {sector!r} exposure {exposure:.4f} exceeds max_sector {max_s}"
        )


def test_pipeline_orders_positive_quantities(pipeline_snapshot):
    """Quantities are fractional by default (DJ-121); assert numeric and positive."""
    for order in pipeline_snapshot.orders:
        assert isinstance(order["quantity"], int | float)
        assert not isinstance(order["quantity"], bool)
        assert order["quantity"] > 0


def test_pipeline_commission_reasonable(pipeline_snapshot):
    total_notional = sum(
        float(o.get("estimated_value", 0.0)) for o in pipeline_snapshot.orders
    )
    total_commission = sum(
        float(o.get("estimated_cost", 0.0)) for o in pipeline_snapshot.orders
    )
    if total_notional > 0:
        commission_rate = total_commission / total_notional
        assert commission_rate < 0.005, (
            f"Commission rate {commission_rate:.4%} exceeds 0.5% of notional"
        )


def test_pipeline_weights_sum_at_most_one(pipeline_snapshot):
    total = sum(pipeline_snapshot.weights.values())
    assert total <= 1.0 + 1e-6, f"Weights sum {total:.6f} > 1.0"


def test_pipeline_json_round_trip(pipeline_snapshot):
    raw = pipeline_snapshot.to_json()
    data = json.loads(raw)
    assert data["n_buy"] == pipeline_snapshot.n_buy
    assert data["n_hold"] == pipeline_snapshot.n_hold
    assert abs(data["total_estimated_value"] - pipeline_snapshot.total_estimated_value) < 1.0


def test_pipeline_all_sectors_covered_in_signals(pipeline_snapshot):
    """Input signals span all 11 GICS sectors."""
    assert pipeline_snapshot.n_buy + pipeline_snapshot.n_hold == 22


def test_pipeline_orders_only_for_approved_tickers(pipeline_snapshot):
    """Order tickers must be a subset of approved signals from risk report."""
    approved = set(pipeline_snapshot.risk_report.get("approved_signals", []))
    order_tickers = {o["ticker"] for o in pipeline_snapshot.orders}
    unexpected = order_tickers - approved
    assert not unexpected, f"Orders generated for non-approved tickers: {unexpected}"
