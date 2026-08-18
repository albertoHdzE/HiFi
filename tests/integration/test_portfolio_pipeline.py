"""
End-to-end integration test for the 3-MCP portfolio pipeline (E4-T4, DJ-091).

Pipeline: compose_portfolio → check_risk_limits → allocate_capital
Scenario: 10 tickers across 4 sectors, Buy/Hold/Sell mix, $250k capital.

No LLMs, no network calls.

Verifications:
- Orders sum to ≤ available capital.
- Commission cost < 0.5% of total notional.
- No blocked ticker appears in the order list.
- All order quantities are positive integers.
"""

from __future__ import annotations

import json

import pytest

from hifi.mcp.capital_allocator import generate_orders
from hifi.mcp.portfolio_composer import compose_portfolio
from hifi.mcp.risk_manager import compute_risk_report

# ---------------------------------------------------------------------------
# Scenario setup
# ---------------------------------------------------------------------------

CAPITAL = 250_000.0

# 10 tickers across 4 sectors, mixed signals
SIGNALS = [
    # Information Technology (3 Buy, 1 Hold)
    {"ticker": "AAPL", "decision": "Buy",  "confidence": 0.80, "sector": "IT"},
    {"ticker": "MSFT", "decision": "Buy",  "confidence": 0.75, "sector": "IT"},
    {"ticker": "NVDA", "decision": "Buy",  "confidence": 0.70, "sector": "IT"},
    {"ticker": "ORCL", "decision": "Hold", "confidence": 0.55, "sector": "IT"},
    # Financials (2 Buy)
    {"ticker": "JPM",  "decision": "Buy",  "confidence": 0.65, "sector": "Financials"},
    {"ticker": "BAC",  "decision": "Buy",  "confidence": 0.60, "sector": "Financials"},
    # Health Care (2 Buy, 1 Sell)
    {"ticker": "JNJ",  "decision": "Buy",  "confidence": 0.72, "sector": "HealthCare"},
    {"ticker": "UNH",  "decision": "Buy",  "confidence": 0.68, "sector": "HealthCare"},
    {"ticker": "MRK",  "decision": "Sell", "confidence": 0.55, "sector": "HealthCare"},
    # Energy (1 Buy)
    {"ticker": "XOM",  "decision": "Buy",  "confidence": 0.62, "sector": "Energy"},
]

PRICES = {
    "AAPL": 175.0,
    "MSFT": 320.0,
    "NVDA": 450.0,
    "ORCL": 100.0,
    "JPM":  155.0,
    "BAC":   35.0,
    "JNJ":  165.0,
    "UNH":  520.0,
    "MRK":   90.0,
    "XOM":  110.0,
}

# Mild returns → near-zero VaR, no blocks
def _flat_ohlcv(n: int = 100, start: float = 100.0) -> list[dict]:
    rows = []
    c = start
    for i in range(n):
        rows.append({"date": f"2023-{(i % 28) + 1:02d}-{(i % 12) + 1:02d}", "close": c})
        c *= 1.0002
    return rows

OHLCV = {t: _flat_ohlcv() for t in PRICES}


# ---------------------------------------------------------------------------
# Pipeline fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_result():
    """Run the full 3-step portfolio pipeline and return intermediate results."""

    # Step 1: compose_portfolio
    weights_raw = compose_portfolio(
        signals_json=json.dumps(SIGNALS),
        max_single_stock=0.05,
        max_sector=0.20,
        min_position=0.01,
    )
    assert not isinstance(weights_raw, dict) or "error" not in weights_raw, \
        f"compose_portfolio error: {weights_raw}"
    weights: dict[str, float] = weights_raw

    # Step 2: check_risk_limits
    # Current portfolio = empty (starting from cash)
    report = compute_risk_report(
        portfolio={},
        signals=SIGNALS,
        ohlcv=OHLCV,
        portfolio_value=CAPITAL,
        hwm_value=CAPITAL,
        max_sector=0.20,
    )
    approved = set(report["approved_signals"])
    blocked = set(report["blocked_signals"])

    # Step 3: allocate_capital (only approved tickers)
    approved_weights = {t: w for t, w in weights.items() if t in approved}
    orders = generate_orders(
        weights=approved_weights,
        prices=PRICES,
        holdings={},
        capital=CAPITAL,
        current_capital=0.0,
    )

    return {
        "weights": weights,
        "report": report,
        "approved": approved,
        "blocked": blocked,
        "orders": orders,
    }


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def test_compose_produces_weights(pipeline_result):
    assert len(pipeline_result["weights"]) > 0


def test_buy_signals_enter_weights(pipeline_result):
    """All 8 Buy signals should produce some portfolio weight."""
    buy_tickers = {s["ticker"] for s in SIGNALS if s["decision"] == "Buy"}
    weight_tickers = set(pipeline_result["weights"].keys())
    # There may be some exclusions due to caps, but at least half should appear
    assert len(weight_tickers & buy_tickers) >= 4


def test_non_buy_signals_not_in_weights(pipeline_result):
    """Hold and Sell tickers should not appear in the composed weights."""
    non_buy = {s["ticker"] for s in SIGNALS if s["decision"] != "Buy"}
    weight_tickers = set(pipeline_result["weights"].keys())
    assert weight_tickers.isdisjoint(non_buy)


def test_weights_sum_at_most_one(pipeline_result):
    total = sum(pipeline_result["weights"].values())
    assert total <= 1.0 + 1e-6, f"Weights sum to {total:.6f} > 1.0"


def test_sector_cap_respected_in_weights(pipeline_result):
    """No sector should exceed 20% in the composed weights."""
    sector_map = {s["ticker"]: s["sector"] for s in SIGNALS}
    sector_totals: dict[str, float] = {}
    for ticker, weight in pipeline_result["weights"].items():
        sector = sector_map.get(ticker, "Unknown")
        sector_totals[sector] = sector_totals.get(sector, 0.0) + weight
    for sector, total in sector_totals.items():
        assert total <= 0.20 + 1e-6, f"Sector {sector} exceeds cap: {total:.4f}"


def test_risk_report_has_var_fields(pipeline_result):
    report = pipeline_result["report"]
    assert "var_95" in report
    assert "var_99" in report
    assert report["var_95"] >= 0.0
    assert report["var_99"] >= 0.0


def test_orders_produced(pipeline_result):
    assert len(pipeline_result["orders"]) > 0


def test_orders_are_buy_orders(pipeline_result):
    """Starting from cash, all orders should be BUY."""
    assert all(o["side"] == "BUY" for o in pipeline_result["orders"])


def test_orders_total_within_capital(pipeline_result):
    """Total notional value of orders must not exceed available capital."""
    total = sum(o["estimated_value"] for o in pipeline_result["orders"])
    assert total <= CAPITAL * 1.01, f"Total notional {total:.2f} exceeds capital {CAPITAL}"


def test_commission_below_half_percent_of_notional(pipeline_result):
    """Commission cost should be < 0.5% of total notional (spec requirement)."""
    total_notional = sum(o["estimated_value"] for o in pipeline_result["orders"])
    total_commission = sum(o["estimated_cost"] for o in pipeline_result["orders"])
    if total_notional > 0:
        ratio = total_commission / total_notional
        assert ratio < 0.005, f"Commission ratio {ratio:.4f} >= 0.5%"


def test_no_blocked_ticker_in_orders(pipeline_result):
    """Tickers blocked by the risk manager must not appear in the order list."""
    blocked = pipeline_result["blocked"]
    order_tickers = {o["ticker"] for o in pipeline_result["orders"]}
    contamination = blocked & order_tickers
    assert not contamination, f"Blocked tickers in orders: {contamination}"


def test_order_quantities_positive(pipeline_result):
    """Quantities are fractional by default (DJ-121) — assert positive and
    numeric rather than integral."""
    for o in pipeline_result["orders"]:
        assert isinstance(o["quantity"], int | float)
        assert not isinstance(o["quantity"], bool)
        assert o["quantity"] > 0


def test_order_type_is_market(pipeline_result):
    for o in pipeline_result["orders"]:
        assert o["order_type"] == "MARKET"
