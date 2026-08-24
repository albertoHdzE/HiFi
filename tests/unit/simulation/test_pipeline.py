"""
Unit tests for hifi.simulation.pipeline (E1-T1, DJ-108).

Tests PortfolioSnapshot dataclass and run_pipeline() with synthetic signals.
No LLMs, no network calls, no market data files.
"""

from __future__ import annotations

import json

import pytest

from hifi.simulation.pipeline import PortfolioSnapshot, run_pipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SIGNALS = [
    {"ticker": "AAPL", "decision": "Buy",  "confidence": 0.80, "sector": "Information Technology"},
    {"ticker": "CRM",  "decision": "Buy",  "confidence": 0.70, "sector": "Information Technology"},
    {"ticker": "JPM",  "decision": "Buy",  "confidence": 0.65, "sector": "Financials"},
    {"ticker": "BLK",  "decision": "Buy",  "confidence": 0.60, "sector": "Financials"},
    {"ticker": "UNH",  "decision": "Hold", "confidence": 0.55, "sector": "Health Care"},
    {"ticker": "XOM",  "decision": "Sell", "confidence": 0.50, "sector": "Energy"},
]

_PRICES = {
    "AAPL": 170.0,
    "CRM":  200.0,
    "JPM":  130.0,
    "BLK":  700.0,
    "UNH":  500.0,
    "XOM":   95.0,
}

# Minimal OHLCV: only a few rows, enough for risk checks to pass gracefully
_OHLCV = {
    t: [{"date": f"2022-01-{d:02d}", "close": float(p)}
        for d in range(1, 11)]
    for t, p in _PRICES.items()
}

_PORTFOLIO_STATE = {
    "portfolio": {},
    "portfolio_value": 100_000.0,
    "hwm_value": 100_000.0,
    "holdings": {},
    "prices": _PRICES,
}

_CONSTRAINTS = {
    "max_single_stock": 0.10,  # relaxed for small universe
    "max_sector": 0.40,        # relaxed: IT has 2 buys
    "min_position": 0.01,
    "capital": 100_000.0,
    "current_capital": 0.0,
}


# ---------------------------------------------------------------------------
# PortfolioSnapshot tests
# ---------------------------------------------------------------------------


def test_portfolio_snapshot_to_json_roundtrip():
    snap = PortfolioSnapshot(
        signals=[{"ticker": "AAPL", "decision": "Buy", "confidence": 0.8, "sector": "IT"}],
        weights={"AAPL": 0.05},
        risk_report={"approved_signals": ["AAPL"], "blocked_signals": []},
        orders=[{"ticker": "AAPL", "side": "BUY", "quantity": 29}],
        n_buy=1, n_hold=0, n_sell=0,
        sector_exposure={"IT": 0.05},
        total_estimated_value=4930.0,
        constraints={"capital": 100_000.0},
    )
    raw = snap.to_json()
    data = json.loads(raw)
    assert data["n_buy"] == 1
    assert data["weights"]["AAPL"] == pytest.approx(0.05)
    assert data["total_estimated_value"] == pytest.approx(4930.0)


# ---------------------------------------------------------------------------
# run_pipeline tests
# ---------------------------------------------------------------------------


def test_run_pipeline_returns_portfolio_snapshot():
    snap = run_pipeline(_SIGNALS, _OHLCV, _PORTFOLIO_STATE, _CONSTRAINTS)
    assert isinstance(snap, PortfolioSnapshot)


def test_run_pipeline_buy_hold_sell_counts():
    snap = run_pipeline(_SIGNALS, _OHLCV, _PORTFOLIO_STATE, _CONSTRAINTS)
    assert snap.n_buy == 4
    assert snap.n_hold == 1
    assert snap.n_sell == 1


def test_run_pipeline_only_buy_signals_weighted():
    """compose_portfolio (long_only) should only include Buy decisions."""
    snap = run_pipeline(_SIGNALS, _OHLCV, _PORTFOLIO_STATE, _CONSTRAINTS)
    # Hold (UNH) and Sell (XOM) should not appear in weights
    assert "UNH" not in snap.weights
    assert "XOM" not in snap.weights


def test_run_pipeline_weights_are_positive():
    snap = run_pipeline(_SIGNALS, _OHLCV, _PORTFOLIO_STATE, _CONSTRAINTS)
    for ticker, w in snap.weights.items():
        assert w > 0, f"{ticker} has non-positive weight {w}"


def test_run_pipeline_orders_are_positive_quantities():
    snap = run_pipeline(_SIGNALS, _OHLCV, _PORTFOLIO_STATE, _CONSTRAINTS)
    for order in snap.orders:
        assert order["quantity"] > 0
        assert order["side"] in ("BUY", "SELL")


def test_run_pipeline_sector_exposure_keys():
    snap = run_pipeline(_SIGNALS, _OHLCV, _PORTFOLIO_STATE, _CONSTRAINTS)
    # Only sectors of approved Buy signals appear in sector_exposure
    for sector in snap.sector_exposure:
        assert sector in ("Information Technology", "Financials")


def test_run_pipeline_json_serializable():
    snap = run_pipeline(_SIGNALS, _OHLCV, _PORTFOLIO_STATE, _CONSTRAINTS)
    raw = snap.to_json()
    data = json.loads(raw)
    assert "weights" in data
    assert "orders" in data
    assert "risk_report" in data


def test_run_pipeline_all_cash_no_portfolio():
    """Starting from all cash (current_capital=0) should generate BUY orders."""
    snap = run_pipeline(_SIGNALS, _OHLCV, _PORTFOLIO_STATE, _CONSTRAINTS)
    buy_orders = [o for o in snap.orders if o["side"] == "BUY"]
    assert len(buy_orders) > 0


def test_run_pipeline_empty_signals():
    """No signals → empty weights, no orders."""
    snap = run_pipeline([], {}, _PORTFOLIO_STATE, _CONSTRAINTS)
    assert snap.weights == {}
    assert snap.orders == []
    assert snap.n_buy == 0


def test_run_pipeline_risk_report_present():
    snap = run_pipeline(_SIGNALS, _OHLCV, _PORTFOLIO_STATE, _CONSTRAINTS)
    rr = snap.risk_report
    assert "approved_signals" in rr
    assert "blocked_signals" in rr
    assert "var_95" in rr
    assert "var_99" in rr
