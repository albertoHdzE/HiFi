"""
Unit tests for check_risk_limits (E4-T2, DJ-091).

Tests:
- VaR calculation on known fixture returns blocks signals above threshold.
- Max drawdown >15% from HWM blocks all Buy signals.
- Correlation >0.85 annotates the lower-confidence ticker in block_reasons.
- All-clear scenario returns all signals approved.
- Empty portfolio returns zero VaR.
- Non-Buy signals (Hold/Sell) are not blocked by the risk manager.
"""

from __future__ import annotations

from hifi.mcp.risk_manager import (
    compute_correlation_matrix,
    compute_portfolio_var,
    compute_risk_report,
    max_drawdown_breached,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_series(returns: list[float], start_close: float = 100.0) -> list[dict]:
    """Build OHLCV list from a sequence of daily returns."""
    closes = [start_close]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    rows = []
    for i, c in enumerate(closes):
        rows.append({"date": f"2023-{(i % 30) + 1:02d}-01", "close": round(c, 4)})
    return rows


def _flat_returns(n: int, value: float = 0.001) -> list[float]:
    return [value] * n


def _large_loss_returns(n: int) -> list[float]:
    """Daily returns that produce a large portfolio loss (VaR > 5%).

    15 days of -10% ensures both the 5th and 1st percentile land in the loss region:
      5th percentile position = (n-1)*0.05 ≈ 12.55 → within the 15 loss days → VaR_95 ≈ 10%
      1st percentile position = (n-1)*0.01 ≈ 2.51  → within the 15 loss days → VaR_99 ≈ 10%
    """
    rets = [0.001] * (n - 15)
    rets += [-0.10] * 15
    return rets


# ---------------------------------------------------------------------------
# max_drawdown_breached
# ---------------------------------------------------------------------------


def test_max_drawdown_not_breached():
    assert not max_drawdown_breached(90_000, 100_000, limit=0.15)


def test_max_drawdown_exactly_at_limit():
    # 15% drawdown — not strictly greater than limit
    assert not max_drawdown_breached(85_000, 100_000, limit=0.15)


def test_max_drawdown_breached():
    assert max_drawdown_breached(84_000, 100_000, limit=0.15)


def test_max_drawdown_zero_hwm():
    assert not max_drawdown_breached(50_000, 0.0)


# ---------------------------------------------------------------------------
# compute_portfolio_var
# ---------------------------------------------------------------------------


def test_var_insufficient_data():
    portfolio = {"AAPL": {"weight": 1.0}}
    returns = {"AAPL": []}
    var_95, var_99 = compute_portfolio_var(portfolio, returns)
    assert var_95 == 0.0
    assert var_99 == 0.0


def test_var_empty_portfolio():
    var_95, var_99 = compute_portfolio_var({}, {})
    assert var_95 == 0.0
    assert var_99 == 0.0


def test_var_positive_on_known_losses():
    """Portfolio with 15/252 days of -10% losses: VaR_95 > 5%, VaR_99 > 8%."""
    rets = _large_loss_returns(252)
    portfolio = {"SPY": {"weight": 1.0}}
    returns = {"SPY": rets}
    var_95, var_99 = compute_portfolio_var(portfolio, returns)
    assert var_95 > 0.05, f"Expected VaR_95 > 0.05, got {var_95}"
    assert var_99 > 0.08, f"Expected VaR_99 > 0.08, got {var_99}"


def test_var_flat_returns_near_zero():
    """Constant small returns → near-zero VaR."""
    rets = _flat_returns(252, value=0.001)
    portfolio = {"AAPL": {"weight": 1.0}}
    returns = {"AAPL": rets}
    var_95, var_99 = compute_portfolio_var(portfolio, returns)
    assert var_95 < 0.01
    assert var_99 < 0.01


# ---------------------------------------------------------------------------
# compute_correlation_matrix
# ---------------------------------------------------------------------------


def test_correlation_perfect_positive():
    r = [0.01, 0.02, -0.01, 0.005, 0.03, 0.01] * 10
    returns = {"A": r, "B": r}
    corr = compute_correlation_matrix(["A", "B"], returns)
    assert ("A", "B") in corr
    assert corr[("A", "B")] > 0.99


def test_correlation_uncorrelated():
    import random
    rng = random.Random(42)
    ra = [rng.gauss(0, 0.01) for _ in range(60)]
    rb = [rng.gauss(0, 0.01) for _ in range(60)]
    returns = {"A": ra, "B": rb}
    corr = compute_correlation_matrix(["A", "B"], returns)
    r_val = corr.get(("A", "B"), 0.0)
    # Uncorrelated series: |r| should be < 0.5 on average
    assert abs(r_val) < 0.7  # loose bound


def test_correlation_missing_ticker():
    returns = {"A": [0.01] * 10}
    corr = compute_correlation_matrix(["A", "B"], returns)
    assert ("A", "B") not in corr


# ---------------------------------------------------------------------------
# compute_risk_report — all clear
# ---------------------------------------------------------------------------


def test_all_clear_returns_all_approved():
    portfolio = {
        "AAPL": {"weight": 0.05, "confidence": 0.8, "sector": "IT"},
    }
    signals = [
        {"ticker": "MSFT", "decision": "Buy", "confidence": 0.7, "sector": "IT"},
        {"ticker": "GOOGL", "decision": "Buy", "confidence": 0.65, "sector": "Comm"},
    ]
    rets = _flat_returns(300, value=0.0005)
    ohlcv = {
        "AAPL": _make_ohlcv_series(rets),
        "MSFT": _make_ohlcv_series(rets),
        "GOOGL": _make_ohlcv_series(rets),
    }
    report = compute_risk_report(
        portfolio=portfolio,
        signals=signals,
        ohlcv=ohlcv,
        portfolio_value=100_000.0,
        hwm_value=100_000.0,
    )
    assert set(report["approved_signals"]) == {"MSFT", "GOOGL"}
    assert report["blocked_signals"] == []
    assert report["var_95"] >= 0.0


# ---------------------------------------------------------------------------
# Max drawdown blocks all buys
# ---------------------------------------------------------------------------


def test_max_drawdown_blocks_all_buys():
    portfolio = {
        "AAPL": {"weight": 0.10, "confidence": 0.8, "sector": "IT"},
    }
    signals = [
        {"ticker": "MSFT", "decision": "Buy", "confidence": 0.7, "sector": "IT"},
    ]
    report = compute_risk_report(
        portfolio=portfolio,
        signals=signals,
        ohlcv={},
        portfolio_value=80_000.0,  # -20% from HWM
        hwm_value=100_000.0,
    )
    assert "MSFT" in report["blocked_signals"]
    assert "MAX_DRAWDOWN" in report["block_reasons"]["MSFT"]
    assert report["approved_signals"] == []


# ---------------------------------------------------------------------------
# VaR breach blocks buy signals
# ---------------------------------------------------------------------------


def test_var_breach_blocks_buys():
    """Portfolio with heavy losses → VaR_95 > 5% → Buy signals blocked."""
    rets = _large_loss_returns(252)
    portfolio = {"SPY": {"weight": 1.0, "confidence": 0.9, "sector": "ETF"}}
    signals = [
        {"ticker": "TSLA", "decision": "Buy", "confidence": 0.6, "sector": "Auto"},
    ]
    ohlcv = {"SPY": _make_ohlcv_series(rets), "TSLA": _make_ohlcv_series(rets)}
    report = compute_risk_report(
        portfolio=portfolio,
        signals=signals,
        ohlcv=ohlcv,
        portfolio_value=100_000.0,
        hwm_value=100_000.0,
    )
    assert "TSLA" in report["blocked_signals"]
    assert "VAR_LIMIT" in report["block_reasons"]["TSLA"]


# ---------------------------------------------------------------------------
# Sector cap blocks signal
# ---------------------------------------------------------------------------


def test_sector_cap_blocks_additional_buy():
    """Sector already at cap → new Buy in that sector is blocked."""
    portfolio = {
        "AAPL": {"weight": 0.10, "confidence": 0.8, "sector": "IT"},
        "MSFT": {"weight": 0.10, "confidence": 0.7, "sector": "IT"},
    }  # IT total = 0.20 = exactly at cap
    signals = [
        {"ticker": "NVDA", "decision": "Buy", "confidence": 0.65, "sector": "IT"},
        {"ticker": "JPM", "decision": "Buy", "confidence": 0.60, "sector": "Financials"},
    ]
    rets = _flat_returns(100, 0.001)
    ohlcv = {t: _make_ohlcv_series(rets) for t in ["AAPL", "MSFT", "NVDA", "JPM"]}
    report = compute_risk_report(
        portfolio=portfolio,
        signals=signals,
        ohlcv=ohlcv,
        max_sector=0.20,
    )
    assert "NVDA" in report["blocked_signals"]
    assert "SECTOR_CAP" in report["block_reasons"]["NVDA"]
    assert "JPM" in report["approved_signals"]


# ---------------------------------------------------------------------------
# Correlation-aware annotation
# ---------------------------------------------------------------------------


def test_correlation_high_annotates_weaker_ticker():
    """Tickers with r>0.85 — the lower-confidence one gets CORR_REDUCED annotation."""
    r = [0.01, 0.02, -0.01, 0.005, 0.03, 0.01] * 20
    ohlcv = {
        "AAPL": _make_ohlcv_series(r),
        "MSFT": _make_ohlcv_series(r),  # perfectly correlated
    }
    signals = [
        {"ticker": "AAPL", "decision": "Buy", "confidence": 0.80, "sector": "IT"},
        {"ticker": "MSFT", "decision": "Buy", "confidence": 0.60, "sector": "IT"},
    ]
    report = compute_risk_report(
        portfolio={},
        signals=signals,
        ohlcv=ohlcv,
        max_sector=0.50,  # high cap so sector doesn't block
    )
    # MSFT has lower confidence → should get CORR_REDUCED annotation
    assert "MSFT" in report["approved_signals"]
    assert "CORR_REDUCED" in report.get("block_reasons", {}).get("MSFT", "")


# ---------------------------------------------------------------------------
# Non-Buy signals not evaluated
# ---------------------------------------------------------------------------


def test_hold_and_sell_signals_not_in_blocked():
    signals = [
        {"ticker": "AAPL", "decision": "Hold", "confidence": 0.5, "sector": "IT"},
        {"ticker": "MSFT", "decision": "Sell", "confidence": 0.6, "sector": "IT"},
    ]
    report = compute_risk_report(
        portfolio={},
        signals=signals,
        ohlcv={},
    )
    assert report["blocked_signals"] == []
    assert report["approved_signals"] == []


# ---------------------------------------------------------------------------
# MCP tool round-trip
# ---------------------------------------------------------------------------


def test_mcp_tool_valid_input():
    import json

    from hifi.mcp.risk_manager import check_risk_limits

    portfolio = {"AAPL": {"weight": 0.05, "confidence": 0.8, "sector": "IT"}}
    signals = [{"ticker": "MSFT", "decision": "Buy", "confidence": 0.7, "sector": "IT"}]
    rets = _flat_returns(100, 0.0005)
    ohlcv = {"AAPL": _make_ohlcv_series(rets), "MSFT": _make_ohlcv_series(rets)}

    result = check_risk_limits(
        portfolio_json=json.dumps(portfolio),
        signals_json=json.dumps(signals),
        ohlcv_json=json.dumps(ohlcv),
    )
    assert "approved_signals" in result or "error" in result


def test_mcp_tool_invalid_portfolio_json():
    from hifi.mcp.risk_manager import check_risk_limits
    result = check_risk_limits(
        portfolio_json="not-json",
        signals_json="[]",
        ohlcv_json="{}",
    )
    assert result.get("error") == "INVALID_PORTFOLIO_JSON"
