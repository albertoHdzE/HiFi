"""
Unit tests for hifi.simulation.metrics (Phase 15 IC/IR/herding metrics).

Tests cover:
- buy_strength: Buy/Sell/Hold mapping, missing decision → None
- compute_ic: Spearman correlation, edge cases, length mismatch
- compute_ir: IR = mean/std, degenerate cases
- compute_herding_coefficient: fraction agreement
- forward_return_from_ohlcv: offline return computation
"""

from __future__ import annotations

import math

import pytest

from hifi.simulation.metrics import (
    ICResult,
    buy_strength,
    compute_herding_coefficient,
    compute_ic,
    compute_ir,
    forward_return_from_ohlcv,
)

# ---------------------------------------------------------------------------
# buy_strength
# ---------------------------------------------------------------------------


def _output(decision: str | None, confidence: float = 0.7) -> dict:
    return {
        "ensemble_decision": {
            "collective_decision": decision,
            "collective_confidence": confidence,
            "agreement": True,
        }
    }


def test_buy_strength_buy():
    assert buy_strength(_output("Buy", 0.8)) == pytest.approx(0.8)


def test_buy_strength_sell():
    assert buy_strength(_output("Sell", 0.6)) == pytest.approx(-0.6)


def test_buy_strength_hold():
    assert buy_strength(_output("Hold", 0.9)) == pytest.approx(0.0)


def test_buy_strength_none_decision():
    assert buy_strength(_output(None)) is None


def test_buy_strength_missing_decision_block():
    assert buy_strength({}) is None


def test_buy_strength_missing_confidence_defaults():
    out = {"ensemble_decision": {"collective_decision": "Buy"}}
    result = buy_strength(out)
    assert result == pytest.approx(0.5)  # defaults to 0.5


def test_buy_strength_buy_confidence_one():
    assert buy_strength(_output("Buy", 1.0)) == pytest.approx(1.0)


def test_buy_strength_sell_confidence_one():
    assert buy_strength(_output("Sell", 1.0)) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# compute_ic
# ---------------------------------------------------------------------------


def test_compute_ic_perfect_positive():
    """Perfectly monotone positive relationship → IC ≈ 1.0."""
    signals = [0.1, 0.3, 0.5, 0.7, 0.9]
    returns = [0.01, 0.03, 0.05, 0.07, 0.09]
    result = compute_ic(signals, returns)
    assert isinstance(result, ICResult)
    assert result.ic == pytest.approx(1.0)
    assert result.n_pairs == 5


def test_compute_ic_perfect_negative():
    """Perfectly monotone negative relationship → IC ≈ -1.0."""
    signals = [0.9, 0.7, 0.5, 0.3, 0.1]
    returns = [0.01, 0.03, 0.05, 0.07, 0.09]
    result = compute_ic(signals, returns)
    assert result.ic == pytest.approx(-1.0)


def test_compute_ic_zero_correlation():
    """Constant returns → IC is NaN (Spearman undefined for constant array)."""
    signals = [0.1, -0.3, 0.5, -0.7, 0.9]
    returns = [0.05, 0.05, 0.05, 0.05, 0.05]
    result = compute_ic(signals, returns)
    assert math.isnan(result.ic)


def test_compute_ic_returns_p_value():
    signals = [0.1, 0.3, 0.5, 0.7, 0.9]
    returns = [0.01, 0.03, 0.05, 0.07, 0.09]
    result = compute_ic(signals, returns)
    assert 0.0 <= result.p_value <= 1.0


def test_compute_ic_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        compute_ic([0.1, 0.2], [0.01])


def test_compute_ic_too_few_pairs_raises():
    with pytest.raises(ValueError, match="At least 2"):
        compute_ic([0.5], [0.05])


def test_compute_ic_minimum_two_pairs():
    result = compute_ic([0.1, 0.9], [0.01, 0.09])
    assert isinstance(result, ICResult)
    assert result.n_pairs == 2


# ---------------------------------------------------------------------------
# compute_ir
# ---------------------------------------------------------------------------


def test_compute_ir_basic():
    """IR = mean / std."""
    ic_series = [0.10, 0.12, 0.08, 0.11, 0.09]
    mean_ic = sum(ic_series) / len(ic_series)
    n = len(ic_series)
    std_ic = math.sqrt(sum((x - mean_ic) ** 2 for x in ic_series) / (n - 1))
    expected = mean_ic / std_ic
    assert compute_ir(ic_series) == pytest.approx(expected)


def test_compute_ir_single_value_returns_zero():
    assert compute_ir([0.05]) == 0.0


def test_compute_ir_empty_returns_zero():
    assert compute_ir([]) == 0.0


def test_compute_ir_constant_series_returns_zero():
    """All same IC → std=0 → IR=0 (no variation)."""
    assert compute_ir([0.05, 0.05, 0.05]) == 0.0


def test_compute_ir_sign_preserved():
    """Negative mean IC → negative IR."""
    ic_series = [-0.05, -0.07, -0.06]
    ir = compute_ir(ic_series)
    assert ir < 0.0


def test_compute_ir_two_values():
    result = compute_ir([0.1, 0.3])
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_herding_coefficient
# ---------------------------------------------------------------------------


def _outputs_with_agreement(flags: list[bool]) -> list[dict]:
    return [{"ensemble_decision": {"agreement": f, "collective_decision": "Buy"}} for f in flags]


def test_herding_all_agree():
    outputs = _outputs_with_agreement([True, True, True])
    assert compute_herding_coefficient(outputs) == pytest.approx(1.0)


def test_herding_none_agree():
    outputs = _outputs_with_agreement([False, False, False])
    assert compute_herding_coefficient(outputs) == pytest.approx(0.0)


def test_herding_half_agree():
    outputs = _outputs_with_agreement([True, True, False, False])
    assert compute_herding_coefficient(outputs) == pytest.approx(0.5)


def test_herding_empty_list():
    assert compute_herding_coefficient([]) == 0.0


def test_herding_missing_agreement_field():
    """Records without agreement field count as non-herded."""
    outputs = [{"ensemble_decision": {}}, {"ensemble_decision": {"agreement": True}}]
    assert compute_herding_coefficient(outputs) == pytest.approx(0.5)


def test_herding_missing_ensemble_decision():
    outputs = [{"ticker": "AAPL"}]
    assert compute_herding_coefficient(outputs) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# forward_return_from_ohlcv
# ---------------------------------------------------------------------------


def _make_ohlcv(dates: list[str], closes: list[float]):
    import pandas as pd

    idx = pd.to_datetime(dates)
    return pd.DataFrame({"Close": closes}, index=idx)


def test_forward_return_basic():
    """Price goes from 100 to 110 over 60 trading days → +10%."""
    dates = [f"2022-01-{d:02d}" for d in range(1, 10)]
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
    df = _make_ohlcv(dates, closes)
    # horizon=2 trading days from 2022-01-01 (iloc 0) → iloc 2 close=102
    result = forward_return_from_ohlcv(df, "2022-01-01", horizon_trading_days=2)
    assert result == pytest.approx(0.02)  # (102-100)/100


def test_forward_return_none_when_not_enough_data():
    """Horizon extends beyond available data → None."""
    dates = ["2022-01-03", "2022-01-04"]
    closes = [100.0, 102.0]
    df = _make_ohlcv(dates, closes)
    result = forward_return_from_ohlcv(df, "2022-01-03", horizon_trading_days=5)
    assert result is None


def test_forward_return_none_when_date_not_in_index():
    """as_of_date before all data → returns first available row's price."""
    dates = ["2022-01-05", "2022-01-06", "2022-01-07"]
    closes = [100.0, 110.0, 120.0]
    df = _make_ohlcv(dates, closes)
    # as_of_date 2022-01-04 is before index; first row at 2022-01-05 used
    result = forward_return_from_ohlcv(df, "2022-01-04", horizon_trading_days=1)
    assert result == pytest.approx(0.10)  # (110-100)/100


def test_forward_return_zero_start_price_returns_none():
    dates = ["2022-01-03", "2022-01-04", "2022-01-05"]
    closes = [0.0, 100.0, 105.0]
    df = _make_ohlcv(dates, closes)
    result = forward_return_from_ohlcv(df, "2022-01-03", horizon_trading_days=1)
    assert result is None


def test_forward_return_negative_return():
    dates = ["2022-01-03", "2022-01-04", "2022-01-05"]
    closes = [100.0, 95.0, 90.0]
    df = _make_ohlcv(dates, closes)
    result = forward_return_from_ohlcv(df, "2022-01-03", horizon_trading_days=1)
    assert result == pytest.approx(-0.05)
