"""
Unit tests for the technical indicator engine (P2-E3).

Each test verifies a specific indicator formula against manually computed
expected values or well-established mathematical properties. The conftest
synthetic_ohlcv fixture provides a 252-bar GBM series with seed=42.

Tickets covered: P2-E3-T4 through P2-E3-T11.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from hifi.data.schemas import OHLCVBar, ProvenanceRecord
from hifi.engines.technical import (
    compute_technical_indicators,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW_DT = __import__("datetime").datetime(2024, 1, 15, 12, 0, 0)
_PROV = ProvenanceRecord(source="test", fetched_at=_NOW_DT)


def _make_bar(dt: date, close: float, adj: float | None = None) -> OHLCVBar:
    """Create a valid OHLCV bar anchored around close."""
    return OHLCVBar(
        ticker="TEST",
        date=dt,
        open=close * 0.995,
        high=close * 1.010,
        low=close * 0.985,
        close=close,
        volume=1_000_000.0,
        adjusted_close=adj,
        source="test",
    )


def _bars_from_closes(closes: list[float], start: date | None = None) -> list[OHLCVBar]:
    """Build a dated bar list from a plain list of close prices."""
    base = start or date(2023, 1, 2)
    return [_make_bar(base + timedelta(days=i), c) for i, c in enumerate(closes)]


def _gbm_bars(n: int, seed: int = 42, daily_sigma: float = 0.01) -> list[OHLCVBar]:
    """Generate n GBM price bars for use in indicator tests."""
    rng = np.random.default_rng(seed=seed)
    log_returns = rng.normal(0.0, daily_sigma, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    base = date(2023, 1, 2)
    bars = []
    for i, c in enumerate(close):
        hi = c * (1 + abs(rng.normal(0, 0.005)))
        lo = c * (1 - abs(rng.normal(0, 0.005)))
        op = lo + rng.random() * (hi - lo)
        # Ensure OHLCV constraints
        hi = max(hi, c, op)
        lo = min(lo, c, op)
        bars.append(
            OHLCVBar(
                ticker="TEST",
                date=base + timedelta(days=i),
                open=float(op),
                high=float(hi),
                low=float(lo),
                close=float(c),
                volume=1_000_000.0,
                source="test",
            )
        )
    return bars


# ---------------------------------------------------------------------------
# T4: SMA
# ---------------------------------------------------------------------------


class TestSMA:
    def test_sma_window3_on_5_bars(self):
        """T4: SMA(3) on closes [10, 11, 12, 13, 14] == (12+13+14)/3 == 13.0."""
        closes = [10.0, 11.0, 12.0, 13.0, 14.0]
        bars = _bars_from_closes(closes)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=3)
        assert result.sma == pytest.approx(13.0)

    def test_sma_returns_none_when_bars_fewer_than_window(self):
        """SMA is None when len(bars) < window."""
        bars = _bars_from_closes([10.0, 11.0, 12.0])
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=5)
        assert result.sma is None

    def test_sma_equals_close_for_single_valid_bar_window1(self):
        """SMA(1) == last close."""
        bars = _bars_from_closes([42.0, 43.0, 44.0])
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=1)
        assert result.sma == pytest.approx(44.0)

    def test_sma_excludes_bars_after_as_of_date(self):
        """Bars after as_of_date must not affect the SMA."""
        closes = [10.0, 11.0, 12.0, 13.0, 14.0]
        bars = _bars_from_closes(closes)
        as_of = bars[2].date  # only first 3 bars
        result = compute_technical_indicators(bars, as_of, window=3)
        # SMA(3) of [10, 11, 12] == 11.0
        assert result.sma == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# T5: EMA
# ---------------------------------------------------------------------------


class TestEMA:
    def test_ema_of_constant_price_series_equals_constant(self):
        """T5: EMA of a constant-price series equals the constant."""
        bars = _bars_from_closes([100.0] * 25)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.ema == pytest.approx(100.0, rel=1e-6)

    def test_ema_returns_none_when_bars_fewer_than_window(self):
        """EMA is None when len(bars) < window."""
        bars = _bars_from_closes([100.0] * 10)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=15)
        assert result.ema is None

    def test_ema_more_sensitive_to_recent_prices_than_sma(self):
        """EMA should be closer to the last price than SMA for a rising series."""
        closes = list(range(1, 26))  # [1, 2, ..., 25], rising
        bars = _bars_from_closes([float(c) for c in closes])
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.ema is not None
        assert result.sma is not None
        # EMA is more recent-weighted, so it should exceed SMA in a rising series
        assert result.ema > result.sma


# ---------------------------------------------------------------------------
# T6 / T7: RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_rsi_in_valid_range(self):
        """T6: RSI is in [0.0, 100.0] for a GBM series with >= 15 bars."""
        bars = _gbm_bars(252)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.rsi is not None
        assert 0.0 <= result.rsi <= 100.0

    def test_rsi_near_100_for_monotonically_rising(self):
        """T7: RSI approaches 100 for a series with no down days."""
        closes = [100.0 + i for i in range(50)]  # 50 strictly rising bars
        bars = _bars_from_closes(closes)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.rsi is not None
        assert result.rsi > 90.0, f"Expected RSI near 100, got {result.rsi}"

    def test_rsi_near_0_for_monotonically_falling(self):
        """T7: RSI approaches 0 for a series with no up days."""
        closes = [200.0 - i for i in range(50)]  # 50 strictly falling bars
        bars = _bars_from_closes(closes)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.rsi is not None
        assert result.rsi < 10.0, f"Expected RSI near 0, got {result.rsi}"

    def test_rsi_none_when_fewer_than_15_bars(self):
        """T10 (RSI): None when bars < 15."""
        bars = _bars_from_closes([100.0 + i for i in range(10)])
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.rsi is None


# ---------------------------------------------------------------------------
# T8: Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_bollinger_upper_ge_mid_ge_lower(self):
        """T8: Bollinger upper >= mid >= lower for any valid series."""
        bars = _gbm_bars(50)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.bb_upper is not None
        assert result.bb_mid is not None
        assert result.bb_lower is not None
        assert result.bb_upper >= result.bb_mid
        assert result.bb_mid >= result.bb_lower

    def test_bollinger_bands_collapse_for_constant_price(self):
        """Upper == mid == lower when all prices are identical (zero std dev)."""
        bars = _bars_from_closes([100.0] * 25)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.bb_upper == pytest.approx(result.bb_mid, abs=1e-9)
        assert result.bb_lower == pytest.approx(result.bb_mid, abs=1e-9)

    def test_bollinger_none_when_bars_fewer_than_window(self):
        """T10 (Bollinger): None when bars < window."""
        bars = _bars_from_closes([100.0 + i for i in range(10)])
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.bb_upper is None
        assert result.bb_mid is None
        assert result.bb_lower is None


# ---------------------------------------------------------------------------
# T9: ATR
# ---------------------------------------------------------------------------


class TestATR:
    def test_atr_is_strictly_positive(self):
        """T9: ATR is strictly positive for any valid OHLCV series with >= 15 bars."""
        bars = _gbm_bars(50)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.atr is not None
        assert result.atr > 0.0

    def test_atr_none_when_fewer_than_15_bars(self):
        """T10 (ATR): None when bars < 15."""
        bars = _gbm_bars(10)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.atr is None


# ---------------------------------------------------------------------------
# T10: All None when bars < required window (global)
# ---------------------------------------------------------------------------


class TestAllNoneWhenInsufficientData:
    def test_all_none_for_single_bar(self):
        """All indicators return None with a 1-bar series."""
        bars = _bars_from_closes([100.0])
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.sma is None
        assert result.ema is None
        assert result.rsi is None
        assert result.macd is None
        assert result.macd_signal is None
        assert result.macd_hist is None
        assert result.bb_upper is None
        assert result.bb_mid is None
        assert result.bb_lower is None
        assert result.atr is None

    def test_all_none_for_5_bars_default_window(self):
        """T10: All indicators return None with 5 bars (all windows require more)."""
        bars = _bars_from_closes([100.0 + i for i in range(5)])
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.sma is None
        assert result.ema is None
        assert result.rsi is None
        assert result.macd is None
        assert result.bb_upper is None
        assert result.atr is None

    def test_macd_none_when_fewer_than_35_bars(self):
        """MACD is None when bars < 35 (needs 26+9 for signal line)."""
        bars = _gbm_bars(30)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.macd is None
        assert result.macd_signal is None
        assert result.macd_hist is None

    def test_macd_computes_with_35_or_more_bars(self):
        """MACD returns non-None values with 35+ bars."""
        bars = _gbm_bars(50)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.macd is not None
        assert result.macd_signal is not None
        assert result.macd_hist is not None


# ---------------------------------------------------------------------------
# T11: adjusted_close used when present
# ---------------------------------------------------------------------------


class TestAdjustedClosePreference:
    def test_sma_uses_adjusted_close_when_present(self):
        """T11: When adjusted_close differs from close, the SMA uses adjusted_close."""
        base = date(2023, 1, 2)
        # close = 100; adjusted_close = 90 (simulating a 10% split adjustment)
        bars = [
            OHLCVBar(
                ticker="TEST",
                date=base + timedelta(days=i),
                open=99.0,
                high=101.0,
                low=98.0,
                close=100.0,
                volume=1_000_000.0,
                adjusted_close=90.0,
                source="test",
            )
            for i in range(25)
        ]
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        # SMA should use adjusted_close=90.0, not close=100.0
        assert result.sma == pytest.approx(90.0, rel=1e-6)

    def test_sma_falls_back_to_close_when_adjusted_close_absent(self):
        """T11: When adjusted_close is None, SMA uses close."""
        bars = [_make_bar(date(2023, 1, 2) + timedelta(days=i), 100.0) for i in range(25)]
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        assert result.sma == pytest.approx(100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


class TestResultSerialisation:
    def test_result_is_json_serialisable(self):
        """TechnicalIndicatorsResult serialises to a JSON-safe dict."""
        import json
        bars = _gbm_bars(252)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        payload = result.model_dump()
        json.dumps(payload)

    def test_result_has_no_nan_values(self):
        """No NaN values in the serialised result."""
        bars = _gbm_bars(252)
        as_of = bars[-1].date
        result = compute_technical_indicators(bars, as_of, window=20)
        for key, val in result.model_dump().items():
            if val is not None:
                assert not math.isnan(val), f"Field {key} contains NaN"
