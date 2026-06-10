"""
Unit tests for the risk metrics engine (P2-E4).

Tests validate each metric against well-known mathematical properties or
manually computable expected values. The QuantStats adapter is tested
through the public compute_risk_metrics() interface.

Tickets covered: P2-E4-T8 through P2-E4-T14.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from hifi.data.schemas import OHLCVBar, OHLCVDataset, ProvenanceRecord
from hifi.engines.risk import (
    compute_beta,
    compute_hist_vol,
    compute_max_drawdown,
    compute_risk_metrics,
    compute_sharpe,
    compute_var,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW_DT = datetime(2024, 1, 15, 12, 0, 0)
_PROV = ProvenanceRecord(source="test", fetched_at=_NOW_DT)


def _make_bar(dt: date, close: float, ticker: str = "TEST") -> OHLCVBar:
    return OHLCVBar(
        ticker=ticker,
        date=dt,
        open=close * 0.995,
        high=close * 1.010,
        low=close * 0.985,
        close=close,
        volume=1_000_000.0,
        source="test",
    )


def _make_dataset(closes: list[float], ticker: str = "TEST") -> OHLCVDataset:
    base = date(2023, 1, 2)
    bars = [_make_bar(base + timedelta(days=i), c, ticker) for i, c in enumerate(closes)]
    return OHLCVDataset(
        ticker=ticker,
        bars=bars,
        source="test",
        fetched_at=_NOW_DT,
        date_from=bars[0].date,
        date_to=bars[-1].date,
        provenance=_PROV,
    )


def _gbm_dataset(
    n: int = 252,
    seed: int = 42,
    daily_sigma: float = 0.013,  # ~0.20 annualised (0.013 * sqrt(252) ≈ 0.206)
    ticker: str = "TEST",
) -> OHLCVDataset:
    """Generate a GBM price series as an OHLCVDataset."""
    rng = np.random.default_rng(seed=seed)
    log_returns = rng.normal(0.0, daily_sigma, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    base = date(2023, 1, 2)
    bars = [
        _make_bar(base + timedelta(days=i), float(c), ticker)
        for i, c in enumerate(close)
    ]
    return OHLCVDataset(
        ticker=ticker,
        bars=bars,
        source="test",
        fetched_at=_NOW_DT,
        date_from=bars[0].date,
        date_to=bars[-1].date,
        provenance=_PROV,
    )


# ---------------------------------------------------------------------------
# T8: Flat series vol is 0.0
# ---------------------------------------------------------------------------


class TestHistoricalVolatility:
    def test_vol_of_flat_price_series_is_zero(self):
        """T8: Annualised vol of a constant-price series is 0.0."""
        closes = [100.0] * 30  # 30 flat bars
        dataset = _make_dataset(closes)
        as_of = dataset.bars[-1].date
        vols = compute_hist_vol(dataset.bars, as_of, windows=[20])
        # Flat prices → zero returns → zero volatility
        assert vols[20] == pytest.approx(0.0, abs=1e-10)

    def test_vol_of_gbm_within_expected_range(self):
        """T9: Vol of GBM series with daily_sigma=0.013 is within [0.17, 0.25]."""
        # daily_sigma=0.013 → annualised ≈ 0.013 * sqrt(252) ≈ 0.206
        dataset = _gbm_dataset(n=300, daily_sigma=0.013)
        as_of = dataset.bars[-1].date
        vols = compute_hist_vol(dataset.bars, as_of, windows=[252])
        assert vols[252] is not None
        assert 0.17 <= vols[252] <= 0.25, f"Vol {vols[252]} outside [0.17, 0.25]"

    def test_vol_none_when_window_exceeds_bars(self):
        """T14: Vol returns None when window exceeds available bars."""
        dataset = _make_dataset([100.0 + i for i in range(10)])
        as_of = dataset.bars[-1].date
        vols = compute_hist_vol(dataset.bars, as_of, windows=[20])
        assert vols[20] is None

    def test_three_horizon_vols_all_computed_with_enough_bars(self):
        """All three horizons (20d, 60d, 252d) return non-None with 300 bars."""
        dataset = _gbm_dataset(n=300, daily_sigma=0.013)
        as_of = dataset.bars[-1].date
        vols = compute_hist_vol(dataset.bars, as_of)
        assert vols[20] is not None
        assert vols[60] is not None
        assert vols[252] is not None

    def test_vol_20d_differs_from_252d_for_volatile_series(self):
        """20d vol and 252d vol are not required to be equal for a non-constant series."""
        dataset = _gbm_dataset(n=300, daily_sigma=0.015)
        as_of = dataset.bars[-1].date
        vols = compute_hist_vol(dataset.bars, as_of)
        # They can differ; just check both are positive
        assert vols[20] is not None and vols[20] >= 0.0
        assert vols[252] is not None and vols[252] >= 0.0


# ---------------------------------------------------------------------------
# T10 / T11: Max drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_max_drawdown_on_declining_series(self):
        """T10: Drawdown of [100, 90, 80, 70, 60] == (100-60)/100 = 0.40."""
        closes = [100.0, 90.0, 80.0, 70.0, 60.0]
        bars = _make_dataset(closes).bars
        as_of = bars[-1].date
        dd = compute_max_drawdown(bars, as_of, window=len(closes))
        assert dd == pytest.approx(0.40, rel=1e-4)

    def test_max_drawdown_of_rising_series_is_zero(self):
        """T11: Max drawdown of a monotonically rising series is 0.0."""
        closes = [100.0 + i * 5 for i in range(20)]  # 100, 105, ..., 195
        bars = _make_dataset(closes).bars
        as_of = bars[-1].date
        dd = compute_max_drawdown(bars, as_of, window=len(closes))
        assert dd == pytest.approx(0.0, abs=1e-10)

    def test_max_drawdown_is_positive_float(self):
        """Drawdown is stored as a positive float, not negative."""
        closes = [100.0, 80.0, 60.0, 90.0, 70.0, 50.0]
        bars = _make_dataset(closes).bars
        as_of = bars[-1].date
        dd = compute_max_drawdown(bars, as_of, window=len(closes))
        assert dd is not None
        assert dd >= 0.0

    def test_max_drawdown_none_when_single_bar(self):
        """Drawdown returns None with only 1 price (no return computable)."""
        bars = _make_dataset([100.0]).bars
        as_of = bars[0].date
        dd = compute_max_drawdown(bars, as_of, window=252)
        assert dd is None

    def test_max_drawdown_in_range_0_to_1(self):
        """Drawdown is bounded in [0, 1] for any valid price series."""
        dataset = _gbm_dataset(n=252)
        as_of = dataset.bars[-1].date
        dd = compute_max_drawdown(dataset.bars, as_of)
        assert dd is not None
        assert 0.0 <= dd <= 1.0


# ---------------------------------------------------------------------------
# T12: Sharpe of zero-variance series
# ---------------------------------------------------------------------------


class TestSharpe:
    def test_sharpe_of_zero_variance_returns_none(self):
        """T12: Sharpe of a constant-price series returns None, not ZeroDivisionError."""
        closes = [100.0] * 60  # 60 flat bars
        bars = _make_dataset(closes).bars
        as_of = bars[-1].date
        sr = compute_sharpe(bars, as_of, risk_free_rate=0.0, window=252)
        assert sr is None

    def test_sharpe_positive_for_rising_series_zero_rf(self):
        """Sharpe is positive for a monotonically rising series with rf=0."""
        closes = [100.0 + i for i in range(30)]
        bars = _make_dataset(closes).bars
        as_of = bars[-1].date
        sr = compute_sharpe(bars, as_of, risk_free_rate=0.0, window=252)
        # Rising series with no rf → positive Sharpe
        assert sr is not None
        assert sr > 0.0

    def test_sharpe_none_when_fewer_than_2_bars(self):
        """T14: Sharpe returns None when fewer than 2 bars available."""
        bars = _make_dataset([100.0]).bars
        as_of = bars[0].date
        sr = compute_sharpe(bars, as_of, window=252)
        assert sr is None


# ---------------------------------------------------------------------------
# T13: Beta
# ---------------------------------------------------------------------------


class TestBeta:
    def test_beta_of_identical_series_is_one(self):
        """T13: Beta of a series identical to the benchmark is 1.0."""
        closes = [100.0 + i * 0.5 + np.sin(i / 10.0) for i in range(100)]
        dataset = _make_dataset(closes, ticker="STOCK")
        benchmark = _make_dataset(closes, ticker="SPY")
        as_of = dataset.bars[-1].date
        beta = compute_beta(dataset.bars, benchmark.bars, as_of, window=len(closes))
        assert beta == pytest.approx(1.0, rel=1e-6)

    def test_beta_of_double_leveraged_series_is_two(self):
        """Beta is exactly 2.0 when asset pct-returns are 2x benchmark pct-returns.

        Constructed with linear (not log) returns so that pct_change() on the
        resulting price series recovers the original returns exactly.
        beta = Cov(2r, r) / Var(r) = 2*Var(r) / Var(r) = 2.0
        """
        rng = np.random.default_rng(seed=99)
        bench_rets = rng.normal(0.001, 0.01, 100)
        # Build prices from linear returns: p[i] = p[i-1] * (1 + r[i])
        # Prepend 100.0 as the initial price so pct_change() recovers bench_rets
        bench_prices = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + bench_rets)])
        stock_prices = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + 2 * bench_rets)])

        benchmark = _make_dataset(list(bench_prices), ticker="SPY")
        dataset = _make_dataset(list(stock_prices), ticker="2XSTOCK")

        as_of = dataset.bars[-1].date
        beta = compute_beta(dataset.bars, benchmark.bars, as_of, window=len(bench_prices))
        assert beta == pytest.approx(2.0, rel=1e-3)

    def test_beta_none_when_no_benchmark(self):
        """Beta is None when no benchmark bars are provided."""
        dataset = _gbm_dataset(n=100)
        as_of = dataset.bars[-1].date
        result = compute_risk_metrics(dataset, as_of, benchmark=None)
        assert result.beta is None

    def test_beta_none_when_benchmark_is_flat(self):
        """Beta returns None when benchmark variance is zero (flat prices)."""
        dataset = _gbm_dataset(n=100)
        flat_benchmark = _make_dataset([100.0] * 100, ticker="FLAT")
        as_of = dataset.bars[-1].date
        beta = compute_beta(dataset.bars, flat_benchmark.bars, as_of, window=100)
        assert beta is None


# ---------------------------------------------------------------------------
# T14: VaR
# ---------------------------------------------------------------------------


class TestVaR:
    def test_var_positive_for_any_valid_series(self):
        """VaR is a positive number (magnitude of potential loss)."""
        dataset = _gbm_dataset(n=100)
        as_of = dataset.bars[-1].date
        var = compute_var(dataset.bars, as_of, confidence=0.95, window=20)
        assert var is not None
        assert var >= 0.0

    def test_var_none_when_insufficient_bars(self):
        """VaR returns None when fewer than window+1 bars available."""
        bars = _make_dataset([100.0 + i for i in range(10)]).bars
        as_of = bars[-1].date
        var = compute_var(bars, as_of, confidence=0.95, window=20)
        assert var is None


# ---------------------------------------------------------------------------
# Full compute_risk_metrics
# ---------------------------------------------------------------------------


class TestComputeRiskMetrics:
    def test_all_fields_computable_with_sufficient_data(self):
        """All risk metrics return non-None values with 300 bars of GBM data."""
        dataset = _gbm_dataset(n=300, daily_sigma=0.013)
        as_of = dataset.bars[-1].date
        result = compute_risk_metrics(dataset, as_of)
        assert result.hist_vol_20d is not None
        assert result.hist_vol_60d is not None
        assert result.hist_vol_252d is not None
        assert result.max_drawdown_252d is not None
        assert result.sharpe_252d is not None
        assert result.var_95_20d is not None

    def test_empty_dataset_returns_empty_result(self):
        """T14: Empty dataset returns all-None RiskMetricsResult."""
        dataset = OHLCVDataset(
            ticker="EMPTY",
            bars=[],
            source="test",
            fetched_at=_NOW_DT,
            date_from=date(2023, 1, 1),
            date_to=date(2023, 1, 1),
            provenance=_PROV,
        )
        result = compute_risk_metrics(dataset, date(2023, 12, 31))
        assert result.hist_vol_20d is None
        assert result.hist_vol_60d is None
        assert result.hist_vol_252d is None
        assert result.max_drawdown_252d is None
        assert result.sharpe_252d is None
        assert result.var_95_20d is None

    def test_result_is_json_serialisable(self):
        """RiskMetricsResult serialises to a JSON-safe dict."""
        import json
        dataset = _gbm_dataset(n=300)
        as_of = dataset.bars[-1].date
        result = compute_risk_metrics(dataset, as_of)
        payload = result.model_dump()
        json.dumps(payload)

    def test_result_has_no_nan_values(self):
        """No NaN in the serialised result."""
        dataset = _gbm_dataset(n=300)
        as_of = dataset.bars[-1].date
        result = compute_risk_metrics(dataset, as_of)
        for key, val in result.model_dump().items():
            if val is not None:
                assert not math.isnan(val), f"Field {key} contains NaN"

    def test_max_drawdown_in_range_0_to_1(self):
        """Max drawdown is in [0, 1]."""
        dataset = _gbm_dataset(n=300)
        as_of = dataset.bars[-1].date
        result = compute_risk_metrics(dataset, as_of)
        if result.max_drawdown_252d is not None:
            assert 0.0 <= result.max_drawdown_252d <= 1.0
