"""
Unit tests for the fundamental and valuation engine (P2-E2).

Tests validate each formula against manually computed expected values.
All assertions derived by hand or using basic arithmetic so the test
itself is independent of the implementation.

Tickets covered: P2-E2-T4 through P2-E2-T10.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from hifi.data.schemas import (
    FundamentalsSnapshot,
    OHLCVBar,
    OHLCVDataset,
    ProvenanceRecord,
)
from hifi.engines.fundamental import (
    compute_financial_ratios,
    compute_growth_metrics,
    compute_valuation_context,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 15, 12, 0, 0)
_PROV = ProvenanceRecord(source="test", fetched_at=_NOW)


def _make_snapshot(**overrides) -> FundamentalsSnapshot:
    defaults: dict = dict(
        ticker="TEST",
        period_end=date(2023, 12, 31),
        revenue=100_000_000_000.0,
        net_income=20_000_000_000.0,
        total_assets=200_000_000_000.0,
        total_liabilities=80_000_000_000.0,
        total_equity=120_000_000_000.0,
        eps=10.0,
        pe_ratio=25.0,
        market_cap=2_500_000_000_000.0,
        source="test",
        fetched_at=_NOW,
        provenance=_PROV,
    )
    defaults.update(overrides)
    return FundamentalsSnapshot(**defaults)


def _make_bar(dt: date, close: float, ticker: str = "TEST") -> OHLCVBar:
    """Create a valid OHLCV bar anchored around close."""
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


def _make_dataset(bars: list[OHLCVBar], ticker: str = "TEST") -> OHLCVDataset:
    dates = [b.date for b in bars] if bars else [date(2024, 1, 15)]
    return OHLCVDataset(
        ticker=ticker,
        bars=bars,
        source="test",
        fetched_at=_NOW,
        date_from=min(dates),
        date_to=max(dates),
        provenance=_PROV,
    )


# ---------------------------------------------------------------------------
# P2-E2-T4 / T5 / T7: compute_financial_ratios
# ---------------------------------------------------------------------------


class TestComputeFinancialRatios:
    """Tests for compute_financial_ratios()."""

    def test_pe_is_price_divided_by_eps(self):
        """T4: P/E = 250.0 / 10.0 = 25.0."""
        snap = _make_snapshot(eps=10.0)
        result = compute_financial_ratios(snap, current_price=250.0)
        assert result.pe == pytest.approx(25.0)

    def test_pe_is_none_when_eps_is_none(self):
        """T5: P/E is None when eps is None."""
        snap = _make_snapshot(eps=None)
        result = compute_financial_ratios(snap, current_price=250.0)
        assert result.pe is None

    def test_pe_is_none_when_eps_is_zero(self):
        """T5: P/E is None when eps == 0."""
        snap = _make_snapshot(eps=0.0)
        result = compute_financial_ratios(snap, current_price=250.0)
        assert result.pe is None

    def test_pe_is_none_when_eps_is_negative(self):
        """T5: P/E is None when eps < 0 (negative earnings)."""
        snap = _make_snapshot(eps=-3.0)
        result = compute_financial_ratios(snap, current_price=250.0)
        assert result.pe is None

    def test_pb_ratio(self):
        """P/B = market_cap / total_equity = 2.5T / 120B ≈ 20.833."""
        snap = _make_snapshot(market_cap=2_500_000.0, total_equity=120_000.0)
        result = compute_financial_ratios(snap, current_price=250.0)
        assert result.pb == pytest.approx(2_500_000.0 / 120_000.0)

    def test_ps_ratio(self):
        """P/S = market_cap / revenue."""
        snap = _make_snapshot(market_cap=500_000.0, revenue=200_000.0)
        result = compute_financial_ratios(snap, current_price=250.0)
        assert result.ps == pytest.approx(500_000.0 / 200_000.0)

    def test_ps_is_none_when_revenue_is_zero(self):
        """P/S is None when revenue == 0 (positive denominator check)."""
        snap = _make_snapshot(revenue=0.0)
        result = compute_financial_ratios(snap, current_price=250.0)
        assert result.ps is None

    def test_ps_is_none_when_revenue_is_negative(self):
        """P/S is None when revenue < 0 (does not occur in practice but defended)."""
        snap = _make_snapshot(revenue=-1.0)
        result = compute_financial_ratios(snap, current_price=250.0)
        assert result.ps is None

    def test_roe(self):
        """T6: ROE = net_income / total_equity = 150 / 1000 = 0.15."""
        snap = _make_snapshot(net_income=150.0, total_equity=1000.0)
        result = compute_financial_ratios(snap, current_price=100.0)
        assert result.roe == pytest.approx(0.15)

    def test_roa(self):
        """ROA = net_income / total_assets = 20B / 200B = 0.10."""
        snap = _make_snapshot(net_income=20_000.0, total_assets=200_000.0)
        result = compute_financial_ratios(snap, current_price=100.0)
        assert result.roa == pytest.approx(0.10)

    def test_roa_is_none_when_assets_zero(self):
        """ROA is None when total_assets == 0."""
        snap = _make_snapshot(total_assets=0.0)
        result = compute_financial_ratios(snap, current_price=100.0)
        assert result.roa is None

    def test_debt_equity(self):
        """Debt/Equity = total_liabilities / total_equity."""
        snap = _make_snapshot(total_liabilities=80_000.0, total_equity=120_000.0)
        result = compute_financial_ratios(snap, current_price=100.0)
        assert result.debt_equity == pytest.approx(80_000.0 / 120_000.0)

    def test_debt_equity_is_none_when_equity_zero(self):
        """Debt/Equity is None when total_equity == 0."""
        snap = _make_snapshot(total_equity=0.0)
        result = compute_financial_ratios(snap, current_price=100.0)
        assert result.debt_equity is None

    def test_ev_ebitda_always_none_in_phase2(self):
        """EV/EBITDA is always None in Phase 2 (EBITDA not in snapshot)."""
        snap = _make_snapshot()
        result = compute_financial_ratios(snap, current_price=100.0)
        assert result.ev_ebitda is None

    def test_current_ratio_always_none_in_phase2(self):
        """Current ratio is always None in Phase 2 (not in snapshot)."""
        snap = _make_snapshot()
        result = compute_financial_ratios(snap, current_price=100.0)
        assert result.current_ratio is None

    def test_all_ratios_none_when_snapshot_has_no_financial_fields(self):
        """T7: All ratios are None when snapshot has all-None financial fields."""
        snap = _make_snapshot(
            revenue=None,
            net_income=None,
            total_assets=None,
            total_liabilities=None,
            total_equity=None,
            eps=None,
            market_cap=None,
        )
        result = compute_financial_ratios(snap, current_price=100.0)
        assert result.pe is None
        assert result.pb is None
        assert result.ps is None
        assert result.roe is None
        assert result.roa is None
        assert result.debt_equity is None

    def test_result_is_json_serialisable(self):
        """FinancialRatioResult serialises to a JSON-safe dict."""
        import json
        snap = _make_snapshot()
        result = compute_financial_ratios(snap, current_price=250.0)
        payload = result.model_dump()
        # Should not raise
        json.dumps(payload)

    def test_result_tolerates_nan_input(self):
        """NaN fields are converted to None by the base validator."""
        snap = _make_snapshot(eps=float("nan"), revenue=float("nan"))
        result = compute_financial_ratios(snap, current_price=250.0)
        # eps=NaN → pe=None; revenue=NaN → ps=None
        assert result.pe is None
        assert result.ps is None


# ---------------------------------------------------------------------------
# P2-E2-T8: compute_growth_metrics
# ---------------------------------------------------------------------------


class TestComputeGrowthMetrics:
    """Tests for compute_growth_metrics()."""

    def test_net_margin(self):
        """T8: net_margin = net_income / revenue = 20B / 100B = 0.20."""
        snap = _make_snapshot(net_income=20_000.0, revenue=100_000.0)
        result = compute_growth_metrics(snap)
        assert result.net_margin == pytest.approx(0.20)

    def test_net_margin_none_when_revenue_zero(self):
        """net_margin is None when revenue == 0."""
        snap = _make_snapshot(revenue=0.0)
        result = compute_growth_metrics(snap)
        assert result.net_margin is None

    def test_net_margin_none_when_revenue_negative(self):
        """net_margin is None when revenue < 0."""
        snap = _make_snapshot(revenue=-1.0)
        result = compute_growth_metrics(snap)
        assert result.net_margin is None

    def test_net_margin_none_when_net_income_none(self):
        """net_margin is None when net_income is None."""
        snap = _make_snapshot(net_income=None)
        result = compute_growth_metrics(snap)
        assert result.net_margin is None

    def test_yoy_growth_rates_none_in_phase2(self):
        """Phase 2 limitation: YoY growth requires two periods."""
        snap = _make_snapshot()
        result = compute_growth_metrics(snap)
        assert result.revenue_growth_yoy is None
        assert result.earnings_growth_yoy is None

    def test_margins_none_in_phase2(self):
        """gross_margin and operating_margin require unavailable fields."""
        snap = _make_snapshot()
        result = compute_growth_metrics(snap)
        assert result.gross_margin is None
        assert result.operating_margin is None

    def test_result_is_json_serialisable(self):
        """GrowthMetricsResult serialises to a JSON-safe dict."""
        import json
        snap = _make_snapshot()
        result = compute_growth_metrics(snap)
        json.dumps(result.model_dump())


# ---------------------------------------------------------------------------
# P2-E2-T9 / T10: compute_valuation_context
# ---------------------------------------------------------------------------


class TestComputeValuationContext:
    """Tests for compute_valuation_context()."""

    def _make_rising_dataset(
        self,
        n: int,
        start_price: float,
        end_price: float,
        start_date: date,
    ) -> OHLCVDataset:
        """Create a dataset with prices linearly interpolated from start to end."""
        bars = []
        for i in range(n):
            price = start_price + (end_price - start_price) * i / max(n - 1, 1)
            dt = start_date + timedelta(days=i)
            bars.append(_make_bar(dt, price))
        return _make_dataset(bars)

    def test_returns_empty_result_when_no_bars_before_as_of_date(self):
        """T10: Returns ValuationResult with all-None when no bars <= as_of_date."""
        snap = _make_snapshot(eps=10.0)
        dataset = _make_dataset([
            _make_bar(date(2024, 6, 1), close=200.0),
        ])
        result = compute_valuation_context(snap, dataset, as_of_date=date(2024, 1, 1))
        assert result.current_pe is None
        assert result.pe_1y_percentile is None
        assert result.price_to_52w_high is None

    def test_pe_1y_percentile_is_half_when_price_is_median(self):
        """T9: pe_1y_percentile == 0.5 when current P/E is the exact midpoint.

        bars = [100, 200, 150]; as_of = last bar (close=150, eps=10)
        trailing P/E = [10, 20, 15]; pe_min=10, pe_max=20
        percentile = (15-10)/(20-10) = 0.5
        """
        eps = 10.0
        snap = _make_snapshot(eps=eps)
        bars = [
            _make_bar(date(2023, 1, 1), 100.0),
            _make_bar(date(2023, 6, 1), 200.0),
            _make_bar(date(2023, 9, 1), 150.0),
        ]
        dataset = _make_dataset(bars)
        result = compute_valuation_context(
            snap, dataset, as_of_date=date(2023, 9, 1)
        )
        assert result.pe_1y_percentile == pytest.approx(0.5)

    def test_pe_1y_percentile_is_zero_at_historical_minimum(self):
        """Percentile == 0.0 when current P/E equals the trailing minimum.

        bars = [300, 200, 100]; as_of = last bar (close=100, eps=10)
        trailing P/E = [30, 20, 10]; pe_min=10, pe_max=30
        percentile = (10-10)/(30-10) = 0.0
        """
        eps = 10.0
        snap = _make_snapshot(eps=eps)
        bars = [
            _make_bar(date(2023, 1, 1), 300.0),
            _make_bar(date(2023, 6, 1), 200.0),
            _make_bar(date(2023, 9, 1), 100.0),
        ]
        dataset = _make_dataset(bars)
        result = compute_valuation_context(
            snap, dataset, as_of_date=date(2023, 9, 1)
        )
        assert result.pe_1y_percentile == pytest.approx(0.0)

    def test_pe_1y_percentile_is_one_at_historical_maximum(self):
        """Percentile == 1.0 when current P/E equals the trailing maximum."""
        eps = 10.0
        snap = _make_snapshot(eps=eps)
        bars = [
            _make_bar(date(2023, 1, 1), 100.0),
            _make_bar(date(2023, 6, 1), 200.0),
            _make_bar(date(2023, 12, 1), 300.0),
        ]
        dataset = _make_dataset(bars)
        result = compute_valuation_context(
            snap, dataset, as_of_date=date(2023, 12, 1)
        )
        assert result.pe_1y_percentile == pytest.approx(1.0)

    def test_pe_percentile_clamped_when_current_exceeds_trailing_max(self):
        """Percentile is clamped to [0,1] when current P/E is outside trailing range."""
        # current bar is after the trailing window — technically outside range
        eps = 10.0
        snap = _make_snapshot(eps=eps)
        # Build dataset where final price is well above trailing history
        bars = [
            _make_bar(date(2023, 1, 1), 100.0),
            _make_bar(date(2023, 6, 1), 200.0),
            _make_bar(date(2023, 12, 31), 5000.0),  # extreme high
        ]
        dataset = _make_dataset(bars)
        result = compute_valuation_context(
            snap, dataset, as_of_date=date(2023, 12, 31)
        )
        # Percentile must be clamped to [0, 1]
        assert 0.0 <= result.pe_1y_percentile <= 1.0

    def test_pe_percentile_is_half_when_all_prices_equal(self):
        """When all prices are equal, pe_percentile == 0.5 (midpoint by convention)."""
        eps = 10.0
        snap = _make_snapshot(eps=eps)
        bars = [_make_bar(date(2023, 1, 1) + timedelta(days=i), 150.0) for i in range(10)]
        dataset = _make_dataset(bars)
        result = compute_valuation_context(
            snap, dataset, as_of_date=date(2023, 1, 10)
        )
        assert result.pe_1y_percentile == pytest.approx(0.5)

    def test_current_pe_computation(self):
        """current_pe = current_price / eps."""
        eps = 10.0
        snap = _make_snapshot(eps=eps)
        bars = [_make_bar(date(2023, 1, 1), 250.0)]
        dataset = _make_dataset(bars)
        result = compute_valuation_context(snap, dataset, as_of_date=date(2023, 1, 1))
        assert result.current_pe == pytest.approx(250.0 / eps)

    def test_all_pe_fields_none_when_eps_is_none(self):
        """T10: All P/E fields are None when snapshot.eps is None."""
        snap = _make_snapshot(eps=None)
        bars = [_make_bar(date(2023, 1, 1) + timedelta(days=i), 100.0 + i) for i in range(5)]
        dataset = _make_dataset(bars)
        result = compute_valuation_context(snap, dataset, as_of_date=date(2023, 1, 5))
        assert result.current_pe is None
        assert result.pe_1y_min is None
        assert result.pe_1y_max is None
        assert result.pe_1y_percentile is None

    def test_all_pe_fields_none_when_eps_is_negative(self):
        """P/E fields are None when eps <= 0."""
        snap = _make_snapshot(eps=-5.0)
        bars = [_make_bar(date(2023, 1, 1), 100.0)]
        dataset = _make_dataset(bars)
        result = compute_valuation_context(snap, dataset, as_of_date=date(2023, 1, 1))
        assert result.current_pe is None
        assert result.pe_1y_percentile is None

    def test_52w_high_and_low_ratio(self):
        """price_to_52w_high = current / 52w_max; price_to_52w_low = current / 52w_min."""
        snap = _make_snapshot(eps=10.0)
        # Bars spanning < 365 days so all in 52-week window
        bars = [
            _make_bar(date(2023, 3, 1), 100.0),  # low
            _make_bar(date(2023, 6, 1), 200.0),  # current
            _make_bar(date(2023, 9, 1), 300.0),  # high
        ]
        dataset = _make_dataset(bars)
        result = compute_valuation_context(snap, dataset, as_of_date=date(2023, 9, 1))
        # current = 300, high_52w = 300, low_52w = 100
        assert result.price_to_52w_high == pytest.approx(1.0)
        assert result.price_to_52w_low == pytest.approx(300.0 / 100.0)

    def test_result_is_json_serialisable(self):
        """ValuationResult serialises to a JSON-safe dict."""
        import json
        snap = _make_snapshot(eps=10.0)
        bars = [_make_bar(date(2023, 1, 1) + timedelta(days=i), 100.0 + i) for i in range(10)]
        dataset = _make_dataset(bars)
        result = compute_valuation_context(snap, dataset, as_of_date=date(2023, 1, 10))
        json.dumps(result.model_dump())
