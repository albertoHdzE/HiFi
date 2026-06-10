"""
Unit tests for Phase 2 engine result types (P2-E1).

Tickets covered:
- P2-E1-T7: each result type accepts all-None construction without error
- P2-E1-T8: each result type serialises to a JSON-safe dict
             (json.dumps succeeds; no float('nan'); None fields present as null)

Additional coverage:
- NaN-to-None conversion: float('nan') inputs are silently converted to None
- Partial construction: a mix of None and non-None fields is accepted
- Field presence: None fields are included in model_dump() output (not dropped)
"""

import json
import math

import pytest

from hifi.engines.types import (
    FinancialRatioResult,
    GrowthMetricsResult,
    MacroSnapshotResult,
    RiskMetricsResult,
    TechnicalIndicatorsResult,
    ValuationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_RESULT_TYPES = [
    FinancialRatioResult,
    GrowthMetricsResult,
    TechnicalIndicatorsResult,
    RiskMetricsResult,
    ValuationResult,
    MacroSnapshotResult,
]


def _all_none(result_type):
    """Construct an instance of result_type with all fields at their defaults."""
    return result_type()


def _is_json_safe(obj: dict) -> bool:
    """Return True if json.dumps succeeds on obj without raising."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# P2-E1-T7: All-None construction
# ---------------------------------------------------------------------------


class TestAllNoneConstruction:
    """T7: each result type can be constructed with no arguments (all fields None)."""

    def test_financial_ratio_result_all_none(self) -> None:
        result = FinancialRatioResult()
        assert result.pe is None
        assert result.pb is None
        assert result.ps is None
        assert result.ev_ebitda is None
        assert result.roe is None
        assert result.roa is None
        assert result.debt_equity is None
        assert result.current_ratio is None

    def test_growth_metrics_result_all_none(self) -> None:
        result = GrowthMetricsResult()
        assert result.revenue_growth_yoy is None
        assert result.earnings_growth_yoy is None
        assert result.gross_margin is None
        assert result.operating_margin is None
        assert result.net_margin is None

    def test_technical_indicators_result_all_none(self) -> None:
        result = TechnicalIndicatorsResult()
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

    def test_risk_metrics_result_all_none(self) -> None:
        result = RiskMetricsResult()
        assert result.hist_vol_20d is None
        assert result.hist_vol_60d is None
        assert result.hist_vol_252d is None
        assert result.beta is None
        assert result.max_drawdown_252d is None
        assert result.sharpe_252d is None
        assert result.var_95_20d is None

    def test_valuation_result_all_none(self) -> None:
        result = ValuationResult()
        assert result.current_pe is None
        assert result.pe_1y_min is None
        assert result.pe_1y_max is None
        assert result.pe_1y_percentile is None
        assert result.price_to_52w_high is None
        assert result.price_to_52w_low is None

    def test_macro_snapshot_result_all_none(self) -> None:
        result = MacroSnapshotResult()
        assert result.fed_funds_rate is None
        assert result.cpi_yoy is None
        assert result.unemployment_rate is None
        assert result.yield_10y is None
        assert result.yield_2y is None
        assert result.yield_curve_slope is None
        assert result.vix is None
        assert result.gdp_growth is None

    @pytest.mark.parametrize("result_type", _ALL_RESULT_TYPES)
    def test_all_types_accept_no_arguments(self, result_type) -> None:
        """Parametric guard: no result type raises on default construction."""
        instance = result_type()
        dumped = instance.model_dump()
        assert all(v is None for v in dumped.values())


# ---------------------------------------------------------------------------
# P2-E1-T8: JSON serialisation
# ---------------------------------------------------------------------------


class TestJsonSerialisation:
    """T8: each result type serialises to a JSON-safe dict."""

    @pytest.mark.parametrize("result_type", _ALL_RESULT_TYPES)
    def test_all_none_instance_is_json_safe(self, result_type) -> None:
        """All-None instance produces a JSON-safe dict."""
        dumped = result_type().model_dump()
        assert _is_json_safe(dumped), f"{result_type.__name__}.model_dump() is not JSON-safe"

    def test_financial_ratio_with_values_is_json_safe(self) -> None:
        result = FinancialRatioResult(pe=25.0, pb=3.5, roe=0.15)
        assert _is_json_safe(result.model_dump())

    def test_technical_indicators_with_values_is_json_safe(self) -> None:
        result = TechnicalIndicatorsResult(
            sma=150.0, ema=148.5, rsi=62.3,
            macd=1.2, macd_signal=0.9, macd_hist=0.3,
            bb_upper=155.0, bb_mid=150.0, bb_lower=145.0,
            atr=2.5,
        )
        assert _is_json_safe(result.model_dump())

    def test_risk_metrics_with_values_is_json_safe(self) -> None:
        result = RiskMetricsResult(
            hist_vol_20d=0.18, hist_vol_60d=0.20, hist_vol_252d=0.22,
            beta=1.05, max_drawdown_252d=0.15, sharpe_252d=1.2, var_95_20d=0.025,
        )
        assert _is_json_safe(result.model_dump())

    def test_macro_snapshot_with_values_is_json_safe(self) -> None:
        result = MacroSnapshotResult(
            fed_funds_rate=5.25, cpi_yoy=3.1, unemployment_rate=3.8,
            yield_10y=4.2, yield_2y=4.8, yield_curve_slope=-0.6,
            vix=18.5, gdp_growth=2.1,
        )
        assert _is_json_safe(result.model_dump())

    @pytest.mark.parametrize("result_type", _ALL_RESULT_TYPES)
    def test_none_fields_appear_as_null_in_json(self, result_type) -> None:
        """None fields must be present as JSON null, not omitted."""
        dumped = result_type().model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        # Every key that was None in the dict should appear as null in JSON
        for key, value in dumped.items():
            assert key in parsed, f"Key '{key}' was dropped during JSON round-trip"
            if value is None:
                assert parsed[key] is None, (
                    f"Key '{key}' expected null in JSON, got {parsed[key]!r}"
                )


# ---------------------------------------------------------------------------
# NaN-to-None conversion
# ---------------------------------------------------------------------------


class TestNanToNoneConversion:
    """NaN float inputs must be silently converted to None."""

    def test_financial_ratio_nan_pe_becomes_none(self) -> None:
        result = FinancialRatioResult(pe=float("nan"))
        assert result.pe is None

    def test_technical_indicators_nan_rsi_becomes_none(self) -> None:
        result = TechnicalIndicatorsResult(rsi=float("nan"), sma=150.0)
        assert result.rsi is None
        assert result.sma == 150.0

    def test_risk_metrics_all_nan_becomes_all_none(self) -> None:
        nan = float("nan")
        result = RiskMetricsResult(
            hist_vol_20d=nan, hist_vol_60d=nan, hist_vol_252d=nan,
            beta=nan, max_drawdown_252d=nan, sharpe_252d=nan, var_95_20d=nan,
        )
        assert all(v is None for v in result.model_dump().values())

    def test_nan_field_makes_dict_json_safe(self) -> None:
        """A result with NaN input is still JSON-serialisable after conversion."""
        result = MacroSnapshotResult(fed_funds_rate=float("nan"), cpi_yoy=3.1)
        dumped = result.model_dump()
        assert _is_json_safe(dumped)
        assert dumped["fed_funds_rate"] is None
        assert dumped["cpi_yoy"] == pytest.approx(3.1)

    @pytest.mark.parametrize("result_type", _ALL_RESULT_TYPES)
    def test_nan_in_any_field_is_not_nan_after_construction(self, result_type) -> None:
        """Regardless of which field receives NaN, the result is never NaN."""
        fields = result_type.model_fields.keys()
        for field in fields:
            instance = result_type(**{field: float("nan")})
            value = getattr(instance, field)
            assert value is None or not math.isnan(value), (
                f"{result_type.__name__}.{field} is NaN after construction"
            )


# ---------------------------------------------------------------------------
# Partial construction
# ---------------------------------------------------------------------------


class TestPartialConstruction:
    """Mix of None and float values is accepted for all types."""

    def test_financial_ratio_partial(self) -> None:
        result = FinancialRatioResult(pe=20.0, roe=0.12)
        assert result.pe == pytest.approx(20.0)
        assert result.roe == pytest.approx(0.12)
        assert result.pb is None
        assert result.ps is None

    def test_valuation_result_partial(self) -> None:
        result = ValuationResult(current_pe=22.5, pe_1y_percentile=0.75)
        assert result.current_pe == pytest.approx(22.5)
        assert result.pe_1y_percentile == pytest.approx(0.75)
        assert result.pe_1y_min is None

    def test_risk_metrics_partial_with_beta_none(self) -> None:
        """beta=None when no benchmark is available -- standard case."""
        result = RiskMetricsResult(hist_vol_252d=0.22, sharpe_252d=1.1)
        assert result.beta is None
        assert result.hist_vol_252d == pytest.approx(0.22)
