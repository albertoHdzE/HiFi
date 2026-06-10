"""
Engine result types for the HiFi deterministic financial computation layer.

These models define the output contracts for all engine functions. Every
computation in Phase 2+ returns one of these typed result objects -- never
a raw dict, never a bare float.

Design decisions:
- All fields are Optional[float]. An engine that cannot compute a metric
  (insufficient data, missing input, zero denominator) returns None for
  that field. None is propagated to the MCP response as JSON null. The
  agent downstream must interpret None as "not available", not as zero.
- NaN is forbidden: any float that arrives as float('nan') is converted
  to None by the base model validator. NaN is not valid JSON and silently
  corrupts downstream computation.
- Pydantic v2 is used for consistency with Phase 1 data schemas. All
  structured data in HiFi is validated through one library.
- Result types carry no provenance. Provenance belongs to the input data
  (Phase 1 schemas). Engine results are derived quantities; the audit
  trail is the input dataset's ProvenanceRecord plus the engine call
  parameters logged by the MCP server (Phase 5).

David reference: §4.6 (Modularity) -- these types are the interface layer
between engine implementations and their consumers. Replacing an engine
implementation (e.g., a faster RSI algorithm in Phase 8) preserves the
TechnicalIndicatorsResult contract unchanged.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, model_validator


class _NanToNoneBase(BaseModel):
    """Base model that converts float NaN fields to None on construction.

    NaN is not valid JSON (json.dumps raises ValueError on float('nan')).
    Any float field that arrives as float('nan') is silently set to None
    so that all result types are unconditionally JSON-serialisable.
    """

    @model_validator(mode="before")
    @classmethod
    def _convert_nan_to_none(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in data.items()
            }
        return data


# ---------------------------------------------------------------------------
# P2-E1-T1: Financial ratio result
# ---------------------------------------------------------------------------


class FinancialRatioResult(_NanToNoneBase):
    """
    Output of compute_financial_ratios().

    Fundamental valuation and profitability ratios derived from one
    FundamentalsSnapshot plus current price.

    None for a field means the ratio could not be computed: the required
    input was absent, the denominator was zero, or the data source did not
    provide the underlying figures. None is not zero.
    """

    pe: float | None = None
    pb: float | None = None
    ps: float | None = None
    ev_ebitda: float | None = None
    roe: float | None = None
    roa: float | None = None
    debt_equity: float | None = None
    current_ratio: float | None = None


# ---------------------------------------------------------------------------
# P2-E1-T2: Growth metrics result
# ---------------------------------------------------------------------------


class GrowthMetricsResult(_NanToNoneBase):
    """
    Output of compute_growth_metrics().

    Year-over-year growth rates and margin metrics.

    Phase 2 limitation: FundamentalsSnapshot carries only one period's data
    (the most recent annual snapshot from yfinance). Revenue and earnings
    growth rates require two periods and therefore return None in Phase 2.
    Multi-period data from SEC EDGAR will unlock these fields in Phase 7+.
    Agents receiving None here must interpret it as "insufficient data",
    not as "zero growth".

    Margin fields (gross_margin, operating_margin, net_margin) may be
    computable from single-period data where the underlying figures are
    available in the snapshot.
    """

    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None


# ---------------------------------------------------------------------------
# P2-E1-T3: Technical indicators result
# ---------------------------------------------------------------------------


class TechnicalIndicatorsResult(_NanToNoneBase):
    """
    Output of compute_technical_indicators().

    Six technical indicators computed from an OHLCV price series.

    All indicators return None when the series is shorter than the required
    window (e.g., SMA(20) on a 10-bar series). None propagates to the MCP
    response as JSON null.

    Sources: Wilder (1978) for RSI and ATR; Appel (1979) for MACD;
    Bollinger (1992) for Bollinger Bands.
    """

    sma: float | None = None
    ema: float | None = None
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    bb_upper: float | None = None
    bb_mid: float | None = None
    bb_lower: float | None = None
    atr: float | None = None


# ---------------------------------------------------------------------------
# P2-E1-T4: Risk metrics result
# ---------------------------------------------------------------------------


class RiskMetricsResult(_NanToNoneBase):
    """
    Output of compute_risk_metrics().

    Portfolio and risk metrics computed via QuantStats from an OHLCV price
    series. All volatility figures are annualised (252 trading-day convention).

    beta requires a benchmark price series (typically SPY). When no benchmark
    is provided, beta is None.

    max_drawdown_252d is stored as a positive float: a 30% drawdown is 0.30,
    not -0.30.

    sharpe_252d uses the risk-free rate passed to the engine; when none is
    provided the engine logs a WARNING and uses 0.0.

    var_95_20d is a daily figure (not annualised): the loss exceeded with 5%
    probability over the trailing 20 bars.
    """

    hist_vol_20d: float | None = None
    hist_vol_60d: float | None = None
    hist_vol_252d: float | None = None
    beta: float | None = None
    max_drawdown_252d: float | None = None
    sharpe_252d: float | None = None
    var_95_20d: float | None = None


# ---------------------------------------------------------------------------
# P2-E1-T5: Valuation context result
# ---------------------------------------------------------------------------


class ValuationResult(_NanToNoneBase):
    """
    Output of compute_valuation_context().

    P/E valuation context: where does the current P/E stand relative to its
    own trailing 252-day (or available) range?

    pe_1y_percentile is in [0.0, 1.0]: 0.0 means the current P/E is at the
    trailing minimum; 1.0 means it is at the trailing maximum; 0.5 means it
    sits at the median.

    price_to_52w_high and price_to_52w_low are ratios: 1.0 means the current
    price equals the 52-week high/low; values below 1.0 for price_to_52w_high
    mean the price is below its peak.

    All fields are None when eps is None (no P/E series can be constructed).
    """

    current_pe: float | None = None
    pe_1y_min: float | None = None
    pe_1y_max: float | None = None
    pe_1y_percentile: float | None = None
    price_to_52w_high: float | None = None
    price_to_52w_low: float | None = None


# ---------------------------------------------------------------------------
# P2-E1-T6: Macro snapshot result
# ---------------------------------------------------------------------------


class MacroSnapshotResult(_NanToNoneBase):
    """
    Output of get_macro_snapshot().

    A point-in-time cross-section of macroeconomic indicators for a given
    date. Values are forward-filled from the most recent observation on or
    before the requested date, using the same forward-fill logic applied
    by the Phase 1 macro fetcher.

    yield_curve_slope is yield_10y minus yield_2y (the 10-2 spread). A
    negative slope indicates an inverted yield curve.

    All fields are None when no macro data is available for the requested
    date (e.g., a date before the fixture window).
    """

    fed_funds_rate: float | None = None
    cpi_yoy: float | None = None
    unemployment_rate: float | None = None
    yield_10y: float | None = None
    yield_2y: float | None = None
    yield_curve_slope: float | None = None
    vix: float | None = None
    gdp_growth: float | None = None


__all__ = [
    "FinancialRatioResult",
    "GrowthMetricsResult",
    "MacroSnapshotResult",
    "RiskMetricsResult",
    "TechnicalIndicatorsResult",
    "ValuationResult",
]
