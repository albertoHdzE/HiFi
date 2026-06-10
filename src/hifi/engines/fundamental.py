"""
Fundamental and valuation engine for HiFi (P2-E2).

Pure functions that derive financial ratios and valuation context from
FundamentalsSnapshot and OHLCVDataset inputs. No I/O, no side effects.

Phase 2 scope notes:
- FundamentalsSnapshot (Phase 1, yfinance) carries one period's annual data.
  Fields available: revenue, net_income, total_assets, total_liabilities,
  total_equity, eps, pe_ratio, market_cap.
- Fields NOT available: EBITDA, current_assets, current_liabilities, COGS,
  operating_income. Ratios that require them return None.
- YoY growth rates require two periods and return None in Phase 2.
  SEC EDGAR multi-period data (Phase 7+) will unlock these.

David reference: §4.1 (Deterministic-First) — every ratio here is a pure
mathematical function. Given the same snapshot and price, the result is
identical across all agents, all runs, all environments.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from hifi.data.schemas import FundamentalsSnapshot, OHLCVDataset
from hifi.engines.types import FinancialRatioResult, GrowthMetricsResult, ValuationResult

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _div(numerator: float | None, denominator: float | None) -> float | None:
    """
    Safe division that returns None on missing data or zero denominator.

    Does NOT guard against negative denominators: negative equity (ROE with
    negative equity) and negative revenue (not a real case but defensive) are
    preserved as-is. The caller decides whether the sign is meaningful.
    """
    if numerator is None or denominator is None:
        return None
    if denominator == 0.0:
        return None
    return numerator / denominator


def _positive_div(
    numerator: float | None, denominator: float | None
) -> float | None:
    """
    Division that additionally returns None when denominator is <= 0.

    Used for ratios where a non-positive denominator produces a meaningless
    result (e.g., P/E with negative or zero earnings).
    """
    if numerator is None or denominator is None:
        return None
    if denominator <= 0.0:
        return None
    return numerator / denominator


# ---------------------------------------------------------------------------
# P2-E2-T1: Financial ratios
# ---------------------------------------------------------------------------


def compute_financial_ratios(
    snapshot: FundamentalsSnapshot,
    current_price: float,
) -> FinancialRatioResult:
    """
    Compute fundamental valuation and profitability ratios.

    Parameters
    ----------
    snapshot : FundamentalsSnapshot
        Most-recent annual snapshot from yfinance for this ticker.
    current_price : float
        The closing price on the analysis date (unadjusted close).

    Returns
    -------
    FinancialRatioResult
        All fields that can be computed from available data.
        Fields not computable (missing input, zero denominator, not in
        Phase 1 snapshot) are None.
    """
    return FinancialRatioResult(
        # Price / Earnings — requires positive EPS
        pe=_positive_div(current_price, snapshot.eps),
        # Price / Book — market_cap / total_equity
        # Negative equity (some mature companies) produces a negative P/B,
        # which is meaningful and preserved rather than replaced with None.
        pb=_div(snapshot.market_cap, snapshot.total_equity),
        # Price / Sales — requires positive revenue
        ps=_positive_div(snapshot.market_cap, snapshot.revenue),
        # EV/EBITDA — EBITDA not in Phase 1 snapshot; deferred to Phase 7+
        ev_ebitda=None,
        # Return on Equity = net_income / total_equity
        roe=_div(snapshot.net_income, snapshot.total_equity),
        # Return on Assets = net_income / total_assets (assets always positive)
        roa=_positive_div(snapshot.net_income, snapshot.total_assets),
        # Debt / Equity = total_liabilities / total_equity
        debt_equity=_div(snapshot.total_liabilities, snapshot.total_equity),
        # Current Ratio — current_assets/current_liabilities not in Phase 1
        current_ratio=None,
    )


# ---------------------------------------------------------------------------
# P2-E2-T2: Growth metrics
# ---------------------------------------------------------------------------


def compute_growth_metrics(snapshot: FundamentalsSnapshot) -> GrowthMetricsResult:
    """
    Compute growth rates and margin metrics from a single snapshot.

    YoY growth rates (revenue, earnings) require two periods and return None
    in Phase 2. Margin metrics that can be derived from single-period data
    are computed where the underlying fields are available.

    Parameters
    ----------
    snapshot : FundamentalsSnapshot

    Returns
    -------
    GrowthMetricsResult
    """
    return GrowthMetricsResult(
        # Requires two periods — None in Phase 2
        revenue_growth_yoy=None,
        earnings_growth_yoy=None,
        # Gross margin = (revenue - COGS) / revenue — COGS not in snapshot
        gross_margin=None,
        # Operating margin = operating_income / revenue — not in snapshot
        operating_margin=None,
        # Net margin = net_income / revenue
        net_margin=_positive_div(snapshot.net_income, snapshot.revenue),
    )


# ---------------------------------------------------------------------------
# P2-E2-T3: Valuation context
# ---------------------------------------------------------------------------


def compute_valuation_context(
    snapshot: FundamentalsSnapshot,
    dataset: OHLCVDataset,
    as_of_date: date,
) -> ValuationResult:
    """
    Compute valuation context: where does the current P/E sit relative to
    its own trailing history, and where is the price relative to its 52-week
    range?

    Uses the snapshot EPS as a fixed anchor: trailing P/E series = each daily
    close / snapshot.eps. This is an approximation because EPS changes over
    time; a full trailing P/E would require a quarterly EPS series. The
    approximation is documented and acceptable for Phase 2 agent context.

    Parameters
    ----------
    snapshot : FundamentalsSnapshot
    dataset : OHLCVDataset
    as_of_date : date

    Returns
    -------
    ValuationResult
    """
    bars_to_date = sorted(
        [b for b in dataset.bars if b.date <= as_of_date], key=lambda b: b.date
    )
    if not bars_to_date:
        return ValuationResult()

    current_bar = bars_to_date[-1]
    current_price = current_bar.close

    # 52-week range
    cutoff_52w = as_of_date - timedelta(days=365)
    bars_52w = [b for b in bars_to_date if b.date >= cutoff_52w]
    if bars_52w:
        prices_52w = [b.close for b in bars_52w]
        high_52w = max(prices_52w)
        low_52w = min(prices_52w)
        price_to_52w_high = _positive_div(current_price, high_52w)
        price_to_52w_low = _positive_div(current_price, low_52w)
    else:
        price_to_52w_high = price_to_52w_low = None

    # Trailing P/E using snapshot EPS as anchor
    eps = snapshot.eps
    if eps is None or eps <= 0.0:
        return ValuationResult(
            price_to_52w_high=price_to_52w_high,
            price_to_52w_low=price_to_52w_low,
        )

    current_pe = current_price / eps

    # Build trailing P/E series for the available history (up to 252 bars)
    trailing_bars = bars_to_date[-252:]
    trailing_pes = [b.close / eps for b in trailing_bars]

    pe_min = min(trailing_pes)
    pe_max = max(trailing_pes)

    if math.isclose(pe_max, pe_min):
        pe_percentile = 0.5
    else:
        pe_percentile = (current_pe - pe_min) / (pe_max - pe_min)
        # Clamp to [0, 1] in case current P/E is outside the trailing range
        pe_percentile = max(0.0, min(1.0, pe_percentile))

    return ValuationResult(
        current_pe=current_pe,
        pe_1y_min=pe_min,
        pe_1y_max=pe_max,
        pe_1y_percentile=pe_percentile,
        price_to_52w_high=price_to_52w_high,
        price_to_52w_low=price_to_52w_low,
    )
