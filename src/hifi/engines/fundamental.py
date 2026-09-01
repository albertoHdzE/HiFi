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
        # EV/EBITDA (DJ-134). Enterprise value = market cap + total debt - cash;
        # it prices the whole business rather than just the equity, which is why
        # it is the multiple that survives differences in leverage.
        # Requires positive EBITDA: a negative denominator yields a negative
        # multiple that reads as "cheap" and is meaningless.
        ev_ebitda=_positive_div(
            _enterprise_value(snapshot), snapshot.ebitda
        ),
        # Return on Equity = net_income / total_equity
        roe=_div(snapshot.net_income, snapshot.total_equity),
        # Return on Assets = net_income / total_assets (assets always positive)
        roa=_positive_div(snapshot.net_income, snapshot.total_assets),
        # Debt / Equity = total_liabilities / total_equity
        debt_equity=_div(snapshot.total_liabilities, snapshot.total_equity),
        # Current Ratio (DJ-134) = current assets / current liabilities.
        current_ratio=_positive_div(
            snapshot.current_assets, snapshot.current_liabilities
        ),
    )


def _enterprise_value(snapshot: FundamentalsSnapshot) -> float | None:
    """Market cap + total debt - cash, or None when market cap is unknown.

    Debt and cash are treated as zero when absent rather than voiding the whole
    figure: a company that reports no debt line genuinely has none to add. Market
    cap is different -- without it there is no enterprise value at all.
    """
    if snapshot.market_cap is None:
        return None
    debt = snapshot.total_liabilities or 0.0
    cash = snapshot.cash_and_equivalents or 0.0
    return snapshot.market_cap + debt - cash


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
    gross_profit = (
        snapshot.revenue - snapshot.cost_of_revenue
        if snapshot.revenue is not None and snapshot.cost_of_revenue is not None
        else None
    )
    return GrowthMetricsResult(
        # DJ-134. Same fiscal quarter against the same quarter a year earlier,
        # which handles seasonality and needs five reported quarters rather than
        # the eight a TTM-against-TTM comparison would require. Mixing a TTM
        # numerator with a single-quarter base is the standard way to
        # manufacture a 300% growth rate; both sides here are single quarters.
        revenue_growth_yoy=_growth(
            snapshot.revenue_latest_q, snapshot.revenue_year_ago_q
        ),
        earnings_growth_yoy=_growth(
            snapshot.net_income_latest_q, snapshot.net_income_year_ago_q
        ),
        gross_margin=_positive_div(gross_profit, snapshot.revenue),
        operating_margin=_positive_div(snapshot.operating_income, snapshot.revenue),
        net_margin=_positive_div(snapshot.net_income, snapshot.revenue),
    )


def _growth(current: float | None, prior: float | None) -> float | None:
    """Year-on-year growth rate, or None when it would not be meaningful.

    A non-positive base is rejected rather than divided through. Earnings that
    move from -100 to +50 do not have a "-150% growth rate"; the sign flip makes
    the ratio uninterpretable, and feeding one to an LLM invites a confident
    reading of a number that means nothing.
    """
    if current is None or prior is None or prior <= 0:
        return None
    return (current - prior) / prior


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
