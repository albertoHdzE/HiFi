"""
IC/IR/herding metrics for Phase 15 walk-forward evaluation (DJ-096).

Primary metric: Information Coefficient (IC) = Spearman rank correlation of the
ensemble buy-strength signal with the 60-day forward return, computed across all
(date, ticker) pairs in the evaluation period.

  IC > 0.0  : ensemble has predictive signal above random
  IC > 0.05 : practically significant (industry convention)

Secondary metrics: IR = IC / IC_std, herding coefficient by regime.

Buy-strength encoding
---------------------
The ensemble collective_decision is mapped to a scalar signal in [-1, +1]:
  Buy  ->  +collective_confidence
  Sell ->  -collective_confidence
  Hold ->   0.0

This encoding preserves ordinal information (strong Buy > weak Buy > Hold >
weak Sell > strong Sell) while allowing rank correlation computation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ICResult:
    """Information Coefficient for one evaluation window."""

    ic: float         # Spearman rank correlation in [-1, +1]
    p_value: float    # two-sided p-value (0 = perfectly predictive)
    n_pairs: int      # number of (ticker, date) pairs used


def buy_strength(output: dict[str, Any]) -> float | None:
    """
    Convert an EnsembleOutput (dict) to a buy-strength scalar in [-1, +1].

    Parameters
    ----------
    output : dict
        Deserialized EnsembleOutput (via model_dump() or loaded from JSON).

    Returns
    -------
    float | None
        Buy-strength in [-1, +1], or None if no valid collective decision.
    """
    decision_block = output.get("ensemble_decision") or {}
    decision = decision_block.get("collective_decision")
    confidence = float(decision_block.get("collective_confidence") or 0.5)
    if decision == "Buy":
        return confidence
    if decision == "Sell":
        return -confidence
    if decision == "Hold":
        return 0.0
    return None


def compute_ic(
    buy_strengths: list[float],
    forward_returns: list[float],
) -> ICResult:
    """
    Compute Spearman rank IC between buy-strength signals and forward returns.

    Parameters
    ----------
    buy_strengths : list[float]
        Buy-strength signal values in [-1, +1].
    forward_returns : list[float]
        60-day realized price returns, same length as buy_strengths.

    Returns
    -------
    ICResult

    Raises
    ------
    ValueError
        If lengths differ or fewer than 2 pairs are provided.
    """
    if len(buy_strengths) != len(forward_returns):
        raise ValueError(
            f"buy_strengths length ({len(buy_strengths)}) != "
            f"forward_returns length ({len(forward_returns)})"
        )
    if len(buy_strengths) < 2:
        raise ValueError(
            "At least 2 (signal, return) pairs are required to compute IC"
        )

    from scipy.stats import spearmanr

    ic, p_value = spearmanr(buy_strengths, forward_returns)
    return ICResult(
        ic=float(ic),
        p_value=float(p_value),
        n_pairs=len(buy_strengths),
    )


def compute_ir(ic_series: list[float]) -> float:
    """
    Compute Information Ratio: mean(IC) / std(IC) across monthly windows.

    A higher IR indicates a more consistent predictive signal.
    Returns 0.0 when fewer than 2 windows are available or std is zero.

    Parameters
    ----------
    ic_series : list[float]
        Monthly IC values (one per evaluation date).

    Returns
    -------
    float
        Information Ratio, or 0.0 when not computable.
    """
    if len(ic_series) < 2:
        return 0.0
    n = len(ic_series)
    mean_ic = sum(ic_series) / n
    variance = sum((ic - mean_ic) ** 2 for ic in ic_series) / (n - 1)
    std_ic = math.sqrt(variance)
    if std_ic < 1e-10:
        return 0.0
    return mean_ic / std_ic


def compute_herding_coefficient(outputs: list[dict[str, Any]]) -> float:
    """
    Compute herding coefficient: fraction of ensemble runs with full agent agreement.

    Uses the ensemble_decision.agreement field, which is True when all voting agents
    cast identical decisions.

    Parameters
    ----------
    outputs : list[dict]
        Deserialized EnsembleOutputs.

    Returns
    -------
    float
        Herding coefficient in [0, 1].  0.0 for empty list.
    """
    if not outputs:
        return 0.0
    n_herded = sum(
        1 for o in outputs
        if (o.get("ensemble_decision") or {}).get("agreement", False)
    )
    return n_herded / len(outputs)


def forward_return_from_ohlcv(
    ohlcv_df: Any,
    as_of_date: str,
    horizon_trading_days: int = 60,
) -> float | None:
    """
    Compute forward return from OHLCV data without network access.

    Uses the Close price at as_of_date (or next available trading day) and
    the Close price horizon_trading_days later.  This is the offline equivalent
    of fetch_forward_return() in scripts/label_outcomes.py, operating on the
    pre-downloaded OHLCV parquets in data/market/{ticker}/ohlcv.parquet.

    Parameters
    ----------
    ohlcv_df : pandas.DataFrame
        DataFrame with DatetimeIndex (Date) and a 'Close' column.
    as_of_date : str
        ISO 8601 evaluation date.
    horizon_trading_days : int
        Number of *trading* days forward (default 60, corresponding to
        approximately 3 calendar months).

    Returns
    -------
    float | None
        Realized return (price_end / price_start) - 1, or None if data
        is unavailable for either the start or end date.

    Notes
    -----
    A ``None`` here silently removes one (ticker, date) pair from the IC
    denominator, and IC is the headline metric of the whole project. Two very
    different situations produce it:

    * **Legitimate** — the horizon runs past the end of the series, so no
      forward return exists yet. Expected for the last ~60 trading days.
    * **A defect** — the OHLCV frame is malformed or the wrong shape. Under
      DJ-120 the market path was broken for 83 of 98 tickers; a bare
      ``except: return None`` would have quietly computed IC on the survivors
      and reported no anomaly.

    The two are therefore separated: the expected case returns ``None``
    quietly, anything unexpected returns ``None`` *and* logs a warning naming
    the ticker-date, so a broken data path shows up as log noise instead of a
    smaller, unremarked ``n``.
    """
    try:
        df = ohlcv_df["Close"].dropna().sort_index()
        t0_str = as_of_date

        # Price at or after as_of_date
        after_t0 = df[df.index >= t0_str]
        if after_t0.empty:
            return None
        price_start = float(after_t0.iloc[0])

        # Price horizon_trading_days trading days later
        start_iloc = df.index.get_loc(after_t0.index[0])
        end_iloc = start_iloc + horizon_trading_days
        if end_iloc >= len(df):
            return None
        price_end = float(df.iloc[end_iloc])

        if price_start == 0.0:
            logger.warning(
                "forward_return: zero start price for %s; dropping this pair "
                "from the IC sample", as_of_date,
            )
            return None
        return (price_end - price_start) / price_start
    except Exception as exc:
        logger.warning(
            "forward_return failed at %s (%s: %s); this pair is dropped from "
            "the IC sample. Expected only for malformed OHLCV — check the "
            "market data path before trusting n.",
            as_of_date, type(exc).__name__, exc,
        )
        return None
