"""
Forward-return labeler for Phase 10 accuracy measurement (P10-E0, DJ-044, DJ-052).

Responsibilities:
1. compute_forward_return() — load OHLCV Parquet for a ticker and compute the
   horizon_days-trading-day forward return from a given analysis date.
2. label_method_decisions() — extract all four methods' collective decisions from
   a list of EnsembleOutputs and label each against the forward return.
3. label_agent_decisions() — extract per-agent signals from EnsembleOutput.signals
   and label each against the forward return.
4. build_method_accuracy_report() — pure wrapper: construct a MethodAccuracyReport
   from a list of MethodDecisionRecords.
5. compute_divergence_rates() — pairwise method divergence rates across a set of
   MethodDecisionRecords.

Labeling rules (DJ-042):
  BUY correct   if forward_return > +0.02
  SELL correct  if forward_return < -0.02
  HOLD correct  if abs(forward_return) <= 0.02
  outcome_correct = None when forward_return is None (unlabeled, not incorrect)

The ±2% band acknowledges that a neutral signal earns credit only when the market
is genuinely flat. It avoids simultaneously labeling near-zero periods as correct
for Hold and incorrect for Buy/Sell.

Horizon convention (DJ-052):
  horizon_days=60 — primary evaluation horizon (David §D-04)
  horizon_days=20 — secondary horizon; differentiates agents at short-term accuracy

Design notes:
- This module reads Parquet files directly via pandas (raw read) rather than via
  hifi.data.storage.read_ohlcv(), because bootstrap Parquet files may lack HiFi
  metadata (Phase 9 bootstrap wrote them via yfinance directly).
- The core forward-return logic mirrors _forward_return() in run_phase9_bootstrap.py
  but lives here as the canonical, importable implementation for Phase 10+.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from hifi.collective.schemas import (
    CalibrationReport,
    DecisionRecord,
    MethodAccuracyReport,
    MethodDecisionRecord,
)

if TYPE_CHECKING:
    from hifi.collective.schemas import EnsembleOutput  # noqa: F401 (type hint only)

logger = logging.getLogger(__name__)

_LABEL_THRESHOLD = 0.02  # DJ-042: ±2% band


# ---------------------------------------------------------------------------
# Internal OHLCV loader
# ---------------------------------------------------------------------------


def _load_prices(ticker: str, data_dir: str) -> dict[date, float]:
    """
    Load close prices for ticker from the most-recent OHLCV Parquet in data_dir.

    Returns {trading_date: adjusted_close_or_close} as a sorted dict.
    Returns {} if no Parquet file exists or no recognisable price column is found.
    Handles both HiFi-format Parquets (adjusted_close/close columns) and raw
    yfinance Parquets (Adj Close/Close columns with capitalised names).
    """
    import pandas as pd

    from hifi.data.market_store import resolve_ohlcv_path

    try:
        path = resolve_ohlcv_path(ticker, data_dir)
    except FileNotFoundError:
        logger.warning("No OHLCV Parquet for %s in %s", ticker, data_dir)
        return {}

    df = pd.read_parquet(path)

    # Normalise index to DatetimeIndex
    if "date" in df.columns:
        df = df.set_index("date")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    # Prefer adjusted close; fall back to close
    col = next(
        (c for c in ("adjusted_close", "Adj Close", "close", "Close") if c in df.columns),
        None,
    )
    if col is None:
        logger.warning("No close price column found for %s in %s", ticker, path)
        return {}

    prices = df[col].dropna()
    return {ts.date(): float(v) for ts, v in prices.items()}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_forward_return(
    ticker: str,
    analysis_date: str,
    data_dir: str,
    horizon_days: int = 60,
) -> float | None:
    """
    Compute the horizon_days-trading-day forward return for ticker from analysis_date.

    Finds the first available trading day t0 on or after analysis_date (handles
    weekends and holidays), then returns (price_{t0+horizon_days} - price_{t0}) / price_{t0}.

    Returns None when:
    - No Parquet file exists for ticker in data_dir/market/
    - analysis_date is after all available trading days
    - Fewer than horizon_days trading days remain after t0 (insufficient forward data)
    - Price at t0 is zero (division guard)
    """
    prices = _load_prices(ticker, data_dir)
    if not prices:
        return None

    t0_target = date.fromisoformat(analysis_date)
    sorted_dates = sorted(prices)

    # Advance to the first trading day on or after analysis_date
    start_candidates = [d for d in sorted_dates if d >= t0_target]
    if not start_candidates:
        return None
    t0 = start_candidates[0]

    t0_idx = sorted_dates.index(t0)
    t1_idx = t0_idx + horizon_days
    if t1_idx >= len(sorted_dates):
        return None  # insufficient forward data

    p0 = prices[t0]
    if p0 == 0:
        return None
    p1 = prices[sorted_dates[t1_idx]]
    return (p1 - p0) / p0


def _apply_label(decision: str, forward_return: float | None) -> bool | None:
    """
    Apply DJ-042 labeling rules: returns True/False/None.

    None is returned when forward_return is None (unlabeled, not incorrect).
    """
    if forward_return is None:
        return None
    if decision == "Buy":
        return forward_return > _LABEL_THRESHOLD
    if decision == "Sell":
        return forward_return < -_LABEL_THRESHOLD
    # Hold: correct within ±2%
    return abs(forward_return) <= _LABEL_THRESHOLD


# ---------------------------------------------------------------------------
# Labeling functions
# ---------------------------------------------------------------------------


def label_method_decisions(
    ensemble_outputs: list,  # list[EnsembleOutput] — avoids circular import
    data_dir: str,
    horizon_days: int = 60,
) -> list[MethodDecisionRecord]:
    """
    Label each method's collective_decision in a list of EnsembleOutputs.

    For each EnsembleOutput, extracts all four methods from method_comparison,
    computes the horizon_days-day forward return, and applies DJ-042 labeling rules.

    Returns a flat list with len = len(ensemble_outputs) * 4 records
    (one per method per output). Records where forward data is unavailable
    have outcome_correct=None.

    Parameters
    ----------
    ensemble_outputs : list[EnsembleOutput]
        Typically loaded from a phase9_collective.json baseline fixture or from
        a live baseline run.
    data_dir : str
        Root data directory containing market/ Parquet files.
    horizon_days : int
        Evaluation horizon in trading days (60 = primary, 20 = secondary).
    """
    now_iso = datetime.now(tz=UTC).isoformat()
    records: list[MethodDecisionRecord] = []

    for output in ensemble_outputs:
        fwd = compute_forward_return(
            output.ticker, output.as_of_date, data_dir, horizon_days
        )
        for method_name, decision_obj in output.method_comparison.items():
            if decision_obj.collective_decision is None:
                # No valid signals for this output — skip labeling
                continue
            records.append(
                MethodDecisionRecord(
                    ticker=output.ticker,
                    analysis_date=output.as_of_date,
                    method_name=method_name,
                    decision=decision_obj.collective_decision,
                    collective_confidence=decision_obj.collective_confidence,
                    forward_return=fwd,
                    outcome_correct=_apply_label(decision_obj.collective_decision, fwd),
                    horizon_days=horizon_days,
                    outcome_labeled_at=now_iso if fwd is not None else None,
                )
            )

    return records


def label_agent_decisions(
    ensemble_outputs: list,  # list[EnsembleOutput]
    data_dir: str,
    horizon_days: int = 60,
) -> list[DecisionRecord]:
    """
    Label individual agent signals from EnsembleOutput.signals.

    For each EnsembleOutput, extracts all non-None signals captured in
    EnsembleOutput.signals (populated by Phase 9 ensemble_runner), computes the
    forward return, and creates a labeled DecisionRecord per agent.

    Sentiment Agent records (confidence=0.0) are included; their decision=Hold
    will typically be labeled as outcome_correct depending on the forward return.

    Returns a flat list of DecisionRecords suitable for appending to
    AgentPerformanceHistory via performance_store.update_and_save().
    """
    now_iso = datetime.now(tz=UTC).isoformat()
    records: list[DecisionRecord] = []

    for output in ensemble_outputs:
        fwd = compute_forward_return(
            output.ticker, output.as_of_date, data_dir, horizon_days
        )
        for signal in output.signals:
            if signal is None:
                continue
            records.append(
                DecisionRecord(
                    ticker=output.ticker,
                    analysis_date=output.as_of_date,
                    agent_type=signal.agent_type,
                    decision=signal.decision,
                    confidence=signal.confidence,
                    forward_return=fwd,
                    outcome_correct=_apply_label(signal.decision, fwd),
                    horizon_days=horizon_days,
                    outcome_labeled_at=now_iso if fwd is not None else None,
                )
            )

    return records


# ---------------------------------------------------------------------------
# Aggregation + calibration helpers
# ---------------------------------------------------------------------------


def build_method_accuracy_report(
    records: list[MethodDecisionRecord],
    generated_at: str | None = None,
) -> MethodAccuracyReport:
    """
    Construct a MethodAccuracyReport from labeled MethodDecisionRecords.

    accuracy_by_method, n_labeled, tickers, and analysis_dates are derived
    automatically by MethodAccuracyReport's model_validator.

    Parameters
    ----------
    records : list[MethodDecisionRecord]
        May include records with outcome_correct=None (unlabeled). They are
        excluded from accuracy_by_method but included in the report for
        completeness.
    generated_at : str | None
        ISO 8601 timestamp. Defaults to now(UTC).
    """
    ts = generated_at or datetime.now(tz=UTC).isoformat()
    return MethodAccuracyReport(records=records, generated_at=ts)


def compute_divergence_rates(
    records: list[MethodDecisionRecord],
) -> dict[str, float]:
    """
    Compute pairwise method divergence rates across all (ticker, analysis_date) pairs.

    Divergence rate for (m1, m2) = fraction of (ticker, date) pairs where m1 and m2
    produce different collective_decisions.

    Only pairs where both methods have a record for the same (ticker, date, horizon_days)
    contribute to the denominator. Pairs where one method is absent or has
    collective_decision=None are excluded.

    Returns a dict with keys of the form "{m1_abbrev}_vs_{m2_abbrev}" for all 6
    canonical method pairs, sorted lexicographically:
      cw_vs_mv, cw_vs_pw, cw_vs_ca, mv_vs_pw, mv_vs_ca, pw_vs_ca

    where cw=confidence_weighted, mv=majority, pw=performance_weighted,
    ca=contrarian_adjusted.
    """
    abbrev = {
        "confidence_weighted": "cw",
        "majority": "mv",
        "performance_weighted": "pw",
        "contrarian_adjusted": "ca",
    }
    # Canonical order: cw, mv, pw, ca — matches docstring and test expectations
    methods = ["confidence_weighted", "majority", "performance_weighted", "contrarian_adjusted"]
    pairs = [
        (a, b)
        for i, a in enumerate(methods)
        for b in methods[i + 1:]
    ]

    # Index records by (ticker, analysis_date, horizon_days, method_name)
    index: dict[tuple[str, str, int, str], str] = {}
    for r in records:
        key = (r.ticker, r.analysis_date, r.horizon_days, r.method_name)
        index[key] = r.decision

    # Collect all unique (ticker, date, horizon) combos
    combos = {(r.ticker, r.analysis_date, r.horizon_days) for r in records}

    result: dict[str, float] = {}
    for m1, m2 in pairs:
        key_str = f"{abbrev[m1]}_vs_{abbrev[m2]}"
        n_total = 0
        n_diverge = 0
        for ticker, analysis_date, horizon_days in combos:
            d1 = index.get((ticker, analysis_date, horizon_days, m1))
            d2 = index.get((ticker, analysis_date, horizon_days, m2))
            if d1 is None or d2 is None:
                continue
            n_total += 1
            if d1 != d2:
                n_diverge += 1
        result[key_str] = n_diverge / n_total if n_total > 0 else 0.0

    return result


def build_calibration_report(
    bootstrap_records: list[DecisionRecord],
    real_records: list[DecisionRecord],
    method_records: list[MethodDecisionRecord],
    generated_at: str | None = None,
) -> CalibrationReport:
    """
    Compute weight calibration comparison from bootstrap vs real labeled records.

    bootstrap_records: DecisionRecords from the Phase 9/10 bootstrap (heuristic rules)
    real_records: DecisionRecords from live LLM ensemble runs labeled by forward return
    method_records: MethodDecisionRecords for divergence rate computation

    Imports performance_store.compute_weights to reuse the canonical weight formula.
    """
    from hifi.collective.performance_store import compute_weights

    ts = generated_at or datetime.now(tz=UTC).isoformat()

    bootstrap_labeled = [r for r in bootstrap_records if r.outcome_correct is not None]
    real_labeled = [r for r in real_records if r.outcome_correct is not None]
    all_labeled = bootstrap_labeled + real_labeled

    return CalibrationReport(
        bootstrap_weights=compute_weights(bootstrap_labeled),
        real_label_weights=compute_weights(real_labeled),
        combined_weights=compute_weights(all_labeled),
        divergence_rates=compute_divergence_rates(method_records),
        n_bootstrap_labeled=len(bootstrap_labeled),
        n_real_labeled=len(real_labeled),
        generated_at=ts,
    )
