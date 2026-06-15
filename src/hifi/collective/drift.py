"""
Drift detection monitors for Phase 13 (P13-E5, DJ-077).

Three operational monitors for production readiness (Phase 14):

  1. Data drift (KS)        — input feature distributions shift
  2. Agent drift (chi-sq)   — individual agent decision proportions change
  3. Collective drift (CUSUM) — ensemble herding_coefficient drifts upward

Not implemented here (Phase 14+):
  - Concept drift (feature → outcome relationship change; requires live labels)
  - Covariate shift (distributional modelling beyond current scope)

Thresholds (calibrated against Phase 10 data in E5-T5):
  - KS p-value < 0.05 → data drift alert
  - chi-squared p-value < 0.05 → agent drift alert
  - CUSUM C_k > 3σ above baseline → collective drift alert

David reference: §14.4 (Drift Detection)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    import pandas as pd


class DriftResult(BaseModel):
    """
    Result of one drift check (P13-E5-T1).

    Fields
    ------
    drift_type : Literal["data", "agent", "collective"]
        Which kind of drift was checked.
    statistic : float
        The primary test statistic (KS D, chi-squared χ², or CUSUM C_k).
    p_value : float | None
        Two-sided p-value for KS and chi-squared. None for CUSUM (distribution-free).
    alert : bool
        True when the threshold is exceeded (drift detected).
    threshold : float
        The decision threshold applied (p-value cutoff for KS/chi-sq;
        3σ level for CUSUM).
    description : str
        Human-readable summary of the check result.
    feature_alerts : list[str]
        For data drift: list of feature names that individually exceeded p < 0.05.
        Empty for agent / collective drift.
    """

    drift_type: Literal["data", "agent", "collective"]
    statistic: float
    p_value: float | None = None
    alert: bool = False
    threshold: float = 0.05
    description: str = ""
    feature_alerts: list[str] = []


class DriftMonitor:
    """
    Three-monitor drift detection engine (P13-E5, DJ-077).

    Usage::

        monitor = DriftMonitor()
        result = monitor.check_data_drift(recent_df, baseline_df)
        if result.alert:
            ...
    """

    # ---------------------------------------------------------------------------
    # E5-T2: KS test for data drift
    # ---------------------------------------------------------------------------

    def check_data_drift(
        self,
        recent: pd.DataFrame,
        baseline: pd.DataFrame,
        p_threshold: float = 0.05,
    ) -> DriftResult:
        """
        Kolmogorov-Smirnov test for data drift across numeric feature columns (DJ-077).

        Applies scipy.stats.ks_2samp on each numeric column present in both DataFrames.
        Alerts if ANY feature's KS p-value < p_threshold.

        Parameters
        ----------
        recent : pd.DataFrame
            Recent market data window (e.g., last 20 trading days).
        baseline : pd.DataFrame
            Historical reference distribution (e.g., Phase 10 data, 2020-2022).
        p_threshold : float
            p-value below which a feature triggers an alert. Default 0.05.

        Returns
        -------
        DriftResult
            statistic = minimum p-value across all features (worst case).
            alert = True if any feature's p-value < p_threshold.
        """
        from scipy.stats import ks_2samp

        shared_cols = [
            c for c in recent.select_dtypes(include="number").columns
            if c in baseline.columns
        ]

        if not shared_cols:
            return DriftResult(
                drift_type="data",
                statistic=1.0,
                p_value=1.0,
                alert=False,
                threshold=p_threshold,
                description="No shared numeric features to compare.",
            )

        feature_pvalues: dict[str, float] = {}
        for col in shared_cols:
            r = recent[col].dropna().values
            b = baseline[col].dropna().values
            if len(r) < 2 or len(b) < 2:
                continue
            _, pval = ks_2samp(r, b)
            feature_pvalues[col] = float(pval)

        if not feature_pvalues:
            return DriftResult(
                drift_type="data",
                statistic=1.0,
                p_value=1.0,
                alert=False,
                threshold=p_threshold,
                description="Insufficient data for KS test.",
            )

        alerted = [col for col, pv in feature_pvalues.items() if pv < p_threshold]
        min_pval = min(feature_pvalues.values())
        alert = len(alerted) > 0

        desc = (
            f"KS test across {len(feature_pvalues)} features: "
            f"{'ALERT — ' + ', '.join(alerted) + ' below threshold' if alert else 'no drift detected'}. "  # noqa: E501
            f"Minimum p-value: {min_pval:.4f}."
        )

        return DriftResult(
            drift_type="data",
            statistic=min_pval,
            p_value=min_pval,
            alert=alert,
            threshold=p_threshold,
            description=desc,
            feature_alerts=alerted,
        )

    # ---------------------------------------------------------------------------
    # E5-T3: Chi-squared test for agent drift
    # ---------------------------------------------------------------------------

    def check_agent_drift(
        self,
        recent_decisions: list[str],
        baseline_dist: dict[str, float],
        p_threshold: float = 0.05,
    ) -> DriftResult:
        """
        Chi-squared goodness-of-fit test for agent decision distribution drift (DJ-077).

        Compares observed Buy/Hold/Sell proportions in recent_decisions to the
        expected proportions from baseline_dist (Phase 10 bootstrap, DJ-041).

        Parameters
        ----------
        recent_decisions : list[str]
            Recent agent decisions (e.g., last 20 decisions). Each element is
            "Buy", "Hold", or "Sell".
        baseline_dist : dict[str, float]
            Expected proportions, e.g. {"Buy": 0.45, "Hold": 0.38, "Sell": 0.17}.
            Values must sum to 1.0 (within floating-point tolerance).
        p_threshold : float
            Chi-squared p-value threshold. Default 0.05.

        Returns
        -------
        DriftResult
            statistic = chi-squared test statistic.
            p_value = chi-squared p-value.
            alert = True when p_value < p_threshold.
        """
        from scipy.stats import chisquare

        options = ["Buy", "Hold", "Sell"]
        n = len(recent_decisions)

        if n == 0:
            return DriftResult(
                drift_type="agent",
                statistic=0.0,
                p_value=1.0,
                alert=False,
                threshold=p_threshold,
                description="No recent decisions to test.",
            )

        observed = np.array([recent_decisions.count(opt) for opt in options], dtype=float)
        expected_props = np.array(
            [baseline_dist.get(opt, 0.0) for opt in options], dtype=float
        )

        # Normalise expected to sum to 1 (guard against rounding).
        total_exp = expected_props.sum()
        if total_exp == 0:
            return DriftResult(
                drift_type="agent",
                statistic=0.0,
                p_value=1.0,
                alert=False,
                threshold=p_threshold,
                description="Baseline distribution sums to zero; cannot test.",
            )
        expected_props = expected_props / total_exp
        expected_counts = expected_props * n

        # scipy chisquare requires all expected > 0 for valid test.
        if np.any(expected_counts == 0):
            return DriftResult(
                drift_type="agent",
                statistic=0.0,
                p_value=1.0,
                alert=False,
                threshold=p_threshold,
                description="Expected counts contain zeros; test underpowered.",
            )

        stat, pval = chisquare(observed, f_exp=expected_counts)
        alert = float(pval) < p_threshold

        desc = (
            f"Chi-squared test (n={n}): χ²={stat:.3f}, p={pval:.4f}. "
            f"{'ALERT — decision distribution shifted.' if alert else 'No agent drift detected.'}"
        )

        return DriftResult(
            drift_type="agent",
            statistic=float(stat),
            p_value=float(pval),
            alert=alert,
            threshold=p_threshold,
            description=desc,
        )

    # ---------------------------------------------------------------------------
    # E5-T4: CUSUM for collective drift
    # ---------------------------------------------------------------------------

    def check_collective_drift(
        self,
        herding_series: list[float],
        baseline_mean: float,
        baseline_std: float,
        k_sensitivity: float = 0.5,
        alert_multiplier: float = 3.0,
    ) -> DriftResult:
        """
        One-sided CUSUM test for upward drift in herding_coefficient (DJ-077).

        CUSUM formula (one-sided, upward):
            C_0 = 0
            C_k = max(0, C_{k-1} + x_k - (baseline_mean + k_sensitivity * baseline_std))

        Alert threshold: C_k > alert_multiplier * baseline_std

        This detects when the herding coefficient is systematically elevated above
        its baseline level — a signal that ensemble diversity is collapsing over time.

        Parameters
        ----------
        herding_series : list[float]
            Sequence of herding_coefficient values (each in [0, 1]).
        baseline_mean : float
            Expected mean herding from Phase 12 factorial A-condition results.
        baseline_std : float
            Expected standard deviation of herding from baseline.
        k_sensitivity : float
            CUSUM sensitivity parameter. Default 0.5 (standard choice).
        alert_multiplier : float
            Alert threshold = alert_multiplier * baseline_std. Default 3.0.

        Returns
        -------
        DriftResult
            statistic = final CUSUM value C_k.
            p_value = None (CUSUM is distribution-free).
            alert = True when C_k exceeds threshold.
        """
        if not herding_series or baseline_std == 0.0:
            return DriftResult(
                drift_type="collective",
                statistic=0.0,
                p_value=None,
                alert=False,
                threshold=alert_multiplier * max(baseline_std, 1e-9),
                description="Insufficient data or zero baseline_std for CUSUM.",
            )

        slack = baseline_mean + k_sensitivity * baseline_std
        threshold = alert_multiplier * baseline_std

        cusum: float = 0.0
        peak: float = 0.0
        for x in herding_series:
            cusum = max(0.0, cusum + x - slack)
            if cusum > peak:
                peak = cusum

        alert = cusum > threshold

        desc = (
            f"CUSUM (n={len(herding_series)}): C_k={cusum:.4f}, "
            f"threshold={threshold:.4f} (3σ). "
            f"{'ALERT — collective herding drift detected.' if alert else 'No collective drift detected.'}"  # noqa: E501
        )

        return DriftResult(
            drift_type="collective",
            statistic=cusum,
            p_value=None,
            alert=alert,
            threshold=threshold,
            description=desc,
        )
