"""
Ensemble diversity and evaluation metrics (P4-E3, P9-E2).

Standalone metric functions implement the exact formulas from David §5.6.
compute_ensemble_metrics() aggregates across a set of EnsembleOutput objects
to produce phase-level summary statistics.

Phase 9 (P9-E2) adds rolling temporal metrics:
  - herding_coefficient (David §5.6.3): κ = mean agreement rate over W periods
  - consensus_stability (David §5.6.4): S = fraction of consecutive-equal decisions
  - compute_rolling_metrics: all window sizes {5, 10, 20} in one call
"""

from __future__ import annotations

import math
from typing import Any

from hifi.collective.schemas import EnsembleOutput


def disagreement_entropy(decisions: list[str]) -> float:
    """
    Shannon entropy over the vote distribution (David §5.6.1).

    p_k = count(decisions == k) / n
    H = -sum(p_k * log2(p_k) for p_k > 0)

    Returns 0.0 for an empty list or a unanimous list.
    Maximum value is log2(3) ~= 1.585 (three equally likely options).
    """
    if not decisions:
        return 0.0
    n = len(decisions)
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d] = counts.get(d, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p_k = count / n
        if p_k > 0:
            entropy -= p_k * math.log2(p_k)
    return entropy


def opinion_dispersion(confidences: list[float]) -> float:
    """
    Mean absolute deviation of agent confidence scores (David §5.6.2).

    D = (1/N) * sum(|c_i - mean_c|)

    For two agents: D = |c_1 - c_2| / 2.
    Returns 0.0 for empty or single-element lists.
    """
    if len(confidences) < 2:
        return 0.0
    n = len(confidences)
    mean_c = sum(confidences) / n
    return sum(abs(c - mean_c) for c in confidences) / n


def pairwise_diversity(decisions: list[str]) -> float:
    """
    Fraction of agent pairs that disagree (David §5.6.5, categorical variant).

    pairwise_diversity = count(pairs where d[i] != d[j]) / total_pairs

    For n agents: total_pairs = n*(n-1)//2.
    Returns 0.0 for fewer than 2 agents (no pairs to compare).
    """
    n = len(decisions)
    if n < 2:
        return 0.0
    total_pairs = n * (n - 1) // 2
    disagreeing_pairs = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if decisions[i] != decisions[j]
    )
    return disagreeing_pairs / total_pairs


def compute_ensemble_metrics(outputs: dict[str, EnsembleOutput]) -> dict[str, Any]:
    """
    Compute phase-level ensemble metrics across a set of EnsembleOutput objects.

    Parameters
    ----------
    outputs : dict[str, EnsembleOutput]
        Mapping of ticker -> EnsembleOutput.

    Returns
    -------
    dict
        Metrics suitable for inclusion in phase4_ensemble.json.
    """
    if not outputs:
        return {
            "fundamental_compliance_rate": 0.0,
            "technical_compliance_rate": 0.0,
            "ensemble_agreement_rate": 0.0,
            "mean_disagreement_entropy": 0.0,
            "mean_opinion_dispersion": 0.0,
            "pairwise_diversity": 0.0,
            "mean_total_latency_ms": 0.0,
            "n_tickers": 0,
        }

    n = len(outputs)
    vals = list(outputs.values())

    fund_valid = sum(1 for o in vals if o.fundamental_analysis.signal is not None)
    tech_valid = sum(1 for o in vals if o.technical_analysis.signal is not None)
    fund_compliance = fund_valid / n
    tech_compliance = tech_valid / n

    agreements = sum(1 for o in vals if o.ensemble_decision.agreement)
    agreement_rate = agreements / n

    entropies = [o.ensemble_decision.disagreement_entropy for o in vals]
    mean_entropy = sum(entropies) / n if entropies else 0.0

    dispersions = [o.ensemble_decision.opinion_dispersion for o in vals]
    mean_dispersion = sum(dispersions) / n if dispersions else 0.0

    # Pairwise diversity: collect all decisions across all tickers (per ticker)
    per_ticker_diversity = []
    for o in vals:
        ed = o.ensemble_decision
        if ed.n_valid_signals >= 2:
            per_ticker_diversity.append(pairwise_diversity(ed.agent_decisions))
    mean_pairwise = (
        sum(per_ticker_diversity) / len(per_ticker_diversity)
        if per_ticker_diversity
        else 0.0
    )

    latencies = [o.latency_ms for o in vals]
    mean_latency = sum(latencies) / n if latencies else 0.0

    return {
        "fundamental_compliance_rate": round(fund_compliance, 4),
        "technical_compliance_rate": round(tech_compliance, 4),
        "ensemble_agreement_rate": round(agreement_rate, 4),
        "mean_disagreement_entropy": round(mean_entropy, 6),
        "mean_opinion_dispersion": round(mean_dispersion, 6),
        "pairwise_diversity": round(mean_pairwise, 4),
        "mean_total_latency_ms": round(mean_latency, 1),
        "n_tickers": n,
    }


# ---------------------------------------------------------------------------
# Phase 9: Rolling temporal metrics (David §5.6.3, §5.6.4)
# ---------------------------------------------------------------------------


def herding_coefficient(
    agent_votes_per_period: list[list[str]],
    w: int,
) -> float | None:
    """
    Herding coefficient over the last W analysis periods (David §5.6.3).

    agent_votes_per_period[t] = list of all agent vote strings at period t.
    a_t = fraction of agents voting with the plurality at period t.
    κ = mean(a_t) over the last W periods.

    Returns None when len(agent_votes_per_period) < W (insufficient history).
    κ near 1/3 (for three options) indicates independence.
    κ near 1.0 indicates systematic herding.

    Periods with empty vote lists are skipped defensively; the window still
    uses the last W entries regardless of whether some are empty.
    """
    if len(agent_votes_per_period) < w:
        return None

    window = agent_votes_per_period[-w:]
    agreement_rates: list[float] = []
    for votes in window:
        if not votes:
            continue
        n = len(votes)
        counts: dict[str, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        max_count = max(counts.values())
        agreement_rates.append(max_count / n)

    if not agreement_rates:
        return None
    return round(sum(agreement_rates) / len(agreement_rates), 6)


def consensus_stability(
    collective_decisions: list[str],
    w: int,
) -> float | None:
    """
    Consensus stability over the last W analysis periods (David §5.6.4).

    S = (1 / (W-1)) * Σ 𝟙(v_t = v_{t+1})  for t in the last W periods.

    Returns None when len(collective_decisions) < W or W < 2 (undefined).
    S = 1.0 means the collective decision never changed in the window.
    S = 0.0 means the collective decision changed every period.
    """
    if w < 2 or len(collective_decisions) < w:
        return None

    window = collective_decisions[-w:]
    n_stable = sum(1 for t in range(w - 1) if window[t] == window[t + 1])
    return round(n_stable / (w - 1), 6)


def compute_rolling_metrics(
    agent_votes_per_period: list[list[str]],
    collective_decisions: list[str],
    w_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float | None]:
    """
    Compute herding coefficient (κ) and consensus stability (S) for all window sizes.

    Returns a dict with keys "kappa_W{w}" and "stability_W{w}" for each w in
    w_values. Values are float or None when history is insufficient for that window.
    """
    result: dict[str, float | None] = {}
    for w in w_values:
        result[f"kappa_W{w}"] = herding_coefficient(agent_votes_per_period, w)
        result[f"stability_W{w}"] = consensus_stability(collective_decisions, w)
    return result
