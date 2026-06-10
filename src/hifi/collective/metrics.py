"""
Ensemble diversity and evaluation metrics (P4-E3).

Standalone metric functions implement the exact formulas from David §5.6.
compute_ensemble_metrics() aggregates across a set of EnsembleOutput objects
to produce phase-level summary statistics.
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
