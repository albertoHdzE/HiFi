"""
Aggregation methods for the Phase 9 Collective Decision Engine (P4-E3, P9-E1).

Phase 4 (P4-E3) — confidence_weighted_vote: David §12.2.2.
Phase 9 (P9-E1) adds three more methods (David §12.2.1, §12.2.3, §12.3):
  - majority_vote: equal-weight mode
  - performance_weighted_vote: historical accuracy weights per agent_type
  - contrarian_adjusted_vote: confidence-weighted base + contrarian discount
  - run_all_methods: convenience wrapper running all four simultaneously

Diversity metrics (David §5.6) are computed identically across all methods so
per-method comparison is apples-to-apples:
- disagreement_entropy (§5.6.1): Shannon entropy over the vote distribution
- opinion_dispersion (§5.6.2): mean absolute deviation of confidence scores
"""

from __future__ import annotations

import math

from hifi.agents.schemas import AgentSignal, ContrarianAnalysis
from hifi.collective.schemas import EnsembleDecision

_OPTIONS = ("Buy", "Hold", "Sell")


def confidence_weighted_vote(
    signals: list[AgentSignal | None],
) -> EnsembleDecision:
    """
    Aggregate a list of AgentSignals via confidence-weighted voting.

    Parameters
    ----------
    signals : list[AgentSignal | None]
        One entry per agent. None entries indicate agents that failed to
        produce a signal and are excluded from the vote.

    Returns
    -------
    EnsembleDecision
        Collective decision with diversity metrics.
        If no valid signals: collective_decision=None, all scores=0.
    """
    valid = [s for s in signals if s is not None]

    if not valid:
        return EnsembleDecision(
            collective_decision=None,
            collective_confidence=0.0,
            n_valid_signals=0,
            agreement=False,
            disagreement_entropy=0.0,
            opinion_dispersion=0.0,
            agent_decisions=[],
            agent_confidences=[],
            winning_score=0.0,
            total_score=0.0,
        )

    # Confidence-weighted scores per option
    scores: dict[str, float] = {k: 0.0 for k in _OPTIONS}
    for sig in valid:
        scores[sig.decision] += sig.confidence

    total_score = sum(scores.values())
    max_score = max(scores.values())

    # Detect tie: more than one option shares the maximum score
    tied_options = [k for k in _OPTIONS if scores[k] == max_score]
    if len(tied_options) > 1:
        winning_decision: str = "Hold"
        collective_confidence = 0.0
        winning_score = scores["Hold"]
    else:
        winning_decision = tied_options[0]
        winning_score = scores[winning_decision]
        collective_confidence = winning_score / total_score if total_score > 0 else 0.0

    # Disagreement entropy (David §5.6.1) -- count-proportion per option
    n = len(valid)
    vote_counts: dict[str, int] = {k: 0 for k in _OPTIONS}
    for sig in valid:
        vote_counts[sig.decision] += 1
    entropy = 0.0
    for k in _OPTIONS:
        p_k = vote_counts[k] / n
        if p_k > 0:
            entropy -= p_k * math.log2(p_k)

    # Opinion dispersion (David §5.6.2) -- mean absolute deviation of confidences
    confidences = [sig.confidence for sig in valid]
    mean_c = sum(confidences) / n
    dispersion = sum(abs(c - mean_c) for c in confidences) / n

    # Agreement: all agents voted identically
    unique_decisions = {sig.decision for sig in valid}
    agreement = len(unique_decisions) == 1

    return EnsembleDecision(
        collective_decision=winning_decision,  # type: ignore[arg-type]
        collective_confidence=round(collective_confidence, 6),
        n_valid_signals=n,
        agreement=agreement,
        disagreement_entropy=round(entropy, 6),
        opinion_dispersion=round(dispersion, 6),
        agent_decisions=[sig.decision for sig in valid],
        agent_confidences=[sig.confidence for sig in valid],
        winning_score=round(winning_score, 6),
        total_score=round(total_score, 6),
    )


def majority_vote(
    signals: list[AgentSignal | None],
) -> EnsembleDecision:
    """
    Aggregate via simple majority vote (David §12.2.1).

    Each agent casts one vote regardless of confidence. The option with the most
    votes wins. Tie-breaking and diversity metrics follow the same conventions as
    confidence_weighted_vote for cross-method comparability:
      - Tie: "Hold" with collective_confidence = 0.0
      - winning_score = winning vote count; total_score = n_valid
      - collective_confidence = winning_count / n_valid

    contrarian_confidence_discount = 1.0, review_flagged = False (neutral defaults;
    contrarian integration is applied only in contrarian_adjusted_vote).
    """
    valid = [s for s in signals if s is not None]

    if not valid:
        return EnsembleDecision(
            collective_decision=None,
            collective_confidence=0.0,
            n_valid_signals=0,
            agreement=False,
            disagreement_entropy=0.0,
            opinion_dispersion=0.0,
            agent_decisions=[],
            agent_confidences=[],
            winning_score=0.0,
            total_score=0.0,
        )

    n = len(valid)
    vote_counts: dict[str, int] = {k: 0 for k in _OPTIONS}
    for sig in valid:
        vote_counts[sig.decision] += 1

    max_count = max(vote_counts.values())
    tied_options = [k for k in _OPTIONS if vote_counts[k] == max_count]

    if len(tied_options) > 1:
        winning_decision: str = "Hold"
        collective_confidence = 0.0
        winning_score = float(vote_counts["Hold"])
    else:
        winning_decision = tied_options[0]
        winning_score = float(max_count)
        collective_confidence = max_count / n

    # Diversity metrics computed from votes and confidences (same as cw_vote)
    entropy = 0.0
    for k in _OPTIONS:
        p_k = vote_counts[k] / n
        if p_k > 0:
            entropy -= p_k * math.log2(p_k)

    confidences = [sig.confidence for sig in valid]
    mean_c = sum(confidences) / n
    dispersion = sum(abs(c - mean_c) for c in confidences) / n
    unique_decisions = {sig.decision for sig in valid}

    return EnsembleDecision(
        collective_decision=winning_decision,  # type: ignore[arg-type]
        collective_confidence=round(collective_confidence, 6),
        n_valid_signals=n,
        agreement=len(unique_decisions) == 1,
        disagreement_entropy=round(entropy, 6),
        opinion_dispersion=round(dispersion, 6),
        agent_decisions=[sig.decision for sig in valid],
        agent_confidences=[sig.confidence for sig in valid],
        winning_score=round(winning_score, 6),
        total_score=float(n),
    )


def performance_weighted_vote(
    signals: list[AgentSignal | None],
    weights: dict[str, float],
) -> EnsembleDecision:
    """
    Aggregate via historical accuracy weights per agent_type (David §12.2.3).

    Score(k) = Σ w_i · c_i · 𝟙(v_i = k)

    where w_i = weights.get(signal.agent_type, 1.0). Falls back to equal weight
    (1.0) for any agent_type not in weights. When weights is empty or all values
    are equal, this is identical to confidence_weighted_vote.

    Diversity metrics are computed from the unweighted votes (same formulas
    as confidence_weighted_vote for cross-method consistency).
    """
    valid = [s for s in signals if s is not None]

    if not valid:
        return EnsembleDecision(
            collective_decision=None,
            collective_confidence=0.0,
            n_valid_signals=0,
            agreement=False,
            disagreement_entropy=0.0,
            opinion_dispersion=0.0,
            agent_decisions=[],
            agent_confidences=[],
            winning_score=0.0,
            total_score=0.0,
        )

    scores: dict[str, float] = {k: 0.0 for k in _OPTIONS}
    for sig in valid:
        w_i = weights.get(sig.agent_type, 1.0)
        scores[sig.decision] += w_i * sig.confidence

    total_score = sum(scores.values())
    max_score = max(scores.values())
    tied_options = [k for k in _OPTIONS if scores[k] == max_score]

    if len(tied_options) > 1:
        winning_decision: str = "Hold"
        collective_confidence = 0.0
        winning_score = scores["Hold"]
    else:
        winning_decision = tied_options[0]
        winning_score = scores[winning_decision]
        collective_confidence = winning_score / total_score if total_score > 0 else 0.0

    # Diversity metrics from unweighted votes/confidences (per plan D-02)
    n = len(valid)
    vote_counts: dict[str, int] = {k: 0 for k in _OPTIONS}
    for sig in valid:
        vote_counts[sig.decision] += 1
    entropy = 0.0
    for k in _OPTIONS:
        p_k = vote_counts[k] / n
        if p_k > 0:
            entropy -= p_k * math.log2(p_k)

    confidences = [sig.confidence for sig in valid]
    mean_c = sum(confidences) / n
    dispersion = sum(abs(c - mean_c) for c in confidences) / n
    unique_decisions = {sig.decision for sig in valid}

    return EnsembleDecision(
        collective_decision=winning_decision,  # type: ignore[arg-type]
        collective_confidence=round(collective_confidence, 6),
        n_valid_signals=n,
        agreement=len(unique_decisions) == 1,
        disagreement_entropy=round(entropy, 6),
        opinion_dispersion=round(dispersion, 6),
        agent_decisions=[sig.decision for sig in valid],
        agent_confidences=[sig.confidence for sig in valid],
        winning_score=round(winning_score, 6),
        total_score=round(total_score, 6),
    )


def contrarian_adjusted_vote(
    signals: list[AgentSignal | None],
    contrarian: ContrarianAnalysis | None,
) -> EnsembleDecision:
    """
    Confidence-weighted base with contrarian discount and review flag (D-03).

    Steps:
    1. Run confidence_weighted_vote(signals) to get base decision.
    2. If contrarian is None: return base unchanged (discount=1.0, flagged=False).
    3. If contrarian is not None:
       discount = 1.0 - 0.5 * contrarian.confidence   (α = 0.5, D-03)
       collective_confidence = base.collective_confidence * discount
       review_flagged = contrarian.confidence > 0.70   (θ = 0.70, D-03)

    The winning direction (Buy/Hold/Sell) is never changed by discounting.
    contrarian_confidence_discount stores the factor (1 - α*c), not the product,
    so Phase 10 can reconstruct the undiscounted confidence for analysis.
    """
    base = confidence_weighted_vote(signals)

    if contrarian is None:
        return base  # contrarian_confidence_discount=1.0, review_flagged=False by default

    discount = round(1.0 - 0.5 * contrarian.confidence, 6)
    discounted_cc = round(base.collective_confidence * discount, 6)
    flagged = contrarian.confidence > 0.70

    return base.model_copy(
        update={
            "collective_confidence": discounted_cc,
            "contrarian_confidence_discount": discount,
            "review_flagged": flagged,
        }
    )


def run_all_methods(
    signals: list[AgentSignal | None],
    contrarian: ContrarianAnalysis | None,
    weights: dict[str, float],
) -> dict[str, EnsembleDecision]:
    """
    Run all four aggregation methods and return results keyed by method name (D-02).

    Keys: "majority", "confidence_weighted", "performance_weighted",
          "contrarian_adjusted"

    Called by ensemble_runner on every run_ensemble() invocation.
    method_comparison["confidence_weighted"] always equals ensemble_decision
    (same function, same inputs).
    """
    return {
        "majority": majority_vote(signals),
        "confidence_weighted": confidence_weighted_vote(signals),
        "performance_weighted": performance_weighted_vote(signals, weights),
        "contrarian_adjusted": contrarian_adjusted_vote(signals, contrarian),
    }
