"""
Confidence-weighted voting for the Phase 4 ensemble (P4-E3).

Implements David §12.2.2: Score(k) = sum of confidence for agents that voted k.
The winning option is argmax Score(k). On a tie, the conservative default is
"Hold" with collective_confidence = 0.0.

Diversity metrics (David §5.6) are computed inline:
- disagreement_entropy (§5.6.1): Shannon entropy over the vote distribution
- opinion_dispersion (§5.6.2): mean absolute deviation of confidence scores
"""

from __future__ import annotations

import math

from hifi.agents.schemas import AgentSignal
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
