"""
Collective Decision Engine output schemas (P4-E2).

EnsembleDecision captures the aggregated output of confidence-weighted voting
(David §12.2.2) plus the diversity metrics from David §5.6 that are computable
with any number of agents >= 2.

EnsembleOutput is the full Phase 4 analysis envelope: both agent analyses plus
the collective decision. Phase 9 will extend this to N agents.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from hifi.agents.schemas import FundamentalAnalysis, TechnicalAnalysis


class EnsembleDecision(BaseModel):
    """
    Collective decision produced by confidence-weighted voting (David §12.2.2).

    Fields
    ------
    collective_decision : str | None
        The winning option (argmax of confidence-weighted scores). None when no valid
        signals were available (both agents failed to produce a signal).
    collective_confidence : float
        Fraction of total confidence that went to the winning option:
        winning_score / total_score. Range [0, 1]. 0.0 on a tie (conservative signal).
    n_valid_signals : int
        Number of non-None AgentSignals that contributed to this decision.
    agreement : bool
        True when all contributing agents voted identically.
    disagreement_entropy : float
        Shannon entropy over the vote distribution (David §5.6.1).
        0.0 = unanimous; log2(3) ≈ 1.585 = maximum disagreement (3 options).
    opinion_dispersion : float
        Mean absolute deviation of agent confidence scores (David §5.6.2).
        0.0 when all confidences are equal; higher when convictions diverge.
    agent_decisions : list[str]
        Individual agent decisions, in the order they were provided.
    agent_confidences : list[float]
        Individual agent confidence scores, in the same order.
    winning_score : float
        Sum of confidence scores for the winning option.
    total_score : float
        Sum of all confidence scores across all options (equals sum of all confidences).
    """

    collective_decision: Literal["Buy", "Hold", "Sell"] | None
    collective_confidence: float
    n_valid_signals: int
    agreement: bool
    disagreement_entropy: float
    opinion_dispersion: float
    agent_decisions: list[str]
    agent_confidences: list[float]
    winning_score: float
    total_score: float


class EnsembleOutput(BaseModel):
    """
    Full Phase 4 analysis: both agent analyses plus the collective decision.

    ticker and as_of_date identify the analysis context. latency_ms covers the
    total wall-clock time for both agent runs and the aggregation step.
    """

    ticker: str
    as_of_date: str
    fundamental_analysis: FundamentalAnalysis
    technical_analysis: TechnicalAnalysis
    ensemble_decision: EnsembleDecision
    latency_ms: float
