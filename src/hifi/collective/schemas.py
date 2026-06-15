"""
Collective Decision Engine output schemas (P4-E2, P8-E1, P9-E0, P10-E0).

EnsembleDecision captures the aggregated output of any voting method (David §12.2)
plus the diversity metrics from David §5.6 computable with any number of agents >= 2.
Phase 9 adds contrarian_confidence_discount and review_flagged for contrarian
integration (D-01, D-03).

EnsembleOutput is the full analysis envelope. Phase 9 adds signals (voting inputs
captured at ensemble time), aggregation_method (primary method used), and
method_comparison (all four aggregation methods run simultaneously — D-01, D-02).
Named analysis fields are kept for Phase 5/6 verification traceability.

DecisionRecord and AgentPerformanceHistory (D-07) track per-agent historical
accuracy to power performance-weighted aggregation (D-02).

Phase 10 adds:
- MethodDecisionRecord: per-method collective decision with outcome label (DJ-044)
- MethodAccuracyReport: aggregated accuracy across all four methods (DJ-044)
- CalibrationReport: weight calibration analysis from bootstrap vs real labels (DJ-048)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from hifi.agents.schemas import (
    AgentSignal,
    ContrarianAnalysis,
    FundamentalAnalysis,
    MacroAnalysis,
    RiskAnalysis,
    SentimentAnalysis,
    TechnicalAnalysis,
)
from hifi.collective.debate import DebateTranscript


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
    # Phase 9: contrarian integration fields (D-01, D-03)
    contrarian_confidence_discount: float = Field(default=1.0, ge=0.0, le=1.0)
    review_flagged: bool = False


class EnsembleOutput(BaseModel):
    """
    Full analysis envelope: all agent analyses plus the collective decision.

    Phase 4 fields (fundamental, technical) remain required for backward
    compatibility (DJ-038). Phase 8 fields (risk, macro, sentiment, contrarian)
    are Optional with None default — callers that run only the Phase 4 agent
    subset get None for the new fields transparently.

    ticker and as_of_date identify the analysis context. latency_ms covers the
    total wall-clock time for all agent runs and the aggregation step.
    """

    ticker: str
    as_of_date: str
    fundamental_analysis: FundamentalAnalysis
    technical_analysis: TechnicalAnalysis
    ensemble_decision: EnsembleDecision
    latency_ms: float
    # Phase 8 new agents (all Optional, backward-compatible)
    risk_analysis: RiskAnalysis | None = None
    macro_analysis: MacroAnalysis | None = None
    sentiment_analysis: SentimentAnalysis | None = None
    contrarian_analysis: ContrarianAnalysis | None = None
    # Phase 9: N-generic aggregation pipeline (D-01, D-02)
    signals: list[AgentSignal] = Field(default_factory=list)
    aggregation_method: str = "confidence_weighted"
    method_comparison: dict[str, EnsembleDecision] = Field(default_factory=dict)
    # Phase 12: structured debate transcript (DJ-066). None when no-debate path used.
    debate_transcript: DebateTranscript | None = None


class DecisionRecord(BaseModel):
    """
    Single agent decision at a historical quarter-end with optional outcome label (D-07).

    outcome_correct is None until the forward date passes and a label is assigned.
    horizon_days is the evaluation horizon (60 = primary, 20 = secondary per D-04).
    forward_return is the realised return over horizon_days trading days.
    """

    ticker: str
    analysis_date: str   # ISO 8601 quarter-end
    agent_type: str      # "fundamental" | "technical" | "risk" | "macro"
    decision: str        # "Buy" | "Hold" | "Sell"
    confidence: float
    outcome_correct: bool | None = None
    outcome_labeled_at: str | None = None
    horizon_days: int = 60
    forward_return: float | None = None

    @model_validator(mode="after")
    def _validate_fields(self) -> DecisionRecord:
        if self.decision not in {"Buy", "Hold", "Sell"}:
            raise ValueError(
                f"decision must be Buy/Hold/Sell, got {self.decision!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.horizon_days <= 0:
            raise ValueError(
                f"horizon_days must be positive, got {self.horizon_days}"
            )
        return self


class AgentPerformanceHistory(BaseModel):
    """
    Persistent store of per-agent decision records and derived accuracy weights (D-07).

    weights maps agent_type -> historical accuracy [0, 1].
    n_labeled is auto-computed: count of records where outcome_correct is not None.
    last_updated is an ISO 8601 timestamp set by the bootstrap or update routine.
    """

    records: list[DecisionRecord]
    weights: dict[str, float]
    last_updated: str
    n_labeled: int = 0

    @model_validator(mode="after")
    def _compute_n_labeled(self) -> AgentPerformanceHistory:
        self.n_labeled = sum(
            1 for r in self.records if r.outcome_correct is not None
        )
        return self


# ---------------------------------------------------------------------------
# Phase 10: method-level accuracy tracking (DJ-044)
# ---------------------------------------------------------------------------

_VALID_METHODS = frozenset(
    {"confidence_weighted", "majority", "performance_weighted", "contrarian_adjusted"}
)


class MethodDecisionRecord(BaseModel):
    """
    Single collective method decision at a historical date with optional outcome label.

    Tracks whether a specific aggregation method's collective_decision was correct
    for a given (ticker, analysis_date) pair. Distinct from DecisionRecord, which
    tracks individual agent decisions: a method aggregates across all agents and has
    no agent_type (DJ-044).

    Two records are written per (ticker, date, method): one for horizon_days=60
    and one for horizon_days=20 (DJ-052), allowing comparison of method accuracy
    at different evaluation windows.

    outcome_correct is None until the forward window passes and a label is applied.
    forward_return is the realised return over horizon_days trading days starting
    from the first available trading day on or after analysis_date.
    """

    ticker: str
    analysis_date: str              # ISO 8601 quarter-end
    method_name: str                # one of the 4 canonical aggregation method keys
    decision: Literal["Buy", "Hold", "Sell"]
    collective_confidence: float    # from EnsembleDecision.collective_confidence
    forward_return: float | None = None
    outcome_correct: bool | None = None
    horizon_days: int = 60          # 60 = primary (DJ-052); 20 = secondary
    outcome_labeled_at: str | None = None

    @model_validator(mode="after")
    def _validate_fields(self) -> MethodDecisionRecord:
        if self.method_name not in _VALID_METHODS:
            raise ValueError(
                f"method_name must be one of {sorted(_VALID_METHODS)}, "
                f"got {self.method_name!r}"
            )
        if not (0.0 <= self.collective_confidence <= 1.0):
            raise ValueError(
                f"collective_confidence must be in [0, 1], "
                f"got {self.collective_confidence}"
            )
        if self.horizon_days <= 0:
            raise ValueError(
                f"horizon_days must be positive, got {self.horizon_days}"
            )
        return self


class MethodAccuracyReport(BaseModel):
    """
    Aggregated accuracy report across all four aggregation methods (DJ-044).

    accuracy_by_method, n_labeled, tickers, and analysis_dates are all derived
    from the records list by a model_validator — they are never set independently.
    This ensures consistency between the stored records and the summary statistics.

    Accuracy convention (matching Phase 5 HR/GR):
      accuracy(method) = n_correct / (n_correct + n_incorrect)
    Records with outcome_correct=None are excluded from both numerator and
    denominator (unlabeled != incorrect).

    When multiple horizons are present in the records, accuracy_by_method aggregates
    across all of them. Use a filtered subset of records for horizon-specific accuracy.
    """

    records: list[MethodDecisionRecord]
    accuracy_by_method: dict[str, float] = Field(default_factory=dict)
    n_labeled: int = 0
    tickers: list[str] = Field(default_factory=list)
    analysis_dates: list[str] = Field(default_factory=list)
    generated_at: str

    @model_validator(mode="after")
    def _compute_derived_fields(self) -> MethodAccuracyReport:
        labeled = [r for r in self.records if r.outcome_correct is not None]
        self.n_labeled = len(labeled)

        # accuracy per method: n_correct / n_labeled_for_method
        correct_by: dict[str, int] = {}
        total_by: dict[str, int] = {}
        for r in labeled:
            correct_by[r.method_name] = correct_by.get(r.method_name, 0) + (
                1 if r.outcome_correct else 0
            )
            total_by[r.method_name] = total_by.get(r.method_name, 0) + 1

        self.accuracy_by_method = {
            m: correct_by[m] / total_by[m] for m in total_by
        }

        # Derived metadata from records
        self.tickers = sorted({r.ticker for r in self.records})
        self.analysis_dates = sorted({r.analysis_date for r in self.records})
        return self


class CalibrationReport(BaseModel):
    """
    Weight calibration analysis: bootstrap heuristics vs real LLM labels (P10-E4).

    bootstrap_weights: accuracy weights derived from proxy-rule bootstrap records
        (2018-Q1 through 2022-Q4, RSI/Sharpe heuristics, DJ-041).
    real_label_weights: accuracy weights derived from records labeled by live LLM
        ensemble runs (Phase 10+ analysis dates).
    combined_weights: accuracy weights from all labeled records (bootstrap + real).
    divergence_rates: pairwise method divergence — fraction of (ticker, date) pairs
        where two methods produce different collective_decisions. Keys follow the
        pattern "{m1}_vs_{m2}" in canonical order (e.g., "cw_vs_mv").
    """

    bootstrap_weights: dict[str, float]
    real_label_weights: dict[str, float]
    combined_weights: dict[str, float]
    divergence_rates: dict[str, float]
    n_bootstrap_labeled: int
    n_real_labeled: int
    generated_at: str
