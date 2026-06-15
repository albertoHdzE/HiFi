"""
Structured debate schemas and helpers for HiFi Phase 12 (P12-E3-T1, DJ-065, DJ-066).

Implements the Oxford 1-round debate protocol from David SS12.2.4:
  Phase 1: Independent analysis  (existing run_ensemble flow)
  Phase 2: Challenge             (minority agents challenge majority position)
  Phase 3: Response              (majority agents respond to challenges)
  Phase 4: Revision              (all agents revise after seeing full transcript)
  Phase 5: Final vote            (run_all_methods on revised signals)

This module provides:
  - DebateTurn:       atomic unit of debate participation (challenge/response/revision)
  - DebateTranscript: full record of one Oxford round for a (ticker, as_of_date) pair
  - identify_minority(): classify agents as minority or majority given initial signals
  - compute_vote_delta(): measure whether debate caused convergence or divergence

DebateTranscript instances are stored in data/interactions/ as Dataset Family D
artifacts (David SS8.6, DJ-066).

EnsembleOutput.debate_transcript (in collective/schemas.py) is set to None when
the no-debate path is used, so all existing code remains backward-compatible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator

from hifi.agents.schemas import AgentSignal

if TYPE_CHECKING:
    from hifi.observability.tracing import AbstractTracer

# Valid decision options -- must match AgentSignal.decision
_OPTIONS = ("Buy", "Hold", "Sell")


# ---------------------------------------------------------------------------
# Pydantic schemas (DJ-066)
# ---------------------------------------------------------------------------


class DebateTurn(BaseModel):
    """
    One agent's contribution during a single debate phase.

    Fields
    ------
    agent_type : str
        Identifies the agent (e.g. "technical", "fundamental", "risk").
    phase : Literal["challenge", "response", "revision"]
        Which phase of the Oxford round this turn belongs to.
    argument : str
        The text of the challenge, response, or revision rationale.
    revised_decision : str | None
        Only populated in the "revision" phase. Buy/Hold/Sell or None.
    revised_confidence : float | None
        Updated confidence [0, 1]. Only meaningful in "revision" phase.
    model_id : str
        Model that generated this turn (fine-tuned if deployed, base otherwise).
    """

    agent_type: str
    phase: Literal["challenge", "response", "revision"]
    argument: str
    revised_decision: str | None = None
    revised_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model_id: str

    @field_validator("argument")
    @classmethod
    def argument_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("argument must not be empty")
        return v

    @field_validator("revised_decision")
    @classmethod
    def valid_decision(cls, v: str | None) -> str | None:
        if v is not None and v not in _OPTIONS:
            raise ValueError(f"revised_decision must be Buy/Hold/Sell, got {v!r}")
        return v


class DebateTranscript(BaseModel):
    """
    Complete record of one Oxford 1-round debate for a (ticker, as_of_date) pair.

    Fields
    ------
    ticker : str
        The equity ticker being analysed.
    as_of_date : str
        ISO 8601 date string for the analysis.
    initial_signals : list[AgentSignal]
        Signals produced by the independent analysis phase (Phase 1).
    minority_agents : list[str]
        Agent types whose initial vote differed from the majority.
        Empty when debate_skipped=True (unanimous initial vote).
    majority_decision : str
        The plurality decision from initial_signals (Buy/Hold/Sell).
    challenge_turns : list[DebateTurn]
        Challenge arguments from minority agents (Phase 2).
        Empty when debate_skipped=True.
    response_turns : list[DebateTurn]
        Majority agents' responses to challenges (Phase 3).
        Empty when debate_skipped=True.
    revised_signals : list[AgentSignal]
        Signals produced after seeing the full transcript (Phase 4).
        Equal to initial_signals when debate_skipped=True.
    vote_delta : Literal["converged", "diverged", "unchanged"]
        Whether debate moved agents toward (converged) or away (diverged)
        from the initial majority, or had no effect (unchanged).
    n_agents_changed_vote : int
        Number of agents whose decision changed between initial and revised signals.
    debate_skipped : bool
        True when the initial vote was unanimous (no minority to challenge).
    """

    ticker: str
    as_of_date: str
    initial_signals: list[AgentSignal]
    minority_agents: list[str]
    majority_decision: str
    challenge_turns: list[DebateTurn] = Field(default_factory=list)
    response_turns: list[DebateTurn] = Field(default_factory=list)
    revised_signals: list[AgentSignal] = Field(default_factory=list)
    vote_delta: Literal["converged", "diverged", "unchanged"] = "unchanged"
    n_agents_changed_vote: int = 0
    debate_skipped: bool = False

    @field_validator("majority_decision")
    @classmethod
    def valid_majority(cls, v: str) -> str:
        if v not in _OPTIONS:
            raise ValueError(f"majority_decision must be Buy/Hold/Sell, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def identify_minority(
    signals: list[AgentSignal],
) -> tuple[list[str], str]:
    """
    Classify agents into minority and majority given their initial signals.

    The majority decision is the plurality vote (most votes).  If two decisions
    tie for the plurality, "Hold" is used as the tie-breaking default
    (consistent with confidence_weighted_vote tie-breaking in voting.py).

    Parameters
    ----------
    signals : list[AgentSignal]
        Initial signals from the independent analysis phase.

    Returns
    -------
    tuple[list[str], str]
        (minority_agent_types, majority_decision)
        minority_agent_types is empty when all agents agree (unanimous vote).
    """
    if not signals:
        return [], "Hold"

    vote_counts: dict[str, int] = {}
    for sig in signals:
        vote_counts[sig.decision] = vote_counts.get(sig.decision, 0) + 1

    max_count = max(vote_counts.values())
    tied_winners = [d for d, c in vote_counts.items() if c == max_count]

    majority_decision = "Hold" if len(tied_winners) > 1 else tied_winners[0]

    minority_agents = [
        sig.agent_type
        for sig in signals
        if sig.decision != majority_decision
    ]

    return minority_agents, majority_decision


def compute_vote_delta(
    initial_signals: list[AgentSignal],
    revised_signals: list[AgentSignal],
) -> tuple[Literal["converged", "diverged", "unchanged"], int]:
    """
    Compare initial and revised votes to determine debate effect.

    Convergence/divergence is measured relative to the initial majority decision:
    - converged: more agents agreed with the initial majority after debate
                 (minority agents moved toward consensus -- herding signal)
    - diverged:  fewer agents agreed with the initial majority after debate
                 (majority agents moved away -- increased disagreement)
    - unchanged: no agent changed its vote

    Parameters
    ----------
    initial_signals : list[AgentSignal]
        Signals before the debate round.
    revised_signals : list[AgentSignal]
        Signals after the revision phase.

    Returns
    -------
    tuple[Literal["converged", "diverged", "unchanged"], int]
        (delta_type, n_agents_changed_vote)
    """
    if not initial_signals or not revised_signals:
        return "unchanged", 0

    initial_by_type: dict[str, str] = {
        sig.agent_type: sig.decision for sig in initial_signals
    }
    revised_by_type: dict[str, str] = {
        sig.agent_type: sig.decision for sig in revised_signals
    }

    common_agents = set(initial_by_type) & set(revised_by_type)
    n_changed = sum(
        1
        for agent in common_agents
        if initial_by_type[agent] != revised_by_type[agent]
    )

    if n_changed == 0:
        return "unchanged", 0

    # Determine direction relative to initial majority
    _, initial_majority = identify_minority(initial_signals)
    initial_agreement = sum(
        1 for sig in initial_signals if sig.decision == initial_majority
    )
    revised_agreement = sum(
        1
        for sig in revised_signals
        if sig.agent_type in common_agents and sig.decision == initial_majority
    )

    if revised_agreement > initial_agreement:
        return "converged", n_changed
    elif revised_agreement < initial_agreement:
        return "diverged", n_changed
    else:
        # Votes changed but overall agreement with majority unchanged
        return "unchanged", n_changed


# ---------------------------------------------------------------------------
# Oxford debate round runner (P12-E3-T2)
# ---------------------------------------------------------------------------


def run_debate_round(
    signals: list[AgentSignal],
    ticker: str,
    as_of_date: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    llm: object | None = None,
) -> DebateTranscript:
    """
    Run one Oxford 1-round debate on initial agent signals (DJ-065).

    Steps
    -----
    1. identify_minority() — if unanimous, return skipped transcript.
    2. Each minority agent generates a challenge (challenge_node).
    3. Each majority agent generates a response (respond_node).
    4. All agents revise after seeing the full transcript (revise_node).
    5. compute_vote_delta() — measure herding vs. divergence.
    6. Return complete DebateTranscript.

    Parameters
    ----------
    signals : list[AgentSignal]
        Initial signals from the independent analysis phase.
    ticker : str
        Ticker being analysed.
    as_of_date : str
        ISO 8601 analysis date.
    data_dir : str | None
        Reserved for future retrieval integration. Not used in Phase 12.
    tracer : AbstractTracer | None
        Observability tracer (debate spans not yet wired).
    llm : object | None
        Optional LLM override injected for deterministic tests.
        Passed directly to challenge_node / respond_node / revise_node.

    Returns
    -------
    DebateTranscript
        Complete debate record. debate_skipped=True when vote was unanimous.
    """
    from hifi.collective.debate_nodes import challenge_node, respond_node, revise_node

    minority_agents, majority_decision = identify_minority(signals)
    majority_count = len(signals) - len(minority_agents)

    if not minority_agents:
        return DebateTranscript(
            ticker=ticker,
            as_of_date=as_of_date,
            initial_signals=signals,
            minority_agents=[],
            majority_decision=majority_decision,
            challenge_turns=[],
            response_turns=[],
            revised_signals=list(signals),
            vote_delta="unchanged",
            n_agents_changed_vote=0,
            debate_skipped=True,
        )

    minority_set = set(minority_agents)
    minority_signals = [s for s in signals if s.agent_type in minority_set]
    majority_signals = [s for s in signals if s.agent_type not in minority_set]

    # Phase 2: challenge
    challenge_turns: list[DebateTurn] = [
        challenge_node(
            signal=sig,
            majority_decision=majority_decision,
            majority_count=majority_count,
            total_agents=len(signals),
            llm=llm,
        )
        for sig in minority_signals
    ]

    # Phase 3: response
    response_turns: list[DebateTurn] = [
        respond_node(
            signal=sig,
            challenge_turns=challenge_turns,
            majority_decision=majority_decision,
            llm=llm,
        )
        for sig in majority_signals
    ]

    # Phase 4: revision (all agents)
    revised_signals: list[AgentSignal] = []
    for sig in signals:
        _, revised_sig = revise_node(
            signal=sig,
            challenge_turns=challenge_turns,
            response_turns=response_turns,
            majority_decision=majority_decision,
            llm=llm,
        )
        revised_signals.append(revised_sig)

    vote_delta, n_changed = compute_vote_delta(signals, revised_signals)

    return DebateTranscript(
        ticker=ticker,
        as_of_date=as_of_date,
        initial_signals=signals,
        minority_agents=minority_agents,
        majority_decision=majority_decision,
        challenge_turns=challenge_turns,
        response_turns=response_turns,
        revised_signals=revised_signals,
        vote_delta=vote_delta,
        n_agents_changed_vote=n_changed,
        debate_skipped=False,
    )
