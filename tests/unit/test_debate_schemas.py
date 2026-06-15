"""
Unit tests for debate schemas and helper functions (P12-E3-T1).

No LLM or live services required. All tests use deterministic signal fixtures.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hifi.agents.schemas import AgentSignal
from hifi.collective.debate import (
    DebateTranscript,
    DebateTurn,
    compute_vote_delta,
    identify_minority,
)
from hifi.collective.schemas import EnsembleOutput

# ---------------------------------------------------------------------------
# Deterministic AgentSignal factory
# ---------------------------------------------------------------------------

def _make_signal(
    agent_type: str,
    decision: str,
    confidence: float = 0.7,
    ticker: str = "AAPL",
    as_of_date: str = "2023-03-31",
) -> AgentSignal:
    return AgentSignal(
        ticker=ticker,
        as_of_date=as_of_date,
        decision=decision,
        confidence=confidence,
        rationale=f"{agent_type} rationale for {decision}",
        key_concern=f"{agent_type} key concern",
        model_id="qwen2.5-coder-32b",
        agent_type=agent_type,
    )


def _make_turn(
    agent_type: str,
    phase: str,
    argument: str = "Test argument",
    revised_decision: str | None = None,
    revised_confidence: float | None = None,
) -> DebateTurn:
    return DebateTurn(
        agent_type=agent_type,
        phase=phase,
        argument=argument,
        revised_decision=revised_decision,
        revised_confidence=revised_confidence,
        model_id="qwen2.5-coder-32b",
    )


# ---------------------------------------------------------------------------
# DebateTurn validation
# ---------------------------------------------------------------------------

def test_debate_turn_challenge_phase() -> None:
    turn = _make_turn("technical", "challenge")
    assert turn.phase == "challenge"
    assert turn.revised_decision is None
    assert turn.revised_confidence is None


def test_debate_turn_response_phase() -> None:
    turn = _make_turn("fundamental", "response", "I acknowledge the concern.")
    assert turn.phase == "response"


def test_debate_turn_revision_with_decision() -> None:
    turn = _make_turn(
        "risk",
        "revision",
        "After reviewing the debate I revise to Hold.",
        revised_decision="Hold",
        revised_confidence=0.65,
    )
    assert turn.revised_decision == "Hold"
    assert turn.revised_confidence == pytest.approx(0.65)


def test_debate_turn_rejects_invalid_phase() -> None:
    with pytest.raises(ValidationError):
        DebateTurn(
            agent_type="technical",
            phase="opinion",  # invalid
            argument="test",
            model_id="qwen2.5-coder-32b",
        )


def test_debate_turn_rejects_empty_argument() -> None:
    with pytest.raises(ValidationError):
        DebateTurn(
            agent_type="technical",
            phase="challenge",
            argument="   ",  # whitespace only
            model_id="qwen2.5-coder-32b",
        )


def test_debate_turn_rejects_invalid_revised_decision() -> None:
    with pytest.raises(ValidationError):
        DebateTurn(
            agent_type="technical",
            phase="revision",
            argument="My revised view.",
            revised_decision="Strong Buy",  # invalid
            model_id="qwen2.5-coder-32b",
        )


def test_debate_turn_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        DebateTurn(
            agent_type="technical",
            phase="revision",
            argument="My revised view.",
            revised_confidence=1.5,  # > 1.0
            model_id="qwen2.5-coder-32b",
        )


# ---------------------------------------------------------------------------
# DebateTranscript validation and roundtrip
# ---------------------------------------------------------------------------

def test_debate_transcript_minimal() -> None:
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Hold"),
    ]
    transcript = DebateTranscript(
        ticker="AAPL",
        as_of_date="2023-03-31",
        initial_signals=signals,
        minority_agents=["technical"],
        majority_decision="Buy",
        revised_signals=signals,
        vote_delta="unchanged",
        n_agents_changed_vote=0,
    )
    assert transcript.ticker == "AAPL"
    assert transcript.majority_decision == "Buy"
    assert transcript.debate_skipped is False


def test_debate_transcript_skipped() -> None:
    signals = [
        _make_signal("fundamental", "Hold"),
        _make_signal("technical", "Hold"),
        _make_signal("risk", "Hold"),
    ]
    transcript = DebateTranscript(
        ticker="JPM",
        as_of_date="2022-12-31",
        initial_signals=signals,
        minority_agents=[],
        majority_decision="Hold",
        revised_signals=signals,
        vote_delta="unchanged",
        n_agents_changed_vote=0,
        debate_skipped=True,
    )
    assert transcript.debate_skipped is True
    assert transcript.minority_agents == []


def test_debate_transcript_roundtrip_json() -> None:
    signals = [_make_signal("fundamental", "Buy"), _make_signal("technical", "Sell")]
    challenge = _make_turn("technical", "challenge", "RSI is overbought, Buy is premature.")
    response = _make_turn("fundamental", "response", "P/E supports Buy despite RSI.")
    transcript = DebateTranscript(
        ticker="AAPL",
        as_of_date="2023-03-31",
        initial_signals=signals,
        minority_agents=["technical"],
        majority_decision="Buy",
        challenge_turns=[challenge],
        response_turns=[response],
        revised_signals=signals,
        vote_delta="unchanged",
        n_agents_changed_vote=0,
    )
    json_str = transcript.model_dump_json()
    loaded = DebateTranscript.model_validate_json(json_str)
    assert loaded.ticker == transcript.ticker
    assert len(loaded.challenge_turns) == 1
    assert len(loaded.response_turns) == 1
    assert loaded.challenge_turns[0].phase == "challenge"


def test_debate_transcript_rejects_invalid_majority() -> None:
    with pytest.raises(ValidationError):
        DebateTranscript(
            ticker="AAPL",
            as_of_date="2023-03-31",
            initial_signals=[],
            minority_agents=[],
            majority_decision="Strong Buy",  # invalid
            revised_signals=[],
            vote_delta="unchanged",
            n_agents_changed_vote=0,
        )


def test_debate_transcript_rejects_invalid_vote_delta() -> None:
    with pytest.raises(ValidationError):
        DebateTranscript(
            ticker="AAPL",
            as_of_date="2023-03-31",
            initial_signals=[],
            minority_agents=[],
            majority_decision="Hold",
            revised_signals=[],
            vote_delta="polarized",  # invalid
            n_agents_changed_vote=0,
        )


# ---------------------------------------------------------------------------
# identify_minority
# ---------------------------------------------------------------------------

def test_identify_minority_split_vote_2v1() -> None:
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Buy"),
        _make_signal("risk", "Sell"),
    ]
    minority, majority = identify_minority(signals)
    assert majority == "Buy"
    assert minority == ["risk"]


def test_identify_minority_unanimous_empty() -> None:
    signals = [
        _make_signal("fundamental", "Hold"),
        _make_signal("technical", "Hold"),
        _make_signal("risk", "Hold"),
    ]
    minority, majority = identify_minority(signals)
    assert majority == "Hold"
    assert minority == []


def test_identify_minority_single_agent() -> None:
    signals = [_make_signal("fundamental", "Sell")]
    minority, majority = identify_minority(signals)
    assert majority == "Sell"
    assert minority == []


def test_identify_minority_empty_signals() -> None:
    minority, majority = identify_minority([])
    assert minority == []
    assert majority == "Hold"


def test_identify_minority_three_way_split_holds() -> None:
    """Three-way tie defaults to Hold as majority_decision."""
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Sell"),
        _make_signal("risk", "Hold"),
    ]
    minority, majority = identify_minority(signals)
    assert majority == "Hold"
    # All agents whose vote != Hold are minority
    assert set(minority) == {"fundamental", "technical"}


def test_identify_minority_4v1() -> None:
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Buy"),
        _make_signal("risk", "Buy"),
        _make_signal("macro", "Buy"),
        _make_signal("sentiment", "Hold"),
    ]
    minority, majority = identify_minority(signals)
    assert majority == "Buy"
    assert minority == ["sentiment"]


# ---------------------------------------------------------------------------
# compute_vote_delta
# ---------------------------------------------------------------------------

def test_compute_vote_delta_unchanged() -> None:
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Sell"),
    ]
    delta, n_changed = compute_vote_delta(signals, signals)
    assert delta == "unchanged"
    assert n_changed == 0


def test_compute_vote_delta_converged() -> None:
    """Minority agent changes vote to majority -> converged."""
    initial = [
        _make_signal("fundamental", "Buy", confidence=0.8),
        _make_signal("technical", "Buy", confidence=0.8),
        _make_signal("risk", "Sell", confidence=0.7),  # minority
    ]
    # After debate, risk agent moves to Buy
    revised = [
        _make_signal("fundamental", "Buy", confidence=0.8),
        _make_signal("technical", "Buy", confidence=0.8),
        _make_signal("risk", "Buy", confidence=0.5),  # changed
    ]
    delta, n_changed = compute_vote_delta(initial, revised)
    assert delta == "converged"
    assert n_changed == 1


def test_compute_vote_delta_diverged() -> None:
    """Majority agent changes away from consensus -> diverged."""
    initial = [
        _make_signal("fundamental", "Buy", confidence=0.8),
        _make_signal("technical", "Buy", confidence=0.8),
        _make_signal("risk", "Sell", confidence=0.7),
    ]
    # After debate, fundamental and technical both switch to Hold
    revised = [
        _make_signal("fundamental", "Hold", confidence=0.5),
        _make_signal("technical", "Hold", confidence=0.5),
        _make_signal("risk", "Sell", confidence=0.7),
    ]
    delta, n_changed = compute_vote_delta(initial, revised)
    assert delta == "diverged"
    assert n_changed == 2


def test_compute_vote_delta_empty_inputs() -> None:
    delta, n_changed = compute_vote_delta([], [])
    assert delta == "unchanged"
    assert n_changed == 0


def test_compute_vote_delta_counts_all_changers() -> None:
    initial = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Sell"),
        _make_signal("risk", "Hold"),
        _make_signal("macro", "Sell"),
        _make_signal("sentiment", "Buy"),
    ]
    # 3 agents change
    revised = [
        _make_signal("fundamental", "Hold"),  # changed
        _make_signal("technical", "Hold"),    # changed
        _make_signal("risk", "Hold"),         # unchanged
        _make_signal("macro", "Hold"),        # changed
        _make_signal("sentiment", "Buy"),     # unchanged
    ]
    _, n_changed = compute_vote_delta(initial, revised)
    assert n_changed == 3


# ---------------------------------------------------------------------------
# EnsembleOutput.debate_transcript field
# ---------------------------------------------------------------------------

def test_ensemble_output_debate_transcript_defaults_none() -> None:
    """EnsembleOutput.debate_transcript is None when not provided."""
    from hifi.agents.schemas import (
        FundamentalAnalysis,
        TechnicalAnalysis,
    )
    from hifi.collective.schemas import EnsembleDecision

    fund = FundamentalAnalysis(
        signal=_make_signal("fundamental", "Hold"),
        financial_ratios={},
        growth_metrics={},
        valuation_context={},
        macro_snapshot={},
        prompt_version="fundamental_v1",
    )
    tech = TechnicalAnalysis(
        signal=_make_signal("technical", "Hold"),
        technical_indicators={},
        risk_metrics={},
        prompt_version="technical_v1",
    )
    decision = EnsembleDecision(
        collective_decision="Hold",
        collective_confidence=0.7,
        n_valid_signals=2,
        agreement=True,
        disagreement_entropy=0.0,
        opinion_dispersion=0.0,
        agent_decisions=["Hold", "Hold"],
        agent_confidences=[0.7, 0.7],
        winning_score=1.4,
        total_score=1.4,
    )
    output = EnsembleOutput(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_analysis=fund,
        technical_analysis=tech,
        ensemble_decision=decision,
        latency_ms=100.0,
    )
    assert output.debate_transcript is None


def test_ensemble_output_accepts_debate_transcript() -> None:
    """EnsembleOutput.debate_transcript can be set to a DebateTranscript."""
    from hifi.agents.schemas import FundamentalAnalysis, TechnicalAnalysis
    from hifi.collective.schemas import EnsembleDecision

    signals = [_make_signal("fundamental", "Hold"), _make_signal("technical", "Hold")]
    transcript = DebateTranscript(
        ticker="AAPL",
        as_of_date="2023-03-31",
        initial_signals=signals,
        minority_agents=[],
        majority_decision="Hold",
        revised_signals=signals,
        vote_delta="unchanged",
        n_agents_changed_vote=0,
        debate_skipped=True,
    )

    fund = FundamentalAnalysis(
        signal=signals[0],
        financial_ratios={},
        growth_metrics={},
        valuation_context={},
        macro_snapshot={},
        prompt_version="fundamental_v1",
    )
    tech = TechnicalAnalysis(
        signal=signals[1],
        technical_indicators={},
        risk_metrics={},
        prompt_version="technical_v1",
    )
    decision = EnsembleDecision(
        collective_decision="Hold",
        collective_confidence=0.7,
        n_valid_signals=2,
        agreement=True,
        disagreement_entropy=0.0,
        opinion_dispersion=0.0,
        agent_decisions=["Hold", "Hold"],
        agent_confidences=[0.7, 0.7],
        winning_score=1.4,
        total_score=1.4,
    )
    output = EnsembleOutput(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_analysis=fund,
        technical_analysis=tech,
        ensemble_decision=decision,
        latency_ms=100.0,
        debate_transcript=transcript,
    )
    assert output.debate_transcript is not None
    assert output.debate_transcript.debate_skipped is True
