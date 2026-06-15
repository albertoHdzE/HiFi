"""
Unit tests for run_debate_round() and run_debate_ensemble() (P12-E3-T2/T3).

No live LLM required. All LLM calls are intercepted via the `llm` parameter
(debate nodes) or monkeypatching (ensemble runner agents).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from hifi.agents.schemas import AgentSignal, FundamentalAnalysis, TechnicalAnalysis
from hifi.collective.debate import DebateTranscript, run_debate_round
from hifi.collective.schemas import EnsembleDecision, EnsembleOutput

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _StubLLM:
    """Returns a fixed string from invoke(). Used to test debate nodes."""

    model_name = "stub-model"

    def __init__(self, content: str = "") -> None:
        self._content = content

    def invoke(self, messages: list) -> MagicMock:
        result = MagicMock()
        result.content = self._content
        return result


# Revision JSON: forces all agents to revise to "Hold"
_HOLD_REVISION = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "After debate: neutral.",
    "key_concern": "Uncertainty remains.",
})


def _make_signal(
    agent_type: str,
    decision: str,
    ticker: str = "AAPL",
    confidence: float = 0.75,
) -> AgentSignal:
    return AgentSignal(
        ticker=ticker,
        as_of_date="2023-03-31",
        decision=decision,
        confidence=confidence,
        rationale=f"{agent_type} rationale for {decision}.",
        key_concern=f"{agent_type} primary concern.",
        model_id="qwen2.5-coder-32b",
        agent_type=agent_type,
    )


def _make_fundamental(signal: AgentSignal) -> FundamentalAnalysis:
    return FundamentalAnalysis(
        signal=signal,
        financial_ratios={"pe_ratio": 25.0},
        growth_metrics={"revenue_growth": 0.05},
        valuation_context={"dcf": "fair"},
        macro_snapshot={"fed_funds_rate": 5.25},
        prompt_version="fundamental_v1",
        latency_ms=100.0,
    )


def _make_technical(signal: AgentSignal) -> TechnicalAnalysis:
    return TechnicalAnalysis(
        signal=signal,
        technical_indicators={"rsi_14": 52.0},
        risk_metrics={"atr_14": 2.1},
        prompt_version="technical_v1",
        latency_ms=80.0,
    )


def _make_ensemble_decision(decision: str = "Buy") -> EnsembleDecision:
    return EnsembleDecision(
        collective_decision=decision,
        collective_confidence=0.75,
        n_valid_signals=2,
        agreement=True,
        disagreement_entropy=0.0,
        opinion_dispersion=0.0,
        agent_decisions=[decision, decision],
        agent_confidences=[0.75, 0.75],
        winning_score=1.5,
        total_score=1.5,
    )


# ---------------------------------------------------------------------------
# run_debate_round — unanimous vote → debate_skipped
# ---------------------------------------------------------------------------


def test_run_debate_round_unanimous_skips_debate():
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Buy"),
        _make_signal("risk", "Buy"),
    ]
    transcript = run_debate_round(
        signals=signals,
        ticker="AAPL",
        as_of_date="2023-03-31",
        llm=_StubLLM(),
    )
    assert transcript.debate_skipped is True


def test_run_debate_round_unanimous_no_challenge_turns():
    signals = [_make_signal("fundamental", "Hold"), _make_signal("technical", "Hold")]
    transcript = run_debate_round(
        signals=signals, ticker="JPM", as_of_date="2022-12-31", llm=_StubLLM()
    )
    assert transcript.challenge_turns == []


def test_run_debate_round_unanimous_no_response_turns():
    signals = [_make_signal("fundamental", "Sell"), _make_signal("technical", "Sell")]
    transcript = run_debate_round(
        signals=signals, ticker="XOM", as_of_date="2022-09-30", llm=_StubLLM()
    )
    assert transcript.response_turns == []


def test_run_debate_round_unanimous_revised_equals_initial():
    signals = [
        _make_signal("fundamental", "Hold"),
        _make_signal("technical", "Hold"),
        _make_signal("risk", "Hold"),
    ]
    transcript = run_debate_round(
        signals=signals, ticker="AAPL", as_of_date="2023-03-31", llm=_StubLLM()
    )
    assert transcript.revised_signals == signals


def test_run_debate_round_unanimous_vote_delta_unchanged():
    signals = [_make_signal("fundamental", "Buy"), _make_signal("technical", "Buy")]
    transcript = run_debate_round(
        signals=signals, ticker="AAPL", as_of_date="2023-03-31", llm=_StubLLM()
    )
    assert transcript.vote_delta == "unchanged"
    assert transcript.n_agents_changed_vote == 0


def test_run_debate_round_empty_signals_skips():
    transcript = run_debate_round(signals=[], ticker="AAPL", as_of_date="2023-03-31")
    assert transcript.debate_skipped is True
    assert transcript.majority_decision == "Hold"


# ---------------------------------------------------------------------------
# run_debate_round — split vote → full Oxford round
# ---------------------------------------------------------------------------


def test_run_debate_round_split_not_skipped():
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Buy"),
        _make_signal("risk", "Sell"),  # minority
    ]
    transcript = run_debate_round(
        signals=signals, ticker="AAPL", as_of_date="2023-03-31", llm=_StubLLM(_HOLD_REVISION)
    )
    assert transcript.debate_skipped is False


def test_run_debate_round_minority_generates_challenge():
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Buy"),
        _make_signal("risk", "Sell"),
    ]
    transcript = run_debate_round(
        signals=signals, ticker="AAPL", as_of_date="2023-03-31", llm=_StubLLM(_HOLD_REVISION)
    )
    # Only "risk" is minority
    assert len(transcript.challenge_turns) == 1
    assert transcript.challenge_turns[0].agent_type == "risk"
    assert transcript.challenge_turns[0].phase == "challenge"


def test_run_debate_round_majority_generates_responses():
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Buy"),
        _make_signal("risk", "Sell"),
    ]
    transcript = run_debate_round(
        signals=signals, ticker="AAPL", as_of_date="2023-03-31", llm=_StubLLM(_HOLD_REVISION)
    )
    response_agents = {t.agent_type for t in transcript.response_turns}
    assert response_agents == {"fundamental", "technical"}
    for turn in transcript.response_turns:
        assert turn.phase == "response"


def test_run_debate_round_all_agents_get_revised_signals():
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Buy"),
        _make_signal("risk", "Sell"),
    ]
    transcript = run_debate_round(
        signals=signals, ticker="AAPL", as_of_date="2023-03-31", llm=_StubLLM(_HOLD_REVISION)
    )
    assert len(transcript.revised_signals) == 3
    for sig in transcript.revised_signals:
        assert isinstance(sig, AgentSignal)


def test_run_debate_round_revised_signals_have_correct_agent_types():
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Buy"),
        _make_signal("risk", "Sell"),
    ]
    transcript = run_debate_round(
        signals=signals, ticker="AAPL", as_of_date="2023-03-31", llm=_StubLLM(_HOLD_REVISION)
    )
    agent_types = {s.agent_type for s in transcript.revised_signals}
    assert agent_types == {"fundamental", "technical", "risk"}


def test_run_debate_round_returns_valid_transcript():
    signals = [
        _make_signal("fundamental", "Hold"),
        _make_signal("technical", "Buy"),
    ]
    transcript = run_debate_round(
        signals=signals, ticker="XOM", as_of_date="2022-09-30", llm=_StubLLM(_HOLD_REVISION)
    )
    assert isinstance(transcript, DebateTranscript)
    assert transcript.ticker == "XOM"
    assert transcript.as_of_date == "2022-09-30"
    assert transcript.majority_decision in ("Buy", "Hold", "Sell")
    assert transcript.vote_delta in ("converged", "diverged", "unchanged")


def test_run_debate_round_majority_decision_set():
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Buy"),
        _make_signal("risk", "Sell"),
    ]
    transcript = run_debate_round(
        signals=signals, ticker="AAPL", as_of_date="2023-03-31", llm=_StubLLM(_HOLD_REVISION)
    )
    assert transcript.majority_decision == "Buy"
    assert transcript.minority_agents == ["risk"]


def test_run_debate_round_two_minority_agents():
    signals = [
        _make_signal("fundamental", "Buy"),
        _make_signal("technical", "Buy"),
        _make_signal("risk", "Sell"),
        _make_signal("macro", "Sell"),
        _make_signal("sentiment", "Buy"),
    ]
    transcript = run_debate_round(
        signals=signals, ticker="AAPL", as_of_date="2023-03-31", llm=_StubLLM(_HOLD_REVISION)
    )
    assert len(transcript.challenge_turns) == 2
    challenge_agents = {t.agent_type for t in transcript.challenge_turns}
    assert challenge_agents == {"risk", "macro"}


# ---------------------------------------------------------------------------
# run_debate_ensemble — structural tests with monkeypatching
# ---------------------------------------------------------------------------


def test_run_debate_ensemble_returns_ensemble_output(monkeypatch):
    """run_debate_ensemble produces EnsembleOutput with debate_transcript field."""
    import hifi.agents.ensemble_runner as er
    from hifi.agents.ensemble_runner import run_debate_ensemble
    from hifi.observability.tracing import NoOpTracer

    fund_sig = _make_signal("fundamental", "Buy")
    tech_sig = _make_signal("technical", "Hold")

    monkeypatch.setattr(er, "run_analysis", lambda **kw: _make_fundamental(fund_sig))
    monkeypatch.setattr(er, "run_technical_analysis", lambda **kw: _make_technical(tech_sig))
    monkeypatch.setattr(er, "get_weights", lambda **kw: {})
    monkeypatch.setattr(er, "verify_ensemble", lambda output: MagicMock())
    monkeypatch.setattr(er, "log_verification_scores", lambda *a, **kw: None)

    output = run_debate_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json="{}",
        agents=["fundamental", "technical"],
        tracer=NoOpTracer(),
        _debate_llm=_StubLLM(_HOLD_REVISION),
    )

    assert isinstance(output, EnsembleOutput)
    assert output.ticker == "AAPL"
    assert output.debate_transcript is not None


def test_run_debate_ensemble_debate_transcript_type(monkeypatch):
    import hifi.agents.ensemble_runner as er
    from hifi.agents.ensemble_runner import run_debate_ensemble
    from hifi.observability.tracing import NoOpTracer

    fund_sig = _make_signal("fundamental", "Buy")
    tech_sig = _make_signal("technical", "Sell")  # split → debate runs

    monkeypatch.setattr(er, "run_analysis", lambda **kw: _make_fundamental(fund_sig))
    monkeypatch.setattr(er, "run_technical_analysis", lambda **kw: _make_technical(tech_sig))
    monkeypatch.setattr(er, "get_weights", lambda **kw: {})
    monkeypatch.setattr(er, "verify_ensemble", lambda o: MagicMock())
    monkeypatch.setattr(er, "log_verification_scores", lambda *a, **kw: None)

    output = run_debate_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json="{}",
        agents=["fundamental", "technical"],
        tracer=NoOpTracer(),
        _debate_llm=_StubLLM(_HOLD_REVISION),
    )

    assert isinstance(output.debate_transcript, DebateTranscript)


def test_run_debate_ensemble_unanimous_produces_skipped_transcript(monkeypatch):
    """When agents agree, debate is skipped but transcript is still attached."""
    import hifi.agents.ensemble_runner as er
    from hifi.agents.ensemble_runner import run_debate_ensemble
    from hifi.observability.tracing import NoOpTracer

    fund_sig = _make_signal("fundamental", "Buy")
    tech_sig = _make_signal("technical", "Buy")  # unanimous

    monkeypatch.setattr(er, "run_analysis", lambda **kw: _make_fundamental(fund_sig))
    monkeypatch.setattr(er, "run_technical_analysis", lambda **kw: _make_technical(tech_sig))
    monkeypatch.setattr(er, "get_weights", lambda **kw: {})
    monkeypatch.setattr(er, "verify_ensemble", lambda o: MagicMock())
    monkeypatch.setattr(er, "log_verification_scores", lambda *a, **kw: None)

    output = run_debate_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json="{}",
        agents=["fundamental", "technical"],
        tracer=NoOpTracer(),
        _debate_llm=_StubLLM(),
    )

    assert output.debate_transcript is not None
    assert output.debate_transcript.debate_skipped is True


def test_run_debate_ensemble_mutually_exclusive_rag_flags(monkeypatch):
    from hifi.agents.ensemble_runner import run_debate_ensemble
    from hifi.observability.tracing import NoOpTracer

    with pytest.raises(AssertionError):
        run_debate_ensemble(
            ticker="AAPL",
            as_of_date="2023-03-31",
            snapshot_json="{}",
            use_rag=True,
            use_graphrag=True,
            tracer=NoOpTracer(),
        )


def test_run_debate_ensemble_signals_are_final_signals(monkeypatch):
    """EnsembleOutput.signals should be the post-debate signals."""
    import hifi.agents.ensemble_runner as er
    from hifi.agents.ensemble_runner import run_debate_ensemble
    from hifi.observability.tracing import NoOpTracer

    fund_sig = _make_signal("fundamental", "Buy")
    tech_sig = _make_signal("technical", "Sell")  # triggers debate

    monkeypatch.setattr(er, "run_analysis", lambda **kw: _make_fundamental(fund_sig))
    monkeypatch.setattr(er, "run_technical_analysis", lambda **kw: _make_technical(tech_sig))
    monkeypatch.setattr(er, "get_weights", lambda **kw: {})
    monkeypatch.setattr(er, "verify_ensemble", lambda o: MagicMock())
    monkeypatch.setattr(er, "log_verification_scores", lambda *a, **kw: None)

    output = run_debate_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json="{}",
        agents=["fundamental", "technical"],
        tracer=NoOpTracer(),
        _debate_llm=_StubLLM(_HOLD_REVISION),
    )

    # Signals in output should be the revised (post-debate) signals
    assert len(output.signals) == 2
    for sig in output.signals:
        assert isinstance(sig, AgentSignal)
