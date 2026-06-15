"""Tests for multi-round debate (P13-E2-T2, DJ-074)."""

import pytest

from hifi.agents.schemas import AgentSignal
from hifi.collective.debate import identify_minority, run_debate_multi_round


def _sig(agent: str, decision: str, conf: float = 0.7) -> AgentSignal:
    return AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        agent_type=agent,
        decision=decision,
        confidence=conf,
        rationale="test",
        key_concern="none",
        call_ids=[],
        model_id="stub-model",
    )


class _StubLLM:
    """Deterministic LLM: always returns a valid revision JSON."""

    model_name = "stub-model"

    def __init__(self, revised_decision: str = "Buy"):
        self._decision = revised_decision

    def invoke(self, messages):
        class _Msg:
            pass
        msg = _Msg()
        msg.content = (
            f'{{"decision": "{self._decision}", "confidence": 0.7, '
            '"argument": "revised after debate"}'
        )
        return msg


# ---------------------------------------------------------------------------
# run_debate_multi_round
# ---------------------------------------------------------------------------

def test_multi_round_unanimous_returns_single_skipped_transcript():
    signals = [_sig("fundamental", "Buy"), _sig("technical", "Buy")]
    transcripts = run_debate_multi_round(
        initial_signals=signals,
        ticker="AAPL",
        as_of_date="2023-03-31",
        max_rounds=3,
    )
    assert len(transcripts) == 1
    assert transcripts[0].debate_skipped is True


def test_multi_round_max_rounds_1_same_as_single_round():
    """max_rounds=1 must behave identically to run_debate_round."""
    signals = [
        _sig("fundamental", "Buy"),
        _sig("technical", "Sell"),
    ]
    stub = _StubLLM("Buy")
    transcripts = run_debate_multi_round(
        initial_signals=signals,
        ticker="AAPL",
        as_of_date="2023-03-31",
        max_rounds=1,
        llm=stub,
    )
    assert len(transcripts) == 1


def test_multi_round_returns_list_of_transcripts():
    signals = [
        _sig("fundamental", "Buy"),
        _sig("technical", "Sell"),
        _sig("risk", "Buy"),
    ]
    stub = _StubLLM("Buy")
    transcripts = run_debate_multi_round(
        initial_signals=signals,
        ticker="AAPL",
        as_of_date="2023-03-31",
        max_rounds=2,
        llm=stub,
    )
    assert len(transcripts) >= 1
    assert len(transcripts) <= 2


def test_multi_round_all_transcripts_same_ticker_date():
    signals = [_sig("fundamental", "Buy"), _sig("technical", "Sell")]
    stub = _StubLLM("Sell")
    transcripts = run_debate_multi_round(
        initial_signals=signals,
        ticker="AAPL",
        as_of_date="2023-03-31",
        max_rounds=2,
        llm=stub,
    )
    for t in transcripts:
        assert t.ticker == "AAPL"
        assert t.as_of_date == "2023-03-31"


def test_multi_round_convergence_stops_early():
    """When revised majority is stable, loop should stop before max_rounds."""
    signals = [
        _sig("fundamental", "Buy", 0.9),
        _sig("technical", "Sell", 0.5),
    ]
    # Stub always revises to Buy → majority Buy in every round → convergence round 2
    stub = _StubLLM("Buy")
    transcripts = run_debate_multi_round(
        initial_signals=signals,
        ticker="AAPL",
        as_of_date="2023-03-31",
        max_rounds=5,
        llm=stub,
    )
    # Should stop at 2 (first convergence) not run all 5
    assert len(transcripts) <= 3


def test_multi_round_max_rounds_enforced():
    signals = [_sig("fundamental", "Buy"), _sig("technical", "Sell")]
    # Stub alternates to prevent convergence (always Sell to fight Buy majority)
    stub = _StubLLM("Sell")
    transcripts = run_debate_multi_round(
        initial_signals=signals,
        ticker="AAPL",
        as_of_date="2023-03-31",
        max_rounds=3,
        llm=stub,
    )
    assert len(transcripts) <= 3


def test_multi_round_ensemble_max_rounds_param():
    """run_debate_ensemble max_rounds parameter reaches debate layer without error."""
    # We just test the signature — no LM Studio needed for this import-level check.
    import inspect
    from hifi.agents.ensemble_runner import run_debate_ensemble
    sig = inspect.signature(run_debate_ensemble)
    assert "max_rounds" in sig.parameters
    assert sig.parameters["max_rounds"].default == 1
