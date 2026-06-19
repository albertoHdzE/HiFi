"""
Unit tests for debate LangGraph nodes (P12-E3-T4).

challenge_node, respond_node, revise_node — no live LLM required.
Each test injects a _ResponseLLM or _FailLLM via the `llm` parameter.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from hifi.agents.schemas import AgentSignal
from hifi.collective.debate import DebateTurn
from hifi.collective.debate_nodes import challenge_node, respond_node, revise_node

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _ResponseLLM:
    """Returns a fixed string from invoke(). model_name is queryable."""

    model_name = "stub-model"

    def __init__(self, content: str = "Stub argument text.") -> None:
        self._content = content

    def invoke(self, messages: list) -> MagicMock:
        result = MagicMock()
        result.content = self._content
        return result


class _EmptyLLM(_ResponseLLM):
    """Returns empty string — tests whitespace-fallback paths."""

    def __init__(self) -> None:
        super().__init__(content="")


class _FailLLM:
    """Raises on invoke() — tests LLM failure fallback paths."""

    model_name = "fail-model"

    def invoke(self, messages: list) -> None:
        raise RuntimeError("LLM unavailable")


_VALID_REVISION_JSON = json.dumps({
    "decision": "Hold",
    "confidence": 0.55,
    "rationale": "After reviewing the debate I revise to neutral.",
    "key_concern": "Rate uncertainty persists.",
})


def _make_signal(
    agent_type: str = "technical",
    decision: str = "Sell",
    ticker: str = "AAPL",
    confidence: float = 0.7,
) -> AgentSignal:
    return AgentSignal(
        ticker=ticker,
        as_of_date="2023-03-31",
        decision=decision,
        confidence=confidence,
        rationale=f"{agent_type} analysis supports {decision}.",
        key_concern=f"{agent_type} primary risk.",
        model_id="qwen2.5-coder-32b",
        agent_type=agent_type,
    )


def _make_turn(agent_type: str, phase: str, argument: str = "Valid argument.") -> DebateTurn:
    return DebateTurn(
        agent_type=agent_type,
        phase=phase,
        argument=argument,
        model_id="stub-model",
    )


# ---------------------------------------------------------------------------
# challenge_node
# ---------------------------------------------------------------------------


def test_challenge_node_returns_debate_turn():
    turn = challenge_node(
        signal=_make_signal("technical", "Sell"),
        majority_decision="Buy",
        majority_count=3,
        total_agents=4,
        llm=_ResponseLLM("RSI overbought; Buy is premature."),
    )
    assert isinstance(turn, DebateTurn)


def test_challenge_node_phase_is_challenge():
    turn = challenge_node(
        signal=_make_signal("risk", "Hold"),
        majority_decision="Buy",
        majority_count=2,
        total_agents=3,
        llm=_ResponseLLM(),
    )
    assert turn.phase == "challenge"


def test_challenge_node_agent_type_matches_signal():
    signal = _make_signal("technical", "Sell")
    turn = challenge_node(
        signal=signal,
        majority_decision="Buy",
        majority_count=2,
        total_agents=3,
        llm=_ResponseLLM(),
    )
    assert turn.agent_type == "technical"


def test_challenge_node_argument_non_empty():
    turn = challenge_node(
        signal=_make_signal("macro", "Hold"),
        majority_decision="Buy",
        majority_count=3,
        total_agents=4,
        llm=_ResponseLLM("Yield curve inverted."),
    )
    assert turn.argument.strip() != ""


def test_challenge_node_uses_stub_model_id():
    turn = challenge_node(
        signal=_make_signal("fundamental", "Hold"),
        majority_decision="Sell",
        majority_count=2,
        total_agents=3,
        llm=_ResponseLLM(),
    )
    assert turn.model_id == "stub-model"


def test_challenge_node_llm_failure_returns_fallback_turn():
    turn = challenge_node(
        signal=_make_signal("technical", "Sell"),
        majority_decision="Buy",
        majority_count=2,
        total_agents=3,
        llm=_FailLLM(),
    )
    assert isinstance(turn, DebateTurn)
    assert turn.phase == "challenge"
    assert turn.argument.strip() != ""


def test_challenge_node_llm_failure_model_id_is_fail_model():
    turn = challenge_node(
        signal=_make_signal("risk", "Hold"),
        majority_decision="Buy",
        majority_count=3,
        total_agents=4,
        llm=_FailLLM(),
    )
    assert turn.model_id == "fail-model"


def test_challenge_node_empty_response_falls_back():
    """Empty LLM response triggers fallback argument (not empty DebateTurn)."""
    turn = challenge_node(
        signal=_make_signal("technical", "Sell"),
        majority_decision="Buy",
        majority_count=2,
        total_agents=3,
        llm=_EmptyLLM(),
    )
    assert turn.argument.strip() != ""


# ---------------------------------------------------------------------------
# respond_node
# ---------------------------------------------------------------------------


def test_respond_node_returns_debate_turn():
    turn = respond_node(
        signal=_make_signal("fundamental", "Buy"),
        challenge_turns=[_make_turn("technical", "challenge", "RSI overbought.")],
        majority_decision="Buy",
        llm=_ResponseLLM("P/E supports Buy despite RSI."),
    )
    assert isinstance(turn, DebateTurn)


def test_respond_node_phase_is_response():
    turn = respond_node(
        signal=_make_signal("fundamental", "Buy"),
        challenge_turns=[],
        majority_decision="Buy",
        llm=_ResponseLLM(),
    )
    assert turn.phase == "response"


def test_respond_node_agent_type_matches_signal():
    signal = _make_signal("risk", "Hold")
    turn = respond_node(
        signal=signal,
        challenge_turns=[],
        majority_decision="Hold",
        llm=_ResponseLLM(),
    )
    assert turn.agent_type == "risk"


def test_respond_node_argument_non_empty():
    turn = respond_node(
        signal=_make_signal("fundamental", "Buy"),
        challenge_turns=[_make_turn("technical", "challenge")],
        majority_decision="Buy",
        llm=_ResponseLLM("Still bullish based on fundamentals."),
    )
    assert turn.argument.strip() != ""


def test_respond_node_llm_failure_returns_fallback_turn():
    turn = respond_node(
        signal=_make_signal("macro", "Sell"),
        challenge_turns=[],
        majority_decision="Sell",
        llm=_FailLLM(),
    )
    assert isinstance(turn, DebateTurn)
    assert turn.phase == "response"
    assert turn.argument.strip() != ""


def test_respond_node_empty_challenges():
    """No challenges raised should still produce a valid response turn."""
    turn = respond_node(
        signal=_make_signal("fundamental", "Hold"),
        challenge_turns=[],
        majority_decision="Hold",
        llm=_ResponseLLM("Maintaining position."),
    )
    assert turn.phase == "response"


def test_respond_node_multiple_challenges():
    challenges = [
        _make_turn("technical", "challenge", "RSI overbought."),
        _make_turn("risk", "challenge", "VaR elevated."),
    ]
    turn = respond_node(
        signal=_make_signal("fundamental", "Buy"),
        challenge_turns=challenges,
        majority_decision="Buy",
        llm=_ResponseLLM("Acknowledged concerns, still Buy."),
    )
    assert isinstance(turn, DebateTurn)


# ---------------------------------------------------------------------------
# revise_node
# ---------------------------------------------------------------------------


def test_revise_node_returns_tuple():
    result = revise_node(
        signal=_make_signal("technical", "Sell"),
        challenge_turns=[],
        response_turns=[],
        majority_decision="Buy",
        llm=_ResponseLLM(_VALID_REVISION_JSON),
    )
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_revise_node_first_element_is_debate_turn():
    turn, _ = revise_node(
        signal=_make_signal("technical", "Sell"),
        challenge_turns=[],
        response_turns=[],
        majority_decision="Buy",
        llm=_ResponseLLM(_VALID_REVISION_JSON),
    )
    assert isinstance(turn, DebateTurn)
    assert turn.phase == "revision"


def test_revise_node_second_element_is_agent_signal():
    _, revised = revise_node(
        signal=_make_signal("risk", "Hold"),
        challenge_turns=[],
        response_turns=[],
        majority_decision="Sell",
        llm=_ResponseLLM(_VALID_REVISION_JSON),
    )
    assert isinstance(revised, AgentSignal)


def test_revise_node_updates_decision_from_json():
    _, revised = revise_node(
        signal=_make_signal("technical", "Sell"),
        challenge_turns=[],
        response_turns=[],
        majority_decision="Buy",
        llm=_ResponseLLM(_VALID_REVISION_JSON),
    )
    assert revised.decision == "Hold"


def test_revise_node_updates_confidence_from_json():
    _, revised = revise_node(
        signal=_make_signal("technical", "Sell"),
        challenge_turns=[],
        response_turns=[],
        majority_decision="Buy",
        llm=_ResponseLLM(_VALID_REVISION_JSON),
    )
    assert abs(revised.confidence - 0.55) < 0.01


def test_revise_node_preserves_agent_type():
    _, revised = revise_node(
        signal=_make_signal("macro", "Buy"),
        challenge_turns=[],
        response_turns=[],
        majority_decision="Sell",
        llm=_ResponseLLM(_VALID_REVISION_JSON),
    )
    assert revised.agent_type == "macro"


def test_revise_node_preserves_ticker_and_date():
    signal = _make_signal("fundamental", "Buy", ticker="JPM")
    _, revised = revise_node(
        signal=signal,
        challenge_turns=[],
        response_turns=[],
        majority_decision="Hold",
        llm=_ResponseLLM(_VALID_REVISION_JSON),
    )
    assert revised.ticker == "JPM"
    assert revised.as_of_date == "2023-03-31"


def test_revise_node_llm_failure_preserves_original_decision():
    original = _make_signal("risk", "Buy")
    _, revised = revise_node(
        signal=original,
        challenge_turns=[],
        response_turns=[],
        majority_decision="Sell",
        llm=_FailLLM(),
    )
    assert revised.decision == "Buy"
    assert revised.confidence == pytest.approx(0.7)


def test_revise_node_invalid_json_preserves_original():
    original = _make_signal("sentiment", "Hold")
    _, revised = revise_node(
        signal=original,
        challenge_turns=[],
        response_turns=[],
        majority_decision="Sell",
        llm=_ResponseLLM("This is not valid JSON at all."),
    )
    assert revised.decision == "Hold"


def test_revise_node_invalid_decision_in_json_falls_back():
    """LLM returns JSON with invalid decision — original is preserved."""
    invalid_json = json.dumps({
        "decision": "Strong Buy",  # not in Buy/Hold/Sell
        "confidence": 0.9,
        "rationale": "Very bullish.",
        "key_concern": "None.",
    })
    original = _make_signal("fundamental", "Buy")
    _, revised = revise_node(
        signal=original,
        challenge_turns=[],
        response_turns=[],
        majority_decision="Hold",
        llm=_ResponseLLM(invalid_json),
    )
    assert revised.decision == "Buy"


def test_revise_node_confidence_clamped_to_one():
    """LLM returns confidence > 1 — should be clamped to 1.0."""
    too_high = json.dumps({
        "decision": "Buy",
        "confidence": 2.5,
        "rationale": "Very confident.",
        "key_concern": "None.",
    })
    _, revised = revise_node(
        signal=_make_signal("technical", "Sell"),
        challenge_turns=[],
        response_turns=[],
        majority_decision="Buy",
        llm=_ResponseLLM(too_high),
    )
    assert revised.confidence <= 1.0


def test_revise_node_revision_turn_contains_revised_decision():
    turn, _ = revise_node(
        signal=_make_signal("technical", "Sell"),
        challenge_turns=[],
        response_turns=[],
        majority_decision="Buy",
        llm=_ResponseLLM(_VALID_REVISION_JSON),
    )
    assert turn.revised_decision == "Hold"
