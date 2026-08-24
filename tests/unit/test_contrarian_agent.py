"""Unit tests for the Contrarian Agent (P8-E5)."""

import json

import pytest

from hifi.agents.contrarian_agent import (
    _build_ensemble_context,
    _extract_json,
    run_contrarian_analysis,
)
from hifi.agents.schemas import ContrarianAnalysis

# --- helpers ---

def _agent_summaries():
    return [
        {"agent_type": "fundamental", "decision": "Buy", "confidence": 0.75,
         "rationale": "P/E fair.", "key_concern": "High rates."},
        {"agent_type": "technical", "decision": "Buy", "confidence": 0.70,
         "rationale": "RSI recovering.", "key_concern": "ATR elevated."},
    ]


# --- _extract_json ---

def test_extract_json_plain():
    result = _extract_json('{"alternative_thesis": "Bear case."}')
    assert result["alternative_thesis"] == "Bear case."


def test_extract_json_fenced():
    result = _extract_json('```json\n{"confidence": 0.4}\n```')
    assert result["confidence"] == pytest.approx(0.4)


# --- _build_ensemble_context ---

def test_build_ensemble_context_is_valid_json():
    ctx = _build_ensemble_context(
        ticker="AAPL",
        as_of_date="2023-03-31",
        agent_summaries=_agent_summaries(),
        collective_decision="Buy",
        collective_confidence=0.72,
    )
    parsed = json.loads(ctx)
    assert parsed["ticker"] == "AAPL"
    assert parsed["preliminary_collective_decision"] == "Buy"
    assert len(parsed["contributing_agents"]) == 2


def test_build_ensemble_context_none_decision():
    ctx = _build_ensemble_context("AAPL", "2023-03-31", [], None, 0.0)
    parsed = json.loads(ctx)
    assert parsed["preliminary_collective_decision"] is None


# --- run_contrarian_analysis ---

_STUB_RESPONSE = json.dumps({
    "alternative_thesis": "Services deceleration risk is underweighted.",
    "risk_scenario": "Fed hikes 3 more times; consumer discretionary spend falls 15%.",
    "counterargument": "High collective confidence ignores macro inversion signal.",
    "confidence": 0.40,
})


def test_run_contrarian_returns_contrarian_analysis():
    class _ResponseLLM:
        model_name = "mlx-qwen35b"
        def invoke(self, _):
            class _R:
                content = _STUB_RESPONSE
            return _R()

    ctx = _build_ensemble_context("AAPL", "2023-03-31", _agent_summaries(), "Buy", 0.72)
    result = run_contrarian_analysis("AAPL", "2023-03-31", ctx, _test_llm=_ResponseLLM())
    assert isinstance(result, ContrarianAnalysis)
    assert result.confidence == pytest.approx(0.40)
    assert "deceleration" in result.alternative_thesis.lower()
    assert result.prompt_version == "contrarian_v1"


def test_run_contrarian_has_no_signal_field():
    class _ResponseLLM:
        model_name = "m"
        def invoke(self, _):
            class _R:
                content = _STUB_RESPONSE
            return _R()

    ctx = _build_ensemble_context("AAPL", "2023-03-31", _agent_summaries(), "Buy", 0.72)
    result = run_contrarian_analysis("AAPL", "2023-03-31", ctx, _test_llm=_ResponseLLM())
    assert not hasattr(result, "signal")
    assert not hasattr(result, "decision")


def test_run_contrarian_fallback_on_parse_failure():
    """When LLM produces bad JSON, fallback ContrarianAnalysis is returned."""
    class _AlwaysInvalidLLM:
        model_name = "m"
        def invoke(self, _):
            class _R:
                content = "I cannot provide that information."
            return _R()

    ctx = _build_ensemble_context("AAPL", "2023-03-31", [], "Hold", 0.5)
    result = run_contrarian_analysis("AAPL", "2023-03-31", ctx, _test_llm=_AlwaysInvalidLLM())
    assert isinstance(result, ContrarianAnalysis)
    assert result.confidence == 0.0


def test_run_contrarian_json_safe():
    class _ResponseLLM:
        model_name = "m"
        def invoke(self, _):
            class _R:
                content = _STUB_RESPONSE
            return _R()

    ctx = _build_ensemble_context("AAPL", "2023-03-31", _agent_summaries(), "Buy", 0.72)
    result = run_contrarian_analysis("AAPL", "2023-03-31", ctx, _test_llm=_ResponseLLM())
    json.dumps(result.model_dump())
