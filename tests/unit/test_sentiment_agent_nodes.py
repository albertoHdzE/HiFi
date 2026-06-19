"""Unit tests for the Sentiment Agent (P8-E4)."""

import json

import pytest

from hifi.agents.schemas import AgentSignal, SentimentAnalysis
from hifi.agents.sentiment_agent import (
    _default_insufficient_signal,
    _extract_json,
    run_sentiment_analysis,
)

# --- _extract_json ---

def test_extract_json_plain():
    assert _extract_json('{"decision":"Buy"}')["decision"] == "Buy"


def test_extract_json_fenced():
    assert _extract_json('```json\n{"decision":"Hold"}\n```')["decision"] == "Hold"


def test_extract_json_invalid():
    assert _extract_json("not json") is None


# --- default insufficient signal ---

def test_default_insufficient_signal():
    sig = _default_insufficient_signal("AAPL", "2023-03-31")
    assert isinstance(sig, AgentSignal)
    assert sig.decision == "Hold"
    assert sig.confidence == 0.0
    assert sig.agent_type == "sentiment"
    assert "data_gaps" in sig.model_fields_set or sig.data_gaps == ["retrieved_context"]


# --- run_sentiment_analysis: fail-open (no context) ---

def test_run_sentiment_returns_default_when_no_context(monkeypatch, tmp_path):
    """When retrieve_context returns empty string, default Insufficient Data signal returned."""
    import hifi.agents.sentiment_agent as sa

    monkeypatch.setattr(sa, "call_tool", lambda *a, **kw: {"passages": [], "call_id": "x"})

    result = run_sentiment_analysis("AAPL", "2023-03-31", data_dir=str(tmp_path))
    assert isinstance(result, SentimentAnalysis)
    assert result.signal is not None
    assert result.signal.decision == "Hold"
    assert result.signal.confidence == 0.0
    assert "Insufficient Data" in result.sentiment_summary


# --- run_sentiment_analysis: with context calls LLM ---

_STUB_RESPONSE = json.dumps({
    "decision": "Buy",
    "confidence": 0.65,
    "rationale": "MD&A tone highlights strong services growth.",
    "key_concern": "FX headwinds mentioned in risk factors.",
    "sentiment_summary": "Positive management tone with selective risk disclosures.",
    "notable_signals": ["services revenue grew 5% YoY", "FX headwinds cited"],
})


def test_run_sentiment_calls_llm_when_context_available(monkeypatch, tmp_path):
    """When retrieve_context returns passages, LLM is called and result parsed."""
    import hifi.agents.sentiment_agent as sa

    passages = [{"rank": 1, "filing_type": "10-K", "section": "MD&A",
                 "period": "2022-09-30", "text": "Apple services revenue grew."}]
    monkeypatch.setattr(sa, "call_tool", lambda *a, **kw: {"passages": passages, "call_id": "x"})

    class _ResponseLLM:
        model_name = "qwen-32b"
        def invoke(self, _):
            class _R:
                content = _STUB_RESPONSE
            return _R()

    result = run_sentiment_analysis(
        "AAPL", "2023-03-31", data_dir=str(tmp_path), _test_llm=_ResponseLLM()
    )
    assert isinstance(result, SentimentAnalysis)
    assert result.signal is not None
    assert result.signal.decision == "Buy"
    assert result.signal.confidence == pytest.approx(0.65)
    assert "Positive" in result.sentiment_summary
    assert len(result.notable_signals) == 2


def test_run_sentiment_json_safe(monkeypatch, tmp_path):
    import hifi.agents.sentiment_agent as sa

    monkeypatch.setattr(sa, "call_tool", lambda *a, **kw: {"passages": [], "call_id": "x"})

    result = run_sentiment_analysis("AAPL", "2023-03-31", data_dir=str(tmp_path))
    json.dumps(result.model_dump())
