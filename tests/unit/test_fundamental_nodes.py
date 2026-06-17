"""Unit tests for the Fundamental Agent graph nodes (P3-E3)."""

import json

import pytest

from hifi.agents.fundamental_agent import (
    FundamentalistState,
    _build_signal,
    _extract_json,
    load_snapshot_node,
    parse_output_node,
)
from hifi.agents.schemas import AgentSignal

# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_plain_dict():
    text = '{"decision": "Buy", "confidence": 0.8}'
    result = _extract_json(text)
    assert result is not None
    assert result["decision"] == "Buy"


def test_extract_json_strips_markdown_fences():
    text = '```json\n{"decision": "Sell", "confidence": 0.4}\n```'
    result = _extract_json(text)
    assert result is not None
    assert result["decision"] == "Sell"


def test_extract_json_finds_embedded_json():
    text = 'Sure! Here is my answer:\n{"decision": "Hold", "confidence": 0.5}\nHope that helps.'
    result = _extract_json(text)
    assert result is not None
    assert result["decision"] == "Hold"


def test_extract_json_returns_none_on_invalid():
    result = _extract_json("This is not JSON at all.")
    assert result is None


def test_extract_json_returns_none_on_empty():
    result = _extract_json("")
    assert result is None


# ---------------------------------------------------------------------------
# _build_signal
# ---------------------------------------------------------------------------


def _valid_parsed() -> dict:
    return {
        "decision": "Buy",
        "confidence": 0.72,
        "rationale": "P/E of 28.3 is below the 5-year average.",
        "key_concern": "Macro headwinds from high interest rates.",
    }


def test_build_signal_valid():
    sig = _build_signal(
        _valid_parsed(), "AAPL", "2023-03-31",
        "qwen2.5-coder-32b-instruct-mlx", ["abc123"], ["pe"],
    )
    assert sig is not None
    assert sig.decision == "Buy"
    assert sig.confidence == pytest.approx(0.72)


def test_build_signal_returns_none_on_bad_confidence():
    parsed = _valid_parsed()
    parsed["confidence"] = 2.5  # out of range
    sig = _build_signal(parsed, "AAPL", "2023-03-31", "model", [], [])
    assert sig is None


def test_build_signal_returns_none_on_invalid_decision():
    parsed = _valid_parsed()
    parsed["decision"] = "Maybe"
    sig = _build_signal(parsed, "AAPL", "2023-03-31", "model", [], [])
    assert sig is None


# ---------------------------------------------------------------------------
# load_snapshot_node
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> FundamentalistState:
    defaults: FundamentalistState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "snapshot_json": "{}",
        "data_dir": "data",
        "tool_results": {},
        "llm_response": "",
        "signal": None,
        "error": None,
        "start_time": 0.0,
    }
    defaults.update(overrides)
    return defaults


def test_load_snapshot_node_sets_error_on_invalid_json():
    state = _make_state(snapshot_json="this is not json")
    result = load_snapshot_node(state)
    assert "error" in result and result["error"] is not None


def test_load_snapshot_node_passes_on_empty_dict():
    # An empty dict is not a valid FundamentalsSnapshot (missing required fields)
    # but the node should gracefully set error
    state = _make_state(snapshot_json="{}")
    result = load_snapshot_node(state)
    # Either error set (missing fields) or empty (if pydantic allows partial)
    # The key requirement: no exception raised
    assert isinstance(result, dict)


def test_load_snapshot_node_passes_with_valid_snapshot(tmp_path):
    """A minimal valid FundamentalsSnapshot passes load_snapshot_node."""
    from datetime import datetime

    from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

    snap = FundamentalsSnapshot(
        ticker="AAPL",
        period_end="2023-03-31",
        source="test",
        fetched_at=datetime(2023, 4, 1),
        provenance=ProvenanceRecord(source="test", fetched_at=datetime(2023, 4, 1)),
    )
    state = _make_state(snapshot_json=snap.model_dump_json())
    result = load_snapshot_node(state)
    assert result.get("error") is None


# ---------------------------------------------------------------------------
# parse_output_node (unit test with pre-set llm_response in state)
# ---------------------------------------------------------------------------


def test_parse_output_node_extracts_signal_from_valid_json(monkeypatch):
    """parse_output_node should extract an AgentSignal when llm_response is valid JSON."""
    valid_response = json.dumps({
        "decision": "Buy",
        "confidence": 0.8,
        "rationale": "ROE of 0.24 and P/E of 28.3 are strong.",
        "key_concern": "High interest rates compress multiples.",
    })

    # Monkeypatch make_llm to return a stub that won't be called (first parse succeeds)
    class _StubLLM:
        model_name = "qwen2.5-coder-32b-instruct-mlx"

        def invoke(self, messages):
            raise AssertionError("LLM should not be called on successful first parse")

    import hifi.agents.fundamental_agent as fa
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _StubLLM())

    state = _make_state(
        llm_response=valid_response,
        tool_results={
            "financial_ratios": {"pe": 28.3, "roe": 0.24, "call_id": "abc"},
            "growth_metrics": {"net_margin": 0.25, "call_id": "def"},
            "valuation_context": {"pe_1y_percentile": 0.6, "call_id": "ghi"},
            "macro_snapshot": {"fed_funds_rate": 4.75, "call_id": "jkl"},
        },
    )
    result = parse_output_node(state)
    assert result.get("error") is None
    assert isinstance(result["signal"], AgentSignal)
    assert result["signal"].decision == "Buy"


def test_parse_output_node_sets_error_after_failed_retry(monkeypatch):
    """parse_output_node sets error when both parse attempts fail."""
    class _StubLLMAlwaysInvalid:
        model_name = "qwen2.5-coder-32b-instruct-mlx"

        def invoke(self, messages):
            class _Resp:
                content = "Sorry, I cannot provide that information."
            return _Resp()

    import hifi.agents.fundamental_agent as fa
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _StubLLMAlwaysInvalid())

    state = _make_state(
        llm_response="not valid json",
        tool_results={
            "financial_ratios": {}, "growth_metrics": {},
            "valuation_context": {}, "macro_snapshot": {},
        },
    )
    result = parse_output_node(state)
    assert result.get("error") is not None
    assert result.get("signal") is None
