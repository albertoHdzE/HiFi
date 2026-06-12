"""Unit tests for the Macro Agent graph nodes (P8-E3)."""

import json

from hifi.agents.macro_agent import (
    MacroAnalystState,
    _build_macro_signal,
    _extract_json,
    call_mcp_tools_node,
    parse_output_node,
)
from hifi.agents.schemas import AgentSignal


def _valid_parsed() -> dict:
    return {
        "decision": "Hold",
        "confidence": 0.55,
        "rationale": "fed_funds_rate of 4.75 signals tight monetary conditions.",
        "key_concern": "Yield curve inverted; recession risk elevated.",
        "regime_assessment": "Late-cycle tightening",
        "macro_rationale": "High rates and inverted curve signal late-cycle dynamics.",
    }


def _make_state(**overrides) -> MacroAnalystState:
    defaults: MacroAnalystState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "data_dir": "data",
        "tool_results": {},
        "llm_response": "",
        "signal": None,
        "regime_assessment": None,
        "macro_rationale": None,
        "error": None,
        "start_time": 0.0,
    }
    defaults.update(overrides)
    return defaults


# --- _extract_json ---

def test_extract_json_plain():
    assert _extract_json('{"decision": "Hold"}')["decision"] == "Hold"


def test_extract_json_fenced():
    assert _extract_json('```json\n{"decision":"Sell"}\n```')["decision"] == "Sell"


def test_extract_json_invalid():
    assert _extract_json("garbage") is None


# --- _build_macro_signal ---

def test_build_macro_signal_sets_agent_type():
    sig, regime, rat = _build_macro_signal(
        _valid_parsed(), "AAPL", "2023-03-31", "qwen27b", ["abc"], []
    )
    assert sig is not None
    assert sig.agent_type == "macro"
    assert regime == "Late-cycle tightening"
    assert "late-cycle" in rat.lower()


def test_build_macro_signal_returns_none_on_bad_decision():
    parsed = _valid_parsed()
    parsed["decision"] = "Maybe"
    sig, _, _ = _build_macro_signal(parsed, "AAPL", "2023-03-31", "m", [], [])
    assert sig is None


# --- call_mcp_tools_node ---

def test_call_mcp_tools_node_has_macro_snapshot_key(tmp_path):
    """call_mcp_tools_node returns macro_snapshot key (may be error dict, not crash)."""
    state = _make_state(data_dir=str(tmp_path))
    update = call_mcp_tools_node(state)
    assert "macro_snapshot" in update["tool_results"]
    assert "risk_metrics" not in update["tool_results"]
    assert "technical_indicators" not in update["tool_results"]


# --- parse_output_node ---

_STUB_JSON = json.dumps({
    "decision": "Hold", "confidence": 0.55,
    "rationale": "fed_funds_rate of 4.75.",
    "key_concern": "Inverted yield curve.",
    "regime_assessment": "Late-cycle tightening",
    "macro_rationale": "Rates signal late cycle.",
})


def test_parse_output_node_extracts_signal(monkeypatch):
    class _StubLLM:
        model_name = "qwen27b"
        def invoke(self, _):
            class _R:
                content = _STUB_JSON
            return _R()

    import hifi.agents.macro_agent as ma
    monkeypatch.setattr(ma, "make_llm", lambda *a, **kw: _StubLLM())

    state = _make_state(
        llm_response=_STUB_JSON,
        tool_results={"macro_snapshot": {"fed_funds_rate": 4.75, "call_id": "abc"}},
    )
    result = parse_output_node(state)
    assert result.get("error") is None
    assert isinstance(result["signal"], AgentSignal)
    assert result["signal"].agent_type == "macro"
    assert result["regime_assessment"] == "Late-cycle tightening"


def test_parse_output_node_sets_error_after_retry(monkeypatch):
    class _Bad:
        model_name = "m"
        def invoke(self, _):
            class _R:
                content = "not json"
            return _R()

    import hifi.agents.macro_agent as ma
    monkeypatch.setattr(ma, "make_llm", lambda *a, **kw: _Bad())

    state = _make_state(llm_response="bad", tool_results={"macro_snapshot": {}})
    result = parse_output_node(state)
    assert result.get("error") is not None
