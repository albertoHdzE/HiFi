"""Unit tests for Technical Agent graph nodes (P4-E1-T9, T10, T11)."""

import json
import os
import shutil

import pytest

from hifi.agents.schemas import AgentSignal
from hifi.agents.technical_agent import (
    TechnicalAnalystState,
    _build_technical_signal,
    _extract_json,
    call_mcp_tools_node,
    parse_output_node,
)

# ---------------------------------------------------------------------------
# _extract_json (reused helper, same as fundamental agent)
# ---------------------------------------------------------------------------


def test_extract_json_plain_dict():
    text = '{"decision": "Buy", "confidence": 0.8}'
    assert _extract_json(text)["decision"] == "Buy"


def test_extract_json_strips_fences():
    text = '```json\n{"decision": "Sell", "confidence": 0.4}\n```'
    assert _extract_json(text)["decision"] == "Sell"


def test_extract_json_returns_none_on_invalid():
    assert _extract_json("not json at all") is None


# ---------------------------------------------------------------------------
# _build_technical_signal
# ---------------------------------------------------------------------------


def _valid_parsed() -> dict:
    return {
        "decision": "Buy",
        "confidence": 0.72,
        "rationale": "RSI of 42.1 and MACD crossover signal upward momentum.",
        "key_concern": "ATR of 3.82 indicates elevated daily range.",
        "time_horizon": "medium-term",
    }


def test_build_technical_signal_sets_agent_type_technical():
    sig = _build_technical_signal(
        _valid_parsed(), "AAPL", "2023-03-31",
        "test-model", ["abc"], [],
    )
    assert sig is not None
    assert sig.agent_type == "technical"


def test_build_technical_signal_valid_fields():
    sig = _build_technical_signal(
        _valid_parsed(), "AAPL", "2023-03-31",
        "test-model", ["abc", "def"], [],
    )
    assert sig is not None
    assert sig.decision == "Buy"
    assert sig.confidence == pytest.approx(0.72)
    assert sig.call_ids == ["abc", "def"]


def test_build_technical_signal_returns_none_on_bad_confidence():
    parsed = _valid_parsed()
    parsed["confidence"] = 1.5
    assert _build_technical_signal(parsed, "AAPL", "2023-03-31", "m", [], []) is None


def test_build_technical_signal_returns_none_on_invalid_decision():
    parsed = _valid_parsed()
    parsed["decision"] = "Neutral"
    assert _build_technical_signal(parsed, "AAPL", "2023-03-31", "m", [], []) is None


# ---------------------------------------------------------------------------
# call_mcp_tools_node (P4-E1-T9)
# ---------------------------------------------------------------------------


@pytest.fixture
def fixtures_data_dir(tmp_path):
    fixtures_root = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    for subdir in ("market", "macro"):
        dst = tmp_path / subdir
        dst.mkdir()
        src = os.path.join(fixtures_root, subdir)
        for f in os.listdir(src):
            if f.endswith(".parquet"):
                shutil.copy(os.path.join(src, f), dst / f)
    return str(tmp_path)


def _make_state(**overrides) -> TechnicalAnalystState:
    defaults: TechnicalAnalystState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "data_dir": "data",
        "tool_results": {},
        "llm_response": "",
        "signal": None,
        "time_horizon": None,
        "error": None,
        "start_time": 0.0,
    }
    defaults.update(overrides)
    return defaults


def test_call_mcp_tools_node_returns_two_keys(fixtures_data_dir):
    """call_mcp_tools_node returns technical_indicators and risk_metrics keys."""
    state = _make_state(data_dir=fixtures_data_dir)
    update = call_mcp_tools_node(state)
    keys = set(update["tool_results"].keys())
    assert keys == {"technical_indicators", "risk_metrics"}


def test_call_mcp_tools_node_results_have_call_ids(fixtures_data_dir):
    """Each tool result in call_mcp_tools_node has a call_id (when not error)."""
    state = _make_state(data_dir=fixtures_data_dir)
    update = call_mcp_tools_node(state)
    for key, result in update["tool_results"].items():
        if "error" not in result:
            assert "call_id" in result, f"{key} missing call_id"


def test_call_mcp_tools_node_no_fundamental_keys(fixtures_data_dir):
    """call_mcp_tools_node must not call fundamental or macro tools."""
    state = _make_state(data_dir=fixtures_data_dir)
    update = call_mcp_tools_node(state)
    # No fundamental keys should exist
    for forbidden in ("financial_ratios", "growth_metrics", "valuation_context", "macro_snapshot"):
        assert forbidden not in update["tool_results"]


# ---------------------------------------------------------------------------
# parse_output_node (P4-E1-T10, T11)
# ---------------------------------------------------------------------------

_STUB_SIGNAL_JSON = json.dumps({
    "decision": "Hold",
    "confidence": 0.55,
    "rationale": "RSI of 48.0 is in neutral territory. Sharpe of 0.82 is moderate.",
    "key_concern": "Max drawdown of -18% in past year indicates downside risk.",
    "time_horizon": "short-term",
})


def test_parse_output_node_extracts_signal_and_time_horizon(monkeypatch):
    """parse_output_node extracts AgentSignal and time_horizon from valid JSON."""
    class _StubLLM:
        model_name = "test-model"
        def invoke(self, messages):
            raise AssertionError("LLM should not be called when first parse succeeds")

    import hifi.agents.technical_agent as ta
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _StubLLM())

    state = _make_state(
        llm_response=_STUB_SIGNAL_JSON,
        tool_results={
            "technical_indicators": {"rsi": 48.0, "call_id": "abc"},
            "risk_metrics": {"sharpe_252d": 0.82, "max_drawdown_252d": -0.18, "call_id": "def"},
        },
    )
    result = parse_output_node(state)
    assert result.get("error") is None
    assert isinstance(result["signal"], AgentSignal)
    assert result["signal"].decision == "Hold"
    assert result["time_horizon"] == "short-term"


def test_parse_output_node_time_horizon_optional(monkeypatch):
    """time_horizon is optional; if absent from JSON, no error and value is None."""
    class _StubLLM:
        model_name = "test-model"
        def invoke(self, messages):
            raise AssertionError("LLM should not be called")

    import hifi.agents.technical_agent as ta
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _StubLLM())

    response_no_horizon = json.dumps({
        "decision": "Sell",
        "confidence": 0.70,
        "rationale": "RSI of 72 is overbought.",
        "key_concern": "ATR elevated.",
    })
    state = _make_state(
        llm_response=response_no_horizon,
        tool_results={
            "technical_indicators": {"rsi": 72.0, "call_id": "abc"},
            "risk_metrics": {"call_id": "def"},
        },
    )
    result = parse_output_node(state)
    assert result.get("error") is None
    assert result["signal"] is not None
    assert result.get("time_horizon") is None


def test_parse_output_node_sets_error_after_failed_retry(monkeypatch):
    """parse_output_node sets error when both attempts fail."""
    class _StubLLMAlwaysInvalid:
        model_name = "test-model"
        def invoke(self, messages):
            class _Resp:
                content = "I cannot provide that."
            return _Resp()

    import hifi.agents.technical_agent as ta
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _StubLLMAlwaysInvalid())

    state = _make_state(
        llm_response="not valid json",
        tool_results={
            "technical_indicators": {},
            "risk_metrics": {},
        },
    )
    result = parse_output_node(state)
    assert result.get("error") is not None
    assert result.get("signal") is None
