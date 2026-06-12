"""Unit tests for the Risk Agent graph nodes (P8-E2)."""

import json
import os
import shutil

import pytest

from hifi.agents.risk_agent import (
    RiskAnalystState,
    _build_risk_signal,
    _extract_json,
    call_mcp_tools_node,
    parse_output_node,
)
from hifi.agents.schemas import AgentSignal


def _valid_parsed() -> dict:
    return {
        "decision": "Hold",
        "confidence": 0.60,
        "rationale": "hist_vol_20d of 0.22 is moderate. Sharpe of 0.82 is acceptable.",
        "key_concern": "max_drawdown_252d of -0.28 is a tail risk.",
        "risk_assessment": "Moderate volatility regime with acceptable Sharpe.",
        "recommended_position_size": 0.05,
    }


def _make_state(**overrides) -> RiskAnalystState:
    defaults: RiskAnalystState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "data_dir": "data",
        "tool_results": {},
        "llm_response": "",
        "signal": None,
        "risk_assessment": None,
        "recommended_position_size": None,
        "error": None,
        "start_time": 0.0,
    }
    defaults.update(overrides)
    return defaults


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


# --- _extract_json ---

def test_extract_json_plain():
    assert _extract_json('{"decision": "Hold"}')["decision"] == "Hold"


def test_extract_json_fenced():
    assert _extract_json('```json\n{"decision":"Sell"}\n```')["decision"] == "Sell"


def test_extract_json_invalid():
    assert _extract_json("not json") is None


# --- _build_risk_signal ---

def test_build_risk_signal_sets_agent_type():
    sig, assessment, pos = _build_risk_signal(
        _valid_parsed(), "AAPL", "2023-03-31", "gemma", ["abc"], []
    )
    assert sig is not None
    assert sig.agent_type == "risk"
    assert assessment == "Moderate volatility regime with acceptable Sharpe."
    assert pos == pytest.approx(0.05)


def test_build_risk_signal_returns_none_on_bad_confidence():
    parsed = _valid_parsed()
    parsed["confidence"] = 2.0
    sig, _, _ = _build_risk_signal(parsed, "AAPL", "2023-03-31", "m", [], [])
    assert sig is None


# --- call_mcp_tools_node ---

def test_call_mcp_tools_node_has_risk_metrics_key(fixtures_data_dir):
    state = _make_state(data_dir=fixtures_data_dir)
    update = call_mcp_tools_node(state)
    assert "risk_metrics" in update["tool_results"]
    assert "technical_indicators" not in update["tool_results"]
    assert "macro_snapshot" not in update["tool_results"]


# --- parse_output_node ---

_STUB_JSON = json.dumps(_valid_parsed() | {"decision": "Hold", "confidence": 0.60,
    "rationale": "hist_vol_20d of 0.22.", "key_concern": "drawdown -28%.",
    "risk_assessment": "Moderate.", "recommended_position_size": 0.05})


def test_parse_output_node_extracts_signal(monkeypatch):
    class _StubLLM:
        model_name = "gemma"
        def invoke(self, _):
            class _R:
                content = _STUB_JSON
            return _R()

    import hifi.agents.risk_agent as ra
    monkeypatch.setattr(ra, "make_llm", lambda *a, **kw: _StubLLM())

    state = _make_state(
        llm_response=_STUB_JSON,
        tool_results={"risk_metrics": {"hist_vol_20d": 0.22, "call_id": "abc"}},
    )
    result = parse_output_node(state)
    assert result.get("error") is None
    assert isinstance(result["signal"], AgentSignal)
    assert result["signal"].decision == "Hold"
    assert result["risk_assessment"] == "Moderate."
    assert result["recommended_position_size"] == pytest.approx(0.05)


def test_parse_output_node_sets_error_after_retry(monkeypatch):
    class _Bad:
        model_name = "m"
        def invoke(self, _):
            class _R:
                content = "not json"
            return _R()

    import hifi.agents.risk_agent as ra
    monkeypatch.setattr(ra, "make_llm", lambda *a, **kw: _Bad())

    state = _make_state(llm_response="bad", tool_results={"risk_metrics": {}})
    result = parse_output_node(state)
    assert result.get("error") is not None
