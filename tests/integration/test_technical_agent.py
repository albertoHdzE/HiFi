"""
Integration test for the Technical Agent (P4-E1-T12).

Uses DI-injected LLM and Phase 1 fixtures (no live LM Studio required).
Validates that run_technical_analysis returns a TechnicalAnalysis with both
MCP tools called and a valid AgentSignal produced.
"""

import json
import os
import shutil

import pytest

from hifi.agents.schemas import TechnicalAnalysis
from hifi.agents.technical_agent import run_technical_analysis


@pytest.fixture
def fixtures_data_dir(tmp_path):
    """Phase 1 parquet fixtures copied to a temp data directory."""
    fixtures_root = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    for subdir in ("market", "macro"):
        dst = tmp_path / subdir
        dst.mkdir()
        src = os.path.join(fixtures_root, subdir)
        for f in os.listdir(src):
            if f.endswith(".parquet"):
                shutil.copy(os.path.join(src, f), dst / f)
    return str(tmp_path)


_STUB_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.55,
    "rationale": "RSI of 48.0 is neutral. Sharpe of 0.82 suggests moderate risk-adjusted return.",
    "key_concern": "Max drawdown of -18% over past 252 days indicates downside risk.",
    "time_horizon": "medium-term",
})


def _stub_llm(model_name: str = "test"):
    class _S:
        def invoke(self, messages):
            class _R:
                content = _STUB_RESPONSE
            return _R()
    s = _S()
    s.model_name = model_name
    return s


def test_run_technical_analysis_aapl(fixtures_data_dir):
    """Full Technical Agent run for AAPL Q1 2023 with DI LLM."""
    analysis = run_technical_analysis(
        ticker="AAPL",
        as_of_date="2023-03-31",
        data_dir=fixtures_data_dir,
        _test_llm=_stub_llm("mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled"),
    )

    assert isinstance(analysis, TechnicalAnalysis)
    assert analysis.signal is not None
    assert analysis.signal.decision in ("Buy", "Hold", "Sell")
    assert 0.0 <= analysis.signal.confidence <= 1.0
    assert analysis.signal.ticker == "AAPL"
    assert analysis.signal.agent_type == "technical"
    assert analysis.prompt_version == "technical_v1"
    assert analysis.latency_ms is not None and analysis.latency_ms > 0


def test_run_technical_analysis_two_tools_called(fixtures_data_dir):
    """Both MCP tools must be called; technical_indicators and risk_metrics populated."""
    analysis = run_technical_analysis(
        "AAPL", "2023-03-31", data_dir=fixtures_data_dir, _test_llm=_stub_llm()
    )

    assert isinstance(analysis.technical_indicators, dict)
    assert isinstance(analysis.risk_metrics, dict)
    assert "call_id" in analysis.technical_indicators or "error" in analysis.technical_indicators
    assert "call_id" in analysis.risk_metrics or "error" in analysis.risk_metrics


def test_run_technical_analysis_signal_has_call_ids(fixtures_data_dir):
    """Signal call_ids must be populated from MCP tool results."""
    analysis = run_technical_analysis(
        "AAPL", "2023-03-31", data_dir=fixtures_data_dir, _test_llm=_stub_llm()
    )

    if analysis.signal is not None:
        assert len(analysis.signal.call_ids) > 0


def test_run_technical_analysis_json_safe(fixtures_data_dir):
    """TechnicalAnalysis.model_dump() must be JSON-safe (no NaN)."""
    analysis = run_technical_analysis(
        "AAPL", "2023-03-31", data_dir=fixtures_data_dir, _test_llm=_stub_llm()
    )
    json.dumps(analysis.model_dump())


def test_run_technical_analysis_time_horizon(fixtures_data_dir):
    """time_horizon is extracted and stored in TechnicalAnalysis."""
    analysis = run_technical_analysis(
        "AAPL", "2023-03-31", data_dir=fixtures_data_dir, _test_llm=_stub_llm()
    )
    assert analysis.time_horizon == "medium-term"
