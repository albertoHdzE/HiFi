"""
Integration tests for run_ensemble (P4-E4-T2, T3, T4).

Both LLMs are stubbed; Phase 1 fixtures provide real MCP data.
"""

import json
import os
import shutil
from datetime import datetime

import pytest

from hifi.agents.ensemble_runner import run_ensemble
from hifi.collective.schemas import EnsembleOutput
from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord


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


@pytest.fixture
def aapl_snapshot_json():
    snap = FundamentalsSnapshot(
        ticker="AAPL",
        period_end="2023-03-31",
        eps=6.11,
        market_cap=2_500_000_000_000,
        total_equity=62_146_000_000,
        revenue=394_330_000_000,
        net_income=99_803_000_000,
        total_assets=352_755_000_000,
        total_liabilities=290_437_000_000,
        source="test",
        fetched_at=datetime(2023, 4, 1),
        provenance=ProvenanceRecord(source="test", fetched_at=datetime(2023, 4, 1)),
    )
    return snap.model_dump_json()


_FUND_STUB = json.dumps({
    "decision": "Hold",
    "confidence": 0.65,
    "rationale": "P/E of 28.3 and ROE of 0.24 are solid fundamentals.",
    "key_concern": "High fed funds rate of 4.75 compresses multiples.",
})

_TECH_STUB = json.dumps({
    "decision": "Buy",
    "confidence": 0.70,
    "rationale": "RSI of 42.1 signals recovery; MACD histogram turned positive.",
    "key_concern": "ATR of 3.82 indicates elevated daily price swings.",
    "time_horizon": "short-term",
})


def _make_fund_stub():
    class _StubLLM:
        model_name = "qwen2.5-coder-32b-instruct-mlx"
        def invoke(self, messages):
            class _R:
                content = _FUND_STUB
            return _R()
    return _StubLLM()


def _make_tech_stub():
    class _StubLLM:
        model_name = "mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled"
        def invoke(self, messages):
            class _R:
                content = _TECH_STUB
            return _R()
    return _StubLLM()


def test_run_ensemble_returns_ensemble_output(
    monkeypatch, fixtures_data_dir, aapl_snapshot_json
):
    """run_ensemble returns EnsembleOutput with both analyses populated."""
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _make_fund_stub())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _make_tech_stub())

    output = run_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        agents=["fundamental", "technical"],
    )

    assert isinstance(output, EnsembleOutput)
    assert output.ticker == "AAPL"
    assert output.as_of_date == "2023-03-31"
    assert output.fundamental_analysis is not None
    assert output.technical_analysis is not None
    assert output.ensemble_decision is not None
    assert output.latency_ms > 0


def test_run_ensemble_independent_agents(
    monkeypatch, fixtures_data_dir, aapl_snapshot_json
):
    """Fundamental and technical analyses are produced independently."""
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _make_fund_stub())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _make_tech_stub())

    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
    )

    # Each analysis uses a different model_id (their stubs have different model_name)
    fa_signal = output.fundamental_analysis.signal
    ta_signal = output.technical_analysis.signal
    if fa_signal is not None and ta_signal is not None:
        assert fa_signal.model_id != ta_signal.model_id
        assert fa_signal.agent_type == "fundamental"
        assert ta_signal.agent_type == "technical"


def test_run_ensemble_json_safe(
    monkeypatch, fixtures_data_dir, aapl_snapshot_json
):
    """EnsembleOutput.model_dump() must be JSON-safe."""
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _make_fund_stub())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _make_tech_stub())

    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
    )
    # Must not raise
    json.dumps(output.model_dump())


def test_run_ensemble_phase9_fields_populated(
    monkeypatch, fixtures_data_dir, aapl_snapshot_json
):
    """Phase 9: signals, aggregation_method, and method_comparison are populated."""
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _make_fund_stub())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _make_tech_stub())

    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
    )

    assert output.aggregation_method == "confidence_weighted"
    assert isinstance(output.signals, list)
    assert all(s is not None for s in output.signals)
    assert set(output.method_comparison.keys()) == {
        "majority", "confidence_weighted", "performance_weighted", "contrarian_adjusted"
    }


def test_run_ensemble_method_comparison_cw_equals_ensemble_decision(
    monkeypatch, fixtures_data_dir, aapl_snapshot_json
):
    """method_comparison['confidence_weighted'] equals ensemble_decision."""
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _make_fund_stub())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _make_tech_stub())

    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
    )

    cw = output.method_comparison["confidence_weighted"]
    ed = output.ensemble_decision
    assert cw.collective_decision == ed.collective_decision
    assert cw.collective_confidence == ed.collective_confidence
    assert cw.n_valid_signals == ed.n_valid_signals


def test_run_ensemble_backward_compat_agents_list_still_gets_method_comparison(
    monkeypatch, fixtures_data_dir, aapl_snapshot_json
):
    """agents=['fundamental','technical'] still populates method_comparison with 4 keys."""
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _make_fund_stub())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _make_tech_stub())

    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
    )

    assert len(output.method_comparison) == 4
    assert output.signals  # non-empty


def test_run_ensemble_output_json_roundtrip_with_phase9_fields(
    monkeypatch, fixtures_data_dir, aapl_snapshot_json
):
    """EnsembleOutput with Phase 9 fields is JSON-safe and round-trips."""

    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _make_fund_stub())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _make_tech_stub())

    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
    )

    serialised = output.model_dump_json()
    restored = type(output).model_validate_json(serialised)
    assert restored.ticker == output.ticker
    assert set(restored.method_comparison.keys()) == set(output.method_comparison.keys())


def test_run_ensemble_disagreement_valid(
    monkeypatch, fixtures_data_dir, aapl_snapshot_json
):
    """When agents disagree, ensemble_decision has agreement=False."""
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta

    # fundamental -> Hold, technical -> Buy: they disagree
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _make_fund_stub())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _make_tech_stub())

    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
    )

    ed = output.ensemble_decision
    fa_sig = output.fundamental_analysis.signal
    ta_sig = output.technical_analysis.signal

    if fa_sig is not None and ta_sig is not None and fa_sig.decision != ta_sig.decision:
        assert ed.agreement is False
        assert ed.disagreement_entropy > 0.0
