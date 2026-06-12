"""
Integration tests for agent observability instrumentation (P6-E3, P6-E4, P6-E5, P6-E6).

Tests cover:
E3-T3: run_analysis() with NoOpTracer produces valid FundamentalAnalysis (regression)
E3-T4: run_technical_analysis() with NoOpTracer produces valid TechnicalAnalysis (regression)
E3-T5: run_analysis() with stub tracer records start_trace() and flush() called once
E3-T6: run_technical_analysis() with stub tracer records start_trace() and flush() called once
E4-T2: call_tool() with no active trace context -- no span created, result unchanged
E4-T3: call_tool() with active trace_context -- tracer.span() called with correct name and input
E4-T4: call_tool() span captures tool result as output on exit
E4-T5: call_tool() span captures call_id in metadata when present in result
E6-T2: run_ensemble() with NoOpTracer passes tracer to run_analysis() and run_technical_analysis()
E6-T3: run_ensemble() calls log_verification_scores() with the ensemble trace_id
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import pytest

from hifi.agents.mcp_client import call_tool
from hifi.observability.tracing import (
    NoOpTracer,
    SpanContext,
    _current_trace_id,
    trace_context,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingTracer(NoOpTracer):
    """NoOpTracer that records method calls for test assertions."""

    def __init__(self) -> None:
        self.start_trace_calls: list[dict] = []
        self.flush_count: int = 0
        self.spans: list[dict] = []
        self.scores: list[tuple[str, str, float]] = []

    def start_trace(self, name: str, ticker: str, as_of_date: str, **metadata) -> str:
        self.start_trace_calls.append(
            {"name": name, "ticker": ticker, "as_of_date": as_of_date}
        )
        return f"rec-trace-{len(self.start_trace_calls)}"

    def flush(self) -> None:
        self.flush_count += 1

    def log_score(self, trace_id: str, name: str, value: float) -> None:
        self.scores.append((trace_id, name, value))

    from collections.abc import Generator
    from contextlib import contextmanager

    @contextmanager
    def span(self, trace_id, name, input=None) -> Generator[SpanContext, None, None]:
        ctx = SpanContext()
        self.spans.append({"trace_id": trace_id, "name": name, "input": input, "ctx": ctx})
        yield ctx


def _make_snapshot():
    from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

    return FundamentalsSnapshot(
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


_HOLD_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "Market appears fairly balanced at current levels.",
    "key_concern": "Rate sensitivity remains elevated.",
    "time_horizon": "medium-term",
})


def _stub_llm(name="stub-model"):
    class _Stub:
        def invoke(self, _):
            class _R:
                content = _HOLD_RESPONSE
            return _R()
        model_name = name
    return _Stub()


@pytest.fixture
def fixtures_data_dir(tmp_path):
    fixtures_root = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    for subdir in ("market", "macro"):
        dst = tmp_path / subdir
        dst.mkdir()
        src = os.path.join(fixtures_root, subdir)
        if os.path.isdir(src):
            for f in os.listdir(src):
                if f.endswith(".parquet"):
                    shutil.copy(os.path.join(src, f), dst / f)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# E3-T3: run_analysis() with NoOpTracer produces valid FundamentalAnalysis
# ---------------------------------------------------------------------------


def test_run_analysis_with_noop_tracer(monkeypatch, fixtures_data_dir):
    import hifi.agents.fundamental_agent as fa
    from hifi.agents.fundamental_agent import run_analysis
    from hifi.agents.schemas import FundamentalAnalysis

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_llm("fund-model"))

    snap = _make_snapshot()
    result = run_analysis(
        "AAPL", "2023-03-31", snap.model_dump_json(),
        data_dir=fixtures_data_dir,
        tracer=NoOpTracer(),
    )

    assert isinstance(result, FundamentalAnalysis)
    assert result.prompt_version == "fundamental_v1"
    assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# E3-T4: run_technical_analysis() with NoOpTracer produces valid TechnicalAnalysis
# ---------------------------------------------------------------------------


def test_run_technical_analysis_with_noop_tracer(monkeypatch, fixtures_data_dir):
    import hifi.agents.technical_agent as ta
    from hifi.agents.schemas import TechnicalAnalysis
    from hifi.agents.technical_agent import run_technical_analysis

    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_llm("tech-model"))

    result = run_technical_analysis(
        "AAPL", "2023-03-31",
        data_dir=fixtures_data_dir,
        tracer=NoOpTracer(),
    )

    assert isinstance(result, TechnicalAnalysis)
    assert result.prompt_version == "technical_v1"
    assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# E3-T5: run_analysis() with RecordingTracer records start_trace() and flush()
# ---------------------------------------------------------------------------


def test_run_analysis_recording_tracer_start_and_flush(monkeypatch, fixtures_data_dir):
    import hifi.agents.fundamental_agent as fa
    from hifi.agents.fundamental_agent import run_analysis

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_llm("fund-model"))

    tracer = RecordingTracer()
    snap = _make_snapshot()
    run_analysis(
        "AAPL", "2023-03-31", snap.model_dump_json(),
        data_dir=fixtures_data_dir,
        tracer=tracer,
    )

    assert len(tracer.start_trace_calls) == 1
    assert tracer.start_trace_calls[0]["name"] == "fundamental_agent"
    assert tracer.start_trace_calls[0]["ticker"] == "AAPL"
    assert tracer.flush_count == 1


# ---------------------------------------------------------------------------
# E3-T6: run_technical_analysis() with RecordingTracer records start_trace() + flush()
# ---------------------------------------------------------------------------


def test_run_technical_analysis_recording_tracer_start_and_flush(monkeypatch, fixtures_data_dir):
    import hifi.agents.technical_agent as ta
    from hifi.agents.technical_agent import run_technical_analysis

    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_llm("tech-model"))

    tracer = RecordingTracer()
    run_technical_analysis(
        "AAPL", "2023-03-31",
        data_dir=fixtures_data_dir,
        tracer=tracer,
    )

    assert len(tracer.start_trace_calls) == 1
    assert tracer.start_trace_calls[0]["name"] == "technical_agent"
    assert tracer.start_trace_calls[0]["ticker"] == "AAPL"
    assert tracer.flush_count == 1


# ---------------------------------------------------------------------------
# E4-T2: call_tool() with no active trace -- no span created, result unchanged
# ---------------------------------------------------------------------------


def test_call_tool_no_trace_context_returns_unchanged(fixtures_data_dir):
    """When no trace context is active, call_tool behaves exactly as before."""
    assert _current_trace_id.get() is None  # precondition

    result = call_tool(
        tool_name="get_macro_snapshot",
        params={"date": "2022-06-15"},
        data_dir=fixtures_data_dir,
    )

    assert isinstance(result, dict)
    assert "call_id" in result


# ---------------------------------------------------------------------------
# E4-T3: call_tool() with active trace_context calls tracer.span() with correct name
# ---------------------------------------------------------------------------


def test_call_tool_with_trace_context_span_name(monkeypatch, fixtures_data_dir):
    """When a trace context is active, call_tool creates a span with the tool name."""
    import hifi.agents.mcp_client as mc

    tracer = RecordingTracer()
    monkeypatch.setattr(mc, "get_tracer", lambda: tracer)

    with trace_context("test-trace-for-mcp"):
        call_tool(
            tool_name="get_macro_snapshot",
            params={"date": "2022-06-15"},
            data_dir=fixtures_data_dir,
        )

    assert len(tracer.spans) == 1
    span_record = tracer.spans[0]
    assert span_record["name"] == "mcp_get_macro_snapshot"
    assert span_record["trace_id"] == "test-trace-for-mcp"
    assert span_record["input"] == {"date": "2022-06-15"}


# ---------------------------------------------------------------------------
# E4-T4: call_tool() span captures tool result as output on exit
# ---------------------------------------------------------------------------


def test_call_tool_span_captures_output(monkeypatch, fixtures_data_dir):
    import hifi.agents.mcp_client as mc

    tracer = RecordingTracer()
    monkeypatch.setattr(mc, "get_tracer", lambda: tracer)

    with trace_context("test-trace-output"):
        result = call_tool(
            tool_name="get_macro_snapshot",
            params={"date": "2022-06-15"},
            data_dir=fixtures_data_dir,
        )

    assert len(tracer.spans) == 1
    ctx = tracer.spans[0]["ctx"]
    assert ctx.output is not None
    assert ctx.output == result


# ---------------------------------------------------------------------------
# E4-T5: call_tool() span captures call_id in metadata when present
# ---------------------------------------------------------------------------


def test_call_tool_span_captures_call_id_in_metadata(monkeypatch, fixtures_data_dir):
    import hifi.agents.mcp_client as mc

    tracer = RecordingTracer()
    monkeypatch.setattr(mc, "get_tracer", lambda: tracer)

    with trace_context("test-trace-callid"):
        result = call_tool(
            tool_name="get_macro_snapshot",
            params={"date": "2022-06-15"},
            data_dir=fixtures_data_dir,
        )

    assert "call_id" in result, "Expected macro_snapshot result to have call_id"
    ctx = tracer.spans[0]["ctx"]
    assert ctx.metadata is not None
    assert ctx.metadata.get("call_id") == result["call_id"]


# ---------------------------------------------------------------------------
# E6-T2: run_ensemble() passes tracer to run_analysis() and run_technical_analysis()
# ---------------------------------------------------------------------------


def test_run_ensemble_passes_tracer_to_agents(monkeypatch, fixtures_data_dir):
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta
    from hifi.agents.ensemble_runner import run_ensemble

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_llm("fund-model"))
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_llm("tech-model"))

    tracer = RecordingTracer()
    snap = _make_snapshot()
    run_ensemble(
        "AAPL", "2023-03-31", snap.model_dump_json(), fixtures_data_dir,
        tracer=tracer, agents=["fundamental", "technical"],
    )

    # run_ensemble starts one trace, each agent also starts one trace via tracer
    # Total: 1 (ensemble) + 1 (fundamental) + 1 (technical) = 3 start_trace calls
    assert len(tracer.start_trace_calls) == 3
    names = [c["name"] for c in tracer.start_trace_calls]
    assert "run_ensemble" in names
    assert "fundamental_agent" in names
    assert "technical_agent" in names


# ---------------------------------------------------------------------------
# E6-T3: run_ensemble() calls log_verification_scores() with the ensemble trace_id
# ---------------------------------------------------------------------------


def test_run_ensemble_logs_verification_scores(monkeypatch, fixtures_data_dir):
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta
    from hifi.agents.ensemble_runner import run_ensemble

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_llm("fund-model"))
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_llm("tech-model"))

    tracer = RecordingTracer()
    snap = _make_snapshot()
    run_ensemble(
        "AAPL", "2023-03-31", snap.model_dump_json(), fixtures_data_dir,
        tracer=tracer, agents=["fundamental", "technical"],
    )

    # Six verification scores must be logged
    assert len(tracer.scores) == 6
    score_names = {name for (_tid, name, _val) in tracer.scores}
    assert score_names == {
        "fundamental_hr", "fundamental_gr",
        "technical_hr", "technical_gr",
        "disagreement_entropy", "n_contradictions",
    }

    # All scores attached to the ensemble trace_id (which was "rec-trace-1")
    ensemble_tid = "rec-trace-1"  # first start_trace returns this
    for tid, _name, _val in tracer.scores:
        assert tid == ensemble_tid
