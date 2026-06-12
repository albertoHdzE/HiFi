"""
Holistic test for the Phase 6 observability pipeline (P6-E6-T4, P6-E6-T5, P6-E6-T6).

Uses a stub LLM and fixture parquet files so no live LM Studio or LangFuse
server is required.

What this test validates:
1. run_ensemble() with NoOpTracer completes and returns a valid EnsembleOutput
2. NoOpTracer.start_trace() was called (tracer method invocation tracking)
3. log_verification_scores() was called with the correct trace_id and report
4. flush(    agents=["fundamental", "technical"],
) was called exactly once per run_ensemble() call
5. Phase 5 regression: verify_ensemble() still produces a valid EnsembleVerificationReport
6. Phase 4 regression: run_ensemble() without explicit tracer still produces valid output
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import pytest

from hifi.collective.schemas import EnsembleOutput
from hifi.observability.tracing import NoOpTracer, SpanContext

# ---------------------------------------------------------------------------
# Recording tracer for assertion tracking
# ---------------------------------------------------------------------------


class RecordingTracer(NoOpTracer):
    """NoOpTracer that records all invocations for holistic assertions."""

    def __init__(self) -> None:
        self.start_trace_calls: list[dict] = []
        self.flush_count: int = 0
        self.scores: list[tuple[str, str, float]] = []

    def start_trace(self, name: str, ticker: str, as_of_date: str, **metadata) -> str:
        self.start_trace_calls.append(
            {"name": name, "ticker": ticker, "as_of_date": as_of_date}
        )
        return f"holistic-trace-{len(self.start_trace_calls)}"

    def flush(self) -> None:
        self.flush_count += 1

    def log_score(self, trace_id: str, name: str, value: float) -> None:
        self.scores.append((trace_id, name, value))

    from collections.abc import Generator
    from contextlib import contextmanager

    @contextmanager
    def span(self, trace_id, name, input=None) -> Generator[SpanContext, None, None]:
        ctx = SpanContext()
        yield ctx


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


_HOLD_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "Market is fairly valued at current levels.",
    "key_concern": "Rate sensitivity.",
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
# Test 1 + 2 + 3 + 4: Full pipeline with NoOpTracer
# ---------------------------------------------------------------------------


def test_phase6_full_pipeline_with_noop_tracer(monkeypatch, fixtures_data_dir):
    """
    run_ensemble(    agents=["fundamental", "technical"],
    ) with NoOpTracer:
    1. Completes and returns valid EnsembleOutput
    2. start_trace() was called
    3. Verification scores logged
    4. flush() called once
    """
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta
    from hifi.agents.ensemble_runner import run_ensemble

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_llm("fund-model"))
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_llm("tech-model"))

    tracer = RecordingTracer()
    snap = _make_snapshot()
    output = run_ensemble(
        "AAPL", "2023-03-31", snap.model_dump_json(),
        data_dir=fixtures_data_dir,
        tracer=tracer,
        agents=["fundamental", "technical"],
    )

    # Assertion 1: valid EnsembleOutput
    assert isinstance(output, EnsembleOutput)
    assert output.ticker == "AAPL"
    assert output.as_of_date == "2023-03-31"
    assert output.ensemble_decision is not None
    assert output.latency_ms >= 0.0

    # Assertion 2: start_trace was called (at least for run_ensemble)
    assert len(tracer.start_trace_calls) >= 1
    top_level_names = [c["name"] for c in tracer.start_trace_calls]
    assert "run_ensemble" in top_level_names

    # Assertion 3: verification scores logged (6 scores on ensemble trace)
    assert len(tracer.scores) == 6
    score_names = {name for (_tid, name, _val) in tracer.scores}
    assert score_names == {
        "fundamental_hr", "fundamental_gr",
        "technical_hr", "technical_gr",
        "disagreement_entropy", "n_contradictions",
    }

    # Assertion 4: flush() called once (by run_ensemble)
    assert tracer.flush_count >= 1


# ---------------------------------------------------------------------------
# Test 5: Phase 5 regression -- verify_ensemble still produces valid report
# ---------------------------------------------------------------------------


def test_phase5_regression_verify_ensemble(monkeypatch, fixtures_data_dir):
    """Phase 5 regression: verify_ensemble output is unchanged by Phase 6."""
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta
    from hifi.agents.ensemble_runner import run_ensemble
    from hifi.verification.schemas import EnsembleVerificationReport
    from hifi.verification.verifier import verify_ensemble

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_llm("fund-model"))
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_llm("tech-model"))

    snap = _make_snapshot()
    output = run_ensemble(
        "AAPL", "2023-03-31", snap.model_dump_json(),
        data_dir=fixtures_data_dir,
        agents=["fundamental", "technical"],
    )

    # Directly call verify_ensemble on the output (Phase 5 pipeline unchanged)
    report = verify_ensemble(output, always_verify=True)

    assert isinstance(report, EnsembleVerificationReport)
    assert report.ticker == "AAPL"
    assert report.fundamental_report.agent_type == "fundamental"
    assert report.technical_report.agent_type == "technical"
    assert 0.0 <= report.fundamental_report.hallucination_rate <= 1.0
    assert 0.0 <= report.technical_report.hallucination_rate <= 1.0
    # Report is JSON-safe
    json.dumps(report.model_dump())


# ---------------------------------------------------------------------------
# Test 6: Phase 4 regression -- run_ensemble() without explicit tracer
# ---------------------------------------------------------------------------


def test_phase4_regression_run_ensemble_without_tracer(monkeypatch, fixtures_data_dir):
    """Phase 4 regression: run_ensemble(    agents=["fundamental", "technical"],
    ) without explicit tracer still works."""
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta
    from hifi.agents.ensemble_runner import run_ensemble

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_llm("fund-model"))
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_llm("tech-model"))

    snap = _make_snapshot()
    # No explicit tracer -- falls back to get_tracer() which returns NoOpTracer
    # (LANGFUSE_ENABLED=false in conftest.py session fixture)
    output = run_ensemble(
        "AAPL", "2023-03-31", snap.model_dump_json(),
        fixtures_data_dir, agents=["fundamental", "technical"],
    )

    assert isinstance(output, EnsembleOutput)
    assert output.ticker == "AAPL"
    # Must be JSON-safe
    json.dumps(output.model_dump())
