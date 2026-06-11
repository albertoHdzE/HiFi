"""
Unit tests for hifi.observability.tracing (P6-E2, P6-E5).

Tests cover:
- get_tracer() returns NoOpTracer when LANGFUSE_ENABLED=false (E2-T6)
- NoOpTracer.start_trace() returns a non-empty string trace ID (E2-T7)
- NoOpTracer.get_callback_handler() returns None (E2-T8)
- NoOpTracer.span() context manager enters and exits without exception (E2-T9)
- NoOpTracer.log_score() with valid args does not raise (E2-T10)
- trace_context() sets _current_trace_id within block, resets after (E2-T11)
- nested trace_context() calls restore outer value correctly (E2-T12)
- log_verification_scores() calls tracer.log_score() exactly 6 times (E5-T2)
- score values match the six named metrics (E5-T3)
- log_verification_scores() with NoOpTracer does not raise (E5-T4)
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from hifi.observability.tracing import (
    NoOpTracer,
    SpanContext,
    _current_trace_id,
    get_tracer,
    log_verification_scores,
    trace_context,
)

# ---------------------------------------------------------------------------
# Helpers: recording tracer and synthetic verification data
# ---------------------------------------------------------------------------


class RecordingTracer(NoOpTracer):
    """NoOpTracer that records all method invocations for test assertions.

    This is a real implementation (not a mock) that extends NoOpTracer and
    adds a call log. Used to verify that instrumentation code calls the tracer
    interface with the expected arguments.
    """

    def __init__(self) -> None:
        self.scores: list[tuple[str, str, float]] = []

    def log_score(self, trace_id: str, name: str, value: float) -> None:
        self.scores.append((trace_id, name, value))


def _make_synthetic_verification_report():
    """Build minimal EnsembleVerificationReport and EnsembleDecision for testing."""
    from hifi.agents.schemas import AgentSignal, FundamentalAnalysis, TechnicalAnalysis
    from hifi.collective.schemas import EnsembleDecision, EnsembleOutput
    from hifi.verification.verifier import verify_ensemble

    fund_signal = AgentSignal(
        ticker="TEST",
        as_of_date="2023-03-31",
        decision="Buy",
        confidence=0.80,
        rationale="P/E of 28.3 looks attractive.",
        key_concern="Rate risk.",
        call_ids=["fundcall_001"],
        model_id="test-model",
        agent_type="fundamental",
    )
    fundamental = FundamentalAnalysis(
        signal=fund_signal,
        financial_ratios={"pe": 28.3, "call_id": "fundcall_001"},
        growth_metrics={"call_id": "fundcall_002"},
        valuation_context={"call_id": "fundcall_003"},
        macro_snapshot={"call_id": "fundcall_004"},
        prompt_version="fundamental_v1",
        latency_ms=1000.0,
    )

    tech_signal = AgentSignal(
        ticker="TEST",
        as_of_date="2023-03-31",
        decision="Hold",
        confidence=0.60,
        rationale="RSI of 52.0 is neutral.",
        key_concern="Volatility elevated.",
        call_ids=["techcall_001"],
        model_id="test-model",
        agent_type="technical",
    )
    technical = TechnicalAnalysis(
        signal=tech_signal,
        technical_indicators={"rsi": 52.0, "call_id": "techcall_001"},
        risk_metrics={"sharpe_252d": 0.95, "call_id": "techcall_002"},
        time_horizon="medium-term",
        prompt_version="technical_v1",
        latency_ms=1500.0,
    )

    decision = EnsembleDecision(
        collective_decision="Buy",
        collective_confidence=0.571,
        n_valid_signals=2,
        agreement=False,
        disagreement_entropy=0.954,
        opinion_dispersion=0.10,
        agent_decisions=["Buy", "Hold"],
        agent_confidences=[0.80, 0.60],
        winning_score=0.80,
        total_score=1.40,
    )

    output = EnsembleOutput(
        ticker="TEST",
        as_of_date="2023-03-31",
        fundamental_analysis=fundamental,
        technical_analysis=technical,
        ensemble_decision=decision,
        latency_ms=2500.0,
    )

    report = verify_ensemble(output, always_verify=True)
    return report, decision


# ---------------------------------------------------------------------------
# E2-T6: get_tracer() returns NoOpTracer when LANGFUSE_ENABLED=false
# ---------------------------------------------------------------------------


def test_get_tracer_returns_noop_when_disabled():
    # conftest.py session fixture already sets LANGFUSE_ENABLED=false,
    # but we use patch here to be explicit and isolated.
    with patch.dict(os.environ, {"LANGFUSE_ENABLED": "false"}):
        tracer = get_tracer()
    assert isinstance(tracer, NoOpTracer)


def test_get_tracer_returns_noop_for_all_falsy_variants():
    for value in ("false", "False", "FALSE", "0", "no", "No", "off", "Off"):
        with patch.dict(os.environ, {"LANGFUSE_ENABLED": value}):
            tracer = get_tracer()
        assert isinstance(tracer, NoOpTracer), f"Expected NoOpTracer for LANGFUSE_ENABLED={value!r}"


def test_get_tracer_returns_noop_when_keys_missing():
    env = {"LANGFUSE_ENABLED": "true", "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}
    with patch.dict(os.environ, env, clear=False):
        # Remove the keys entirely to simulate missing env vars
        env_without_keys = dict(os.environ)
        env_without_keys.pop("LANGFUSE_PUBLIC_KEY", None)
        env_without_keys.pop("LANGFUSE_SECRET_KEY", None)
        env_without_keys["LANGFUSE_ENABLED"] = "true"
        with patch.dict(os.environ, env_without_keys, clear=True):
            tracer = get_tracer()
    assert isinstance(tracer, NoOpTracer)


# ---------------------------------------------------------------------------
# E2-T7: NoOpTracer.start_trace() returns non-empty string trace ID
# ---------------------------------------------------------------------------


def test_noop_tracer_start_trace_returns_nonempty_string():
    tracer = NoOpTracer()
    trace_id = tracer.start_trace("test", ticker="AAPL", as_of_date="2023-03-31")
    assert isinstance(trace_id, str)
    assert len(trace_id) > 0


def test_noop_tracer_start_trace_returns_fixed_noop_id():
    tracer = NoOpTracer()
    trace_id = tracer.start_trace("fundamental_agent", ticker="JPM", as_of_date="2023-03-31")
    assert trace_id == "noop-trace"


# ---------------------------------------------------------------------------
# E2-T8: NoOpTracer.get_callback_handler() returns None
# ---------------------------------------------------------------------------


def test_noop_tracer_get_callback_handler_returns_none():
    tracer = NoOpTracer()
    handler = tracer.get_callback_handler("noop-trace")
    assert handler is None


# ---------------------------------------------------------------------------
# E2-T9: NoOpTracer.span() context manager enters and exits without exception
# ---------------------------------------------------------------------------


def test_noop_tracer_span_enters_and_exits_cleanly():
    tracer = NoOpTracer()
    with tracer.span("noop-trace", "mcp_get_technical_indicators", input={"ticker": "AAPL"}) as ctx:
        assert ctx is not None
        assert isinstance(ctx, SpanContext)
    # No exception raised; span exited cleanly


def test_noop_tracer_span_ctx_output_can_be_set():
    tracer = NoOpTracer()
    with tracer.span("noop-trace", "test_span") as ctx:
        ctx.output = {"result": 42}
        ctx.metadata = {"call_id": "abc123"}
    assert ctx.output == {"result": 42}
    assert ctx.metadata == {"call_id": "abc123"}


# ---------------------------------------------------------------------------
# E2-T10: NoOpTracer.log_score() with valid args does not raise
# ---------------------------------------------------------------------------


def test_noop_tracer_log_score_does_not_raise():
    tracer = NoOpTracer()
    tracer.log_score("noop-trace", "fundamental_hr", 0.0)
    tracer.log_score("noop-trace", "technical_gr", 1.0)
    tracer.log_score("noop-trace", "n_contradictions", 3.0)
    # All calls completed without exception


def test_noop_tracer_flush_does_not_raise():
    tracer = NoOpTracer()
    tracer.flush()


# ---------------------------------------------------------------------------
# E2-T11: trace_context() sets _current_trace_id within block, resets after
# ---------------------------------------------------------------------------


def test_trace_context_sets_and_resets():
    assert _current_trace_id.get() is None, "Precondition: no active trace"

    with trace_context("test-trace-001"):
        assert _current_trace_id.get() == "test-trace-001"

    assert _current_trace_id.get() is None


def test_trace_context_resets_on_exception():
    assert _current_trace_id.get() is None

    with pytest.raises(ValueError), trace_context("test-trace-002"):
        assert _current_trace_id.get() == "test-trace-002"
        raise ValueError("test exception")

    assert _current_trace_id.get() is None


# ---------------------------------------------------------------------------
# E2-T12: nested trace_context() restores outer value correctly
# ---------------------------------------------------------------------------


def test_trace_context_nested_restores_outer():
    """Nested trace_context uses ContextVar token semantics to restore correctly."""
    assert _current_trace_id.get() is None

    with trace_context("outer-trace"):
        assert _current_trace_id.get() == "outer-trace"

        with trace_context("inner-trace"):
            assert _current_trace_id.get() == "inner-trace"

        # After inner block exits, outer trace ID is restored
        assert _current_trace_id.get() == "outer-trace"

    assert _current_trace_id.get() is None


def test_trace_context_triple_nesting():
    with trace_context("level-1"):
        with trace_context("level-2"):
            with trace_context("level-3"):
                assert _current_trace_id.get() == "level-3"
            assert _current_trace_id.get() == "level-2"
        assert _current_trace_id.get() == "level-1"
    assert _current_trace_id.get() is None


# ---------------------------------------------------------------------------
# E5-T2: log_verification_scores() calls log_score() exactly 6 times
# ---------------------------------------------------------------------------


def test_log_verification_scores_calls_log_score_six_times():
    tracer = RecordingTracer()
    report, decision = _make_synthetic_verification_report()
    log_verification_scores(tracer, "test-trace", report, decision)
    assert len(tracer.scores) == 6


# ---------------------------------------------------------------------------
# E5-T3: score values match the six named metrics
# ---------------------------------------------------------------------------


def test_log_verification_scores_correct_names_and_values():
    tracer = RecordingTracer()
    report, decision = _make_synthetic_verification_report()
    log_verification_scores(tracer, "tid-001", report, decision)

    score_dict = {name: value for (_tid, name, value) in tracer.scores}

    assert "fundamental_hr" in score_dict
    assert "fundamental_gr" in score_dict
    assert "technical_hr" in score_dict
    assert "technical_gr" in score_dict
    assert "disagreement_entropy" in score_dict
    assert "n_contradictions" in score_dict

    assert score_dict["fundamental_hr"] == report.fundamental_report.hallucination_rate
    assert score_dict["fundamental_gr"] == report.fundamental_report.grounding_rate
    assert score_dict["technical_hr"] == report.technical_report.hallucination_rate
    assert score_dict["technical_gr"] == report.technical_report.grounding_rate
    assert score_dict["disagreement_entropy"] == decision.disagreement_entropy
    assert score_dict["n_contradictions"] == float(report.n_contradictions)


def test_log_verification_scores_trace_id_passed_to_all_scores():
    tracer = RecordingTracer()
    report, decision = _make_synthetic_verification_report()
    log_verification_scores(tracer, "specific-trace-id", report, decision)

    for trace_id, _name, _value in tracer.scores:
        assert trace_id == "specific-trace-id"


# ---------------------------------------------------------------------------
# E5-T4: log_verification_scores() with NoOpTracer does not raise
# ---------------------------------------------------------------------------


def test_log_verification_scores_with_noop_tracer_does_not_raise():
    tracer = NoOpTracer()
    report, decision = _make_synthetic_verification_report()
    log_verification_scores(tracer, "noop-trace", report, decision)
