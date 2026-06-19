"""
Observability tracing module for HiFi (P6-E2).

Provides a backend-agnostic AbstractTracer interface, a NoOpTracer for tests
and disabled-state production runs, and a LangFuseTracer that wraps the
LangFuse v3 Python SDK.

Architecture (DJ-023)
---------------------
Two integration mechanisms used together:

1. LangFuse CallbackHandler -- automatic tracing of all LangChain/LangGraph
   LLM calls. Zero changes to graph node implementations. Enabled by passing
   the handler in the LangGraph config dict ({"callbacks": [handler]}).

2. ContextVar-based MCP span injection -- _current_trace_id is set by
   trace_context() at the start of each agent run. call_tool() reads it to
   attach MCP tool spans to the active trace without any signature changes.
   Works correctly in synchronous LangGraph execution (Python ContextVar
   inherits through synchronous call stacks).

NoOpTracer is NOT a test mock (DJ-025). It is a real implementation of the
no-op case that ships in production code. Tests run with LANGFUSE_ENABLED=false
via the conftest.py session fixture, which causes get_tracer() to return
NoOpTracer. No test requires a live LangFuse instance.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hifi.collective.schemas import EnsembleDecision
    from hifi.verification.schemas import EnsembleVerificationReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ContextVar for MCP span injection (DJ-023)
# ---------------------------------------------------------------------------

# Thread-safe (and async-safe) current trace ID.
# Set by trace_context() via run_analysis() / run_ensemble().
# Read by call_tool() to create MCP child spans without signature changes.
_current_trace_id: ContextVar[str | None] = ContextVar("_current_trace_id", default=None)


@contextmanager
def trace_context(trace_id: str) -> Generator[None, None, None]:
    """Set the active trace ID for the duration of a with block.

    Uses ContextVar token semantics so nested trace_context() calls correctly
    restore the outer trace ID when the inner block exits.
    """
    token = _current_trace_id.set(trace_id)
    try:
        yield
    finally:
        _current_trace_id.reset(token)


# ---------------------------------------------------------------------------
# SpanContext: mutable output carrier yielded by span() context managers
# ---------------------------------------------------------------------------


@dataclass
class SpanContext:
    """Mutable container yielded by AbstractTracer.span() context managers.

    Callers set output and metadata within the with block so that the tracer
    can pass them to span.end() in the finally clause. This avoids a separate
    set_output() method on the tracer interface.

    Example usage in call_tool():
        with tracer.span(trace_id, "mcp_get_technical_indicators", input=params) as ctx:
            result = _call_subprocess(...)
            ctx.output = result
            ctx.metadata = {"call_id": result.get("call_id")}
    """

    output: dict | None = None
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# AbstractTracer interface
# ---------------------------------------------------------------------------


class AbstractTracer:
    """Interface that all tracer implementations must satisfy.

    AbstractTracer is the boundary between HiFi agent code and the
    observability backend. All instrumentation is written against this
    interface, not against LangFuse directly. If the backend changes,
    only tracing.py changes.
    """

    def start_trace(
        self, name: str, ticker: str, as_of_date: str, **metadata: Any
    ) -> str:
        """Create a new top-level trace. Returns the trace ID."""
        raise NotImplementedError

    def get_callback_handler(self, trace_id: str) -> Any | None:
        """Return a LangChain/LangGraph CallbackHandler scoped to this trace.

        Returns None when tracing is disabled (agents pass {} as config,
        preserving pre-Phase-6 behaviour exactly).
        """
        raise NotImplementedError

    @contextmanager
    def span(
        self, trace_id: str, name: str, input: dict | None = None
    ) -> Generator[SpanContext, None, None]:
        """Context manager for a child span (e.g., one MCP tool call).

        Yields a SpanContext. Callers set ctx.output and ctx.metadata inside
        the with block. The tracer reads them on exit to call span.end().
        """
        raise NotImplementedError
        yield SpanContext()  # pragma: no cover

    def log_score(self, trace_id: str, name: str, value: float) -> None:
        """Attach a named numeric score to an existing trace."""
        raise NotImplementedError

    def flush(self) -> None:
        """Flush any buffered events to the LangFuse server."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NoOpTracer (DJ-025)
# ---------------------------------------------------------------------------


class NoOpTracer(AbstractTracer):
    """No-op implementation of AbstractTracer.

    Identical interface as LangFuseTracer; performs no network operations and
    has no external dependencies. Used when LANGFUSE_ENABLED=false (all tests)
    and as a fail-open fallback when LangFuse is misconfigured or unavailable.

    NOT a test mock -- a real implementation of the no-op case (DJ-025).
    """

    _NOOP_TRACE_ID: str = "noop-trace"

    def start_trace(
        self, name: str, ticker: str, as_of_date: str, **metadata: Any
    ) -> str:
        return self._NOOP_TRACE_ID

    def get_callback_handler(self, trace_id: str) -> None:
        return None

    @contextmanager
    def span(
        self, trace_id: str, name: str, input: dict | None = None
    ) -> Generator[SpanContext, None, None]:
        ctx = SpanContext()
        yield ctx

    def log_score(self, trace_id: str, name: str, value: float) -> None:
        pass

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# LangFuseTracer
# ---------------------------------------------------------------------------


class LangFuseTracer(AbstractTracer):
    """LangFuse v3 implementation of AbstractTracer.

    Wraps the langfuse.Langfuse SDK client. The langfuse package is imported
    lazily inside __init__ -- the module is importable even if langfuse is
    not installed. If construction fails for any reason, get_tracer() catches
    the exception and falls back to NoOpTracer.

    Environment variables read at construction time:
    - LANGFUSE_HOST       (default: http://localhost:3000)
    - LANGFUSE_PUBLIC_KEY (required)
    - LANGFUSE_SECRET_KEY (required)
    """

    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        # Lazy import: do not move to module level.
        # Ensures tracing.py is importable without the langfuse package.
        from langfuse import Langfuse  # noqa: PLC0415

        self._host = host
        self._public_key = public_key
        self._secret_key = secret_key
        self._client = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
        )

    def start_trace(
        self, name: str, ticker: str, as_of_date: str, **metadata: Any
    ) -> str:
        trace = self._client.trace(
            name=name,
            metadata={"ticker": ticker, "as_of_date": as_of_date, **metadata},
        )
        return trace.id

    def get_callback_handler(self, trace_id: str) -> Any | None:
        try:
            from langfuse.callback import CallbackHandler  # noqa: PLC0415

            return CallbackHandler(
                public_key=self._public_key,
                secret_key=self._secret_key,
                host=self._host,
                trace_id=trace_id,
            )
        except Exception as exc:
            logger.warning("Failed to create LangFuse CallbackHandler: %s", exc)
            return None

    @contextmanager
    def span(
        self, trace_id: str, name: str, input: dict | None = None
    ) -> Generator[SpanContext, None, None]:
        span_obj = self._client.span(
            trace_id=trace_id,
            name=name,
            input=input or {},
        )
        ctx = SpanContext()
        try:
            yield ctx
        finally:
            span_obj.end(
                output=ctx.output,
                metadata=ctx.metadata,
            )

    def log_score(self, trace_id: str, name: str, value: float) -> None:
        self._client.score(
            trace_id=trace_id,
            name=name,
            value=value,
        )

    def flush(self) -> None:
        self._client.flush()


# ---------------------------------------------------------------------------
# Factory (DJ-025)
# ---------------------------------------------------------------------------


# Warn-once flags: avoid repeating the same fallback warning on every agent call.
# Each key maps to True once the warning has been emitted for this process.
_WARNED: dict[str, bool] = {}


def get_tracer() -> AbstractTracer:
    """Return LangFuseTracer when enabled, NoOpTracer otherwise.

    Controlled by LANGFUSE_ENABLED env var (default: true when unset).
    NoOpTracer is returned when:
      - LANGFUSE_ENABLED is false / 0 / no / off (case-insensitive)
      - LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY are not set
      - langfuse package is not installed
      - any other initialisation error (fail-open design)

    Fail-open means a misconfigured LangFuse instance never prevents an
    agent from running. The agent's functional behaviour is identical whether
    get_tracer() returns a NoOpTracer or a LangFuseTracer.

    Warnings are emitted at most once per process per failure mode to avoid
    log noise during batch evaluation runs (e.g. E0-T5 with 120+ agent calls).
    """
    enabled_raw = os.environ.get("LANGFUSE_ENABLED", "true").lower().strip()
    if enabled_raw in ("false", "0", "no", "off"):
        return NoOpTracer()

    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        if not _WARNED.get("no_keys"):
            logger.warning(
                "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set; "
                "falling back to NoOpTracer."
            )
            _WARNED["no_keys"] = True
        return NoOpTracer()

    try:
        return LangFuseTracer(host=host, public_key=public_key, secret_key=secret_key)
    except ImportError:
        if not _WARNED.get("no_pkg"):
            logger.warning("langfuse package not installed; falling back to NoOpTracer.")
            _WARNED["no_pkg"] = True
        return NoOpTracer()
    except Exception as exc:
        if not _WARNED.get("init_fail"):
            logger.warning(
                "LangFuse initialisation failed (%s); falling back to NoOpTracer.", exc
            )
            _WARNED["init_fail"] = True
        return NoOpTracer()


# ---------------------------------------------------------------------------
# Verification score logging (P6-E5, DJ-024)
# ---------------------------------------------------------------------------


def log_verification_scores(
    tracer: AbstractTracer,
    trace_id: str,
    verification_report: EnsembleVerificationReport,
    ensemble_decision: EnsembleDecision,
) -> None:
    """Log Phase 5 verification metrics as LangFuse scores on a trace.

    Six scores are logged per ensemble trace (DJ-024):
      fundamental_hr      -- Fundamental Agent hallucination rate
      fundamental_gr      -- Fundamental Agent grounding rate
      technical_hr        -- Technical Agent hallucination rate
      technical_gr        -- Technical Agent grounding rate
      disagreement_entropy -- Shannon entropy over the vote distribution
      n_contradictions    -- Number of cross-agent field contradictions

    These become the data points from which the RAG hypothesis (Phase 7) and
    backtesting analysis (Phase 10) are tested. Every run_ensemble() call
    from Phase 6 onward extends the time series.
    """
    tracer.log_score(
        trace_id, "fundamental_hr",
        verification_report.fundamental_report.hallucination_rate,
    )
    tracer.log_score(
        trace_id, "fundamental_gr",
        verification_report.fundamental_report.grounding_rate,
    )
    tracer.log_score(
        trace_id, "technical_hr",
        verification_report.technical_report.hallucination_rate,
    )
    tracer.log_score(
        trace_id, "technical_gr",
        verification_report.technical_report.grounding_rate,
    )
    tracer.log_score(
        trace_id, "disagreement_entropy",
        ensemble_decision.disagreement_entropy,
    )
    tracer.log_score(
        trace_id, "n_contradictions",
        float(verification_report.n_contradictions),
    )
