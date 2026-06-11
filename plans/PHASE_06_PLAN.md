# Phase 6: Observability — LangFuse

**Status:** PLANNED

| Epic | Title | Status |
|---|---|---|
| P6-E1 | LangFuse infrastructure (Docker + setup doc) | PLANNED |
| P6-E2 | Tracing module (AbstractTracer, NoOpTracer, LangFuseTracer) | PLANNED |
| P6-E3 | Agent instrumentation (LLM call tracing via CallbackHandler) | PLANNED |
| P6-E4 | MCP span instrumentation (ContextVar-based span injection) | PLANNED |
| P6-E5 | Verification score logging (HR/GR as LangFuse scores) | PLANNED |
| P6-E6 | Ensemble trace + holistic test + one-time baseline run | PLANNED |

**David Sections:** §14 Observability (full section), §4.4 Observability principle
**Learning Guide Topics:** 6.1 LLM Observability, 6.4 Experiment Tracking (foundations)
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 6

---

## Governing Philosophy for This Phase

Phase 6 is the scientific instrumentation layer of HiFi.

A multi-agent collective intelligence system is only as scientifically rigorous as the
measurements it produces about itself. Phases 3–5 built agents that reason and a verifier
that measures factual accuracy at a single point in time. But complexity science is not a
science of snapshots — it is a science of trajectories. The hallucination rate measured on
three tickers at one date (Phase 5 baseline) is a single observation. The hallucination rate
measured across hundreds of tickers over months, with the distribution evolving as the
system improves through RAG and fine-tuning, is a time series. A time series is what allows
scientific claims about systemic behaviour.

Phase 6 creates the apparatus that turns single observations into trajectories. Every
`run_ensemble()` call from Phase 6 onward produces a structured, queryable record: which
agents ran, what they received from MCP tools, what they inferred, what the verifier
measured, and what the collective decided. These records are the raw material for the
scientific analysis in Phase 10 (Evaluation and Backtesting).

The David's definition (§4.4) is exact: "Observability is not debugging. Observability is
the scientific instrumentation of the system." A debugger helps you find a bug. A
measurement apparatus helps you understand a system. Phase 6 builds the latter.

**The three measurement scales.** David §14.1 defines three observability functions:
(1) operational monitoring (is the system working?), (2) scientific instrumentation (what
are the dynamics of the agent population?), and (3) auditability (can every decision be
traced to its inputs?). These map directly to the LangFuse trace hierarchy:

- **Macro (trace):** One `run_ensemble()` = one trace. Metadata: ticker, as_of_date, total
  latency, verification scores, collective decision. This is the unit of scientific observation.
- **Meso (span):** One agent execution = one span group. Within it: each MCP tool call is a
  child span. This is the agent-population level — heterogeneity of behaviour across agents.
- **Micro (generation):** One LLM inference = one generation. Token counts, prompt, completion,
  latency. This is the model-level signal — calibration, verbosity, reasoning patterns.

All three levels are populated by Phase 6. The LangFuse dashboard gives immediate visual access
to the time series at all three scales.

**Why LangFuse, and why Docker Compose.** LangFuse is chosen in David §14.2 because it is
open-source, self-hostable, and supports traces, spans, generations, and evaluation scores in
a unified interface. It integrates natively with LangChain/LangGraph, which means LLM calls
are traced automatically without touching the agent code. The self-hosted requirement is
non-negotiable: financial data processed by HiFi (market data, macro indicators, rationale
text containing specific ticker names and numerical claims) must not leave the local machine.
Cloud LangFuse (langfuse.com) is therefore excluded regardless of its technical capabilities.

Docker Compose is the only viable path for self-hosted LangFuse v3. The v3 stack (Postgres,
ClickHouse, Redis, LangFuse server + worker) is heavier than a single process but provides
the ClickHouse analytics backend that makes historical queries over thousands of traces
fast. On the development machine (Apple M3 Ultra, 192 GB RAM) running 35B LLMs in LM
Studio, the Docker Compose overhead is trivial.

**Why NoOpTracer is not a mock.** The existing test suite (551 tests) is entirely
deterministic. It must remain so. No test should require a live LangFuse instance. The
solution is a `NoOpTracer` that implements the identical interface as `LangFuseTracer` but
performs no network operations. This is not a mock in the test-double sense — it is a
real implementation of the no-op case that ships in production code. The same code path
runs in tests (with `NoOpTracer`) and in production (with `LangFuseTracer`). The only
difference is whether spans are transmitted over HTTP. This follows the same architectural
principle as the MCP client's subprocess isolation: the interface is the boundary.

---

## Background: What the Existing Pipeline Lacks

The Phase 3–5 pipeline produces rich structured outputs (AgentSignal, EnsembleOutput,
EnsembleVerificationReport) but all of this information is local to the Python process that
created it. Nothing is persisted between runs in a queryable form. The Phase 3, 4, and 5
baseline JSON fixtures are snapshots — three tickers, one date, recorded once. There is no
infrastructure to answer questions like:

- "What is the mean latency of the Technical Agent over the last 20 runs?"
- "Has the hallucination rate changed since the Phase 5 baseline?"
- "Which tickers consistently produce low-confidence signals?"
- "What was the ensemble disagreement entropy distribution over Q1 2023?"

Phase 6 makes all of these questions answerable. Every `run_ensemble()` call writes to a
persistent, queryable store. The LangFuse dashboard renders this as time series charts,
distributions, and per-trace drilldowns. When Phase 7 (RAG) is added, the LangFuse score
history will immediately show whether the hallucination rate is decreasing — the direct
empirical test of the RAG hypothesis.

**The call_id hook pays off again.** Phase 2 embedded `call_id` in every MCP tool result.
Phase 5 used it for per-claim verification. Phase 6 uses it to name MCP spans in LangFuse:
each span's name includes the call_id, creating a direct link between the LangFuse trace
and the specific tool invocation that produced a result. A human auditor looking at a
LangFuse trace can identify exactly which MCP call produced each verified or hallucinated
claim.

---

## Key Decisions To Make in This Phase

**DJ-022: LangFuse deployment — Docker Compose v3 (confirmed)**

LangFuse v3 requires Docker Compose with five services: `langfuse-web` (UI + API),
`langfuse-worker` (background task processor), `db` (Postgres 16), `clickhouse`
(ClickHouse 24, analytics backend), and `redis` (message queue). The v2 setup without
ClickHouse would be simpler but does not support the analytics query performance needed
for Phase 10's historical analysis. The decision is to use v3 from the start.

Confirmed at Phase 6 plan creation based on:
(1) local hardware can absorb the overhead,
(2) the ClickHouse analytics backend is scientifically necessary at Phase 10 scale, and
(3) migrating from v2 to v3 after data is accumulated is more disruptive than starting with v3.

This decision to be recorded as DJ-022 at P6-E1.

**DJ-023: LangGraph integration — hybrid (confirmed)**

Three integration paths exist for LangFuse + LangGraph:
1. LangFuse `CallbackHandler`: automatically traces all LangGraph node executions and all
   LangChain LLM calls. Zero agent code changes for the LLM tracing. Does not capture MCP
   subprocess calls (which are not LangChain operations).
2. Manual SDK: explicit `trace.span()` for every operation. Full control, maximum code surface.
3. Hybrid: `CallbackHandler` for LLM calls + `ContextVar` for MCP calls.

The hybrid approach is chosen. The `CallbackHandler` handles everything that goes through
LangChain automatically. MCP spans are added via a Python `ContextVar` that holds the
current trace ID — set at the start of `run_analysis()`, read by `call_tool()` without
requiring the tracer or trace ID to be threaded through the LangGraph state schema.

The `ContextVar` approach has a key advantage: it is transparent to existing function
signatures. `call_tool()` gains no new parameters. The LangGraph state schema gains no new
fields. The instrumentation is a side-channel that does not couple the tracing logic to the
agent logic.

This decision to be recorded as DJ-023 at P6-E4.

**DJ-024: Verification scores — log immediately (confirmed)**

Phase 5 HR/GR metrics are the primary quality signal for the system. Deferring their logging
to Phase 10 would mean losing the time series for all Phase 6–9 runs. Every `run_ensemble()`
after Phase 6 should produce six scores on its trace: `fundamental_hr`, `fundamental_gr`,
`technical_hr`, `technical_gr`, `disagreement_entropy`, `n_contradictions`. These become the
data points from which trend analysis and the Phase 7 RAG hypothesis are tested.

This decision to be recorded as DJ-024 at P6-E5.

**DJ-025: Test isolation — NoOpTracer + LANGFUSE_ENABLED env var (confirmed)**

No test in the existing suite (551 tests) or any new Phase 6 test should require a live
LangFuse server. The tracing module is enabled/disabled via the `LANGFUSE_ENABLED`
environment variable (default: `true` in production, `false` in tests via pytest `conftest.py`
fixture). When disabled, `get_tracer()` returns a `NoOpTracer`. Tests that specifically
validate tracing behaviour do so against `NoOpTracer` (verifying the correct methods are
called with correct arguments) without any network operations.

This decision to be recorded as DJ-025 at P6-E2.

---

## Epic P6-E1: LangFuse Infrastructure

**Objective:** Provide a reproducible, single-command LangFuse v3 setup. Every developer
who clones HiFi should be able to start the observability stack with one command and see
the dashboard within 60 seconds. The infrastructure is documented, not assumed.

**Docker Compose setup.**

File: `docker/langfuse/docker-compose.yml` — references the official LangFuse v3 image
(`langfuse/langfuse:3`) with all five services configured. The Compose file uses environment
variables loaded from `docker/langfuse/.env.example` (which is committed with safe defaults
for local development). A developer copies `.env.example` to `.env`, then runs:

```
docker compose -f docker/langfuse/docker-compose.yml up -d
```

LangFuse UI is accessible at `http://localhost:3000`. First-time setup: create an
organisation and project in the UI, then copy the public and secret keys to `.env`.

**Environment variables for the Python SDK.** The Python side reads:
- `LANGFUSE_HOST` — default `http://localhost:3000`
- `LANGFUSE_PUBLIC_KEY` — project public key from LangFuse UI
- `LANGFUSE_SECRET_KEY` — project secret key from LangFuse UI
- `LANGFUSE_ENABLED` — `true` | `false` (default `true`; overridden to `false` in tests)

These are documented in `doc/setup/LANGFUSE_SETUP.md` alongside a step-by-step setup
walkthrough. The setup doc is a perennial document (not tied to a specific version) that
will be updated if the LangFuse version changes.

**Note:** `docker/langfuse/.env` is in `.gitignore` (contains local API keys). Only
`.env.example` is committed.

| Ticket | Description | Status |
|---|---|---|
| P6-E1-T1 | Create docker/langfuse/docker-compose.yml with langfuse-web, langfuse-worker, db, clickhouse, redis services | PLANNED |
| P6-E1-T2 | Create docker/langfuse/.env.example with all required env vars and safe defaults | PLANNED |
| P6-E1-T3 | Add docker/langfuse/.env to .gitignore | PLANNED |
| P6-E1-T4 | Create doc/setup/LANGFUSE_SETUP.md: step-by-step guide from docker compose up to first trace in dashboard | PLANNED |
| P6-E1-T5 | Manual verification: docker compose up, UI accessible at localhost:3000, project created, keys extracted | PLANNED |
| P6-E1-T6 | Record DJ-022: LangFuse v3 Docker Compose confirmed as deployment method | PLANNED |

**Files to create:**
- `docker/langfuse/docker-compose.yml`
- `docker/langfuse/.env.example`
- `doc/setup/LANGFUSE_SETUP.md`

---

## Epic P6-E2: Tracing Module

**Objective:** Build the abstraction layer between HiFi and LangFuse. This layer ensures
that all instrumentation code in the agents and verifier is written against a stable
interface (`AbstractTracer`), not against LangFuse directly. If LangFuse is ever replaced
by a different observability backend, only `tracing.py` changes.

**Interface design.**

```python
# src/hifi/observability/tracing.py

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator

# Thread-safe (and async-safe) current trace ID.
# Set by run_ensemble() via trace_context(); read by call_tool() for MCP spans.
_current_trace_id: ContextVar[str | None] = ContextVar("_current_trace_id", default=None)

@contextmanager
def trace_context(trace_id: str) -> Generator[None, None, None]:
    """Set the active trace ID for the duration of a with block."""
    token = _current_trace_id.set(trace_id)
    try:
        yield
    finally:
        _current_trace_id.reset(token)


class AbstractTracer:
    """Interface that all tracer implementations must satisfy."""

    def start_trace(
        self, name: str, ticker: str, as_of_date: str, **metadata: Any
    ) -> str:
        """Create a new top-level trace. Returns the trace ID."""
        raise NotImplementedError

    def get_callback_handler(self, trace_id: str) -> Any | None:
        """Return a LangChain/LangGraph CallbackHandler scoped to this trace.
        Returns None when tracing is disabled (agents pass empty callbacks list)."""
        raise NotImplementedError

    @contextmanager
    def span(
        self, trace_id: str, name: str, input: dict | None = None
    ) -> Generator[None, None, None]:
        """Context manager for a child span (e.g., one MCP tool call)."""
        raise NotImplementedError
        yield  # pragma: no cover

    def log_score(self, trace_id: str, name: str, value: float) -> None:
        """Attach a named numeric score to an existing trace."""
        raise NotImplementedError

    def flush(self) -> None:
        """Flush any buffered events to the LangFuse server."""
        raise NotImplementedError
```

**NoOpTracer.** Implements `AbstractTracer` with no-ops. Returns a fixed dummy trace ID
(`"noop-trace"`) from `start_trace()`. Returns `None` from `get_callback_handler()`.
The `span()` context manager enters and exits without doing anything. `log_score()` and
`flush()` are no-ops. The `NoOpTracer` is importable without the `langfuse` package
installed — it has no external dependencies.

**LangFuseTracer.** Wraps the `langfuse.Langfuse` SDK client. Initialised with `host`,
`public_key`, `secret_key` from environment variables. `start_trace()` calls
`self._client.trace(name=name, metadata={...})` and returns `trace.id`.
`get_callback_handler(trace_id)` constructs a `langfuse.callback.CallbackHandler` scoped
to the trace. `span()` creates a child span via `self._client.span(trace_id=trace_id,
name=name, input=input)` and ends it on context manager exit. `log_score()` calls
`self._client.score(trace_id=trace_id, name=name, value=value)`.

**get_tracer() factory.**

```python
def get_tracer() -> AbstractTracer:
    """Return LangFuseTracer when enabled, NoOpTracer otherwise.

    Controlled by LANGFUSE_ENABLED env var (default: true).
    NoOpTracer is returned when:
      - LANGFUSE_ENABLED=false
      - LANGFUSE_HOST / PUBLIC_KEY / SECRET_KEY are not set
      - langfuse package is not installed
    """
```

The factory tries to import and initialise `LangFuseTracer`. Any failure (missing env vars,
missing package, connection refused) falls back to `NoOpTracer` with a logged warning.
This fail-open design ensures that a misconfigured LangFuse instance never prevents an
agent from running.

**conftest.py update.** A session-scoped pytest fixture `disable_langfuse` sets
`LANGFUSE_ENABLED=false` for all tests. This ensures `get_tracer()` always returns
`NoOpTracer` in the test suite, without requiring any test-level patching.

| Ticket | Description | Status |
|---|---|---|
| P6-E2-T1 | Implement AbstractTracer, trace_context(), _current_trace_id in src/hifi/observability/tracing.py | PLANNED |
| P6-E2-T2 | Implement NoOpTracer; importable with no langfuse dependency | PLANNED |
| P6-E2-T3 | Implement LangFuseTracer wrapping langfuse.Langfuse; lazy-import to avoid hard dependency at module load | PLANNED |
| P6-E2-T4 | Implement get_tracer() factory with fail-open fallback to NoOpTracer | PLANNED |
| P6-E2-T5 | Add LANGFUSE_ENABLED=false fixture to tests/conftest.py (session-scoped; sets env var) | PLANNED |
| P6-E2-T6 | Unit test: get_tracer() returns NoOpTracer when LANGFUSE_ENABLED=false | PLANNED |
| P6-E2-T7 | Unit test: NoOpTracer.start_trace() returns a non-empty string trace ID | PLANNED |
| P6-E2-T8 | Unit test: NoOpTracer.get_callback_handler() returns None | PLANNED |
| P6-E2-T9 | Unit test: NoOpTracer.span() context manager enters and exits without exception | PLANNED |
| P6-E2-T10 | Unit test: NoOpTracer.log_score() called with valid args does not raise | PLANNED |
| P6-E2-T11 | Unit test: trace_context() sets _current_trace_id within block, resets after | PLANNED |
| P6-E2-T12 | Unit test: nested trace_context() calls restore outer value correctly (ContextVar token semantics) | PLANNED |
| P6-E2-T13 | Record DJ-025: NoOpTracer + LANGFUSE_ENABLED env var confirmed as test isolation strategy | PLANNED |

**Files to create/modify:**
- `src/hifi/observability/tracing.py` (new — observability package stub exists from Phase 0)
- `tests/conftest.py` (add session-scoped fixture)
- `tests/unit/test_tracing_module.py` (new)

---

## Epic P6-E3: Agent Instrumentation

**Objective:** Instrument the Fundamental and Technical agents so that every LLM inference
inside a LangGraph graph execution produces a LangFuse generation record. The LangFuse
`CallbackHandler` is the right tool here: it integrates directly with LangChain's callback
system and automatically captures input messages, output completions, token counts,
model name, and latency for every ChatOpenAI call — without any changes to the graph
node implementations themselves.

**Integration pattern.** Each `run_analysis()` and `run_technical_analysis()` function
accepts an optional `tracer: AbstractTracer | None = None` parameter. When called:

```python
def run_analysis(snapshot: FundamentalsSnapshot, tracer: AbstractTracer | None = None) -> FundamentalAnalysis:
    _tracer = tracer or get_tracer()
    trace_id = _tracer.start_trace(
        "fundamental_agent", ticker=snapshot.ticker, as_of_date=str(snapshot.period_end)
    )
    handler = _tracer.get_callback_handler(trace_id)
    config = {"callbacks": [handler]} if handler is not None else {}
    with trace_context(trace_id):          # makes trace_id available to call_tool()
        result = graph.invoke(initial_state, config=config)
    _tracer.flush()
    return _build_result(result)
```

The `trace_context(trace_id)` call sets `_current_trace_id` for the duration of the
LangGraph execution, making it available to `call_tool()` in E4 without passing it through
the graph state.

When `tracer=None` and `get_tracer()` returns `NoOpTracer`, `get_callback_handler()`
returns `None`, `config` is `{}`, and the agents run exactly as before Phase 6. The
behavioural footprint of the tracing layer, when disabled, is zero.

**Note on graph-level trace vs. agent-level trace.** At Phase 6, each agent creates its own
trace. At Phase 8+ (multi-agent), the parent trace is created by `run_ensemble()` and the
agents create child spans within it. The Phase 6 design deliberately makes the trace
creation point a parameter: when `run_analysis()` is called standalone (as in Phase 3
tests), it creates its own trace. When called from `run_ensemble()` (Phase 6 E6), it
receives the ensemble's trace as its trace context. This transition requires no refactoring.

| Ticket | Description | Status |
|---|---|---|
| P6-E3-T1 | Modify fundamental_agent.run_analysis() to accept optional tracer; start trace, pass CallbackHandler to graph.invoke(), wrap with trace_context() | PLANNED |
| P6-E3-T2 | Modify technical_agent.run_technical_analysis() identically | PLANNED |
| P6-E3-T3 | Unit test: run_analysis() with NoOpTracer produces valid FundamentalAnalysis (regression) | PLANNED |
| P6-E3-T4 | Unit test: run_technical_analysis() with NoOpTracer produces valid TechnicalAnalysis (regression) | PLANNED |
| P6-E3-T5 | Unit test: run_analysis() with stub tracer records start_trace() and flush() called once | PLANNED |
| P6-E3-T6 | Unit test: run_technical_analysis() with stub tracer records start_trace() and flush() called once | PLANNED |
| P6-E3-T7 | Regression: all existing Phase 3 and Phase 4 agent tests pass without modification | PLANNED |

**Files to modify:**
- `src/hifi/agents/fundamental_agent.py`
- `src/hifi/agents/technical_agent.py`

---

## Epic P6-E4: MCP Span Instrumentation

**Objective:** Add a child span to the active trace for each `call_tool()` invocation.
The span records the tool name, arguments (input), and the tool result (output), with the
call_id as the span ID. This makes the LangFuse trace a complete audit trail from LLM
inference down to the exact subprocess call that produced each tool result.

**ContextVar-based injection.** The `_current_trace_id` ContextVar (set by
`trace_context()` in E3) makes the active trace ID available inside `call_tool()`
without changing its signature:

```python
# src/hifi/agents/mcp_client.py

from hifi.observability.tracing import _current_trace_id, get_tracer

def call_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    trace_id = _current_trace_id.get()
    if trace_id is None:
        return _call_tool_impl(tool_name, args)
    tracer = get_tracer()
    with tracer.span(trace_id, name=f"mcp_{tool_name}", input=args):
        return _call_tool_impl(tool_name, args)
```

When `trace_id is None` (no active trace context — e.g., in tests, or in standalone
`call_tool()` calls), the function behaves exactly as before. The ContextVar is only
populated during an instrumented `run_analysis()` or `run_ensemble()` call.

**What the MCP span captures:**
- `name`: `"mcp_get_financial_ratios"`, `"mcp_get_technical_indicators"`, etc.
- `input`: the tool arguments dict (ticker, date)
- `output`: the tool result dict (set when the span exits)
- `metadata.call_id`: the 12-character SHA-256 prefix embedded in the tool result

The `call_id` in the span metadata creates a direct cross-reference between the LangFuse
span and the `call_ids` list in `AgentSignal`, and the `VerificationResult.tool_field` in
the Phase 5 report. A human auditor can look at any LangFuse trace, find the MCP span
for `get_technical_indicators`, read its `call_id`, and trace it to the specific
`VerificationResult` that checked the RSI claim.

**Design note on ContextVar threading.** Python's `contextvars.ContextVar` is correctly
propagated through the LangGraph synchronous execution model used in Phases 3–5. LangGraph
calls Python functions synchronously, and `ContextVar` values are inherited by inner
function calls. If Phase 9 introduces `asyncio`-based parallelism, `ContextVar` works
correctly with async context management as well — no refactoring required.

| Ticket | Description | Status |
|---|---|---|
| P6-E4-T1 | Modify call_tool() in mcp_client.py to read _current_trace_id and create span when active | PLANNED |
| P6-E4-T2 | Unit test: call_tool() with no active trace context → no span created, result unchanged | PLANNED |
| P6-E4-T3 | Unit test: call_tool() with active trace_context → tracer.span() called with correct name and input | PLANNED |
| P6-E4-T4 | Unit test: call_tool() span captures tool result as output on exit | PLANNED |
| P6-E4-T5 | Unit test: call_tool() span captures call_id in metadata when present in result | PLANNED |
| P6-E4-T6 | Regression: all existing MCP client tests pass (call_tool behaviour unchanged when no trace context) | PLANNED |
| P6-E4-T7 | Record DJ-023: ContextVar-based MCP span injection confirmed; no state schema changes required | PLANNED |

**Files to modify:**
- `src/hifi/agents/mcp_client.py`

---

## Epic P6-E5: Verification Score Logging

**Objective:** After each `verify_ensemble()` call, log the Phase 5 verification metrics
as LangFuse scores on the ensemble trace. This creates the historical time series of
system quality that later phases will analyse and improve against.

**Scores logged per ensemble trace:**

| Score name | Source | Phase 5 baseline value (AAPL/JPM/XOM mean) |
|---|---|---|
| `fundamental_hr` | fundamental_report.hallucination_rate | 0.000 |
| `fundamental_gr` | fundamental_report.grounding_rate | 1.000 |
| `technical_hr` | technical_report.hallucination_rate | 0.067 |
| `technical_gr` | technical_report.grounding_rate | 0.667 |
| `disagreement_entropy` | ensemble_decision.disagreement_entropy | 0.000 |
| `n_contradictions` | ensemble_report.n_contradictions (as float) | 0.000 |

These six scores are sufficient to track the primary quality signals at the ensemble level.
Individual-claim-level scores (e.g., per-ticker HR) are retrievable from the span metadata
for deep dives but are not logged as top-level LangFuse scores to avoid polluting the
dashboard with excessive series.

**Implementation.** A standalone helper `log_verification_scores()` in
`src/hifi/observability/tracing.py`:

```python
def log_verification_scores(
    tracer: AbstractTracer,
    trace_id: str,
    verification_report: EnsembleVerificationReport,
    ensemble_decision: EnsembleDecision,
) -> None:
    """Log Phase 5 verification metrics as LangFuse scores on a trace."""
    tracer.log_score(trace_id, "fundamental_hr",
                     verification_report.fundamental_report.hallucination_rate)
    tracer.log_score(trace_id, "fundamental_gr",
                     verification_report.fundamental_report.grounding_rate)
    tracer.log_score(trace_id, "technical_hr",
                     verification_report.technical_report.hallucination_rate)
    tracer.log_score(trace_id, "technical_gr",
                     verification_report.technical_report.grounding_rate)
    tracer.log_score(trace_id, "disagreement_entropy",
                     ensemble_decision.disagreement_entropy)
    tracer.log_score(trace_id, "n_contradictions",
                     float(verification_report.n_contradictions))
```

This function is called by `run_ensemble()` (in E6) after `verify_ensemble()`. The
`log_verification_scores` helper is also independently testable.

| Ticket | Description | Status |
|---|---|---|
| P6-E5-T1 | Implement log_verification_scores() in src/hifi/observability/tracing.py | PLANNED |
| P6-E5-T2 | Unit test: log_verification_scores() calls tracer.log_score() exactly 6 times with correct names | PLANNED |
| P6-E5-T3 | Unit test: score values match fundamental_hr, fundamental_gr, technical_hr, technical_gr, disagreement_entropy, n_contradictions | PLANNED |
| P6-E5-T4 | Unit test: log_verification_scores() with NoOpTracer raises no exception | PLANNED |
| P6-E5-T5 | Record DJ-024: verification scores logged immediately in Phase 6; six named scores per ensemble trace | PLANNED |

**Files to modify:**
- `src/hifi/observability/tracing.py` (add log_verification_scores())

---

## Epic P6-E6: Ensemble Trace + Holistic Test + One-Time Baseline Run

**Objective:** Wire all instrumentation into `run_ensemble()` as a cohesive parent trace.
Test the full instrumented pipeline holistically. Provide a one-time script that generates
a real LangFuse trace from the Phase 4 fixture data so the trace hierarchy can be visually
verified in the dashboard.

**Ensemble trace design.** `run_ensemble()` becomes the owner of the top-level trace:

```python
def run_ensemble(ticker: str, snapshot: FundamentalsSnapshot, tracer: AbstractTracer | None = None) -> EnsembleOutput:
    _tracer = tracer or get_tracer()
    trace_id = _tracer.start_trace("run_ensemble", ticker=ticker, ...)
    with trace_context(trace_id):
        # 1. Run Fundamental Agent (creates sub-trace via E3; MCP spans via E4)
        fundamental = run_analysis(snapshot, tracer=_tracer)
        # 2. Run Technical Agent
        technical = run_technical_analysis(ticker, ..., tracer=_tracer)
        # 3. Collective decision
        decision = _make_collective_decision(fundamental.signal, technical.signal)
        # 4. Verification
        output = EnsembleOutput(...)
        verification = verify_ensemble(output)
        # 5. Log verification scores
        log_verification_scores(_tracer, trace_id, verification, decision)
    _tracer.flush()
    return output
```

**Note on trace hierarchy vs. separate per-agent traces.** At Phase 6, each agent call
within `run_ensemble()` passes the ensemble's `_tracer` but starts its own trace within
the callback handler. This creates two separate LangFuse traces per `run_ensemble()` call
(one per agent) plus the parent ensemble trace. Phase 9 will consolidate these into a
single parent-child hierarchy using LangFuse's session linking. The Phase 6 design is the
minimal working instrumentation that does not require Phase 9 refactoring; it is not
architecturally wrong, just less visually consolidated in the dashboard.

**Holistic test.** `tests/holistic/test_phase6_observability_pipeline.py` verifies:
1. `run_ensemble()` with `NoOpTracer` completes and returns a valid `EnsembleOutput`
2. `NoOpTracer.start_trace()` was called (tracer method invocation tracking)
3. `log_verification_scores()` was called with the correct trace_id and non-None report
4. `flush()` was called exactly once per `run_ensemble()` call
5. Phase 5 regression: `verify_ensemble()` still produces a valid `EnsembleVerificationReport`
6. Phase 4 regression: `run_ensemble()` without an explicit tracer still produces valid output

**One-time baseline script.** `scripts/run_phase6_tracing.py` requires a live LangFuse
instance (`LANGFUSE_ENABLED=true`, keys configured). It reads the Phase 4 ensemble fixture
(no LLM required — it replays verification on already-recorded outputs) and logs the Phase 5
metrics as LangFuse scores, creating one trace per ticker. After running, the LangFuse
dashboard should show three traces with six scores each, matching the Phase 5 baseline values.
This script is the manual verification that the entire E2–E5 stack is wired correctly end-to-end.

| Ticket | Description | Status |
|---|---|---|
| P6-E6-T1 | Modify run_ensemble() in ensemble_runner.py to create parent trace, pass tracer to agents, call log_verification_scores() | PLANNED |
| P6-E6-T2 | Unit test: run_ensemble() with NoOpTracer passes tracer to run_analysis() and run_technical_analysis() | PLANNED |
| P6-E6-T3 | Unit test: run_ensemble() calls log_verification_scores() with the ensemble trace_id | PLANNED |
| P6-E6-T4 | Holistic test: full pipeline with NoOpTracer (all 6 assertions listed above) | PLANNED |
| P6-E6-T5 | Holistic test: Phase 5 regression — verify_ensemble output unchanged | PLANNED |
| P6-E6-T6 | Holistic test: Phase 4 regression — run_ensemble() without explicit tracer works | PLANNED |
| P6-E6-T7 | Write scripts/run_phase6_tracing.py: reads phase4_ensemble.json, logs verification scores to live LangFuse | PLANNED |
| P6-E6-T8 | Manual verification: run phase6_tracing.py with live LangFuse; confirm 3 traces + 6 scores in dashboard | PLANNED |

**Files to modify/create:**
- `src/hifi/agents/ensemble_runner.py`
- `scripts/run_phase6_tracing.py` (new)
- `tests/integration/test_agent_tracing.py` (new — integration tests for E3–E5)
- `tests/holistic/test_phase6_observability_pipeline.py` (new)

---

## Epic Dependency Graph

```
P6-E1 (Infrastructure)    [independent, manual verification]
         |
         | (human sets up Docker, confirms UI works)
         |
P6-E2 (Tracing Module)
    AbstractTracer, NoOpTracer, LangFuseTracer, get_tracer(), trace_context()
         |
         +----------------------------+----------------------------+
         |                            |                            |
P6-E3 (Agent Instrumentation)  P6-E4 (MCP Spans)         P6-E5 (Verification Scores)
 fundamental_agent.py           mcp_client.py             log_verification_scores()
 technical_agent.py             ContextVar injection       tracing.py addition
         |                            |                            |
         +----------------------------+----------------------------+
                                      |
                              P6-E6 (Ensemble Trace + Holistic)
                              run_ensemble() orchestrates all
```

E1 is infrastructure with no code dependency. E2 is the foundation for E3, E4, and E5,
which are independent of each other. E6 depends on all of E2–E5.

---

## New Dependencies

**Production dependency:** `langfuse>=2.0`

Added to `pyproject.toml` under `[project.dependencies]`. The `langfuse` package is a
lazy import in `LangFuseTracer` — it is only imported when `get_tracer()` attempts to
create a `LangFuseTracer`. If the package is not installed, `get_tracer()` falls back to
`NoOpTracer` with a warning. This ensures the codebase is importable and all tests pass
even if `langfuse` is not installed (e.g., in a minimal CI environment).

**Infrastructure dependency:** Docker (not a Python package). Required only to run the
LangFuse observability stack locally. Not required for any test to pass.

---

## Phase 6 Quality Gates

| Gate | Criterion | Measured By |
|---|---|---|
| All unit tests pass | pytest tests/unit/, 0 failures | Manual run |
| All integration tests pass | pytest tests/integration/, 0 failures | Manual run |
| Holistic test passes | pytest tests/holistic/test_phase6_observability_pipeline.py | Manual run |
| Phase 4 regression | pytest tests/holistic/test_phase4_ensemble_pipeline.py still passes | Manual run |
| Phase 5 regression | pytest tests/holistic/test_phase5_verification_pipeline.py still passes | Manual run |
| No live LangFuse required | All tests pass with LANGFUSE_ENABLED=false | pytest run |
| Existing 551 tests pass | Full pytest suite, 0 regressions | Manual run |
| Lint clean | ruff check src/ tests/ scripts/, 0 errors | Manual run |
| DJ-022 recorded | LangFuse v3 Docker Compose confirmed | P6-E1-T6 |
| DJ-023 recorded | ContextVar MCP span injection confirmed | P6-E4-T7 |
| DJ-024 recorded | Verification scores logged immediately | P6-E5-T5 |
| DJ-025 recorded | NoOpTracer test isolation confirmed | P6-E2-T13 |
| Dashboard verified | 3 traces visible in LangFuse with 6 scores each | scripts/run_phase6_tracing.py |

---

## Commit Strategy

| Commit | Epic | Key Files |
|---|---|---|
| Phase 6 / E1: LangFuse infrastructure | P6-E1 | docker/langfuse/, doc/setup/LANGFUSE_SETUP.md |
| Phase 6 / E2: Tracing module | P6-E2 | observability/tracing.py, tests/unit/test_tracing_module.py, conftest.py |
| Phase 6 / E3: Agent instrumentation | P6-E3 | fundamental_agent.py, technical_agent.py |
| Phase 6 / E4: MCP span injection | P6-E4 | mcp_client.py |
| Phase 6 / E5: Verification score logging | P6-E5 | tracing.py (log_verification_scores) |
| Phase 6 / E6: Ensemble trace + holistic | P6-E6 | ensemble_runner.py, scripts/run_phase6_tracing.py, tests/holistic/test_phase6_observability_pipeline.py |

---

## Open Questions This Phase Will Answer

**OQ-P6-01: Does the LangFuse CallbackHandler correctly trace the LangGraph node
execution in our agents?**
The agents use a custom LangGraph graph structure (not a standard Chain). The
CallbackHandler's LangGraph support covers `StateGraph` invocations, but the exact
behaviour depends on which callback events LangGraph emits. Phase 6 will verify that the
generated LangFuse trace contains at least one generation record for each agent's LLM call.

**OQ-P6-02: What is the latency overhead of LangFuse tracing on the agent pipeline?**
LangFuse batches and sends spans asynchronously. The overhead is expected to be < 5ms per
trace in the event-emission path. Phase 6 will measure wall-clock time for
`run_ensemble()` with and without `LANGFUSE_ENABLED=true` to confirm this. If the overhead
is material, span transmission can be deferred to a background thread.

**OQ-P6-03: Does the ContextVar approach correctly propagate trace_id through LangGraph
node execution?**
LangGraph executes graph nodes in the same Python thread as the caller (synchronous mode).
Python's `ContextVar` inherits from outer scope in synchronous calls. The expected answer
is yes — the trace_id set by `trace_context()` in `run_analysis()` is visible inside
`call_mcp_tools_node()` when it calls `call_tool()`. Phase 6 will confirm this in the
integration test.

**OQ-P6-04: Are the six verification score names the right ones to track in the dashboard?**
This can only be assessed after seeing the LangFuse dashboard with real data. If the
dashboard suggests additional useful signals (e.g., per-claim grounding rate, parse
retry count), they can be added without breaking changes in Phase 9.

---

## Connections to Earlier and Later Phases

**Depends on Phase 5:**
- `EnsembleVerificationReport`, `AgentVerificationReport` are the sources for score logging
- `verify_ensemble()` must be called before `log_verification_scores()`
- The six score names are derived directly from Phase 5 metric names

**Phase 7 (RAG) depends on Phase 6:**
- The primary hypothesis test for RAG is: does `technical_hr` decrease after RAG is added?
- This comparison requires a Phase 6 score baseline (pre-RAG) and a Phase 7 score run (post-RAG)
- Without Phase 6, the RAG improvement is not measurable in the LangFuse dashboard

**Phase 9 (Collective Decision Engine) depends on Phase 6:**
- Phase 9 extends `run_ensemble()` to N agents. The tracing architecture extends naturally:
  pass `_tracer` to each of N agents, all using the same `trace_id` for their spans
- The `log_verification_scores()` helper will log N agents' metrics as scores

**Phase 10 (Evaluation and Backtesting):**
- Phase 10 can query LangFuse directly for the historical HR/GR/entropy time series
  using the LangFuse SDK's fetch API or the PostgreSQL/ClickHouse backend
- Every decision made from Phase 6 onward is traceable to its inputs without rerunning the pipeline
- The Phase 6 baseline run (`run_phase6_tracing.py`) seeds the time series with the Phase 5
  baseline values, making Phase 10 comparisons start from a known reference point
