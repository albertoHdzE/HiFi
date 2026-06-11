# Phase 6 Scientific Bitacora: Observability — LangFuse

**Date:** 2026-06-11
**Tests:** 582 passing (551 inherited + 31 new), 0 skipped, 0 lint errors
**Commits:** 6 atomic commits (E1–E6)

---

## What Was Built

Phase 6 adds the scientific instrumentation layer to HiFi. Every `run_ensemble()`
call from this phase forward produces a structured, queryable LangFuse trace containing:
- LLM generation records (automatic via CallbackHandler)
- MCP tool spans with call_id linkage (ContextVar injection, no signature changes)
- Six verification scores per trace (HR/GR for each agent, disagreement entropy, contradictions)

**Files created:**
- `docker/langfuse/docker-compose.yml` — LangFuse v3 stack (Postgres 16, ClickHouse 24, Redis 7)
- `docker/langfuse/.env.example` — local credentials template
- `doc/setup/LANGFUSE_SETUP.md` — perennial setup guide
- `src/hifi/observability/tracing.py` — AbstractTracer, NoOpTracer, LangFuseTracer, get_tracer(), trace_context(), SpanContext, log_verification_scores()
- `scripts/run_phase6_tracing.py` — baseline script to seed LangFuse dashboard

**Files modified:**
- `src/hifi/agents/fundamental_agent.py` — tracer parameter, trace_context(), CallbackHandler
- `src/hifi/agents/technical_agent.py` — same pattern
- `src/hifi/agents/mcp_client.py` — ContextVar span injection
- `src/hifi/agents/ensemble_runner.py` — parent trace, verify_ensemble(), score logging
- `tests/conftest.py` — session-scoped disable_langfuse fixture
- `pyproject.toml` — langfuse>=3.0 dependency

---

## Key Decisions Made

**DJ-022 (confirmed):** LangFuse v3 Docker Compose with ClickHouse analytics backend.
The ClickHouse backend is required for Phase 10's historical queries. Starting with v3
is less disruptive than migrating from v2 after data accumulates.

**DJ-023 (confirmed):** Hybrid integration — CallbackHandler for automatic LLM call
tracing + ContextVar for MCP spans. The ContextVar (`_current_trace_id`) is the elegant
solution: it propagates through synchronous call stacks without touching function signatures
or LangGraph state schemas. The MCP client gained tracing awareness with zero interface
changes to `call_tool()`.

**DJ-024 (confirmed):** Six verification scores logged per ensemble trace immediately.
Deferring to Phase 10 would lose the pre-RAG baseline needed to measure RAG improvement.
The Phase 5 baseline run (AAPL/JPM/XOM, HR fundamental=0.000, technical=0.067) is now
the seed point for the LangFuse time series.

**DJ-025 (confirmed):** NoOpTracer + LANGFUSE_ENABLED=false for test isolation. The
session-scoped autouse fixture in conftest.py ensures all 582 tests run with NoOpTracer.
No test requires a live LangFuse server. The NoOpTracer is a real implementation (not a
mock) — the same code path runs in tests and production.

---

## Architectural Surprises and Observations

**SpanContext yield pattern.** The plan's interface specified `Generator[None, None, None]`
for `span()`, but capturing MCP tool output for `span.end(output=...)` required the context
manager to yield a mutable object. The `SpanContext` dataclass solved this cleanly: callers
set `ctx.output` and `ctx.metadata` inside the `with` block; the tracer reads them in the
`finally` clause. This pattern has a precedent in Python's `contextlib` ecosystem and added
zero coupling between the tracing layer and the agent layer.

**Config parameter for graph.invoke().** LangGraph's `graph.invoke(input, config={...})`
accepts a plain dict with a `callbacks` key. When `get_callback_handler()` returns `None`
(NoOpTracer), `config={}` is passed, which is identical to no config. The existing 551 tests
confirmed this — no test modification was required.

**call_id continuity across phases.** The Phase 2 decision to embed `call_id` in every MCP
tool result paid off across three phases: Phase 5 used it for hallucination verification,
Phase 6 uses it to link LangFuse spans to specific tool invocations. This is a satisfying
example of upfront design yielding cumulative compounding value.

**Observability is a side-channel, not a concern.** The cleanest result of this phase is
that NONE of the existing code needed to know about observability in any meaningful way.
The ContextVar approach means `call_tool()` reads a value it didn't set and doesn't need
to understand. The agent functions forward a `tracer` parameter they don't inspect. The
verification layer is unchanged. Instrumentation is genuinely orthogonal to function.

---

## Open Questions Resolved

**OQ-P6-03 (resolved):** ContextVar propagates correctly through LangGraph synchronous
execution. Confirmed by E4 integration tests: `_current_trace_id` set in `trace_context()`
is visible inside `call_mcp_tools_node()` → `call_tool()` with no extra work.

**OQ-P6-01 and OQ-P6-02** remain open until a live LangFuse instance is connected (manual
verification step E6-T8). The CallbackHandler LangGraph integration and latency overhead
will be measured at that point.

---

## Connections Forward

**Phase 7 (RAG):** The primary RAG hypothesis is now measurable. The `technical_hr` score
in LangFuse should decrease after RAG is added. The Phase 6 baseline (0.067) is the
reference point. Every Phase 7 `run_ensemble()` call will add a data point to the time series.

**Phase 9 (N-agent ensemble):** The tracer interface is already multi-agent ready. Pass the
same `tracer` instance to N agents; all add spans to the same ensemble trace via trace_id.
`log_verification_scores()` will generalize to N agents' metrics with one loop change.

**Phase 10 (Backtesting):** The ClickHouse backend in LangFuse v3 supports SQL-style queries
over the trace store. Phase 10 can query `SELECT avg(score_value) WHERE score_name='technical_hr'
GROUP BY week` directly from the analytics backend without re-running the pipeline.
