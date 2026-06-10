# Phase 3 Bitacora: First Agent (Baseline)

**Phase:** 3 -- First Agent (Baseline)
**Status:** COMPLETE
**Dates:** 2026-06-10
**Author:** Alberto Espinosa
**Tests at completion:** 377 passing, 8 skipped (baseline fixture tests await live LLM run); 0 lint errors

---

## Objective

Build the first HiFi agent: a Fundamental Analyst that receives deterministic numbers
from the Phase 2 MCP server and produces a structured investment opinion (Buy/Hold/Sell)
with confidence and rationale. Establish the measurement infrastructure needed to quantify
what one unverified agent can do without ensemble, fine-tuning, RAG, or verification.
The baseline is not expected to be high quality. It is expected to be measurable.

---

## E1: Inference and Orchestration Stack

**Decision DJ-013: LM Studio chosen over Ollama.**

The protocol originally specified Ollama. At the start of Phase 3, LM Studio was already
running at `http://localhost:1234/v1` with five models loaded, including a 32B model. The
OpenAI-compatible API is identical from a client perspective -- `langchain-openai` points
to it with a custom `base_url`. No install, no model downloads, no configuration change.
The decision was immediate. LM Studio serves; MLX tools enter at Phase 11 for fine-tuning.

**Decision DJ-014: qwen2.5-coder-32b-instruct-mlx as the Phase 3 baseline model.**

A structured output test on the live LM Studio instance confirmed the model produces exact
JSON on the first attempt with no extraneous text. At 32B parameters (MLX-native on M3
Ultra) it is significantly stronger than the 7B baseline originally planned -- a deliberate
upgrade that sets a higher floor for Phase 4 to beat. The Phase 4 diversity models (two
Claude Opus 4.6 reasoning-distilled models) are already available in LM Studio.

**Decision DJ-015: LangGraph for agent orchestration.**

LangGraph was specified in the David document and confirmed here. The single-agent graph
in Phase 3 (four sequential nodes) becomes multi-agent in Phase 4 (additional specialized
agents as parallel branches feeding an aggregation node) with no structural change. The
conditional edge pattern used for error handling in Phase 3 (`_should_abort`) is the same
pattern used in Phase 9 for triggering the contrarian agent only when ensemble consensus
is high. These are the right primitives.

**MCP client pattern.** The agent calls the Phase 2 MCP server as a subprocess. The client
(`mcp_client.call_tool`) starts the server via `subprocess.run`, passes the tool call as
JSON over stdin, and reads the response from stdout. This is synchronous and lightweight --
no async complexity, no persistent connection management. One MCP call per subprocess
invocation is the right scope for Phase 3 given the four-tool call count. Phase 9 will
revisit if latency becomes the bottleneck.

**LLM fixture recording decision (vcrpy deferred).** The `openai` client uses `httpx`,
not `requests`. The `responses` library used in Phase 1 cannot intercept httpx. The
correct tool is `vcrpy` with httpx support. However, Phase 3 tests use monkeypatched
LLM stubs instead of recorded cassettes. This is the right decision: the structural
pipeline is fully tested deterministically, and the LLM's content output is not what
Phase 3 tests assert. Phase 5 and Phase 10 will introduce content-level assertions where
vcrpy cassettes become necessary.

---

## E2: Agent Output Schemas

**Separation of concerns at the schema level.** Two types were designed:

- `AgentSignal`: the atomic output unit. Every agent (Fundamental, Technical, Risk, Macro,
  Contrarian) will produce this same schema. The collective decision engine (Phase 9)
  consumes a list of AgentSignal objects, one per agent per ticker per period. Having all
  agents produce the same schema is the mechanism that makes the ensemble generic. An
  ensemble that needed different input types for different agents would be fragile. This
  one does not.

- `FundamentalAnalysis`: the full analysis envelope that wraps AgentSignal with the raw
  MCP tool results. Traceability requires that every number in the rationale can be
  located in a specific tool output. The envelope is the link.

**`call_ids` field rationale.** Each MCP tool call in Phase 2 returns a `call_id` (12-char
SHA-256 prefix of the serialised inputs). The agent is instructed to cite these IDs in its
rationale. `AgentSignal.call_ids` stores the IDs the agent acknowledged using.
Phase 5 (Verification) uses this list to check every numerical claim in the rationale
against the corresponding tool output. This is the Phase 2 call_id field earning its value
three phases later -- the design cost was near zero and the payoff is full numerical
auditability.

**`data_gaps` field rationale.** When a tool returns `None` for a field (e.g.,
`revenue_growth_yoy` is always None in Phase 2 because yfinance provides only a single
snapshot), the agent has no right to say anything about that field. If it does, it has
hallucinated. The `data_gaps` list records which fields were None. The hallucination
checker uses this list: if a gap field appears in the rationale without an acknowledgment
phrase ("unavailable", "insufficient", etc.), it is flagged. This is not Phase 5 rigor --
it is Phase 3's best approximation.

---

## E3: Base Agent Infrastructure

**LangGraph graph design.** The four-node graph is:

```
load_snapshot -> call_mcp_tools -> generate_analysis -> parse_output -> END
```

`load_snapshot_node` is a validation gate. It deserialises the snapshot JSON and verifies
it parses as a `FundamentalsSnapshot`. If not, an error is set in state and a conditional
edge (`_should_abort`) routes to END, skipping all LLM calls. This prevents silent
failures: a bad snapshot produces an explicit error in the `FundamentalAnalysis.signal`
field (None) rather than a confused LLM output. Explicit failures are measurable.

`call_mcp_tools_node` calls four tools: `get_financial_ratios`, `get_growth_metrics`,
`get_valuation_context`, `get_macro_snapshot`. Each call is wrapped in a try/except that
catches subprocess failures and stores `{"error": "COMPUTATION_ERROR"}` in the result
dict. The LLM receives this error dict as context, which is the correct behaviour -- the
agent should know when a tool failed, not receive a fabricated value.

`generate_analysis_node` fills the `fundamental_v1` prompt template and calls the LLM.
The state carries the raw response string. No parsing happens here -- parsing is
deliberately separated so that the parse-and-retry logic in `parse_output_node` is clean.

`parse_output_node` implements the parse-and-retry pattern. First attempt: parse the LLM
response as JSON, validate as AgentSignal. If this fails, send a correction message and
attempt once more. If both fail, set `error` in state. Two observations:
1. The retry message needs to be explicit: not "try again" but "produce ONLY the JSON
   object with fields: decision, confidence, rationale, key_concern."
2. The second attempt failure is set as an error in state, not silently swallowed. The
   baseline metrics capture this failure rate. It is data, not noise.

**Prompt versioning.** Prompt files live in `src/hifi/agents/prompts/` as versioned
Markdown files. The version identifier (`fundamental_v1`) is embedded in
`FundamentalAnalysis.prompt_version`. When the prompt changes (Phase 10 experimentation,
Phase 11 fine-tuning data curation), the version increments and every output produced with
the new prompt is distinguishable from outputs produced with the old one. This is
observability for prompt engineering.

---

## E4: Fundamental Analyst Agent -- Integration

**run_analysis() entrypoint.** This is the only public function callers need. It:
1. Builds and compiles the LangGraph graph (on every call -- compilation is cheap)
2. Constructs the initial state from ticker, date, snapshot JSON, and data directory
3. Invokes the graph
4. Assembles `FundamentalAnalysis` from the final state

On parse failure (both attempts exhausted), `signal` is None. The `FundamentalAnalysis`
is returned anyway with the raw tool results intact. This is important: a failed run still
carries the MCP tool results, which means the baseline metrics can assess whether tool
calls succeeded even when the LLM parsing failed. The failure is granular.

**Information restriction.** The Fundamental Agent has access only to balance sheet and
income statement ratios, valuation context, and macro background. It does NOT receive
OHLCV data, technical indicators, or sentiment. This is a deliberate architectural choice,
not a limitation. Genuine ensemble diversity (Phase 4) requires genuine information
diversity. An agent that receives everything is not diverse -- it is noisy. The
restrictions are enforced by which MCP tools are called in `call_mcp_tools_node`.

**Hallucination checker (Phase 3 approximation).** `baseline_metrics.count_hallucinated_numbers`
extracts all numeric values from the rationale, then checks each one against the flat
union of all four tool result dicts (within 1% relative tolerance). Numbers that do not
appear in any tool result are flagged as hallucination candidates. Small integers (|n| <= 3)
are excluded because "the company has 2 segments" is not a financial hallucination.

This is deliberately a Phase 3 approximation. Phase 5 (Verification) will do this
rigorously: check specific claims against specific tool calls using the call_ids. The
Phase 3 checker is a lower bound on the hallucination rate. It will report false negatives
(a hallucinated number that happens to match an unrelated tool result value) and may
report false positives (a ratio the LLM computed correctly but rounded differently). Both
error types are documented. What matters at Phase 3 is having a number at all.

---

## E5: Baseline Evaluation Fixtures and Metrics

**Fixture approach.** The baseline runner (`scripts/run_phase3_baseline.py`) produces
`tests/fixtures/baseline/phase3_baseline.json`. This file is the ground state: the floor
against which Phase 4 improvements are measured. The tests in `test_phase3_baseline.py`
validate the structure and minimum quality threshold (compliance_rate >= 0.90) but are
skipped when the fixture does not exist. The fixture is generated once, committed, and
thereafter serves as a regression guard.

Reference snapshots for AAPL, JPM, and XOM are hardcoded in the script using approximate
Q1 2023 values from public 10-Q filings. This is the right scope: the baseline measures
LLM interpretation quality, not data acquisition quality. The distinction is deliberate.

**Why 8 tests are skipped.** The baseline fixture tests require the output of a live LLM
run. Tests that would always fail when the fixture is absent are worse than skipped tests --
they create noise that obscures real failures. The skip condition is transparent: the
test output says "phase3_baseline.json not generated yet -- run scripts/run_phase3_baseline.py".
This is the correct ergonomics.

---

## E6: Holistic Pipeline Test

The holistic test (`tests/holistic/test_phase3_agent_pipeline.py`) validates the full
pipeline end-to-end using a monkeypatched LLM and real Phase 1 Parquet fixtures.
Six assertions:
1. `call_mcp_tools_node` returns results for all four tools
2. All tool results that did not error include a `call_id`
3. `parse_output_node` extracts a valid AgentSignal from a pre-set valid JSON response
4. `run_analysis` produces a structurally valid `FundamentalAnalysis` (decision, confidence, ticker)
5. The output is JSON-safe (no NaN in any field)
6. The Phase 2 holistic test still passes (regression guard)

The holistic test deliberately does not assert the content of decisions or rationale.
Content quality is a Phase 10 concern. Phase 3 asserts structure and auditability.
These are different and must be kept separate.

---

## Surprises and Insights

**FundamentalsSnapshot field mismatch.** Test code written during Phase 3 used
`as_of_date` as a field name when the actual schema field is `period_end`. The schema
also requires `source` and `fetched_at` at the top level (not only in the nested
`ProvenanceRecord`). These errors were not caught at write time because the tests that
exercised the snapshot construction were not run until the full suite was assembled.
The fix was trivial but revealed a pattern: when a schema has both a top-level field
and a nested record with the same semantic (fetched_at appears in both `FundamentalsSnapshot`
and `ProvenanceRecord`), test authors will conflate them. A doc comment on the schema
field clarifying "this top-level field is for direct queries; provenance is for audit
trail" would have prevented the confusion.

**data_gap_acknowledged logic requires field name in rationale.** The initial test
rationale was "Revenue growth is unavailable for this snapshot." This sounds correct but
fails the check: `data_gap_acknowledged` looks for the exact field name `revenue_growth_yoy`
in the rationale, not a natural-language description of it. The rationale needed to say
"revenue_growth_yoy is unavailable" not "Revenue growth is unavailable". This is a
fundamental design question: should the agent be required to cite field names verbatim,
or should the checker perform semantic matching? Phase 3 takes the strict position
(field name verbatim) because it is simpler to test and audit. Phase 5 can introduce
semantic matching if verbatim citation proves too brittle in practice.

**MCP transport is synchronous subprocess per call.** Each of the four MCP tool calls
starts a fresh subprocess. At Phase 3 scale (four tools, one ticker) this is acceptable
at ~200-300ms overhead per call. At Phase 8 scale (many tools, 10+ tickers, concurrent
agents), subprocess-per-call will become the latency bottleneck. The Phase 9 design
should include a persistent MCP server process per agent session. This is noted as a
forward dependency, not a Phase 3 concern.

---

## Open Questions Resolved

**OQ-P3-01 (10 vs 3 tickers):** Resolved as DEFERRED. Phase 3 baseline covers AAPL,
JPM, XOM. The 10-ticker expansion is a Phase 8 task when the full data universe is
established.

**OQ-P3-02 (Qwen vs Llama):** Resolved as MOOT. The Phase 3 baseline uses
qwen2.5-coder-32b-instruct-mlx (32B), which is substantially more capable than the
originally planned 7B models. The comparison between 7B models is not informative at
this scale.

**OQ-P3-03 (failure mode taxonomy):** Documented above (data_gap_acknowledged logic,
field-name verbatim vs semantic). Full failure mode analysis requires the actual baseline
run. Categories: (a) JSON parse failure, (b) hallucinated numbers, (c) ignored None
fields, (d) incoherent rationale. Category (c) is the one already revealed by test design.

---

## Connections Forward

**Phase 4** adds a second agent (Claude-distilled model from LM Studio). The LangGraph
graph in Phase 3 becomes the foundation: add parallel agent branches, add an aggregation
node. The AgentSignal schema is already the uniform interface. Phase 3's baseline metrics
are the target to beat.

**Phase 5** (Verification) will use `call_ids` from every AgentSignal to check numerical
claims against specific MCP tool outputs. The Phase 3 simple hallucination checker is the
proof-of-concept. Phase 5 replaces it with a verifier that is itself an MCP tool,
returning structured verification reports.

**Phase 10** (Evaluation) uses `phase3_baseline.json` as the floor for all improvement
measurements. The metric schema established in `baseline_metrics.py` (compliance_rate,
hallucinated_numbers, data_gaps_acknowledged, mean_call_id_coverage, mean_latency_ms)
becomes the standard format for evaluation reports across all phases.

**Phase 11** (Fine-Tuning) uses `fundamental_v1.md` as the training document template.
The prompt version in `FundamentalAnalysis.prompt_version` makes Phase 3 outputs
distinguishable from outputs produced with later, fine-tuned prompts. This
distinguishability is necessary for curating training data.
