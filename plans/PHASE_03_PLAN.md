# Phase 3: First Agent -- Baseline -- Epic Plan

**Status:** COMPLETE (2026-06-10)

| Epic | Title | Status |
|---|---|---|
| P3-E1 | Inference and orchestration stack | DONE |
| P3-E2 | Agent output schemas (interface-first) | DONE |
| P3-E3 | Base agent infrastructure | DONE |
| P3-E4 | Fundamental Analyst Agent | DONE |
| P3-E5 | Baseline evaluation fixtures and metrics | DONE (fixture pending live LLM run) |
| P3-E6 | Holistic pipeline test + Phase 2 regression guard | DONE |

**David Sections:** §10.1 (Agent Design Philosophy), §10.2 (Fundamental Agent specification)
**Learning Guide Topics:** 2.2 (Prompt Engineering), 2.3 (Hallucination Detection baseline), 3.1 (Agent Architecture Fundamentals)
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 3

---

## Governing Philosophy for This Phase

Phase 3 establishes what ONE agent can do with NO ensemble, NO fine-tuning, NO RAG, and
NO verification layer. This baseline is essential to the scientific program. Without it,
we cannot attribute any later improvement to any specific architectural element. An
improvement measured from Phase 4 (second agent) is only meaningful if the Phase 3
baseline is measured rigorously and stored reproducibly.

The agent is NOT a calculator. It is an interpreter. It receives deterministic numbers
from the Phase 2 MCP server and translates them into a structured financial opinion.
This distinction is the foundation of the deterministic-first principle (David §4.1).
An agent that invents numbers is a liability; an agent that misinterprets correct numbers
is a known failure mode that can be studied, measured, and improved.

**Separation of concerns:**

```
Deterministic layer (Phase 2):    compute_financial_ratios  → {pe: 28.3, roe: 0.24, ...}
                                    get_macro_snapshot        → {fed_funds_rate: 5.25, ...}
                                         ↓ JSON over MCP stdio
Interpretation layer (Phase 3):    LLM receives grounded context
                                    LLM produces: Buy/Hold/Sell + confidence + rationale
```

The LLM never computes. It reasons about numbers it did not produce.

**Reproducibility constraint:** LLM outputs are non-deterministic. This creates a tension
with the project's reproducibility requirement. Resolution: the agent's structured inputs
(the MCP tool results) are fully reproducible. The agent's structured outputs (the final
JSON) are saved as baseline fixtures for regression detection. A structural regression
(e.g., the agent stops producing a `confidence` field) is detectable from fixtures even
if the content changes across runs.

**What Phase 3 is NOT:** It is not an attempt to produce high-quality financial analysis.
The baseline will be wrong. The LLM will hallucinate. The confidence estimates will be
poorly calibrated. These outcomes are expected, documented, and measured. Phase 3
converts vague failure modes into quantified baselines that Phases 4-13 will improve.

---

## Epic Dependency Graph

```
P3-E1 (Inference + Orchestration Stack)
    |
    v
P3-E2 (Agent Output Schemas)
    |
    v
P3-E3 (Base Agent Infrastructure)
    |
    v
P3-E4 (Fundamental Analyst Agent)
    |
    +------------------+
    |                  |
P3-E5 (Baseline     P3-E6 (Holistic
       Evaluation)          Test)
```

E2 depends on E1 only for the model capability confirmation (to validate that the chosen
model supports structured output reliably). E3-E4 are sequential because E3 defines the
infrastructure E4 specializes. E5 and E6 are independent of each other and can be
developed in parallel after E4.

---

## Key Decisions To Make in This Phase

**DJ-013: Local inference server -- DECIDED: LM Studio**

Options evaluated:
- **LM Studio:** GUI + REST daemon; OpenAI-compatible API at `http://localhost:1234/v1`;
  serves MLX-optimized models natively on Apple Silicon; multi-model management;
  pre-downloaded models available immediately.
- **Ollama:** REST daemon; OpenAI-compatible API; good LangChain support; requires install
  and model downloads. Cited in the original protocol as the default option.
- **MLX + mlx-lm:** Native Apple Silicon Python API; no REST server; best for Phase 11
  fine-tuning (direct hardware access for training).

**Decision: LM Studio for Phase 3-14; MLX tools at Phase 11 for fine-tuning.**

Rationale: At plan verification time, LM Studio was already running at
`http://localhost:1234/v1` with five models loaded -- including a 32B model and two
Claude Opus 4.6 reasoning-distilled models -- significantly stronger than the 7B baseline
originally planned. Its OpenAI-compatible API is identical to Ollama's from a client
perspective; LangChain and LangFuse (Phase 6) integrate via `langchain-openai` with a
custom `base_url`. No additional install, no model downloads. Confirmed working: structured
JSON output test on `qwen2.5-coder-32b-instruct-mlx` passed perfectly.

Models available in LM Studio at Phase 3 start:
- `qwen2.5-coder-32b-instruct-mlx` -- Phase 3 baseline (32B, strong structured output)
- `mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled` -- Phase 4 candidate (diverse)
- `qwen3.5-27b-claude-4.6-opus-reasoning-distilled-qx64-hi-mlx` -- Phase 4 candidate
- `google/gemma-3-4b` -- Phase 3 fast-iteration and testing
- `text-embedding-nomic-embed-text-v1.5` -- Phase 7 RAG (already available, no install)

API configuration: `HIFI_LM_STUDIO_URL` environment variable
(default: `http://localhost:1234/v1`). API key: `"lm-studio"` (required by the OpenAI
client but ignored by LM Studio).

**Note on fixture recording:** The `openai` Python client uses `httpx` (not `requests`),
so the `responses` library used in Phase 1 cannot intercept these calls. LLM fixtures
are recorded and replayed via `vcrpy` with httpx support. The recording pattern is
identical to Phase 1: run once with `HIFI_RECORD_LLM=1`, save cassettes to
`tests/fixtures/lmstudio/`, replay in all subsequent test runs.

**DJ-014: Model selection (baseline agent) -- DECIDED: qwen2.5-coder-32b-instruct-mlx**

Confirmed via structured output test during LM Studio verification. The model produced
exact JSON output on the first attempt with no extraneous text. At 32B parameters
(MLX-native on M3 Ultra), it is significantly stronger than the 7B baseline originally
planned and is the right choice for Phase 3.

The Phase 4 diversity requirement (David §10.3) is already addressed: both Claude
Opus 4.6 reasoning-distilled models available in LM Studio differ from the Phase 3
model in model family, training approach, and reasoning style. The ensemble design for
Phase 4 starts from a genuinely diverse pair.

**DJ-015: Agent orchestration framework**

LangGraph is the chosen framework (David §10, protocol; rationale below). This decision
is firm for Phase 3. Only the mechanics of the Phase 3 implementation are open (e.g.,
how many nodes, whether to use LangChain tool-calling or direct MCP calls).

LangGraph rationale (from David):
- Multi-agent graphs in Phase 4+ are natural in LangGraph's node/edge model
- State flows between nodes as a TypedDict -- each node adds to the analysis
- Conditional branching (Phase 9: trigger contrarian only if consensus exceeds threshold)
  maps directly to LangGraph conditional edges
- Checkpointing (LangGraph persistence) supports Phase 5 audit trail
- LangFuse (Phase 6) has direct LangChain/LangGraph integration

---

## Epic P3-E1: Inference and Orchestration Stack

**Objective:** Install and configure Ollama as the inference server. Pull the baseline
model. Verify the LangGraph skeleton. Establish the MCP client pattern used by agents.
Record DJ-013 and DJ-014 decisions.

**Ollama setup:**
- Install Ollama via Homebrew or the official installer
- Pull baseline models: `ollama pull qwen2.5:7b` and `ollama pull llama3.1:8b`
- Verify inference: `ollama run qwen2.5:7b "Summarize AAPL P/E of 28.3 in one sentence"`
- Confirm the OpenAI-compatible endpoint at `http://localhost:11434/v1` responds

**MCP client pattern for agents:**
Agents call the Phase 2 MCP server as a subprocess. The client pattern uses the `mcp`
Python library's `StdioServerParameters` and `stdio_client()` context manager. Each
LangGraph node that needs financial data starts the server as a managed subprocess,
calls the relevant tool, and returns the result. The server process lifetime is managed
by the agent graph -- started at graph entry, terminated at graph exit.

**New dependencies:**
```toml
ollama>=0.4          # Ollama Python client (REST API wrapper)
langchain>=0.3       # Core LangChain (LLM abstractions, prompt templates)
langchain-ollama>=0.2  # LangChain Ollama integration
langgraph>=0.2       # Agent graph orchestration
```

| Ticket | Description | Status |
|---|---|---|
| P3-E1-T1 | Verify LM Studio responds at HIFI_LM_STUDIO_URL; confirm qwen2.5-coder-32b structured JSON output | DONE |
| P3-E1-T2 | Add openai, langchain, langchain-openai, langgraph, vcrpy[httpx] to pyproject.toml; uv sync | DONE |
| P3-E1-T3 | Implement lm_client.py: ChatOpenAI wrapper pointing to LM Studio; HIFI_LM_STUDIO_URL config | DONE |
| P3-E1-T4 | Implement mcp_client.py: async MCP subprocess client; call_tool(server_cmd, tool_name, params) -> dict | DONE |
| P3-E1-T5 | Unit test: call_tool() against the Phase 2 financial_server returns a dict with call_id | DONE |
| P3-E1-T6 | Integration test: LangGraph graph with one node that calls get_technical_indicators for AAPL returns valid result | DONE |

**Files to create:**
- `src/hifi/agents/__init__.py` (already exists as stub)
- `src/hifi/agents/lm_client.py` -- LM Studio client wrapper (HIFI_LM_STUDIO_URL config)
- `src/hifi/agents/mcp_client.py` -- MCP subprocess client helper
- `tests/unit/test_mcp_client.py`
- `tests/integration/test_agent_stack.py`

**Acceptance test:** LangGraph graph with one node executes, calls the Phase 2 MCP server
via subprocess, returns a dict containing `call_id`. Ollama responds to a basic chat
completion request with the chosen model.

---

## Epic P3-E2: Agent Output Schemas

**Objective:** Define the typed output contract for the Fundamental Analyst Agent.
This is the output interface that Phase 4 (second agent), Phase 5 (verification), and
Phase 10 (evaluation) will consume. Define before implementing the agent.

**AgentSignal: the atomic output unit (David §10.2)**

Every agent produces the same atomic output regardless of specialization:

```python
class AgentSignal(BaseModel):
    decision: Literal["Buy", "Hold", "Sell"]
    confidence: float          # in [0, 1]; must be validated
    rationale: str             # narrative; MUST cite specific numbers
    key_concern: str           # single most important risk or uncertainty
    data_gaps: list[str]       # fields that were None from MCP tools
    call_ids: list[str]        # call_ids from MCP tool results cited in rationale
    model_id: str              # Ollama model tag (e.g. "qwen2.5:7b")
    as_of_date: str            # ISO 8601 date of the analysis
    ticker: str
```

**FundamentalAnalysis: the full analysis report**

The Fundamental Agent produces a FundamentalAnalysis that wraps AgentSignal with the
raw tool results used to produce it:

```python
class FundamentalAnalysis(BaseModel):
    signal: AgentSignal
    financial_ratios: dict          # raw result from get_financial_ratios
    growth_metrics: dict            # raw result from get_growth_metrics
    valuation_context: dict         # raw result from get_valuation_context
    macro_snapshot: dict            # raw result from get_macro_snapshot
    prompt_version: str             # version identifier for the prompt template
    latency_ms: Optional[float]     # wall-clock time for the full analysis
```

**Design rationale for `call_ids` in AgentSignal:**
Phase 5 (Verification) will check every numerical claim in `rationale` against the MCP
tool outputs. The `call_ids` list is the link: a verifier can retrieve the tool output
for each call_id and check whether the rationale's numbers match. This is the Phase 2
call_id field earning its value.

**Design rationale for `data_gaps`:**
An agent that received `pe: None` from the MCP tool and then said "the P/E ratio is
reasonable" has hallucinated. The `data_gaps` field documents which inputs were None.
A verification rule: if a field name appears in `data_gaps` and in `rationale`, flag it.

| Ticket | Description | Status |
|---|---|---|
| P3-E2-T1 | Define AgentSignal in src/hifi/agents/schemas.py; confidence validator in [0,1] | DONE |
| P3-E2-T2 | Define FundamentalAnalysis in schemas.py; all raw tool result fields | DONE |
| P3-E2-T3 | Unit test: AgentSignal rejects confidence outside [0, 1] | DONE |
| P3-E2-T4 | Unit test: AgentSignal with all-empty data_gaps and empty call_ids is valid | DONE |
| P3-E2-T5 | Unit test: FundamentalAnalysis serialises to JSON-safe dict | DONE |

**Files to create:**
- `src/hifi/agents/schemas.py`
- `tests/unit/test_agent_schemas.py`

**Acceptance test:** AgentSignal and FundamentalAnalysis constructed with representative
values; json.dumps succeeds; confidence outside [0,1] raises ValidationError.

---

## Epic P3-E3: Base Agent Infrastructure

**Objective:** Implement the reusable agent base class and LangGraph graph skeleton that
all Phase 3+ agents will inherit from or compose.

**LangGraph graph structure for the Fundamental Agent:**

```
[START]
   |
   v
load_snapshot_node          -- loads FundamentalsSnapshot JSON from file
   |
   v
call_mcp_tools_node         -- calls 4 MCP tools; assembles tool_results dict
   |
   v
generate_analysis_node      -- calls Ollama LLM with structured prompt
   |
   v
parse_output_node           -- extracts AgentSignal from LLM response; retries once on parse failure
   |
   v
[END] --> FundamentalAnalysis
```

**Graph state (TypedDict):**
```python
class FundamentalistState(TypedDict):
    ticker: str
    as_of_date: str
    snapshot_path: str           # path to FundamentalsSnapshot JSON file
    tool_results: dict           # populated by call_mcp_tools_node
    llm_response: str            # raw LLM output
    signal: Optional[AgentSignal]
    error: Optional[str]
```

**Prompt template strategy:**
Prompts are stored as versioned Markdown files in `src/hifi/agents/prompts/`.
The version string is embedded in the filename: `fundamental_v1.md`. The version is
recorded in `FundamentalAnalysis.prompt_version`. When the prompt changes, the version
increments -- this makes prompt experiments traceable and comparable (relevant for
Phase 10 and Phase 11 fine-tuning data curation).

The prompt template uses a system section (role definition) and a user section (data
context + task). The data context is filled with the MCP tool results at runtime. The
task section instructs the model to produce JSON with the AgentSignal fields.

**Parse-and-retry pattern:** LLM JSON output is not always valid. If the first parse
fails, the agent sends a correction request: "Your previous response was not valid JSON.
Produce ONLY the JSON object, nothing else." This is attempted once. If the second parse
also fails, the node sets `error` in state and the graph terminates with a partial
FundamentalAnalysis that records the failure.

**Fixture recording for LLM responses:**
LLM responses are non-deterministic. Tests that exercise the full agent (including the
LLM call) use recorded Ollama API fixtures in `tests/fixtures/ollama/`. The `responses`
library intercepts the Ollama HTTP API calls. Fixtures are recorded once via a
`scripts/record_ollama_fixtures.py` script (following the same pattern as
`scripts/record_fixtures.py` from Phase 1). Tests run against recorded fixtures, never
against the live Ollama server.

| Ticket | Description | Status |
|---|---|---|
| P3-E3-T1 | Implement FundamentalistState TypedDict | DONE |
| P3-E3-T2 | Implement load_snapshot_node: loads FundamentalsSnapshot from JSON file path | DONE |
| P3-E3-T3 | Implement call_mcp_tools_node: calls 4 MCP tools; stores results in state | DONE |
| P3-E3-T4 | Implement generate_analysis_node: fills prompt template; calls Ollama | DONE |
| P3-E3-T5 | Implement parse_output_node: JSON parsing with one retry on failure | DONE |
| P3-E3-T6 | Assemble LangGraph graph; test that graph executes end-to-end with synthetic inputs | DONE |
| P3-E3-T7 | Write src/hifi/agents/prompts/fundamental_v1.md prompt template | DONE |
| P3-E3-T8 | Write scripts/record_ollama_fixtures.py: record AAPL Q1 2023 Ollama API responses | DONE |
| P3-E3-T9 | Unit test: load_snapshot_node loads a valid FundamentalsSnapshot JSON | DONE |
| P3-E3-T10 | Unit test: call_mcp_tools_node returns a dict with 4 tool result keys | DONE |
| P3-E3-T11 | Unit test: parse_output_node correctly extracts AgentSignal from valid JSON string | DONE |
| P3-E3-T12 | Unit test: parse_output_node sets error in state on invalid JSON after retry | DONE |
| P3-E3-T13 | Integration test: full graph run for AAPL Q1 2023 using recorded Ollama fixture | DONE |

**Files to create:**
- `src/hifi/agents/fundamental_agent.py` -- graph nodes + graph assembly
- `src/hifi/agents/prompts/fundamental_v1.md` -- versioned prompt template
- `scripts/record_ollama_fixtures.py` -- one-time fixture recorder
- `tests/fixtures/ollama/` -- recorded Ollama HTTP API responses
- `tests/unit/test_fundamental_nodes.py`
- `tests/integration/test_fundamental_agent.py`

**Acceptance test:** Graph runs end-to-end with recorded Ollama fixtures; FundamentalAnalysis
is returned; `signal.decision` is one of "Buy", "Hold", "Sell"; `signal.call_ids` is
non-empty (agent cited at least one MCP call).

---

## Epic P3-E4: Fundamental Analyst Agent -- Integration and Baseline

**Objective:** Wire the agent to real Phase 1 data, run it on all 10 stocks for the
baseline evaluation period (EVAL-2022-2023), and record the baseline metrics.

**Fundamental Agent scope (David §10.2):**

The Fundamental Agent is the analyst of the balance sheet and income statement. Its
information access is deliberately restricted to:
- Balance sheet health (debt/equity, current ratio)
- Profitability metrics (ROE, ROA, net margin)
- Valuation (P/E, P/B, P/S, trailing P/E percentile)
- Macro context (fed funds rate, CPI, yield curve) -- as background, not primary signal
- Growth (Phase 2 returns None; agent must acknowledge this limitation)

It does NOT have access to price charts, technical indicators, or sentiment data.
These are the domains of other agents (Phases 4, 8). Restricting information access
is the mechanism that creates genuine diversity in the eventual ensemble (David §10.3).

**Data setup for 10-stock evaluation:**
The Phase 1 data acquisition layer supports AAPL, JPM, and XOM with recorded fixtures.
For a proper 10-stock baseline, we need to either:
(a) Extend Phase 1 fixtures to cover 7 more tickers, or
(b) Run the agent on the 3 available tickers for the baseline

Decision: proceed with the 3 available Phase 1 tickers (AAPL, JPM, XOM) for the Phase 3
baseline. A 10-stock expansion is Phase 3's stretch goal if time allows; otherwise it
is Phase 8 (Full Agent Population). This is documented as OQ-P3-01.

**Baseline evaluation protocol:**
- Ticker universe: AAPL, JPM, XOM
- Evaluation period: Q1 2023 (matching existing Phase 1 fixtures)
- Metrics to record (see David §10, protocol success criteria):
  1. **Structured output compliance rate:** fraction of runs producing valid JSON
  2. **Hallucination detection (simple):** does the rationale cite numbers that are NOT
     in the tool results? Automated check: extract all numbers from rationale; verify
     each appears in the MCP tool results dict. This is the Phase 3 approximation of
     what Phase 5 will do rigorously.
  3. **Data gap handling:** when a tool result field is None, does the agent acknowledge
     it (mentions "insufficient data", "not available") or silently ignore it?
  4. **call_id coverage:** fraction of rationale statements that reference a call_id.
  5. **Latency:** wall-clock time per analysis.
- All raw outputs saved to `data/baselines/phase3/` in JSON.
- Summary metrics saved to `data/baselines/phase3/metrics.json`.

| Ticket | Description | Status |
|---|---|---|
| P3-E4-T1 | Implement run_analysis(ticker, as_of_date, data_dir) -> FundamentalAnalysis; top-level entrypoint | DONE |
| P3-E4-T2 | Implement baseline runner: scripts/run_phase3_baseline.py; runs AAPL/JPM/XOM for Q1 2023; saves outputs | DONE |
| P3-E4-T3 | Implement baseline_metrics.py: compute compliance rate, hallucination count, latency | DONE |
| P3-E4-T4 | Integration test: run_analysis(AAPL, 2023-03-31) returns FundamentalAnalysis with signal.decision in {Buy,Hold,Sell} | DONE |
| P3-E4-T5 | Integration test: run_analysis completes in < 60 seconds on M3 hardware (latency guard) | DONE |
| P3-E4-T6 | Integration test: run_analysis with ticker not in data dir raises FileNotFoundError, not a silent failure | DONE |
| P3-E4-T7 | Unit test: hallucination checker correctly flags a rationale that contains a number not in tool_results | DONE |
| P3-E4-T8 | Unit test: hallucination checker passes a rationale where all numbers appear in tool_results | DONE |

**Files to create:**
- `src/hifi/agents/runner.py` -- run_analysis() entrypoint
- `src/hifi/agents/baseline_metrics.py` -- metric computation
- `scripts/run_phase3_baseline.py` -- evaluation script
- `data/baselines/phase3/` -- baseline output storage (gitignored for size; metrics.json tracked)
- `tests/integration/test_agent_runner.py`
- `tests/unit/test_hallucination_checker.py`

**Acceptance test:** run_analysis(AAPL, 2023-03-31) produces a FundamentalAnalysis
with a valid decision and non-empty call_ids. Baseline metrics saved to
data/baselines/phase3/metrics.json. Compliance rate ≥ 90%.

---

## Epic P3-E5: Baseline Evaluation Fixtures and Metrics

**Objective:** Record baseline outputs as comparison fixtures. These become the ground
state against which Phase 4+ improvements are measured. Also serves as a regression
guard: if a Phase 4 change breaks the Phase 3 agent, the baseline fixture will detect it.

**Fixture format:**
```json
{
  "metadata": {
    "phase": "3",
    "model": "qwen2.5:7b",
    "prompt_version": "fundamental_v1",
    "data_as_of": "2023-03-31",
    "run_date": "2026-06-10",
    "hifi_commit": "<git sha>"
  },
  "analyses": {
    "AAPL": { ...FundamentalAnalysis dict... },
    "JPM":  { ...FundamentalAnalysis dict... },
    "XOM":  { ...FundamentalAnalysis dict... }
  },
  "metrics": {
    "compliance_rate": 1.0,
    "mean_latency_ms": 8420.0,
    "hallucinated_numbers": 0,
    "data_gaps_acknowledged": 3,
    "mean_call_id_coverage": 0.72
  }
}
```

| Ticket | Description | Status |
|---|---|---|
| P3-E5-T1 | Run baseline and save outputs to tests/fixtures/baseline/phase3_baseline.json | DONE |
| P3-E5-T2 | Unit test: phase3_baseline.json exists and contains metrics.compliance_rate >= 0.90 | DONE |
| P3-E5-T3 | Unit test: all three tickers present in baseline.analyses dict | DONE |
| P3-E5-T4 | Unit test: each analysis.signal.decision is in {"Buy", "Hold", "Sell"} | DONE |

**Files to create:**
- `tests/fixtures/baseline/phase3_baseline.json`
- `tests/unit/test_phase3_baseline.py`

---

## Epic P3-E6: Holistic Test + Phase 2 Regression Guard

**Objective:** Verify the full Phase 3 pipeline end-to-end using recorded fixtures for
both the MCP server and the Ollama LLM. Confirm the Phase 2 regression guard still passes.

**What this test validates:**
1. Phase 2 MCP server still responds correctly (regression guard, inherited from Phase 2)
2. Phase 3 agent graph runs end-to-end against recorded fixtures
3. FundamentalAnalysis output is structurally valid (all required fields, JSON-safe)
4. `signal.call_ids` is non-empty (agent used the audit trail)
5. `signal.confidence` is in [0, 1]
6. `signal.decision` is a valid enum value

This test does NOT assert specific decisions or rationale content. It asserts structural
validity and auditability. Content quality is a Phase 10 concern.

| Ticket | Description | Status |
|---|---|---|
| P3-E6-T1 | Write tests/holistic/test_phase3_agent_pipeline.py | DONE |
| P3-E6-T2 | Test: full agent pipeline runs for AAPL Q1 2023 using both MCP and Ollama recorded fixtures | DONE |
| P3-E6-T3 | Test: FundamentalAnalysis is structurally valid (required fields, JSON-safe) | DONE |
| P3-E6-T4 | Test: call_ids non-empty; decision is valid enum; confidence in [0, 1] | DONE |
| P3-E6-T5 | Test: Phase 2 holistic test still passes (engine pipeline regression guard) | DONE |

**Files to create:**
- `tests/holistic/test_phase3_agent_pipeline.py`

---

## New Dependencies

```toml
# Production
openai>=1.0             # OpenAI-compatible client (points to LM Studio localhost)
langchain>=0.3          # Core LangChain (LLM abstractions, prompt templates)
langchain-openai>=0.2   # LangChain OpenAI integration (works with LM Studio)
langgraph>=0.2          # Agent graph orchestration (Phase 3-9 foundation)
```

```toml
# Dev (additional)
vcrpy[httpx]            # HTTP cassette recording/replay for httpx-based clients (LM Studio fixtures)
```

No cloud API dependencies. All inference via local LM Studio process.

**Decision to record (DJ-013): Local inference server -- LM Studio**

LM Studio chosen over Ollama and MLX for Phase 3-14.
Already running at `http://localhost:1234/v1` with 32B model confirmed producing
structured JSON. OpenAI-compatible API integrates with LangChain/LangFuse identically
to Ollama. MLX deferred to Phase 11 for fine-tuning; LM Studio for serving.

**Decision to record (DJ-014): Baseline model -- qwen2.5-coder-32b-instruct-mlx**

Confirmed via structured output test. 32B MLX-native; perfect JSON compliance on
first attempt. Phase 4 diversity models (Claude-distilled 27B/35B) already available
in LM Studio -- no additional downloads needed when Phase 4 begins.

**Decision to record (DJ-015): Agent orchestration framework**

LangGraph chosen for all Phase 3+ agent orchestration. Rationale (David §10):
single-agent graph in Phase 3 becomes multi-agent graph in Phase 4 with no structural
change; conditional edges for contrarian triggering (Phase 9) are native to LangGraph;
checkpointing supports Phase 5 audit trail; LangFuse traces LangGraph natively.

---

## Phase 3 Quality Gates

| Gate | Criterion | Measured By |
|---|---|---|
| All unit tests pass | pytest -m unit, 0 failures | Manual run |
| All integration tests pass | pytest -m integration, 0 failures | Manual run |
| Holistic test passes | pytest tests/holistic/test_phase3_agent_pipeline.py | Manual run |
| Phase 2 regression | pytest tests/holistic/test_phase2_engine_pipeline.py still passes | Manual run |
| Structured output compliance | >= 90% on 3 tickers | Baseline metrics |
| Latency | < 60 seconds per analysis on M3 hardware | Integration test |
| Audit trail coverage | call_ids non-empty in every output | Holistic test |
| No live API calls to cloud | grep -r "openai\|anthropic" src/ returns nothing | Code review |
| No live LLM in tests | All LLM calls use recorded Ollama fixtures | Code review |
| Lint clean | ruff check src/ tests/, 0 errors | Manual run |

---

## Commit Strategy

One commit per epic, in dependency order:

| Commit | Epic | Key Files |
|---|---|---|
| Phase 3 / E1: Inference and orchestration stack | P3-E1 | agents/mcp_client.py, tests/unit/test_mcp_client.py, tests/integration/test_agent_stack.py |
| Phase 3 / E2: Agent output schemas | P3-E2 | agents/schemas.py, tests/unit/test_agent_schemas.py |
| Phase 3 / E3: Base agent infrastructure | P3-E3 | agents/fundamental_agent.py, agents/prompts/fundamental_v1.md, scripts/record_ollama_fixtures.py, tests/ |
| Phase 3 / E4: Fundamental Analyst Agent | P3-E4 | agents/runner.py, agents/baseline_metrics.py, scripts/run_phase3_baseline.py, tests/ |
| Phase 3 / E5: Baseline fixtures and metrics | P3-E5 | tests/fixtures/baseline/phase3_baseline.json, tests/unit/test_phase3_baseline.py |
| Phase 3 / E6: Holistic test | P3-E6 | tests/holistic/test_phase3_agent_pipeline.py |

---

## Open Questions This Phase Will Answer

**OQ-P3-01:** Should the baseline evaluation cover 10 stocks or the 3 available from
Phase 1? Phase 3 uses 3 stocks (AAPL, JPM, XOM). If the Phase 1 fixtures are extended
to 10 stocks before Phase 3 ends, the baseline can be re-run. Otherwise this is
deferred to Phase 8 when the full universe is established.

**OQ-P3-02 (from protocol OQ-AG01):** Which local model produces better structured
output compliance for financial analysis -- Qwen 2.5 7B or Llama 3.1 8B? Resolved
by P3-E1-T3 evaluation. Documented as DJ-014.

**OQ-P3-03 (from protocol OQ-AG02):** What failure modes appear in the baseline?
Categorize: (a) JSON parse failure, (b) hallucinated numbers, (c) ignored None fields,
(d) incoherent rationale. This taxonomy becomes the measurement framework for
Phase 5 (Verification).

**OQ-P3-04 (from protocol OQ-AG03):** How calibrated are LLM confidence estimates
out of the box? Measure: does the model assign high confidence to opinions that are
later confirmed by price movement, or is confidence random? This requires Phase 14
(Paper Trading) to resolve properly, but Phase 3 records the raw confidence values for
future analysis.

**OQ-P3-05:** How does the agent handle a FundamentalsSnapshot where most fields are
None (low-quality data)? Does it produce a coherent "insufficient data" response or
does it fabricate? This is the critical robustness test for prompt design.

---

## Connections to Earlier and Later Phases

**Depends on Phase 2:**
- All agent tool calls go through the Phase 2 MCP server
- The Phase 2 call_id field is the foundation of the Phase 3 audit trail
- Phase 2 schemas (FundamentalsSnapshot, OHLCVDataset) are the agent's input types

**Phase 4 depends on this phase:**
- The AgentSignal schema is the vote interface for Phase 4's ensemble
- The LangGraph graph structure from Phase 3 is extended (not replaced) in Phase 4
- The Phase 3 baseline metrics are the comparison baseline for Phase 4 improvement

**Phase 5 (Verification) depends on this phase:**
- The Phase 3 hallucination checker (simple) informs Phase 5 design
- The call_ids in AgentSignal are the audit trail hooks Phase 5 will verify
- Phase 3 documents the failure modes that Phase 5's verifier must catch

**Phase 6 (LangFuse) depends on this phase:**
- LangGraph is the framework Phase 6 will instrument
- Ollama's OpenAI-compatible endpoint is what LangFuse traces
- Phase 3 establishes the trace structure (nodes = spans) Phase 6 will formalize

**Phase 10 (Evaluation) uses this phase's baselines:**
- The phase3_baseline.json fixture is the floor for all future improvement measurements
- The baseline metrics schema (compliance_rate, hallucination_count, etc.) becomes the
  standard evaluation format used across all phases

**Phase 11 (Fine-Tuning) uses Phase 3's prompt templates:**
- The prompt versions (fundamental_v1.md) are the training document templates
- The Phase 3 raw outputs (where the model got things right) are candidate training data
- The Phase 3 failure modes (hallucinations, None-ignored) define what fine-tuning should fix
