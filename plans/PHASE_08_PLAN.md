# Phase 8: Full Agent Population

**Status:** PLANNED

| Epic | Title | Status |
|---|---|---|
| P8-E0 | venvs/ta/ activation (indicators MCP server) | PLANNED |
| P8-E1 | Schema extension (4 new agents + EnsembleOutput) | PLANNED |
| P8-E2 | Risk Agent (8a) | PLANNED |
| P8-E3 | Macro Agent (8b) | PLANNED |
| P8-E4 | Sentiment Agent (8c) | PLANNED |
| P8-E5 | Contrarian Agent (8d) | PLANNED |
| P8-E6 | Ensemble runner extension (N-agent voting + orchestration) | PLANNED |
| P8-E7 | Baseline measurement + bitacora | PLANNED |

**David Sections:** §10.2 Agent Specifications, §10.3 Diversity Requirements, §12.2.2 Confidence-weighted vote, §5.6.1-5.6.5 Complexity metrics
**Learning Guide Topics:** 3.1 Agent Architecture Fundamentals, 3.3 Collective Intelligence, 8.2 Collective Intelligence, 8.3 Emergence & Measurement
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 8
**Context Reference:** plans/PHASE_08_CONTEXT.md (DJ-032 through DJ-038)

---

## Governing Philosophy for This Phase

Phase 8 answers the central empirical question of HiFi's first year: **does agent diversity actually help?**

The marginal contribution curve — ensemble performance as a function of the number of agents — is one of the most interesting results HiFi will produce. If adding the fifth agent does not improve the collective decision, that is valuable evidence about the limits of ensemble diversity under current architectural constraints. If it does, that validates the complexity-science hypothesis at the core of the project: that a heterogeneous population of bounded-rational agents produces better collective decisions than any individual agent.

The phase adds four agents in a fixed order, each with strictly separated information access. The order is not arbitrary: Risk (8a) exercises the existing financial MCP infrastructure; Macro (8b) adds economic context orthogonal to firm-level data; Sentiment (8c) activates the Phase 7 knowledge system with qualitative filing analysis; Contrarian (8d) is a second-pass agent that stress-tests the emerging consensus before the final output is produced.

**Diversity is enforced architecturally, not just nominally.** Each new agent differs from every existing agent on at least two of the five diversity dimensions defined in David §10.3: model family, information access, prompt structure, fine-tuning status, and role. This is not a cosmetic design choice — it is the mechanism by which the ensemble avoids correlated failures.

**The Contrarian Agent is architecturally distinct.** It does not vote. It does not produce a Buy/Hold/Sell signal. It receives the other agents' analyses and the preliminary collective decision, then articulates what could go wrong with that consensus. In Phase 8, its output is logged and included in EnsembleOutput but does not modify the collective_decision. The mechanism by which the Contrarian influences decisions is the subject of Phase 9.

**The scientific measurement protocol is embedded in the plan.** After each agent addition (8a, 8b, 8c, 8d), the standard evaluation runs on AAPL/JPM/XOM at 2023-03-31 and records: disagreement entropy H, pairwise inter-agent correlation, and marginal contribution of the new agent. These measurements are the primary deliverable of Phase 8.

---

## Pre-Phase Decisions (from plans/PHASE_08_CONTEXT.md)

All decisions are pre-decided. No empirical decisions required in this phase (unlike Phase 7's DJ-030/DJ-031 chunking experiment).

**DJ-032 — Model assignments (diversity requirement §10.3):**

| Agent | Model | max_tokens | Env Var |
|---|---|---|---|
| Fundamental | qwen2.5-coder-32b-instruct-mlx | 1024 | HIFI_FUNDAMENTAL_MODEL |
| Technical | mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled | 4096 | HIFI_TECHNICAL_MODEL |
| Risk | google/gemma-3-4b | 1024 | HIFI_RISK_MODEL |
| Macro | qwen3.5-27b-claude-4.6-opus-reasoning-distilled-qx64-hi-mlx | 4096 | HIFI_MACRO_MODEL |
| Sentiment | qwen2.5-coder-32b-instruct-mlx | 1024 | HIFI_SENTIMENT_MODEL |
| Contrarian | mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled | 4096 | HIFI_CONTRARIAN_MODEL |

max_tokens=4096 is required for reasoning-distilled models (internal "think" tokens consume visible budget). max_tokens=1024 is sufficient for standard models. This is established in Phase 4 (technical agent, DJ-032 reference).

**DJ-033 — Contrarian is second-pass:** receives all other agents' analyses + preliminary collective_decision before formulating counter-thesis. Does NOT produce AgentSignal. Does NOT participate in voting.

**DJ-034 — Sentiment Agent information source:** SEC 8-K + 10-K MD&A + 10-Q MD&A from knowledge store (already indexed). No earnings call transcripts in Phase 8. Sentiment Agent uses RAG as its ONLY information source (not MCP financial tools).

**DJ-035 — venvs/ta/ activated in Phase 8:** `src/hifi/mcp/indicators_server.py` runs inside `venvs/ta/` with pinned pandas 1.5.3 + pandas-ta 0.3.14b0. The Risk Agent uses existing `get_risk_metrics` (financial_server.py), not the new indicators server. The venvs/ta/ activation is an architectural unlock for Phase 9+ extended TA.

**DJ-036 — New schemas:** `RiskAnalysis`, `MacroAnalysis`, `SentimentAnalysis`, `ContrarianAnalysis` added to `src/hifi/agents/schemas.py`.

**DJ-037 — Incremental evaluation:** after each sub-phase (8a, 8b, 8c, 8d), run AAPL/JPM/XOM and record H, pairwise correlation, marginal contribution in the bitacora.

**DJ-038 — Ensemble runner extended:** `run_ensemble()` gains `agents: list[str] | None` parameter. Default None = all agents. Contrarian always runs last regardless of agents list order.

---

## Interface Design

### New Agent Schemas

```python
# src/hifi/agents/schemas.py — additions

class RiskAnalysis(BaseModel):
    """Output schema for the Risk Agent (P8-E2)."""
    signal: AgentSignal           # Buy/Hold/Sell from risk-management perspective
    risk_assessment: str          # structured risk profile (volatility regime, tail risk, etc.)
    recommended_position_size: float | None  # fraction of portfolio [0,1]; None if insufficient data
    prompt_version: str           # "risk_v1"

class MacroAnalysis(BaseModel):
    """Output schema for the Macro Agent (P8-E3)."""
    signal: AgentSignal           # Buy/Hold/Sell based on macro environment alignment
    regime_assessment: str        # current macro regime classification
    rationale: str                # how macro environment supports or threatens the thesis
    prompt_version: str           # "macro_v1"

class SentimentAnalysis(BaseModel):
    """Output schema for the Sentiment Agent (P8-E4)."""
    signal: AgentSignal           # Buy/Hold/Sell based on qualitative filing analysis
    sentiment_summary: str        # qualitative assessment of management tone + disclosures
    notable_signals: list[str]    # specific statements or events flagged as significant
    prompt_version: str           # "sentiment_v1"

class ContrarianAnalysis(BaseModel):
    """Output schema for the Contrarian Agent (P8-E5). No AgentSignal — does not vote."""
    alternative_thesis: str       # what could go wrong with the consensus
    risk_scenario: str            # specific adverse scenario with estimated probability
    counterargument: str          # structured argument against the dominant position
    confidence: float             # confidence in the contrarian view [0, 1]
    prompt_version: str           # "contrarian_v1"
```

### Extended EnsembleOutput

```python
# src/hifi/collective/schemas.py — extended

class EnsembleOutput(BaseModel):
    """Full N-agent ensemble result envelope."""
    ticker: str
    as_of_date: str
    # Existing agents (Phase 4)
    fundamental: FundamentalAnalysis
    technical: TechnicalAnalysis
    # New agents (Phase 8) — optional for backward compatibility
    risk: RiskAnalysis | None = None
    macro: MacroAnalysis | None = None
    sentiment: SentimentAnalysis | None = None
    contrarian: ContrarianAnalysis | None = None
    # Collective decision (voting over participating agents only)
    decision: EnsembleDecision
    # Verification (Phase 5)
    verification: EnsembleVerificationReport | None = None
    elapsed_seconds: float = 0.0
```

### Extended run_ensemble() signature

```python
# src/hifi/agents/ensemble_runner.py — extended

def run_ensemble(
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    use_rag: bool = False,
    agents: list[str] | None = None,  # NEW: None = all; subset by name for ablations
) -> EnsembleOutput:
    """
    Run N agents sequentially and aggregate.

    agents parameter controls which agents participate:
    - None: all agents (fundamental, technical, risk, macro, sentiment, contrarian)
    - ["fundamental", "technical"]: Phase 4 behavior (backward compatible)
    - ["fundamental", "technical", "risk"]: Phase 8a state

    Contrarian always runs last (it needs the preliminary consensus as input).
    The agents list controls which non-contrarian agents contribute to voting;
    contrarian is automatically appended if "contrarian" is in agents (or agents is None).
    """
```

### New Agent Entry Points

```python
# src/hifi/agents/risk_agent.py
def run_risk_analysis(
    ticker: str,
    as_of_date: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    use_rag: bool = False,
) -> RiskAnalysis: ...

# src/hifi/agents/macro_agent.py
def run_macro_analysis(
    ticker: str,
    as_of_date: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    use_rag: bool = False,
) -> MacroAnalysis: ...

# src/hifi/agents/sentiment_agent.py
def run_sentiment_analysis(
    ticker: str,
    as_of_date: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
) -> SentimentAnalysis: ...
# Note: use_rag is always True for Sentiment Agent (RAG is its primary information source)

# src/hifi/agents/contrarian_agent.py
def run_contrarian_analysis(
    ticker: str,
    as_of_date: str,
    ensemble_context: str,   # formatted string of all other agents' outputs + preliminary decision
    tracer: AbstractTracer | None = None,
) -> ContrarianAnalysis: ...
```

### Marginal Contribution Measurement

```python
# scripts/run_phase8_baseline.py

def run_incremental_evaluation(
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    agent_sets: list[list[str]],  # e.g., [["fundamental","technical"], [...,"risk"], ...]
    data_dir: str,
) -> list[dict]:
    """
    Run run_ensemble() for each agent_set subset.
    For each result, record:
    - n_agents: int
    - agents: list[str]
    - collective_decision: str
    - collective_confidence: float
    - disagreement_entropy: float
    - pairwise_diversity: float (mean of off-diagonal pairwise correlation)
    - mean_hr: float (from verify_ensemble)
    Returns list of measurement dicts for marginal contribution table.
    """
```

---

## Epic P8-E0: venvs/ta/ Activation

**Objective:** Activate the `venvs/ta/` scaffold (created in Phase 2) as a working MCP server. Creates `src/hifi/mcp/indicators_server.py` — a FastMCP server that runs inside the isolated virtual environment using pinned `pandas 1.5.3` + `pandas-ta 0.3.14b0`. This is an architectural unlock for Phase 9+ extended TA capabilities; Phase 8 agents use the existing `financial_server.py` MCP tools.

**Architecture recap (from DAVID.md §17 DJ-010, DJ-035).** The MCP subprocess boundary is a process boundary. `venvs/ta/bin/python -m hifi.mcp.indicators_server` runs with a pinned dependency stack isolated from the main process. The main process calls it via `call_tool("get_extended_indicators", ..., server_script="venvs/ta/bin/python -m hifi.mcp.indicators_server")` and receives JSON results. No changes to agents, verification, or observability layers are required.

**Scope for Phase 8.** A minimal initial server exposing two tools:
- `get_extended_indicators(ticker, start_date, end_date)` — returns additional indicators not in the Phase 2 custom set: VWAP, OBV, STOCH, ADX, CCI, Williams %R (pandas-ta implementations)
- `health_check()` — returns version info and available indicator list

This is a foundation; full TA library integration happens in Phase 9.

**Fixture strategy.** The indicators_server runs as a subprocess with its own venv. Unit tests stub the subprocess call via `call_tool` monkeypatching (same pattern as existing MCP server tests). An integration test runs the actual server subprocess — skipped if `venvs/ta/` is not set up.

| Ticket | Description | Status |
|---|---|---|
| P8-E0-T1 | Run `scripts/setup_ta_venv.sh` and verify `venvs/ta/bin/python -c "import pandas_ta; print(pandas_ta.__version__)"` succeeds | PLANNED |
| P8-E0-T2 | Create `src/hifi/mcp/indicators_server.py` — FastMCP server with `health_check()` tool returning `{"pandas_ta_version": str, "available_indicators": list[str]}` | PLANNED |
| P8-E0-T3 | Implement `get_extended_indicators(ticker, start_date, end_date)` tool: loads OHLCV parquet, computes VWAP/OBV/STOCH/ADX/CCI/WILLR via pandas-ta, returns JSON | PLANNED |
| P8-E0-T4 | Unit test: `health_check()` response is a dict with `pandas_ta_version` and `available_indicators` keys (server module imported directly, not via subprocess) | PLANNED |
| P8-E0-T5 | Unit test: `get_extended_indicators()` with synthetic OHLCV returns non-null values for all 6 indicators | PLANNED |
| P8-E0-T6 | Integration test: `call_tool("health_check", {}, server_module="hifi.mcp.indicators_server")` returns valid response when `venvs/ta/` is available (skip if not) | PLANNED |
| P8-E0-T7 | Update `Makefile` with `ta-venv-setup`, `ta-venv-test` targets | PLANNED |
| P8-E0-T8 | Update `scripts/check_env.py` with `--check ta-venv` prerequisite check | PLANNED |

**Files to create:**
- `src/hifi/mcp/indicators_server.py`
- `tests/unit/test_indicators_server.py`
- `tests/integration/test_ta_venv.py` (skip unless ta-venv available)

---

## Epic P8-E1: Schema Extension

**Objective:** Add four new analysis schemas to `src/hifi/agents/schemas.py` and extend `EnsembleOutput` and `EnsembleDecision` in `src/hifi/collective/schemas.py` for N-agent support. This epic is the data model foundation that all subsequent agent epics build on.

**Backward compatibility guarantee.** `EnsembleOutput.risk`, `.macro`, `.sentiment`, `.contrarian` are all `Optional` with `None` defaults. All existing tests (Phase 4-7) continue to pass without modification. The `EnsembleDecision` voting logic is already generic (takes a list of AgentSignals) — adding more signals is naturally supported.

**Pairwise diversity matrix.** `EnsembleDecision` currently stores `pairwise_diversity: float` (scalar mean of the 2-agent matrix). With N agents, this becomes a full N×N matrix. For backward compatibility, `pairwise_diversity` remains the scalar mean; a new `pairwise_diversity_matrix: list[list[float]] | None` field stores the full matrix when N > 2.

| Ticket | Description | Status |
|---|---|---|
| P8-E1-T1 | Add `RiskAnalysis` schema to `schemas.py`: `signal: AgentSignal`, `risk_assessment: str`, `recommended_position_size: float | None`, `prompt_version: str` | PLANNED |
| P8-E1-T2 | Add `MacroAnalysis` schema: `signal: AgentSignal`, `regime_assessment: str`, `rationale: str`, `prompt_version: str` | PLANNED |
| P8-E1-T3 | Add `SentimentAnalysis` schema: `signal: AgentSignal`, `sentiment_summary: str`, `notable_signals: list[str]`, `prompt_version: str` | PLANNED |
| P8-E1-T4 | Add `ContrarianAnalysis` schema: `alternative_thesis: str`, `risk_scenario: str`, `counterargument: str`, `confidence: float`, `prompt_version: str` (NO AgentSignal) | PLANNED |
| P8-E1-T5 | Extend `EnsembleOutput` with `risk: RiskAnalysis | None = None`, `macro: MacroAnalysis | None = None`, `sentiment: SentimentAnalysis | None = None`, `contrarian: ContrarianAnalysis | None = None` | PLANNED |
| P8-E1-T6 | Add `pairwise_diversity_matrix: list[list[float]] | None = None` to `EnsembleDecision` | PLANNED |
| P8-E1-T7 | Unit test: `RiskAnalysis`, `MacroAnalysis`, `SentimentAnalysis`, `ContrarianAnalysis` validate correctly with valid inputs | PLANNED |
| P8-E1-T8 | Unit test: `EnsembleOutput` with only `fundamental` + `technical` populated (None for new fields) passes validation — backward compat | PLANNED |
| P8-E1-T9 | Unit test: `EnsembleOutput` with all 6 agents populated validates correctly | PLANNED |
| P8-E1-T10 | Unit test: `ContrarianAnalysis.confidence` rejects values outside [0, 1] (Pydantic validator) | PLANNED |

**Files to modify:**
- `src/hifi/agents/schemas.py`
- `src/hifi/collective/schemas.py`
- `tests/unit/test_agent_schemas.py` (extend existing)

---

## Epic P8-E2: Risk Agent (Sub-phase 8a)

**Objective:** Implement the Risk Agent — the first addition to the ensemble beyond the Phase 4 agents. The Risk Agent reasons exclusively from risk metrics (historical volatility, beta, max drawdown, Sharpe, VaR) and produces a risk-management-perspective investment signal and a position sizing recommendation.

**Information access (strict isolation).** The Risk Agent calls only `get_risk_metrics` from `financial_server.py`. It does NOT call `get_financial_ratios`, `get_growth_metrics`, `get_technical_indicators`, or `get_macro_snapshot`. This isolation is enforced by the prompt and by the MCP tool list passed to the graph — only `get_risk_metrics` is registered.

**LangGraph graph structure.** Three nodes, identical pattern to technical_agent.py:
```
call_mcp_tools → generate_analysis → parse_output
```
`call_mcp_tools` calls `get_risk_metrics` only. `generate_analysis` formats the tool results into the prompt and calls the LLM. `parse_output` extracts `RiskAnalysis` from the JSON response.

**Model.** `google/gemma-3-4b` (HIFI_RISK_MODEL env var). Standard instruction-following model, not reasoning-distilled — use max_tokens=1024. Gemma's information restriction is enforced by info access alone (different model family than Fundamental/Technical satisfies §10.3).

**Position size interpretation.** The Risk Agent is asked to recommend a position size as a fraction of a hypothetical portfolio, justified by the risk profile. This is a novel output field not present in other agents. The prompt instructs the model to output a value in [0, 1] (e.g., 0.05 = 5% of portfolio) or null if insufficient data. Pydantic validates the range.

**Prompt (risk_v1.md).** System prompt: risk management perspective, focus on downside protection and volatility regimes. User template: structured risk metrics block with interpretation guidelines (VaR interpretation, volatility regime thresholds, beta context). Output: JSON with all RiskAnalysis fields.

| Ticket | Description | Status |
|---|---|---|
| P8-E2-T1 | Create `RiskAnalystState` TypedDict in `risk_agent.py`: `ticker`, `as_of_date`, `data_dir`, `tool_results`, `llm_response`, `signal`, `error`, `start_time` | PLANNED |
| P8-E2-T2 | Implement `call_mcp_tools_node`: calls `get_risk_metrics` only; stores result in `tool_results` | PLANNED |
| P8-E2-T3 | Create `src/hifi/agents/prompts/risk_v1.md`: system prompt (risk management perspective) + user template (ticker, as_of_date, risk_metrics block, JSON format instructions) | PLANNED |
| P8-E2-T4 | Implement `generate_analysis_node`: loads `risk_v1.md`, formats tool results, calls `make_llm(HIFI_RISK_MODEL, max_tokens=1024)`, stores raw LLM response | PLANNED |
| P8-E2-T5 | Implement `parse_output_node`: extracts JSON from LLM response, builds `RiskAnalysis` with `AgentSignal` (agent_type="risk"), handles parse errors | PLANNED |
| P8-E2-T6 | Implement `build_risk_graph()` and `run_risk_analysis()` entry point with same signature pattern as `run_technical_analysis()` | PLANNED |
| P8-E2-T7 | Unit test: `call_mcp_tools_node` with monkeypatched `call_tool` returns tool_results with risk metrics keys | PLANNED |
| P8-E2-T8 | Unit test: `parse_output_node` with valid stub JSON returns `RiskAnalysis` with `signal.agent_type == "risk"` | PLANNED |
| P8-E2-T9 | Unit test: `parse_output_node` with invalid JSON returns error state gracefully (does not raise) | PLANNED |
| P8-E2-T10 | Unit test: `recommended_position_size` is None when LLM response omits the field | PLANNED |
| P8-E2-T11 | Integration test: full graph with stub LLM + stub call_tool returns `RiskAnalysis` with valid signal | PLANNED |
| P8-E2-T12 | Integration test: `run_risk_analysis()` with monkeypatched LLM returns valid `RiskAnalysis` in < 5 seconds | PLANNED |

**Files to create:**
- `src/hifi/agents/risk_agent.py`
- `src/hifi/agents/prompts/risk_v1.md`
- `tests/unit/test_risk_agent_nodes.py`
- `tests/integration/test_risk_agent.py`

---

## Epic P8-E3: Macro Agent (Sub-phase 8b)

**Objective:** Implement the Macro Agent — the second addition to the ensemble. The Macro Agent reasons from macroeconomic indicators (interest rates, inflation, employment, GDP, yield curve) and assesses how the current macro regime affects the investment thesis for a specific stock.

**Information access.** The Macro Agent calls only `get_macro_snapshot` from `financial_server.py`. This is economy-wide data, not ticker-specific. The agent's distinctive challenge is interpreting macro signals in the context of a specific stock — the prompt must supply sector context (e.g., "AAPL is a large-cap technology company; consider its sensitivity to consumer spending and interest rate changes").

**Ticker-to-sector mapping.** For the Phase 8 baseline tickers (AAPL, JPM, XOM), sector context is hardcoded in the prompt template as a conditional block. A general-purpose mapping is deferred to Phase 10 (evaluation framework) when sector databases are available.

**Macro snapshot interpretation.** The Phase 2 `get_macro_snapshot` already computes: FEDFUNDS rate, CPI YoY, UNRATE, GS10, GS2, VIXCLS, GDP growth rate, yield curve spread (GS10 - GS2). The Macro Agent prompt provides interpretation guidance: yield curve inversion as recession signal, VIXCLS > 25 as elevated risk regime, etc.

**Model.** `qwen3.5-27b-claude-4.6-opus-reasoning-distilled-qx64-hi-mlx` (HIFI_MACRO_MODEL). Reasoning-distilled — use max_tokens=4096.

| Ticket | Description | Status |
|---|---|---|
| P8-E3-T1 | Create `MacroAnalystState` TypedDict in `macro_agent.py`: `ticker`, `as_of_date`, `data_dir`, `tool_results`, `llm_response`, `signal`, `error`, `start_time` | PLANNED |
| P8-E3-T2 | Implement `call_mcp_tools_node`: calls `get_macro_snapshot` only; macro data is not ticker-scoped but ticker is available for prompt context | PLANNED |
| P8-E3-T3 | Create `src/hifi/agents/prompts/macro_v1.md`: system prompt (macroeconomic analysis perspective) + user template with ticker, macro indicators, sector context block, JSON format | PLANNED |
| P8-E3-T4 | Implement `generate_analysis_node`: formats macro snapshot + sector context, calls `make_llm(HIFI_MACRO_MODEL, max_tokens=4096)`, stores response | PLANNED |
| P8-E3-T5 | Implement `parse_output_node`: extracts JSON, builds `MacroAnalysis` with `AgentSignal` (agent_type="macro"), handles parse errors | PLANNED |
| P8-E3-T6 | Implement `build_macro_graph()` and `run_macro_analysis()` entry point | PLANNED |
| P8-E3-T7 | Unit test: `call_mcp_tools_node` calls `get_macro_snapshot` (not ticker-specific tools) | PLANNED |
| P8-E3-T8 | Unit test: `parse_output_node` returns `MacroAnalysis` with non-empty `regime_assessment` | PLANNED |
| P8-E3-T9 | Unit test: `parse_output_node` with malformed JSON sets `error` state without raising | PLANNED |
| P8-E3-T10 | Integration test: full graph with stub LLM + stub call_tool returns `MacroAnalysis` with valid signal | PLANNED |
| P8-E3-T11 | Integration test: `run_macro_analysis()` with monkeypatched LLM completes and returns `MacroAnalysis` | PLANNED |

**Files to create:**
- `src/hifi/agents/macro_agent.py`
- `src/hifi/agents/prompts/macro_v1.md`
- `tests/unit/test_macro_agent_nodes.py`
- `tests/integration/test_macro_agent.py`

---

## Epic P8-E4: Sentiment Agent (Sub-phase 8c)

**Objective:** Implement the Sentiment Agent — the third addition. The Sentiment Agent uses only the Phase 7 knowledge system (SEC filings via RAG) and produces a qualitative assessment of management tone, key disclosures, and notable signals. This is the first agent whose primary information source is unstructured text rather than numerical MCP tools.

**Information access.** The Sentiment Agent calls `retrieve_context` from `knowledge_server.py` ONLY. It does NOT call any financial MCP tools. This is the primary diversity dimension that distinguishes it from the Fundamental Agent (which uses the same model but has access to financial ratios and growth metrics, not qualitative filings).

**RAG is mandatory (not optional).** Unlike the Fundamental and Technical agents where `use_rag=False` is the default and RAG is an enhancement, for the Sentiment Agent RAG IS the information source. There is no `use_rag` parameter — the `retrieve_context_node` always runs. If retrieval returns no passages (e.g., knowledge store not populated), the agent returns a default "Insufficient Data" signal rather than fabricating sentiment from pre-training.

**Query design.** The Sentiment Agent issues three targeted retrieval queries per ticker:
1. Management tone and forward guidance
2. Risk disclosures and regulatory concerns
3. Strategic initiatives and capital allocation

Results from all three queries are merged and deduplicated before injection into the prompt.

**Model.** `qwen2.5-coder-32b-instruct-mlx` (HIFI_SENTIMENT_MODEL). Same model family as Fundamental Agent — diversity is maintained through information access (RAG-only vs. financial ratios) and role (qualitative narrative vs. quantitative fundamentals).

**LangGraph graph structure:**
```
retrieve_context → generate_analysis → parse_output
```
No `call_mcp_tools` node (no financial tools). The `retrieve_context` node runs three queries and formats passages into a context block.

| Ticket | Description | Status |
|---|---|---|
| P8-E4-T1 | Create `SentimentAnalystState` TypedDict in `sentiment_agent.py`: `ticker`, `as_of_date`, `data_dir`, `retrieved_context`, `llm_response`, `signal`, `error`, `start_time` | PLANNED |
| P8-E4-T2 | Implement `retrieve_context_node`: issues 3 queries (management tone, risk disclosures, strategic initiatives) via `call_tool("retrieve_context", ...)` for each; merges and deduplicates passages; returns formatted context or empty string on failure (fail-open) | PLANNED |
| P8-E4-T3 | Create `src/hifi/agents/prompts/sentiment_v1.md`: system prompt (qualitative analysis, narrative focus) + user template with retrieved_context block and JSON format instructions | PLANNED |
| P8-E4-T4 | Implement `generate_analysis_node`: uses retrieved_context in prompt; if empty string, returns default `SentimentAnalysis` with signal="Hold", confidence=0.0, sentiment_summary="Insufficient filing data available", notable_signals=[] | PLANNED |
| P8-E4-T5 | Implement `parse_output_node`: extracts JSON, builds `SentimentAnalysis` with `AgentSignal` (agent_type="sentiment"), validates `notable_signals` is a list | PLANNED |
| P8-E4-T6 | Implement `build_sentiment_graph()` and `run_sentiment_analysis()` entry point (no `use_rag` parameter) | PLANNED |
| P8-E4-T7 | Unit test: `retrieve_context_node` with stub `call_tool` returning 2 passages returns non-empty formatted context string | PLANNED |
| P8-E4-T8 | Unit test: `retrieve_context_node` with `call_tool` raising exception returns empty string (fail-open) | PLANNED |
| P8-E4-T9 | Unit test: `generate_analysis_node` with empty retrieved_context returns default "Insufficient Data" sentiment without calling LLM | PLANNED |
| P8-E4-T10 | Unit test: `parse_output_node` with valid stub JSON returns `SentimentAnalysis` with `signal.agent_type == "sentiment"` | PLANNED |
| P8-E4-T11 | Unit test: `parse_output_node` with malformed JSON sets error state | PLANNED |
| P8-E4-T12 | Integration test: full graph with stub call_tool (passages) + stub LLM returns valid `SentimentAnalysis` | PLANNED |
| P8-E4-T13 | Integration test: `run_sentiment_analysis()` with monkeypatched components completes successfully | PLANNED |

**Files to create:**
- `src/hifi/agents/sentiment_agent.py`
- `src/hifi/agents/prompts/sentiment_v1.md`
- `tests/unit/test_sentiment_agent_nodes.py`
- `tests/integration/test_sentiment_agent.py`

---

## Epic P8-E5: Contrarian Agent (Sub-phase 8d)

**Objective:** Implement the Contrarian Agent — the fourth and final addition. The Contrarian Agent is architecturally distinct: it is a second-pass agent that receives all other agents' outputs and the preliminary collective decision, then produces a structured counter-thesis. It does NOT vote.

**Design rationale (from DAVID.md §10.2).** A contrarian that simply votes the opposite of the consensus is noise. A contrarian that articulates *why* the consensus might be wrong is intelligence. The Contrarian Agent is designed to be the latter. Its output — `alternative_thesis`, `risk_scenario`, `counterargument` — is structured prose that can be read by a human operator or used as input to a future debate mechanism (Phase 9).

**Input construction.** The `run_contrarian_analysis()` function receives a pre-formatted `ensemble_context: str` that includes:
- Each participating agent's signal, confidence, and rationale summary
- The preliminary `collective_decision` and `collective_confidence`
- The `disagreement_entropy` H

This context is formatted by `format_ensemble_context()` in the ensemble runner before passing to the Contrarian.

**No LangGraph graph.** The Contrarian Agent is simpler than others — it takes a formatted string context and calls the LLM once. It does not need a LangGraph graph (no MCP tools, no state machine). It is a direct LLM call with structured output.

**Model.** `mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled` (HIFI_CONTRARIAN_MODEL). Same model family as Technical Agent — diversity maintained through role (adversarial second-pass vs. independent price-analysis first-pass) and information access (all agents' outputs vs. price-only MCP tools). Reasoning-distilled — use max_tokens=4096.

**Confidence calibration note.** The Contrarian's `confidence` field represents its confidence in the counter-thesis, not in a market direction. A high-confidence contrarian (0.9) says "the consensus is likely wrong." A low-confidence contrarian (0.1) says "the consensus seems sound, though here are possible risks." This interpretation is specified in the prompt.

| Ticket | Description | Status |
|---|---|---|
| P8-E5-T1 | Implement `format_ensemble_context(agents_dict: dict, decision: EnsembleDecision) -> str` in `ensemble_runner.py`: formats all agent signals/rationales + preliminary collective decision into a structured text block | PLANNED |
| P8-E5-T2 | Create `src/hifi/agents/prompts/contrarian_v1.md`: system prompt (adversarial stress-tester, articulate counter-thesis) + user template with ensemble_context block and JSON format | PLANNED |
| P8-E5-T3 | Implement `run_contrarian_analysis(ticker, as_of_date, ensemble_context, tracer)`: direct LLM call with `make_llm(HIFI_CONTRARIAN_MODEL, max_tokens=4096)`; returns `ContrarianAnalysis` | PLANNED |
| P8-E5-T4 | Implement parse logic: extract JSON from LLM response, build `ContrarianAnalysis`; on parse error return default with confidence=0.0 and filled alternative_thesis/risk_scenario/counterargument indicating parse failure | PLANNED |
| P8-E5-T5 | Unit test: `format_ensemble_context()` with 3-agent dict returns non-empty string containing each agent's signal decision and preliminary collective_decision | PLANNED |
| P8-E5-T6 | Unit test: `run_contrarian_analysis()` with monkeypatched LLM returning valid JSON returns `ContrarianAnalysis` with confidence in [0,1] | PLANNED |
| P8-E5-T7 | Unit test: `run_contrarian_analysis()` with LLM returning malformed JSON returns default ContrarianAnalysis without raising | PLANNED |
| P8-E5-T8 | Unit test: `ContrarianAnalysis` has no `signal` field (not an AgentSignal bearer — not a voting agent) | PLANNED |
| P8-E5-T9 | Integration test: `run_contrarian_analysis()` with stub LLM completes and returns valid `ContrarianAnalysis` | PLANNED |

**Files to create:**
- `src/hifi/agents/contrarian_agent.py`
- `src/hifi/agents/prompts/contrarian_v1.md`
- `tests/unit/test_contrarian_agent.py`
- `tests/integration/test_contrarian_agent.py`

---

## Epic P8-E6: Ensemble Runner Extension

**Objective:** Extend `run_ensemble()` in `ensemble_runner.py` to orchestrate all 6 agents (Fundamental, Technical, Risk, Macro, Sentiment, Contrarian) with an `agents` registry parameter. Update `confidence_weighted_vote()` to handle N participating agents. Add `pairwise_diversity_matrix` to `compute_ensemble_metrics()`. Ensure the Phase 4-7 behavior is fully preserved when `agents=["fundamental", "technical"]` or `agents=None` with only Phase 4 agents installed.

**Contrarian execution order.** Regardless of agent list order, the Contrarian always runs last. The ensemble runner:
1. Runs all non-contrarian agents in the order specified (or default order)
2. Aggregates their signals via `confidence_weighted_vote()`
3. If "contrarian" is in agents (or agents is None): runs `run_contrarian_analysis()` with the preliminary decision as context
4. Returns the final `EnsembleOutput` with all results

**N-agent voting.** `confidence_weighted_vote()` already accepts a list of `AgentSignal` objects and is naturally extensible. Risk, Macro, and Sentiment all contribute `AgentSignal` objects. The Contrarian does not. No changes needed to the voting logic itself — only the ensemble runner needs to collect the right signals.

**Pairwise diversity matrix.** `compute_ensemble_metrics()` in `collective/metrics.py` currently computes `pairwise_diversity(signals)` as a scalar mean. For N > 2 agents, it additionally computes the full N×N matrix and stores it in `EnsembleDecision.pairwise_diversity_matrix`.

**Holistic test.** `tests/holistic/test_phase8_agent_population.py` runs the full 6-agent ensemble (all stubs) and verifies:
- 6 agents contribute to EnsembleOutput
- Contrarian analysis present and valid
- Phase 4-6 regression: use of `agents=["fundamental", "technical"]` produces identical structure to Phase 6 output

| Ticket | Description | Status |
|---|---|---|
| P8-E6-T1 | Add `agents: list[str] | None = None` parameter to `run_ensemble()`; validate against known agent names; raise `ValueError` for unknown names | PLANNED |
| P8-E6-T2 | Implement agent dispatch in `run_ensemble()`: call `run_risk_analysis()`, `run_macro_analysis()`, `run_sentiment_analysis()` based on `agents` list; collect their `AgentSignal` objects for voting | PLANNED |
| P8-E6-T3 | Implement Contrarian dispatch: after voting, if "contrarian" in agents (or agents is None), call `format_ensemble_context()` + `run_contrarian_analysis()`; attach to EnsembleOutput | PLANNED |
| P8-E6-T4 | Update `compute_ensemble_metrics()` in `collective/metrics.py`: compute full `pairwise_diversity_matrix` (N×N) when N > 2; store in `EnsembleDecision` | PLANNED |
| P8-E6-T5 | Maintain backward compat: `run_ensemble()` with `agents=["fundamental", "technical"]` or with no new agent modules installed produces EnsembleOutput identical in structure to Phase 6 | PLANNED |
| P8-E6-T6 | Integration test: `run_ensemble()` with agents=["fundamental","technical","risk"] (stub Risk LLM) returns EnsembleOutput with `risk` field populated | PLANNED |
| P8-E6-T7 | Integration test: `run_ensemble()` with agents=None (all stubs) returns EnsembleOutput with all 6 fields populated | PLANNED |
| P8-E6-T8 | Integration test: `run_ensemble()` with agents=["fundamental","technical"] returns EnsembleOutput where `risk`, `macro`, `sentiment`, `contrarian` are all None | PLANNED |
| P8-E6-T9 | Holistic test `test_phase8_agent_population.py`: full N-agent ensemble with all stubs; verify pairwise_diversity_matrix is 5×5 (excluding Contrarian); verify Contrarian analysis present | PLANNED |
| P8-E6-T10 | Holistic test: Phase 6 regression — `run_ensemble(agents=["fundamental","technical"])` produces EnsembleOutput structurally identical to Phase 6 behavior | PLANNED |

**Files to modify:**
- `src/hifi/agents/ensemble_runner.py`
- `src/hifi/collective/metrics.py`
- `src/hifi/collective/schemas.py` (already modified in E1)

**Files to create:**
- `tests/holistic/test_phase8_agent_population.py`
- `tests/integration/test_phase8_ensemble.py`

---

## Epic P8-E7: Baseline Measurement + Bitacora

**Objective:** Run the incremental evaluation protocol (Protocol Phase 8 §success criteria). Measure disagreement entropy H, pairwise inter-agent correlation, and marginal contribution after each agent addition (8a, 8b, 8c, 8d). Write the scientific bitacora. Update STATUS.md.

**Incremental evaluation protocol.** The baseline script runs the ensemble five times on each of AAPL/JPM/XOM at 2023-03-31, each time with one more agent:

| Run | agents | N voting agents |
|---|---|---|
| Phase 4 baseline | ["fundamental", "technical"] | 2 |
| After 8a | + "risk" | 3 |
| After 8b | + "macro" | 4 |
| After 8c | + "sentiment" | 5 |
| After 8d | + "contrarian" | 5 (contrarian doesn't vote) |

For each run, record: `collective_decision`, `collective_confidence`, `disagreement_entropy`, `mean_pairwise_diversity`, agent-level signals, and (when applicable) `ContrarianAnalysis` summary.

**Marginal contribution metric.** Marginal contribution of agent A_n is defined as the change in collective_confidence between the N-1 and N agent runs, averaged over the 3 tickers:
```
MC(A_n) = mean_tickers(collective_confidence(N agents) - collective_confidence(N-1 agents))
```
A positive MC indicates the new agent adds information that resolves ambiguity. A negative MC indicates the new agent introduces conflicting signals that reduce ensemble confidence. Both are valuable empirical findings.

**LM Studio required.** This script requires all configured models to be running in LM Studio simultaneously (or sequentially). An `--agents` flag allows partial runs for testing.

| Ticket | Description | Status |
|---|---|---|
| P8-E7-T1 | Create `scripts/run_phase8_baseline.py`: loads snapshots, runs 5-level incremental evaluation on AAPL/JPM/XOM, records all measurements | PLANNED |
| P8-E7-T2 | Implement `run_incremental_evaluation()` helper: iterates agent_sets, calls `run_ensemble()` for each, collects metrics | PLANNED |
| P8-E7-T3 | Implement marginal contribution calculation and output as structured JSON + human-readable table | PLANNED |
| P8-E7-T4 | Save to `tests/fixtures/baseline/phase8_agent_population.json` | PLANNED |
| P8-E7-T5 | Update `scripts/check_env.py` with `--check phase8-fixture` | PLANNED |
| P8-E7-T6 | Update `Makefile` with `baseline-phase8` target | PLANNED |
| P8-E7-T7 | Create `tests/unit/test_phase8_baseline.py`: skip if fixture absent; when present, validate JSON schema (correct agent keys, H values in [0, log2(3)], all tickers present) | PLANNED |
| P8-E7-T8 | Write `doc/bitacora/PHASE_08_AGENT_POPULATION.md`: governing philosophy, marginal contribution table for AAPL/JPM/XOM, interpretation, open questions for Phase 9 | PLANNED |
| P8-E7-T9 | Update `plans/STATUS.md`: Phase 8 COMPLETE, update test count, update David proximity matrix | PLANNED |

**Files to create:**
- `scripts/run_phase8_baseline.py`
- `tests/fixtures/baseline/phase8_agent_population.json` (generated by script)
- `tests/unit/test_phase8_baseline.py`
- `doc/bitacora/PHASE_08_AGENT_POPULATION.md`

**Files to modify:**
- `plans/STATUS.md`
- `scripts/check_env.py`
- `Makefile`

---

## Epic Dependency Graph

```
P8-E0 (venvs/ta/)
  |
  +-- [independent — can run in parallel with E1-E5]

P8-E1 (Schemas)
  |
  +-- P8-E2 (Risk Agent)
  |     |
  |     +-- P8-E3 (Macro Agent)
  |           |
  |           +-- P8-E4 (Sentiment Agent)
  |                 |
  |                 +-- P8-E5 (Contrarian Agent)
  |                       |
  |                       +-- P8-E6 (Ensemble Extension)
  |                             |
  |                             +-- P8-E7 (Baseline + Bitacora)
```

E0 is independent and can be developed in parallel. E1 establishes schemas; E2-E5 implement agents in order (each measurable after integration into E6). E6 integrates all; E7 measures.

Note: E2-E5 each have standalone unit and integration tests. The full N-agent evaluation requires E6 (ensemble extension) to be complete. Sub-phase measurements (8a, 8b, 8c) can be approximated by running `run_ensemble(agents=[...])` after each epic, but the formal baseline in E7 requires all agents.

---

## New Dependencies

**Production:** No new Python packages required.
- LangGraph (already a dep): used for E2, E3, E4 agent graphs
- openai client (already a dep): used in E5 Contrarian's direct LLM call
- mcp/FastMCP (already a dep): used in E0 indicators_server.py

**Dev (venvs/ta/ only, not main env):** `pandas-ta 0.3.14b0` + `pandas 1.5.3` (already in `venvs/ta/requirements.txt` — no changes needed).

**No new pyproject.toml dependencies** unless venvs/ta/ integration requires a subprocess management package (unlikely — existing `call_tool()` subprocess pattern is sufficient).

---

## Phase 8 Quality Gates

| Gate | Criterion | Measured By |
|---|---|---|
| All unit tests pass | pytest tests/unit/, 0 failures | `make test` |
| All integration tests pass | pytest tests/integration/, 0 failures | `make test` |
| Holistic tests pass | pytest tests/holistic/test_phase8_*.py | `make test` |
| Phase 4-7 regression | All existing holistic tests pass unchanged | `make test` |
| No LM Studio required for tests | All tests pass without LM Studio (monkeypatched LLMs) | `make test` |
| Lint clean | ruff check src/ tests/ scripts/, 0 errors | `make lint` |
| Diversity verified | Each new agent differs from existing agents on >= 2 dimensions (§10.3) | Architecture review |
| Contrarian no-vote confirmed | ContrarianAnalysis has no AgentSignal field; not included in confidence_weighted_vote | P8-E5-T8 |
| Marginal contribution measured | MC for each agent (8a, 8b, 8c) recorded in bitacora table | P8-E7-T8 |
| Disagreement entropy tracked | H reported for each N-agent run in baseline JSON | P8-E7-T4 |
| Pairwise correlation measured | Full 5×5 correlation matrix in EnsembleDecision for N=5 agents | P8-E6-T9 |
| Backward compatibility | `run_ensemble(agents=["fundamental","technical"])` identical to Phase 6 output | P8-E6-T10 |

---

## Commit Strategy

| Commit | Epic | Key Files |
|---|---|---|
| Phase 8 / E0: venvs/ta indicators server | P8-E0 | mcp/indicators_server.py, tests/unit/test_indicators_server.py |
| Phase 8 / E1: Agent population schemas | P8-E1 | agents/schemas.py, collective/schemas.py |
| Phase 8 / E2: Risk Agent (8a) | P8-E2 | agents/risk_agent.py, prompts/risk_v1.md |
| Phase 8 / E3: Macro Agent (8b) | P8-E3 | agents/macro_agent.py, prompts/macro_v1.md |
| Phase 8 / E4: Sentiment Agent (8c) | P8-E4 | agents/sentiment_agent.py, prompts/sentiment_v1.md |
| Phase 8 / E5: Contrarian Agent (8d) | P8-E5 | agents/contrarian_agent.py, prompts/contrarian_v1.md |
| Phase 8 / E6: Ensemble runner extension | P8-E6 | ensemble_runner.py, collective/metrics.py, holistic test |
| Phase 8 / E7: Population baseline + bitacora | P8-E7 | scripts/run_phase8_baseline.py, bitacora, STATUS.md |

---

## Open Questions This Phase Will Answer

**OQ-P8-01: Does agent diversity improve collective_confidence?**
Measured by the marginal contribution table in E7. If MC > 0 for all three new voting agents, diversity helps. If MC <= 0 for any agent, that agent introduces conflicting noise rather than complementary signal.

**OQ-P8-02: Does disagreement entropy H increase as more agents are added?**
Adding more diverse agents should increase H (greater spread of opinions). If H stays flat or decreases, the agents may be too correlated despite nominal diversity — a design problem requiring investigation.

**OQ-P8-03: Is the Contrarian Agent's counter-thesis coherent and non-trivial?**
Qualitative evaluation during E7 bitacora writing. A trivial contrarian says "what if the stock goes down?" A useful contrarian identifies a specific mechanism, cites a data point from the filing corpus, and estimates a probability. This is a qualitative assessment of prompt engineering quality.

**OQ-P8-04: Does the Sentiment Agent improve HR/GR vs. Phase 7 baseline?**
The Sentiment Agent uses RAG (knowledge store). Its HR/GR can be compared to the Phase 7 RAG baseline. If Sentiment Agent HR is lower than Phase 7 Fundamental Agent HR, RAG helps qualitative agents too.

---

## Connections to Earlier and Later Phases

**Depends on Phase 7:**
- Knowledge store (LanceDB) and knowledge_server.py: used by Sentiment Agent (P8-E4)
- Chunking config (DJ-030) and embedding model (DJ-031): Sentiment Agent inherits Phase 7 production configuration
- RAG pattern (retrieve_context_node, use_rag): Risk and Macro agents will use this pattern if RAG enhancement is added in Phase 9

**Depends on Phase 5:**
- `verify_ensemble()` continues to work on Phase 8 EnsembleOutput (new agents' analyses are verified if they contain numerical claims)
- HR/GR metrics are the verification targets for the Sentiment Agent's RAG-based claims

**Depends on Phase 6:**
- LangFuse tracing extended to new agents; each agent run creates a separate span
- `get_tracer()` is passed to all new agent entry points for observability continuity

**Phase 9 (Collective Decision Engine) depends on Phase 8:**
- The Contrarian Agent's `ContrarianAnalysis` output is the input to Phase 9's debate mechanism
- The pairwise diversity matrix from Phase 8 is the baseline for Phase 9's aggregation experiments
- The marginal contribution results inform which agents to include in Phase 9's ablation studies
- Performance-weighted voting (Phase 9) requires per-agent decision histories established in Phase 8

**Phase 10 (Evaluation & Backtesting) depends on Phase 8:**
- The full 5-agent ensemble is the system being evaluated in walk-forward backtesting
- Agent-level decision histories (Phase 8) feed into Phase 10's calibration analysis
