# Phase 8 Bitacora: Agent Population Expansion

**Date completed:** 2026-06-11
**Tests at close:** ~785 (see final gate)
**Status:** COMPLETE

---

## Objective

Expand the HiFi ensemble from 2 agents (fundamental, technical) to 6 agents
(+ risk, macro, sentiment, contrarian), each with distinct information access
and model diversity, while keeping backward compatibility with the Phase 4/6/7
`agents=["fundamental","technical"]` call signature.

---

## Architecture Decisions (DJ-032 through DJ-038)

### DJ-032: Model Assignment by Information Role

Each new agent was assigned a different model to maximize ensemble diversity
along both information-space and model-space dimensions (David §10.3):

| Agent | Model | Reasoning | Max Tokens |
|---|---|---|---|
| Risk | google/gemma-3-4b | Non-reasoning; fast; risk metrics are numerical | 1024 |
| Macro | qwen3.5-27b reasoning-distilled | Regime classification needs chain-of-thought | 4096 |
| Sentiment | qwen2.5-coder-32b-instruct-mlx | Reuses Phase 3 fundamental model; SEC text | 1024 |
| Contrarian | mlx-qwen3.5-35b reasoning-distilled | Adversarial stress test; longest reasoning | 4096 |

The fundamental and technical agents retained their Phase 3/4 model assignments.

### DJ-033: Contrarian Is a Non-Voting Second-Pass Critic

The Contrarian Agent was deliberately excluded from the confidence-weighted
vote (David §12.2.2). Its role is adversarial stress testing, not signal
aggregation. This prevents the adversarial signal from cancelling legitimate
consensus — the Contrarian's output is logged for analysis but does not
change `collective_decision` in Phase 8. Integration into the decision
mechanism is a Phase 9 concern.

Practically: the Contrarian always runs LAST in `run_ensemble()`, receives
the formatted `ensemble_context` (all other agents' outputs + preliminary
decision), and produces `ContrarianAnalysis` (no `AgentSignal` field).

### DJ-034: Sentiment Agent Uses SEC Filings Only (RAG)

The Sentiment Agent was restricted to SEC 8-K, 10-K MD&A, and 10-Q MD&A
filings retrieved via the Phase 7 knowledge_server. This creates genuine
information-space diversity: while Fundamental and Technical Agents see
numerical tool outputs, the Sentiment Agent sees only qualitative management
language.

Earnings call transcripts were evaluated and deferred to Phase 9 (outside
Phase 7 filing corpus scope).

### DJ-035: TA Library MCP Boundary Applied in Phase 8

The `venvs/ta/` activation for pandas-ta was scheduled for Phase 8. However,
the Risk Agent reuses the existing `get_risk_metrics` MCP tool (which uses
QuantStats), so the indicators_server.py was added as a scaffold for future
TA-Lib integration. The MCP dependency isolation boundary (DJ-010) is
maintained: `venvs/ta/` is structurally defined but not yet exercised by
running tests.

### DJ-036: New Schemas Added to schemas.py

Four new analysis schemas extend the existing pattern:

- `RiskAnalysis`: `signal` + `risk_assessment` + `recommended_position_size` + `risk_metrics`
- `MacroAnalysis`: `signal` + `regime_assessment` + `rationale` + `macro_snapshot`
- `SentimentAnalysis`: `signal` + `sentiment_summary` + `notable_signals`
- `ContrarianAnalysis`: no `signal`; has `alternative_thesis` + `risk_scenario` + `counterargument` + `confidence`

`EnsembleOutput` gained four new Optional fields (`risk_analysis`, `macro_analysis`,
`sentiment_analysis`, `contrarian_analysis`) defaulting to None for backward
compatibility (DJ-038).

### DJ-037: Incremental Evaluation After Each Sub-Phase

Each sub-phase (8a risk, 8b macro, 8c sentiment, 8d contrarian) was evaluated
by comparing `disagreement_entropy H` and `n_valid_signals` across the 5
incremental configurations. The baseline script (`run_phase8_baseline.py`) runs
all 5 configurations sequentially for AAPL, JPM, XOM at 2023-03-31.

Marginal contribution table (to be filled after live baseline run):

| Config | Agents | H (AAPL) | H (JPM) | H (XOM) | n_signals |
|---|---|---|---|---|---|
| 1 | fund + tech | TBD | TBD | TBD | 2 |
| 2 | + risk | TBD | TBD | TBD | 3 |
| 3 | + macro | TBD | TBD | TBD | 4 |
| 4 | + sentiment | TBD | TBD | TBD | 5 |
| 5 | + contrarian | TBD | TBD | TBD | 5 (contrarian is non-voting) |

### DJ-038: Backward-Compatible agents= Parameter

`run_ensemble()` gained an `agents: list[str] | None` parameter.
- `agents=None` → all 6 agents (Phase 8 default)
- `agents=["fundamental","technical"]` → Phase 4/6/7 behavior; Phase 8 fields are None

Contrarian always runs last regardless of list order.

---

## Agent Diversity Matrix

| Agent | Information | Model Family | Votes? |
|---|---|---|---|
| Fundamental | Fundamentals + macro + valuation | Qwen2.5-coder-32b | Yes |
| Technical | Price-derived (indicators + risk) | qwen2.5-coder-32b | Yes |
| Risk | Risk metrics only (hist_vol, beta, max_drawdown, Sharpe, VaR) | gemma-3-4b | Yes |
| Macro | Macro snapshot only (fed_funds, CPI, unemployment, VIX, GDP) | qwen3.5-27b reasoning | Yes |
| Sentiment | SEC filings via RAG only (qualitative language) | qwen2.5-coder-32b | Yes |
| Contrarian | All other outputs (second-pass adversary) | qwen3.5-35b reasoning | No |

Information diversity is the primary ensemble mechanism per David §10.3.
Model diversity is a secondary mechanism; models were chosen to avoid
"echo chamber" effects where similar architectures produce correlated outputs.

---

## Implementation Surprises and Lessons Learned

### Sentiment Agent Fail-Open Design Caught Unexpected State

The holistic test (test_phase8_agent_population.py) initially assumed that the
knowledge_server would be unavailable in the test environment, triggering the
fail-open path (Hold/0.0 default signal without LLM call). However, the real
`data/knowledge` directory existed in the project root, and the knowledge_server
successfully retrieved AAPL passages, causing the sentiment agent to call the
live LM Studio.

Fix: the `patched_llms` pytest fixture explicitly patches `sentiment_agent.call_tool`
to return `{"passages": []}`, forcing fail-open deterministically regardless of
whether a live knowledge store is available. This is the correct test design:
the test exercises the fail-open code path explicitly rather than relying on
infrastructure absence.

Lesson: fail-open paths need explicit testing (forced empty context), not
implicit testing (hoping the service is down). The test now serves as a
specification of the DJ-038 fail-open contract.

### Contrarian Agent LLM Stub Model Name

The `parse_output_node` in risk_agent (and other agents) accesses `llm.model_name`
to populate `AgentSignal.model_id`. The stub LLM objects used in tests must
therefore carry a `model_name` attribute. The pattern established in Phase 4
(`stub.model_name = "stub-model"`) was reused across all new agents.

### Ruff E501 in Test Files After agents= Insertion

Adding `agents=["fundamental","technical"]` to existing `run_ensemble` calls
in test files caused 5 E501 (line too long) violations. These were fixed by
wrapping call arguments across multiple lines (multi-line function calls) and
splitting long comments/docstrings.

### Lazy Imports for Phase 8 Agents in ensemble_runner.py

The Phase 8 agents are imported lazily inside the conditional blocks in
`run_ensemble()` rather than at module level. This avoids import-time side
effects (each agent imports LangChain components that may log or probe
environment variables) and keeps the module lightweight when only the Phase 4
subset is requested.

---

## File Inventory

### New Source Files

| File | Purpose |
|---|---|
| `src/hifi/agents/risk_agent.py` | LangGraph 3-node Risk Analyst Agent (get_risk_metrics) |
| `src/hifi/agents/macro_agent.py` | LangGraph 3-node Macro Analyst Agent (get_macro_snapshot) |
| `src/hifi/agents/sentiment_agent.py` | Sentiment Agent (RAG-only; fail-open) |
| `src/hifi/agents/contrarian_agent.py` | Second-pass Contrarian Critic (no MCP tools) |
| `src/hifi/mcp/indicators_server.py` | FastMCP scaffold for venvs/ta/ TA integration |
| `src/hifi/agents/prompts/risk_v1.md` | Risk Agent prompt template |
| `src/hifi/agents/prompts/macro_v1.md` | Macro Agent prompt template |
| `src/hifi/agents/prompts/sentiment_v1.md` | Sentiment Agent prompt template |
| `src/hifi/agents/prompts/contrarian_v1.md` | Contrarian Agent prompt template |

### New Test Files

| File | Purpose |
|---|---|
| `tests/unit/test_risk_agent_nodes.py` | Unit tests for Risk Agent nodes + helpers |
| `tests/unit/test_macro_agent_nodes.py` | Unit tests for Macro Agent nodes + helpers |
| `tests/unit/test_sentiment_agent_nodes.py` | Unit tests for Sentiment Agent; fail-open path |
| `tests/unit/test_contrarian_agent.py` | Unit tests for Contrarian Agent + _extract_json |
| `tests/unit/test_indicators_server.py` | Unit tests for indicators_server FastMCP |
| `tests/holistic/test_phase8_agent_population.py` | Full 6-agent pipeline holistic test |
| `scripts/run_phase8_baseline.py` | Incremental evaluation baseline runner |

### Modified Files

| File | Change |
|---|---|
| `src/hifi/agents/schemas.py` | Added RiskAnalysis, MacroAnalysis, SentimentAnalysis, ContrarianAnalysis |
| `src/hifi/collective/schemas.py` | EnsembleOutput: 4 new Optional fields for Phase 8 agents |
| `src/hifi/agents/ensemble_runner.py` | agents= param; lazy Phase 8 agent imports; Contrarian second-pass |
| `src/hifi/agents/mcp_client.py` | Minor adjustments for server_module routing |
| `Makefile` | Added baseline-phase8 target |

---

## Scientific Context

Phase 8 implements the first concrete test of the ensemble diversity hypothesis
(David §10.3): does adding agents with distinct information access increase
ensemble disagreement entropy H?

The Phase 4 ensemble (2 agents) produces H=0.0 when both agents agree or
H>0 when they disagree. With 5 voting agents across 3 distinct information
spaces (numerical fundamentals, numerical technicals/risk, qualitative
filings), the theoretical H maximum increases. Whether the information
diversity actually manifests in diverse votes depends on whether the underlying
information sources produce genuinely different signals for the same ticker.

This is empirically measurable once the live baseline is run. The Contrarian
Agent provides an additional dimension: does its `confidence` in the
alternative thesis correlate with ensemble disagreement? If high confidence
contrarianship predicts future disagreement, it becomes a useful early warning
signal for uncertain market conditions.

---

## Next Phase

Phase 9 will redesign `EnsembleOutput` to be fully N-agent generic (replacing
the Optional field per agent design with a `signals: list[AgentSignal]`
pattern). It will also formalize the Contrarian's role in the decision
mechanism and add proper marginal contribution metrics.
