# Phase 8 Context: Full Agent Population

**Generated:** 2026-06-11 (--auto mode)
**Status:** Ready for planning

---

## Domain Boundary

Expand from 2 agents (Fundamental + Technical) to 4 agents by adding Risk, Macro, Sentiment, and Contrarian agents incrementally. Each sub-phase (8a-8d) adds one agent, measures diversity and marginal contribution, and stops to record results before adding the next.

This phase proves whether agent diversity actually helps — the marginal contribution curve is the central empirical result.

---

## Decisions

### DJ-032: Model Assignments (Diversity Requirement §10.3)

Agents must differ on >= 2 diversity dimensions. Current assignments:

| Agent | Model | Info Access | Sub-phase |
|---|---|---|---|
| Fundamental | qwen2.5-coder-32b-instruct-mlx | Financials snapshot | Existing |
| Technical | mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled | Price-derived only | Existing |
| Risk (new) | google/gemma-3-4b | Risk metrics only | 8a |
| Macro (new) | qwen3.5-27b-claude-4.6-opus-reasoning-distilled-qx64-hi-mlx | Macro indicators only | 8b |
| Sentiment (new) | qwen2.5-coder-32b-instruct-mlx | SEC filings via RAG only | 8c |
| Contrarian (new) | mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled | All agent outputs (2nd pass) | 8d |

Diversity dimensions per new agent:
- Risk: model (Gemma vs Qwen) + info access (risk metrics only)
- Macro: model (27B distilled vs 32B coder) + info access (macro indicators only)
- Sentiment: info access (RAG only) + role (qualitative/narrative) — same model family as Fundamental but different information and role
- Contrarian: role (adversarial, no vote) + processing order (2nd pass)

Env vars: `HIFI_RISK_MODEL`, `HIFI_MACRO_MODEL`, `HIFI_SENTIMENT_MODEL`, `HIFI_CONTRARIAN_MODEL`.

### DJ-033: Contrarian Agent Design — Second-Pass

The Contrarian Agent is a **second-pass agent**: it receives the outputs of agents 8a-8c before formulating its counter-thesis. This makes it an informed stress-tester rather than random noise.

Execution order:
```
Risk → Macro → Sentiment → [vote aggregation] → Contrarian(sees consensus) → final ensemble
```

The Contrarian does NOT produce a Buy/Hold/Sell vote. Its output is:
- `alternative_thesis` — what could go wrong with the consensus
- `risk_scenario` — specific adverse scenario with estimated probability
- `counterargument` — structured argument against the dominant position
- `confidence` — confidence in the contrarian view [0, 1]

The Contrarian's output is logged and displayed but does NOT change the collective_decision from Phase 4's confidence_weighted_vote. In Phase 9, the Contrarian's output will be integrated into the collective decision mechanism.

### DJ-034: Earnings Transcripts Source for Sentiment Agent

Earnings call transcripts are NOT added in Phase 8. The Sentiment Agent (8c) uses:
- SEC 8-K filings (earnings announcements, already indexed in knowledge store)
- 10-K MD&A sections (management discussion, already indexed)
- 10-Q MD&A sections (quarterly, already indexed)

This is sufficient for a qualitative/narrative-focused agent in Phase 8. Actual earnings call transcripts (audio/text) are deferred to Phase 9 per DJ-028.

The Sentiment Agent's RAG queries are tuned for qualitative language (management tone, forward guidance, risk language) rather than numerical data.

### DJ-035: venvs/ta/ Activation in Phase 8

The `venvs/ta/` scaffold (created in Phase 2, pinned at pandas 1.5.3 + pandas-ta 0.3.14b0) is activated in Phase 8 as Epic 8a (prerequisite for Risk Agent).

Deliverable: `src/hifi/mcp/indicators_server.py` — a FastMCP server that runs inside `venvs/ta/` and exposes extended TA indicators via MCP stdio. The main process calls it via the existing `call_tool()` mechanism with `server_module="hifi.mcp.indicators_server"` but the server binary is `venvs/ta/bin/python -m hifi.mcp.indicators_server`.

Phase 8 does NOT require the Risk Agent to use venvs/ta/ — the existing `get_risk_metrics` MCP tool (financial_server.py) already covers the Risk Agent's information needs (hist_vol, beta, max_drawdown, Sharpe, VaR). venvs/ta/ activation unlocks extended indicators for future phases. If activation is complex, it can be a standalone epic before 8a.

### DJ-036: New Agent Schemas

Four new schemas added to `src/hifi/agents/schemas.py`:

```python
class RiskAnalysis(BaseModel):
    signal: AgentSignal
    risk_assessment: str          # structured risk profile
    recommended_position_size: float | None  # fraction of portfolio [0,1]
    prompt_version: str

class MacroAnalysis(BaseModel):
    signal: AgentSignal
    regime_assessment: str        # macro regime classification
    rationale: str
    prompt_version: str

class SentimentAnalysis(BaseModel):
    signal: AgentSignal
    sentiment_summary: str        # qualitative assessment
    notable_signals: list[str]    # specific statements/events flagged
    prompt_version: str

class ContrarianAnalysis(BaseModel):
    # No AgentSignal — Contrarian does not vote
    alternative_thesis: str
    risk_scenario: str
    counterargument: str
    confidence: float             # [0, 1]
    prompt_version: str
```

`EnsembleOutput` in `src/hifi/collective/schemas.py` extended to include optional fields for each new agent's analysis.

### DJ-037: Evaluation Cadence

After each sub-phase (8a, 8b, 8c, 8d), run the standard eval on AAPL/JPM/XOM at 2023-03-31 and measure:

1. Directional agreement with Phase 5 baseline (as proxy for accuracy — no ground truth yet)
2. Disagreement entropy H (from Phase 4 metrics)
3. Pairwise correlation matrix across all agents
4. Marginal contribution of new agent (delta in ensemble confidence + delta in H)

Results recorded in `doc/bitacora/PHASE_08_AGENT_POPULATION.md` as a marginal contribution table.

### DJ-038: Ensemble Runner Extension

`run_ensemble()` in `ensemble_runner.py` extended to accept new agents via a registry pattern:

```python
def run_ensemble(
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    use_rag: bool = False,
    agents: list[str] | None = None,  # NEW: ["fundamental", "technical", "risk", "macro", "sentiment", "contrarian"]
) -> EnsembleOutput:
```

Default `agents=None` means all enabled agents. This is backward-compatible — existing callers get the Phase 7 behavior if they don't pass `agents`.

Contrarian always runs last regardless of the `agents` list order.

### Execution Order (Sequential, Contrarian Last)

Phase 8 keeps sequential execution. No parallelism yet (Phase 9+ optimization).

```
load_snapshot → call_financial_mcp → [fundamental_agent, technical_agent, risk_agent, macro_agent, sentiment_agent] → vote() → contrarian_agent(sees_consensus) → EnsembleOutput
```

---

## Sub-Phase Epics (for Planning)

| Sub-phase | Epic | Deliverable |
|---|---|---|
| Pre-8a | venvs/ta/ activation | indicators_server.py in venvs/ta/ (if needed this phase) |
| 8a | Risk Agent | risk_agent.py + RiskAnalysis schema + prompts/risk_v1.md + eval |
| 8b | Macro Agent | macro_agent.py + MacroAnalysis schema + prompts/macro_v1.md + eval |
| 8c | Sentiment Agent | sentiment_agent.py + SentimentAnalysis schema + prompts/sentiment_v1.md + eval |
| 8d | Contrarian Agent | contrarian_agent.py + ContrarianAnalysis schema + prompts/contrarian_v1.md + eval |
| 8e | Phase baseline + bitacora | scripts/run_phase8_baseline.py + doc/bitacora/PHASE_08_AGENT_POPULATION.md |

---

## Reusable Assets (from Prior Phases)

- `src/hifi/agents/lm_client.py` — `make_llm()` works for any model in LM Studio
- `src/hifi/agents/mcp_client.py` — `call_tool()` used as-is for all new agents
- `src/hifi/agents/schemas.py` — `AgentSignal` reused in Risk/Macro/Sentiment
- `src/hifi/mcp/financial_server.py` — `get_risk_metrics` for Risk Agent; `get_macro_snapshot` for Macro Agent
- `src/hifi/mcp/knowledge_server.py` — `retrieve_context` for Sentiment Agent (RAG)
- `src/hifi/collective/metrics.py` — All diversity metrics already implemented
- `src/hifi/collective/voting.py` — `confidence_weighted_vote()` extended to N agents
- `tests/conftest.py` — `DeterministicEmbeddingModel` + synthetic fixture generators

---

## Canonical References

- `doc/HIFI_DAVID.md` §10.2 — All Agent Specifications (information access, output fields)
- `doc/HIFI_DAVID.md` §10.3 — Diversity Requirements (2-dimension minimum rule)
- `doc/HIFI_DAVID.md` §12.2.2 — Confidence-weighted vote (already implemented)
- `doc/HIFI_DAVID.md` §5.6 — Complexity metrics (H, D, κ, S)
- `doc/HIFI_PROTOCOL_V1.md` Phase 8 — Sub-phases 8a-8d, success criteria
- `plans/PHASE_07_PLAN.md` — RAG pattern (retrieve_context_node, use_rag)
- `src/hifi/agents/technical_agent.py` — Reference implementation for new agents
- `src/hifi/agents/fundamental_agent.py` — Reference implementation (with RAG)
- `venvs/ta/requirements.txt` — Pinned pandas 1.5.3 + pandas-ta 0.3.14b0

---

## Open Questions (Deferred to Planning)

- OQ-P8-01: Should Risk Agent use get_risk_metrics directly, or does it also get fundamentals for context? (David §10.2 says "risk metrics only" — planner to decide what "only" means operationally)
- OQ-P8-02: venvs/ta/ activation complexity — if indicators_server.py needs significant work, defer to Phase 9 and use existing 6 indicators for Risk Agent in Phase 8
- OQ-P8-03: EnsembleOutput schema extension — how to represent N-agent results without breaking Phase 5-7 verification layer
- OQ-P8-04: Contrarian Agent context window — how much of the other agents' outputs to include (full rationale or just signal+confidence)

---

## Deferred Ideas

- Parallel agent execution (Phase 9+ optimization)
- Earnings call transcript acquisition (Phase 9, per DJ-028)
- Agent memory / calibration data (David §10.4 — Phase 10+)
- Performance-weighted voting using historical accuracy (Phase 9)
- Valuation Agent (not in Phase 8 protocol order; appears in David §10.2 but not in Phase 8 sub-phases)
