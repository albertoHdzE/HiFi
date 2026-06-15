# Phase 12: GraphRAG + Structured Debate — Context and Pre-Phase Decisions

**Gathered:** 2026-06-13
**Status:** Ready for planning

---

## Phase Boundary

Phase 12 delivers two independent mechanisms that extend the system's collective
intelligence capacity:

1. **GraphRAG** — structural/relational retrieval extending the Phase 7 dense
   RAG pipeline. Answers OQ-K02 (does GraphRAG improve over plain RAG?) and
   records DJ-016 with measured evidence.

2. **Structured Debate** — adversarial deliberation between agents before final
   collective vote. Tests the David's hypothesis (SS12.2.4) that minority
   opinions improve decision quality when they carry valid private information.
   Addresses the herding risk identified in SS5.6.3.

Additionally, Phase 12 resolves two open items from Phase 11:

3. **technical_v1 compliance fix** — the adapter's GR=0.000 is a compliance:domain
   ratio problem (0.19% compliance examples), not a training failure. Must be
   resolved before debate evaluation is interpretable.

4. **Multi-date diversity measurement** — Phase 11's OQ-M02 was vacuously
   answered (zero disagreement on a single date). Phase 12 runs 10 quarterly
   dates to produce real evidence.

Explicitly OUT of scope:
- Fine-tuning Risk/Macro/Sentiment agents (requires verification layer extension
  first — verify_agent() only supports FundamentalAnalysis | TechnicalAnalysis)
- Adaptive aggregation (SS12.2.5) — requires substantially more labeled data
- Agent memory (SS10.4) — Phase 13
- Paper trading — Phase 14

Explicitly IN scope as a preparatory artifact:
- Design document for Sentiment Agent fine-tuning labels (to inform Phase 13
  decision, not executed in Phase 12)
- Verification layer gap analysis for Phase 8 agents (what must extend before
  Risk/Macro/Sentiment can be fine-tuned)

---

## Evidence Base for Decisions

### Phase 11 Training Results (Positive)

| Metric | Value | Implication |
|---|---|---|
| Rank sweep convergence | All 4 ranks quality_ok=True | Pipeline proven |
| Training examples | 26,433 per agent | 55x above 400 minimum |
| fundamental_v1 GR | 1.000 maintained | Fine-tuning preserves quality when compliance ratio adequate |
| Adapter quality checks | Both PASS | Adapters are structurally healthy |

### Phase 11 Evaluation Results (Mixed)

| Metric | Value | Root cause |
|---|---|---|
| technical_v1 GR | 1.000 -> 0.000 | Compliance:domain ratio = 0.19% (50/26,433) |
| Diversity (pairwise) | 0.000 both conditions | Uninformative: unanimous votes on single date |
| Base technical GR | 1.000 (vs Phase 5: 0.667) | Model config change between Phase 5 and 11 |

### Phase 8 Model Diversity Matrix (DJ-032)

| Agent | Model | Family | Fine-Tuned? |
|---|---|---|---|
| Fundamental | qwen2.5-coder-32b | Qwen 2.5 | Yes (fundamental_v1) |
| Technical | qwen2.5-coder-32b | Qwen 2.5 | Yes (technical_v1, not deployed) |
| Risk | gemma-3-4b | Google Gemma | No |
| Macro | qwen3.5-27b reasoning | Qwen 3.5 | No |
| Sentiment | qwen2.5-coder-32b | Qwen 2.5 | No |
| Contrarian | qwen3.5-35b reasoning | Qwen 3.5 | No (non-voting) |

Three agents share qwen2.5-coder-32b: Fundamental, Technical, Sentiment.
Risk and Macro already have architectural diversity from different model families.

### Structural Constraint: Verification Layer

`verify_agent()` in `src/hifi/verification/verifier.py` accepts only
`FundamentalAnalysis | TechnicalAnalysis`. The Phase 5 verification layer has
no path for Risk, Macro, or Sentiment agents. HR/GR baselines cannot be
established without extending verify_agent() to their schemas. This is the
structural blocker for fine-tuning Phase 8 agents (David SS9.4 requires
"demonstrably outperforms base" before deployment).

### Phase 10 Bootstrap Accuracy

| Agent | Accuracy | Signal type |
|---|---|---|
| Risk | 0.349 | Sharpe/drawdown thresholds |
| Technical | 0.254 | RSI thresholds |
| Fundamental | 0.079 | Hold bias |
| Macro | 0.079 | Hold bias |
| Sentiment | 0.000 | 3 records only |

---

## Pre-Phase Decisions (DJ-061 through DJ-069)

### DJ-061: technical_v1 Compliance Ratio Fix

The technical_v1 adapter collapsed GR from 1.000 to 0.000. The root cause is
identified in the Phase 11 bitacora: the training set contained ~50 compliance
examples against 26,433 domain examples (0.19% compliance). The domain signal
overwhelmed the format prior.

The fundamental_v1 adapter (same rank, same iterations, same base model)
preserved GR=1.000, proving the approach works with adequate compliance ratio.

Fix strategy:
1. Generate additional compliance examples from Phase 4 ensemble fixture
   (multi-ticker format, both agents) and Phase 5 verification fixture
   (verified outputs with HR=0.000). Target: >= 200 compliance examples
   (up from ~50), achieving ~0.75% ratio.
2. Re-train at 500 iterations (half of original 1000) to reduce overfitting
   to the domain distribution.
3. Re-evaluate with three-tier protocol (DJ-058).
4. Deploy if GR >= 0.720. If GR still < 0.720 after two mitigation attempts,
   formally abandon technical_v1 and document as a limitation. Debate
   evaluation uses base model for Technical Agent in that case.

### DJ-062: GraphRAG Library — NetworkX + LanceDB Extension

Options evaluated:

| Library | Pros | Cons |
|---|---|---|
| Microsoft GraphRAG | Full pipeline, community detection | Cloud-oriented, heavyweight, Azure assumptions |
| LlamaIndex PropertyGraphIndex | Well-maintained, LLM integration | Large dependency, abstracts away graph structure |
| Neo4j + Cypher | Industry standard, powerful queries | External service, heavyweight for 12-node graph |
| NetworkX + custom | Pure Python, zero new deps, full control | No built-in LLM entity extraction |

Decision: **NetworkX (already available in Python ecosystem) + LanceDB (already
deployed for Phase 7 dense retrieval)**. The Phase 12 graph schema is small
(~12 nodes, ~40 edges). A custom implementation is simpler than any framework
abstraction at this scale. The graph is serialized as JSON at
`data/knowledge_graph/financial_graph.json` (NetworkX JSON serialization).

No new venv required — NetworkX is pure Python and compatible with the main
uv project env.

Rationale from governing principles:
- "The simplest version that produces evidence" (Protocol SS3): a custom
  NetworkX graph is the minimum viable GraphRAG implementation that answers
  OQ-K02.
- "Do not add complexity without evidence" (DJ-016 original framing): if
  this simple implementation doesn't show Precision@k improvement, a heavier
  framework would not either.

### DJ-063: Graph Schema — Tight Scope for Phase 12

**Node types:**

| Type | Fields | Source |
|---|---|---|
| Company | ticker, name, sector, industry | yfinance metadata (deterministic) |
| Sector | name, sector_code | yfinance sector field |
| MacroFactor | name, series_id | Phase 1 FRED data (VIX, FFR, CPI) |

**Edge types:**

| Type | From -> To | Source | Symmetric? |
|---|---|---|---|
| BELONGS_TO | Company -> Sector | yfinance metadata | No |
| COMPETES_WITH | Company <-> Company | Static curated seed | Yes |
| SENSITIVE_TO | Sector -> MacroFactor | Domain knowledge | No |

**Scope:** 3 evaluation tickers (AAPL, JPM, XOM) + their sector peers from
the 15-ticker Phase 10 universe:

- AAPL sector peers: MSFT, GOOGL, NVDA, AMZN, META (Technology)
- JPM sector peers: BAC, GS (Financials)
- XOM sector peers: CVX (Energy)

Total: ~10-12 company nodes, 3-5 sector nodes, 3 macro nodes, ~30-40 edges.

Competitor edges (curated seed):
- Technology: AAPL <-> MSFT, AAPL <-> GOOGL, MSFT <-> GOOGL, NVDA <-> AMZN
- Financials: JPM <-> BAC, JPM <-> GS, BAC <-> GS
- Energy: XOM <-> CVX

Macro sensitivity edges:
- Technology -> FFR (rate-sensitive growth stocks)
- Financials -> FFR (net interest margin driven)
- Energy -> VIX (commodity volatility proxy)
- All sectors -> CPI (inflation affects all)

Full LLM-extracted competitor relationships from SEC filings deferred to
Phase 13 (the David marks this as an open question: "Manual vs. automatic
knowledge graph construction?" SS11.3).

### DJ-064: GraphRAG Query Expansion Mechanism

Given a query for ticker T:
1. Look up T in the graph.
2. Expand to 1-hop neighbors: T's sector, T's direct competitors.
3. Expand to 2-hop: sector peers (other companies in same sector), sector's
   sensitive macro factors.
4. Collect the expanded entity set: {T, competitor_1, ..., peer_1, ...}.
5. Pass expanded ticker set as a filter to LanceDB dense search:
   `WHERE ticker IN (T, competitor_1, peer_1, ...)`.
6. Return top-k results by cosine similarity (same as Phase 7 RAG).

The expansion adds relational context (SEC filings from competitors and sector
peers are now in scope) without changing the retrieval mechanism (still dense
ANN search). The graph provides the "what else is relevant" signal; LanceDB
provides the "what is semantically similar" signal.

New class: `GraphRetriever` extends `KnowledgeRetriever` with the graph
expansion step. Same interface (`retrieve()` -> `list[DocumentChunk]`).

Measurement: Precision@k on the Phase 7 20-query evaluation set
(`tests/fixtures/retrieval/evaluation_queries.json`), comparing:
- Baseline: KnowledgeRetriever (Phase 7 dense RAG)
- Experimental: GraphRetriever (graph-expanded dense RAG)

### DJ-065: Structured Debate Protocol — Oxford 1-Round, 5 Voting Agents

Protocol design from David SS12.2.4 ("Structured Debate — Experimental"):

```
Phase 1: INDEPENDENT ANALYSIS (existing run_ensemble flow)
    All 5 voting agents run independently.
    Initial votes collected.

Phase 2: CHALLENGE (new)
    Identify minority agents: those whose vote differs from plurality.
    If no minority (unanimous): skip debate, proceed to vote.
    Each minority agent generates a "challenge" argument (max 150 words):
      - Given: its own analysis + the majority decision + majority confidence
      - Produce: structured counter-argument citing specific evidence

Phase 3: RESPONSE (new)
    Each majority agent sees the challenge transcript.
    Each majority agent generates a "response" (max 100 words):
      - Acknowledge or refute the challenge
      - Optionally revise its position

Phase 4: REVISION (new)
    All agents see the full debate transcript (challenges + responses).
    Each agent produces a revised AgentSignal:
      - Same schema as initial signal
      - decision, confidence, rationale may change
      - must reference debate evidence in rationale

Phase 5: FINAL VOTE (existing run_all_methods flow)
    run_all_methods() on revised signals.
    Contrarian runs last on final signals (unchanged role).
```

Integration: new `run_debate_ensemble()` function in `ensemble_runner.py`.
The existing `run_ensemble()` is unchanged — debate is additive, not modifying.

All 5 voting agents participate in debate because they have genuine
architectural diversity (gemma-3-4b, qwen3.5-27b, qwen2.5-coder-32b). A debate
between architecturally different models is closer to the David's vision of
collective intelligence from heterogeneous agents (SS5.2).

LLM routing: each agent uses its deployed model for debate turns (fine-tuned
if deployed, base otherwise). This is consistent with the agent's analytical
voice and enables the 2x2 factorial design (base/fine-tuned x no-debate/debate).

Debate termination: one round only for Phase 12. Multi-round debate is a
Phase 13 extension (requires convergence criteria, which we don't have data
to calibrate yet).

### DJ-066: Debate Transcript Storage — Dataset Family D

New Pydantic schemas in `src/hifi/collective/debate.py`:

```
DebateTurn:
    agent_type: str
    phase: Literal["challenge", "response", "revision"]
    argument: str
    revised_decision: str | None  (only in revision phase)
    revised_confidence: float | None
    model_id: str

DebateTranscript:
    ticker: str
    as_of_date: str
    initial_signals: list[AgentSignal]
    minority_agents: list[str]
    majority_decision: str
    challenge_turns: list[DebateTurn]
    response_turns: list[DebateTurn]
    revised_signals: list[AgentSignal]
    vote_delta: Literal["converged", "diverged", "unchanged"]
    n_agents_changed_vote: int
    debate_skipped: bool  (True when initial vote is unanimous)
```

`EnsembleOutput` gains an optional field:
`debate_transcript: DebateTranscript | None = None`

Transcripts saved to `data/interactions/` (Dataset Family D, David SS8.6).
This is the first population of Dataset Family D artifacts.

### DJ-067: Multi-Date Evaluation Protocol (2x2 Factorial Design)

Phase 12 runs a 2x2 factorial experiment across 10 quarterly dates:

```
                   No debate          With debate
Base models           A                   C
Fine-tuned            B                   D
```

- **Condition A:** Phase 9 collective baseline (base models, no debate)
- **Condition B:** Phase 11 fine-tuned evaluation (fine-tuned, no debate)
- **Condition C:** New (base models + Oxford debate)
- **Condition D:** New (fine-tuned models + Oxford debate)

The interaction effect (D-B) - (C-A) answers: "Does debate produce greater
benefit when agents are heterogeneously fine-tuned?"

**Evaluation dates (quarterly):**
2022-03-31, 2022-06-30, 2022-09-30, 2022-12-31,
2023-03-31, 2023-06-30, 2023-09-30, 2023-12-31,
2024-03-31, 2024-06-30

**Tickers:** AAPL, JPM, XOM (evaluation universe)

**Total runs:** 10 dates x 3 tickers x 4 conditions = 120 ensemble runs

**Metrics per run:**
- Tier 1: HR, GR per agent (Phase 5 verifier)
- Tier 2: 60-day forward accuracy per method (Phase 10 labeler)
- Tier 3: pairwise_diversity, disagreement_entropy, herding_coefficient

**Metrics across runs (publishable):**
- Debate herding index: distribution of vote_delta across 30 ticker-dates
- Diversity change: mean pairwise_diversity(debate) vs mean pairwise_diversity(no debate)
- Accuracy change: mean method_accuracy(debate) vs mean method_accuracy(no debate)
- Interaction effect: does fine-tuning amplify or dampen debate benefit?
- GraphRAG effect: Precision@k(GraphRAG) vs Precision@k(RAG) on eval queries

**Hardware estimate:** ~5 min per run on M3 Ultra = ~10 hours sequential.
Checkpointed (one JSON per run). Resumable from interruption.

### DJ-068: GraphRAG as Drop-In Path Extension

`run_ensemble()` gains `use_graphrag: bool = False` parameter.

When `use_graphrag=True`:
- The existing `retrieve_context_node` in fundamental_agent and technical_agent
  is replaced with `graph_retrieve_context_node` (same interface, graph-expanded
  implementation via GraphRetriever).
- The `knowledge_server.py` MCP tool is extended with a `graph_search` tool
  alongside the existing `search` tool.

`use_rag` and `use_graphrag` are mutually exclusive:
`assert not (use_rag and use_graphrag)`. This keeps A/B testing explicit.

### DJ-069: Fine-Tuning Remaining Agents — Staged Decision

The David's Platonic goal (SS9.4) requires agent-specific fine-tuning for
all agents. The Phase 15 ablation study ("Remove fine-tuning") requires
fine-tuned agents to exist. Fine-tuning is not optional in the long term.

However, the Phase 12 evidence is insufficient to execute now:

1. **Verification layer blocker:** verify_agent() must be extended to support
   RiskAnalysis, MacroAnalysis, SentimentAnalysis before HR/GR baselines can
   be established for those agents (Protocol SS1: "every layer earns its place
   with a measurement").

2. **Model architecture diversity:** Risk (gemma-3-4b) and Macro (qwen3.5-27b)
   use different model families. LoRA training for these architectures requires
   separate investigation (different adapter configurations, different training
   dynamics for reasoning-distilled models).

3. **Sentiment Agent is the highest-priority candidate:** it shares
   qwen2.5-coder-32b with Fundamental and Technical. Fine-tuning is the
   strongest diversity mechanism for reducing rho within this trio. Training
   label design (sentiment-specific, e.g., MD&A management tone) is a Phase 12
   deliverable (design document only, not execution).

**Phase 12 delivers:**
- technical_v1 compliance fix and redeployment decision
- Multi-date diversity measurement (10 dates) to properly assess OQ-M02
- Verification layer gap analysis for Phase 8 agents
- Sentiment Agent training label design document

**Phase 13 receives:**
- Verification layer extension for Phase 8 agent schemas
- Sentiment Agent fine-tuning (if Phase 12 diversity evidence supports it)
- Risk/Macro fine-tuning feasibility study (different architectures)

This staging follows the Protocol's principle: build the measurement first,
then the intervention.

---

## Canonical References

Downstream agents MUST read these before planning or implementing.

### Core Specification
- `doc/HIFI_DAVID.md` SS11.3 — GraphRAG (graph schema, RAG vs GraphRAG question)
- `doc/HIFI_DAVID.md` SS12.2.4 — Structured Debate (Oxford format, herding risk)
- `doc/HIFI_DAVID.md` SS5.3 — Ensemble Learning (rho formula, diversity theorem)
- `doc/HIFI_DAVID.md` SS5.6 — Complexity metrics (disagreement entropy, herding kappa, consensus stability)
- `doc/HIFI_DAVID.md` SS9.4 — Fine-Tuning Strategy (agent-specific, critical requirement)
- `doc/HIFI_DAVID.md` SS10.3 — Diversity Requirements (5 dimensions, 2-dimension minimum)
- `doc/HIFI_DAVID.md` SS15 — Ablation studies (require fine-tuned + GraphRAG to exist)
- `doc/HIFI_PROTOCOL_V1.md` SS Phase 12 — Deliverables and success criteria

### Phase Context
- `plans/PHASE_11_CONTEXT.md` — DJ-053 through DJ-060 rationale
- `plans/PHASE_11_PLAN.md` — Phase 11 epic/ticket structure (pattern to follow)
- `doc/bitacora/PHASE_11_FINE_TUNING.md` — Training results, compliance failure root cause
- `doc/bitacora/PHASE_08_AGENT_POPULATION.md` — DJ-032 model diversity matrix
- `doc/bitacora/PHASE_10_EVALUATION.md` — Bootstrap accuracy, tear sheet infrastructure
- `doc/bitacora/PHASE_09_COLLECTIVE_ENGINE.md` — Rolling metrics (kappa, S), method comparison

### Existing Infrastructure
- `src/hifi/knowledge/` — Phase 7 RAG (vector_store.py, retrieval.py, schemas.py)
- `src/hifi/agents/ensemble_runner.py` — run_ensemble() entry point
- `src/hifi/collective/voting.py` — run_all_methods(), confidence_weighted_vote()
- `src/hifi/verification/verifier.py` — verify_agent(), verify_ensemble()
- `src/hifi/models/` — training_data.py, fine_tune.py (Phase 11 infrastructure)
- `tests/fixtures/retrieval/evaluation_queries.json` — 20-query Precision@k set
- `tests/fixtures/baseline/phase11_evaluation.json` — Three-tier eval results

### Agent Prompts (debate must extend these)
- `src/hifi/agents/prompts/fundamental_v1.md`
- `src/hifi/agents/prompts/technical_v1.md`
- `src/hifi/agents/prompts/risk_v1.md`
- `src/hifi/agents/prompts/macro_v1.md`
- `src/hifi/agents/prompts/sentiment_v1.md`
- `src/hifi/agents/prompts/contrarian_v1.md`

---

## Deferred Ideas

The following were raised during Phase 12 scoping and explicitly deferred:

- **Sentiment Agent fine-tuning:** Training label design produced as Phase 12
  artifact; execution deferred to Phase 13 pending verification layer extension
  and multi-date diversity evidence.

- **Risk/Macro fine-tuning:** Different model architectures (gemma-3-4b,
  qwen3.5-27b) require separate LoRA investigation. Phase 13+.

- **Multi-round debate:** Phase 12 implements one round. Multi-round requires
  convergence criteria calibrated from Phase 12 transcript data. Phase 13.

- **Adaptive aggregation (SS12.2.5):** Learned aggregation function. Requires
  more labeled data than Phase 12 produces. Phase 13+.

- **LLM-extracted knowledge graph:** Automatic competitor extraction from
  MD&A sections. David SS11.3 open question on manual vs. automatic
  construction. Phase 13.

- **Herding/VIX correlation analysis:** kappa vs VIXCLS over 2018-2022.
  Data exists. Analysis deferred from Phase 11. Can be included as a Phase 12
  notebook analysis if time permits.
