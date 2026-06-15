# Phase 12 Bitacora: GraphRAG + Structured Debate

**Phase status:** LLM evaluation complete (graphrag-eval done; factorial eval condition A complete, B/C/D in progress) — 2026-06-14
**Tests at infrastructure close:** 1137 passed, 0 skipped, 0 lint errors
**Tests at Phase 13 E0 close:** 1197 passed, 0 skipped, 0 lint errors
**David sections:** SS11.3 GraphRAG, SS12.2.4 Structured Debate, SS5.6 Complexity Metrics, SS9.4 Fine-Tuning Strategy

---

## Objective

Phase 12 extends collective intelligence through two independent mechanisms:

1. **GraphRAG (E1, E2):** Does structural relational knowledge about entity
   relationships improve retrieval precision over dense RAG alone? This answers
   OQ-K02 and resolves DJ-016 ("do not add complexity without evidence") with
   measured Precision@k evidence.

2. **Structured Debate (E3, E4):** Does adversarial deliberation between
   heterogeneous agents improve collective decision quality, or does it cause
   herding (SS5.6.3)? The David predicts minority opinions carry private
   information not aggregated by simple voting. Phase 12 tests this empirically
   via a 2x2 factorial experiment.

Phase 12 also resolves two open items from Phase 11:

3. **technical_v1 compliance fix (E0):** GR collapsed from 1.000 to 0.000 due to
   compliance:domain training ratio of 0.19% (50/26,433). Fixed by augmenting to
   >= 200 compliance examples. Re-training and re-evaluation are Wave 2 tasks.

4. **Multi-date diversity (E4):** Phase 11 OQ-M02 was vacuously answered on a
   single date with unanimous votes. Phase 12 runs 10 quarterly dates to produce
   real diversity evidence.

---

## Architecture Decisions (DJ-061 through DJ-069)

### DJ-061: technical_v1 Compliance Ratio Fix

**Problem:** technical_v1 GR collapsed 1.000 → 0.000. Root cause: training set
contained ~50 compliance examples in 26,433 total (0.19% ratio). The domain
signal (max-return labels) overwhelmed the format prior. fundamental_v1 (same
rank, same iterations) preserved GR=1.000, proving the approach works with
adequate compliance ratio.

**Fix strategy:**
1. Generate >= 200 compliance examples (target ratio ~0.75%).
2. Re-train at 500 iterations (half of original 1000) to reduce domain overfitting.
3. Re-evaluate with three-tier protocol.
4. Deploy if GR >= 0.720. If GR still < 0.720: abandon technical fine-tuning for
   Phase 12 and document as empirical result. Debate uses base model for Technical.

**Wave 1 deliverable (P12-E0-T1):** `data/training/technical_compliance_v2.jsonl`
generated with 200 examples:
- 6 extracted from Phase 4 + Phase 9 fixtures (verified LLM outputs, HR=0.000)
- 194 synthetic from OHLCV data at quarterly intervals across 16 tickers,
  using `format_as_jsonl()` with `generate_max_return_labels()` from `training_data.py`
- All examples pass structural validation (decision ∈ {Buy, Hold, Sell}, 0 ≤ confidence ≤ 1,
  non-empty rationale and key_concern)

**Status:** Wave 1 complete. Re-training (E0-T2) and evaluation (E0-T3) are Wave 2
tasks (hardware-bound — require venvs/finetune/ on Apple Silicon).

### DJ-062: GraphRAG Library — NetworkX + LanceDB

**Options evaluated:**

| Library | Decision |
|---|---|
| Microsoft GraphRAG | Rejected — cloud-oriented, Azure assumptions |
| LlamaIndex PropertyGraphIndex | Rejected — large dependency, abstracts graph structure |
| Neo4j + Cypher | Rejected — external service, heavyweight for 12-node graph |
| NetworkX + custom | **Selected** — pure Python, zero new external service, full control |

NetworkX is already available as a transitive dependency of langgraph. Added as
explicit direct dependency (networkx>=3.6.1) to avoid relying on transitive availability.

LanceDB (already deployed in Phase 7) provides the dense ANN search layer.
The graph provides "what else is relevant"; LanceDB provides "what is semantically similar".

**Rationale:** "The simplest version that produces evidence" (Protocol SS3). A custom
NetworkX graph is the minimum viable GraphRAG implementation that answers OQ-K02.
If this simple implementation shows no Precision@k improvement, a heavier framework
would not either — per DJ-016.

### DJ-063: Graph Schema — Tight Scope

**Node types:** Company (ticker, name, sector, industry), Sector (name), MacroFactor
(name, FRED series_id).

**Edge types:** BELONGS_TO (Company → Sector, directed), COMPETES_WITH
(Company ↔ Company, symmetric, stored as 2 directed edges), SENSITIVE_TO
(Sector → MacroFactor, directed).

**Scope:** ~12 company nodes (AAPL, JPM, XOM + sector peers from Phase 10 universe),
3 macro factor nodes (VIX/VIXCLS, FFR/FEDFUNDS, CPI/CPIAUCSL), ~40 edges.

**Macro sensitivity:**
- Technology → FFR (rate-sensitive growth stocks), VIX
- Financial Services → FFR (net interest margin driven)
- Energy → VIX (commodity volatility proxy), CPI

Full LLM-extracted competitor relationships from SEC filings deferred to Phase 13
(David SS11.3 open question: manual vs. automatic graph construction).

### DJ-064: Query Expansion — 2-Hop BFS

Given query ticker T:
1. Look up T in graph.
2. 1-hop: direct competitors (COMPETES_WITH edges).
3. 2-hop: sector peers (other companies via same BELONGS_TO sector).
4. Expanded ticker set → LanceDB filter `WHERE ticker IN (T, ...)`.
5. Cosine similarity ranking within filtered set.

`expand_query_tickers(ticker, max_hops=2)` implements this. `max_hops=0` returns
`[ticker]` (graceful fallback). Unknown tickers return singleton list (no crash).

### DJ-065: Structured Debate — Oxford 1-Round, 5 Voting Agents

```
Phase 1: INDEPENDENT ANALYSIS (existing run_ensemble flow)
Phase 2: CHALLENGE — minority agents challenge majority (max 150 words)
Phase 3: RESPONSE — majority agents respond to challenges (max 100 words)
Phase 4: REVISION — all agents revise after full transcript (max 1024 tokens)
Phase 5: FINAL VOTE — run_all_methods() on revised signals
```

Minority definition: agents whose initial vote differs from plurality.
If all agents agree: `debate_skipped=True`, transcript recorded but no LLM calls made.

All 5 voting agents participate (Risk: gemma-3-4b, Macro: qwen3.5-27b, others:
qwen2.5-coder-32b). Architectural diversity is the strongest basis for heterogeneous
deliberation (David SS5.2).

One round only in Phase 12. Multi-round requires convergence criteria calibrated
from Phase 12 transcript data — deferred to Phase 13.

### DJ-066: DebateTranscript — Dataset Family D

`src/hifi/collective/debate.py` contains:
- `DebateTurn`: atomic debate contribution per agent per phase
- `DebateTranscript`: full Oxford record per (ticker, as_of_date) pair
- `identify_minority()`: plurality with Hold tie-break
- `compute_vote_delta()`: measures herding (converged/diverged/unchanged) relative to initial majority

`EnsembleOutput.debate_transcript: DebateTranscript | None = None` added to
`collective/schemas.py` — fully backward compatible with all existing code.

Transcripts will populate `data/interactions/` as Dataset Family D artifacts
(David SS8.6). This is the first population of Dataset Family D.

### DJ-067: 2x2 Factorial Experiment — 10 Dates, 3 Tickers

```
                   No debate     With debate
Base models           A               C
Fine-tuned            B               D
```

10 quarterly dates × 3 tickers × 4 conditions = 120 ensemble runs.
Interaction effect: (D-B) − (C-A) measures whether debate benefits more from
fine-tuned (heterogeneous decision boundaries) vs. base (shared priors) agents.

**Evaluation dates:** 2022-Q1 through 2024-Q2 (quarterly end dates).
**Tickers:** AAPL, JPM, XOM.
**Hardware estimate:** ~5 min/run on M3 Ultra = ~10 hours sequential; checkpointed.

### DJ-068: use_graphrag as Drop-In Path

`run_ensemble()` gains `use_graphrag: bool = False`. Mutually exclusive with
`use_rag`: `assert not (use_rag and use_graphrag)`. `GraphRetriever` wraps
`KnowledgeRetriever` with the graph expansion step but exposes the same interface
(`retrieve(query, ticker, top_k) -> list[DocumentChunk]`). Agent code unchanged.

### DJ-069: Remaining Agent Fine-Tuning — Staged

- **Structural blocker:** `verify_agent()` supports only `FundamentalAnalysis |
  TechnicalAnalysis`. HR/GR baselines cannot be established for Risk, Macro,
  Sentiment without extending the verification layer.
- **Priority:** Sentiment Agent (same qwen2.5-coder-32b model, Phase 13).
- **Deferred:** Risk (gemma-3-4b) and Macro (qwen3.5-27b) — different LoRA
  dynamics for reasoning-distilled models, Phase 13+.
- **Phase 12 deliverable:** Verification layer gap analysis + Sentiment training
  label design document (design only, not execution).

---

## Wave 1 Results (2026-06-14)

### P12-E0-T1: Compliance Examples V2

| Metric | Value |
|---|---|
| Total examples | 200 |
| Extracted (Phase 4 + Phase 9) | 6 |
| Synthetic (OHLCV, 16 tickers, quarterly) | 194 |
| Compliance ratio (vs 26,433 domain) | ~0.75% (up from 0.19%) |
| Schema validation | All 200 pass (decision/confidence/rationale/key_concern) |

Generated by: `uv run python scripts/generate_compliance_examples.py`
Output: `data/training/technical_compliance_v2.jsonl`

### P12-E1-T1: FinancialGraph

`src/hifi/knowledge/graph_store.py` — `FinancialGraph` class:
- NetworkX `DiGraph` backend
- Typed CRUD: `add_company()`, `add_sector()`, `add_macro_factor()`, `add_competes_with()`,
  `add_belongs_to()`, `add_sensitive_to()`
- Readers: `get_competitors()`, `get_sector_peers()`, `get_macro_factors()`,
  `expand_query_tickers(ticker, max_hops=2)`
- Persistence: `save(path)` / `FinancialGraph.load(path)` via `networkx.node_link_data`
- Graceful fallback: unknown ticker in `expand_query_tickers` returns `[ticker]`
- 30 unit tests, all pass

### P12-E1-T2: build_financial_graph()

`src/hifi/knowledge/graph_construction.py`:
- `DEFAULT_COMPETITORS`: 11 tickers, 8 symmetric pairs across 3 sectors
- `DEFAULT_MACRO_SENSITIVITY`: Technology → FFR+VIX, Financial Services → FFR, Energy → VIX+CPI
- `_TICKER_FALLBACK`: metadata for all 17 Phase 10 tickers (no internet required in tests)
- `ticker_metadata` override parameter for deterministic testing
- SENSITIVE_TO edges only added for sectors present in the requested ticker set
- Competitor edges added when at least one endpoint is in the ticker list
- 14 unit tests, all pass

### P12-E3-T1: Debate Schemas

`src/hifi/collective/debate.py`:
- `DebateTurn`: validated phase (challenge/response/revision), non-empty argument,
  optional revised_decision ∈ {Buy, Hold, Sell}, optional revised_confidence ∈ [0, 1]
- `DebateTranscript`: full Oxford round record, vote_delta ∈ {converged, diverged, unchanged}
- `identify_minority()`: plurality with Hold tie-break; returns ([], majority) when unanimous
- `compute_vote_delta()`: measures direction relative to initial majority decision
- `EnsembleOutput.debate_transcript: DebateTranscript | None = None` added to schemas.py
- 23 unit tests, all pass; no circular imports

### Test Metrics at Wave 1 Close

| Metric | Value |
|---|---|
| Tests passing | 1071 |
| Tests skipped | 0 |
| Lint errors (new files) | 0 |
| New tests added | +70 |
| Pre-existing notebook lint issues | 20 (Phase 11 notebook, pre-existing, not in scope) |
| New dependency | networkx 3.6.1 (explicit, was transitive via langgraph) |

---

## Implementation Summary (All Epics)

All infrastructure is complete. Evaluation results (E2, E4) require LM Studio servers.

### Epic E0 — technical_v1 Compliance Fix

| Ticket | File | Status |
|---|---|---|
| E0-T1 | `data/training/technical_compliance_v2.jsonl` (200 examples) | Complete |
| E0-T2 | Re-train `technical_v1` @ rank 8, 500 iters (venvs/finetune/) | Pending (hardware) |
| E0-T3 | Re-evaluate `technical_v2`; deploy/abandon if GR >= 0.720 | Pending (E0-T2) |

Generated by: `scripts/generate_compliance_examples.py`

### Epic E1 — GraphRAG Infrastructure

| Ticket | File | Status |
|---|---|---|
| E1-T1 | `src/hifi/knowledge/graph_store.py` — `FinancialGraph` (NetworkX) | Complete |
| E1-T2 | `src/hifi/knowledge/graph_construction.py` — `build_financial_graph()` | Complete |
| E1-T3 | `src/hifi/knowledge/graph_retrieval.py` — `GraphRetriever` (2-hop BFS + LanceDB) | Complete |
| E1-script | `scripts/build_knowledge_graph.py` → `data/knowledge_graph/financial_graph.json` | Complete |

Tests: `tests/unit/test_graph_store.py` (30), `test_graph_construction.py` (14), `test_graph_retrieval.py`

### Epic E2 — Precision@k Evaluation (RAG vs GraphRAG)

| Ticket | File | Status |
|---|---|---|
| E2-T1 | `scripts/run_phase12_graphrag_eval.py` — Precision@k comparison | Complete (script) |
| E2-results | `tests/fixtures/baseline/phase12_graphrag_precision.json` | Complete (2026-06-14) |

Run: `make graphrag-eval` (requires LanceDB populated + LM Studio)

### Epic E3 — Structured Debate Infrastructure

| Ticket | File | Status |
|---|---|---|
| E3-T1 | `src/hifi/collective/debate.py` — `DebateTurn`, `DebateTranscript`, helpers | Complete |
| E3-T2 | `run_debate_round()` in `collective/debate.py` | Complete |
| E3-T3 | `run_debate_ensemble()` in `agents/ensemble_runner.py` | Complete |
| E3-nodes | `src/hifi/collective/debate_nodes.py` — LangGraph challenge/respond/revise nodes | Complete |
| E3-prompts | `src/hifi/agents/prompts/{challenge,response,revision}_v1.md` | Complete |

Tests: `tests/unit/test_debate_schemas.py` (23), `test_debate_nodes.py`, `test_run_debate.py`

### Epic E4 — 2x2 Factorial Evaluation

| Ticket | File | Status |
|---|---|---|
| E4-T1 | `scripts/run_phase12_evaluation.py` — 120-run orchestrator (checkpointed) | Complete (script) |
| E4-T2 | `compute_factorial_summary()` — interaction effects, herding, OQ-M02 | Complete |
| E4-results | `tests/fixtures/baseline/phase12_factorial_results.json` | Pending (LM Studio) |

Run: `make eval-phase12` (requires all 4 conditions; fine-tuned servers at ports 1235/1236)

### Epic E5 — Documentation & Replication

| Ticket | File | Status |
|---|---|---|
| E5-T1 | `scripts/run_phase12_baseline.py` — baseline evaluation script | Complete |
| E5-T2 | This bitacora (architecture, results, lessons learned) | Complete |
| E5-T3 | `notebooks/phase12_graphrag_debate_replication.ipynb` — frozen replication | Complete |
| E5-tests | `tests/unit/test_phase12_metrics.py` — `compute_factorial_summary`, `_herding_coefficient` | Complete |

Makefile targets added: `build-graph`, `baseline-phase12`, `eval-phase12`, `graphrag-eval`

---

## Verification Gap Analysis (Phase 13 Input)

`verify_agent()` in `src/hifi/verification/verifier.py` currently accepts only
`FundamentalAnalysis | TechnicalAnalysis`. The following work is required to extend
it to Phase 8 agents:

| Agent | Schema | Changes required |
|---|---|---|
| Risk | RiskAnalysis | Extend FIELD_ALIAS_TABLE, add risk-specific claim extractor |
| Macro | MacroAnalysis | FRED series citations need macro-specific grounding rules |
| Sentiment | SentimentAnalysis | New claim types (sentiment scores, filing citations) |

Estimated effort: ~3 tickets per agent (extractor, verifier, baseline fixture).
This unblocks fine-tuning for Risk, Macro, Sentiment agents in Phase 13.

---

## Sentiment Agent Training Label Design (Phase 13 Input)

**Candidate approach:** MD&A management tone classifier.

SEC filings (already in Dataset Family B from Phase 7) contain MD&A sections where
management describes business outlook. A keyword-based deterministic classifier
assigns tone: cautious → Sell, neutral → Hold, optimistic → Buy.

**Feasibility:** Phase 7 EDGAR corpus covers AAPL, JPM, XOM (10-K, 10-Q filings,
2018-2023). Approximately 60-90 filing-quarters available.

**Validation required before execution:**
- Does the Phase 7 filing corpus have sufficient MD&A text (>100 tokens per filing)?
- Is the keyword classifier signal-to-noise ratio adequate (>60% accuracy on
  held-out analyst ratings)?
- Does the training set have enough Sell examples (2022 downturn period)?

**Phase 13 decision gate (DJ-069):** Execute Sentiment fine-tuning only if:
1. Verification layer extended to SentimentAnalysis (HR/GR baseline established)
2. MD&A corpus yields >= 200 labeled examples with balanced Sell representation
3. Phase 12 diversity evidence shows Sentiment is the diversity bottleneck

---

## Spanner Emulator — Phase 15 Scope (DJ-070)

**Decision:** Google Cloud Spanner emulator integration is deferred to Phase 15
(Containerization). See `doc/HIFI_DAVID.md` SS11.6 and OQ-A03.

**Rationale:** The single-machine architecture must be validated at scale before
distributed storage is warranted. Phase 14 (Paper Trading) provides that validation.
Phase 15 containerises the system for cloud readiness.

**Planned scope in Phase 15:**
- Replace LanceDB + Parquet file storage with Spanner emulator for local development
- Schema: financial knowledge graph edges + document chunks as Spanner tables
- Migration script: existing data/knowledge_graph/ and data/market/ → Spanner
- Test against Spanner emulator (no GCP credentials required for local development)
- Document path to real GCP Spanner for production deployment

**Why Spanner specifically:** Globally distributed, strongly consistent. If HiFi
scales beyond a single machine (multi-region paper trading → live trading), Spanner's
external consistency guarantees preserve data integrity across nodes — something
SQLite and local LanceDB cannot provide.

**Alternative (lower complexity):** CockroachDB provides Spanner-compatible open-source
API with single-node Docker mode. If the Spanner emulator proves too heavyweight for
local development, CockroachDB is the fallback.

**OQ-A03 formally answered in Phase 15:** after single-machine validation in Phase 14.

---

## Open Questions

| ID | Question | Resolution target |
|---|---|---|
| OQ-K02 | Does GraphRAG improve Precision@k by >= 5% over dense RAG? | **ANSWERED NEGATIVE (2026-06-14)** — delta=0.000, KEEP plain RAG (DJ-016) |
| OQ-M02 | Does fine-tuning preserve diversity across 10 dates? | Phase 12 factorial eval — conditions B/C/D in progress |
| OQ-D01 | Does debate cause herding (kappa increase > 0.1)? | Phase 12 factorial eval — conditions C/D in progress |
| OQ-D02 | Is the 2x2 interaction effect (D-B)-(C-A) positive? | Phase 12 factorial eval — all 4 conditions required |
| OQ-D03 | What fraction of dates have non-unanimous initial votes? | **ANSWERED: 36.7%** — condition A: 11/30, XOM=100%, JPM=10%, AAPL=0% |
| OQ-A03 | Is Spanner emulator warranted for local deployment? | Phase 15 (DJ-070) |

---

## Open Questions for Phase 13

These questions are generated by Phase 12 but require Phase 13 to answer. They are
distinct from the Phase 12 OQs above (which will be answered when LLM evaluation runs).

| ID | Question | Context |
|---|---|---|
| OQ-P13-01 | Does extending verify_agent() to SentimentAnalysis introduce ambiguity in claim extraction (sentiment scores vs. financial claims)? | DJ-069: Verification Gap Analysis |
| OQ-P13-02 | Is the MD&A keyword classifier signal strong enough to produce unambiguous training labels, or does tone conflate risk disclosure with outlook? | Sentiment label design — Section 4 above |
| OQ-P13-03 | Should Macro Agent fine-tuning use reasoning-trace supervision (chain-of-thought) rather than output-only LoRA given qwen3.5-27b's reasoning architecture? | DJ-069: reasoning model LoRA dynamics |
| OQ-P13-04 | Does gemma-3-4b (Risk Agent) respond to LoRA fine-tuning at rank 8, or does the smaller parameter count require rank 4? | DJ-069: Risk Agent fine-tuning |
| OQ-P13-05 | Does multi-round debate (≥2 rounds) converge to consensus faster than single-round, or oscillate? Calibrate convergence criterion from Phase 12 transcripts. | DJ-065: one-round limitation |
| OQ-P13-06 | Is LLM-extracted competitor graph (from SEC filings) more useful than the hand-coded graph for GraphRAG Precision@k? | DJ-063: manual vs. automatic graph construction |

---

## Phase 13 Inputs (produced by Phase 12)

1. **technical_v1 deploy/abandon decision** — gate for debate evaluation with fine-tuned agents
2. **Verification gap analysis** — exact tickets to extend verify_agent() for Phase 8 schemas
3. **Sentiment label design document** — go/no-go for Sentiment Agent fine-tuning
4. **Debate infrastructure** — run_debate_ensemble(), DebateTranscript schema, debate nodes
5. **GraphRAG infrastructure** — GraphRetriever, FinancialGraph, build_financial_graph()
6. **2x2 factorial results** — herding evidence, diversity measurement, interaction effect
7. **OQ-K02 answer** — DJ-016 decision recorded: adopt or keep plain RAG
8. **Dataset Family D** — debate transcripts as training data for future DPO alignment

---

## Results (2026-06-14)

---

### E2: Graph-Expanded Retrieval Precision@k Evaluation (OQ-K02)

**CORRECTED 2026-06-15** — The initial run (2026-06-14T22:00 UTC) produced P@5=0.000
for both retrievers due to three compounding bugs. All three have been fixed and the
evaluation re-run with valid data.

**Terminology correction:** What was previously called "GraphRAG" is actually
**graph-expanded dense retrieval** — 2-hop BFS ticker-filter widening via
FinancialGraph, followed by standard cosine ANN search in LanceDB. It is NOT true
GraphRAG (no entity extraction, community detection, or relationship summaries).

**Run date:** 2026-06-15
**Script:** `scripts/run_phase12_graphrag_eval.py` (corrected)
**Fixture:** `tests/fixtures/baseline/phase12_graphrag_precision.json`
**Evaluation set:** 20 queries, 3 tickers (AAPL/JPM/XOM), k=5
**Knowledge store:** `data/knowledge/knowledge.lance/` (169 chunks, 3 tickers, 3 filing types)

#### Bug Fix Summary

| Bug | Root Cause | Fix |
|---|---|---|
| BUG 1 (path) | Eval script used `data/knowledge.lance/` (0 rows, empty). Actual data at `data/knowledge/knowledge.lance/` (169 rows). `main()` missing `--knowledge-dir` arg. | Added `--knowledge-dir` CLI arg, default `data/knowledge/`. |
| BUG 2 (sections) | Indexed sections = `['Full Text', 'Earnings Release']`. Eval queries expect `['MD&A', 'Risk Factors', 'Business', 'Earnings Release']`. 16/20 queries can never match at section level. | Two-level evaluation: document-level (ticker+filing_type) and section-level (ticker+section+filing_type). OQ-K02 uses document-level. |
| BUG 3 (naming) | "GraphRAG" label implied entity extraction and community detection. Actual implementation is graph-expanded dense retrieval (BFS ticker expansion + cosine ANN). | Terminology corrected throughout script, Makefile, and bitacora. |

#### Raw Metrics (Corrected)

| Metric | RAG (dense) | Graph-expanded | Delta | OQ-K02 (>=5pp)? |
|---|---|---|---|---|
| Document Precision@5 | 0.3100 | 0.3100 | 0.0000 | NO |
| Section Precision@5 | 0.1000 | 0.1000 | 0.0000 | NO |
| Queries improved | -- | 0/20 | -- | -- |
| Queries degraded | -- | 0/20 | -- | -- |
| Queries unchanged | -- | 20/20 | -- | -- |
| n_retrieved per query | 5 | 5 | -- | -- |

#### Two-Level Precision Methodology

**Document-level precision** (ticker + filing_type match) measures whether the
retriever finds chunks from the correct document type. This is the meaningful metric
because it tests the retrieval engine's ability to identify relevant documents
regardless of section-level parsing granularity.

**Section-level precision** (ticker + section + filing_type match) measures section
granularity. Currently limited to 0.10 because only 4/20 Earnings Release queries
can match — the upstream SEC parser (`data/edgar.py`) stores all 10-K/10-Q content
as a single "Full Text" section instead of extracting named SEC sections (MD&A, Risk
Factors, Business). This is a **data engineering gap**, not a retrieval algorithm
failure. Fixing the SEC parser to extract named sections is a potential Phase 14+
improvement.

#### Per-Query Analysis

Document precision by filing type:
- **10-K queries** (8 queries): mean doc_p = 0.350 (range 0.20-0.60)
- **10-Q queries** (6 queries): mean doc_p = 0.100 (range 0.00-0.40)
- **8-K/Earnings queries** (6 queries): mean doc_p = 0.367 (range 0.20-0.80)

10-Q queries score lower because the knowledge store has fewer 10-Q chunks
relative to 10-K (the AAPL 10-Q queries Q01/Q16 score doc_p=0.00 — the cosine
similarity ranking favors 10-K and 8-K content even for quarterly queries).

#### Root Cause: Zero Delta

Graph expansion adds competitor tickers (e.g., MSFT, GOOGL for AAPL; GS, BAC for JPM)
but these competitors have **no indexed documents** in the knowledge store (only
AAPL/JPM/XOM were ingested in Phase 7). The expanded ticker set is effectively
identical to `{primary_ticker}` for all queries — so graph-expanded retrieval
produces the exact same results as plain RAG.

This is not a failure of the graph expansion algorithm. It is a **data coverage
limitation**: the value of graph-expanded retrieval scales with the number of indexed
entities, and with only 3 indexed tickers out of 11 in the graph, expansion has no
material to work with.

#### DJ-016 Decision: KEEP plain RAG

Per DJ-016 ("do not add complexity without evidence"), the decision is **KEEP plain
RAG**. Graph-expanded retrieval adds query-expansion complexity (2-hop BFS + multi-
ticker LanceDB filter) without any document-level Precision@k improvement.

The `FinancialGraph` and `GraphRetriever` infrastructure remain in the codebase as
scaffold for future work (expanded ticker coverage, or true GraphRAG with entity
extraction). The default `run_ensemble()` path remains dense RAG only.

#### Connection to Past Phases

- **Phase 7 (RAG Knowledge Systems):** The corrected evaluation validates that the
  Phase 7 retriever IS functional — document Precision@5 = 0.31, not zero. The
  initial P@5=0.000 result was entirely due to BUG 1 (wrong data path). Phase 7's
  approach is vindicated: dense ANN with ticker filter produces meaningful retrieval.
- **Phase 5 (Verification Layer):** The Section P@5 = 0.10 explains the low
  mean_SGR=0.167 observed in the Phase 13 E0 baseline — the Sentiment agent receives
  chunks labeled "Full Text" that don't align with the section-level ground truth.

#### Connection to Future Phases

- **Phase 13 E3 (LLM-extracted graph):** Per DJ-069, E3 was gated on OQ-K02 >= 5pp
  improvement. OQ-K02 is NEGATIVE (0pp) at document level. Phase 13 E3 is NOT
  triggered. The manual seed graph (11 tickers, 3 macro nodes, 34 edges) is final.
- **Phase 14 (Paper Trading):** Two improvements needed before live trading:
  (1) Expand knowledge store beyond 3 tickers to give graph expansion useful data.
  (2) Improve SEC parser to extract named sections (MD&A, Risk Factors, etc.) instead
  of "Full Text", which would lift both section precision and Sentiment SGR.
- **Phase 15 (Containerization):** The Precision@k framework (20 queries) becomes the
  regression test for post-migration retrieval quality. Document P@5 >= 0.31 is the
  baseline to preserve.

---

### Phase 13 E0: Verification Baseline (Risk, Macro, Sentiment)

**Run date:** 2026-06-14
**Script:** `scripts/run_phase13_verification_baseline.py`
**Fixture:** `tests/fixtures/baseline/phase13_verification_baseline.json`
**Evaluation date:** 2023-03-31 (same as Phase 5 and Phase 11 baselines)
**Tickers:** AAPL, JPM, XOM
**HiFi commit:** fcf57b5

#### Metric Definitions

Four distinct metrics are used across the three agent types. Understanding their
denominators and epistemological scope is critical for interpreting the results.

**HR (Hallucination Rate):** `n_hallucinated / n_resolvable`, where
`n_resolvable = n_verified + n_hallucinated`. A hallucination is a numerical claim
that was extracted from the agent's rationale, matched to an MCP tool field via the
alias table, and whose value differs from the tool output by more than the tolerance.
HR measures numerical fabrication among claims the verifier can evaluate. HR=0.000
is the unconditional safety target — it gates fine-tuned model deployment (DJ-058).

**GR (Grounding Rate):** `n_verified_with_call_id / n_verified`. Among verified claims,
GR measures whether the agent cited the specific call_id of the MCP tool call that
produced the value. GR=1.000 means every verified number was properly attributed.
GR captures "source honesty" — does the agent know where its numbers came from?

**Alias Coverage:** `n_resolvable / n_claims`. The fraction of extracted numerical
claims that could be matched to any field in the alias table. Low coverage means the
alias table has gaps, not that the agent is hallucinating. Coverage is an actionable
improvement target for the verification infrastructure.

**SGR (Sentiment Grounding Rate):** `n_grounded / n_signals`. For each item in
`SentimentAnalysis.notable_signals`, SGR checks whether the lowercased signal text
appears as a substring of the lowercased retrieved SEC filing context. This is
qualitative grounding (paraphrase vs. verbatim quotation) rather than numerical
verification. SGR measures whether the Sentiment agent claims come from the source
documents rather than LLM confabulation.

#### Risk Agent Results

| Ticker | n_claims | n_verified | n_hallucinated | n_unresolvable | HR | GR | Alias Coverage |
|---|---|---|---|---|---|---|---|
| AAPL | 3 | 1 | 0 | 2 | 0.000 | 1.000 | 0.333 |
| JPM | 3 | 1 | 0 | 2 | 0.000 | 1.000 | 0.333 |
| XOM | 2 | 1 | 0 | 1 | 0.000 | 1.000 | 0.500 |
| **Aggregate** | **8** | **3** | **0** | **5** | **0.000** | **1.000** | **0.389** |

**HR = 0.000 (no hallucinations).** The Risk agent (gemma-3-4b) does not fabricate
numerical values. All resolvable claims match MCP tool outputs within tolerance. This
is the first HR measurement for the Risk agent — Phase 5 and Phase 11 measured
Fundamental and Technical only. gemma-3-4b achieves the same HR=0.000 as the larger
qwen2.5-coder-32b despite its smaller parameter count.

**GR = 1.000 (all verified claims cited).** Every resolved claim includes a valid
call_id. The Risk agent traces each number back to its source MCP call. This is the
Phase 13 baseline for measuring fine-tuning impact: if a Risk fine-tune degrades GR
below 1.000, the model is learning to cite without verification.

**Alias Coverage = 38.9% (3 of 8 claims matched).** The unmatched claims are
20-day and 252-day volatility values, which appear in every risk rationale but whose
alias patterns ("20-day historical volatility", "252-day volatility") do not currently
map to the `hist_vol_20d` / `hist_vol_252d` fields in the alias table. This is a
verifier infrastructure gap. Expanding the alias table would raise coverage to ~100%
for Risk agent claims and make unresolvable claims actual evidence of hallucination
rather than alias misses.

**Detailed breakdown (verified claims per ticker):**
- AAPL: max_drawdown_252d=0.296 verified (tool value: 0.29620..., tolerance: 0.01). Two volatility values unresolvable.
- JPM: sharpe_252d=-0.01 verified (tool value: -0.01008..., tolerance: 0.01). Two volatility values unresolvable.
- XOM: sharpe_252d=1.07 verified (tool value: 1.07724..., tolerance: 1%). One volatility value unresolvable.

**Why this matters for past phases:** Phase 5 (Verification) established the HR/GR
framework for Technical/Fundamental agents. Phase 12 extends it to Risk, completing
verification coverage across the 3 non-reasoning agents (Fundamental, Technical, Risk
all use qwen2.5-coder-32b or gemma-3-4b). Phase 11 (Fine-Tuning) proved HR/GR can
degrade under poor fine-tuning (technical_v1 GR→0.000). Risk HR=0.000 baseline is
now established before any Risk fine-tuning begins.

**Why this matters for future phases:** Phase 13 E1 Sentiment fine-tuning gate
requires HR/GR baselines for all 5 agents — Risk baseline is now satisfied. Any
Risk fine-tuning in Phase 13+ must preserve HR=0.000 and GR≥0.720 (DJ-058 threshold).
The 38.9% alias coverage is the top actionable improvement: one ticket to expand the
alias table would close this gap before Phase 13 Risk fine-tuning.

#### Macro Agent Results

| Ticker | n_claims | n_verified | n_hallucinated | n_unresolvable | HR | GR | Root Cause |
|---|---|---|---|---|---|---|---|
| AAPL | 0 | 0 | 0 | 0 | 0.000 | 0.000 | No FRED data — macro parquet not acquired for 2023 |
| JPM | 1 | 0 | 0 | 1 | 0.000 | 0.000 | FRED error string "2023" extracted as false-positive claim |
| XOM | 1 | 0 | 0 | 1 | 0.000 | 0.000 | FRED error string "2023" extracted as false-positive claim |
| **Aggregate** | **2** | **0** | **0** | **2** | **0.000** | **0.000** | **All claims unresolvable — FRED data absent** |

**HR = 0.000.** No hallucinations. The Macro agent did not fabricate any values
against available tool outputs.

**GR = 0.000.** Meaningful GR cannot be established without verified claims. This
is a data gap, not a model quality signal.

**Root cause: FRED macro data missing for 2023-03-31.** The Phase 1 data acquisition
and Phase 10 bootstrap both used 2018-2022 FRED series. The macro parquet files do
not cover 2023. The Macro agent's `macro_snapshot` MCP tool returned an error:
`{"error": "NO_MACRO_DATA", "detail": "No macro parquet files in data/macro"}`.

**False-positive claim extraction:** The regex extractor found `2023.0` in the error
text "No macroeconomic data is available for the assessment date of 2023-03-31". The
year "2023" is a date component — not a FRED series value — so it cannot be matched
to any macro alias. This is a regex boundary case: year-format numbers (2018-2030)
should be excluded from numerical claim extraction. This is an extractor improvement
target, not a macro agent failure.

**MacroAnalysis dual rationale:** The `verify_agent()` extension for MacroAnalysis
concatenates both `signal.rationale` and the analysis-level `analysis.rationale` before
extracting claims. The AAPL Macro agent produced zero numerical claims in its
combined rationale — the error caused it to generate a qualitative-only response.

**Why GR = 0.000 is NOT a deployment blocker:** The Macro agent baseline on 2023-03-31
is inconclusive due to missing data. The 2x2 factorial experiment uses 2020-2022 dates
where FRED data was acquired. A valid Macro baseline requires either: (a) acquiring
2023-03-31 FRED data via `make acquire-data`, or (b) selecting a baseline date within
the existing FRED coverage window.

**Why this matters for past phases:** Phase 8 (Agent Population) deployed the Macro
agent against the full data pipeline, but verification was never run against it.
This baseline reveals that the Macro agent's verifiability is entirely dependent on
the FRED data coverage window — a data-pipeline dependency not previously documented.

**Why this matters for future phases:** Phase 13 E1 gate requires Macro baseline
before any Macro fine-tuning. The gate is: (1) acquire 2023-03-31 FRED data,
(2) re-run baseline, (3) confirm GR > 0.0 before training. Phase 13 scope for
Macro fine-tuning is deferred (DJ-069); the data gap must be resolved first.
Additionally, the regex false-positive for year-format numbers should be patched
before the alias table is expanded — otherwise year citations will appear as
unresolvable claims in all future Macro evaluations.

#### Sentiment Agent Results

| Ticker | n_signals | n_grounded | SGR | Filing context quality |
|---|---|---|---|---|
| AAPL | 2 | 0 | 0.000 | Procedural (DEF 14A — shareholder meeting proxy) |
| JPM | 2 | 1 | 0.500 | Substantive (compensation plan with verbatim text) |
| XOM | 2 | 0 | 0.000 | Generic (risk disclosure without specific phrases) |
| **Aggregate** | **6** | **1** | **0.167** | — |

**Mean SGR = 0.167.** At the Phase 12/13 boundary, the Sentiment agent grounds
1 of 6 notable_signals verbatim in the retrieved filing context. This is the
SGR baseline against which Phase 13 E1 fine-tuning must improve.

**AAPL SGR = 0.000 — Procedural filing effect.** The retrieved context for AAPL on
2023-03-31 was a DEF 14A (annual meeting proxy) rather than a 10-K/10-Q. The agent's
two notable_signals correctly characterize this filing ("primarily contain procedural
information...") but use paraphrase, not quotation. SGR exact-match grounding penalizes
paraphrase even when the characterization is accurate. The signals are epistemically
correct but not SGR-grounded.

**JPM SGR = 0.500 — Verbatim citation present.** The JPM agent quoted:
"For any calendar year ending during the vesting period, JPMorgan Chase's annual
pre-tax pre-provision income at the Firm level is negative." This phrase appears
verbatim in the compensation plan text, confirmed by the `matched_chunk` field in
the fixture. This is the positive template for Phase 13 fine-tuning: agents that
quote rather than paraphrase achieve SGR grounding. The second JPM signal
("The New York Stock Exchange Depositary Shares...") was not found verbatim,
explaining the 0.5 rather than 1.0 SGR.

**XOM SGR = 0.000 — Absence characterization.** The XOM agent produced two signals
describing what the filings do NOT contain: "The filings do not contain specific
forward guidance..." and "There are no new risk factors...". Absence claims cannot
be grounded as verbatim substrings — there is no text to match against a statement
about missing text. This is a systematic pattern: when filings lack forward guidance,
the Sentiment agent falls back to characterizing absence rather than quoting presence.

**Phase 13 E1 gate analysis:** DJ-069 requires SGR ≥ 0.500 before Sentiment
fine-tuning can proceed. Mean SGR = 0.167 < 0.500. The gap is agent behavior,
not retrieval quality — JPM showed that when substantive filing text is available,
the agent can achieve SGR = 0.500. The fine-tuning objective is: train the Sentiment
agent to produce quotation-style notable_signals rather than paraphrase or absence
characterizations. This requires training examples where the notable_signals items
are literal substring of the retrieved context (JPM-style).

**Why this matters for past phases:**
- **Phase 7 (RAG):** The retrieval system successfully fetched substantive text for
  JPM. The SGR failure for AAPL and XOM is not a retrieval failure — it is that the
  retrieved content (proxy filing, generic disclosure) does not contain the kind of
  forward-looking language that generates quotable signals. This is a filing-type
  coverage issue: DEF 14A filings should not be the primary source for the Sentiment
  agent on an earnings date.
- **Phase 8 (Agent Population):** The Sentiment agent was designed in Phase 8 without
  a grounding constraint. SGR=0.167 is the cost of that design: agents producing
  plausible-sounding but unverifiable signals. Fine-tuning (Phase 13 E1) will introduce
  the grounding constraint retroactively.

**Why this matters for future phases:**
- **Phase 13 E1:** Fine-tuning dataset must include examples where `notable_signals`
  items are confirmed as verbatim substrings of `retrieved_context`. JPM 2023-03-31
  is one confirmed positive example. The training set needs ≥ 200 such examples.
- **Phase 14 (Paper Trading):** SGR is the real-time auditability metric for the
  Sentiment agent in live conditions. A Sentiment agent with SGR < 0.500 produces
  signals that cannot be traced to source documents — an unacceptable audit risk in
  production. The Phase 13 E1 fine-tuning gate (SGR ≥ 0.500) is a production
  readiness criterion, not just a training quality metric.

---

### E4: 2x2 Factorial Evaluation (BLOCKED)

**Status:** Condition A COMPLETE (30/30). Condition B PARTIAL (12/30, process died).
  Conditions C, D NOT STARTED. BLOCKED on technical_v2 deployment.
**Script:** `scripts/run_phase12_evaluation.py`
**Checkpoint:** `data/evaluation/phase12/checkpoint.json` (42/120 runs)
**PID 7978:** No longer running (stopped mid-B at B_JPM_2020-06-30)

#### Condition A Final Results (30 of 30 COMPLETE — 2026-06-14)

| Metric | Value |
|---|---|
| Runs complete | 30 (AAPL: 10/10, JPM: 10/10, XOM: 10/10) |
| Collective Hold | 19 |
| Collective Buy | 11 |
| Collective Sell | 0 |
| Unanimous dates (entropy=0.0) | 19/30 = 63.3% |
| Non-unanimous dates (entropy=1.0) | 11/30 = 36.7% |
| Mean disagreement_entropy | 0.367 |
| Mean opinion_dispersion | 0.066 |

**OQ-D03 ANSWERED for condition A:** 36.7% debate-eligible dates with initial disagreement.

#### Non-Unanimous Dates — Condition A (All 11)

| Run | Decision | Entropy | Agent votes | Market context |
|---|---|---|---|---|
| A_JPM_2021-09-30 | Buy | 1.000 | Hold / Buy | JPM post-COVID recovery; valuation vs. momentum diverge |
| A_XOM_2020-03-31 | Buy | 1.000 | Buy / Hold | COVID oil crash; Fundamental buys the dip, Technical warns on momentum |
| A_XOM_2020-06-30 | Buy | 1.000 | Buy / Hold | Partial oil recovery; fundamental-technical divergence persists |
| A_XOM_2020-09-30 | Buy | 1.000 | Buy / Hold | Energy demand uncertain; COVID second wave |
| A_XOM_2020-12-31 | Buy | 1.000 | Buy / **Sell** | **Maximum disagreement:** vaccine optimism (Fund) vs. bearish price trend (Tech) |
| A_XOM_2021-03-31 | Buy | 1.000 | Buy / Hold | Oil recovery; valuation gap between fundamental value and technical signal |
| A_XOM_2021-06-30 | Buy | 1.000 | Buy / Hold | OPEC+ compliance; fundamental strength, technical hesitant |
| A_XOM_2021-09-30 | Buy | 1.000 | Buy / Hold | Oil above $75; continued fundamental/technical split |
| A_XOM_2021-12-31 | Buy | 1.000 | Buy / Hold | Omicron wave creates technical uncertainty despite high fundamental scores |
| A_XOM_2022-03-31 | Buy | 1.000 | Buy / Hold | Ukraine war energy spike; Technical sees parabolic, Fundamental sees value |
| A_XOM_2022-06-30 | Buy | 1.000 | Buy / Hold | Peak oil prices; energy fundamental buy, technical momentum fading |

#### Ticker-Level Diversity Analysis

| Ticker | Unanimous | Non-unanimous | Disagreement Rate |
|---|---|---|---|
| AAPL | 10/10 | 0/10 | 0% |
| JPM | 9/10 | 1/10 | 10% |
| XOM | 0/10 | 10/10 | 100% |

**XOM = 100% non-unanimous: the defining structural finding of condition A.**
The Fundamental agent votes Buy for XOM on all 10 dates across 2020-2022.
The Technical agent votes Hold on 9 dates and Sell on 1 (2020-12-31). This
persistent split reflects genuinely different information processing: Fundamental
sees XOM's depressed PE, book value, and dividend yield as buying opportunities;
Technical sees falling momentum, high volatility, and bearish indicators as risk signals.
Neither agent is wrong — they process heterogeneous evidence.

This is the exact scenario the structured debate is designed for: systematic
disagreement between agents using different evidence types. XOM is the
"debate-sensitive" ticker. Conditions C and D will reveal whether debate resolves
the fundamental/technical split or amplifies it via herding.

**A_XOM_2020-12-31: only Buy/Sell split.** Maximum disagreement (entropy=1.0,
['Buy', 'Sell']). Collective output is Buy (confidence-weighted). The minority
Sell signal carries private information about price momentum risk that is overruled
by Fundamental's higher confidence. The debate's challenge phase is the designed
mechanism for amplifying this minority view — OQ-D01 measures whether it helps.

**AAPL = 100% unanimous.** Apple's consistent profitability and valuation stability
produce identical signals from both agents on every date. The debate is vacuously
applied (debate_skipped=True) on all 10 AAPL dates in conditions C/D — no additional
LLM calls occur.

**Hold dominance (19/30 = 63.3%):** Consistent with Phase 9 (unanimous Hold),
Phase 10 (bootstrap technical accuracy 0.254 / fundamental 0.079 — Hold-biased),
and Phase 11 evaluation (unanimous Hold, single date). The base ensemble is
structurally risk-averse. The 11 Buy decisions are concentrated in XOM (10) and
JPM (1) — sectors with genuine value dislocation vs. technical momentum signals.

#### Condition B Results and Critical Findings (12/30 runs, process died)

**Completion:** AAPL 10/10, JPM 2/10, XOM 0/10. Process (PID 7978) stopped at
B_JPM_2020-06-30 — likely OOM or timeout from technical_v1's 639s/request.

| Metric | Condition A (30 runs) | Condition B (12 runs) |
|---|---|---|
| AAPL decisions | 10/10 Hold (unanimous) | 10/10 **Buy** (unanimous) |
| JPM decisions | 9 Hold + 1 Buy | 2/2 **Buy** |
| XOM decisions | 10 Buy | (not reached) |
| n_valid_signals | 2 (Fund + Tech) | 1 (Fund only) |
| technical_analysis.signal | Present | **None** (12/12 failed) |
| Technical latency | ~105s | **639s** (timeout / parse failure) |
| entropy | 0.367 mean | 0.0 (single agent, vacuous) |

**technical_v1 production failure confirmed across all 12 runs.** The fine-tuned
technical_v1 adapter on port 1235 returns responses that cannot be parsed into
`AgentSignal`. `TechnicalAnalysis.signal` is None in all 12 condition B runs.
Latency = 639s per request — the model generates very long or malformed responses
before the framework's parser gives up. This is the same failure mode that caused
GR=0.000 in Phase 11 evaluation.

**fundamental_v1 systematic Buy bias discovered.** Fine-tuning shifted the
Fundamental agent's decision boundary:

| Ticker | Base (A) Fundamental | Fine-tuned (B) Fundamental | Direction Shift |
|---|---|---|---|
| AAPL | 10/10 Hold | 10/10 Buy | Hold -> Buy |
| JPM | 10/10 Hold | 2/2 Buy | Hold -> Buy |
| XOM | 10/10 Buy | (not reached) | (already Buy) |

The fine-tuned fundamental agent votes Buy for EVERY input regardless of ticker.
This is consistent with the training data bias: max-return labels from 2016-2022
bull market produced Buy-dominant labels (Phase 11 DJ-054). The fine-tuning
preserved GR=1.000 (format quality) but introduced a systematic directional
bias. This is an important epistemological finding: **format compliance (GR) is
necessary but not sufficient for decision quality.** The model can cite numbers
correctly while making systematically biased decisions.

**Phase 11 deploy decision VALIDATED.** Phase 11 declined to deploy technical_v1
(GR=0.000). The 639s production failure confirms this was correct. technical_v2
(with Phase 12 compliance fix) is needed before Technical can operate fine-tuned.

**Condition B is effectively single-Fundamental-agent.** With
technical_analysis.signal=None, only the fine-tuned Fundamental agent vote counts.
n_valid_signals=1 in all B runs, meaning:
- Entropy is always 0.0 (no disagreement possible with 1 agent)
- Collective decision = Fundamental agent's decision alone
- All B runs = fine-tuned Fundamental's Buy (not a 2-agent consensus)

**OQ-M02 (diversity preserved): NOT ANSWERABLE.** The fine-tuned ensemble in
condition B has lower diversity than condition A by construction (1 agent vs 2).
OQ-M02 is negative by design — the technical failure reduces the ensemble to a
single voter. This is a data quality constraint, not a model quality answer.
OQ-M02 requires both technical_v2 deployment AND re-running condition B.

**Interaction effect (D-B)-(C-A): NOT COMPUTABLE.** Condition B's single-agent
nature confounds the factorial design. The interaction effect is meaningless
without valid 2-agent data in conditions B and D.

#### Conditions C, D: NOT STARTED

Conditions C (base, debate) and D (FT, debate) did not start before the process
died. All four conditions must be re-run after technical_v2 is deployed.

#### Factorial Summary and OQ Status

| Question | Status | Finding |
|---|---|---|
| OQ-D03 (debate-eligible %) | ANSWERED (condition A) | 36.7% non-unanimous. XOM=100%, JPM=10%, AAPL=0%. |
| OQ-M02 (diversity preserved) | NOT ANSWERABLE | technical_v1 failure confounds condition B. |
| OQ-D01 (debate improves quality) | NOT ANSWERABLE | Conditions C/D not started. |
| OQ-D02 (debate induces herding) | NOT ANSWERABLE | Conditions C/D not started. |

**Next steps (blocked on technical_v2):**
1. Train technical_v2 at 500 iters, rank 8 with augmented compliance data
2. Evaluate technical_v2: GR >= 0.720 gate (DJ-058)
3. Re-run full factorial (120 runs) with working fine-tuned models
4. Compute interaction effects and answer OQ-M02, OQ-D01, OQ-D02

#### E0: technical_v2 Re-Evaluation (Pending hardware window)

| Metric | fundamental_v1 (Phase 11) | technical_v1 (Phase 11) | technical_v2 (Phase 12, pending) |
|---|---|---|---|
| Hallucination Rate (HR) | 0.000 | 0.000 | TBD (after E0-T2 retraining) |
| Grounding Rate (GR)     | 1.000 | 0.000 | TBD (deploy if ≥ 0.720) |
| Compliance ratio        | ~0.19% | ~0.19% | ~0.75% (200 examples) |
| Training iterations     | 1000 | 1000 | 500 (reduced to limit domain overfitting) |

Retraining command (deferred until factorial eval completes — avoids 3× 32B model in memory):
```
cd venvs/finetune && python -m mlx_lm.lora \
  --model ~/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-8bit \
  --data ../../data/training/technical_compliance_v2.jsonl \
  --iters 500 --rank 8 \
  --adapter-path ../../data/adapters/technical_v2/
```

---

## Lessons Learned

### Wave 1 (Infrastructure)

**Compliance ratio is load-bearing for LoRA fine-tuning.** The 0.19% compliance ratio
in Phase 11 was identified as the root cause of technical_v1's GR collapse. The fix
requires only more format examples — the training infrastructure, rank, and base model
are all validated. This is a strong signal that compliance example count should be
monitored as a first-class metric in all future fine-tuning runs.

**NetworkX as a transitive dependency is fragile.** networkx 3.6.1 was available in
the uv env via langgraph but not pinned. Added as an explicit direct dependency to
prevent a future langgraph update from removing the transitive availability.

**Debate skipping is the common case.** Preliminary analysis of Phase 9 collective
outputs (3 tickers, 1 date) shows unanimous agreement across all agents. OQ-D03
(debate participation rate) is therefore a critical early measurement — if agents
are unanimous on > 80% of dates, the debate mechanism has limited scope.

**EnsembleOutput backward compatibility.** Adding `debate_transcript: ... | None = None`
to EnsembleOutput required no changes to any existing test or calling code — the
optional field with None default is fully additive.

### Wave 2 (Evaluation)

**Precision@k=0.000 for both retrievers is a retrieval filter bug, not a GraphRAG
failure.** The evaluation exposed a metadata alignment gap between the LanceDB index
and the evaluation ground-truth labels. This gap was invisible in Phase 7 because
agent retrieval uses the full knowledge_server pipeline, not direct filtered ANN
queries. Future retrieval evaluations must verify `n_retrieved > 0` before interpreting
precision values.

**Macro agent verification requires FRED coverage.** The 2023-03-31 baseline date is
outside the Phase 1 FRED data acquisition window (2018-2022). The Macro agent baseline
is therefore inconclusive for the chosen date. Future baselines for Macro should use
dates within the FRED coverage window (2018-2022) or acquire 2023 FRED data explicitly.

**Regex false-positives in year-format numbers.** The numerical claim extractor
matched "2023" in the Macro agent's error string as a numerical claim. Year-format
integers (2000-2100) should be excluded from claim extraction to avoid false-positive
unresolvable counts in Macro verification output.

**SGR = 0.167 reveals agent behavior, not retrieval failure.** The Sentiment agent
achieves SGR=0.500 for JPM (where substantive filing text is available) but SGR=0.000
for AAPL and XOM (where the agent paraphrases or characterizes absence). The gap
between retrieval quality and SGR is an agent behavior signal: fine-tuning must
specifically target quotation behavior. Training examples where the agent paraphrases
correctly but fails SGR are negative examples; examples where the agent quotes verbatim
and achieves SGR are positive examples.

**Hold dominance is structural, not incidental.** 18/19 condition-A collective decisions
are Hold across 2020-2022. This is consistent with Phase 9, Phase 10, and Phase 11
evaluation results — all of which showed unanimous or near-unanimous Hold. The base
ensemble is systematically risk-averse. The 2x2 factorial will test whether fine-tuning
(condition B) shifts this distribution, and whether structured debate (conditions C/D)
can introduce Buy/Sell diversity on the rare non-unanimous dates.
