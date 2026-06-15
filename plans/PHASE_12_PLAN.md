# Phase 12: GraphRAG + Structured Debate

**Status:** PLANNED

| Epic | Title | Status |
|---|---|---|
| P12-E0 | technical_v1 compliance fix + redeployment decision | PLANNED |
| P12-E1 | GraphRAG infrastructure (graph store, construction, retrieval) | PLANNED |
| P12-E2 | GraphRAG evaluation (A/B: RAG vs GraphRAG, OQ-K02, DJ-016) | PLANNED |
| P12-E3 | Structured debate infrastructure (schemas, debate runner, LangGraph nodes) | PLANNED |
| P12-E4 | Multi-date evaluation (2x2 factorial, 120 runs, OQ-M02, herding) | PLANNED |
| P12-E5 | Baseline script + bitacora + replication notebook | PLANNED |

**David Sections:** SS11.3 GraphRAG, SS12.2.4 Structured Debate, SS5.3 Ensemble
Learning, SS5.6 Complexity Metrics, SS9.4 Fine-Tuning Strategy, SS10.3 Diversity
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 12
**Decision IDs:** DJ-061 through DJ-069 (context: plans/PHASE_12_CONTEXT.md)

---

## Governing Philosophy for This Phase

Phase 11 produced the first fine-tuned agents and demonstrated that LoRA
training converges reliably on financial reasoning tasks (26,433 examples,
all ranks quality_ok). One agent preserved quality perfectly (fundamental_v1,
GR=1.000); one suffered a fixable compliance failure (technical_v1, GR=0.000,
root cause: 0.19% compliance ratio). The diversity measurement is entirely
uninformative (single date, unanimous votes).

Phase 12 addresses two independent mechanisms for improving collective
intelligence:

**GraphRAG** asks: does structural knowledge about entity relationships
improve retrieval quality over dense RAG alone? This is the empirical question
OQ-K02 from the David, and the original DJ-016 framing: "do not add complexity
without evidence."

**Structured Debate** asks: does adversarial deliberation between heterogeneous
agents improve collective decision quality, or does it cause herding? This is
the empirical question from David SS12.2.4. The literature on group
deliberation is mixed (Sunstein, 2006): deliberation can produce "group
polarization" where the group moves toward a more extreme position than the
average individual. Phase 12 measures this directly.

The 2x2 factorial design (base/fine-tuned x no-debate/debate) enables
a genuinely publishable complexity science result: the interaction between
individual agent specialization (fine-tuning) and collective process design
(debate). This is the core of the David's research agenda.

---

## Pre-Phase Decisions

See `plans/PHASE_12_CONTEXT.md` for full rationale (DJ-061 through DJ-069).

Key constraints:
- technical_v1 must be fixed before debate evaluation (DJ-061)
- GraphRAG uses NetworkX + LanceDB, no new venv (DJ-062)
- Graph: ~12 company nodes, 3 macro nodes, ~40 edges (DJ-063)
- Debate: Oxford 1-round, all 5 voting agents (DJ-065)
- Multi-date: 10 dates x 3 tickers x 4 conditions = 120 runs (DJ-067)
- GraphRAG is use_graphrag parameter, mutually exclusive with use_rag (DJ-068)
- Remaining agent fine-tuning is a staged decision (DJ-069)

---

## Wave Plan

```
Wave 1 (parallel — all independent):
  E0-T1: Generate augmented compliance examples
  E1-T1: src/hifi/knowledge/graph_store.py (FinancialGraph schema)
  E1-T2: src/hifi/knowledge/graph_construction.py (build_financial_graph)
  E3-T1: src/hifi/collective/debate.py (DebateTurn, DebateTranscript schemas)

Wave 2 (depends on Wave 1):
  E0-T2: Re-train technical_v1 @ 500 iters (hardware-bound, background)
  E1-T3: src/hifi/knowledge/graph_retrieval.py (GraphRetriever)
  E3-T2: debate.py run_debate_round() logic
  E3-T3: run_debate_ensemble() in ensemble_runner.py

Wave 3 (depends on Wave 2):
  E0-T3: Re-evaluate technical_v1, record deploy decision      <- unblocks E4
  E1-T4: scripts/build_knowledge_graph.py + Makefile target
  E2-T1: use_graphrag parameter in run_ensemble() (DJ-068)
  E3-T4: LangGraph challenge/respond/revise nodes
  E3-T5: Debate prompt templates (challenge, response, revision)

Wave 4 (depends on Wave 3):
  E2-T2: Precision@k comparison RAG vs GraphRAG (Phase 7 eval queries)
  E2-T3: Agent quality comparison with/without GraphRAG
  E4-T1: Multi-date baseline: 10 dates x 3 tickers x 4 conditions
  E4-T2: Herding analysis + diversity measurement

Wave 5 (depends on Wave 4):
  E2-T4: Record OQ-K02 + DJ-016 decision
  E4-T3: Record OQ-M02 + deliberation quality score
  E4-T4: Verification layer gap analysis + Sentiment label design doc
  E5: Baseline script, Makefile, bitacora, notebook
```

Critical path: E0 (technical_v1) -> E4 (debate eval) -> E5.
GraphRAG (E1 -> E2) is parallel and does not block debate evaluation.

---

## Epic P12-E0: technical_v1 Compliance Fix

**Scope:** Resolve the technical_v1 adapter's GR=0.000 format compliance
failure identified in Phase 11. Root cause: compliance:domain ratio of 0.19%
(50 compliance examples in 26,433 total). This epic is a hard gate for debate
evaluation — debate with a format-broken agent produces uninterpretable results.

### Tickets

**P12-E0-T1: Generate augmented compliance examples**

Extend `scripts/generate_compliance_examples.py` to extract additional
verified outputs from:
- Phase 4 ensemble fixture (`tests/fixtures/baseline/phase4_ensemble.json`):
  multi-ticker format, both agents
- Phase 5 verification fixture (`tests/fixtures/baseline/phase5_verification.json`):
  verified outputs with HR=0.000
- Phase 9 collective fixture (`tests/fixtures/baseline/phase9_collective.json`):
  full 6-agent outputs

Target: >= 200 compliance examples (up from ~50).

Output: `data/training/technical_compliance_v2.jsonl`

Tests:
- Unit: compliance example count >= 200
- Unit: JSONL format matches training schema
- Unit: all examples parse as valid TechnicalAnalysis

**P12-E0-T2: Re-train technical_v1 at 500 iterations**

Modify `scripts/run_phase11_finetune.py` to accept `--max-iters` flag.
Re-train with:
- Rank 8 (confirmed optimal)
- 500 iterations (half of original 1000)
- Augmented compliance set from T1
- Same base model path

Output: `data/adapters/technical_v2/`

Tests:
- Unit: check_adapter_quality() returns True
- Unit: adapter directory contains expected files

**P12-E0-T3: Re-evaluate technical_v2 with three-tier protocol**

Run `scripts/run_phase11_evaluation.py` with technical_v2 adapter on AAPL/JPM/XOM
at 2023-03-31.

Decision criteria:
- GR >= 0.720: DEPLOY (record as technical_v2, update serving scripts)
- GR < 0.720 after this attempt: ABANDON technical fine-tuning for Phase 12.
  Record as DJ-061 empirical result. Debate uses base model for Technical Agent.

Output: `tests/fixtures/baseline/phase12_technical_v2_eval.json`

Tests:
- Unit: fixture validates against FineTuneEvaluationResult schema

---

## Epic P12-E1: GraphRAG Infrastructure

**Scope:** Build the financial knowledge graph (NetworkX-backed), graph
construction pipeline, and graph-expanded retrieval. No LLM required —
all construction is deterministic from yfinance metadata and curated seed.

### Tickets

**P12-E1-T1: src/hifi/knowledge/graph_store.py — FinancialGraph**

Pydantic-compatible wrapper around a NetworkX graph with typed operations:

```python
class FinancialGraph:
    """NetworkX-backed financial entity graph."""

    def add_company(self, ticker: str, name: str, sector: str, industry: str) -> None
    def add_sector(self, name: str) -> None
    def add_macro_factor(self, name: str, series_id: str) -> None
    def add_competes_with(self, ticker_a: str, ticker_b: str) -> None
    def add_belongs_to(self, ticker: str, sector: str) -> None
    def add_sensitive_to(self, sector: str, macro_factor: str) -> None
    def get_competitors(self, ticker: str) -> list[str]
    def get_sector_peers(self, ticker: str) -> list[str]
    def get_macro_factors(self, ticker: str) -> list[str]
    def expand_query_tickers(self, ticker: str, max_hops: int = 2) -> list[str]
    def save(self, path: Path) -> None
    def load(cls, path: Path) -> FinancialGraph
    def node_count(self) -> int
    def edge_count(self) -> int
```

Storage: JSON serialization via `networkx.node_link_data()` /
`networkx.node_link_graph()` at `data/knowledge_graph/financial_graph.json`.

Tests:
- Unit: add/get operations roundtrip correctly
- Unit: expand_query_tickers returns correct 1-hop and 2-hop neighborhoods
- Unit: save/load roundtrip preserves graph structure
- Unit: symmetric edges (COMPETES_WITH) work both directions

**P12-E1-T2: src/hifi/knowledge/graph_construction.py — build_financial_graph()**

Deterministic construction from existing data:

```python
def build_financial_graph(
    tickers: list[str],
    competitor_seed: dict[str, list[str]],
    macro_sensitivity: dict[str, list[str]],
    data_dir: str | None = None,
) -> FinancialGraph:
    """Build the financial knowledge graph from yfinance metadata + curated seed."""
```

Steps:
1. For each ticker, read yfinance metadata (sector, industry, name) from
   OHLCV Parquet metadata or a cached info file.
2. Create Company and Sector nodes.
3. Add BELONGS_TO edges (Company -> Sector).
4. Add COMPETES_WITH edges from curated seed.
5. Add MacroFactor nodes (VIX, FFR, CPI).
6. Add SENSITIVE_TO edges from macro_sensitivity dict.

The `competitor_seed` and `macro_sensitivity` are passed as dicts (not
hardcoded) so tests can use minimal graphs.

Default seed provided in `graph_construction.py`:
```python
DEFAULT_COMPETITORS = {
    "AAPL": ["MSFT", "GOOGL"],
    "MSFT": ["AAPL", "GOOGL"],
    "GOOGL": ["AAPL", "MSFT"],
    "JPM": ["BAC", "GS"],
    "BAC": ["JPM", "GS"],
    "GS": ["JPM", "BAC"],
    "XOM": ["CVX"],
    "CVX": ["XOM"],
}

DEFAULT_MACRO_SENSITIVITY = {
    "Technology": ["FFR", "VIX"],
    "Financial Services": ["FFR"],
    "Energy": ["VIX", "CPI"],
}
```

Tests:
- Unit: build_financial_graph produces correct node/edge counts
- Unit: BELONGS_TO edges match yfinance sector metadata
- Unit: COMPETES_WITH edges are symmetric
- Unit: SENSITIVE_TO edges connect sectors to macro factors
- Integration: build from real Phase 10 Parquet metadata

**P12-E1-T3: src/hifi/knowledge/graph_retrieval.py — GraphRetriever**

Extends `KnowledgeRetriever` with graph-based query expansion:

```python
class GraphRetriever:
    """Graph-expanded retrieval: expand query via graph, then dense ANN search."""

    def __init__(
        self,
        store: KnowledgeStore,
        embedding_model: object,
        graph: FinancialGraph,
    ) -> None

    def retrieve(
        self,
        query: str,
        ticker: str,
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """Expand ticker to graph neighbors, then retrieve from expanded set."""

    def format_context(self, chunks: list[DocumentChunk]) -> str:
        """Same as KnowledgeRetriever.format_context — inherited interface."""
```

Implementation of `retrieve()`:
1. `expanded = graph.expand_query_tickers(ticker, max_hops=2)`
2. For each ticker in expanded, search LanceDB with query embedding
3. Merge results by cosine similarity, take top_k
4. Return as list[DocumentChunk]

Tests:
- Unit: GraphRetriever.retrieve returns chunks from expanded ticker set
- Unit: expanded set includes competitors and sector peers
- Unit: format_context produces same format as KnowledgeRetriever
- Unit: falls back gracefully when ticker not in graph

**P12-E1-T4: scripts/build_knowledge_graph.py + Makefile target**

Script: `scripts/build_knowledge_graph.py`
- Reads yfinance ticker info (cached or from Parquets)
- Calls `build_financial_graph()` with default seed
- Saves to `data/knowledge_graph/financial_graph.json`
- Prints summary: node counts, edge counts, ticker coverage

Makefile target: `build-graph`

`scripts/check_env.py` gains a `knowledge-graph` check.

Tests:
- Unit: script produces valid JSON at expected path
- Unit: graph loads and has expected structure

---

## Epic P12-E2: GraphRAG Evaluation

**Scope:** Integrate GraphRetriever into the agent pipeline, run A/B
comparison against Phase 7 RAG, and record the OQ-K02 / DJ-016 decision.

### Tickets

**P12-E2-T1: use_graphrag parameter in run_ensemble()**

Add `use_graphrag: bool = False` to `run_ensemble()` and
`run_debate_ensemble()`.

When True:
- Assert `not use_rag` (mutually exclusive)
- Load FinancialGraph from `data/knowledge_graph/financial_graph.json`
- Pass GraphRetriever to agents' retrieve_context nodes instead of
  KnowledgeRetriever

This requires extending the fundamental_agent and technical_agent
`retrieve_context_node` to accept either retriever type. Both implement
the same interface (`retrieve(query, ticker, top_k)` ->
`list[DocumentChunk]`), so the change is a parameter type widening.

Tests:
- Unit: use_graphrag=True triggers GraphRetriever path
- Unit: use_rag=True and use_graphrag=True raises AssertionError
- Integration: run_ensemble with use_graphrag=True produces valid output

**P12-E2-T2: Precision@k comparison — RAG vs GraphRAG**

Script: `scripts/run_phase12_graphrag_eval.py`

Runs `evaluate_precision_at_k()` (from `src/hifi/knowledge/retrieval.py`)
twice:
1. With KnowledgeRetriever (Phase 7 baseline)
2. With GraphRetriever (Phase 12)

Both use the same 20-query evaluation set and embedding model.

Output: `tests/fixtures/baseline/phase12_graphrag_precision.json`
```json
{
    "rag_precision_at_5": <float>,
    "graphrag_precision_at_5": <float>,
    "delta": <float>,
    "queries_improved": <int>,
    "queries_degraded": <int>,
    "queries_unchanged": <int>
}
```

Tests:
- Unit: fixture validates against expected schema
- Unit: delta is computed correctly

**P12-E2-T3: Agent quality comparison with/without GraphRAG**

Run ensemble on AAPL/JPM/XOM at 2023-03-31 with:
- `use_rag=True` (Phase 7 dense RAG)
- `use_graphrag=True` (Phase 12 graph-expanded RAG)

Compare agent-level GR and ensemble-level accuracy.

Output: `tests/fixtures/baseline/phase12_graphrag_agent_quality.json`

Tests:
- Unit: fixture validates
- Holistic: structural pipeline test for GraphRAG path

**P12-E2-T4: Record OQ-K02 + DJ-016 decision**

Based on T2 and T3 results, record:
- OQ-K02 answer: measured Precision@k improvement (or lack thereof)
- DJ-016 decision: is GraphRAG worth the complexity?
  - If Precision@k improves by >= 5% absolute: ADOPT GraphRAG
  - If Precision@k improves < 5%: DOCUMENT AND KEEP plain RAG
  - Record in `doc/bitacora/PHASE_12_GRAPHRAG_DEBATE.md`

No tests — this is a documentation/decision artifact.

---

## Epic P12-E3: Structured Debate Infrastructure

**Scope:** Build the Oxford 1-round debate mechanism including schemas,
debate runner logic, LangGraph nodes, and prompt templates. All infrastructure
is unit-testable with deterministic stubs (no LLM required).

### Tickets

**P12-E3-T1: src/hifi/collective/debate.py — Schemas**

Pydantic models:

```python
class DebateTurn(BaseModel):
    agent_type: str
    phase: Literal["challenge", "response", "revision"]
    argument: str
    revised_decision: str | None = None
    revised_confidence: float | None = None
    model_id: str

class DebateTranscript(BaseModel):
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
    debate_skipped: bool = False
```

Helper functions:
```python
def identify_minority(signals: list[AgentSignal]) -> tuple[list[str], str]:
    """Return (minority_agent_types, majority_decision)."""

def compute_vote_delta(
    initial_signals: list[AgentSignal],
    revised_signals: list[AgentSignal],
) -> tuple[Literal["converged", "diverged", "unchanged"], int]:
    """Compare initial vs revised votes. Return (delta_type, n_changed)."""
```

Tests:
- Unit: DebateTurn validates all three phases
- Unit: DebateTranscript round-trips to JSON
- Unit: identify_minority correctly classifies minority/majority
- Unit: identify_minority returns ([], majority) when unanimous
- Unit: compute_vote_delta detects convergence, divergence, unchanged
- Unit: EnsembleOutput.debate_transcript field is optional None

**P12-E3-T2: debate.py — run_debate_round()**

Core debate logic:

```python
def run_debate_round(
    signals: list[AgentSignal],
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    use_rag: bool = False,
    use_graphrag: bool = False,
) -> DebateTranscript:
    """Run one Oxford debate round on initial agent signals."""
```

Steps:
1. Call `identify_minority(signals)` — if no minority, return transcript
   with `debate_skipped=True`.
2. For each minority agent, call the agent's `challenge_node` (LLM call).
3. For each majority agent, call the agent's `respond_node` (LLM call).
4. For each agent, call the agent's `revise_node` (LLM call with full
   transcript).
5. Collect revised signals.
6. Call `compute_vote_delta()`.
7. Return complete DebateTranscript.

Tests:
- Unit: run_debate_round with stubbed LLM produces valid transcript
- Unit: debate_skipped=True when all signals agree
- Unit: minority agents generate challenge turns
- Unit: revised signals have same schema as initial signals

**P12-E3-T3: run_debate_ensemble() in ensemble_runner.py**

New entry point that wraps run_ensemble() + debate:

```python
def run_debate_ensemble(
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    use_rag: bool = False,
    use_graphrag: bool = False,
    agents: list[str] | None = None,
) -> EnsembleOutput:
    """Run ensemble with structured debate before final vote."""
```

Steps:
1. Run all agents independently (reuse existing agent-running code from
   run_ensemble()).
2. Collect initial signals.
3. Call `run_debate_round()` on initial signals.
4. If debate_skipped: run_all_methods on initial signals (same as no debate).
5. Else: run_all_methods on revised_signals from debate transcript.
6. Run contrarian on final signals.
7. Run verify_ensemble on final output.
8. Attach debate_transcript to EnsembleOutput.
9. Return.

The existing `run_ensemble()` remains unchanged — full backward compatibility.

Tests:
- Unit: run_debate_ensemble produces EnsembleOutput with debate_transcript
- Unit: when debate is skipped, output matches run_ensemble output
- Integration: structural test with stubbed LLMs

**P12-E3-T4: LangGraph debate nodes**

Three new LangGraph node functions, one per debate phase. Each agent module
(fundamental_agent.py, technical_agent.py, risk_agent.py, macro_agent.py,
sentiment_agent.py) gains:

```python
def challenge_node(state: dict) -> dict:
    """Generate a challenge argument against the majority position."""

def respond_node(state: dict) -> dict:
    """Generate a response to a challenge argument."""

def revise_node(state: dict) -> dict:
    """Revise vote after seeing full debate transcript. Return AgentSignal."""
```

Each node uses the agent's deployed model (fine-tuned if deployed, base
otherwise). Max token limits: challenge=512, response=256, revision=1024.

These nodes are NOT added to the existing agent graphs — they are called
as standalone functions by `run_debate_round()`. This avoids modifying the
tested Phase 3-8 graph structure.

Tests:
- Unit per agent: challenge_node produces valid argument text
- Unit per agent: respond_node produces valid response text
- Unit per agent: revise_node produces valid AgentSignal
- All tests use LLM stubs (no live inference)

**P12-E3-T5: Debate prompt templates**

New prompt templates:

| Template | Location | Purpose |
|---|---|---|
| `challenge_v1.md` | `src/hifi/agents/prompts/` | Challenge argument template |
| `response_v1.md` | `src/hifi/agents/prompts/` | Response to challenge template |
| `revision_v1.md` | `src/hifi/agents/prompts/` | Vote revision with debate context |

Templates are agent-agnostic (parameterized by agent_type, analysis data,
debate transcript). Each includes:
- The agent's original analysis summary
- The majority decision and confidence
- The challenge/response transcript (for later phases)
- Instruction to produce structured output (AgentSignal for revision)

Tests:
- Unit: templates render without errors for each agent type
- Unit: rendered prompts fit within max_tokens budget

---

## Epic P12-E4: Multi-Date Evaluation

**Scope:** Run the 2x2 factorial experiment (DJ-067) across 10 quarterly
dates and 3 tickers. Produce the diversity, herding, and accuracy evidence
needed to answer OQ-M02 and inform DJ-069 (remaining agent fine-tuning).

### Tickets

**P12-E4-T1: scripts/run_phase12_evaluation.py — Multi-date runner**

Orchestrator script that runs all 4 experimental conditions:

```
Condition A: run_ensemble(use_rag=False)           # base, no debate
Condition B: run_ensemble(use_rag=False)            # fine-tuned, no debate
             (with HIFI_*_FINETUNE_URL set)
Condition C: run_debate_ensemble(use_rag=False)     # base, with debate
Condition D: run_debate_ensemble(use_rag=False)     # fine-tuned, with debate
             (with HIFI_*_FINETUNE_URL set)
```

For each of 10 dates x 3 tickers:
- Run all 4 conditions
- Save per-run JSON to `data/evaluation/phase12/{condition}_{ticker}_{date}.json`
- Checkpoint after each run (resumable from interruption)

Tests:
- Unit: script argument parsing
- Unit: checkpoint/resume logic
- Holistic: structural test verifying output schema per condition

**P12-E4-T2: Herding and diversity analysis**

Compute across the 120 runs:

| Metric | Formula | Source |
|---|---|---|
| Mean pairwise_diversity per condition | mean(pd_i) for i in condition | EnsembleOutput |
| Mean disagreement_entropy per condition | mean(H_i) | EnsembleOutput |
| Herding coefficient per condition | mean(a_t) where a_t = majority fraction | David SS5.6.3 |
| Vote delta distribution | count(converged, diverged, unchanged) | DebateTranscript |
| Accuracy per method per condition | Phase 10 labeler | 60-day forward return |
| Interaction effect | (D-B) - (C-A) per metric | 2x2 factorial |

Output: `tests/fixtures/baseline/phase12_factorial_results.json`

Tests:
- Unit: metric computations match expected values on synthetic data
- Unit: interaction effect sign is computed correctly

**P12-E4-T3: Record OQ-M02 + deliberation quality score**

Based on T2 results:

- **OQ-M02 (diversity preserved):** Compare mean pairwise_diversity across
  conditions A vs B (fine-tuning effect) and A vs C (debate effect).
  Threshold: diversity preserved if degradation < 10%.

- **Deliberation quality:** Compare accuracy(C) vs accuracy(A) and
  accuracy(D) vs accuracy(B). Does debate improve, degrade, or not affect
  collective accuracy?

- **Herding assessment:** If herding coefficient increases by > 0.1 between
  no-debate and debate conditions, flag debate as inducing herding.

Record in bitacora.

**P12-E4-T4: Verification gap analysis + Sentiment label design**

Two deliverables (design documents, not code):

1. **Verification gap analysis:** Document exactly what must change in
   `verify_agent()`, `FIELD_ALIAS_TABLE`, and `extractor.py` to support
   RiskAnalysis, MacroAnalysis, and SentimentAnalysis schemas. Estimate
   effort in tickets. Output: section in bitacora.

2. **Sentiment Agent training label design:** Propose training labels for
   Sentiment Agent fine-tuning. Candidate: MD&A management tone
   (cautious/neutral/optimistic) mapped to Sell/Hold/Buy via keyword-based
   deterministic classifier on SEC filing text. Evaluate feasibility with
   the Phase 7 SEC filing corpus. Output: section in bitacora.

No code produced. These are Phase 13 input artifacts.

---

## Epic P12-E5: Baseline Script + Bitacora + Notebook

**Scope:** Produce the repeatable baseline generation script, Makefile targets,
scientific bitacora, and frozen replication notebook.

### Tickets

**P12-E5-T1: scripts/run_phase12_baseline.py**

Runs a minimal baseline (3 tickers, 1 date, all conditions) to populate
`tests/fixtures/baseline/phase12_baseline.json`.

**P12-E5-T2: Makefile targets**

| Target | Command | Purpose |
|---|---|---|
| `build-graph` | `uv run python scripts/build_knowledge_graph.py` | Build knowledge graph |
| `baseline-phase12` | `uv run python scripts/run_phase12_baseline.py` | Generate baseline fixture |
| `eval-phase12` | `uv run python scripts/run_phase12_evaluation.py` | Full 120-run evaluation |
| `graphrag-eval` | `uv run python scripts/run_phase12_graphrag_eval.py` | Precision@k comparison |

Each `baseline-*` target runs fixture unit tests + holistic tests after
generation (closed-loop validation, Phase 9 Makefile policy).

`scripts/check_env.py` gains: `knowledge-graph`, `phase12-fixture` checks.

**P12-E5-T3: doc/bitacora/PHASE_12_GRAPHRAG_DEBATE.md**

Scientific bitacora following established pattern:
- Objective and governing philosophy
- Architecture decisions (DJ-061 through DJ-069) with rationale
- GraphRAG results (Precision@k comparison, DJ-016 decision)
- Debate results (herding, diversity, accuracy)
- 2x2 factorial results table
- OQ-K02 and OQ-M02 answers
- Verification gap analysis
- Sentiment label design document
- Implementation surprises and lessons learned
- Open questions for Phase 13

**P12-E5-T4: notebooks/phase12_graphrag_debate_replication.ipynb**

Frozen narrative notebook that reads Phase 12 artifacts (no LLM calls,
runs in < 30s). Sections:
1. Knowledge graph visualization (NetworkX drawing)
2. GraphRAG vs RAG Precision@k comparison
3. 2x2 factorial results table
4. Debate transcript examples (1 per condition)
5. Diversity and herding metrics visualization
6. OQ-K02 and OQ-M02 conclusions

---

## Success Criteria

### Mandatory (phase complete when all pass)

- [ ] technical_v1 either redeployed (GR >= 0.720) or formally abandoned with evidence
- [ ] Knowledge graph constructed with >= 10 company nodes, correct sector/competitor edges
- [ ] GraphRetriever operational and passing unit + integration tests
- [ ] OQ-K02 answered: Precision@k measured for RAG vs GraphRAG
- [ ] DJ-016 decision recorded with quantitative evidence
- [ ] Structured debate implemented (Oxford 1-round, 5 voting agents)
- [ ] run_debate_ensemble() produces valid EnsembleOutput with DebateTranscript
- [ ] Multi-date evaluation on >= 10 dates (30+ ticker-date pairs per condition)
- [ ] OQ-M02 resolved with multi-date evidence (not single-date vacuous result)
- [ ] Herding coefficient computed for debate vs no-debate conditions
- [ ] All existing tests pass (>= 1001) + Phase 12 tests (estimated +80-100)
- [ ] 0 lint errors
- [ ] Bitacora complete
- [ ] Replication notebook complete

### Publication-Grade

- [ ] 2x2 factorial interaction effect computed and interpretable
- [ ] GraphRAG latency cost quantified (ms overhead per query)
- [ ] Debate transcript examples saved as Dataset Family D artifacts
- [ ] Verification gap analysis and Sentiment label design documented
- [ ] Vote delta distribution (converged/diverged/unchanged) reported across 30+ runs

---

## Open Questions to Resolve During Phase 12

1. **Does GraphRAG improve Precision@k by >= 5% over dense RAG?** (OQ-K02)
   If not, the additional graph infrastructure is not justified.

2. **Does debate increase herding (kappa)?** If kappa increases by > 0.1
   between no-debate and debate conditions, debate is inducing the pathology
   it was designed to prevent. This would be a negative but publishable result.

3. **Is the interaction effect positive?** (D-B) - (C-A): does debate benefit
   more from fine-tuned agents (heterogeneous decision boundaries) than from
   base agents (shared priors)?

4. **Does the base Technical Agent still achieve GR=1.000 across multiple dates?**
   Phase 11 showed GR=1.000 on 2023-03-31 (vs Phase 5 baseline 0.667). If this
   holds across 10 dates, the Phase 5 GR weakness was a model configuration
   artifact, not a persistent problem.

5. **What is the debate participation rate?** On how many of the 30 ticker-dates
   do agents actually disagree (non-trivial debate)? If agents are unanimous on
   > 80% of dates, the debate mechanism has limited scope for improvement.
