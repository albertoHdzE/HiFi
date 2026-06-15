# Phase 13: Verification Completeness, Sentiment Intelligence, and System Resilience
## Context and Pre-Phase Decisions

**Gathered:** 2026-06-14
**Status:** Ready for planning

---

## Phase Boundary

Phase 13 absorbs two independent obligation streams:

### Stream 1 — Phase 12 Deferred Items (Tier 1, critical path)

Phase 12 explicitly deferred the following because they required evidence that only
Phase 12 could produce. That evidence is now available (or pending LLM runs that are
in progress):

1. **Verification layer extension for Phase 8 agents** — `verify_agent()` accepts only
   `FundamentalAnalysis | TechnicalAnalysis`. HR/GR baselines cannot be established
   for Risk, Macro, or Sentiment without extending the verification layer. This blocks
   all fine-tuning for those agents (Protocol SS1: every layer earns its place with
   a measurement).

2. **Sentiment Agent fine-tuning** — qwen2.5-coder-32b is shared by Fundamental,
   Technical, and Sentiment. Fine-tuning Sentiment is the primary diversity mechanism
   for the qwen2.5-coder-32b trio (David SS10.3, DJ-069). Go/no-go requires: (a)
   verification layer extended to SentimentAnalysis, (b) >= 200 MD&A training examples
   with adequate Sell representation, (c) Phase 12 diversity evidence shows Sentiment
   is the bottleneck.

3. **Multi-round debate** — Phase 12 implements one Oxford round. The convergence
   criterion for additional rounds cannot be designed without Phase 12 transcript
   evidence of how often one round is sufficient. Phase 13 has that data.

4. **LLM-extracted knowledge graph** — David SS11.3 / OQ-K03: "Manual vs. automatic
   graph construction?" Phase 13 executes this **only if Phase 12 OQ-K02 is positive**
   (GraphRAG improves Precision@k by >= 5%). If OQ-K02 is negative, additional graph
   nodes cannot fix a fundamental retrieval precision problem at this scale.

### Stream 2 — Protocol Phase 13 Items (Tier 2, parallel)

From `doc/HIFI_PROTOCOL_V1.md` Phase 13 (SS13a-d):

- **Agent memory (SS13a / David SS10.4):** Agents accumulate decision history,
  outcome feedback, and calibration data. Enables adaptation.
- **Drift detection (SS13c / David SS14.4):** Statistical monitors for data drift,
  agent drift, and collective drift. Required before Phase 14 (paper trading).
- **Synthetic scenarios (SS13b / David SS8.7):** Stress test under rare events.
  Dataset Family F population.
- **Dataset families completion (SS13d / David SS8.5-8.6):** Standardize Dataset
  Family E (agent interactions), audit all families.

### Explicitly OUT of scope for Phase 13

- Risk Agent fine-tuning (gemma-3-4b — different architecture, different LoRA
  dynamics, requires separate investigation; Phase 14+)
- Macro Agent fine-tuning (qwen3.5-27b reasoning-distilled — similar caveat)
- Adaptive aggregation (SS12.2.5 — requires substantially more labeled data than
  Phase 13 produces)
- OQ-AG03 (LLM calibration) — Phase 14 (paper trading data needed)
- Spanner emulator — Phase 15 (DJ-070)

---

## Evidence Base for Decisions

### Phase 12 Empirical Inputs (some pending LLM runs)

| Input | Status | Phase 13 dependency |
|---|---|---|
| technical_v2 GR (post compliance fix) | Pending (E0-T2/T3) | Determines if technical FT enters Phase 13 evaluation |
| 2x2 factorial OQ-M02 (diversity preservation) | Pending (E4) | Confirms/refutes Sentiment as diversity bottleneck |
| Debate OQ-D01 (herding increase A→C) | Pending (E4) | Calibrates multi-round convergence criterion |
| Debate OQ-D03 (participation rate) | Pending (E4) | If >80% unanimous, multi-round adds limited value |
| GraphRAG OQ-K02 (Precision@k delta) | **ANSWERED NEGATIVE** (doc P@5 delta=0.000, 2026-06-15) | E3 NOT triggered |
| Phase 12 debate transcripts | Pending (E4) | Raw material for convergence criterion calibration |
| MD&A corpus assessment | Phase 12 design only | Must validate label count, Sell balance in Phase 13 |

### Verification Layer: Current State

```
verify_agent(FundamentalAnalysis) → HR/GR ✓ (Phase 5)
verify_agent(TechnicalAnalysis)   → HR/GR ✓ (Phase 5)
verify_agent(RiskAnalysis)        → NOT IMPLEMENTED
verify_agent(MacroAnalysis)       → NOT IMPLEMENTED
verify_agent(SentimentAnalysis)   → NOT IMPLEMENTED (structural constraint)
```

`_named_tool_results()` in `verifier.py` has two branches only. `FIELD_ALIAS_TABLE`
in `extractor.py` covers numerical fields from financial_ratios, growth_metrics,
valuation_context, macro_snapshot, technical_indicators, and risk_metrics.

RiskAnalysis and MacroAnalysis have numerical tool results (risk_metrics,
macro_snapshot) that are already partially in the FIELD_ALIAS_TABLE. Extension
is straightforward.

SentimentAnalysis has NO numerical MCP tools — it operates purely on RAG-retrieved
text. The regex + alias approach produces 0 extractable claims. A distinct
verification strategy is required (DJ-072).

### Agent Schemas (Phase 8)

| Agent | Analysis Schema | Tool results | Fine-tunable via existing HR/GR? |
|---|---|---|---|
| Fundamental | FundamentalAnalysis | financial_ratios, growth_metrics, valuation_context, macro_snapshot | ✓ (fundamental_v1 deployed) |
| Technical | TechnicalAnalysis | technical_indicators, risk_metrics | ✓ (technical_v2 pending) |
| Risk | RiskAnalysis | risk_metrics | ✗ — verify_agent not implemented |
| Macro | MacroAnalysis | macro_snapshot | ✗ — verify_agent not implemented |
| Sentiment | SentimentAnalysis | None (RAG-only) | ✗ — different verification strategy needed |

### Dataset Family Status (David SS8)

| Family | Contents | Status |
|---|---|---|
| A — Market Observation | OHLCV, macro series (Parquet) | Complete (Phase 1) |
| B — Feature Datasets | SEC filings in LanceDB | Complete (Phase 7) |
| C — Reference Strategy | Training labels (Phase 11 JSONL) | Complete (Phase 11) |
| D — Explanation Datasets | Agent reasoning traces, debate transcripts | Partially populated (Phase 12) |
| E — Agent Interaction | Votes, confidence, disagreement records | Exists as EnsembleOutput JSON; needs schema |
| F — Synthetic Scenarios | Stress-test scenarios | NOT STARTED |
| G — Evaluation Datasets | Phase 3/4/5/7/9/10 baselines (fixtures) | Partially populated |

---

## Pre-Phase Decisions (DJ-071 through DJ-079)

### DJ-071: Phase 13 Scope — Deferred Items Take Tier 1 Priority

**Problem:** The Protocol's Phase 13 (memory, scenarios, drift) and the Phase 12
deferred items (verification, Sentiment FT, debate, graph) are separate work streams
that both belong in Phase 13. Priority must be set.

**Decision:** Tier 1 = Phase 12 deferred items (verification layer, Sentiment FT,
multi-round debate). Tier 2 = Protocol Phase 13 items (memory, drift, scenarios).
Tier 1 items are prerequisites for the Phase 15 ablation study ("Remove fine-tuning"
requires fine-tuned Sentiment to exist; "Remove verification" requires full verification
coverage). Tier 2 items run in parallel where independent. If Tier 1 exceeds Phase 13
capacity, Tier 2 items carry forward to Phase 14 prep.

**Rationale:** The ablation study (David SS15) requires the full system — including
all available fine-tuned agents and full verification coverage — to exist before
measuring the contribution of each component. Sentiment fine-tuning cannot be deferred
past Phase 13 without jeopardising Phase 15.

### DJ-072: verify_agent() Extension for SentimentAnalysis

**Problem:** SentimentAnalysis has no numerical MCP tool results. The Phase 5
regex + alias-table + tolerance-check approach is inapplicable. The agent cites
qualitative assertions ("management tone was cautious") not numerical claims.

**Options evaluated:**

| Option | Approach | Drawback |
|---|---|---|
| Compliance-only | HR=0.0 by definition, measure only format | Uninformative — no content grounding |
| LLM-as-judge | Second LLM assesses whether claims match context | Introduces inference in verification chain (DJ-019) |
| Citation grounding (SGR) | Check if notable_signals appear in retrieved context | Deterministic, specific to Sentiment |
| Skip verification | Accept verification gap | Violates Protocol SS1 |

**Decision:** Implement **Sentiment Grounding Rate (SGR)** via citation grounding:
each item in `notable_signals` is checked against the retrieved context chunks that
were passed to the agent. An item is "grounded" if it appears verbatim (exact string
match after normalisation) in any retrieved chunk. SGR = grounded_items / total_items.

This gives:
- HR = 0.0 by definition (Sentiment cannot hallucinate numerical values it never cites)
- SGR replaces GR for Sentiment: measures citation honesty, not numerical accuracy

The metric is defensible: "Did the agent fabricate a quotation from a document it
didn't retrieve?" is a binary, deterministic question. It is analogous to GR in
epistemological intent (grounding the rationale in MCP tool results) but adapted to
the Sentiment agent's information access pattern.

Phase 13 deliverable: SGR baseline on AAPL/JPM/XOM at 2023-03-31, establishing
whether notable_signals are properly grounded before fine-tuning.

Remaining open: edit-distance tolerance for "near-verbatim" matches. Phase 13 uses
exact match only; calibration deferred to Phase 14.

Implementation: `verify_sentiment_agent(analysis: SentimentAnalysis, retrieved_context: str)`
as a separate function (not overloading `verify_agent`). `EnsembleVerificationReport`
gains an optional `sentiment_report: SentimentGroundingReport | None = None`.

### DJ-073: Sentiment Agent Training Label Strategy

**Problem:** MD&A management tone → Buy/Hold/Sell labels. Phase 12 proposed this
approach; Phase 13 validates feasibility before execution.

**Confirmation criteria (must all pass before fine-tuning):**
1. Phase 7 MD&A corpus (EDGAR, 10-K/10-Q, 2018-2023) yields >= 200 labeled examples
2. Sell class has >= 30 examples (2022 bear market period)
3. SGR baseline >= 0.5 on retrieved context (agent uses retrieved text meaningfully)

**Label strategy:**
- Primary: keyword-based deterministic tone classifier
  - Cautious keywords (↓guidance, impairment, headwinds, uncertainty, restructuring) → Sell
  - Optimistic keywords (strong growth, record revenue, expanding margins, accelerating) → Buy
  - Neutral (default for no dominant signal) → Hold
- Sell class augmentation: for 2022-01 to 2022-12, supplement with outcome-based
  labels: if management tone is cautious AND 60-day forward return < -10%, override
  label to Sell. This provides signal anchoring without pure look-ahead bias (the
  tone signal is still deterministic; the price filter only selects the clearest examples).

**Fallback (if < 30 Sell examples):** Abandon MD&A tone as primary approach for
Phase 13. Sentinel outcome: document as an empirical limitation. Sentiment Agent
fine-tuning deferred to Phase 14 using paper trading data as the Sell signal source.

**Decision gate document:** `scripts/validate_sentiment_corpus.py` — runs the
classifier on the EDGAR corpus, reports class distribution, confirms go/no-go.

### DJ-074: Multi-Round Debate — Convergence Criterion

**Problem:** Phase 12 hard-codes one debate round. Multi-round is warranted if Phase 12
shows participation rate > 20% (OQ-D03) and non-trivial vote_delta distribution.

**Design criteria:**
Phase 12 OQ-D03 answers: "On what fraction of dates does debate actually run?" If
< 20% of dates have non-unanimous initial votes, multi-round is of limited scope.
If > 20%, multi-round adds value for those dates.

**Decision:** Implement multi-round with:

```
Convergence criterion: vote_stability — stop when majority_decision(round_k)
  == majority_decision(round_{k-1}), i.e., the collective decision did not change
Hard cap: max_rounds=3 (prevents oscillation)
Minimum: 1 round always runs (even if initial vote is unanimous, 1 round records
  the transcript; debate_skipped=True only when majority is unanimous AND no
  minority exists)
```

New parameter: `run_debate_ensemble(max_rounds: int = 1)`. Phase 12 default preserved.
Phase 13 evaluates max_rounds=2 (one additional round from Phase 12 baseline).

**New OQ:** OQ-D04: Does adding a second round reduce herding compared to 1 round?
(Expected: no — herding is determined by architecture diversity, not round count.
But this is an empirical claim.)

### DJ-075: LLM-Extracted Knowledge Graph — Gate Criterion

**Problem:** Expanding the FinancialGraph from 11 hand-coded companies to 25-30 via
LLM extraction from SEC MD&A requires justification that the graph adds retrieval value.

**Gate:** Execute LLM extraction ONLY if Phase 12 OQ-K02 result shows mean Precision@k
(GraphRAG) ≥ mean Precision@k (dense RAG) + 5%.

**If gate passes (OQ-K02 positive):**
- Extract competitor entities from MD&A "Competition" sections using few-shot LLM prompt
- Target: 15-20 additional company nodes, cross-sector relationships
- Evaluation: measure Precision@k with expanded graph vs. Phase 12 hand-coded graph
- Answers OQ-K03 (manual vs. automatic graph construction)

**If gate fails (OQ-K02 negative or < 5% delta):**
- Document per DJ-016: "GraphRAG at 12-node scale does not improve Precision@k.
  LLM-extracted graph would have the same ceiling — the bottleneck is retrieval
  mechanism, not graph size."
- Convert E3 into a brief analysis: WHY doesn't graph expansion help?
  Hypothesis: BFS expansion at 2-hop over 11 nodes retrieves the same documents
  as dense search because the corpus density (documents per ticker) is too low
  for graph expansion to reach meaningfully different documents.
- This hypothesis, if confirmed, is a publishable negative result (David SS1 principle:
  document what doesn't work as carefully as what does).

### DJ-076: Agent Memory Implementation Pattern

**Problem:** David SS10.4 requires agents to accumulate decision history and outcome
feedback. Three implementation options with different complexity/benefit tradeoffs.

**Options evaluated:**

| Pattern | Complexity | Context window cost | Retrieval latency |
|---|---|---|---|
| In-context prefix (last N decisions) | Low | ~200 tokens / 3 records | None |
| RAG-based (LanceDB embedding) | Medium | Dynamic | LanceDB query |
| No memory | None | None | None |

**Decision:** In-context structured prefix (last 3 decisions per ticker).

Format injected before analytical prompt:
```
[Agent Memory — last 3 decisions for {ticker}]
{as_of_date}: {decision} (confidence={confidence:.2f}) → actual_60d_return={return:.1%}
...
If no history: "No prior decisions recorded for {ticker}."
```

**Rationale:**
- Phase 10 performance_store.py already stores (ticker, date, decision, outcome) records
- 3 decisions × ~200 tokens = 600 tokens total — well within qwen2.5-coder-32b 32K context
- Deterministic, interpretable, no new infrastructure
- RAG-based memory adds latency and complexity for no benefit at < 100 records/ticker

**OQ-AG02 (David):** Memory decay not implemented in Phase 13 (uniform weight for
last 3 decisions). Recency-weighted decay calibrated from paper trading in Phase 14.

**Measurement:** Does memory change decision rate vs. no-memory baseline? Run
memory vs. no-memory comparison on the 30 Phase 12 evaluation dates (ticker-date pairs
already evaluated). Decision agreement rate gives a "memory influence" signal.

### DJ-077: Drift Detection Scope

**Problem:** David SS14.4 defines four drift types. Full concept drift requires labeled
outcomes across multiple market regimes — insufficient data exists before paper trading.

**Decision:** Phase 13 implements three monitors:

1. **Data drift (Kolmogorov-Smirnov):** Compare recent 20-day distribution of key
   market features (realized_vol, P/E, RSI) to Phase 10 historical baseline (2020-2022).
   Alert threshold: KS p-value < 0.05 for any feature. Implementation in
   `src/hifi/collective/drift.py`.

2. **Agent drift (chi-squared):** Compare last 20 decisions per agent to baseline
   Buy/Hold/Sell proportions from Phase 10 bootstrap. Alert if chi-squared p < 0.05.
   Detects whether an agent has started systematically biasing toward one decision.

3. **Collective drift (CUSUM):** Track herding_coefficient over time. Alert if
   CUSUM statistic exceeds 3σ above baseline herding level from Phase 12 evaluation.

**Not implemented in Phase 13:**
- Concept drift (relationship between features and outcomes) — requires Phase 14 data
- Covariate shift detection — requires distributional modeling beyond current scope

**Answers:** OQ-E01 (window sizes and thresholds) — Phase 13 calibrates on Phase 10
data; Phase 14 validates on live paper trading observations.

### DJ-078: Synthetic Scenarios — Historical Scenarios for Phase 13

**Problem:** David SS8.7 (Dataset Family F) requires synthetic scenarios for stress
testing. True synthetic financial data generation is an open research problem
(David: "Most generators fail precisely when they are most needed").

**Decision:** Phase 13 uses **historical scenarios** (not generated data) for Dataset
Family F. Rationale: historical scenarios avoid generation methodology artifacts while
still achieving the goal of testing agent behavior under extreme conditions.

Three scenarios, all using Phase 1 market data already acquired:

| Scenario ID | Date | Ticker(s) | Event | Expected behavior |
|---|---|---|---|---|
| F-001 | 2020-03-16 (Black Monday II) | AAPL, JPM, XOM | COVID crash: single-day drop >10% | Risk agent: Sell; Macro: Sell; Fundamental: Hold or Sell |
| F-002 | 2022-03-31 (rate shock) | AAPL, JPM, XOM | FFR rising, CPI at 8.5% | Macro agent: Sell or Hold for rate-sensitive; Risk: Sell |
| F-003 | 2023-02-02 (AAPL earnings beat) | AAPL | Q1 2023 beat $0.06 EPS | Fundamental agent: Buy or Hold |

**ScenarioEvaluator:** Runs `run_ensemble()` on the scenario (ticker, date), compares
the collective decision to the "expected direction" (not a hard label, but a documented
plausible outcome). Results go to `data/scenarios/` (Dataset Family F).

**Methodological note (honest limitation):** Historical scenarios are not synthetic.
They are subset of the existing evaluation universe. They do NOT test agent behavior
on truly unseen distribution shifts. True generative synthetic scenario creation
(GARCH, VAE) is deferred to Phase 16 (publication research).

### DJ-079: Dataset Families Audit

**Decision:** Phase 13 produces and standardizes the following:

- **Dataset Family E (Agent Interaction):** Standardize `EnsembleOutput` JSON
  files as the canonical Dataset Family E artifact. Add `data/interactions/README.md`
  with schema documentation. Debate transcripts from Phase 12 (when available)
  go to `data/interactions/debate/`.

- **Dataset Family F (Synthetic Scenarios):** Three historical scenario evaluations
  from E6 (DJ-078). Stored at `data/scenarios/{F-001,F-002,F-003}.json`.

- **Dataset Family G (Evaluation):** Audit existing baseline fixtures. Ensure
  phase12_factorial_results.json (when generated) is added as the primary
  multi-date evaluation dataset.

Full public release (dataset cards, Hugging Face) is Phase 16 scope.

---

## Open Questions Generated by Phase 13

| ID | Question | Resolution target |
|---|---|---|
| OQ-V01 | Is the SGR baseline for SentimentAnalysis >= 0.5? | Phase 13 E0 evaluation |
| OQ-V02 | Does extending verify_agent to RiskAnalysis reveal systematic macro hallucinations? | Phase 13 E0 evaluation |
| OQ-S01 | Does MD&A corpus yield >= 200 labeled examples with >= 30 Sell? | Phase 13 E1-T1 gate |
| OQ-S02 | Does sentiment_v1 achieve SGR >= 0.720 post fine-tuning? | Phase 13 E1-T3 evaluation |
| OQ-D04 | Does a second debate round reduce herding vs. one round? | Phase 13 E2 |
| OQ-M03 | Does in-context agent memory change decision rate? | Phase 13 E4 |
| OQ-DR01 | Does drift monitoring detect the 2022 regime change in the Phase 10 data? | Phase 13 E5 |

---

## Canonical References

- `doc/HIFI_DAVID.md` SS10.3 — Diversity Requirements (5 dimensions, 2-minimum)
- `doc/HIFI_DAVID.md` SS10.4 — Agent Memory (decision history, outcome feedback)
- `doc/HIFI_DAVID.md` SS14.4 — Drift Detection (KS, chi-squared, CUSUM)
- `doc/HIFI_DAVID.md` SS8.7 — Synthetic Scenarios (Dataset Family F)
- `doc/HIFI_DAVID.md` SS11.3 — GraphRAG (OQ-K03 manual vs. automatic)
- `doc/HIFI_DAVID.md` SS9.4 — Fine-Tuning Strategy (demonstrably outperforms)
- `doc/HIFI_DAVID.md` SS15 — Ablation studies (require fine-tuned Sentiment)
- `src/hifi/verification/verifier.py` — verify_agent (extension target)
- `src/hifi/verification/extractor.py` — FIELD_ALIAS_TABLE (extension target)
- `src/hifi/agents/schemas.py` — RiskAnalysis, MacroAnalysis, SentimentAnalysis
- `doc/bitacora/PHASE_12_GRAPHRAG_DEBATE.md` — Phase 13 inputs, OQ-K02 results
- `plans/PHASE_12_CONTEXT.md` — DJ-069 (fine-tuning staging rationale)

### DJ-080: Gemma 4 12B as Sentiment Agent Base Model

**Problem:** Three agents (Fundamental, Technical, Sentiment) share qwen2.5-coder-32b
(DJ-032). Fine-tuning is the primary diversity mechanism for this trio, but Sentiment
is not yet fine-tuned. Gemma 4 12B (released June 3, 2026, Apache 2.0, Google) is now
available for local inference on Apple Silicon via MLX.

**Evaluation summary:**
- Gemma 4 12B: 11.95B params, dense encoder-free multimodal Transformer, 256K context
- MLX Q4 quantized: ~6.7 GB (fits easily alongside qwen2.5-coder-32b ~18 GB)
- Available: `mlx-community/gemma-4-12B-it-4bit` and official QAT checkpoint
- LM Studio: GGUF and MLX backends both supported
- Fine-tuning: Unsloth LoRA supported; mlx_lm conversion available
- LiveCodeBench v6: 72% (competitive for a 12B general model)
- Note: The "gemma-4-12b-coder-fable-5" name refers to a community fine-tune by
  yuxinlu1, NOT an official Google release. The base model is google/gemma-4-12B-it.

**Decision:** Switch Sentiment agent base model from qwen2.5-coder-32b to Gemma 4 12B
as a **Phase 13 E1 prerequisite task**. Rationale:

1. **Maximum diversity impact:** Breaks the 3-agent qwen2.5-coder-32b monopoly into
   2+1 (two agents on Qwen 2.5, one on Gemma 4). This is the highest-ROI model change.
2. **Zero baselines invalidated:** Sentiment has no fine-tuned version, no factorial
   condition results (factorial uses only Fundamental + Technical). Phase 13 E0 SGR
   baseline (0.167) was designed to be measured BEFORE fine-tuning — re-measuring
   after the model swap is the intended workflow.
3. **Optimal timing:** Phase 13 E1 (Sentiment fine-tuning) has not started. Switching
   the base model before fine-tuning is free; switching after would invalidate the
   adapter.
4. **Memory efficiency:** Gemma 4 12B Q4 (~6.7 GB) is lighter than qwen2.5-coder-32b
   Q8 (~18 GB), reducing total inference memory when all agents run concurrently.
5. **Architecture diversity:** Gemma 4's encoder-free multimodal Transformer is
   architecturally distinct from Qwen 2.5's decoder-only design. This adds a third
   architecture family to the ensemble (Qwen dense, Gemma dense multimodal,
   Qwen reasoning-distilled).

**Model diversity table after DJ-080:**

| Agent | Model | Family | Change |
|---|---|---|---|
| Fundamental | qwen2.5-coder-32b (+ fundamental_v1) | Qwen 2.5 | -- |
| Technical | qwen2.5-coder-32b (+ technical_v2 pending) | Qwen 2.5 | -- |
| Risk | gemma-3-4b | Google Gemma 3 | -- |
| Macro | qwen3.5-27b reasoning | Qwen 3.5 | -- |
| Sentiment | **gemma-4-12b-it** | **Google Gemma 4** | **NEW** |
| Contrarian | qwen3.5-35b reasoning | Qwen 3.5 | -- |

**Secondary recommendation (NOT implemented in Phase 13):** Upgrade Risk agent from
gemma-3-4b to gemma-4-12b (3x capacity within same Google family). This could improve
alias coverage (38.9% → higher via better terminology) and risk metric quality. But
it does not address the diversity bottleneck (same family). Deferred to Phase 14+.

**Implementation tasks (Phase 13 E1 prerequisite):**
- T0-1: Download `mlx-community/gemma-4-12B-it-4bit` or `-qat-4bit` to LM Studio
- T0-2: Add Gemma 4 12B model config to `src/hifi/agents/lm_client.py`
- T0-3: Update `sentiment_agent.py` to use the Gemma 4 model identifier
- T0-4: Re-run Phase 13 E0-T6 SGR baseline with Gemma 4 Sentiment (re-measure SGR)
- T0-5: Update DJ-032 model diversity table in STATUS.md

**Baselines requiring re-measurement after swap:**
- Phase 13 E0 SGR baseline (0.167 → TBD with Gemma 4)
- Phase 10 bootstrap Sentiment accuracy (if applicable)
- No factorial results affected (factorial uses only Fundamental + Technical)

---

## Deferred from Phase 13

- **Risk Agent fine-tuning** (gemma-3-4b): LoRA dynamics for small non-reasoning
  models require separate investigation. Phase 14+.
- **Macro Agent fine-tuning** (qwen3.5-27b reasoning): reasoning-trace supervision
  vs. output-only LoRA is an open question (OQ-P13-03). Phase 14+.
- **Adaptive aggregation** (David SS12.2.5): requires substantially more labeled
  outcome data than Phase 13 produces. Phase 14 after paper trading.
- **Concept drift detection**: requires Phase 14 live data to calibrate.
- **OQ-AG01 (Contrarian design)**: test both "sees" / "doesn't see" consensus.
  Phase 14 (live evaluation context).
- **True synthetic data generation** (GARCH, VAE for Dataset Family F): Phase 16
  publication research.
