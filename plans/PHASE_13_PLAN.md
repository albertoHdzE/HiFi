# Phase 13: Verification Completeness, Sentiment Intelligence, and System Resilience
## Epic and Ticket Plan

**Phase status:** NOT STARTED
**Pre-phase decisions:** DJ-071 through DJ-079 (see PHASE_13_CONTEXT.md)
**David sections:** SS9.4, SS10.3, SS10.4, SS11.3, SS14.4, SS8.7, SS15
**Next DJ number at phase start:** DJ-080

---

## Success Criteria

- [ ] verify_agent() covers all 5 voting agents (Risk, Macro, Sentiment)
- [ ] Sentiment Agent fine-tuning: go/no-go decided with evidence; if go, SGR >= 0.720
- [ ] Multi-round debate (max 2 additional rounds) implemented and evaluated
- [ ] Agent memory implemented and memory influence measured on 30 evaluation dates
- [ ] Three drift monitors operational (KS, chi-squared, CUSUM)
- [ ] Three synthetic scenarios evaluated (Dataset Family F populated)
- [ ] Dataset Family E (agent interactions) standardized with README
- [ ] Replication notebook: frozen, no LLMs, < 30s
- [ ] Tests: >= 1250 passing, 0 lint errors

---

## Wave Structure

```
Wave 1 (no LLM runs required, parallel work):
  E0: Verification layer extension (Risk + Macro + Sentiment)
  E4: Agent memory schema and store
  E5: Drift detection module
  E6: Synthetic scenario framework

Wave 2 (requires Phase 12 LLM results and E0 complete):
  E1: Sentiment Agent fine-tuning (gate on OQ-S01)
  E2: Multi-round debate calibration + implementation
  E3: LLM-extracted graph (gate on OQ-K02 positive)

Wave 3 (integration, documentation, validation):
  E7: Dataset families audit and standardization
  E8: Replication notebook + bitacora + memory update
```

---

## Epic E0: Verification Layer Generalization (DJ-072)

**Objective:** Extend `verify_agent()` and `verify_ensemble()` to all 5 voting agents.
Risk and Macro get standard numerical claim verification via their tool result dicts.
Sentiment gets the new SGR (Sentiment Grounding Rate) metric.

**Ticket E0-T1: RiskAnalysis support in verify_agent()**

File: `src/hifi/verification/verifier.py`

- Add `RiskAnalysis` branch to `_named_tool_results()`:
  `return [("risk_metrics", analysis.risk_metrics)]`
- Add `RiskAnalysis` to `verify_agent()` type union and agent_type inference
- `FIELD_ALIAS_TABLE` already covers risk_metrics fields (hist_vol, beta, Sharpe,
  max_drawdown, VaR) from Phase 5. Verify coverage on RiskAnalysis rationale text.

Tests: verify RiskAnalysis returns AgentVerificationReport, check field resolution
for known risk_metrics keys.

**Ticket E0-T2: MacroAnalysis support in verify_agent()**

File: `src/hifi/verification/verifier.py`

- Add `MacroAnalysis` branch to `_named_tool_results()`:
  `return [("macro_snapshot", analysis.macro_snapshot)]`
- Add `MacroAnalysis` to type union
- `FIELD_ALIAS_TABLE` already covers macro_snapshot (fed_funds_rate, CPI, VIX, etc.)
  Verify coverage on MacroAnalysis rationale field.

Tests: verify MacroAnalysis returns AgentVerificationReport with resolvable claims.

**Ticket E0-T3: SentimentGroundingReport schema**

File: `src/hifi/verification/schemas.py`

New schemas:
```python
class SentimentGroundingResult(BaseModel):
    signal_text: str
    grounded: bool
    matched_chunk: str | None = None

class SentimentGroundingReport(BaseModel):
    ticker: str
    as_of_date: str
    n_signals: int
    n_grounded: int
    grounding_rate: float
    results: list[SentimentGroundingResult]
```

Tests: schema validation, grounding_rate = n_grounded / n_signals (ZeroDivisionError
guard when n_signals == 0 → grounding_rate = 0.0).

**Ticket E0-T4: verify_sentiment_agent() implementation**

File: `src/hifi/verification/verifier.py`

```python
def verify_sentiment_agent(
    analysis: SentimentAnalysis,
    retrieved_context: str,
) -> SentimentGroundingReport:
```

- Normalise each item in `analysis.notable_signals` (lowercase, strip whitespace)
- Normalise `retrieved_context` identically
- Check if normalised signal appears in normalised context (str.__contains__)
- Return SentimentGroundingReport

Tests:
- All signals grounded → grounding_rate = 1.0
- No signals → grounding_rate = 0.0, n_signals = 0
- Partial grounding → correct fraction
- Empty retrieved_context → all ungrounded

**Ticket E0-T5: EnsembleVerificationReport extension**

File: `src/hifi/verification/schemas.py` and `verifier.py`

- Add `sentiment_report: SentimentGroundingReport | None = None` to
  `EnsembleVerificationReport` (optional, backward compatible)
- `verify_ensemble()` gains optional `sentiment_analysis: SentimentAnalysis | None`
  and `sentiment_context: str | None` parameters.

Tests: backward compatibility (existing verify_ensemble calls unchanged), optional
field populated when sentiment arguments provided.

**Ticket E0-T6: Establish HR/GR baselines for Risk, Macro, Sentiment**

Script: `scripts/run_phase13_verification_baseline.py`

Run `verify_agent()` for Risk and Macro agents on Phase 12 evaluation dates
(AAPL/JPM/XOM, 2023-03-31). Run `verify_sentiment_agent()` on same dates.
Output: `tests/fixtures/baseline/phase13_verification_baseline.json`

Report: HR, GR (Risk), HR, GR (Macro), SGR (Sentiment), unresolvable rates.
This establishes the measurement prior to fine-tuning.

---

## Epic E1: Sentiment Agent Fine-Tuning (DJ-073)

**Objective:** Fine-tune `sentiment_v1` on MD&A tone labels. Deploy if SGR >= 0.720
on three-tier evaluation.

**Gate (must all pass before E1-T2):**
1. E0 complete (SGR baseline established)
2. `validate_sentiment_corpus.py` reports >= 200 labeled examples, >= 30 Sell
3. Phase 12 OQ-M02 shows entropy(A) - entropy(FT) < 10% (diversity preserved by FT)
   AND Sentiment agent contributes to observed diversity spread

**Ticket E1-T1: Corpus validation + label generation**

Script: `scripts/validate_sentiment_corpus.py`

- Load Phase 7 EDGAR corpus (LanceDB)
- For each filing (10-K/10-Q MD&A sections, 2018-2023):
  - Apply keyword-based tone classifier (DJ-073)
  - Assign Buy/Hold/Sell label + confidence
  - For 2022 filings: apply Sell augmentation rule (cautious tone + 60-day return < -10%)
- Report: class distribution, total count, Sell count, example quality sample
- Output: `data/training/sentiment_labels_v1.jsonl` (if gate passes)
- Decision: PROCEED or ABORT with documented reason

**Ticket E1-T2: Sentiment training data assembly**

Script: `scripts/generate_sentiment_examples.py`

Analogous to `scripts/generate_compliance_examples.py` (Phase 12).
- Format as `{"prompt": "<sentiment analysis prompt>", "completion": "<Buy/Hold/Sell JSON>"}`
- Compliance examples (same structure as technical/fundamental) interleaved at ~1% ratio
- Target: >= 200 domain examples + >= 5 compliance examples
- Output: `data/training/sentiment_v1.jsonl`

Tests: `tests/unit/test_sentiment_training.py` — schema validation, class balance,
compliance ratio.

**Ticket E1-T3: Fine-tune sentiment_v1**

Uses existing `venvs/finetune/` infrastructure (mlx_lm, rank 8, 500 iters).
Mirrors Phase 11 technical adapter training.

Makefile target: `finetune-sentiment`

Command structure:
```bash
cd venvs/finetune && python -m mlx_lm.lora \
  --model qwen2.5-coder-32b-instruct-mlx \
  --data data/training/sentiment_v1.jsonl \
  --iters 500 --rank 8 \
  --adapter-path data/adapters/sentiment_v1/
```

**Ticket E1-T4: Three-tier evaluation of sentiment_v1**

- Tier 1 (SGR): `verify_sentiment_agent()` on AAPL/JPM/XOM at 2023-03-31
  Compare base SGR vs. fine-tuned SGR. Deploy threshold: SGR >= 0.720.
- Tier 2 (accuracy): 60-day forward accuracy on 10 Phase 12 dates
- Tier 3 (diversity): pairwise_diversity(sentiment_v1, others) vs. base

Output: `tests/fixtures/baseline/phase13_sentiment_evaluation.json`

**Ticket E1-T5: Deploy sentiment_v1 or document abandonment**

Decision criterion: SGR >= 0.720 AND no regression in diversity (pairwise not worse
than base by > 10%).

If deployed: `mlx_lm.server` port assignment (decide at phase start, likely 1237).
If abandoned: document as empirical result in bitacora. Sentiment uses base model
for Phase 14.

Makefile target: `finetune-serve-sentiment` (conditional on deploy decision).

---

## Epic E2: Multi-Round Debate Calibration (DJ-074)

**Objective:** Extend `run_debate_round()` to support multi-round with vote stability
convergence. Measure OQ-D04 (does round 2 reduce herding?).

**Prerequisite:** Phase 12 debate transcripts (from E4 LLM evaluation run).
If participation rate (OQ-D03) < 20%, downscope to documentation only.

**Ticket E2-T1: Phase 12 transcript analysis**

Script: `scripts/analyze_debate_transcripts.py`

- Load Phase 12 factorial results (phase12_factorial_results.json)
- Compute: participation rate, vote_delta distribution, n_agents_changed_vote distribution
- Answer OQ-D03 quantitatively
- Determine whether multi-round investment is warranted

**Ticket E2-T2: Multi-round debate implementation**

File: `src/hifi/collective/debate.py`

Extend `run_debate_round()` (or add `run_debate_multi_round()`):
```python
def run_debate_multi_round(
    initial_signals: list[AgentSignal],
    llm: BaseChatModel,
    max_rounds: int = 2,
) -> list[DebateTranscript]:
    """Run up to max_rounds debate rounds. Stop on vote stability."""
```

Convergence logic:
- After each round, compare majority_decision to previous round
- If unchanged: set converged=True, stop
- If max_rounds reached without convergence: record as unconverged

Tests: vote stability convergence (mock LLM producing identical revision), max_rounds
enforcement, single-agent degenerate case.

**Ticket E2-T3: run_debate_ensemble() multi-round parameter**

File: `src/hifi/agents/ensemble_runner.py`

Add `max_rounds: int = 1` parameter to `run_debate_ensemble()`. Default=1 preserves
Phase 12 behaviour exactly.

**Ticket E2-T4: Evaluate OQ-D04**

Script: `scripts/run_phase13_debate_eval.py`

Re-run Phase 12 conditions C and D with max_rounds=2 on a subset of dates (5 dates,
3 tickers = 30 additional runs). Compare herding_coefficient 1-round vs 2-round.

Output: `tests/fixtures/baseline/phase13_debate_multiround.json`

---

## Epic E3: LLM-Extracted Knowledge Graph (DJ-075)

**Gate: Execute only if Phase 12 OQ-K02 positive (GraphRAG Precision@k delta >= 5%).**

**Ticket E3-T1 (if gate passes): SEC MD&A competitor extraction**

Script: `scripts/extract_competitors.py`

- Load MD&A "Competition" sections from Phase 7 EDGAR corpus (10-K filings only)
- Few-shot LLM prompt: "List the direct competitors named in this text."
- Extract (company_a, company_b, relationship) triples
- Filter: keep only companies where OHLCV data exists in data/market/
- Output: `data/knowledge_graph/extracted_competitors.json`

Tests: extraction on a fixture MD&A excerpt, entity deduplication (AAPL vs. Apple Inc.)

**Ticket E3-T2 (if gate passes): Merge with hand-coded graph**

File: `src/hifi/knowledge/graph_construction.py`

- `build_financial_graph(include_extracted=True)` parameter
- Merge extracted edges with DEFAULT_COMPETITORS (deduplication, symmetric enforcement)
- Report: nodes added, edges added, conflicts with hand-coded graph

**Ticket E3-T3 (if gate passes): Evaluate expanded graph Precision@k**

Script: `scripts/run_phase13_graphrag_expanded.py`

Compare Precision@k: hand-coded graph (Phase 12) vs. expanded graph (Phase 13).
Answers OQ-K03 (does automatic extraction improve retrieval precision?).

**Ticket E3-T4 (if gate fails): Negative result analysis**

Document: Why didn't GraphRAG improve Precision@k at 12-node scale?
Hypothesis test: measure average number of unique documents retrieved via BFS
expansion vs. single-ticker search. If overlap > 80%, BFS is not adding new
information — the corpus density is too low.
Output: `doc/bitacora/PHASE_13_GRAPHRAG_NEGATIVE.md` (if applicable).

---

## Epic E4: Agent Memory (DJ-076)

**Objective:** Implement in-context decision history for all 5 voting agents.
Measure whether memory changes decision rates on the Phase 12 evaluation dates.

**Ticket E4-T1: AgentMemoryRecord schema**

File: `src/hifi/collective/schemas.py` (or new `src/hifi/memory.py`)

```python
class AgentMemoryRecord(BaseModel):
    ticker: str
    as_of_date: str
    agent_type: str
    decision: str
    confidence: float
    actual_60d_return: float | None = None
    outcome_correct: bool | None = None
```

**Ticket E4-T2: AgentMemoryStore**

File: `src/hifi/collective/memory.py`

```python
class AgentMemoryStore:
    def __init__(self, store_path: Path): ...
    def record(self, record: AgentMemoryRecord) -> None: ...
    def recall(self, ticker: str, agent_type: str, n: int = 3) -> list[AgentMemoryRecord]: ...
    def format_for_prompt(self, records: list[AgentMemoryRecord]) -> str: ...
```

Persistence: JSON file per agent per ticker at `data/memory/{agent_type}/{ticker}.json`.
`recall()` returns last N records by as_of_date, most recent first.
`format_for_prompt()` returns the structured prefix string (DJ-076 format).

Tests: record/recall round-trip, n_records=0 returns empty string, partial history
(< 3 records) returns available records only.

**Ticket E4-T3: Memory injection into agent prompts**

Files: `src/hifi/agents/fundamental_agent.py`, `technical_agent.py`,
       `risk_agent.py`, `macro_agent.py`, `sentiment_agent.py`

Each agent node gains an optional `memory_store: AgentMemoryStore | None = None`
parameter. If provided, memory prefix is prepended to the analytical prompt before
the LLM call.

`run_ensemble()` gains `memory_store: AgentMemoryStore | None = None` parameter.

Tests: prompt includes memory prefix when store provided, prompt unchanged when None.

**Ticket E4-T4: OQ-M03 measurement**

Script: `scripts/run_phase13_memory_eval.py`

- Run `run_ensemble()` with and without memory on the Phase 12 30 evaluation
  dates (AAPL/JPM/XOM, 10 quarterly dates)
- Compare decision rate changes per agent (fraction of dates where decision differs)
- Compute: memory influence rate = fraction of dates where memory changes the decision

Output: `tests/fixtures/baseline/phase13_memory_eval.json`
Expected result: influence rate 5-15% (memory provides context but does not dominate)

---

## Epic E5: Drift Detection (DJ-077)

**Objective:** Three operational drift monitors for production readiness (Phase 14).

**Ticket E5-T1: Drift module skeleton**

File: `src/hifi/collective/drift.py`

```python
class DriftMonitor:
    def check_data_drift(self, recent: pd.DataFrame, baseline: pd.DataFrame) -> DriftResult
    def check_agent_drift(self, recent_decisions: list[str], baseline_dist: dict[str, float]) -> DriftResult
    def check_collective_drift(self, herding_series: list[float], baseline_mean: float, baseline_std: float) -> DriftResult

class DriftResult(BaseModel):
    drift_type: Literal["data", "agent", "collective"]
    statistic: float
    p_value: float | None
    alert: bool
    threshold: float
    description: str
```

**Ticket E5-T2: KS test for data drift**

`check_data_drift()` implementation:
- `scipy.stats.ks_2samp()` for each numeric feature column
- Alert if any feature KS p-value < 0.05
- Baseline: Phase 10 data distributions (2020-2022 OHLCV + macro features)

Tests: identical distributions → p > 0.05 (no alert), obviously different
distributions → p < 0.05 (alert triggered).

**Ticket E5-T3: Chi-squared test for agent drift**

`check_agent_drift()` implementation:
- Expected proportions from Phase 10 bootstrap decision distribution
- `scipy.stats.chisquare()` on recent decision counts
- Alert if chi-squared p < 0.05

**Ticket E5-T4: CUSUM for collective drift**

`check_collective_drift()` implementation:
- CUSUM statistic: C_k = max(0, C_{k-1} + x_k - (μ + k_delta))
  where k_delta = 0.5σ (standard CUSUM sensitivity parameter)
- Alert if C_k > 3σ above baseline
- Baseline statistics from Phase 12 herding_coefficient mean/std

**Ticket E5-T5: Calibration on Phase 10 regime data**

Script: `scripts/calibrate_drift_monitors.py`

- Run all three drift monitors on Phase 10 data split at the 2022-01-01 regime change
- Check: do monitors correctly detect the 2022 rate shock regime change?
  (KS should alert on vol regime change; CUSUM should detect herding increase in 2022)
- Output: `tests/fixtures/baseline/phase13_drift_calibration.json`

Tests for E5: deterministic statistical properties of KS/chi-squared (not phase-specific).

---

## Epic E6: Synthetic Scenarios (DJ-078)

**Objective:** Populate Dataset Family F with three historical stress-test scenarios.

**Ticket E6-T1: ScenarioEvaluator class**

File: `src/hifi/collective/scenarios.py`

```python
class ScenarioDefinition(BaseModel):
    scenario_id: str
    ticker: str
    as_of_date: str
    event_description: str
    expected_direction: Literal["Buy", "Hold", "Sell", "Risk-Off"]
    regime: str

class ScenarioResult(BaseModel):
    scenario_id: str
    ticker: str
    as_of_date: str
    collective_decision: str
    expected_direction: str
    aligned: bool  # collective_decision direction matches expected
    ensemble_output: dict

class ScenarioEvaluator:
    def run(self, scenario: ScenarioDefinition) -> ScenarioResult: ...
```

**Ticket E6-T2: Run three scenarios**

Script: `scripts/run_phase13_scenarios.py`

Run F-001 (2020-03-16), F-002 (2022-03-31), F-003 (2023-02-02) using `run_ensemble()`.
Requires LM Studio. Output: `data/scenarios/{F-001,F-002,F-003}.json`

**Ticket E6-T3: Dataset Family F documentation**

File: `data/scenarios/README.md`

Schema documentation, scenario descriptions, methodological limitations
(historical scenarios, not generated), alignment results.

Tests: ScenarioDefinition schema validation, ScenarioResult serialisation.

---

## Epic E7: Dataset Families Audit (DJ-079)

**Ticket E7-T1: Dataset Family E standardization**

File: `data/interactions/README.md`

Document the EnsembleOutput JSON schema as the canonical Dataset Family E artifact.
Create index: `data/interactions/index.json` listing all available runs by
(phase, ticker, date, condition).

**Ticket E7-T2: Dataset Family G audit**

Verify all baseline fixtures exist and are documented:
`tests/fixtures/baseline/` inventory with phase, date, and content hash.
Add `tests/fixtures/baseline/MANIFEST.md`.

---

## Epic E8: Documentation and Replication

**Ticket E8-T1: Phase 13 bitacora**

File: `doc/bitacora/PHASE_13_ADVANCED_FEATURES.md`

Sections: Objective, Architecture Decisions (DJ-071 through DJ-079), Implementation
Summary, Results (verification baselines, Sentiment FT decision, memory influence,
drift calibration, scenario results), Open Questions, Lessons Learned.

**Ticket E8-T2: Phase 13 replication notebook**

File: `notebooks/phase13_replication.ipynb`

No LLM calls, < 30s runtime. Sections:
1. Verification baseline comparison (Fund/Tech Phase 5 vs. Risk/Macro/Sentiment Phase 13)
2. Sentiment training corpus analysis (class balance, examples)
3. Sentiment_v1 evaluation (SGR, if deployed)
4. Drift calibration charts (KS p-values, herding CUSUM)
5. Scenario results table
6. OQ-V01, OQ-V02, OQ-M03, OQ-DR01 conclusions

**Ticket E8-T3: plans/STATUS.md update**

Update Phase 12 to COMPLETE (after LLM evaluation runs), Phase 13 to IN PROGRESS,
DJ decisions index updated through DJ-079.

**Ticket E8-T4: MEMORY.md update**

Update Phase 13 completion, test count, next DJ number, Phase 14 scope.

---

## Test Coverage Summary

| New test file | Scope |
|---|---|
| `test_verification_risk.py` | E0-T1: RiskAnalysis verify_agent |
| `test_verification_macro.py` | E0-T2: MacroAnalysis verify_agent |
| `test_verification_sentiment.py` | E0-T3/T4: SGR schema + verify_sentiment_agent |
| `test_verification_ensemble_v2.py` | E0-T5: Extended EnsembleVerificationReport |
| `test_sentiment_training.py` | E1-T2: Training data schema, class balance |
| `test_debate_multiround.py` | E2-T2: Convergence logic |
| `test_memory.py` | E4-T1/T2: AgentMemoryRecord, AgentMemoryStore |
| `test_drift.py` | E5-T1/T2/T3/T4: DriftMonitor statistical tests |
| `test_scenarios.py` | E6-T1: ScenarioEvaluator schema |

Target: 1250+ tests, 0 lint errors at phase close.

---

## Makefile Targets (additions)

| Target | Command | Requires |
|---|---|---|
| `verification-baseline-p13` | Run E0-T6 script | Phase 12 LLM evaluation complete |
| `validate-sentiment-corpus` | Run E1-T1 script | LanceDB populated (Phase 7) |
| `generate-sentiment-training` | Run E1-T2 script | E1-T1 gate pass |
| `finetune-sentiment` | mlx_lm.lora for sentiment_v1 | venvs/finetune/ + E1-T2 |
| `eval-sentiment` | Three-tier sentiment evaluation | LM Studio + E1-T3 |
| `analyze-debate-transcripts` | Run E2-T1 script | Phase 12 factorial results |
| `eval-debate-multiround` | Run E2-T4 script | LM Studio |
| `extract-competitors` | Run E3-T1 script (if OQ-K02 positive) | LM Studio |
| `eval-memory` | Run E4-T4 script | LM Studio |
| `calibrate-drift` | Run E5-T5 script | No LLMs |
| `run-scenarios` | Run E6-T2 script | LM Studio |

---

## Phase 14 Handoff (produced by Phase 13)

1. **Full verification coverage** — HR/GR for Risk/Macro, SGR for Sentiment
2. **Sentiment_v1 deploy/abandon decision** — with SGR evidence
3. **Multi-round debate** — calibrated convergence, OQ-D04 answer
4. **LLM-extracted graph** — expanded or documented as non-improving
5. **Agent memory** — operational, memory influence quantified (OQ-M03)
6. **Drift monitors** — operational, calibrated on 2022 regime change
7. **Dataset Family F** — 3 historical scenarios
8. **Dataset Family E** — standardized schema and index
9. **Open questions for Phase 14** — OQ-AG01 (Contrarian design), OQ-AG02 (memory decay),
   OQ-AG03 (calibration), concept drift detection
