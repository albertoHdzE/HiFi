# Phase 13 Bitacora: Verification Completeness, Sentiment Intelligence, System Resilience

**Phase status:** Wave 2 LLM evaluation in progress — 2026-06-15
**Tests at Wave 2 code close:** 1271 passed, 0 skipped, 0 lint errors (src/ + tests/)
**David sections:** SS9.4, SS10.3, SS10.4, SS14.4, SS8.7, SS15

---

## Objective

Phase 13 addresses three strategic gaps identified after Phase 12:

1. **Verification completeness (E0):** `verify_agent()` covered only Fundamental and
   Technical agents. Risk, Macro, and Sentiment agents executed but were not verified.
   This created an unmonitored blind spot across 3/5 voting agents.

2. **Sentiment intelligence (E1):** The Sentiment Agent ran on the base qwen2.5-coder-32b
   model (no domain fine-tuning). OQ-S01 gated further investment on corpus sufficiency.

3. **System resilience (E4, E5, E6):** No mechanism existed for the system to learn
   from its own decisions (memory), detect market regime changes (drift), or evaluate
   behavior under stress conditions (scenarios). These are preconditions for Phase 14
   (paper trading).

Phase 13 also re-evaluates the multi-round debate hypothesis from Phase 12 (E2) using
the completed Phase 12 factorial baselines, and standardizes Dataset Families E and G (E7).

---

## Architecture Decisions (DJ-071 through DJ-087)

### DJ-071: Phase 13 Scope
Phase 13 executes three workstreams in parallel: verification extension, sentiment
upgrade, and resilience infrastructure. These are decoupled — none blocks another.

### DJ-072: Verification Layer Extension Strategy
Extend `verify_agent()` to Risk and Macro agents using their tool result dicts:
- Risk: `[("risk_metrics", analysis.risk_metrics)]`
- Macro: `[("macro_snapshot", analysis.macro_snapshot)]`
The `FIELD_ALIAS_TABLE` from Phase 5 already covers these fields — no new aliases needed
for numerical verification. Sentiment requires a new verification pathway (SGR).

### DJ-073: SGR (Sentiment Grounding Rate)
New metric for Sentiment verification. SGR = n_grounded / n_signals where a signal
is "grounded" when its text can be found verbatim in the retrieved RAG context.
Rationale: Sentiment is unique among agents in that its claims are qualitative
(signal phrases) rather than quantitative (numerical metrics). HR/GR cannot apply.
SGR tests whether the agent is hallucinating signals or extracting from real context.

### DJ-074: Multi-Round Debate (max_rounds=2)
Extends the Phase 12 single-round Oxford debate to allow up to 2 rounds with
vote-stability convergence. Convergence: if the majority decision after round k
equals the majority decision after round k-1, stop early. At max_rounds=2, this
means at most one additional round beyond the Phase 12 baseline.

### DJ-075: Debate Hypothesis Pre-Registration
OQ-D04 hypothesis pre-registered BEFORE running the evaluation: NEGLIGIBLE or NEGATIVE.
Rationale: herding is an architectural property (model diversity), not a function of
deliberation round count. A second round is likely to reinforce the majority view
(anchoring) rather than reduce herding. Pre-registration prevents HARKing.

### DJ-076: Agent Memory Architecture
In-context prefix (last 3 decisions) chosen over external vector store retrieval.
Decision rationale:
- Zero retrieval latency (no network call during inference)
- 3 × ~200 tokens ≈ 600 tokens — within qwen2.5-coder-32b 32K context
- Clean rollback: memory is passed per-call, not persisted in the LLM
- Memory decay calibration deferred to Phase 14 (OQ-AG02)
Storage: JSON append-only files at `{store_path}/{agent_type}/{ticker}.json`.

### DJ-077: OQ-M03 Experimental Design
Memory influence test uses synthetic prior records (alternating Buy/Hold/Sell with
outcome metadata) rather than real historical decisions. Rationale: using real
decisions would require running the full historical eval first, creating a dependency
loop. Synthetic records with known outcomes test the maximum-influence scenario.

### DJ-078: Scenario Methodology — Historical vs Generative
Phase 13 uses historical market data for scenarios (NOT generative synthetic data).
Rationale: generative methods (GARCH, VAE) introduce distribution methodology
artifacts that would confound agent evaluation. Known historical events provide
interpretable ground truth for expected_direction labels.
Limitation (honest, documented): historical scenarios are a SUBSET of the existing
evaluation universe. They do not test agent behavior on truly unseen distributions.
True generative scenarios deferred to Phase 16.

### DJ-079: Drift Monitor Design
Three complementary monitors chosen for statistical coverage:
- KS test: sensitive to distribution shape changes (feature-level)
- Chi-squared: tests decision distribution shifts (output-level)
- CUSUM: detects persistent trend shifts with low latency (time-series level)
No single monitor covers all regime change types. The three together provide
overlapping coverage to reduce false-negative rate.

### DJ-080: Gemma 4 as Sentiment Base (SUPERSEDED by DJ-087)
**Decision:** Switch Sentiment Agent from qwen2.5-coder-32b to google/gemma-4-e4b.
**Superseded by:** DJ-086 diagnosis confirmed E4B has chat-template failure in LM Studio.

### DJ-086: E4B Failure Diagnosis
`diagnose_sentiment_sgr.py` revealed two failure modes:
1. AAPL/JPM: LM Studio chat-template issue — model echoes user prompt verbatim,
   no JSON generated. (json.JSONDecodeError on response parsing)
2. XOM: JSON parsed but notable_signals are paraphrases → SGR=0 with exact-match
3. gemma-4-12b-it-mlx: listed in LM Studio API but fails to load ("Failed to load model")

### DJ-087: Revert Sentiment to qwen2.5-coder + Verbatim Rule 5
**Revert:** Sentiment Agent reverted to qwen2.5-coder-32b-instruct-mlx.
**Fix:** Added Rule 5 to `prompts/sentiment_v1.md`:
  "If you extracted a notable_signal from retrieved context, the signal_text MUST be
   an EXACT verbatim phrase from that context. Copy the exact phrase — do not paraphrase."
**Result:** SGR improved 0.167 → 0.667 (4/6 signals grounded). AAPL remains at 0.000
because its RAG context is an 8-K boilerplate header with no quotable financial signals.
**E1 implication:** SGR gate for FT (>= 0.720) is conditional on having quotable context.
The AAPL failure is a content issue (corpus), not a model issue.

---

## Epic E0: Verification Layer Extension (DJ-072, DJ-073)

**Status:** COMPLETE

Extended `verify_agent()` and `verify_ensemble()` to all 5 voting agents.

### Risk Agent Verification (E0-T1)
- Added `RiskAnalysis` branch to `_named_tool_results()`: `[("risk_metrics", analysis.risk_metrics)]`
- `FIELD_ALIAS_TABLE` covers risk_metrics fields from Phase 5: hist_vol, beta, Sharpe, max_drawdown, VaR
- **Baseline (2023-03-31):** HR=0.000, GR=1.000 ✓
- **Note:** alias_coverage=38.9% — vol aliases (historical_volatility, annualised_vol, etc.)
  are unresolvable against `hist_vol` without normalization. Documented, not fixed.

### Macro Agent Verification (E0-T2)
- Added `MacroAnalysis` branch: `[("macro_snapshot", analysis.macro_snapshot)]`
- **Baseline (2023-03-31):** HR=0.000, GR=0.000, n_claims=0-1
- **Root cause:** FRED data absent/sparse for 2023-03-31 in the fixture corpus.
  When macro_snapshot has no numerical values, the agent produces no verifiable claims.
  This is correct behavior — the agent does not hallucinate figures from missing data.

### Sentiment Grounding Rate (E0-T3, E0-T4)
- New schemas: `SentimentGroundingResult`, `SentimentGroundingReport`
- `verify_sentiment_agent()` checks each notable_signal substring against retrieved_context
- **Baseline (2023-03-31, DJ-087 verbatim Rule 5):** mean_SGR=0.667
  - AAPL: SGR=0.000 (8-K boilerplate header, no quotable signals)
  - JPM: SGR=1.000 (2/2 grounded)
  - XOM: SGR=1.000 (2/2 grounded)

### E0 Lesson
Verification completeness revealed a structural gap: HR=0 (Hallucination Rate) is the
RIGHT result for agents that extract correct numerical values — they are being grounded,
not guessing. The Macro agent's GR=0 is a DATA problem (no FRED values in fixture),
not a model failure. SGR=0 for AAPL is a CORPUS problem (boilerplate context), not
a grounding failure. All three findings are empirically correct.

---

## Epic E1: Sentiment Fine-Tuning Gate (OQ-S01)

**Status:** ABORT — corpus insufficient

### E1-T1: Corpus Gate Analysis
**Gate criterion:** >= 100 Sell-class examples required for class-balanced training.

**Finding:**
- Fixture-only corpus: 3 tickers (AAPL, JPM, XOM) × ~2 filings each = ~6 examples
- Class distribution: all Buy/Hold — 0 Sell examples
- EDGAR filings not ingested in any Phase 1-13 script
- Phase 13 fixture corpus is insufficient by 2+ orders of magnitude

**Decision (OQ-S01: NEGATIVE):** Fine-tuning requires full EDGAR ingestion (>= 10
tickers, multi-year coverage) before a class-balanced corpus exists.

**Phase 14 prerequisite:** `make acquire-edgar-filings` must ingest at minimum
the 10 tickers used in Phase 10 bootstrap, covering 2018-2022.

---

## Epic E2: Multi-Round Debate Calibration (DJ-074)

**Status:** E2-T1/T2/T3 code COMPLETE; E2-T4 evaluation IN PROGRESS

### Implementation (E2-T1 through E2-T3)
- `run_debate_multi_round()` in `src/hifi/collective/debate.py`
- Vote-stability convergence: stops when revised majority equals previous round majority
- `max_rounds` parameter added to `run_debate_ensemble()` (default=1, Phase 12 compat)
- `max_rounds` exposed in `EnsembleOutput` API

### Evaluation (E2-T4 — script: scripts/run_phase13_debate_eval.py)
**Design:** 5 dates × 3 tickers = 15 runs, max_rounds=2.
**Baseline:** Phase 12 condition C mean_herding = 0.950 (base models, 1-round debate).
**OQ-D04 results:** See `tests/fixtures/baseline/phase13_debate_multiround.json`.

| Metric | Value |
|---|---|
| 1-round herding (Phase 12 condition C) | 0.950 |
| 2-round herding (this eval) | 0.929 (14/15 runs, 1 fail: LM Studio model unload) |
| Delta (2-round minus 1-round) | -0.021 |
| OQ-D04 answer | **NEGLIGIBLE** — \|Δ\|=0.021 < threshold 0.050 |

**Scientific interpretation:** On 12/14 valid runs, both agents agreed unanimously
(herding=1.000) and debate was skipped — round count is irrelevant when agents agree.
On 2 runs (AAPL 2020-12-31 and XOM 2020-12-31), agents disagreed (herding=0.500,
one Buy one Hold). The 2-round debate did not change the split (vote_delta=unchanged).
This confirms DJ-075 pre-registration: herding is determined by the initial diversity of
the base models, not by deliberation round count. Additional rounds reinforce the status
quo (or are skipped entirely).

---

## Epic E4: Agent Memory (DJ-076)

**Status:** E4-T1/T2/T3 code COMPLETE; E4-T4 evaluation PENDING

### Implementation (E4-T1 through E4-T3)
- `AgentMemoryRecord` and `AgentMemoryStore` in `src/hifi/collective/memory.py`
- Storage: JSON append-only per `{agent_type}/{ticker}.json`
- `format_for_prompt()` builds structured prefix: `[Agent Memory — last N decisions]`
- Memory injected into all 5 voting agents via `memory_prefix` parameter
- `memory_prefixes` dict passed through `run_ensemble()` and `run_debate_ensemble()`

### Evaluation (E4-T4 — script: scripts/run_phase13_memory_eval.py)
**Design:** 10 dates × 3 tickers = 30 pairs, agents=[fundamental, technical].
Each pair run twice: (a) no memory, (b) with 3 synthetic prior records per agent.
**Synthetic priors:** Buy (2019-09-30) → Hold (2019-12-31) → Sell (2020-01-31)
with outcome metadata, creating maximum conflicting signal to detect anchoring.
**Metric:** fraction of pairs where ≥1 agent changed decision.

| Metric | Value |
|---|---|
| Pairs evaluated | 30/30 (0 failures) |
| Pairs where memory changed ≥1 decision | 9/30 |
| Changed fraction | 0.300 |
| OQ-M03 answer | **YES** — memory prefix has measurable influence on agent decisions |

**Scientific interpretation:** Memory prefix changed decisions in 30% of (date, ticker) pairs,
exceeding the 10% threshold. The Fundamental Agent was most susceptible (influenced by the
alternating Buy/Hold/Sell prior records). The Technical Agent was also influenced on several
dates (e.g., JPM 2020-06-30: both agents changed; JPM 2020-09-30: both changed).
Interpretation: the structured `[Agent Memory — last N decisions]` prefix creates
an anchoring effect that shifts Fundamental Agent toward recent prior decisions.
This confirms the memory mechanism is active and operational, but also suggests
Phase 14 should monitor for over-anchoring (OQ-AG02 deferred to Phase 14).

---

## Epic E5: Drift Detection (DJ-079)

**Status:** COMPLETE (OQ-DR01 ANSWERED)

### Implementation
- `DriftMonitor` base + three concrete monitors in `src/hifi/collective/drift.py`
- KS test: two-sample KS on realized_vol + RSI between reference and current windows
- Chi-squared: discrete vote distribution shift test
- CUSUM: cumulative sum on fraction of tickers below 50-day moving average

### Calibration Results (E5-T5 — fixture: phase13_drift_calibration.json)
Calibrated against the 2022 rate-shock regime change (FFR 0% → 4.25%, CPI 8.5%):

| Monitor | Statistic | Alert |
|---|---|---|
| KS test (vol + RSI) | p=0.000 | YES |
| Chi-squared (momentum decisions) | p=0.000 | YES |
| CUSUM (frac < 50d MA) | C_k=48.57 >> threshold=0.534 | YES |

**OQ-DR01: YES** — All three monitors detect the 2022 rate-shock regime change.
CUSUM C_k is 91× the threshold, confirming the 2022 shift is a severe outlier.

### Operational Notes
- KS window: 2020-2021 (reference) vs 2022-2023 (current) — 2 full years per window
- Chi-squared uses momentum_proxy labels from the evaluation dataset
- CUSUM h parameter (threshold) calibrated at 0.534 from 2021 in-control baseline
- All three monitors are production-ready; activation per-trigger is configured in Phase 14

---

## Epic E6: Synthetic Scenario Framework (DJ-078)

**Status:** Code COMPLETE; E6-T2 evaluation PENDING

### Implementation
- `ScenarioDefinition`, `ScenarioResult`, `ScenarioEvaluator` in `src/hifi/collective/scenarios.py`
- 7 Phase 13 scenarios across 3 regimes (crash, rate_shock, earnings_beat)
- `expected_direction`: exact label (Buy/Hold/Sell) or "Risk-Off" (Hold or Sell satisfies)
- Results written to `data/scenarios/{scenario_id}.json`

### Scenarios Defined (PHASE13_SCENARIOS)

| Scenario | Ticker | Date | Regime | Expected |
|---|---|---|---|---|
| F-001  | AAPL | 2020-03-16 | crash | Risk-Off |
| F-001b | JPM  | 2020-03-16 | crash | Risk-Off |
| F-001c | XOM  | 2020-03-16 | crash | Sell |
| F-002  | AAPL | 2022-03-31 | rate_shock | Risk-Off |
| F-002b | JPM  | 2022-03-31 | rate_shock | Hold |
| F-002c | XOM  | 2022-03-31 | rate_shock | Buy |
| F-003  | AAPL | 2023-02-02 | earnings_beat | Buy |

### Evaluation (E6-T2 — script: scripts/run_phase13_scenarios.py)
Results: See `data/scenarios/scenario_summary.json`.

| Scenario | Ticker | Expected | Decision | Aligned |
|---|---|---|---|---|
| F-001  | AAPL | Risk-Off | Hold | YES |
| F-001b | JPM  | Risk-Off | Hold | YES |
| F-001c | XOM  | Sell     | Hold | NO |
| F-002  | AAPL | Risk-Off | Hold | YES |
| F-002b | JPM  | Hold     | Hold | YES |
| F-002c | XOM  | Buy      | Hold | NO |
| F-003  | AAPL | Buy      | Hold | NO |

**Overall alignment: 4/7 = 57%** (crash: 2/3=67%, rate_shock: 2/3=67%, earnings_beat: 0/1=0%)

**Scientific interpretation:**
The ensemble has a structural Hold bias — 6/7 scenarios resulted in Hold. This is
consistent with the Phase 13 verification baseline (agents default to Hold when uncertain).
Key findings:
- **F-001c (XOM Sell):** Hold instead of Sell. Ensemble correctly goes defensive (Risk-Off
  satisfied) but does not reach full Sell conviction even in COVID + oil war combined shock.
- **F-002c (XOM Buy):** Hold instead of Buy. Energy macro tailwind from Russia/Ukraine not
  captured — the Macro Agent had sparse FRED data for 2022-03-31.
- **F-003 (AAPL earnings beat → Buy):** Hold. The earnings beat on 2023-02-02 is not
  captured because (a) the Sentiment Agent's context is from the fixture corpus, not the
  actual earnings call, and (b) the Fundamental snapshot is 2022-12-31 (not Q1 FY2023).

**Methodological limitation (DJ-078):** snapshot_json uses 2022-12-31 reference
financials for all scenario dates. The Fundamental Agent's price-based signals
(via market data at the scenario date) are correct, but the snapshot fundamentals
may not match the scenario date. The low earnings_beat alignment (0/1) is a direct
consequence of this limitation.

---

## Epic E7: Dataset Families Audit and Standardization (DJ-071)

**Status:** COMPLETE

### Dataset Family E: Agent Interactions (E7-T1)
- Schema documented in `data/interactions/README.md`
- 120 interaction records from Phase 12 factorial (conditions A-D, 3 tickers, 10 dates)
- Format: `{ticker}_{as_of_date}_{condition}.json` per record

### Dataset Family G: Verification Baselines (E7-T2)
- Inventory: `tests/fixtures/baseline/MANIFEST.md`
- 12 baseline fixtures documented with schema, generator script, and OQ linkage
- phase13_verification_baseline.json: 3 agents × 3 tickers × 1 date

---

## Open Question Resolution Summary

| OQ | Question | Answer | Evidence |
|---|---|---|---|
| OQ-S01 | Sentiment corpus sufficient for FT? | **NEGATIVE** | E1-T1: 0 Sell examples; FT deferred to Phase 14 |
| OQ-D04 | 2nd debate round reduces herding? | **NEGLIGIBLE** (Δ=-0.021) | E2-T4: debate_multiround.json; 14/15 runs |
| OQ-M03 | Memory prefix changes decisions? | **YES** (30% pairs changed) | E4-T4: memory_eval.json; 9/30 pairs, 0 failures |
| OQ-DR01 | All 3 monitors detect 2022 regime? | **YES** | E5-T5: drift_calibration.json; all p=0.000 |

---

## Phase 13 Outputs

### Code artifacts
- `src/hifi/verification/verifier.py` — Risk + Macro + Sentiment branches added
- `src/hifi/verification/schemas.py` — SentimentGroundingResult, SentimentGroundingReport
- `src/hifi/collective/memory.py` — AgentMemoryRecord, AgentMemoryStore
- `src/hifi/collective/drift.py` — DriftMonitor, KSDriftMonitor, ChiSqDriftMonitor, CUSUMDriftMonitor
- `src/hifi/collective/scenarios.py` — ScenarioDefinition, ScenarioResult, ScenarioEvaluator, PHASE13_SCENARIOS
- `src/hifi/collective/debate.py` — run_debate_multi_round() (max_rounds support)
- `src/hifi/agents/ensemble_runner.py` — max_rounds + memory_prefixes in both runners

### Evaluation artifacts
- `tests/fixtures/baseline/phase13_verification_baseline.json`
- `tests/fixtures/baseline/phase13_drift_calibration.json`
- `tests/fixtures/baseline/phase13_sentiment_corpus.json`
- `tests/fixtures/baseline/phase13_debate_multiround.json` (PENDING)
- `tests/fixtures/baseline/phase13_memory_eval.json` (PENDING)
- `data/scenarios/{scenario_id}.json` (PENDING)

### Documentation
- `data/interactions/README.md` — Dataset Family E schema
- `tests/fixtures/baseline/MANIFEST.md` — Dataset Family G inventory
- `prompts/sentiment_v1.md` — verbatim Rule 5 added (DJ-087)
- `scripts/diagnose_sentiment_sgr.py` — DJ-086 diagnosis tool

### Calibration scripts
- `scripts/calibrate_drift_monitors.py`
- `scripts/run_phase13_e0_baseline.py`
- `scripts/run_phase13_debate_eval.py`
- `scripts/run_phase13_memory_eval.py`
- `scripts/run_phase13_scenarios.py`

---

## Complexity Science Notes

Phase 13's contribution to the complexity science framing:

**Memory as emergence inhibitor:** If memory anchors agents to prior positions,
it could reduce the emergent disagreement that drives collective intelligence.
OQ-M03 tests whether memory causes over-convergence (reduces entropy) or enriches
deliberation (increases calibration). The direction of the effect is an open question
in multi-agent epistemology.

**Drift detection as regime awareness:** The three monitors implement a formal
early-warning system for distribution shift. From a complexity science lens, this
is equivalent to detecting phase transitions in the financial system. The CUSUM
C_k=48.57 for 2022 confirms that the rate-shock period was a genuine phase transition
(not gradual drift), consistent with critical transition theory.

**Historical scenarios as counterfactual tests:** Scenario F-001 (COVID crash)
tests whether the agent collective would have issued defensive signals on 2020-03-16.
If the system is truly intelligence-augmenting (David §1.1), it should produce
Risk-Off signals even without hindsight. Alignment rate measures this property.
