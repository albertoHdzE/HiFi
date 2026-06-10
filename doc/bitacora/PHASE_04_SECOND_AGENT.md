# Phase 4 Bitacora: Second Agent — First Ensemble

**Phase:** 4 -- Second Agent (First Ensemble)
**Status:** COMPLETE
**Date:** 2026-06-10
**Author:** Alberto Espinosa
**Tests at completion:** 463 passing, 0 skipped, 0 lint errors

---

## Objective

Add a second agent — the Technical Analyst — and wire both agents into a collective
decision engine using confidence-weighted voting. The scientific purpose is not to produce
better financial predictions. It is to establish whether two architecturally diverse agents,
each restricted to a different information domain, produce genuinely different opinions.

If they do, the ensemble design is structurally sound and Phase 8 (full agent population)
is justified. If they do not, the information restriction strategy is insufficient and must
be revised before scaling.

This phase is the first empirical test of the central hypothesis.

---

## E1: Technical Analyst Agent

**Graph structure.** The Technical Agent uses the same LangGraph pattern as the Fundamental
Agent but without `load_snapshot_node`. It has no access to financial statements. Its three
nodes — `call_mcp_tools`, `generate_analysis`, `parse_output` — are structurally identical
to nodes 2, 3, and 4 of the Fundamental Agent. The information restriction is enforced at
`call_mcp_tools_node`: only `get_technical_indicators` and `get_risk_metrics` are called.
No code path gives the Technical Agent access to P/E ratios, revenue, or macro context.

**time_horizon extraction.** The Technical Agent's prompt asks for a `time_horizon` field
("short-term", "medium-term", "long-term") in the JSON output. This field is extracted from
the parsed dict before constructing the AgentSignal. The reason it is not added to
AgentSignal: AgentSignal is the uniform interface consumed by the collective decision engine.
Adding a Technical-Agent-specific field to the universal schema would couple the aggregation
logic to one agent's capabilities. time_horizon lives in TechnicalAnalysis, which wraps
AgentSignal with agent-specific context. The distinction between the uniform interface and
the full analysis envelope is the same as Phase 3.

**Decision DJ-016: Technical Agent model confirmed as mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled.**

The MoE 35B model (3.5B active parameters, distilled from Claude Opus 4.6) was chosen over
the dense 27B Q4 model. Both are Qwen 3.5 base models with Claude Opus 4.6 reasoning
distillation, but the MoE variant has faster inference due to sparse activation. The
training objective of both is multi-step reasoning rather than code generation — a genuine
difference from the Phase 3 `qwen2.5-coder-32b` model.

**Decision DJ-017: Prompt strategy confirmed as Strategy 1 (indicator-state framing).**

The `technical_v1` prompt defines the interpretation framework explicitly in the system
section: RSI thresholds (< 30 oversold, 30-50 recovering, 50-70 momentum, > 70 overbought),
MACD crossover semantics, Bollinger Band position interpretation, ATR regime classification,
and Sharpe/drawdown benchmarks. The model is not expected to apply its own domain knowledge
to raw numbers. The framework is the controlled variable.

Strategy 1 produces reproducible, auditable interpretations. The risk — mechanical rule
application without price context — is acceptable at Phase 4 because the baseline's purpose
is measurement, not accuracy. Phase 10 can test Strategy 2 as an ablation once forward
outcome data exists to judge which produces better directional calls.

**Critical bug found during live testing: max_tokens truncation.**

The first live run produced 100% fundamental compliance but only 33% technical compliance
(1/3 tickers). JPM and XOM both failed on the first parse attempt and the retry. AAPL
succeeded.

Root cause: reasoning-distilled models consume internal chain-of-thought tokens before
generating visible output. With `max_tokens=1024`, the model had approximately 900 tokens
consumed by internal reasoning, leaving fewer than 130 tokens for the JSON response. The
JSON was truncated mid-rationale-string. AAPL succeeded because its technical rationale
happened to be shorter than 130 tokens. JPM and XOM had longer rationales (more indicator
values to cite) that crossed the token boundary.

The fix is `max_tokens=4096` in the Technical Agent's `make_llm()` calls. This is the
fundamental difference between code-generation models (Phase 3's qwen2.5-coder, which
does not think before responding) and reasoning-distilled models (Phase 4, which do). Any
future agent built on a reasoning model must use a larger max_tokens budget.

After the fix: 100% technical compliance, 0 parse failures on first attempt for AAPL and
JPM; XOM required one retry (first response malformed) but the retry succeeded. The retry
path was exercised exactly as designed.

---

## E2: Ensemble Output Schemas

**TechnicalAnalysis.** The schema mirrors FundamentalAnalysis: wraps AgentSignal with the
raw tool results (technical_indicators dict, risk_metrics dict) plus time_horizon and
prompt_version. The tool_results_flat() method merges both dicts for the hallucination
checker. The information boundary is enforced: TechnicalAnalysis carries no financial
ratio or macro fields.

**EnsembleDecision.** Nine fields capture the full state of a confidence-weighted vote:
collective_decision, collective_confidence, n_valid_signals, agreement, disagreement_entropy,
opinion_dispersion, agent_decisions, agent_confidences, winning_score, total_score. The
agent_decisions and agent_confidences lists preserve the individual signals for auditing.
Phase 5 can inspect these to trace why a particular collective decision was reached.

**EnsembleOutput.** The top-level envelope: ticker, as_of_date, fundamental_analysis,
technical_analysis, ensemble_decision, latency_ms. This is the schema Phase 9 will extend
to N agents by replacing the two named analysis fields with a list. The two-agent version
is explicit about which analysis is which; the N-agent version will use a dict keyed by
agent_id. The transition is the expected structural change.

---

## E3: Collective Decision Engine

**Confidence-weighted voting (David §12.2.2).** Implementation matches the formula exactly:

```
Score(k) = sum of confidence for agents voting k
Decision = argmax Score(k)
```

With 2 agents voting the same decision, collective_confidence = 1.0 (all conviction on the
winner). With 2 agents voting differently (confidence c1 and c2), collective_confidence =
max(c1, c2) / (c1 + c2). The higher-confidence agent's vote wins.

**Tie handling.** A tie occurs when two options share the maximum confidence-weighted score.
With 2 agents on 3 options, a tie requires identical scores for two or more options. The
conservative resolution is "Hold" with collective_confidence = 0.0. This is the right
default: when agents disagree with equal conviction, the system should not manufacture a
direction. The 0.0 confidence is a signal to downstream consumers that the decision is
unreliable.

**Disagreement entropy (David §5.6.1).** Computed over the vote distribution by count
(not by confidence). This is deliberate: entropy measures the spread of opinions, not the
strength of conviction. Two agents voting Buy(0.9) and Sell(0.1) have the same entropy
as two agents voting Buy(0.5) and Sell(0.5) — both are maximally split. The confidence
difference is captured by opinion_dispersion and collective_confidence separately. Using
count-proportion for entropy preserves the David formula's intent.

**Opinion dispersion (David §5.6.2).** Mean absolute deviation of confidence scores.
For 2 agents: |c1 - c2| / 2. This measures how far apart the agents' convictions are,
independent of which option they chose. High dispersion with agreement means one agent is
much more confident than the other. High dispersion with disagreement means the agents
not only disagree on direction but also on how strongly to hold their views.

**Pairwise diversity (David §5.6.5 categorical adaptation).** For Phase 4 with 2 agents,
this is 0 if they agree and 1 if they disagree. The function generalises to N agents
naturally: it counts the fraction of agent pairs that disagree. At Phase 9 with 6 agents,
this will produce meaningful gradations.

---

## E4: Ensemble Runner

**Independence guarantee.** `run_ensemble` calls `run_analysis` (Fundamental) and
`run_technical_analysis` (Technical) as sequential, independent function calls with no
shared state. Neither agent sees the other's output during reasoning. The independence
is enforced by the function call boundary: no mutable object is passed between them,
no shared queue, no side channel. This is the independence condition from David §10.1.

A system where Agent 2 could observe Agent 1's reasoning before producing its own would
not be an ensemble — it would be a chain. Chains have correlated errors by construction
and provide no ensemble benefit. The sequential-but-independent design keeps the benefit
while deferring the parallelisation complexity to Phase 9.

**Sequential vs concurrent.** With 2 agents and 3 tickers, sequential execution adds
roughly one agent's latency per ticker. At Phase 4's mean 75 seconds per ticker ensemble,
sequential adds 25-35 seconds compared to parallel. This is acceptable. Concurrency
(thread safety, exception propagation, shared logging) would add implementation surface
area that is not earned at this scale. Phase 9 will introduce asyncio-based parallelism
with 6+ agents where the latency argument reverses.

---

## E5: Baseline Evaluation Fixture

**Live run results (phase4_ensemble.json, 2026-06-10):**

| Ticker | Fund. decision | Fund. conf. | Tech. decision | Tech. conf. | Horizon | Collective |
|---|---|---|---|---|---|---|
| AAPL | Hold | 0.75 | Hold | 0.72 | long-term | Hold (1.0) |
| JPM | Hold | 0.65 | Hold | 0.52 | short-term | Hold (1.0) |
| XOM | Hold | 0.65 | Hold | 0.50 | medium-term | Hold (1.0) |

**Ensemble metrics:**

| Metric | Value | Interpretation |
|---|---|---|
| fundamental_compliance_rate | 1.00 | All 3 tickers produced valid signals |
| technical_compliance_rate | 1.00 | All 3 tickers produced valid signals (after max_tokens fix) |
| ensemble_agreement_rate | 1.00 | Both agents agreed on every ticker |
| pairwise_diversity | 0.00 | Zero disagreement across 3 tickers |
| mean_disagreement_entropy | 0.00 | Unanimous in every case |
| mean_opinion_dispersion | 0.0517 | Confidence gap averages ~0.10 per ticker pair |
| mean_total_latency_ms | 75344 | ~75 seconds per ensemble (including both agents + MCP) |

**What the zero diversity result means.** Both agents voted Hold for all three tickers.
Pairwise diversity = 0.0, disagreement entropy = 0.0. The Phase 4 plan identified this
as a possible outcome (pairwise_diversity < 0.1 → investigate information restriction).
Two interpretations are consistent with the data:

1. **Training prior dominance.** Both models were trained on large financial corpora and
   have strong prior beliefs that Q1 2023 blue-chip equities (AAPL, JPM, XOM) are Hold
   candidates. The training prior may dominate over the specific indicator context
   provided in the prompt. This is a model calibration problem, not an architecture problem.

2. **The indicators genuinely support Hold.** Q1 2023 was a period of elevated rate
   uncertainty (FEDFUNDS at 4.1%, CPI at 3.2%, no yield curve or VIX data available).
   A rational analyst with incomplete macro data and mid-range technical indicators might
   genuinely choose Hold. Both agents may be correct.

The distinction between these interpretations cannot be resolved from this baseline alone.
Phase 10 (evaluation with forward outcome data) will reveal whether the Hold calls for
Q1 2023 were accurate. If AAPL, JPM, and XOM rose significantly after 2023-03-31, the
Hold calls were wrong (agents were too conservative). If they were flat or declined, Hold
was correct and the zero diversity is a sign that both agents converged on the right answer.

**What to do about zero diversity.** The Phase 4 plan specifies: document, do not gate.
Zero diversity is a measurement, not a failure condition at this phase. The architectural
test (can two agents with different information produce different calls?) has not been
falsified — it has not been confirmed either. Three tickers at one point in time is not
a sufficient sample. The diversity question will be answered with more tickers, more dates,
and more volatile market regimes (Phase 8 will include trending markets and crisis periods
where technical and fundamental views typically diverge).

**Opinion dispersion is non-zero.** Even though both agents agreed on direction, the
confidence scores differ meaningfully: AAPL (0.75 vs 0.72, dispersion 0.015), JPM (0.65
vs 0.52, dispersion 0.065), XOM (0.65 vs 0.50, dispersion 0.075). The technical model
is systematically less confident than the fundamental model across all three tickers. This
is interpretable: the technical indicators available (only FEDFUNDS and CPI from macro,
limited by the 2022 fixture date range) provide sparser context than the balance sheet
data available to the fundamental agent. Lower confidence in the face of sparser data is
the correct calibration behaviour.

---

## E6: Holistic Pipeline Test and Phase 3 Regression

The holistic test exercises six scenarios: both agents return values, both agree, they
disagree (confidence-weighted winner selected), EnsembleOutput is JSON-safe, time_horizon
stored in TechnicalAnalysis, and Phase 3 Fundamental Agent still produces valid
FundamentalAnalysis in isolation. All pass.

Phase 3 regression guard passed. Adding the Technical Agent, the collective package, and
the ensemble runner introduced no regressions in the Phase 3 pipeline. The LangGraph
graphs are independent and share no global state.

---

## Surprises and Insights

**max_tokens is model-family-specific, not universal.** The same `make_llm()` default
(1024 tokens) that works correctly for the code-generation model fails silently for the
reasoning-distilled model. The failure mode is insidious: the response appears to start
correctly (valid JSON opening) and then is cut off mid-string. `_extract_json` returns
None because the JSON is incomplete, which the parse-and-retry path treats as a content
error and retries — but the retry faces the same token budget constraint and fails again.
The symptom (parse failure) points to a content problem when the actual cause is an
infrastructure parameter.

The fix is not to increase the global default. The code-generation model genuinely only
needs 1024 tokens for its responses. The right design is to make each agent responsible
for specifying the token budget appropriate for its model family. `technical_agent.py`
now passes `max_tokens=4096` to `make_llm()`. If a future Phase 8 agent uses a reasoning
model with even longer thinking chains, it may need 8192. This is documented.

**Data directory setup for live baseline runs.** The `scripts/run_phase*_baseline.py`
scripts default to `data/` as the data directory. In a development environment where
Phase 1 live data has not been fetched (the Phase 1 data acquisition was run against
real APIs and committed separately), the only available data is in `tests/fixtures/`.
The fixture Parquet files (market) and XML files (macro) cannot be used directly by the
MCP server, which expects Parquet files in `data/market/` and `data/macro/`.

The solution: a one-time preparation step converts the XML macro fixtures to Parquet via
`MacroDataFetcher.fetch_series()` with a patched `fredapi.fred.urlopen`, then copies the
market Parquets. This produces a complete data directory at `/tmp/hifi_live_data/` that
the baseline scripts can use. The conversion is deterministic (same XML → same Parquet
every time) and takes under 5 seconds.

This is a practical finding for future sessions: `--data-dir /tmp/hifi_live_data` (built
from the fixture preparation step) is the correct invocation for baseline scripts in the
development environment.

**XOM parse retry on second run.** XOM's technical analysis required a retry on the
second live run. The first LLM response was not valid JSON. The retry produced a valid
response. This confirms that the parse-and-retry pattern is necessary and functions as
designed. A system that failed permanently on the first parse failure would have reported
33% technical compliance. The retry path recovered it to 100%. The failure rate on first
attempt (~33% at this model/prompt combination) is higher than expected and is a candidate
for prompt refinement in Phase 10.

---

## Decisions Recorded

| ID | Decision | Status |
|---|---|---|
| DJ-016 | Technical Agent model: mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled | CONFIRMED |
| DJ-017 | Technical Agent prompt strategy: Strategy 1 (indicator-state framing) | CONFIRMED |
| DJ-018 | Aggregation method: confidence-weighted voting; tie → Hold, confidence=0.0 | CONFIRMED |

---

## Open Questions Resolved

**OQ-P4-01 (DJ-016 model selection).** Resolved. MoE 35B model confirmed for Technical
Agent. Both candidate models are available in LM Studio; the MoE was chosen for inference
speed. JSON compliance confirmed.

**OQ-P4-02 (genuine diversity?).** Partially resolved. The Phase 4 baseline shows zero
pairwise diversity on 3 tickers at one date. This is insufficient to confirm or deny the
hypothesis. The question passes to Phase 8 as an open empirical question requiring more
data.

**OQ-P4-03 (ensemble outperforms individual?).** Not resolvable in Phase 4. Requires
forward outcome data from Phase 10. The structural precondition (two independent agents
producing valid signals) is confirmed; the performance question is deferred.

**OQ-P4-04 (technical agent confidence calibration?).** Preliminary finding: technical
agent is systematically less confident than fundamental agent (mean 0.58 vs 0.68 across
3 tickers). Consistent with sparser data context. Full calibration analysis requires Phase
10 outcome data.

**OQ-P4-05 (confidence-weighted vs majority vote?).** Not meaningful to test with zero
disagreement. With all unanimous votes, both methods produce identical results. The
comparison becomes informative when disagreement exists. Deferred to Phase 9.

---

## Connections Forward

**Phase 5 (Verification)** can now operate on two agent types. EnsembleDecision carries
agent_decisions and agent_confidences for both agents. When they disagree, the discrepancy
is a trigger condition for Phase 5's review flag. The first cross-agent numerical
contradiction check — does the fundamental rationale cite a number that contradicts the
technical signal? — is now architecturally possible.

**Phase 8 (Full Agent Population)** depends on the zero-diversity finding. Before adding
four more agents, the diversity question must be addressed. Two candidates: (a) test with
more volatile market data where technical and fundamental views typically diverge, (b) add
prompt diversity on top of information diversity (different instruction framing, not just
different data). The Phase 4 baseline is the control condition.

**Phase 9 (Collective Decision Engine)** will extend `confidence_weighted_vote` to N
agents and add performance-weighted voting (David §12.2.3) using Phase 10 outcome data.
The EnsembleDecision schema scaffolded here (agent_decisions, agent_confidences as lists
of length N) already supports N > 2 without schema changes.

**Phase 10 (Evaluation)** has a concrete comparison target: phase4_ensemble.json. The
evaluation table (Fundamental vs Technical vs Ensemble, by metric) was specified in the
Phase 4 plan and is now populated with the baseline row. Every subsequent improvement
(better prompts, RAG, fine-tuning) will be measured against this row.
