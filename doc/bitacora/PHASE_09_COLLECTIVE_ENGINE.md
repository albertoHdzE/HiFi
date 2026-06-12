# Phase 9 Bitacora: Collective Decision Engine

**Date completed:** 2026-06-12
**Tests at close:** 843 passed, 2 skipped, 0 lint errors
**Status:** COMPLETE (scripts and bitacora committed separately from source)

---

## Objective

Formalize the Contrarian Agent's role in the decision mechanism and expand the
ensemble from one aggregation method (confidence-weighted) to four, enabling
structural comparison across methods on every run. Seed a historical performance
store from 20 bootstrap quarter-ends to enable performance-weighted aggregation
from Phase 9 onward.

Phase 8 produced a 6-agent ensemble with the Contrarian as a non-voting
second-pass critic. Phase 9 makes the Contrarian's output quantitatively
actionable via a discount formula and adds three alternative aggregation methods
as permanent diagnostic instruments alongside the primary confidence-weighted
vote.

---

## Architecture Decisions (DJ-039 through DJ-043)

### DJ-039: Run All Four Methods on Every Ensemble Call

All four aggregation methods are computed on every `run_ensemble()` invocation
and stored in `EnsembleOutput.method_comparison`. The primary method remains
`confidence_weighted` (David §12.2.2) — `ensemble_decision` always reflects
this method. The three alternative methods are structural diagnostics, not
competing primaries.

Rationale for running all four simultaneously:
- **Method divergence as a free signal.** When `majority_vote` disagrees with
  `confidence_weighted`, the discrepancy reveals whether confidence weighting
  is doing real work or simply amplifying the plurality. This is diagnostic
  without requiring ground-truth labels.
- **Zero marginal cost.** All four methods operate on the same `signals` list
  already materialised in memory. The added computation is negligible compared
  to LLM agent latency.
- **Foundation for Phase 10 accuracy tracking.** Phase 10 will record outcomes
  per method, enabling an empirical comparison of method accuracy once 60-day
  forward returns are available. That comparison requires historical divergence
  to be non-zero, and structural divergence must be observed before labels
  arrive.

The four canonical method keys in `method_comparison` are:

| Key | Algorithm | Reference |
|---|---|---|
| `confidence_weighted` | Weighted vote by agent confidence | David §12.2.2 |
| `majority_vote` | Unweighted plurality vote | David §12.2.1 |
| `performance_weighted` | Weighted by historical accuracy from bootstrap | David §12.2.3 |
| `contrarian_adjusted` | Confidence-weighted with contrarian discount | David §12.3 |

### DJ-040: Contrarian Confidence Discount Formula

The `contrarian_adjusted` method applies a multiplicative discount to the
ensemble's collective confidence based on the Contrarian Agent's conviction:

```
contrarian_confidence_discount = 1 - alpha * contrarian_confidence
```

where `alpha = 0.5` (provisional, to be calibrated in Phase 10).

The discount is stored as a separate field (`contrarian_confidence_discount`)
on `EnsembleDecision` rather than being silently absorbed into `collective_confidence`.
This preserves auditability: Phase 10 can reconstruct the undiscounted confidence
as `cc_undiscounted = cc / discount` (when discount > 0).

The `review_flagged` field is set to `True` when contrarian confidence exceeds
`theta = 0.70` (provisional). This threshold was chosen to flag the top ~30%
of contrarian certainty without triggering on every run. Calibration from Phase 8
live runs will refine both `alpha` and `theta` in Phase 10.

When no Contrarian Agent output is available (e.g., `agents=["fundamental","technical"]`),
`contrarian_confidence_discount` defaults to `1.0` (no discount) and `review_flagged`
defaults to `False`, preserving full backward compatibility.

### DJ-041: Bootstrap Proxy Signals Are Documented Heuristics

The bootstrap script (`run_phase9_bootstrap.py`) seeds `agent_performance_history.json`
with 60 analysis runs (20 quarter-ends × 3 tickers) using deterministic threshold
rules instead of live LLM agents:

| Agent | Bootstrap rule |
|---|---|
| Technical | RSI < 40 → Buy; RSI > 60 → Sell; else Hold |
| Risk | Sharpe_252d > 0.8 → Buy; Sharpe_252d < 0.3 → Sell; else Hold |
| Fundamental | Hold (uniform prior; LLM not run) |
| Macro | Hold (uniform prior; LLM not run) |
| Sentiment | Skipped (fail-open; no historical RAG corpus) |
| Contrarian | Skipped (second-pass; requires other agent outputs) |

Fixed confidence values: 0.65 for directional signals (Buy/Sell), 0.50 for Hold.

These rules produce a rough prior. Their limitations are intentional and documented:
1. The Fundamental and Macro agents always Hold, so their accuracy reflects the
   frequency of flat market periods (expected ~40-50%), not analytical quality.
2. The RSI and Sharpe thresholds are not calibrated for these specific tickers.
   They will underperform well-tuned models but provide a non-trivial starting point.
3. Phase 10 will add true LLM agent outputs as labeled records, progressively
   replacing the bootstrap prior with empirical accuracy.

### DJ-042: 60-Day Forward Return Labeling

Each DecisionRecord is labeled using the 60-trading-day forward close price:

```
forward_return = (price_{t+60} - price_{t0}) / price_{t0}
```

where `t0` is the first trading day on or after the analysis quarter-end.

Labeling rules:
- **BUY correct** if forward_return > +0.02 (2% gain over ~3 calendar months)
- **SELL correct** if forward_return < -0.02
- **HOLD correct** within ±0.02 (flat market)
- **Unlabeled** (outcome_correct=None) when price data is insufficient to compute
  the forward return (e.g., Q4 2022 forward window extends past the Parquet horizon)

The ±2% band was chosen to avoid labeling near-zero return periods as simultaneously
correct for both Hold and wrong for Buy/Sell. It acknowledges that a neutral signal
earns credit only when the market is genuinely flat, not merely when the directional
signal happened to miss a small move.

Horizon of 60 trading days matches the primary evaluation window from D-04. A
secondary 20-day horizon (D-04 provision) is deferred to Phase 10, when the expanded
ticker universe will provide enough short-horizon data to differentiate agents.

### DJ-043: Atomic Write Pattern for Performance Store

`save_history()` in `performance_store.py` writes the performance JSON atomically
via `write-to-tmp + rename`:

```python
tmp = path.with_suffix(".json.tmp")
tmp.write_text(history.model_dump_json(indent=2))
tmp.rename(path)  # atomic on POSIX filesystems (same device)
```

This prevents corrupted JSON if the bootstrap is interrupted mid-write. On macOS
and Linux, `rename()` on the same filesystem is atomic at the OS level — the
reader either sees the old complete file or the new complete file, never a partial
write. Phase 10 will add file-level locking when multiple concurrent writers
(live ensemble runs) may update the store simultaneously.

---

## Structural Comparison Design

The method comparison is a structural instrument — it measures method divergence
in the current run, not accuracy against future labels. The key observable is the
**divergence rate**: how often do two methods produce different `collective_decision`
values for the same ticker and date?

Expected divergence patterns:
- `confidence_weighted` vs `majority_vote`: diverges when high-confidence outlier
  agents dominate the weighted vote but lose the plurality. This exposes cases where
  one very confident agent outweighs a majority of moderate agents.
- `performance_weighted` vs `confidence_weighted`: diverges after the bootstrap
  weights differentiate agent accuracy. Before enough labeled data accumulates,
  these will be nearly identical (uniform weights).
- `contrarian_adjusted` vs `confidence_weighted`: always diverges when contrarian
  confidence > 0, since the discount reduces collective_confidence. Decision
  divergence (different `collective_decision`) occurs when the discount pushes
  confidence below the decision boundary.

Phase 10 will compute empirical divergence rates and correlate them with label
accuracy to answer: does higher method divergence predict lower future accuracy?

---

## Rolling Metrics

Phase 9 implements two rolling temporal metrics (David §5.6.3-4):

**Herding coefficient (κ):**
```
kappa_W = mean(a_t)  for t in last W analysis periods
a_t = fraction of agents voting with the plurality at period t
```
kappa near 1/3 (random three-option assignment) indicates independent agents.
kappa near 1.0 indicates systematic herding.

**Consensus stability (S):**
```
S_W = (1 / (W-1)) * sum(1[v_t == v_{t+1}])  for t in last W periods
```
S = 1.0 means the collective decision was unchanged throughout the window.
S = 0.0 means it changed at every period.

Both metrics are computed for W ∈ {5, 10, 20} periods. They return `None` when
history length < W, making them naturally undefined before sufficient data
accumulates. The bootstrap provides 20 periods per ticker, enabling all three
windows for the first time at Phase 9 close.

---

## Baseline Structural Results

Method comparison table (from live baseline run — fill in from `run_phase9_baseline.py` output):

| Ticker | confidence_weighted | majority_vote | performance_weighted | contrarian_adjusted |
|---|---|---|---|---|
| AAPL | TBD | TBD | TBD | TBD |
| JPM | TBD | TBD | TBD | TBD |
| XOM | TBD | TBD | TBD | TBD |

Rolling metrics from bootstrap (20 quarter-ends, 2018-Q1 through 2022-Q4):

| Ticker | kappa_W5 | kappa_W10 | kappa_W20 | S_W5 | S_W10 | S_W20 |
|---|---|---|---|---|---|---|
| AAPL | TBD | TBD | TBD | TBD | TBD | TBD |
| JPM | TBD | TBD | TBD | TBD | TBD | TBD |
| XOM | TBD | TBD | TBD | TBD | TBD | TBD |

*Run `scripts/run_phase9_bootstrap.py` followed by `scripts/run_phase9_baseline.py` to populate these tables.*

---

## Implementation Surprises and Lessons Learned

### Pydantic v2 model_copy for Immutable EnsembleDecision Updates

`contrarian_adjusted_vote()` needs to derive a new `EnsembleDecision` from an
existing one with only the discount and `review_flagged` fields overridden.
Pydantic v2 `model_copy(update={...})` handles this cleanly:

```python
return base.model_copy(update={
    "contrarian_confidence_discount": discount,
    "collective_confidence": base.collective_confidence * discount,
    "review_flagged": review_flagged,
})
```

The immutable copy pattern avoids the footgun of mutating a shared object that
`method_comparison["confidence_weighted"]` also holds a reference to.

### ruff N803 and Lowercase Window Parameter

`herding_coefficient` and `consensus_stability` use lowercase `w` (not `W`) for
the window parameter — ruff N803 prohibits uppercase variable names in function
arguments. All call sites must use `w=5`, not `W=5`. This is documented here
because the mathematical notation in David §5.6.3-4 uses uppercase W, creating
a systematic mismatch between the spec and code that can cause argument errors.

### compute_weights Returns Uniform When No Labeled Records

`compute_weights([])` returns `{agent_type: 0.25}` for the four canonical types.
This means `performance_weighted_vote` is identical to `confidence_weighted_vote`
with equal weights on first run (before bootstrap), because both distribute votes
proportionally to confidence. The divergence only appears after bootstrap labels
differentiate agent accuracy.

---

## File Inventory

### New Source Files

| File | Purpose |
|---|---|
| `src/hifi/collective/performance_store.py` | Load/save/compute_weights/get_weights/update_and_save |

### New Test Files

| File | Purpose |
|---|---|
| `tests/unit/test_aggregation_methods.py` | 28 tests: all 4 voting methods + run_all_methods |
| `tests/unit/test_rolling_metrics.py` | 27 tests: herding_coefficient, consensus_stability, compute_rolling_metrics |
| `tests/unit/test_performance_store.py` | 16 tests: load/save/atomic write/compute_weights/get_weights |
| `tests/holistic/test_phase9_collective_engine.py` | 12 holistic tests: full pipeline with method_comparison |
| `tests/unit/test_phase9_baseline.py` | Fixture validation test (skipif fixture absent) |
| `scripts/run_phase9_bootstrap.py` | Deterministic bootstrap (no LLM required) |
| `scripts/run_phase9_baseline.py` | Live baseline measurement (requires LM Studio) |

### Modified Files

| File | Change |
|---|---|
| `src/hifi/collective/schemas.py` | EnsembleDecision: contrarian_confidence_discount + review_flagged; EnsembleOutput: signals, aggregation_method, method_comparison; DecisionRecord + AgentPerformanceHistory added |
| `src/hifi/collective/voting.py` | majority_vote, performance_weighted_vote, contrarian_adjusted_vote, run_all_methods added |
| `src/hifi/collective/metrics.py` | herding_coefficient, consensus_stability, compute_rolling_metrics added |
| `src/hifi/agents/ensemble_runner.py` | Loads perf weights, runs all 4 methods, logs divergence, populates signals and method_comparison |

---

## Scientific Context

Phase 9 establishes the measurement infrastructure for comparing aggregation
methods. The central question from David §12.2-12.3 is: does adding information
(confidence, historical accuracy, adversarial stress) to the aggregation process
improve collective decisions?

Phase 9 cannot answer this empirically — it lacks outcome labels for 2023-03-31
(those won't be available for 60 trading days, and the bootstrap only covers
2018-2022). What Phase 9 establishes is:

1. **The four methods are structurally distinct.** `majority_vote` treats all
   agents as equal. `confidence_weighted` amplifies high-conviction agents.
   `performance_weighted` amplifies historically accurate agents. `contrarian_adjusted`
   flags ensemble overconfidence. Each method encodes a different hypothesis about
   what makes a good collective decision.

2. **Divergence is observable before accuracy is.** Method disagreement is
   immediate. Accuracy requires labels. Measuring divergence now creates a
   historical record that will be analyzed with labels in Phase 10.

3. **The bootstrap prior is intentionally weak.** Starting with RSI/Sharpe
   heuristics instead of LLM outputs means the initial `performance_weighted`
   results will be nearly identical to `confidence_weighted`. This is correct:
   we should not trust bootstrapped priors too heavily. The weights will shift
   as real agent labels accumulate, and the shift itself will be scientifically
   interesting — it will reveal whether agent types differ in accuracy.

---

## Open Questions for Phase 10

1. **Method accuracy ordering.** After 60-day labels are available for 2023-03-31
   (approximately 2023-06-15), does any method outperform the others on Buy/Sell
   precision? Does `performance_weighted` converge faster than `confidence_weighted`
   on a larger ticker universe?

2. **Contrarian calibration.** Does high `contrarian_confidence` (review_flagged)
   predict larger forecast errors? If so, `alpha` should be increased above 0.5.
   If not, the Contrarian is generating noise and `alpha` should decrease toward 0.

3. **Bootstrap quality.** How much do the RSI/Sharpe heuristic weights diverge
   from true LLM agent accuracy? If the RSI Technical signal happens to have
   good accuracy over 2018-2022, the `performance_weighted` method will give it
   more weight — potentially for the wrong reason (the bootstrap heuristic
   happened to be well-calibrated to the LLM agent's actual behavior by accident).

4. **Herding dynamics.** Does kappa increase during high-VIX periods (2020 COVID,
   2022 rate hikes) when all agents should rationally converge? If so, the
   herding coefficient is a macro-stress indicator, not just an ensemble health metric.

---

## Next Phase

Phase 10 will add accuracy outcome labels to the 2023-03-31 baseline run (after
the 60-day window closes), extend the bootstrap to 20+ tickers, and compute the
first empirical method accuracy comparison. It will also add a QuantStats tear-sheet
output path and the first portfolio-level aggregation across tickers.
