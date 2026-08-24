# Phase 15: Historical Walk-Forward Simulation
## Context and Pre-Phase Decisions

**Gathered:** 2026-06-16
**Status:** NOT STARTED — awaits Phase 14 completion
**Depends on:** Phase 14 (5-org ensemble, 100-stock pipeline, namespace infrastructure)

---

## Why This Phase Exists

Phase 14 builds infrastructure. Phase 15 uses it to answer the central scientific
question of the entire project. Without this phase there is no thesis — just a system
that was built and never rigorously evaluated.

The question: **Does architectural diversity in LLM ensembles produce measurable
predictive skill in financial markets, stable across regimes?**

This tests Page's diversity theorem empirically. Page (2007) argues that a diverse
group of problem-solvers systematically outperforms a homogeneous group of higher-average
performers. HiFi's Phase 12.1 confirmed the homogeneous failure mode (entropy=0.000,
herding=1.000). Phase 14 builds the diverse system. Phase 15 measures the difference.

---

## DJ-095: Walk-Forward Methodology Design

**Problem:** How to structure the historical simulation so results are scientifically
defensible — no look-ahead, no parameter fitting on the test set, reproducible.

**Decision:** Strict walk-forward with temporal namespace partitioning.

```
Period          Dates           Role
Training        2004-2019       Calibration, EDGAR corpus, bootstrap labels
Validation      2020-2021       COVID regime; no parameter re-fitting after this
Held-out test   2022-2023       Primary scientific result; rate-shock regime
Walk-forward    2024-2025       Sequential monthly; causal order enforced
```

**Temporal discipline:** The `hifi-eval` LanceDB namespace (DJ-093) is populated only
with data available at each evaluation date. When evaluating 2022-Q1, the namespace
contains only SEC filings with period_of_report ≤ 2022-Q1, episodes with decision_date
< 2022-Q1, and market data through 2022-Q1. No future information can leak.

**Reproducibility:** `make eval-reset` + `make eval-ingest-through DATE=D` produces
exactly the same namespace state from scratch. Any researcher can reproduce the result.

**Why 2022-2023 as held-out test:** This regime (Fed Funds Rate rising 500bps, CPI at
8.5%, SPY down 20%) is the same regime that OQ-DR01 confirmed all three drift monitors
detect. It is the hardest evaluation period in recent history. If the ensemble shows
positive IC here, the result is robust to adversarial conditions.

---

## DJ-096: Primary vs. Ablation Comparisons

**Decision:** Run the held-out 2022-2023 period under four conditions:

| Condition | Ensemble mode | Key variable |
|---|---|---|
| Full | Sequential, 5-org, episodic RAG | Baseline |
| Parallel | Parallel (no inter-agent sharing), 5-org | Tests OQ-P14-03 |
| Homogeneous | Sequential, qwen-dominant (Phase 13 config) | Tests Page theorem |
| No-memory | Sequential, 5-org, no episodic prefix | Tests OQ-P14-04 |

The Homogeneous condition re-runs with the pre-Phase-14 model assignments (qwen2.5-coder-32b
Fundamental + Technical + Sentiment, qwen3.5 Macro + Contrarian, gemma-3-4b Risk). This
is the direct "remove diversity" ablation.

**Primary metric:** Information Coefficient (IC) = Spearman rank correlation of ensemble
Buy-strength signal with 60-day forward return, computed across all (date, ticker) pairs.

IC > 0.0: ensemble has predictive signal above random.
IC > 0.05: practically significant (industry convention).

**Secondary metrics:** IR = IC / IC_std, Sharpe Ratio, hit rate vs. SPY buy-and-hold,
herding coefficient by regime.

---

## DJ-097: Simulation Pipeline Architecture

**Decision:** `scripts/run_phase15_walkforward.py` — single script, stateless per run,
reads from `hifi-eval` namespace exclusively.

For each (month, ticker) in the walk-forward universe:
1. `make eval-ingest-through DATE={month-end}` (if not already done for this date)
2. `run_sequential_ensemble(ticker, date, namespace="hifi-eval")`
3. Store `EnsembleOutput` to `data/walkforward/{year}/{month}/{ticker}.json`
4. After 60 trading days: `make label-outcomes` populates forward_return + outcome_correct

Parallelism within a date: tickers can be evaluated in parallel (no cross-ticker
dependency). Across dates: strictly sequential (causal order).

**Frequency:** Monthly rebalancing. Each month-end date: all 100 tickers evaluated.
21 years × 12 months × 100 tickers = 25,200 ensemble calls total (across all conditions).
With 4 conditions: ~100,800 calls. At ~5 min/call sequentially: multi-day GPU batch job.
Solution: batch overnight; checkpoint-resume support in the pipeline script.

---

## Open Questions

| ID | Question | Resolution |
|---|---|---|
| OQ-P14-02 | IC of ensemble on 2022-2023 held-out test | Phase 15 primary result |
| OQ-P14-03 | Sequential vs. parallel IC comparison | Phase 15 ablation (DJ-096) |
| OQ-P14-06 | Sharpe vs. SPY buy-and-hold | Phase 15 secondary metric |
| OQ-AG03 | Is the ensemble calibrated? (confidence vs. accuracy) | Phase 15 calibration curve |

---

## Phase 15 → Phase 16 Handoff

1. IC/IR/Sharpe results across all 4 conditions documented
2. Regime-conditional performance table (bull/bear/rate_shock/recovery)
3. OQ-P14-03 answer: sequential vs. parallel — which is better and by how much?
4. Dataset Family G primary artifact: all 25,200+ EnsembleOutputs as JSON
5. Open questions for Phase 16: does live performance match walk-forward?
