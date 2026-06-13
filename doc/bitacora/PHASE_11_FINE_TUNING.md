# Phase 11 Bitacora: Fine-Tuning Infrastructure and Training Pipeline

**Date completed:** 2026-06-12
**Tests at close:** 991 passed, 10 skipped (training JSONL absent), 0 lint errors
**Status:** COMPLETE (infrastructure and pipeline; LoRA training pending hardware run)

---

## Objective

Build the complete infrastructure for LoRA fine-tuning of the Technical and
Fundamental agents, measure the diversity-quality tradeoff empirically, and
answer two open questions from the Phase 10 close:

- **OQ-M01:** What is the minimum training data quantity needed for stable LoRA
  convergence with mlx_lm at rank 4-32?
- **OQ-M02:** Does fine-tuning with heterogeneous labels preserve ensemble
  diversity (pairwise_diversity >= 0.9 x base)?

Phase 11 is designed as a controlled experiment: each agent is trained toward
a different objective function (max-return for Technical, risk-adjusted Sharpe
for Fundamental) to preserve the ρ-diversity that gives the ensemble its
collective advantage over any single agent. The scientific hypothesis is that
heterogeneous labels enable individual improvement without collective regression.

---

## Architecture Decisions (DJ-053 through DJ-060)

### DJ-053: Scope = Fine-Tuning Only; Structured Debate Deferred to Phase 12

Phase 11 is scoped to LoRA fine-tuning of individual agents. Structured debate
(multi-agent adversarial deliberation) is deferred to Phase 12 after we have
empirical evidence of fine-tuned agent behavior. Combining fine-tuning and
debate architecture changes in one phase would make the experiment uninterpretable.

### DJ-054: Dataset Family C -- Heterogeneous Labels per Agent

Two label strategies for two agents:

- **Technical Agent:** Max-return labels. Forward 60-day return > +2% = Buy,
  < -2% = Sell, otherwise Hold. Optimizes for directional accuracy.
- **Fundamental Agent:** Risk-adjusted Sharpe labels. Forward 60-day rolling
  Sharpe > 0.8 = Buy, < 0.3 = Sell, otherwise Hold. Optimizes for
  risk-adjusted performance.

Training universe: 15 tickers, 2016-2022 (approximately 1,500 trading days,
yielding ~1,440 labeled periods per ticker after the 60-day lookahead window).
Target >= 400 training examples per agent after class balancing.

Look-ahead bias is acknowledged and intentional (David §8.4). These are
reference strategy labels for supervised fine-tuning, not production signals.

### DJ-055: Fine-Tune Technical (GR=0.667 → 0.720 target) + Fundamental

Technical Agent has GR=0.667 from Phase 5 baseline -- the only agent with
meaningful room for improvement. Fundamental Agent has GR=1.000 from Phase 5
but low accuracy (0.079 from Phase 10 bootstrap), indicating format compliance
without predictive quality. Fine-tuning addresses complementary weaknesses.

Success threshold: GR improvement >= 0.05 for Technical (0.667 → 0.717+).

### DJ-056: mlx_lm in venvs/finetune/ (Python 3.13, Same Pattern as venvs/ta/)

The `venvs/finetune/` environment follows the established isolation pattern:
- Created by `scripts/setup_finetune_venv.sh` (idempotent, `--clear` on each run)
- Pins `mlx==0.31.1` and `mlx-lm==0.31.1` to match system-installed versions
- Never imported by the main uv project environment

This ensures: (1) mlx_lm's Apple Silicon GPU code stays isolated from the
pandas/langchain stack; (2) the main env tests remain fast on non-Apple hardware
since mlx is not a dependency.

### DJ-057: Fine-Tuned Serving via mlx_lm.server Ports 1235/1236

Two mlx_lm.server instances run alongside LM Studio (port 1234):
- Port 1235: Technical Agent adapter (data/adapters/technical_v1/)
- Port 1236: Fundamental Agent adapter (data/adapters/fundamental_v1/)

Both serve OpenAI-compatible API, so the existing agent code only needs an
environment variable override (HIFI_TECHNICAL_MODEL/HIFI_FUNDAMENTAL_MODEL) to
switch from base to fine-tuned model. No architectural changes to agents.

Scripts: `scripts/serve_finetune_models.sh`, `make finetune-serve`,
`make finetune-stop`.

### DJ-058: Three-Tier Evaluation Protocol

The evaluation measures three independent dimensions:

- **Tier 1:** HR/GR grounding (Phase 5 verifier infrastructure). Measures whether
  fine-tuning changes the hallucination rate (HR must not increase; GR should
  improve for Technical Agent).
- **Tier 2:** Forward-return accuracy (Phase 10 labeler infrastructure). Compares
  base vs fine-tuned method accuracy on held-out dates.
- **Tier 3:** Ensemble diversity (pairwise_diversity, disagreement_entropy). The
  primary risk-control check -- fine-tuning that increases ρ beyond the threshold
  where ensemble error > single-agent error must be rejected.

Fine-tuned models are not deployed unless Tier 1 and Tier 3 pass. Tier 2
improvement is desirable but not gating.

### DJ-059: New Package src/hifi/models/

New source package with two modules:

- `training_data.py` -- label generators (max-return, risk-adjusted Sharpe),
  JSONL formatter, FineTuneEvaluationResult schema
- `fine_tune.py` -- run_lora_training() (subprocess to mlx_lm.lora),
  check_adapter_quality(), rank sweep utilities

Separation from `hifi.agents` and `hifi.collective` is intentional: the models
package is the training-time interface; agents are the inference-time interface.

### DJ-060: label-outcomes Makefile Target for Incremental Weight Updates

`make label-outcomes` (script: `scripts/run_label_outcomes.py`) labels any
performance history records where 60 trading days have elapsed since analysis_date.
It runs automatically at the end of `make test-live`. This creates a feedback loop:
every live run incrementally expands the labeled dataset used by Phase 10 accuracy
tracking and Phase 11 fine-tuning data augmentation.

No LM Studio required. Pure Parquet + pandas computation.

---

## Files Created

### Source Package

| File | Purpose |
|---|---|
| `src/hifi/models/__init__.py` | Package stub |
| `src/hifi/models/training_data.py` | generate_max_return_labels(), generate_risk_adjusted_labels(), format_as_jsonl(), FineTuneEvaluationResult |
| `src/hifi/models/fine_tune.py` | run_lora_training(), check_adapter_quality(), load_rank_sweep_results(), optimal_rank_from_sweep() |

### Scripts

| File | Purpose |
|---|---|
| `scripts/setup_finetune_venv.sh` | Create venvs/finetune/ (idempotent) |
| `scripts/serve_finetune_models.sh` | Start mlx_lm.server on ports 1235/1236 |
| `scripts/generate_reference_strategies.py` | Dataset Family C Parquets |
| `scripts/generate_training_jsonl.py` | JSONL per agent |
| `scripts/generate_compliance_examples.py` | Structured output compliance examples |
| `scripts/run_phase11_finetune.py` | Rank sweep + agent training pipeline |
| `scripts/run_phase11_evaluation.py` | Three-tier evaluation |
| `scripts/run_label_outcomes.py` | Incremental outcome labeling |
| `scripts/analyze_rank_sweep.py` | Rank sweep table + recommendation |

### Tests

| File | Count | Notes |
|---|---|---|
| `tests/unit/test_training_data.py` | 18 | Label generators, JSONL format |
| `tests/unit/test_training_jsonl.py` | 6 | All skipif training files absent |
| `tests/unit/test_fine_tune.py` | 12 | LoRA pipeline, subprocess mocking |
| `tests/unit/test_evaluation_schema.py` | 9 | FineTuneEvaluationResult schema |
| `tests/unit/test_label_outcomes.py` | 4 | Incremental labeling |
| `tests/unit/test_phase11_baseline.py` | 4 | skipif fixture absent |
| `tests/holistic/test_phase11_evaluation.py` | 6 | Structural pipeline test |

---

## Training Data Summary

### Dataset Family C -- Reference Strategy Labels

Generated by `scripts/generate_reference_strategies.py` from:
- 15 tickers: AAPL, JPM, XOM, MSFT, NVDA, GOOGL, BAC, GS, CVX, JNJ, UNH, AMZN,
  WMT, CAT, NEE
- Training window: 2016-01-01 through 2022-12-31
- Horizon: 60 trading days
- Output: `data/reference_strategies/max_return/{ticker}_60d.parquet`
           `data/reference_strategies/risk_adjusted/{ticker}_60d.parquet`

**Status:** Pending `make acquire-data-phase10` (internet required) +
`make generate-reference-strategies`.

### Training JSONL (Target)

| File | Agent | Labels | Min Examples |
|---|---|---|---|
| `data/training/technical_max_return_60d.jsonl` | Technical | max-return | >= 400 |
| `data/training/fundamental_risk_adjusted_60d.jsonl` | Fundamental | risk-adjusted Sharpe | >= 400 |
| `data/training/technical_compliance.jsonl` | Technical | Phase 3/5 verified | ~50 |
| `data/training/fundamental_compliance.jsonl` | Fundamental | Phase 3/5 verified | ~50 |

**Status:** Pending (requires reference strategies + `make baseline-phase3` for
compliance examples).

---

## LoRA Rank Sweep Results

**Status:** PENDING -- requires Apple Silicon GPU + venvs/finetune/ + training JSONL.

Command: `uv run python scripts/run_phase11_finetune.py --rank-sweep --sweep-iters 300`

Expected output table (placeholder):

| Rank | Train Loss | Duration (s) | Quality OK |
|---|---|---|---|
| 4  | -- | -- | -- |
| 8  | -- | -- | -- |
| 16 | -- | -- | -- |
| 32 | -- | -- | -- |

Results will be written to `data/training/rank_sweep_results.json` and
`data/training/optimal_rank.json`.

**OQ-M01 answer:** Pending empirical run. Expected finding: rank 8 or 16 yields
the best loss-to-duration tradeoff for a ~450-example dataset. Rank 4 may
underfit (insufficient LoRA capacity for structured JSON output); rank 32 may
overfit given the training set size. The sweep will establish this empirically.

---

## Three-Tier Evaluation Results

**Status:** PENDING -- requires all of: training data, fine-tuned adapters,
LM Studio on 1234, mlx_lm.server on 1235 and 1236.

Command: `make baseline-phase11`

Expected FineTuneEvaluationResult structure:

```json
{
  "ticker": "AAPL",
  "analysis_date": "2023-03-31",
  "base_technical_gr": 0.667,
  "finetuned_technical_gr": null,
  "base_fundamental_gr": 1.000,
  "finetuned_fundamental_gr": null,
  "base_pairwise_diversity": null,
  "finetuned_pairwise_diversity": null,
  "diversity_preserved": null,
  "gr_improved_technical": null,
  "gr_improved_fundamental": null
}
```

**OQ-M02 answer:** Pending empirical run. The null hypothesis (diversity
degrades) will be tested against the experimental hypothesis (heterogeneous
labels preserve diversity). The threshold is finetuned_pairwise_diversity >=
0.9 x base_pairwise_diversity.

---

## Implementation Surprises and Lessons Learned

### Two-File Wave Parallelization for Infrastructure

Wave 1 (E0 + E5) ran in parallel because the serving infrastructure
(venvs/finetune/, Makefile targets) and the label-outcomes script have no shared
dependencies. This pattern -- identify independent wave groups, implement in
parallel -- is consistently the fastest way to build Phase 11 without waiting on
external resources (GPU, internet).

### JSONL Skipif Strategy for Test Hygiene

The 10 tests marked `skipif(training_file_absent)` in `test_training_jsonl.py`
and `test_phase11_baseline.py` run cleanly without generated data. This avoids
the false negative of marking the phase "complete" before training data exists,
while keeping `make test` fast and deterministic. The count will jump from
991 to >= 1001 after training data is generated.

### mlx_lm Subprocess Pattern

`run_lora_training()` invokes mlx_lm.lora as a subprocess via venvs/finetune/
rather than importing mlx_lm directly into the main env. This mirrors the
venvs/ta/ pattern for pandas-ta. The benefit: (1) version pinning is guaranteed;
(2) the main uv project env never acquires an Apple-Silicon-only dependency;
(3) CI on Linux (if added in Phase 15+) will not fail on mlx import.

### Compliance Examples as Secondary Signal

The compliance JSONL (from Phase 3/5 verified outputs where HR=0.000) serves as
a regularization signal: it teaches the model what perfect structured output
looks like independent of the decision direction. This prevents fine-tuning from
degrading format compliance while improving directional accuracy.

### Forward-Return Label Convention

The label-outcomes script uses business-day offset via pandas `BDay(60)` to
compute the 60-trading-day label date. This matches the Phase 9/10 convention
for bootstrap labels. Using calendar days would misalign with the OHLCV Parquet
index which contains only trading days.

---

## Open Questions for Phase 12

1. **Structured debate after fine-tuning.** Phase 11 produces two fine-tuned
   agents with potentially different decision boundaries. Does structured debate
   between them improve collective accuracy, or does it cause polarization?
   Phase 12 designs the debate mechanism and measures this interaction.

2. **Adapter versioning convention.** Phase 11 produces `technical_v1` and
   `fundamental_v1`. A versioning scheme (semantic: v{major}.{minor}) should be
   established before Phase 12 introduces structural debate, because debate may
   require retraining with debate-aware prompts as fine-tuning signal.

3. **GraphRAG vs fine-tuning complement.** Phase 7 implemented standard RAG.
   Phase 12 implements GraphRAG. The Phase 11 fine-tuned fundamental agent
   (trained without RAG context) provides the cleanest baseline for measuring
   whether GraphRAG retrieval adds value on top of domain fine-tuning, or whether
   one subsumes the other.

4. **Performance-weighted differentiation.** After real LLM outputs are labeled
   through `make test-live` + `make label-outcomes`, does performance_weighted
   diverge from confidence_weighted? Phase 10 showed all methods at 0.0 on 3
   tickers. Phase 11's fine-tuned agents on the expanded universe should provide
   the signal differentiation needed for performance weighting to matter.

---

## Phase 11 Completion Status

| Criterion | Status |
|---|---|
| Tests >= 1000, 0 failures | 991 + 10 skipped (10 will activate after data gen) |
| Lint clean | PASS -- 0 errors |
| Infrastructure (finetune-setup, finetune-serve) | IMPLEMENTED -- pending first run |
| Training data (>= 400 examples/agent) | PENDING -- internet + GPU |
| Adapters (technical_v1, fundamental_v1) | PENDING -- GPU |
| OQ-M01 (rank sweep) | PENDING -- GPU |
| OQ-M02 (diversity preserved) | PENDING -- full evaluation run |
| Tier 1 (GR improved) | PENDING -- evaluation run |
| Bitacora | COMPLETE |

The infrastructure phase is complete. The empirical phase (training, evaluation,
OQ answers) requires: (1) `make acquire-data-phase10` for the 15-ticker OHLCV
dataset, (2) `make generate-reference-strategies` for Dataset Family C,
(3) `make finetune-setup` for the venvs/finetune/ environment, and
(4) `make finetune-train` for the actual LoRA runs (~hours on Apple Silicon).
