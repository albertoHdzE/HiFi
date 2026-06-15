# Phase 12.1: Completion and Correction
## Context and Pre-Phase Decisions

**Gathered:** 2026-06-15
**Status:** Executing
**Parent phase:** Phase 12 (GraphRAG + Structured Debate)
**Trigger:** Phase 12 evaluation produced three categories of findings requiring
corrective action before Phase 13 entry.

---

## Why Phase 12.1 Exists

Phase 12 infrastructure is complete (1197 tests, 0 lint). Phase 12 LLM evaluation
revealed three independent blockers that prevent scientifically valid Phase 13 entry:

### Blocker 1 — technical_v1 Failure (GR=0.000)

Phase 11 trained technical_v1 (rank 8, 1000 iters, 26,433 examples). Evaluation showed
GR collapsed from 1.000 (base) to 0.000 (fine-tuned). Root cause: compliance:domain
ratio = 0.19% (50/26,433). The format prior was overwhelmed by domain signal.

Phase 12 E0-T1 generated 200 compliance examples (`technical_compliance_v2.jsonl`),
raising the ratio to ~0.75%. Retraining (E0-T2) and evaluation (E0-T3) were deferred
as "Wave 2" hardware-bound tasks. They remain unexecuted.

**Impact:** The 2x2 factorial experiment (DJ-067) ran 42/120 conditions. Condition B
(fine-tuned, no debate) produced signal=None in all 12 runs — effectively single-agent.
OQ-M02 (diversity preservation under fine-tuning) cannot be answered. Conditions C/D
(debate) never started.

### Blocker 2 — Incomplete Factorial Evaluation

| Condition | Runs | Status |
|---|---|---|
| A (base, no debate) | 30/30 | COMPLETE |
| B (FT, no debate) | 12/30 | PARTIAL — technical_v1 signal=None |
| C (base, debate) | 0/30 | NOT STARTED |
| D (FT, debate) | 0/30 | NOT STARTED |

OQ-D01 (herding), OQ-D02 (interaction effect), OQ-M02 (diversity) all require
conditions C and D. Phase 12 cannot close without these measurements.

### Blocker 3 — Model Family Monopoly (DJ-080)

Three agents (Fundamental, Technical, Sentiment) share qwen2.5-coder-32b. This
creates correlated failure modes and artificially suppresses ensemble disagreement
(David SS5.2). Phase 13 E1 (Sentiment fine-tuning) would train a third LoRA adapter
on the same base model — amplifying the monopoly rather than breaking it.

DJ-080 (recorded 2026-06-15) prescribes switching Sentiment to gemma-4-12b-it before
Phase 13 begins. This is a Phase 12.1 deliverable because it changes the baseline
that Phase 13 builds upon.

---

## Phase 12.1 Scope

Phase 12.1 is strictly corrective. It produces no new infrastructure — only:

1. **Retrain technical_v2** with augmented compliance data
2. **Complete the factorial evaluation** (all 120 runs)
3. **Implement DJ-080** (Sentiment model swap to Gemma 4 12B)
4. **Re-baseline verification** with new Sentiment model
5. **Close Phase 12** with all OQs answered or scientifically documented

### Explicitly OUT of scope

- New debate infrastructure (Phase 13 E2)
- Multi-round debate (Phase 13 E2)
- Sentiment fine-tuning (Phase 13 E1)
- LLM-extracted graph (Phase 13 E3 — NOT triggered per OQ-K02)
- Agent memory, drift detection, synthetic scenarios (Phase 13)

---

## Evidence Base

### Confirmed Results (from Phase 12)

| Finding | Evidence | Decision |
|---|---|---|
| OQ-K02 NEGATIVE | Doc P@5 delta=0.000 (20 queries, 3 tickers) | DJ-016: KEEP plain RAG |
| OQ-D03 = 36.7% | Condition A: 11/30 non-unanimous dates | XOM=100%, JPM=10%, AAPL=0% |
| fundamental_v1 Buy bias | AAPL Hold->Buy, JPM Hold->Buy (all dates) | Training data bias (2016-2022 bull market) |
| technical_v1 broken | signal=None, 639s/request, GR=0.000 | Compliance ratio root cause confirmed |
| Phase 13 E0 baselines | Risk HR=0.000/GR=1.000, Macro GR=0.000, Sentiment SGR=0.167 | Verification extension working |

### Hardware Constraints

- Mac Studio Ultra M3, 98 GB unified memory
- qwen2.5-coder-32b (8-bit) occupies ~34 GB
- gemma-4-12b-it (Q4) occupies ~6.7 GB
- Training: batch_size=1, grad_accumulation=4 (32B peaks ~52 GB)
- Concurrent serving: max 2x 32B (LM Studio + 1 mlx_lm.server) safely

---

## Decisions

### DJ-081: Phase 12.1 as Decimal Sub-Phase

**Decision:** Create Phase 12.1 (Completion and Correction) as a formal sub-phase
with its own CONTEXT, PLAN, and bitacora documents.

**Rationale:** Phase 12 produced valid infrastructure and partial results. The
corrective work (technical_v2 retrain, factorial completion, model swap) is
scientifically distinct from Phase 12's design work and from Phase 13's new
capabilities. A decimal phase captures this without disrupting the Protocol's
18-phase structure.

**Precedent:** Standard practice in experimental protocols — a "corrigendum" phase
addresses issues discovered during evaluation before the next experimental cycle begins.

### DJ-082: technical_v2 Training Parameters

**Decision:** Train at rank 8, 500 iterations (half of v1's 1000), with augmented
compliance data (200 examples merged with 26,430 domain examples = 26,630 total).

**Rationale:**
- Rank 8 confirmed optimal in Phase 11 sweep (loss 0.299 vs 0.314/0.296/0.298)
- 500 iters reduces domain overfitting risk while maintaining convergence
- fundamental_v1 succeeded at 1000 iters with same compliance ratio — halving iters
  is a conservative correction for the format/domain imbalance

**Gate:** GR >= 0.720 (DJ-058). If FAIL: abandon technical fine-tuning, document as
empirical result, run factorial with base Technical model.

### DJ-083: Factorial Re-run Strategy

**Decision:** Full 120-run re-execution (not incremental from checkpoint).

**Rationale:** The existing 42 runs used technical_v1 (broken) for condition B.
These results are scientifically invalid for cross-condition comparison. A clean
re-run with technical_v2 (or base model if v2 fails) ensures all conditions use
the same model configuration. The checkpoint infrastructure supports this — set
`--force-rerun` or delete `checkpoint.json`.

### DJ-084: Gemma 4 12B Model Variant Selection

**Decision:** Use `mlx-community/gemma-4-12b-it-4bit` (Q4 quantized) via LM Studio.

**Rationale:**
- Q4 quantization: ~6.7 GB memory, fits alongside 32B base model (~34 GB)
- Total memory with both: ~41 GB out of 98 GB available (comfortable margin)
- MLX-optimized variant ensures Apple Silicon compatibility
- LM Studio provides the OpenAI-compatible API layer (same as all other agents)

**Alternative considered:** `gemma-4-12b-it-qat-4bit` (quantization-aware training).
QAT variants may have slightly better quality at Q4. If available in LM Studio,
prefer QAT; otherwise standard Q4 is sufficient for base model evaluation.

---

## Open Questions (Phase 12.1 Scope)

| ID | Question | Resolution target |
|---|---|---|
| OQ-M02 | Does fine-tuning preserve diversity (< 10% entropy degradation)? | Factorial conditions A vs B |
| OQ-D01 | Does debate cause herding (entropy decrease A->C > 0.10)? | Factorial conditions A vs C |
| OQ-D02 | Is interaction effect (D-B)-(C-A) positive? | All 4 conditions required |
| OQ-SGR01 | Does Gemma 4 12B improve Sentiment SGR over qwen2.5-coder-32b? | Re-baseline with new model |
