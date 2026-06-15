# Phase 12.1: Completion and Correction — Plan

**Created:** 2026-06-15
**Context:** plans/PHASE_12.1_CONTEXT.md
**Bitacora:** doc/bitacora/PHASE_12.1_COMPLETION.md
**Dependencies:** Phase 12 infrastructure (complete), LM Studio, venvs/finetune/

---

## Goal

Close Phase 12 with all open questions answered (or scientifically documented as
unanswerable with root cause), and establish the corrected baseline for Phase 13 entry.

**Success criteria:**
1. technical_v2 trained and evaluated (GR >= 0.720 PASS, or documented FAIL)
2. Full factorial (120 runs) complete with valid models
3. OQ-M02, OQ-D01, OQ-D02 answered
4. Sentiment agent running on Gemma 4 12B (DJ-080)
5. SGR re-baseline captured with new model
6. All results documented in bitacora with scientific justification
7. Phase 12 status -> COMPLETE in STATUS.md

---

## Dependency Graph

```
Track A (Fine-Tuning)          Track B (Model Swap)
=====================          ====================
W1: Train technical_v2         W5: Download Gemma 4 12B
         |                              |
W2: Evaluate GR gate           W6: Sentiment code swap
         |                              |
W3: Full factorial (120)       W7: SGR re-baseline
         |                              |
W4: Phase 12 close  <------->  W4: Phase 12 close
                     merge
```

Track A and Track B are independent. W5-W7 can execute in parallel with W1-W3.
W4 requires both tracks complete.

---

## Work Items

### W1: Train technical_v2 (Hardware-Bound)

**Ticket:** P12.1-W1
**Inputs:** `data/training/technical_max_return_60d.jsonl` (26,430) +
`data/training/technical_compliance_v2.jsonl` (200)
**Output:** `data/adapters/technical_v2/`
**Parameters:** rank 8, 500 iters, batch_size=1, grad_accumulation=4, lr=1e-5

**Command:**
```bash
uv run python scripts/run_phase11_finetune.py \
    --agent technical \
    --rank 8 \
    --iters 500 \
    --adapter-name technical_v2
```

**Estimated time:** ~1.5-2 hours on M3 Ultra
**Gate:** Training loss < 0.400

### W2: Evaluate technical_v2 (Hardware-Bound)

**Ticket:** P12.1-W2
**Depends on:** W1
**Input:** `data/adapters/technical_v2/`

**Steps:**
1. Update `scripts/serve_finetune_models.sh`: change `technical_v1` -> `technical_v2`
2. Start servers: `make finetune-serve`
3. Wait 15s for model loading
4. Run evaluation:
```bash
uv run python scripts/run_phase11_finetune.py --evaluate-only \
    --agent technical --adapter-name technical_v2
```
5. Record HR, GR, accuracy

**Gate (DJ-058):**
- GR >= 0.720: DEPLOY technical_v2 -> proceed to W3 with fine-tuned models
- GR < 0.720: ABANDON technical fine-tuning -> proceed to W3 with base Technical model
  (condition B uses base Technical, same as condition A for Technical agent)

### W3: Full Factorial Evaluation (Hardware-Bound)

**Ticket:** P12.1-W3
**Depends on:** W2 (to know which models to use)
**Output:** `tests/fixtures/baseline/phase12_factorial_results.json`

**Steps:**
1. Delete stale checkpoint: `rm data/evaluation/phase12/checkpoint.json`
2. Start all required servers (LM Studio + fine-tuned on 1235/1236)
3. Run:
```bash
make eval-phase12  # 120 runs, checkpointed
```
4. If W2 was FAIL: run with `HIFI_TECHNICAL_FINETUNE_URL` unset (base model for all)

**Estimated time:** ~10 hours sequential (checkpointed, resumable)
**Answers:** OQ-M02, OQ-D01, OQ-D02, OQ-D03 (confirmation)

### W4: Phase 12 Close-Out

**Ticket:** P12.1-W4
**Depends on:** W3 + W7 (both tracks complete)

**Steps:**
1. Update bitacora with final factorial results and all OQ answers
2. Update notebook sections 3, 5, 6 with real data from fixtures
3. Update STATUS.md: Phase 12 -> COMPLETE, Phase 12.1 -> COMPLETE
4. Update MEMORY.md with final results
5. Commit: "Phase 12.1 complete: factorial evaluation + Gemma 4 Sentiment baseline"

### W5: Download Gemma 4 12B (User Action)

**Ticket:** P12.1-W5
**No code dependency. Can start immediately.**

**Steps:**
1. Open LM Studio
2. Search: `gemma-4-12b-it`
3. Download: `mlx-community/gemma-4-12b-it-4bit` (prefer QAT variant if available)
4. Load model and verify inference with test prompt:
   "Analyze the sentiment of this text: Apple reported record quarterly revenue."

**Gate:** Model loads, produces coherent JSON-structured output, response time < 30s

### W6: Sentiment Agent Model Swap (DJ-080)

**Ticket:** P12.1-W6
**Depends on:** W5 (model available in LM Studio)

**Files to modify:**
1. `src/hifi/agents/sentiment_agent.py` — change `_DEFAULT_SENTIMENT_MODEL`
2. `src/hifi/agents/lm_client.py` — no changes expected (model name is just a string)

**Tests:** All existing sentiment tests must pass (they mock the LLM, so model name
change is transparent). Integration test requires LM Studio with Gemma 4 loaded.

### W7: SGR Re-Baseline

**Ticket:** P12.1-W7
**Depends on:** W6 + LM Studio with Gemma 4 loaded

**Command:**
```bash
uv run python scripts/run_phase13_verification_baseline.py
```

**Current baseline (qwen2.5-coder-32b):** mean_SGR=0.167 (1/6 grounded)
**Expected:** Different value (better or worse — this is a measurement, not an optimization)
**Output:** Update `tests/fixtures/baseline/phase13_verification_baseline.json`

---

## Test Checkpoints

| Checkpoint | Expected |
|---|---|
| After W6 (model swap code) | 1197 tests pass, 0 lint |
| After W4 (full close) | >= 1197 tests pass, 0 lint |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| technical_v2 GR < 0.720 | Medium | Factorial loses FT-specific conditions | Run A/C with base; document as empirical finding |
| Gemma 4 12B not in LM Studio | Low | Cannot complete Track B | Use `lmstudio-community` or `bartowski` quant |
| Factorial exceeds 10h | Medium | Schedule | Checkpointed; run overnight |
| 3x 32B OOM | Low | Cannot serve simultaneously | Schedule B/D after A/C (max 2x 32B concurrent) |
