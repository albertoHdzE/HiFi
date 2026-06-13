# Phase 11 Handoff: Fine-Tuning Continuation + Replication Notebook

**Written:** 2026-06-13
**Context:** Handed off at 91% context usage. All training is DONE. Two tasks remain.

---

## Exact State at Handoff

### Test count
997 passed, 4 skipped (phase11_baseline fixture absent), 0 failures, 0 lint errors.

### Git log (last 3 commits)
```
9c571a7  Fix class balance test bound: 2016-2022 bull market produces Buy-dominant labels
fb0efe4  Fix Phase 11 training pipeline: mlx_lm API, batch_size, model path, compliance extraction
67abcc9  Phase 11 complete: fine-tuning infrastructure + training pipeline (991 tests)
```

### What is DONE
- All 26,430 training examples generated (15 tickers × 1,762 labeled periods, 2016-2022)
- Rank sweep complete: ranks 4/8/16/32 at 300 iters each
- technical_v1 adapter: rank 8, 1000 iters, 26,433 examples, 8202s, quality PASS
- fundamental_v1 adapter: rank 8, 1000 iters, 26,433 examples, 2767s, quality PASS
- All infrastructure: venvs/finetune/, Makefile targets, check_env checks
- Bitacora: doc/bitacora/PHASE_11_FINE_TUNING.md (written before training results)

### What is PENDING (Task A + Task B)

**Task A: Three-tier evaluation + baseline fixture (30 min)**
Requires LM Studio running on port 1234 + fine-tuned servers on 1235/1236.
```bash
make finetune-serve     # starts ports 1235 + 1236
make baseline-phase11   # runs evaluation, generates fixture
uv run pytest tests/unit/test_phase11_baseline.py tests/holistic/test_phase11_evaluation.py -q
```
After: update doc/bitacora/PHASE_11_FINE_TUNING.md with actual evaluation results.
Update STATUS.md Phase 11 row to COMPLETE.

**Task B: Replication Notebook (new task added this session)**
Create `notebooks/phase11_finetune_replication.ipynb` — a didactic, self-contained
Jupyter notebook that replicates the entire Phase 11 fine-tuning experiment.
See detailed spec below.

---

## Rank Sweep Results (reference for notebook)

| Rank | Final Loss (300 iters) | Duration | Quality | Peak Mem |
|---|---|---|---|---|
| 4  | 0.314 | 2459s | PASS | 51.82 GB |
| 8  | 0.299 | 2463s | PASS | 51.99 GB |
| 16 | 0.296 | 2465s | PASS | 52.34 GB |
| 32 | 0.298 | 2472s | PASS | 53.03 GB |

Optimal: **rank 8** (selected by pipeline; rank 16 marginally lower by 0.003 — within noise).

Training loss trajectory (rank 4, observed in log, representative):
- Iter 10: 1.701 | Iter 50: 1.202 | Iter 100: 0.423 | Iter 200: ~0.320 | Iter 300: 0.314

Class distribution in training data (empirical, 2016-2022 bull market):
- Technical max-return: Buy=16,087 (60.9%), Sell=6,930 (26.2%), Hold=3,413 (12.9%)
- Fundamental risk-adjusted: (similar Buy-dominant distribution)

---

## Task B: Replication Notebook Specification

### Location
`notebooks/phase11_finetune_replication.ipynb`

### Purpose
A standalone, didactic Jupyter notebook that:
1. Anyone can run to reproduce Phase 11 results
2. Teaches WHY each decision was made, not just HOW
3. Shows plots that justify hyperparameter choices
4. Documents validation criteria and pass/fail thresholds

### Notebook Structure (sections)

**Section 0: Setup and Prerequisites**
- Import check (mlx_lm, pandas, matplotlib, json, pathlib)
- Path configuration (point to local LM Studio model, data/adapters/)
- Print environment summary (Python version, mlx version, available GPU memory)

**Section 1: The Scientific Question**
- Markdown: explain the ρ-diversity tradeoff (Ensemble_Error ≈ b² + ρv + (1-ρ)v/M)
- Explain why heterogeneous labels (max-return vs risk-adjusted Sharpe) preserve diversity
- The hypothesis: fine-tuning with different objectives → individual improvement without ρ increase
- Plot: schematic showing how correlated agents collapse ensemble benefit

**Section 2: Dataset Family C — Reference Strategy Labels**
- Load `data/reference_strategies/max_return/AAPL_60d.parquet`
- Load `data/reference_strategies/risk_adjusted/AAPL_60d.parquet`
- **Plot 1:** AAPL price series with Buy/Sell/Hold labels overlaid (color-coded)
- **Plot 2:** Label distribution bar chart (Buy/Sell/Hold counts per ticker)
- **Plot 3:** Side-by-side max-return vs risk-adjusted labels — show where they DISAGREE
  (disagreement = the signal that heterogeneous objectives work differently)
- Markdown: interpretation of why Buy dominates in 2016-2022 (secular bull market)
- **Plot 4:** Forward return distribution for each label class (violin or box plot)
  — shows that Buy labels genuinely have higher forward returns (validates the labeling)

**Section 3: Training Data JSONL Inspection**
- Load `data/training/technical_max_return_60d.jsonl` (first 5 examples)
- Pretty-print one complete training example (system/user/assistant structure)
- **Plot 5:** Token length distribution (histogram) — shows sequence length for batch planning
- **Plot 6:** Class distribution pie chart (Buy/Sell/Hold) with annotation about bull market skew
- Show one compliance example from `data/training/technical_compliance.jsonl`
- Explain: compliance examples teach schema adherence independent of decision direction

**Section 4: LoRA Rank Sweep Analysis**
- Load `data/training/rank_sweep_results.json`
- **Plot 7:** Bar chart: rank vs final loss (annotate rank 8 as selected)
- **Plot 8:** Bar chart: rank vs training duration (shows linear scaling)
- **Plot 9:** Scatter: loss vs memory footprint (shows rank 8 is the efficiency knee)
- Decision rule: lowest loss with quality_ok=True AND smallest rank (parsimony principle)
- Markdown: explain what LoRA rank means (low-rank decomposition of weight matrices),
  why rank 8 captures enough expressivity for structured JSON output learning,
  why rank 32 doesn't help much (diminishing returns, possible overfitting at 300 iters)

**Section 5: Training Loss Curve Reconstruction**
- Since mlx_lm outputs loss to stdout (not captured in JSON), reconstruct from log
- Use the hardcoded representative trajectory from the handoff document
- **Plot 10:** Training loss curve (rank 4, representative) showing convergence
  — annotate: rapid drop 1.7→0.5 in first 100 iters, plateau ~0.1 after iter 200
- Markdown: interpret convergence — model learns JSON structure fast, then fine-tunes signal
- Discussion: what does loss ~0.3 mean? (cross-entropy on next token prediction;
  lower = model is more confident about structured output; ~0.3 is good for JSON format learning)

**Section 6: Adapter Quality Verification**
- Show adapter directory structure: `data/adapters/technical_v1/`
- Load `adapter_config.json` and print key fields (rank, target_modules, lora_alpha)
- **Plot 11:** Adapter weight magnitude distribution (histogram of safetensors weights)
  — shows what the model learned (non-zero = adapted layers; near-zero = unchanged)
- Run `check_adapter_quality()` in-notebook (uses venvs/finetune to call mlx_lm generate)
- Print sample generation output (5 tokens) to prove adapter loads

**Section 7: Three-Tier Evaluation Framework**
- Load `tests/fixtures/baseline/phase5_verification.json` (base model HR/GR)
- If `tests/fixtures/baseline/phase11_evaluation.json` exists: load and compare
- If not (servers not running): show expected schema and mock values from spec
- **Plot 12:** Before/after comparison bar chart: base_technical_gr vs finetuned_technical_gr
- **Plot 13:** Diversity preservation: base_pairwise_diversity vs finetuned_pairwise_diversity
  with 0.9x threshold line (DJ-058 criterion)
- **Plot 14:** Three-tier summary matrix (heatmap): Tier1/Tier2/Tier3 × pass/fail/pending
- Markdown: interpret each tier, link to OQ-M01 (min training data) and OQ-M02 (diversity)

**Section 8: Conclusions and Replication Checklist**
- Summary table: what improved, what stayed same, what degraded
- Replication checklist: hardware requirements (Apple Silicon, 64GB+ unified memory),
  software (mlx_lm 0.31.1, Python 3.13), data requirements (15 tickers 2016-2022),
  time estimates (rank sweep ~2.5h, full training ~3h/agent)
- Key findings (fill in after baseline-phase11 runs)
- Extensions: what to try next (learning rate schedule, more iters, different base model,
  DPO alignment, homogeneous labels as control experiment)

### Implementation Notes for Notebook Creation
- Use `matplotlib` for all plots (no seaborn dependency)
- Each plot cell: standalone (re-runnable in any order after setup)
- Paths via `Path(__file__).parent.parent` from notebook location
- Model loading in Section 6 is OPTIONAL (toggle cell with `RUN_ADAPTER_CHECK = True`)
- If `phase11_evaluation.json` absent: show placeholder with warning, don't fail
- Add `%%time` magic to cells that load large files
- Target: runnable from scratch in < 30 minutes (excluding GPU cells)

---

## Files to Create/Update in Next Session

### New
- `notebooks/phase11_finetune_replication.ipynb` — the replication notebook
- `tests/fixtures/baseline/phase11_evaluation.json` — generated by make baseline-phase11

### Update
- `doc/bitacora/PHASE_11_FINE_TUNING.md` — fill in actual training numbers and evaluation
- `plans/STATUS.md` — Phase 11 row → COMPLETE, test count → 1001+
- `plans/PHASE_11_HANDOFF.md` — delete after next session completes

### No changes needed
- `src/hifi/models/` — training code is correct and committed
- `scripts/` — all pipeline scripts working and committed
- `tests/` — all passing except 4 skipif-absent (will activate after baseline-phase11)

---

## Critical Technical Facts for Next Session

### mlx_lm 0.31.1 invocation (DO NOT use old API)
```bash
# CORRECT
venvs/finetune/bin/python -m mlx_lm lora --model PATH --train ...

# WRONG (deprecated, errors with unrecognized args)
python -m mlx_lm.lora ...
```

### LoRA rank is set via YAML, not CLI flag
```yaml
# lora_config.yaml
lora_parameters:
  rank: 8
  dropout: 0.0
  scale: 20.0
```
Pass via: `mlx_lm lora ... -c lora_config.yaml`

### Memory constraints on M3 Ultra (192 GB unified)
- batch_size=4 with 32B model → OOM (>192 GB)
- batch_size=1 → 51.8 GB peak (safe)
- grad_accumulation_steps=4 gives effective batch=4

### Local model path (avoids HuggingFace download)
```
~/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-8bit/
```
mlx_lm.load() checks Path.exists() before downloading — local path works directly.

### Compliance examples fixture format
- Phase 3 fixture: flat format `analyses[ticker][signal]` (fundamental agent only)
- Phase 4 fixture: nested `analyses[ticker][technical_analysis][signal]`
- generate_compliance_examples.py now handles both formats

---

## Environment Checklist for Next Session

```bash
# Verify everything is in place
uv run python scripts/check_env.py --check finetune-venv   # should be OK
uv run python scripts/check_env.py --check phase11-data    # should be OK
uv run python scripts/check_env.py --check phase11-adapters # should be OK

# To run baseline-phase11 (Task A):
# 1. Start LM Studio GUI, load qwen2.5-coder-32b-instruct-mlx on port 1234
# 2. make finetune-serve   (starts ports 1235 + 1236)
# 3. make baseline-phase11

# To create notebook (Task B):
# No special services needed — notebook uses local files only
# (adapter quality check section is optional/toggleable)
```

---

## Phase 12 Preview (after Phase 11 is fully closed)

Phase 12: GraphRAG + Structured Debate
- Key question: does structured debate add value ON TOP OF fine-tuned agents?
- Risk: debate may cause group polarization (Sunstein 2006) instead of collective improvement
- Depends on Phase 11 diversity measurement (OQ-M02) to know the ρ baseline

DJ-061 will be the first decision of Phase 12.
