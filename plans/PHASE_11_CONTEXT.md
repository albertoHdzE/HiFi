# Phase 11: Fine-Tuning — Context and Pre-Phase Decisions

**Gathered:** 2026-06-12
**Status:** Ready for planning

---

## Phase Boundary

Phase 11 delivers domain-specific LoRA fine-tuning for two agents (Technical and
Fundamental), empirically answers OQ-M01 (minimum data for measurable improvement)
and OQ-M02 (does fine-tuning reduce ensemble diversity), and establishes the
fine-tuning infrastructure pattern (venvs/finetune/, mlx_lm.server, adapters/).

Explicitly OUT of scope:
- Structured debate (§12.2.4) — deferred to Phase 12 (entangled variables)
- Herding/VIX correlation analysis — data analysis only, no code; included as
  an analysis script/notebook in Phase 12 when more live data exists
- Paper trading, GraphRAG, Agent Memory — later phases

Explicitly IN scope as a trivial addition:
- Incremental weight update hook (Makefile trigger to auto-label 60d after live runs)

---

## Pre-Phase Decisions (DJ-053 through DJ-060)

### DJ-053: Scope Boundary — Fine-Tuning Only in Phase 11

Structured debate (David §12.2.4) and fine-tuning are independent perturbations
on the ensemble. Running both in the same phase entangles variables: an improvement
after Phase 11 would be unattributable to either mechanism. Scientific rigor
requires measuring fine-tuning effect first, then adding debate as a perturbation
in Phase 12 on the measured baseline.

Herding/VIX correlation requires only data analysis (no new code); it is deferred
as an analysis artifact. Incremental weight updates are 5-10 lines of Makefile
code and are included in Phase 11 because they directly extend the fine-tuning
feedback loop.

### DJ-054: Training Data Strategy — Heterogeneous Labels Preserve Diversity

Primary training signal: Dataset Family C (reference strategies, David §8.4),
generated deterministically from Phase 2 financial engines. Different labeling
strategies assigned to different agents to preserve ensemble decorrelation (David
§5.3, ensemble error formula: Ensemble_Error ≈ b² + ρv + (1-ρ)v/M):

| Agent       | Reference strategy      | Labeling rule                                     |
|---|---|---|
| Technical   | Max-return (60-day)     | BUY if forward_return > +0.02; SELL if < -0.02   |
| Fundamental | Risk-adjusted (60-day)  | BUY if Sharpe_60d > 0.8; SELL if Sharpe_60d < 0.3|

Rationale: Technical analysis naturally captures price momentum (max-return is
its appropriate reference). Fundamental analysis naturally captures risk-adjusted
value creation (Sharpe is its appropriate reference). Training on identical
signals would increase inter-agent correlation ρ; heterogeneous signals preserve
diversity by design.

Secondary training signal (both agents): structured output compliance examples.
Correct JSON schema + MCP field citations from verified Phase 3/5 baseline runs.
This directly targets Technical Agent's GR=0.667 weakness (Phase 5 baseline).

Data volume: 15 tickers × ~31 non-overlapping 60-day periods (2016-2022) =
~465 labeled signal periods per agent. This meets the lower bound of LoRA
literature (200-500 examples for measurable adaptation on chat models).

Reference strategy data stored in: `data/reference_strategies/`

### DJ-055: Agents to Fine-Tune — Technical + Fundamental

Rationale:
- Technical Agent: GR=0.667 at Phase 5 baseline — measurable, attributable
  improvement target. Weakest grounding of any measured agent.
- Fundamental Agent: HR=0.000, GR=1.000 at Phase 5 baseline — already strong.
  Phase 11 tests whether fine-tuning improves *accuracy* (Phase 10 metric)
  without degrading the strong HR/GR. This provides the A/B structure:
  improving a weak baseline (Technical) vs. improving from a strong baseline
  (Fundamental).
- Risk and Macro agents lack Phase 5 HR/GR baselines (introduced in Phase 8)
  so before/after comparison is not possible for them.
- Contrarian agent has no Buy/Hold/Sell vote; fine-tuning it is a different
  architecture (no reference strategy applies directly). Deferred to Phase 13.

### DJ-056: Fine-Tuning Framework — mlx_lm in pyenv, venvs/finetune/ Pattern

mlx 0.31.1 and mlx_lm 0.31.1 are already installed in the system pyenv
(Python 3.13.12, `/Users/alberto/.pyenv/versions/3.13.12/`). They are NOT
in the project's uv virtual environment (Python 3.12.13).

Following the `venvs/ta/` isolation pattern (DJ-010):
- `venvs/finetune/` created with `uv venv --python 3.13 --seed --clear`
- mlx and mlx-lm installed pinned to tested versions
- All fine-tuning scripts invoked via `venvs/finetune/bin/python`
- `scripts/setup_finetune_venv.sh` bootstraps the environment
- `scripts/check_env.py` gets a `finetune-venv` check

This pattern ensures version pinning, reproducibility, and isolation from the
main project stack. In Phase 15 (containerization), venvs/finetune/ maps to a
training Docker service with its own base image.

LoRA via `mlx_lm.lora` (CLI). Fine-tuned adapters stored in `data/adapters/`:
- `data/adapters/technical_v1/` — Technical Agent LoRA adapters
- `data/adapters/fundamental_v1/` — Fundamental Agent LoRA adapters

Rank sweep: 4, 8, 16, 32 on Technical Agent only to identify optimal rank
(DJ-015). Fundamental Agent uses the rank identified as optimal. This halves
sweep compute while still providing empirical calibration.

### DJ-057: Fine-Tuned Model Serving — mlx_lm.server on Separate Port

mlx_lm.server provides an OpenAI-compatible HTTP API (same protocol as LM
Studio). Fine-tuned models are served alongside LM Studio rather than replacing it:

| Service          | Port | Model                              | Purpose                  |
|---|---|---|---|
| LM Studio        | 1234 | qwen2.5-coder-32b (base)           | All existing agents      |
| mlx_lm.server    | 1235 | qwen2.5-coder-32b + technical_v1   | Fine-tuned Technical eval|
| mlx_lm.server    | 1236 | qwen2.5-coder-32b + fundamental_v1 | Fine-tuned Fundamental eval|

New environment variables:
- `HIFI_TECHNICAL_FINETUNE_URL=http://localhost:1235/v1`
- `HIFI_FUNDAMENTAL_FINETUNE_URL=http://localhost:1236/v1`

Evaluation runs the ensemble twice: once with base model (LM Studio, existing
`HIFI_LM_STUDIO_URL`), once with fine-tuned models (the two new URLs). The
before/after comparison is deterministic given identical inputs.

Makefile targets:
- `finetune-setup` — runs setup_finetune_venv.sh
- `finetune-train-technical` — runs LoRA training for Technical Agent
- `finetune-train-fundamental` — runs LoRA training for Fundamental Agent
- `finetune-serve` — starts both mlx_lm.server instances (background)
- `finetune-stop` — stops mlx_lm.server instances

### DJ-058: Three-Tier Evaluation Protocol (Answer to OQ-M01 + OQ-M02)

Fine-tuning is only deployed if it demonstrably outperforms the base model
(David §9.4 critical requirement). Three evaluation tiers:

**Tier 1 — Individual quality (required for deployment decision):**
- HR and GR from Phase 5 verification layer (existing, reuse)
- JSON schema compliance rate (parse success vs. failure)
- Success criterion: GR improvement ≥ 0.05 for Technical (0.667 → ≥ 0.72)

**Tier 2 — Collective accuracy (aggregate evidence):**
- Run full ensemble with fine-tuned agents on Phase 10 labeled tickers/dates
- Method-level accuracy comparison: fine-tuned ensemble vs. base ensemble
- Success criterion: at least one method shows accuracy improvement ≥ 0.05

**Tier 3 — Diversity impact (the publishable result, OQ-M02):**
- Measure pairwise_diversity and disagreement_entropy before/after fine-tuning
- Measure inter-agent vote correlation ρ directly
- Success criterion: diversity metrics do NOT decrease by > 10% (diversity
  degradation is the primary risk of shared training data)

The Pareto frontier of (GR improvement) vs. (pairwise_diversity change) across
LoRA ranks provides the primary scientific result of Phase 11.

### DJ-059: Training Data Module Placement

New package: `src/hifi/models/` — consistent with David spec (`src/models/`) but
adapted to the established `src/hifi/*` convention.

Files:
- `src/hifi/models/__init__.py` — package stub
- `src/hifi/models/training_data.py` — generate_technical_training_data(),
  generate_fundamental_training_data(), format_as_jsonl()
- `src/hifi/models/fine_tune.py` — run_lora_training(), check_adapter_quality()

Scripts:
- `scripts/generate_reference_strategies.py` — generate Dataset Family C Parquets
- `scripts/run_phase11_finetune.py` — full fine-tuning pipeline (data → train → eval)
- `scripts/setup_finetune_venv.sh` — bootstrap venvs/finetune/

### DJ-060: Incremental Weight Update Hook

After any `make baseline-phaseN` or `make test-live` run, the Makefile
automatically calls `scripts/run_phase10_labeling.py` with a 60-day lookback
from the analysis date to add real LLM labels to `agent_performance_history.json`.

This means every live run automatically extends the labeled dataset, making
`performance_weighted` aggregation progressively more data-driven.

New Makefile target: `label-outcomes` — labels any unlabeled records in the
performance history where 60 trading days have elapsed since analysis_date.
Added as a post-step in `test-live`.

---

## Canonical References

Downstream agents MUST read these before planning or implementing.

### Core Specification
- `doc/HIFI_DAVID.md` §9.4 — Fine-Tuning Strategy (method choices, framework, data)
- `doc/HIFI_DAVID.md` §8.4 — Dataset Family C (reference strategy labeling rules)
- `doc/HIFI_DAVID.md` §10.3 — Diversity Requirements (agents must differ ≥2 dimensions)
- `doc/HIFI_DAVID.md` §5.3 — Ensemble Learning (ρ formula, diversity theorem)
- `doc/HIFI_PROTOCOL_V1.md` §Phase 11 — Deliverables and success criteria

### Phase Context
- `plans/PHASE_10_PLAN.md` — Phase 10 decisions DJ-044 through DJ-052
- `doc/bitacora/PHASE_10_EVALUATION.md` — Phase 10 results: accuracy=0.0, bootstrap accuracy per agent
- `doc/bitacora/PHASE_05_VERIFICATION.md` — HR/GR baseline: Technical GR=0.667, Fundamental GR=1.000
- `doc/bitacora/PHASE_09_COLLECTIVE_ENGINE.md` — DJ-039 through DJ-043

### Environment
- `scripts/setup_ta_venv.sh` — Pattern to follow for setup_finetune_venv.sh
- `scripts/check_env.py` — Pattern for adding finetune-venv check
- `Makefile` — Established target patterns (finetune-setup must follow langfuse-setup style)
- `docker/langfuse/docker-compose.yml` — LangFuse stack (note: ClickHouse unhealthy on macOS)

### Agent Prompts (training data must match these templates)
- `src/hifi/agents/prompts/technical_v1.md` — Technical Agent prompt template (base)
- `src/hifi/agents/prompts/technical_v2.md` — Technical Agent prompt with RAG
- `src/hifi/agents/prompts/fundamental_v1.md` — Fundamental Agent prompt (base)
- `src/hifi/agents/prompts/fundamental_v2.md` — Fundamental Agent prompt with RAG

---

## Deferred Ideas

The following were raised during Phase 11 scoping and explicitly deferred:

- **Structured debate (§12.2.4):** Multi-turn agent exchange before final vote.
  Requires independent baseline measurement first. Phase 12.

- **Herding/VIX correlation:** κ vs VIXCLS over 2018-2022. Data exists now;
  analysis requires FRED VIX download + correlation computation. Phase 12 or
  standalone analysis notebook.

- **Axolotl / Unsloth comparison:** David §9.4 lists these as fallback/comparison
  frameworks. Phase 11 uses mlx_lm exclusively (hardware-appropriate). If mlx_lm
  produces suboptimal results, Axolotl evaluation defers to Phase 13+.

- **Adaptive aggregation (§12.2.5):** Learned aggregation function from data.
  Requires substantially more labeled data than Phase 11 produces. Phase 13+.
