# Phase 11: Fine-Tuning

**Status:** PLANNED

| Epic | Title | Status |
|---|---|---|
| P11-E0 | Fine-tuning infrastructure (venvs/finetune/, mlx_lm.server, Makefile) | PLANNED |
| P11-E1 | Dataset Family C generation (reference strategies) | PLANNED |
| P11-E2 | Training data formatting (JSONL + structured output examples) | PLANNED |
| P11-E3 | LoRA fine-tuning pipeline (Technical + Fundamental, rank sweep) | PLANNED |
| P11-E4 | Three-tier evaluation (HR/GR + accuracy + diversity) | PLANNED |
| P11-E5 | Incremental weight update hook (label-outcomes) | PLANNED |
| P11-E6 | Baseline measurement + bitacora | PLANNED |

**David Sections:** §9.4 Fine-Tuning Strategy, §8.4 Dataset Family C,
§10.3 Diversity Requirements, §5.3 Ensemble Learning
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 11 (critical capstone path)
**Decision IDs:** DJ-053 through DJ-060 (context: plans/PHASE_11_CONTEXT.md)

---

## Governing Philosophy for This Phase

Phase 10 established the measurement infrastructure. Phase 11 asks the first
controllable question: can we improve individual agent quality through domain
fine-tuning without eroding the ensemble's collective intelligence?

The central tension is the ρ problem (David §5.3):

    Ensemble_Error ≈ b² + ρv + (1-ρ)v/M

Fine-tuning pushes agents toward a common training signal, which risks increasing
inter-agent correlation ρ. If ρ → 1, the ensemble reduces to a single agent with
extra latency cost. Phase 11 is designed as a controlled experiment to measure
this tradeoff empirically.

The scientific hypothesis:
- Heterogeneous training labels (max-return for Technical, risk-adjusted Sharpe
  for Fundamental) will improve individual quality while preserving diversity,
  because each agent is trained toward a different objective function.
- Homogeneous training (same labels for both) would be the null hypothesis
  control -- not implemented in Phase 11 but framed as a future comparison.

This framing is what makes Phase 11 a publishable experiment, not just an
engineering milestone.

---

## Pre-Phase Decisions

See `plans/PHASE_11_CONTEXT.md` for full rationale (DJ-053 through DJ-060).

Key constraints:
- mlx 0.31.1 + mlx_lm 0.31.1 in pyenv Python 3.13.12 (not in uv project env)
- venvs/finetune/ follows the venvs/ta/ pattern (DJ-056)
- Fine-tuned models served via mlx_lm.server alongside LM Studio (DJ-057)
- Training data: ~465 periods per agent from 15 tickers, 2016-2022 (DJ-054)
- Evaluation must measure diversity impact, not just individual improvement (DJ-058)
- src/hifi/models/ is the new package (DJ-059)

---

## Epic P11-E0: Fine-Tuning Infrastructure

**Scope:** Create the venvs/finetune/ isolated environment, mlx_lm.server serving
pattern, and Makefile targets. This epic establishes the infrastructure that all
subsequent Phase 11 epics depend on.

### Tickets

**P11-E0-T1: scripts/setup_finetune_venv.sh**

Creates and populates venvs/finetune/ following setup_ta_venv.sh pattern:

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/venvs/finetune"

echo "Creating venvs/finetune/ with Python 3.13..."
uv venv "$VENV_DIR" --python 3.13 --seed --clear

echo "Installing mlx and mlx-lm..."
"$VENV_DIR/bin/pip" install --quiet \
    "mlx==0.31.1" \
    "mlx-lm==0.31.1"

echo "Verifying installation..."
"$VENV_DIR/bin/python" -c "import mlx_lm; print('mlx_lm', mlx_lm.__version__, 'OK')"
echo "venvs/finetune/ ready."
```

Must be idempotent: --clear recreates each time (ensures pinned versions).
chmod +x after creation.

**P11-E0-T2: check_env.py -- finetune-venv check**

New check function `check_finetune_venv()`:
- Verifies `venvs/finetune/bin/python` exists
- Verifies `import mlx_lm` succeeds in that python
- Returns error list with fix instructions if absent

Registered in the CHECK_REGISTRY dict under key `"finetune-venv"`.

**P11-E0-T3: Makefile targets**

New targets following the established style:

```makefile
FINETUNE_VENV := venvs/finetune/bin/python

finetune-setup: ## Create venvs/finetune/ with pinned mlx+mlx-lm (idempotent)
    bash scripts/setup_finetune_venv.sh
    uv run python scripts/check_env.py --check finetune-venv

finetune-train: ## Run LoRA fine-tuning for both agents (requires finetune-setup)
    uv run python scripts/check_env.py --check finetune-venv
    uv run python scripts/check_env.py --check phase11-data || { \
        echo "Generate training data first: make generate-reference-strategies"; exit 1; }
    uv run python scripts/run_phase11_finetune.py

finetune-serve: ## Start mlx_lm.server for fine-tuned models (background, requires adapters)
    uv run python scripts/check_env.py --check phase11-adapters || { \
        echo "Train first: make finetune-train"; exit 1; }
    bash scripts/serve_finetune_models.sh

finetune-stop: ## Stop mlx_lm.server instances
    pkill -f "mlx_lm.server" 2>/dev/null || true

generate-reference-strategies: ## Generate Dataset Family C JSONL (no LM Studio required)
    uv run python scripts/check_env.py --check market-data || $(MAKE) acquire-data
    uv run python scripts/check_env.py --check phase10-data || $(MAKE) acquire-data-phase10
    uv run python scripts/generate_reference_strategies.py
    uv run python scripts/check_env.py --check phase11-data

label-outcomes: ## Label unlabeled performance records where 60d has elapsed (no LM Studio)
    uv run python scripts/run_label_outcomes.py
```

Also add `label-outcomes` as post-step to `test-live`:
```makefile
test-live: ... existing steps ...
    $(MAKE) label-outcomes
    uv run pytest -q --tb=short
```

**P11-E0-T4: scripts/serve_finetune_models.sh**

Starts both mlx_lm.server instances in the background:

```bash
#!/usr/bin/env bash
set -euo pipefail
VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/venvs/finetune"
MODEL_BASE="mlx-community/Qwen2.5-Coder-32B-Instruct-8bit"
ADAPTERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/adapters"

echo "Starting mlx_lm.server for Technical Agent (port 1235)..."
"$VENV/bin/python" -m mlx_lm.server \
    --model "$MODEL_BASE" \
    --adapter-path "$ADAPTERS_DIR/technical_v1" \
    --port 1235 \
    --log-level warning &
echo "Technical fine-tuned server PID: $!"

echo "Starting mlx_lm.server for Fundamental Agent (port 1236)..."
"$VENV/bin/python" -m mlx_lm.server \
    --model "$MODEL_BASE" \
    --adapter-path "$ADAPTERS_DIR/fundamental_v1" \
    --port 1236 \
    --log-level warning &
echo "Fundamental fine-tuned server PID: $!"

echo "Servers starting. Health check in 10s..."
sleep 10
curl -s http://localhost:1235/health >/dev/null && echo "Port 1235 OK" || echo "Port 1235 NOT READY"
curl -s http://localhost:1236/health >/dev/null && echo "Port 1236 OK" || echo "Port 1236 NOT READY"
```

Note: The model path `mlx-community/Qwen2.5-Coder-32B-Instruct-8bit` is the
HuggingFace Hub ID. mlx_lm.server will download it on first run if not cached.
The LM Studio local model path should also be documented as a fallback.

---

## Epic P11-E1: Dataset Family C Generation

**Scope:** Generate reference strategy training labels for all 15 tickers,
2016-2022, at both 60-day and 20-day horizons. Store as Parquet in
data/reference_strategies/. These are NOT training JSONL files yet (that is E2).

This epic generates the ground-truth signal. Two strategies per DJ-054:
- Max-return labels for Technical Agent training
- Risk-adjusted (Sharpe) labels for Fundamental Agent training

### Tickets

**P11-E1-T1: src/hifi/models/__init__.py + src/hifi/models/training_data.py**

New package stub. `training_data.py` contains:

```python
def generate_max_return_labels(
    ticker: str,
    data_dir: str,
    horizon_days: int = 60,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Generate max-return reference strategy labels for a ticker.

    For each trading day t with at least horizon_days remaining:
      forward_return = (close[t+horizon_days] - close[t]) / close[t]
      label = "Buy"  if forward_return > threshold
      label = "Sell" if forward_return < -threshold
      label = "Hold" otherwise

    Returns DataFrame with columns: date, ticker, label, forward_return, horizon_days.
    Excludes any period where insufficient data remains (unlabeled).
    """

def generate_risk_adjusted_labels(
    ticker: str,
    data_dir: str,
    horizon_days: int = 60,
    sharpe_buy: float = 0.8,
    sharpe_sell: float = 0.3,
) -> pd.DataFrame:
    """
    Generate risk-adjusted reference strategy labels using rolling Sharpe.

    For each trading day t, compute Sharpe over the forward horizon_days window:
      forward_returns = daily_pct_change(close[t:t+horizon_days])
      sharpe = mean(forward_returns) / std(forward_returns) * sqrt(252)
      label = "Buy"  if sharpe > sharpe_buy
      label = "Sell" if sharpe < sharpe_sell
      label = "Hold" otherwise

    Returns same DataFrame schema as generate_max_return_labels.
    Note: Look-ahead bias is acknowledged and intentional (David §8.4).
    """

def format_as_jsonl(
    labels_df: pd.DataFrame,
    ticker: str,
    agent_type: str,
    data_dir: str,
    prompt_template_path: str,
    mcp_tool_outputs: dict | None = None,
) -> list[dict]:
    """
    Format labeled periods as mlx_lm-compatible JSONL chat messages.

    Each example: {"messages": [system, user, assistant]}
    - system: agent system prompt from prompt_template_path
    - user: analysis request with real MCP tool outputs for (ticker, date)
    - assistant: JSON matching AgentSignal schema with label as decision

    When mcp_tool_outputs is None, uses synthetic indicator values from the
    label period's OHLCV data (direct numpy computation, no MCP subprocess).
    This avoids requiring LM Studio for training data generation.

    Returns list of dicts ready for json.dumps per line.
    """
```

**P11-E1-T2: scripts/generate_reference_strategies.py**

```
Usage: uv run python scripts/generate_reference_strategies.py [--data-dir DIR]
       [--tickers AAPL,JPM,...] [--horizon 60] [--output-dir data/reference_strategies]

Steps:
1. Load all 15-ticker OHLCV Parquets from data/market/
2. For each ticker: generate_max_return_labels() and generate_risk_adjusted_labels()
3. Save Parquets: data/reference_strategies/max_return/{ticker}_60d.parquet
                  data/reference_strategies/risk_adjusted/{ticker}_60d.parquet
4. Print summary: n_buy, n_sell, n_hold per strategy per ticker
5. Validate: assert n_total >= 400 (sufficient for fine-tuning)
```

Idempotent: skip tickers whose output Parquet already exists with correct row count.
Error handling: skip ticker on failure, log warning, continue.

**P11-E1-T3: check_env.py -- phase11-data check**

New check `check_phase11_data()`:
- Verifies `data/reference_strategies/max_return/` contains >= 12 Parquet files
- Verifies `data/reference_strategies/risk_adjusted/` contains >= 12 Parquet files
- Returns error with fix instructions if absent

**P11-E1-T4: Unit tests for training_data.py**

`tests/unit/test_training_data.py`:
- `test_max_return_labels_buy_case`: seeded OHLCV, known forward return > 0.02 → Buy
- `test_max_return_labels_insufficient_data`: fewer than horizon_days remaining → row excluded
- `test_risk_adjusted_labels_high_sharpe`: seeded returns with Sharpe > 0.8 → Buy
- `test_risk_adjusted_labels_low_sharpe`: seeded returns with Sharpe < 0.3 → Sell
- `test_labels_no_lookahead_leakage`: verify label computation does not use data before t
- `test_format_as_jsonl_structure`: output is list of dicts with "messages" key;
  each message has "role" and "content"; assistant content is valid JSON
- `test_format_as_jsonl_schema_compliance`: assistant JSON validates as AgentSignal
  (or subset thereof for the targeted agent)

---

## Epic P11-E2: Training Data Formatting and Structured Output Examples

**Scope:** Convert reference strategy Parquets into two JSONL training datasets
(one per agent). Add structured output compliance examples as secondary signal.

Training data volume target: >= 400 examples per agent (DJ-054).

### Tickets

**P11-E2-T1: scripts/generate_training_jsonl.py**

```
Usage: uv run python scripts/generate_training_jsonl.py \
       [--agent technical|fundamental] [--horizon 60] [--output-dir data/training/]

Steps for Technical Agent:
1. Load data/reference_strategies/max_return/ for all 15 tickers
2. For each labeled period (ticker, date, label):
   a. Compute technical indicators from OHLCV using hifi.engines.technical directly
      (no MCP subprocess -- uses the engine functions directly for speed)
   b. Format user message as technical agent prompt with indicator values inline
   c. Format assistant message as AgentSignal JSON with decision=label, confidence=0.7
3. Filter: exclude Hold examples beyond 2x the Buy+Sell count (class balance)
4. Shuffle with fixed seed (42) for reproducibility
5. Write data/training/technical_max_return_60d.jsonl (one JSON dict per line)
6. Print: n_examples, class distribution, avg tokens per example

Steps for Fundamental Agent:
1. Load data/reference_strategies/risk_adjusted/ for all 15 tickers
2. For each labeled period:
   a. Compute fundamental ratios from FundamentalsSnapshot (use Phase 1 data or
      synthetic values when historical fundamentals unavailable for that date)
   b. Format user message as fundamental agent prompt
   c. Format assistant message as FundamentalAnalysis JSON with decision=label
3. Same class balancing, shuffling, and output
4. Write data/training/fundamental_risk_adjusted_60d.jsonl

**P11-E2-T2: Structured output compliance examples**

A secondary JSONL set (50-100 examples) derived from verified Phase 3/5 baseline
runs where HR=0.000 and GR=1.000. These show the model what perfect structured
output looks like (correct JSON schema, correct MCP field names, no hallucinated
values). Purpose: reinforce format compliance without changing decision logic.

Script: `scripts/generate_compliance_examples.py`
Source: `tests/fixtures/baseline/phase3_baseline.json`,
        `tests/fixtures/baseline/phase5_verification.json`
Output: `data/training/technical_compliance.jsonl`,
        `data/training/fundamental_compliance.jsonl`

**P11-E2-T3: Training data validation**

`tests/unit/test_training_jsonl.py` (skipif training files absent):
- `test_technical_jsonl_min_examples`: n_examples >= 400
- `test_fundamental_jsonl_min_examples`: n_examples >= 400
- `test_technical_class_balance`: Buy/(Buy+Hold+Sell) in [0.25, 0.50]
- `test_all_examples_have_valid_assistant_json`: every assistant content parses as JSON
- `test_no_lookahead_contamination`: analysis_date in user message is before label date

---

## Epic P11-E3: LoRA Fine-Tuning Pipeline

**Scope:** Run LoRA training via mlx_lm.lora for both agents. Perform rank sweep
(4, 8, 16, 32) on Technical Agent to identify optimal rank (DJ-015). Use optimal
rank for Fundamental Agent. Store adapters in data/adapters/.

### Tickets

**P11-E3-T1: src/hifi/models/fine_tune.py**

```python
def run_lora_training(
    model_path: str,            # HuggingFace or local path to base model
    train_file: str,            # path to .jsonl training file
    output_dir: str,            # path to save adapters (data/adapters/{name}/)
    lora_rank: int = 8,
    lora_layers: int = 16,
    batch_size: int = 4,
    num_iters: int = 1000,
    learning_rate: float = 1e-5,
    venv_python: str = "venvs/finetune/bin/python",
) -> dict:
    """
    Invoke mlx_lm.lora training as a subprocess via the finetune venv.

    Builds the mlx_lm.lora command and runs it. Returns a dict with:
      - output_dir: path to saved adapters
      - train_loss: final training loss (parsed from stdout)
      - n_examples: number of training examples used
      - duration_seconds: wall-clock training time

    Raises RuntimeError if training fails or adapters are not produced.
    """

def check_adapter_quality(
    adapter_dir: str,
    venv_python: str = "venvs/finetune/bin/python",
) -> bool:
    """
    Verify that adapters in adapter_dir are loadable.
    Runs a minimal mlx_lm.generate call and checks the output is non-empty.
    Returns True if adapters are valid, False otherwise.
    """
```

**P11-E3-T2: scripts/run_phase11_finetune.py -- Main fine-tuning pipeline**

```
Usage: uv run python scripts/run_phase11_finetune.py \
       [--agent technical|fundamental|both] [--rank N] [--rank-sweep]

For --rank-sweep (Technical Agent only):
  Runs training at ranks 4, 8, 16, 32.
  For each rank:
    - Train on data/training/technical_max_return_60d.jsonl
    - Save adapters to data/adapters/technical_rank{N}/
    - Run check_adapter_quality()
  Produces data/training/rank_sweep_results.json:
    {rank: {train_loss, duration_seconds}} for each rank

For --agent technical (default: uses rank 8 or optimal from sweep if available):
  - Combine technical_max_return_60d.jsonl + technical_compliance.jsonl
  - Train with selected rank
  - Save to data/adapters/technical_v1/

For --agent fundamental:
  - Combine fundamental_risk_adjusted_60d.jsonl + fundamental_compliance.jsonl
  - Train with optimal rank from sweep (or 8 if sweep not run)
  - Save to data/adapters/fundamental_v1/

Progress output: live loss every 100 iterations, ETA, current rank
```

**P11-E3-T3: check_env.py -- phase11-adapters check**

New check `check_phase11_adapters()`:
- Verifies `data/adapters/technical_v1/` exists and contains adapter files
- Verifies `data/adapters/fundamental_v1/` exists
- Returns error list with fix instruction if absent

**P11-E3-T4: Unit tests for fine_tune.py**

`tests/unit/test_fine_tune.py`:
- `test_run_lora_training_missing_venv`: raises RuntimeError with useful message
  when venv_python does not exist (no finetune venv)
- `test_check_adapter_quality_missing_dir`: returns False when adapter_dir absent
- `test_run_lora_training_returns_expected_keys`: mock subprocess returns a known
  stdout; assert output dict contains output_dir, train_loss, duration_seconds
  (monkeypatching subprocess here is justified -- we are testing the Python
  parsing logic, not mlx_lm itself)

---

## Epic P11-E4: Three-Tier Evaluation

**Scope:** Run the three-tier evaluation protocol (DJ-058) comparing base model
vs fine-tuned model for both agents. This epic produces the quantitative results
that answer OQ-M01 and OQ-M02.

### Tickets

**P11-E4-T1: scripts/run_phase11_evaluation.py**

```
Usage: uv run python scripts/run_phase11_evaluation.py \
       [--ticker AAPL,JPM,...] [--analysis-date 2023-03-31] [--held-out-date DATE]

Requires: LM Studio running on 1234 (base model)
          mlx_lm.server running on 1235 (technical fine-tuned)
          mlx_lm.server running on 1236 (fundamental fine-tuned)

Steps:
1. check_env: verify all three servers are reachable
2. BASE MODEL RUN:
   - run_ensemble(ticker, date, agents=["fundamental", "technical"], use_rag=False)
   - run verify_ensemble() on the output
   - Extract: HR, GR per agent; pairwise_diversity, disagreement_entropy
3. FINE-TUNED MODEL RUN:
   - Temporarily set HIFI_LM_STUDIO_URL for fundamental_agent to port 1236
   - Temporarily set HIFI_LM_STUDIO_URL for technical_agent to port 1235
   - run_ensemble() again with same inputs
   - run verify_ensemble() on the output
4. DIVERSITY COMPARISON:
   - Compute pairwise_diversity for base and fine-tuned runs
   - Compute inter-agent vote correlation ρ directly from vote arrays
5. ACCURACY (if Phase 10 labeled data available for this date):
   - Compare base vs. fine-tuned method accuracy on labeled periods
6. Print and save results to data/evaluation/phase11_evaluation.json
```

**P11-E4-T2: Evaluation result schema**

New `src/hifi/models/training_data.py` addition (or separate file):

```python
class FineTuneEvaluationResult(BaseModel):
    ticker: str
    analysis_date: str
    base_technical_gr: float
    base_fundamental_gr: float
    finetuned_technical_gr: float
    finetuned_fundamental_gr: float
    base_pairwise_diversity: float
    finetuned_pairwise_diversity: float
    base_disagreement_entropy: float
    finetuned_disagreement_entropy: float
    diversity_preserved: bool  # True if finetuned diversity >= 0.9 * base diversity
    gr_improved_technical: bool  # True if improvement >= 0.05
    gr_improved_fundamental: bool
    generated_at: str
```

**P11-E4-T3: Rank sweep analysis**

Script: `scripts/analyze_rank_sweep.py`
Reads `data/training/rank_sweep_results.json`, prints table:

```
LoRA rank sweep results (Technical Agent)
rank    train_loss    duration_s    quality_ok
4       X.XXX         XXXX          True
8       X.XXX         XXXX          True
16      X.XXX         XXXX          True
32      X.XXX         XXXX          True

Recommended rank: N (lowest loss with quality_ok=True)
```

Saves recommendation to `data/training/optimal_rank.json`.

**P11-E4-T4: Unit tests for evaluation**

`tests/unit/test_evaluation_schema.py`:
- `test_finetune_evaluation_result_schema`: construct with known values, JSON roundtrip
- `test_diversity_preserved_computation`: diversity_preserved=True when finetuned >= 0.9 * base
- `test_gr_improved_computation`: gr_improved=True when delta >= 0.05

**P11-E4-T5: Holistic evaluation test**

`tests/holistic/test_phase11_evaluation.py`:
- `test_evaluation_pipeline_structure`: construct a base EnsembleOutput and a
  fine-tuned EnsembleOutput (different signals, same ticker/date), run the
  diversity comparison logic, assert FineTuneEvaluationResult is valid
- `test_three_tier_logic_wired`: verify that the three tiers produce consistent
  outputs (no LLM required -- test with pre-specified signal lists)

---

## Epic P11-E5: Incremental Weight Update Hook (DJ-060)

**Scope:** Implement the `label-outcomes` Makefile target so every `make test-live`
automatically labels any performance records where 60 trading days have elapsed.

### Tickets

**P11-E5-T1: scripts/run_label_outcomes.py**

```
Usage: uv run python scripts/run_label_outcomes.py [--data-dir DIR]
       [--horizon 60] [--dry-run]

Steps:
1. Load data/agent_performance_history.json
2. For each record where outcome_correct is None:
   - analysis_date + 60 trading days = label_date
   - If label_date <= today: compute forward_return, label record (DJ-042 rules)
3. Save updated history (atomic write via filelock, same pattern as performance_store.py)
4. Print: n_newly_labeled, n_still_unlabeled

No LM Studio required. Pure Parquet + pandas computation.
```

**P11-E5-T2: Unit tests for label_outcomes**

`tests/unit/test_label_outcomes.py`:
- `test_unlabeled_records_get_labeled`: synthetic record with analysis_date 70 trading
  days ago and Parquet data available → outcome_correct is set after labeling
- `test_future_records_stay_unlabeled`: analysis_date only 30 days ago → outcome_correct
  stays None
- `test_idempotent`: running labeler twice produces same result

---

## Epic P11-E6: Baseline Measurement + Bitacora

**Scope:** Generate the Phase 11 baseline fixture, run the full evaluation
pipeline, and write the scientific bitacora.

### Tickets

**P11-E6-T1: tests/fixtures/baseline/phase11_evaluation.json**

Generated by `scripts/run_phase11_evaluation.py`.
Content: FineTuneEvaluationResult for AAPL/JPM/XOM, 2023-03-31.
Fixture validation test: `tests/unit/test_phase11_baseline.py` (skipif absent).

**P11-E6-T2: Makefile target baseline-phase11**

```makefile
baseline-phase11: ## Phase 11 fine-tuning eval: generate + validate (requires LM Studio + finetune servers)
    uv run python scripts/check_env.py --check lm-studio
    uv run python scripts/check_env.py --check finetune-venv
    uv run python scripts/check_env.py --check phase11-adapters || $(MAKE) finetune-train
    $(MAKE) finetune-serve
    sleep 15
    uv run python scripts/run_phase11_evaluation.py
    $(MAKE) finetune-stop
    uv run pytest tests/unit/test_phase11_baseline.py \
                 tests/holistic/test_phase11_evaluation.py \
                 -q --tb=short
```

**P11-E6-T3: doc/bitacora/PHASE_11_FINE_TUNING.md**

Standard structure:
- Objective, architecture decisions (DJ-053 through DJ-060)
- Training data summary: n_examples, class distribution per agent
- Rank sweep results (Technical Agent): loss vs. rank table
- Evaluation results (Tier 1/2/3)
- OQ-M01 answer: minimum data quantity observed empirically
- OQ-M02 answer: diversity impact measured
- Implementation surprises
- Open questions for Phase 12

---

## Interface Design Summary

### New Source Files

```
src/hifi/models/__init__.py
src/hifi/models/training_data.py        — generate labels, format JSONL
src/hifi/models/fine_tune.py            — run_lora_training, check_adapter_quality
scripts/setup_finetune_venv.sh          — bootstrap venvs/finetune/
scripts/serve_finetune_models.sh        — start mlx_lm.server on 1235/1236
scripts/generate_reference_strategies.py — Dataset Family C Parquets
scripts/generate_training_jsonl.py      — JSONL per agent
scripts/generate_compliance_examples.py — structured output examples
scripts/run_phase11_finetune.py         — main fine-tuning pipeline
scripts/run_phase11_evaluation.py       — three-tier evaluation
scripts/run_label_outcomes.py           — incremental weight update
scripts/analyze_rank_sweep.py           — rank sweep analysis
```

### Modified Files

```
scripts/check_env.py    — finetune-venv, phase11-data, phase11-adapters checks
Makefile                — finetune-setup, finetune-train, finetune-serve, finetune-stop,
                          generate-reference-strategies, label-outcomes, baseline-phase11;
                          label-outcomes added as post-step in test-live
```

### New Test Files

```
tests/unit/test_training_data.py        — training_data.py unit tests
tests/unit/test_training_jsonl.py       — JSONL validation (skipif absent)
tests/unit/test_fine_tune.py            — fine_tune.py unit tests
tests/unit/test_evaluation_schema.py    — FineTuneEvaluationResult schema
tests/unit/test_label_outcomes.py       — incremental labeling tests
tests/unit/test_phase11_baseline.py     — fixture validation (skipif absent)
tests/holistic/test_phase11_evaluation.py — structural pipeline test
```

### New Data Artifacts

```
data/reference_strategies/max_return/{ticker}_60d.parquet
data/reference_strategies/risk_adjusted/{ticker}_60d.parquet
data/training/technical_max_return_60d.jsonl
data/training/fundamental_risk_adjusted_60d.jsonl
data/training/technical_compliance.jsonl
data/training/fundamental_compliance.jsonl
data/training/rank_sweep_results.json
data/training/optimal_rank.json
data/adapters/technical_v1/             — LoRA adapter weights
data/adapters/fundamental_v1/           — LoRA adapter weights
data/evaluation/phase11_evaluation.json
tests/fixtures/baseline/phase11_evaluation.json
```

---

## Execution Order and Wave Structure

```
Wave 1 (parallel -- no dependencies):
  P11-E0  Fine-tuning infrastructure (venvs, Makefile, check_env)
  P11-E5  label-outcomes script (standalone, no fine-tuning needed)

Wave 2 (after Wave 1):
  P11-E1  Dataset Family C generation (needs Makefile + check_env)
  P11-E2  Training data formatting (needs E1 Parquets)
  -- these run sequentially: E1 then E2

Wave 3 (after Wave 2):
  P11-E3  LoRA fine-tuning (needs E2 JSONL)
  -- rank sweep, then both agents

Wave 4 (after Wave 3):
  P11-E4  Three-tier evaluation (needs adapters + servers)
  P11-E6  Baseline + bitacora
```

---

## Verification Criteria

Phase 11 is complete when all hold:

1. **Tests:** `pytest -q --tb=no` passes with >= 1000 tests, 0 skipped, 0 lint errors

2. **Infrastructure:** `make finetune-setup` completes without error;
   `make finetune-serve` starts both servers and health checks pass

3. **Training data:** `data/training/technical_max_return_60d.jsonl` and
   `data/training/fundamental_risk_adjusted_60d.jsonl` each have >= 400 examples

4. **Adapters:** `data/adapters/technical_v1/` and `data/adapters/fundamental_v1/`
   both exist and pass `check_adapter_quality()`

5. **OQ-M01 answered:** `data/training/rank_sweep_results.json` documents
   training loss for all four ranks; `data/training/optimal_rank.json` specifies
   the recommended rank with justification

6. **OQ-M02 answered:** `data/evaluation/phase11_evaluation.json` contains
   `pairwise_diversity` for both base and fine-tuned runs; the bitacora
   explicitly states whether diversity was preserved, degraded, or improved

7. **Tier 1 met:** `finetuned_technical_gr >= 0.72` (improvement from 0.667 baseline)
   OR documented explanation of why improvement was not achieved

8. **Makefile:** `make test` passes; `make baseline-phase11` is runnable end-to-end;
   `make label-outcomes` can be invoked standalone without error

---

## Scientific Context

Phase 11 is the first phase that directly modifies agent behavior through learning
rather than prompt engineering or architectural expansion. This creates a new risk:
agents that individually improve may collectively regress if fine-tuning increases
their correlation.

The key observable is the pairwise diversity before vs. after fine-tuning. If
diversity is preserved while GR improves, the result is: heterogeneous reference
strategies are a valid approach to individual improvement without collective
regression. This is a non-obvious finding -- most multi-agent fine-tuning work
(if it exists in this domain at all) uses identical training data across agents.

If diversity degrades, the Phase 11 result is equally valuable: it establishes
an empirical bound on how much fine-tuning can be done before the ensemble
benefit erodes. In either case, the result is publishable.

The MUST-KNOW.txt interview framework (§9 Numbers, §12 Evaluation Frameworks)
requires real numbers. Phase 11 must produce: training dataset sizes, LoRA ranks
tested, training durations, before/after GR, before/after diversity metrics.
These are the specifics that distinguish builders from readers.

---

## Open Questions for Phase 12

1. **Structured debate.** Now that individual agents are fine-tuned, does
   structured debate add value on top of better individual agents, or does
   debate cause group polarization (Sunstein 2006) that degrades the collective?
   Phase 12 designs the debate mechanism and measures this interaction.

2. **Performance-weighted convergence.** After Phase 11 real LLM outputs are
   labeled, does performance_weighted become meaningfully different from
   confidence_weighted? The Phase 10 baseline showed all methods at 0.0.
   Phase 11's expanded labeled set (real fine-tuned agent outputs) should
   provide differentiation.

3. **Adapter versioning.** Phase 11 produces technical_v1 and fundamental_v1
   adapters. A versioning convention for adapter management (analogous to
   prompt versioning: v1.md, v2.md) should be established before Phase 12
   introduces structural debate.

4. **GraphRAG vs standard RAG.** Phase 7 implemented standard RAG. Phase 12
   implements GraphRAG. Phase 11's fine-tuned fundamental agent (trained
   without RAG context) provides the cleanest baseline for comparing whether
   GraphRAG retrieval adds value on top of domain fine-tuning.
