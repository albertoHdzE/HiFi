# Phase 15 Plan: Historical Walk-Forward Simulation

**Status:** COMPLETE — 2026-07-06
**Branch:** phase14/heterogeneous-ensemble
**Context:** plans/PHASE_15_CONTEXT.md (DJ-095, DJ-096, DJ-097)
**Bitacora:** doc/bitacora/PHASE_15_WALK_FORWARD_SIMULATION.md

---

## What Phase 15 Actually Was

Phase 15 had **no new code** to write. All execution infrastructure was delivered by
Phase 14.1 (scripts/run_phase15_orchestrator.py, scripts/compute_phase15_ic.py,
scripts/watchdog_walkforward.sh). Phase 15 was the **scientific experiment itself**:
running the 4-condition ablation, monitoring it for ~10 days, fixing two production
bugs discovered during the run, and computing IC results.

This is by design. The David (SS15) separates infrastructure (Phase 14.1) from
scientific execution (Phase 15). The experiment is the deliverable.

---

## Execution Plan (as run)

### Step 1: Launch condition=full
- Command: `nohup uv run python scripts/run_phase15_orchestrator.py --agent all --aggregate --pipeline --condition full --period held-out-test --quiet >> /tmp/walkforward_full.log 2>&1 &`
- Watchdog cron: every 30 min, auto-restarts orchestrator on death
- Duration: ~2 days (2026-06-24 → ~2026-06-26)

### Step 2: Launch condition=no-memory
- Same command, `--condition no-memory`
- Duration: ~2 days

### Step 3: Launch condition=parallel
- Same command, `--condition parallel`
- Duration: ~2 days
- **Bug fixed mid-run:** unload_all() added to model_manager.py (commit 27d6639)

### Step 4: Launch condition=homogeneous
- First run failed: 3/6 agents had wrong model keys in _HOMOGENEOUS_AGENT_CONFIG
  - gemma-3-4b-it → google/gemma-3-4b
  - mlx-community-qwen3-235b-a22b → mlx-qwen3.5-35b-a3b (x2)
- Fix: commit ec9a529
- Clean rerun required (sequential mode, causal history integrity)
- Delete all homogeneous data: `data/runs/homogeneous-*/` + `data/walkforward/homogeneous/`
- Rerun: 2026-07-03 → 2026-07-06 (~3.5 days)

### Step 5: IC computation
- Command: `uv run python scripts/compute_phase15_ic.py --period held-out-test --quiet`
- Runs automatically after all 4 conditions complete (watchdog triggers it)

---

## Results

See doc/bitacora/PHASE_15_WALK_FORWARD_SIMULATION.md for full analysis.

```
Condition     IC       p-value    IR       Herding
parallel     +0.0642   0.0019    +0.316    0.000   ** p < 0.01
full         +0.0232   0.2603    +0.567    0.361
no-memory    +0.0251   0.2236    +0.262    0.220
homogeneous  -0.0428   0.0380    nan       0.862   *  p < 0.05 (negative)
```

---

## Verification Criteria (all met)

- [x] All 4 conditions complete: 2352 sidecars per agent per condition
- [x] All 4 conditions: 24 portfolio snapshots
- [x] IC computed for all 4 conditions
- [x] Primary finding documented in bitacora
- [x] STATUS.md updated
- [x] Replication notebook updated

---

## Phase 15 → Phase 16 Handoff

Per DJ-095/DJ-097:

1. IC/IR results across 4 conditions: DONE (see bitacora)
2. Regime-conditional analysis: pending (compute_phase15_ic.py regime breakdown)
3. OQ-P14-03 answer (sequential vs parallel): parallel wins; herding explains gap
4. Dataset Family G primary artifact: 56,448 EnsembleOutput JSONs in data/walkforward/
5. Open question for Phase 16: does live performance match walk-forward parallel IC?
