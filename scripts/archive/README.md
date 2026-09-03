# Archived phase scripts

Every script here produced a result that a phase was closed on. They are kept —
not deleted — because the bitácoras in `doc/bitacora/` and the phase plans in
`plans/` cite them by name as the provenance of published numbers, and a
reproducibility claim whose script no longer exists is not a reproducibility
claim.

They are **not** part of any running path. Nothing in `hifi.live`, the nightly
cycle, or the walk-forward sweep imports from this directory. That is the whole
point of the move (DJ-135): `scripts/` had grown to 60 files and 16,500 lines,
of which the two that actually ran every night were indistinguishable from the
28 that had run once, years of phases ago.

## Still runnable

Each script resolves the repository root through `parents[2]` rather than
`parent.parent`, adjusted when they moved one directory deeper. Invoke them
exactly as before, with `archive/` inserted:

```
uv run python scripts/archive/run_phase9_baseline.py     # was scripts/run_phase9_baseline.py
```

The Makefile targets that call them (`baseline-phase*`, `bootstrap*`,
`calibrate-drift`, `eval-*`, `finetune-train`) were updated and still work.

**Paths in `doc/bitacora/` and `plans/PHASE_0*`–`PHASE_1*` were deliberately
left pointing at the old locations.** Those documents are the dated record of
what was done and when; rewriting them to match today's layout would make them
say something that was not true at the time. This file is the redirection.

## What is here

| Script | Phase | Produced |
|---|---|---|
| `acquire_phase1_data.py` | 1 | Initial OHLCV + FRED macro parquets |
| `acquire_phase10_data.py` | 10 | Market data for the 12 added tickers |
| `acquire_macro_phase14.py` | 14 | FRED extension to 2004–2025 |
| `run_phase3_baseline.py` | 3 | Fundamental agent baseline fixture |
| `run_phase4_ensemble.py` | 4 | First ensemble baseline |
| `run_phase5_verification.py` | 5 | Verification-layer baseline |
| `run_phase6_tracing.py` | 6 | LangFuse trace seeding |
| `run_phase7_rag_baseline.py` | 7 | EDGAR RAG baseline |
| `run_phase8_baseline.py` | 8 | Agent-population ablation ladder (2→6 agents) |
| `run_phase9_bootstrap.py` | 9 | Performance history seed, 20 quarter-ends |
| `run_phase9_baseline.py` | 9 | Collective engine baseline |
| `run_phase10_bootstrap.py` | 10 | 15-ticker history seed (supersedes the Phase 9 one) |
| `run_phase10_baseline.py` | 10 | Accuracy labelling + tear sheets |
| `run_phase10_calibration.py` | 10 | Confidence calibration |
| `run_phase11_finetune.py` | 11 | LoRA training for both agents |
| `run_phase11_evaluation.py` | 11 | Fine-tune evaluation — the gate DJ-058 failed |
| `generate_compliance_examples.py` | 11 | Compliance training examples |
| `run_phase12_baseline.py` | 12 | GraphRAG + debate pilot |
| `run_phase12_evaluation.py` | 12 | 2×2 factorial: GraphRAG × debate |
| `run_phase12_graphrag_eval.py` | 12 | Precision@k, plain vs graph-expanded |
| `run_phase13_verification_baseline.py` | 13 | Risk/macro/sentiment verification |
| `run_phase13_debate_eval.py` | 13 | Multi-round debate → OQ-D04 |
| `run_phase13_memory_eval.py` | 13 | Agent memory influence → OQ-M03 |
| `run_phase13_scenarios.py` | 13 | Synthetic scenarios F-001/002/003 |
| `diagnose_sentiment_sgr.py` | 13 | DJ-086 SGR=0.000 root cause |
| `diag_sentiment_signals.py` | 13 | Sentiment signal/context inspection |
| `calibrate_drift_monitors.py` | 13 | Drift monitors on the 2022 rate shock |
| `run_phase14_e0_full.py` | 14 | Full E0 sweep |
| `run_phase14_model_diagnostic.py` | 14 | Per-model diagnostic |
| `analyze_rank_sweep.py` | 11 | LoRA rank sweep analysis |
| `genesis2_reset.sh` | 16 | `data/live/_genesis1_archive` and the Genesis II clear (2026-08-24). Superseded by `scripts/genesis_reset.sh` |

## What deliberately stayed in `scripts/`

Operational, or needed by work that is still open:

- `hifi_live.py`, `hifi_walkforward.py`, `nightly_live_execute.sh`,
  `watchdog_walkforward.sh` — the running experiment.
- `run_phase15_walkforward.py`, `run_phase15_smoke.py`, `compute_phase15_ic.py`
  — Phase 15 must be **re-run** on repaired data before any Page-theorem claim
  (the original result is retracted), so these are pending work, not history.
- `refresh_data.py`, `verify_agent_repair.py`, `simulate_next_run.py`,
  `check_env.py`, `genesis_reset.sh` — operations.
- `build_phase16_report_notebook.py`, `run_personality_shadow.py`,
  `label_outcomes.py`, `run_label_outcomes.py` — reporting and labelling that
  run against the live record.
- `ingest_edgar_mda.py`, `ingest_episodes.py`, `manage_namespaces.py`,
  `acquire_phase14_data.py`, `record_sec_fixtures.py`,
  `validate_sentiment_corpus.py`, `build_knowledge_graph.py` — data and index
  maintenance for the current universe.
- `generate_reference_strategies.py`, `generate_training_jsonl.py`,
  `setup_finetune_venv.sh`, `serve_finetune_models.sh` — fine-tuning
  infrastructure. The adapters are retired (DJ-124) but the pipeline is intact
  and the paper discusses the negative result.
