# Dataset Family G: Evaluation Baseline Fixtures

**Phase:** 13 E7-T2 audit (DJ-079)
**Last updated:** 2026-06-15

This directory contains deterministic evaluation baselines for HiFi Phase 3–13.
Each file is the authoritative reference for its phase's quantitative results.
Do not edit these files manually; they are produced by the corresponding
`scripts/run_phase{N}_*.py` script.

---

## Inventory

| File | Phase | Epic/Ticket | As-of Date | SHA-256 (16 chars) | Size (B) | Description |
|---|---|---|---|---|---|---|
| `phase3_baseline.json` | 3 | — | 2023-03-31 | `c4472fdb86f03f11` | 8,161 | Fundamental Agent baseline: single-model analyses for AAPL/JPM/XOM |
| `phase4_ensemble.json` | 4 | — | 2023-03-31 | `c0373ddf7864da1d` | 16,393 | Two-agent ensemble (Fundamental + Technical); majority + weighted voting |
| `phase5_verification.json` | 5 | — | 2023-03-31 | `31c880e17189e985` | 12,505 | Claim Verification: HR/GR baselines for Fundamental and Technical agents |
| `phase7_rag_baseline.json` | 7 | — | 2023-03-31 | `c830fd4cb19223ac` | 19,420 | RAG-augmented ensemble; Sentiment Agent added; LanceDB retrieval |
| `phase9_collective.json` | 9 | — | 2023-03-31 | `39c7395cdeaeb6cf` | 56,120 | Collective engine: 5-agent ensemble, voting methods, bootstrap history |
| `phase10_accuracy.json` | 10 | — | 2023-03-31 | `c72a42a2f83b633c` | 4,460 | 60-day forward accuracy labeling; majority/weighted/contrarian methods |
| `phase11_evaluation.json` | 11 | — | 2023-03-31 | `cb06e87e4f688aa6` | 1,864 | Fine-tuned Fundamental_v1 + Technical_v1 evaluation; HR/GR post-FT |
| `phase12_factorial_results.json` | 12 | E4-T1/T2 | 10 dates 2020–2022 | `e599e729ff9d8eab` | 2,611 | 2x2 factorial (120 runs): debate × FT across AAPL/JPM/XOM; herding/entropy |
| `phase12_graphrag_precision.json` | 12 | E2-T2 | — | `81d45d386d4d5a24` | 17,862 | GraphRAG Precision@5: plain RAG vs graph-expanded; OQ-K02 NEGATIVE |
| `phase13_verification_baseline.json` | 13 | E0-T6 | 2023-03-31 | `0297c30db108ba3f` | 11,746 | HR/GR for Risk+Macro; SGR for Sentiment (qwen2.5-coder + verbatim Rule 5, DJ-087) |
| `phase13_drift_calibration.json` | 13 | E5-T5 | 2020–2023 | `4a737ba75dc620b6` | 3,119 | Drift monitor calibration on 2022 rate-shock; all three monitors DETECTED |

---

## Key Results Summary

| Phase | Metric | Value | Notes |
|---|---|---|---|
| 5 | Fundamental HR | — | Established claim extraction baseline |
| 5 | Technical GR | — | Grounding rate for technical indicators |
| 12 | Disagreement entropy (cond. A) | 0.367 | No debate, no FT |
| 12 | Herding coefficient (cond. A) | 0.817 | High consensus without debate |
| 12 | GraphRAG Precision@5 delta | 0.000 | OQ-K02 NEGATIVE — plain RAG sufficient |
| 13 | Risk HR | 0.000 | No FRED data for 2023-03-31 evaluation date |
| 13 | Risk GR | 1.000 | All risk_metrics claims grounded |
| 13 | Macro HR/GR | 0.000 | No FRED claims for date (FRED data absent) |
| 13 | Sentiment SGR (DJ-087, qwen2.5 + Rule 5) | 0.667 | 4/6 signals grounded; JPM=1.0, XOM=1.0, AAPL=0 |
| 13 | Sentiment SGR (Gemma 4 E4B, no Rule 5) | 0.000 | Chat template failure + paraphrasing |
| 13 | KS drift p-value (2022 regime) | 0.000 | ALERT: vol+RSI distribution shifted |
| 13 | Chi-sq agent drift p-value | 0.000 | ALERT: momentum proxy decisions shifted |
| 13 | CUSUM herding C_k | 48.57 | ALERT: sustained herding proxy above 3σ |

---

## Production Notes

- **Phase 13 E0 baseline** (`phase13_verification_baseline.json`): generated with
  `qwen2.5-coder-32b` + DJ-087 prompt (Rule 5 verbatim-quoting). SGR=0.667 (4/6
  signals grounded). AAPL SGR=0.000 persists — retrieved context for AAPL contains
  low-information passages (boilerplate 8-K header text) that the model cannot quote
  as meaningful signals. Phase 14 should investigate AAPL retrieval quality.
- **FRED data absence**: Risk and Macro baselines on 2023-03-31 show HR=GR=0 because
  FRED API was not live during the evaluation run. This is expected and documented
  (alias_coverage=38.9% for Risk, n_claims=0 for Macro).
- **Phase 12 factorial** (`phase12_factorial_results.json`): covers 10 quarterly dates
  (2020-Q1 through 2022-Q2) × 3 tickers × 4 conditions = 120 runs. Fundamental+Technical
  only (Sentiment not included in factorial per DJ-080 rationale).
- **Drift calibration** (`phase13_drift_calibration.json`): CUSUM uses 2021-only as
  baseline (COVID-2020 excluded to avoid extreme volatility inflating the threshold).
  KS and chi-sq use 2020-2021 as baseline.

---

## Reproducing a Fixture

Each file is fully reproducible from the corresponding script:

```bash
# Phase 13 drift calibration (no LLM)
uv run python scripts/calibrate_drift_monitors.py

# Phase 13 verification baseline (requires LM Studio)
uv run python scripts/run_phase13_verification_baseline.py

# Phase 12 factorial (requires LM Studio, ~4h)
uv run python scripts/run_phase12_evaluation.py
```
