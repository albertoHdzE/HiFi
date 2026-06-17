# Phase 18: Publication + Open Source Release
## Context and Pre-Phase Decisions

**Gathered:** 2026-06-16
**Status:** NOT STARTED — post-graduation; awaits Phase 17 ablation results
**Depends on:** Phase 17 (ablation results for publication evidence)

---

## Why This Phase Is Post-Graduation

Phase 17 (capstone) is the graduation requirement. Phase 18 extracts the scientific
contribution from the finished system and makes it available to the research community.
This is the "long tail" of the project — the work that converts an MScFE capstone
into a published contribution to complexity science and computational finance.

The data is already being collected from Phase 14 onward. By Phase 18, HiFi has:
- 21 years of walk-forward simulation results (Phase 15)
- 12-16 weeks of live paper trading (Phase 16)
- Ablation results isolating each component (Phase 17)
- 7 populated dataset families (Phases 1-16)
- Complete decision journal (DJ-001 through DJ-10X)

Publication requires analysis and writing, not new engineering.

---

## DJ-103: Primary Publication Strategy

**Central claim (falsifiable, empirically grounded):**
Architectural diversity in LLM agent ensembles — agents from different organizations
with distinct pre-training distributions — produces measurably higher Information
Coefficient than homogeneous ensembles, stable across market regimes. This is an
empirical test of Page's (2007) diversity theorem applied to LLM populations.

**Evidence base:**
- 100 stocks × 21 years × 5 market regimes
- Homogeneous ablation (Phase 17): direct IC comparison
- Regime-conditional analysis: diversity effect holds in bull AND bear AND rate-shock

**Primary target:** ICAIF (ACM International Conference on AI in Finance)
Annual; typically October submission for February conference.
Fits our contribution (empirical, financial, AI agents).

**Secondary target:** ACM Collective Intelligence Conference
Directly relevant to the Page diversity theorem framing.

**Tertiary (methods paper):** FinNLP workshop at EMNLP
For the HR/GR/SGR verification framework — narrower but defensible standalone contribution.

---

## DJ-104: Open Source Release Strategy

**License:** Apache 2.0 — permissive, commercial use allowed, maximizes reach in
the ML/finance community. Not GPL (too restrictive for commercial derivative work).

**What gets released:**
- Full source code (src/, scripts/, tests/, notebooks/)
- fundamental_v1 and technical_v2 LoRA adapters (with model cards)
- Dataset Families A (OHLCV), B (SEC MD&A chunks), D (debate transcripts),
  E (ensemble interaction records), F (historical scenarios), G (evaluation baselines)
- Docker stack for reproducible deployment
- Complete decision journal (DJ-001 through final) as supplementary material

**What does NOT get released:**
- IBKR credentials (never in repo, per DJ-099)
- Raw LM Studio model weights (user-provided; covered by each model's license)
- Any personally identifying data

**Dataset cards (Hugging Face):**
Each dataset family gets a dataset card documenting: schema, provenance, period,
tickers covered, intended use, limitations.
Family E (agent interactions) and Family F (scenarios) are novel contributions —
no comparable public datasets exist for LLM ensemble financial decision records.

---

## DJ-105: Containerization Design

**What is containerized:** All deterministic infrastructure.
- MCP servers (hifi-financial-calculator, hifi-portfolio-composer, hifi-risk-manager,
  hifi-capital-allocator): each as a Docker container
- LanceDB: Docker volume with data directory mounted
- LangFuse + ClickHouse: existing docker-compose from Phase 6

**What is NOT containerized:** LM Studio (model weights are user-provided, ~50-100 GB;
cannot bundle in Docker image). Documentation provides the LM Studio setup steps.

`docker compose up` starts all MCP servers, LanceDB, and LangFuse. Users then
configure LM Studio separately and run `make diagnose-models` to verify.

---

## Open Questions

| ID | Question | Resolution |
|---|---|---|
| OQ-P18-01 | Which venue accepts the primary paper? | Phase 18 submission process |
| OQ-P18-02 | Is Family E (agent interaction dataset) novel enough for standalone dataset paper? | Phase 18 evaluation |
| OQ-P18-03 | Should HiFi support GPU-based inference (not just Apple Silicon)? | Phase 18 scope decision |
