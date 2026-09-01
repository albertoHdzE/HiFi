# Paper I — Statistical Analysis Plan (SAP)

**Version:** 0.1 — DRAFT, NOT FROZEN
**Created:** 2026-09-01
**Governed by:** `02-charter.md`
**Freeze condition:** this document must reach v1.0 and be frozen, with its hash
recorded in the bitácora and its substance filed as OSF amendment 002, **before
the confirmatory re-run executes**. Analysis of data collected before the freeze
is exploratory by definition (charter §2.2).

> **Status of every number below.** Design parameters (cells, endpoints, tests)
> are decisions. Thresholds marked `⟨TBD-Dn⟩` are *deliverables* — they must be
> computed from the archived corpus and written in before the freeze. No
> threshold in this document may be chosen after seeing the data it will be
> applied to. Nothing here is a result.

---

## 1. Objective

Estimate whether **information partitioning** and **model heterogeneity** have
different effects on the effective decision independence of an LLM agent
collective, and whether either is large relative to the collective's own
reproducibility floor.

This operationalises H-PRIMARY (charter §1.2). Document 01 asked for
"independence" repeatedly and never defined it; §3 below defines it.

---

## 2. Design

### 2.1 Factorial structure

A 2 × 2 factorial over the two manipulations, plus a replicate of the reference
cell. Five cells:

| Cell | Models | Information | Role |
|---|---|---|---|
| **R** | heterogeneous | shared | Reference (current production configuration) |
| **M** | homogeneous | shared | Model-heterogeneity removed |
| **I** | heterogeneous | partitioned | Information sharing removed |
| **MI** | homogeneous | partitioned | Both removed |
| **R′** | heterogeneous | shared | **Replicate of R** — noise-floor estimator |

Topology (parallel vs sequential aggregation) is **not** a confirmatory factor.
It is nested as an exploratory arm (§7.3). Document 01's eleven-condition matrix
is withdrawn; five cells is what the available compute and the multiplicity
budget support.

### 2.2 Operational definition of each manipulation

**Model heterogeneity.**
- *Heterogeneous* — the production assignment: distinct families across agents
  (Llama 3.3 70B, Mistral Small 3.2 24B, DeepSeek-R1-distill-Qwen 32B, Gemma 3
  12B, Qwen3.5). Exact identifiers and quantisations recorded per run.
- *Homogeneous* — all five voting agents served by a single model
  (`qwen2.5-coder-32b-instruct-mlx`), prompts unchanged.

**Information sharing.**
- *Shared* — production behaviour, including the two known sharing channels: the
  GraphRAG context broadcast that injects identical filing text into both the
  fundamental and technical agents (`ensemble_runner.py:261,272`, identified as a
  diversity violation in the 2026-08-23 evaluation §4.2), and any prior-agent
  context read by downstream agents.
- *Partitioned* — strict single-channel isolation. Each agent sees exactly one
  evidence family and no other agent's output: fundamental → EDGAR filings;
  technical → OHLCV-derived indicators; risk → risk metrics; macro →
  macroeconomic series; sentiment → news. No broadcast context, no prior-agent
  summaries.

The partition is **verified, not assumed** (gate G6, §6). Phase 15's memory
ablation ablated a store that was empty for all 2,352 records (DJ-128); an
unverified manipulation is the failure mode this programme has already
committed once.

### 2.3 Randomisation and interleaving — mandatory

Phase 15 ran its four conditions in four disjoint wall-clock windows across
model-server reloads, making **condition perfectly confounded with time**
(DJ-128). That single design error produced a 43.2% disagreement between two
identical configurations.

Therefore, binding:

1. Conditions are **interleaved at the (date, ticker) decision level** in
   randomised order, not run block-wise.
2. The randomisation seed and the realised ordering are recorded per run.
3. `clear_run` is called between conditions. Duplicate `(run_id, agent_type)`
   rows must be zero (gate G5).
4. Serving-stack state (model server version, load order, any reload) is logged
   per decision, and included as a nuisance covariate in the sensitivity
   analysis (§8.2).

### 2.4 Sample

- Universe: the 98-ticker S&P subset, membership fixed and documented.
- Date grid: `⟨TBD-D1⟩` evaluation dates. Phase 15 used 24. **24 is very likely
  underpowered for the primary contrast** (see §5) because the effective sample
  size is the number of dates, not the number of decisions. The grid is extended
  to whatever D1 requires, or the study is declared underpowered in advance.
- Per cell: 98 tickers × D dates decision records, 5 voting agents each.

---

## 3. Endpoints

### 3.1 Per-agent signal

For agent *a* on decision *i*, map the agent's own decision and confidence to a
scalar, by the same encoding `metrics.py` already uses for the ensemble:

```
s(a,i) = +c(a,i)  if decision = Buy
         -c(a,i)  if decision = Sell
          0       if decision = Hold
```

with c ∈ [0,1] the agent's stated confidence. This preserves ordinal strength.
The encoding's known weakness — that stated LLM confidence is not calibrated —
is addressed by the rank-based secondary endpoint (§3.4) and disclosed.

### 3.2 PRIMARY ENDPOINT — effective number of independent agents

For each cell and each date *t*, form the A × A Pearson correlation matrix
**C**(t) of the agent signals across the 98 tickers on that date. Let λ₁…λ_A be
its eigenvalues. The primary endpoint is the participation ratio:

```
                 (Σ λ_k)²        (tr C)²         A²
    n_eff(t) =  ───────────  =  ─────────  =  ─────────
                  Σ λ_k²         tr(C²)        Σ λ_k²
```

with A = 5 voting agents (the contrarian is a reviewer and does not vote —
`voting_agents = [a for a in active if a != "contrarian"]`; an earlier internal
analysis mis-scored this and the manuscript states it once, correctly).

- n_eff = A ⟺ agents are mutually uncorrelated (full independence)
- n_eff → 1 ⟺ agents are perfectly correlated (one agent in five costumes)

**Why this endpoint.** It is the quantity Page's identity actually rewards, it is
label-free (§3.5), it is scale-free, it has a hard theoretical range that makes
effect sizes interpretable, and it is standard in complexity science and in
random-matrix treatments of correlated systems. The existing herding coefficient
— fraction of unanimous decisions — is a coarse binarisation of the same idea and
becomes a secondary endpoint.

**Primary estimand.** Averaging over dates within a cell, write n̄_eff(cell).
Define:

```
    Δ_info  = n̄_eff(I) − n̄_eff(R)      effect of removing information sharing
    Δ_model = n̄_eff(R) − n̄_eff(M)      effect of model heterogeneity
    θ       = Δ_info − Δ_model          THE PRIMARY ESTIMAND
```

**H-PRIMARY predicts θ > 0**: partitioning information buys more independence
than diversifying models.

### 3.3 Reproducibility floor

```
    δ₀ = | n̄_eff(R′) − n̄_eff(R) |
```

measured on the replicate cell. δ₀ and its upper confidence bound are reported
in the abstract. **No effect smaller than the upper bound of δ₀ is
interpretable** (charter §2.5). This is the single most important number in the
paper after θ.

### 3.4 Secondary endpoints (pre-specified, label-free)

| # | Endpoint | Definition |
|---|---|---|
| S1 | Herding κ | fraction of decisions with unanimous voting agents (`compute_herding_coefficient`) |
| S2 | Disagreement entropy | Shannon entropy of the {Buy, Hold, Sell} distribution across the 5 agents per decision, normalised by log 3 |
| S3 | Mean pairwise \|ρ\| | mean absolute off-diagonal element of **C**(t) |
| S4 | Rank-based n_eff | §3.2 recomputed on Spearman correlations, guarding against the confidence-calibration weakness of §3.1 |
| S5 | Confidence dispersion | cross-agent SD of stated confidence |

### 3.5 Secondary endpoints (label-dependent) — reported with a caveat

| # | Endpoint | Definition |
|---|---|---|
| L1 | Ensemble IC | Spearman rank correlation of ensemble buy-strength with 60-day forward return (`compute_ic`) |
| L2 | Individual-agent IC | as L1, per agent |
| L3 | Page decomposition | collective squared error, mean individual squared error, and the realised diversity term, on a common numeric target |

**L1–L3 depend on forward-return labels and therefore on the historical
evaluation protocol, which the 2026-08-23 evaluation found to violate
point-in-time accuracy (C1), survivorship control (C2), corporate-action
consistency (C3) and purged/embargoed validation (C8).** They are reported as
secondary, with those violations stated in the same table, and no confirmatory
claim rests on them. The primary endpoint is deliberately label-free precisely
so that the paper does not sit downstream of that repair programme (charter
§3.1).

L3 is reported as a *decomposition*, never as a test: the identity cannot fail
(charter §1.1).

### 3.6 Endpoints explicitly excluded from Paper I

Cumulative return, volatility, Sharpe, Sortino, maximum drawdown, Calmar,
turnover, transaction-cost-adjusted performance, exposure, concentration. These
belong to Paper II. Document 01 §7 listed seventeen performance measures with no
primary; that design cannot be defended against multiplicity.

---

## 4. Effective sample size

The Phase 15 p-values treated 2,352 (date, ticker) observations as independent.
They are not, in two distinct ways, and the correction is presented in the
manuscript as a methodological contribution rather than buried.

1. **Cross-sectional dependence.** Decisions on 98 tickers on the same date share
   the market factor, the macro context and the serving-stack state. Within a
   date, the effective number of independent observations is far below 98.
2. **Overlapping horizons.** 60-trading-day forward returns on a date grid finer
   than 60 days overlap, inducing serial dependence of order (horizon / spacing).

**Resolution, binding for all confirmatory inference:**

- The **unit of analysis is the date**. Per-date statistics are computed across
  tickers, then compared across cells date-by-date. Nominal n for the primary
  test is D, the number of dates — not D × 98.
- Cells are compared **paired by date**, which removes date-level common factors
  from the contrast.
- Uncertainty comes from a **stationary block bootstrap over dates**, block
  length `⟨TBD-D2⟩` chosen by an automatic rule fixed before the freeze, plus a
  **permutation test** that permutes cell labels within date.
- For any label-dependent secondary, the overlap correction is applied and the
  reported n is the number of non-overlapping horizons.

No result in this paper reports a p-value computed under an i.i.d. assumption.

---

## 5. Power — computed before the run, not after

Document 01 never asked whether the effect is detectable. It is the most likely
fatal objection and it is answered first.

**Deliverable D1 (blocking).** From the archived Phase 15 corpus, estimate the
date-to-date standard deviation of the paired n_eff difference, σ_d. Then the
minimum detectable θ for a paired two-sided test at α = 0.05 and power 0.80 is
approximately

```
    MDE ≈ (z_{0.975} + z_{0.80}) · σ_d / √D  ≈  2.80 · σ_d / √D
```

Report a table of MDE against D ∈ {24, 36, 48, 60, 90}. Then:

- Fix the **smallest scientifically meaningful θ**, θ_min, *before* seeing MDE.
  Recommended anchor: θ_min = 0.5 effective agents (a tenth of the A = 5 range,
  and half an agent is the smallest difference a reader would act on).
  `⟨TBD-D3⟩` — Alberto ratifies θ_min.
- Choose D such that MDE ≤ min(θ_min, upper bound of δ₀). **If no feasible D
  satisfies this, the study is declared underpowered in the SAP, before running**,
  and the paper is written as a protocol-and-mechanism contribution with the
  power limitation stated in the abstract.

An honest underpowered study, declared in advance, is publishable at a
complexity venue. An underpowered study discovered afterwards is not.

---

## 6. Negative-control gates — run first, halt on failure

These are gates, not robustness checks. They execute before the primary analysis
and a failure halts it.

| Gate | Control | Pass condition | Failure means |
|---|---|---|---|
| **G1** | Label shuffle — permute the ticker–label alignment | L1 IC indistinguishable from 0 | the label pipeline manufactures signal; all label-dependent endpoints void |
| **G2** | Replicate cell R′ | δ₀ upper bound < \|θ̂\| | the design cannot resolve its own effect; study void (charter §4) |
| **G3** | Evidence-free probe | agents abstain or emit low confidence when given empty evidence | the DJ-120 failure mode is live: confidence is uninformative, and S5/L1 are reported as such |
| **G4** | Data coverage | ≥ 99% of (agent, ticker, date) cells receive real evidence | a DJ-120 repeat; the cell is void, not corrected post hoc |
| **G5** | Store hygiene | zero duplicate `(run_id, agent_type)` rows; `clear_run` called per condition | a DJ-128 repeat (60% of homogeneous records were fed doubled context) |
| **G6** | **Manipulation check** | evidence-payload overlap (Jaccard on tool-payload hashes, per agent pair) ≈ 0 in partitioned cells and materially higher in shared cells; model identity verified per call | the manipulation did not manipulate — the Phase 15 memory-ablation error |
| **G7** | Random-agent null | replacing agents with random draws yields n_eff ≈ A and IC ≈ 0 | the metric is not measuring what we think |

G6 is the gate this project exists to have. Phase 15's headline contrast was
between two configurations that were argument-for-argument identical, and nobody
checked.

---

## 7. Analysis specification

### 7.1 Primary analysis

1. Compute n_eff(t) per cell per date.
2. Form the paired per-date quantities Δ_info(t), Δ_model(t), θ(t).
3. Point estimate θ̂ = mean over dates.
4. 95% CI by stationary block bootstrap over dates (§4).
5. p-value by permutation of cell labels within date, `⟨TBD-D4⟩` permutations
   (≥ 10,000).
6. **Decision rule, fixed now:** H-PRIMARY is *supported* iff the lower bound of
   the 95% CI for θ exceeds both 0 and the upper bound of δ₀. It is *rejected*
   iff the upper bound of the CI lies below 0. Otherwise the result is
   *inconclusive* and is reported as inconclusive.

There is exactly one primary test. It is not corrected for multiplicity because
there is nothing to correct.

### 7.2 Secondary analyses

S1–S5 and L1–L3, each by the same paired-by-date bootstrap. The secondary family
is controlled at FDR 0.05 by Benjamini–Yekutieli (valid under arbitrary
dependence, which is what we have). The corrected and uncorrected values are both
shown.

### 7.3 Exploratory analyses — labelled as such throughout

- Topology: parallel vs sequential aggregation, and its interaction with the two
  factors.
- The 2 × 2 interaction term itself (cell MI) beyond its role in θ.
- Per-agent contribution to n_eff (leave-one-agent-out).
- Regime conditioning.
- Synthetic-market mechanism isolation (§7.4).
- Aggregation-rule comparison, including the implemented-but-unselected
  contrarian-aware discount (`voting.py:272`), which would have applied a 0.66
  multiplier on the AAPL sample and has never been the active aggregator.

No exploratory result appears in the abstract, and none is described with
hypothesis-testing language.

### 7.4 Synthetic mechanism isolation (exploratory, high value)

Construct an environment with known signal, known noise, and a **tunable
correlation floor** injected between agents. Verify that n_eff recovers the
imposed floor and that θ behaves as the mechanism predicts. For a complexity
venue this is often worth more than the market result: it shows the instrument
measures what it claims, in a system where the truth is known by construction.

---

## 8. Robustness and sensitivity

### 8.1 Pre-specified robustness

Re-run the primary analysis under: rank correlations (S4) instead of Pearson;
decision-only signals (confidence discarded, s ∈ {−1,0,+1}); leave-one-date-out;
leave-one-ticker-decile-out; alternative block lengths.

The primary conclusion must survive all five, or the failures are reported in the
abstract.

### 8.2 Serving-stack sensitivity

Include model-server epoch as a nuisance covariate; report θ̂ with and without.
DJ-128 established this is not hypothetical.

### 8.3 Alternative explanations, answered by design

| Objection | Answered by |
|---|---|
| Ensemble wins only because there are more agents | cells M and MI hold agent count fixed |
| One agent dominates | leave-one-agent-out (§7.3) |
| Leakage | primary endpoint is label-free (charter §3.1) |
| Survivorship | primary endpoint is label-free; disclosed for L1–L3 |
| Shared information is the whole story | that *is* the manipulation |
| Model choice drove it | cells M and MI |
| Favourable regime | paired-by-date design; regime conditioning exploratory |
| Weak baseline | R is production, the strongest configuration we run |
| Test set used in development | pre-registration + freeze date on this SAP |

---

## 9. Data exclusions — specified before analysis

Excluded, with reasons stated in the manuscript:

1. Phase 15 `homogeneous` records with duplicated `(run_id, agent_type)` context
   (7,056 of 11,760 pairs, 60%) — unless regenerated under G5.
2. Any live-arm record before 2026-08-24 (Genesis II start).
3. Arm A's genesis window carrying the DJ-131 cash drag — bounded and stated.
4. Any cell failing G4 coverage.

No exclusion may be added after the analysis begins. Additions are logged in §11
as deviations and reclassify affected results as exploratory.

---

## 10. Provenance and reproducibility

Every run carries a `run_id` binding: config hash · dataset hash · model identity
and quantisation per agent · prompt hash · git commit · randomisation seed ·
evaluation protocol version · SAP version · serving-stack epoch.

The registry stays a flat file. Scientific provenance matters; tooling fashion
does not (document 01 §12, retained). Prompt *content* hashing is added — the
2026-08-23 evaluation found only a `prompt_version` string, which is not a hash.

Where local model weights cannot be redistributed, the limitation is stated and
the closest reproducible substitute is documented.

---

## 11. Deviation log

Every departure from this SAP after the freeze is recorded here, dated, with its
reason, and with the affected results reclassified.

| Date | SAP § | Deviation | Reason | Effect on claims |
|---|---|---|---|---|
| | | | | |

---

## 12. Pre-freeze checklist

- [ ] D1 — σ_d estimated from archived corpus; MDE × D table produced
- [ ] D2 — block-bootstrap length rule fixed
- [ ] D3 — θ_min ratified by Alberto
- [ ] D4 — permutation count fixed
- [ ] Partitioned-information configuration implemented and G6-verified
- [ ] Interleaved randomisation implemented (§2.3)
- [ ] Replicate cell R′ scheduled
- [ ] All seven gates implemented and dry-run
- [ ] Date grid D chosen, or underpowered status declared in writing
- [ ] SAP hash recorded in bitácora
- [ ] OSF amendment 002 filed — **blocks the run**
- [ ] v1.0 frozen — date:
