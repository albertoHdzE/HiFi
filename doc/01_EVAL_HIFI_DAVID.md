# HiFi: Evaluation of Distance to the David

## Gap Analysis — `doc/HIFI_DAVID.md` (the Ideal) vs. the Implemented System

**Status:** Evaluation snapshot — 2026-08-23
**Scope:** Full repository (~38K LOC across `src/hifi/` and `scripts/`), four parallel subsystem reviews plus targeted personal verification of every load-bearing claim.
**Method:** Adversarial read. Negative claims ("X does not exist") verified by grep/read on 2026-08-23. Line references are valid against commit `4be5daa`. See Appendix A for the evidence chain.
**Relation to the David:** This document practices what HIFI_DAVID.md §1.1 preaches: simplifications and deviations recorded *explicitly, against* the reference — except here it is the implementation that is measured against the reference.

---

## Table of Contents

1. [Executive verdict](#1-executive-verdict)
2. [Section-by-section scorecard](#2-section-by-section-scorecard)
3. [Overall distance estimate](#3-overall-distance-estimate)
4. [Direct contradictions (C1–C8)](#4-where-the-implementation-directly-contradicts-the-david)
5. [Missing elements (silent absences)](#5-what-is-simply-missing-silent-absences)
6. [Deficiencies in the David itself](#6-where-the-david-itself-is-deficient)
7. [What is better than the David](#7-what-is-better-than-the-david)
8. [Remediation plan](#8-remediation-plan-to-close-the-gap)
9. [Enhancements and next directions](#9-enhancements-and-next-directions)
10. [Definition of Done against §19](#10-definition-of-done-against-david-19)
11. [Appendix A — Evidence chain](#appendix-a--evidence-chain)
12. [Appendix B — References](#appendix-b--references)

---

## 1. Executive verdict

The project has built **roughly 70% of the David's architecture**. The skeleton is recognizably the Platonic system: deterministic engines behind MCP servers, six specialized local agents, four aggregation methods, RAG + GraphRAG on LanceDB, LoRA fine-tuning infrastructure, Langfuse observability, drift monitors, walk-forward simulation, and paper execution through Alpaca. Several parts *exceed* the spec (Section 7).

But the David's **epistemology has fared worse than its architecture**. The document's three "Critical Requirements" for market observation datasets (§8.2 — survivorship control, point-in-time accuracy, consistent corporate-action adjustment) are **all three violated in production code**, and the evaluation framework's central promise (§15.2 — purged cross-validation with embargo, no train/test overlap) **does not exist anywhere in the codebase**. The system that was built can do everything the David described; the *numbers it produces about itself* cannot yet be trusted the way the David demands.

The deepest pattern: where the implementation contradicts the David, it is usually not because the David was silent — it is because the David was **explicit** and was nevertheless overridden by convenience (Section 4, especially C4 and C6).

A second systemic finding, independent of the David: **the live path fails open**. Degraded states silently resolve toward tradeable outputs (empty vote → Hold, missing ensemble → truncated universe, breaker exception → trade), and one headline safety control is provably dead code. On paper mode this costs research validity; pointed at real money it would cost capital.

---

## 2. Section-by-section scorecard

Legend: ✅ faithful · ⚠️ partial/divergent · ❌ absent or violated

| David § | Component | Status | Evidence |
|---|---|---|---|
| §2 | Gap thesis (local/open/auditable/reproducible platform) | ✅ Built | Entire repo; all-local inference verified end-to-end |
| §4.1 | Deterministic-first | ✅ Faithful, ⚠️ blemishes | Engines/MCP clean and pure; two defects: indicators server runs raw `close` while engine prefers adjusted (`indicators_server.py:184` vs `technical.py:102`) — same concept, two sources of truth; NaN inputs crash MCP servers instead of returning the documented error dict (`capital_allocator.py:154–203`) |
| §4.2 | Diversity over scale | ⚠️ Nominal yes, measured no | Three model families among production agents (Llama/Mistral/Gemma); but shared `graphrag_ctx` injects identical filing text into fundamental AND technical agents (`ensemble_runner.py:261,272`), violating information-access diversity at the context level |
| §4.3 | Verifiability | ❌ Weakened in practice | Regex extractor recognizes only `"<alias> of <number>"`; a rationale with zero parseable claims scores Hallucination Rate = 0.0 — a *perfect* score (`verification/schemas.py:219–220`). The David's target HR < 0.02 is currently **unmeasurable**, not achieved |
| §4.4 | Observability | ✅ Strongest area | Tracing + decision audit + tearsheets + drift monitors + AI-ops report panel. Gaps: traces lost on crash (no `atexit` flush), memory/cost-per-decision metrics absent, payloads unredacted |
| §4.5 | Reproducibility | ⚠️ Half | Dataset SHA-256 registry ✅, seed config ✅, temperature 0.0 ✅. But registry overwrites prior versions (`versioning.py:158–171`), model weight checksums ✗, prompt content hashing ✗ (only a `prompt_version` string in metadata), experiment registry ✗ |
| §4.6 / §6.2 | Modularity / MCP-as-nervous-system | ✅ Exemplary | venv-isolated MCP subprocesses (`venvs/ta/`) implement DJ-010 exactly as specified — genuinely rare engineering maturity |
| §4.7 | Local-first | ✅ Holds | LM Studio serving, local embeddings, self-hosted Langfuse, paper Alpaca |
| §4.8 | Open research | ❌ Not started | Nothing released publicly; no dataset cards; no external deployability |
| §7.1–7.2 | Data acquisition / engineering | ⚠️ | yfinance/FRED/EDGAR-MD&A/Alpaca ✅; transcripts ✗, news ✗; quality gates decorative (zero production callers — only tests invoke them); three incompatible price-basis semantics coexist (see C3) |
| §7.3 | Knowledge layer | ✅ Built, ❌ temporally unguarded | LanceDB + nomic-embed + GraphRAG + episodic store all exist and were evaluated (Phase 12); retrieval lacks as_of enforcement (see C1) |
| §7.9 | Execution layer | ⚠️ Partial, safety unwired | Order generation ✅, circuit breakers present but partly dead; **stop-losses never implemented** (grep: zero hits); execution logging partial (no fill reconciliation); **§7.9's safety table exists verbatim as `SafetyConfig`… and is referenced by nothing** |
| §7.10 | Experiment registry | ❌ Never built | No MLflow or equivalent; results live in episode JSONLs and notebooks; the open question stayed open and became a permanent hole |
| §8.2 | Family A critical requirements | ❌ All three violated | Survivorship, point-in-time, corporate actions — see C1–C3 |
| §8.3 | Feature families | ✅ ~80% | Fundamentals subset, technical (custom six + pandas-ta server), risk, macro, embeddings ✅; Piotroski/Altman/DuPont deferred per DJ-012 (acknowledged deferral, legitimate); VWAP/volume-profile/support-resistance ✗ |
| §8.4 | Reference strategy datasets | ⚠️ Single-horizon | Forward-return labeling exists but only 60 trading days; David specifies horizons {5, 10, 20, 60, 120}; per-dataset documentation requirements (objective function, thresholds, costs, known biases) not materialized |
| §8.5–8.6 | Explanation + interaction datasets | ✅ Good | Episodes, agent sidecars, debate transcripts, votes/confidences all recorded — the complexity-science data byproduct exists as envisioned |
| §8.7 | Synthetic scenarios (Family F) | ✅ Built early | `data/scenarios/F-001..003`, `collective/scenarios.py`, Phase 13 harness — ahead of the implied schedule |
| §8.8 | Immutable evaluation sets (Family G) | ❌ Contract broken | No frozen EVAL-* artifacts; fine-tune training set covers 2004–2025 including every evaluation window (`scripts/generate_training_jsonl.py:95–116`) |
| §8.9 | Public release + datasheets | ❌ | Nothing released; Gebru et al.-style dataset cards absent |
| §9 | Model diversity + fine-tuning gate | ⚠️ Infra ✅, gate ❌ | mlx_lm LoRA pipeline ✅; "deployed only if demonstrably outperforms base on a held-out set" violated twice: training contamination (above) and DJ-124 (a rejected adapter served ~2 months) |
| §10.1 | Independence during analysis phase | ❌ Violated | Shared RAG context (above) plus debate challenge prompt receiving majority-size counts (`debate_nodes.py:177–179`) inject correlation at exactly the stage where the David forbade it |
| §10.2 | Agent roster | ⚠️ 6 of 7 | Fundamental/Technical/Risk/Macro/Sentiment/Contrarian ✅; **Valuation Agent never built**; contrarian correctly non-voting (matches the Design Note; DJ-128 confirms "reviewer") |
| §10.3 | Diversity measurement mandate | ⚠️ | Metrics implemented (`collective/metrics.py`) but measurements contaminated by shared-context leak; OQ-AG04 not honestly answerable yet |
| §10.4 | Agent memory | ⚠️ Exists, unsafe | Per-ticker/per-agent files ✅; decay function never designed (OQ-AG02 open), no as-of filter (leaks when backtested), unbounded growth, non-atomic writes |
| §11 | Knowledge systems | ✅ Built | GraphRAG-vs-RAG evaluated empirically (Phase 12); temporal discipline absent (C1) |
| §12.2 | Aggregation methods | ✅ 4 of 5 | Majority/confidence-weighted/performance-weighted/debate ✅; adaptive aggregation ✗ (marked Advanced — legitimate deferral) |
| §12.2.2 OQ | LLM confidence calibration | ⚠️ Measured, unused | Calibration reporting exists (`collective/labeler.py:361–392`); nothing feeds calibrated values back into voting — the loop sketched in the David is half-closed |
| §13 | Verification architecture | ⚠️ Shape ✅, substance weak | Extract→classify→verify-against-MCP pipeline exists incl. cross-agent contradiction detector; but extraction-by-regex was the option the David explicitly did *not* prefer (§13.6), and it now rubber-stamps (C4) |
| §14 | Observability stack + drift | ✅ Best-in-class for scope | KS / χ² / CUSUM implemented across data, agent, and collective layers, with a calibration script; statistical hygiene issues noted (family-wise error uncorrected; CUSUM alerts on final value instead of running max) |
| §15.2 | Evaluation protocol | ❌ Central failure | Walk-forward harness ✅ exists; **purge/embargo machinery: zero occurrences in the codebase (grep-verified)**; bootstrap CIs partial; equal-weight control arm ✅ and external riskbudget arm ✅ actually richer than spec; random/momentum/GPT-4 baselines ✗ |
| §15.7 | Ablation studies | ⚠️ Attempted, buggy | Memory ablation ran and "ablated nothing" (DJ-128); agent-removal and homogenize-model ablations never run |
| §16.1 | Containerization | ❌ Only Langfuse | No hifi-data/models/agents containers; deployment-from-docs-alone fails today (manual LM Studio + venv setup); GENESIS_CHECKLIST is an operational patch acknowledging this |
| §17 | Decision journal | ✅✅ Exceeded | DJ-series alive through DJ-128 with post-hoc audits — the educative-journal principle realized more rigorously than the David asked |
| §19 | Success criteria | Mixed | Capstone ✅ likely met; Engineering: traceability ✅, containerization ✗, HR target unverifiable; Scientific: H1/H2/H4 compromised by leakage channels; Community ✗ |

---

## 3. Overall distance estimate

Two different distances must be reported, because they diverge:

| Dimension | Distance | Reading |
|---|---|---|
| Architectural surface area | **~70% built**, ~15% diverged by design, ~15% missing | The machine resembles the David |
| Scientific trustworthiness of produced results | **~40%** | Every headline empirical claim currently routes through at least one leakage channel, a survivorship filter, or an uncalibrated instrument |

The machine resembles the David; the *evidence* the machine produces does not yet deserve the name High-Fidelity. Closing the first distance without closing the second would produce a beautiful instrument that lies fluently.

---

## 4. Where the implementation directly CONTRADICTS the David

These are not gaps — they are refutations of explicit normative text. Ranked by damage to the David's core claim.

### C1. Point-in-Time accuracy — violated five independent ways
> David §8.2: *"Point-in-time accuracy: Use data as it was available at each historical date, not as it was later revised."* (Also Glossary: "Point-in-Time".)

Reality:
1. Vector retrieval has **no date parameter at all** (`mcp/knowledge_server.py:108–113`; `knowledge/vector_store.py:168–176`) — yet it is the sentiment/fundamental agents' *primary* context path.
2. EDGAR filtering uses fiscal period end, not filing date (`knowledge/edgar_retriever.py:114`); `filed_date` exists at ingestion and is dropped from chunks (`knowledge/schemas.py:32–44`). A 10-Q for a period ending Sept 24 becomes "known" on Oct 1 although filed weeks later.
3. Episodic memory injects outcome labels computed after as_of (`knowledge/episodic_retriever.py:76,113–114`): an episode from as_of−30d passes the `decision_date < as_of` filter while its "+8% (60d)" label encodes prices up to as_of+30d.
4. Macro series forward-filled by reference period, ignoring FRED publication lag (`data/macro.py:206–265`): GDP stamped at quarter start, published ~a month after quarter end. (Remedy exists upstream: FRED's ALFRED vintage database provides as-of-published series.)
5. Performance weights loaded from disk cover full-sample future outcomes (`collective/performance_store.py:106–113`); the bootstrap labels entire history up front — structurally look-ahead whenever performance-weighted voting is evaluated backward.

### C2. Survivorship bias control — violated at both ends
> David §8.2: *"Survivorship bias control: Include delisted companies."* Coverage ambition: *"S&P 500 constituents (including historical members), 1995–present."*

Reality: universe curated in 2026 applied backward to 2004 with no membership history (`data/universe.py:4–16`); the labeler then returns `None` for any series ending before horizon completion — precisely the delistings (`collective/labeler.py:141–144`). The −100% outcomes that would punish bad Buys cannot exist in the evaluation set. Delivery: ~97 hand-picked survivors, 2004–2025.

### C3. Corporate-action consistency — three price bases, one column namespace
> David §8.2: *"Corporate action adjustment: Splits, dividends, mergers must be handled consistently."*

Reality: canonical store built with `auto_adjust=True` (`acquire_phase14_data.py`), nightly Alpaca appends split-adjusted-only raw bars onto it (`execution/market_data.py:64–148`), fixtures use `auto_adjust=False` with true adjusted_close (`data/market.py:41–48`), and ATR straddles two bases within one formula (`engines/technical.py:223`). Vendor seams become phantom returns feeding labels, signals, and covariance. After any split, the live store fabricates a price cliff.

### C4. Verifiability — the David's own preferred solution rejected
> David §13.6: structured output templates are "**strongly preferred because it eliminates the extraction problem entirely**."

Implementation chose regex extraction anyway — the disfavored path — and now suffers exactly the predicted failure: phrasing outside the alias grammar yields zero extractable claims and a perfect hallucination score (HR := 0.0 when nothing is resolvable, `verification/schemas.py:219–220`). Combined with the extractor's narrow grammar (`verification/extractor.py:172–175`), measured HR/GR trends can *improve* while fabrication worsens — the inverse of the metric's purpose. The spec predicted this bug and was overruled by convenience.

### C5. Fine-tuning deployment gate — violated in both possible directions
> David §9.4: *"A fine-tuned model is only deployed if it demonstrably outperforms the base model on a held-out evaluation set."*

Violated twice: (a) the held-out set isn't held out — training JSONL spans every eval window (C1-family contamination); (b) a model that failed quality judgment was served regardless (DJ-124: rejected adapter emitting constant Buy @ 0.70 across 98 tickers for months; serving script checks directory existence only). The rank-sweep artifact compounds this: `train_loss` is hard-coded `None` (`models/fine_tune.py:156`), selection always falls back to default rank, yet `optimal_rank.json` records justification "Lowest train_loss…" — provenance that never happened.

### C6. The safety table transcribed into code — and never wired
David §7.9 specifies: max position 5%, max daily loss 2%, max sector 25%, confidence override 0.95, min agents 4 of voting pool. Reality: `config/loader.py:41–49` defines `SafetyConfig` matching those values **value-for-value** — and it is referenced by nothing outside the loader. Meanwhile the live pipeline hardcodes separate constants, the quorum is enforced nowhere, stop-losses (also §7.9) were never generated, and the drawdown circuit breaker receives `hwm_value = portfolio_value` (`run_phase16_live.py:371`), making `(hwm−pv)/hwm > limit` identically zero — an unfireable control. This is ritual compliance: the table exists so the spec reads as satisfied, disconnected from machinery.

### C7. Independence condition undermined at the source
> David §10.1(4): agents have "**NO access to other agents' outputs during independent reasoning phase**"; §5.2 quotes Surowiecki's independence condition.

Reality: shared GraphRAG context flows into two supposedly independent agents (`ensemble_runner.py:245,261,272`); the debate challenge prompt embeds majority size, pressuring conformity (`debate_nodes.py:177–179`). Condition 2 of collective intelligence is structurally violated, then herding coefficient κ is measured as if it held.

### C8. Evaluation protocol — the primary defense against self-deception never constructed
> David §15.2: *"Walk-forward validation with purged cross-validation (López de Prado, 2018) … Embargo period between train and test to prevent leakage."*

No purge or embargo machinery exists (grep-verified across src/ and scripts/). Combined with C5's contamination and C1's label leakage, overlapping 60-day label horizons silently correlate adjacent folds. The one technique the David cited by name and book was skipped.

---

## 5. What is simply MISSING (silent absences)

| Missing element | David ref | Consequence |
|---|---|---|
| Valuation Agent (7th role) | §10.2 | Valuation dimension folded into fundamentals; ensemble smaller and less diverse than designed |
| Stop-losses in order generation | §7.9 | Downside exits rely entirely on the nightly rebalance cycle |
| Experiment registry (MLflow/custom) | §7.10 | No queryable provenance chain result → config hash → data hash → model hash |
| Earnings call transcripts & news ingestion | §7.1, §8.2 | Sentiment agent runs on SEC MD&A proxy — the sentiment role sees the least sentiment-like data in the system |
| Multi-horizon reference labels {5,10,20,60,120} | §8.4 | Horizon-risk questions unanswerable |
| Frozen immutable EVAL datasets | §8.8 | Benchmark drift possible; cross-month comparisons unofficial |
| Purge/embargo module | §15.2 | Fold contamination (see C8) |
| Random & momentum baselines; cloud-LLM comparison | §15.6 | Ensemble wins have weaker nulls than demanded (equal-weight and riskbudget arms partially compensate) |
| Containerized deployability | §16, §19.4 | Community success criterion unreachable today |
| Dataset cards & public release | §8.9, §19.4 | Open-research principle entirely prospective |
| Fill reconciliation (order terminal states) | §7.9 "every order, fill, rejection…recorded" | Rejections/partial fills invisible; broker/book drift absorbed silently |
| Turnover metric | §15.2 | Listed in the David's own metric table; computed nowhere (only a docstring mentions it, `execution/portfolio_recorder.py:10`) |
| Idempotent order submission | *(absent from David too)* | Crash-rerun duplicate-order window — see §6.1 |

---

## 6. Where the DAVID ITSELF is deficient

An honest gap analysis must also indict the spec. The two areas where reality failed hardest are precisely where the David said least:

1. **Execution lifecycle safety is unspecified.** No word on idempotency, client order IDs, crash-mid-submission recovery, partial fills, or reconciliation. The worst operational finding (duplicate-order window after the documented 2026-08-17 arm-A crash — see `run_phase16_live.py:1066` comment) lives entirely in this silence. The David designed a brain and sketched a spinal cord; it never considered reflexes. (Alpaca's API has supported deterministic `client_order_id` deduplication throughout — the omission cost nothing to prevent.)
2. **Retrieval temporality is implied, never mandated.** Point-in-Time is declared for *data*, but §11 (Knowledge Systems) never states that retrieval must be as-of bounded. Implementation exploited the ambiguity. One normative sentence — "retrieval SHALL filter documents by availability date ≤ as_of" — would have prevented half of C1.
3. **Fail-open vs fail-closed is never decided.** §5.1 prizes graceful degradation but never distinguishes "degrade the opinion" from "degrade the interlock." Breakers failing open, empty retrievals indistinguishable from emptiness (the exact defect class of incident DJ-120), and Hold-by-default on total parse failure are all compliant with a vague "degrade gracefully."
4. **No transaction-cost model anywhere** — not even as an open question. Fees/slippage absent from simulation; the David's own metric table lists Turnover, which nothing computes. Any simulated-vs-live comparison is therefore systematically flattering.
5. **Open questions without resolution deadlines become permanent holes.** Layer 10's registry question (§7.10) stayed open through sixteen executed phases and was never built. The David needs a rule: an OQ older than N phases either gets resolved or blocks dependent work.
6. **Minor:** §7.9's "halt after 3 consecutive losses >1%" breaker is statistically naive (loss streaks are common noise); hardware tiers assume away the model-swapping latency problem flagged elsewhere (OQ-A04 remains unanswered and in practice bounds universe size per nightly run).

---

## 7. What is BETTER than the David

Credit where the machine exceeded its god:

1. **Operational hardening born from incidents**: broker/book sync with vanish-probe delisting detection, watchdog process, impact-scaled position halt, per-account isolation, retry taxonomy separating idempotent GETs from non-idempotent submits. The David's execution layer is one paragraph; reality is a hardened daemon.
2. **`analytics/decision_audit.py`** — provenance-first analytics answering the DJ-120 class of question ("did the tool fail, or did the agent think?") that the David never anticipated.
3. **External riskbudget strategy arm** in the live report — a real third-party baseline, richer than the David's baseline list.
4. **MCP venv isolation** (`venvs/ta/`) — dependency isolation executed exactly as DJ-010 imagined; each venv maps cleanly to a future Docker service.
5. **Drift monitoring across all four layers** (data/concept/agent/collective) with a dedicated calibration script — most production shops lack this.
6. **Synthetic scenario harness (Family F)** built and evaluated in Phase 13, far ahead of the roadmap's implication.
7. **The bitácora/DJ discipline itself** — self-auditing post-hoc: GENESIS_CHECKLIST documents a retracted headline IC result; DJ-128 caught its own null ablation ("the memory ablation ablated nothing"). The epistemic immune system the David hoped for *exists* — it currently catches leaks faster than they are fixed, which is both its strength and the project's race condition.

---

## 8. Remediation plan to close the gap

Ordered so that each tier unblocks the next. Phase numbers continue the existing Protocol sequence (currently at Phase 16). Each phase follows house style: bitácora entry, DJ decisions where architectural, tests before merge, acceptance criteria stated falsifiably.

### Phase 17 — Temporal Integrity *(blocks all scientific claims)*
Goal: make every historical decision reproducible from information available at its date.
1. Add mandatory `as_of_date` to `retrieve_context` and all vector/graph search paths; fail closed when absent; thread `filed_date` through chunk schema and storage; filter on availability, keep period for display. *(Closes C1.1–C1.2)*
2. Episodic/memory recall: require `decision_date + labeling_horizon < as_of` when serving historical evaluations; add `as_of` to `memory.recall()` signature before any walk-forward wiring. *(Closes C1.3, §10.4 leak)*
3. Switch macro snapshots to vintage-aware ingestion (FRED ALFRED vintage series or release-date tables carried locally); assert publication lag ≥ 0. *(Closes C1.4)*
4. As-of performance weights: compute weights strictly from records labeled before as_of. *(Closes C1.5)*
5. Purge/embargo module for walk-forward: enforce López de Prado-style purge of `horizon_days` between folds; unit-test with deliberately overlapping labels. *(Closes C8)*
6. Date-based embargo split for fine-tuning (train ≤ cutoff; eval > cutoff + horizon); rewrite `optimal_rank.json` provenance honestly (capture stdout loss or delete sweep machinery). *(Partially closes C5)*

**Acceptance:** rerun one prior headline evaluation (e.g., Phase 15 IC) under the closed pipeline; record the delta in the bitácora whether it improves or collapses. That number — honest this time — becomes the new baseline.

### Phase 18 — Data Truthfulness
1. Delisting-aware labeler: delisting ⇒ terminal return recorded, never `None`; accuracy denominators include failures. *(Closes C2b)*
2. Universe membership history: as-of constituent lists (even coarse: added/delisted dates per ticker) replacing today-curated list applied backward. *(Closes C2a)*
3. One price-basis contract: pick adjusted-everywhere (recommended for signal paths); nightly updater refetches a rolling window covering the longest indicator horizon on every corp action; wire `quality.py` checks into acquisition so gates stop being decorative. *(Closes C3)*
4. Freeze EVAL-* dataset artifacts with content hashes and dataset cards (Gebru et al. style), even if unreleased publicly. *(Opens §8.8 contract)*

### Phase 19 — Safety Wiring *(blocks any capital beyond paper)*
1. Idempotency: deterministic `client_order_id = account-date-ticker-side`; pre-submit existence check; write intent row before first submit; fill reconciler polling terminal order states after next open.
2. Wire `SafetyConfig` into pipeline; delete hardcoded twin constants; enforce quorum (abort cycle below threshold); persist true running HWM; breakers fail closed on verification failure; re-check breakers immediately pre-submission.
3. Port the ET market-hours guard from shell wrapper into `main()` (zoneinfo; opt-out env mirroring existing convention).
4. Split discontinuity guard in nightly updates (refetch-on-anomaly using existing corp-action ratio check).
5. Adapter approval manifest (trained-from, eval metrics, approved-by/at); serve scripts refuse unmanifested adapters; agents log adapter identity per call.

**Acceptance:** chaos drills in CI — kill mid-submission and rerun (expect zero duplicates), broker error during breaker check (expect halt), quorum violation (expect abort).

### Phase 20 — Instrument Honesty
1. Structured-template claims (the David's original preference) as primary path, regex as fallback; track claims-per-1k-chars as health metric; near-zero density flags the report instead of scoring 0.0 HR. *(Closes C4)*
2. Confidence calibration loop closed: calibrated probabilities feed performance/confidence weighting (OQ-AG03 answered then used).
3. Config: `extra="forbid"`; single source of truth for risk limits.
4. Experiment registry: minimal JSONL registry keyed by run_id carrying config hash, dataset hashes, model identities, prompt hashes — Layer 10 delivered at Phase-1-scale simplicity.
5. Trace durability (`atexit`/context-manager flush) + redaction middleware.

### Phase 21 — Completion of the Roster and Spec Amendments
1. Build the Valuation Agent (7th role) **or** amend the David to six roles with a recorded justification — the David itself demands explicit, recorded simplifications; silent absence satisfies neither branch.
2. Multi-horizon labels {5,10,20,60,120}; turnover metric; random + momentum baselines.
3. Amend the David (living document, per its own §1.2): add execution-idempotency requirement, retrieval-temporality normative sentence, fail-open/fail-closed distinction, transaction-cost modeling as an OQ, and the OQ-resolution-deadline rule. A reference that cannot be violated loudly gets violated quietly — which is the story Sections 4–6 tell.
4. Homogenize-model and remove-one-agent ablations (§15.7), now meaningful once Phase 17 removes correlated-input contamination.

### Phase 22 — Deployment and Open Release *(only if community criteria matter)*
Containerize per §16.1 (each venv → service, as DJ-010 already maps); deployment-from-docs-alone test with a fresh machine; dataset cards + public release; then §19.4 becomes answerable.

---

## 9. Enhancements and next directions

Beyond remediation, the directions with highest leverage on the David's actual research questions (§2.3):

**Research-grade upgrades**
1. **Independence measurement done honestly (OQ-AG04)** — once shared-context injection is removed, measure pairwise decision correlation per model family; this is the paper-worthy core result the platform exists to produce (Breiman's ρ term made empirical).
2. **Adaptive aggregation (§12.2.5)** — now trainable legitimately: logistic regression over (votes, confidences, entropy, regime) with purged CV. The data byproduct (Family E) already accumulates exactly the required features.
3. **Diversity-performance correlation study (§15.4)** — H vs forward decision quality, segmented by regime; requires Phases 17–18 for trustworthy labels.
4. **Regime-conditioned weighting** — performance weights conditioned on regime classifier (`data/regime.py` already provides the segmentation).
5. **Contrarian mechanism comparison (OQ from §12.3)** — confidence discount vs veto vs Bayesian update, evaluated head-to-head; machinery exists, comparison was never run.

**Engineering upgrades**
6. Transaction-cost model in simulation (fees + slippage bps + turnover-aware rebalance), enabling honest sim↔live deltas.
7. Parallel agent inference benchmark (OQ-A04) to shrink nightly wall-clock and enlarge per-cycle universe.
8. Binary serialization at the MCP boundary if OHLCV payload overhead ever binds (OQ-A01 — still unanswered, cheap to measure once).
9. Embedding-space versioning stamp + assertion on query; chunking A/B harness generalized from Phase 7 fixtures (OQ-K01).
10. Memory decay design (OQ-AG02): exponential decay with half-life swept over {30, 90, 250} sessions, evaluated by downstream decision quality, not intuition.

**Strategic options**
11. If community release matters: the fastest credible artifact is the *evaluation harness* (walk-forward + leakage tests + ablations) rather than the trading system — it is novel, fully local, and doesn't require brokerage integration for third parties.
12. If trading matters more than research: Phases 17–19 are the minimum; consider freezing model population (no fine-tune churn) until the approval manifest exists.

---

## 10. Definition of Done against David §19

| Criterion | Currently | Done means |
|---|---|---|
| §19.1 Capstone (end-to-end, agents, collective, verification, paper trading, docs) | Met | Stays met |
| §19.2.2 Reproducible results | Partial | Registry chain run_id → config/data/model/prompt hashes reconstructs any episode |
| §19.2.4 Deployable by external users | No | Fresh-machine setup ≤ 1 day from docs alone |
| §19.2.5 HR < 2% on objective claims | Unmeasurable | Measured with claims-density floor; reported with extraction coverage |
| §19.3.1 Ensemble > average individual (H1) | Compromised | Re-established under embargo + survivorship-correct labels; result either way recorded in bitácora |
| §19.3.2 Diversity contribution measurable (H2) | Contaminated | Pairwise correlation matrix post-decontamination; ρ reported alongside ensemble gain |
| §19.3.3 Verification reduces hallucination (H4) | Weak instrument | Template-based extraction; before/after HR with CIs |
| §19.4 Community success | Not started | Phase 22 |

---

## Appendix A — Evidence chain

Per the project's own provenance discipline (cf. `analytics/decision_audit.py`), claims in this document carry the following warrant:

- **Personally verified by direct read this session:** ATR mixed-series (`technical.py:221–228`); VaR positional alignment and weight-drop (`risk_manager.py:93–113`); dead HWM breaker (`run_phase16_live.py:371` + `risk_manager.py:146–148`); missing client_order_id and post-hoc dedup marker (`alpaca_executor.py:165–187`, `run_phase16_live.py:148,1052,1063,1072`, incl. the :1066 incident comment); composer step ordering (`portfolio_composer.py:240–299`); sentiment 4× initial weight (`performance_store.py:31–32` + `voting.py:223`); config extras silently ignored vs docstring (`loader.py:58–109`); unbounded `retrieve_context` (`knowledge_server.py:108–113`); tearsheet t0 inclusion (`tearsheet.py:160–192`).
- **Verified by grep on 2026-08-23:** absence of stop-loss, purge/embargo, turnover computation, MLflow/experiment registry, transcripts/news modules; presence of `prompt_version` metadata strings; Sortino/Calmar present in tearsheet; no valuation agent file.
- **Sourced from structured subsystem review (traced to cited lines, not independently re-read):** remaining line citations (ensemble_runner context sharing, debate_nodes anchoring, labeler delisting None, macro forward-fill, fine_tune train_loss=None, extractor grammar, SafetyConfig orphan status, docker-compose defaults).

Line references valid at commit `4be5daa`. Any subsequent edit may shift them; the code identifiers quoted alongside remain findable.

## Appendix B — References

Works invoked by this evaluation (full citations in HIFI_DAVID.md §20; listed here where load-bearing for a specific finding):

- **López de Prado (2018)**, *Advances in Financial Machine Learning* — purged K-fold CV and embargo; the technique whose absence constitutes finding C8 (§15.2 names it; implementation omits it).
- **Breiman (2001)**, *Random Forests* — the ρ-correlation term in the ensemble error decomposition; C7 shows the platform inflates its own ρ via shared context.
- **Surowiecki (2005)**, *The Wisdom of Crowds* — independence condition violated by C7; the David quotes the conditions, the code breaks one.
- **Page (2007)**, *The Difference* — diversity decomposition; measurement mandated by §10.3 remains confounded until decontamination.
- **Gebru et al. (2021)**, *Datasheets for Datasets* — the standard for the missing dataset cards (§8.9, Phase 18/22).
- **Knight (1921)**, *Risk, Uncertainty and Profit* — grounds why the David promises well-reasoned rather than optimal decisions; relevant to accepting Tier-1 reruns that may *worsen* headline numbers.
- **ALFRED / FRED vintage databases** (research.stlouisfed.org/alfred) — the concrete remedy for macro point-in-time violations (Phase 17.3).
- **Alpaca Trading API documentation** — `client_order_id` deduplication semantics; the zero-cost prevention for the duplicate-order window (§6.1, Phase 19.1).

---

*Closing note, in the David's own spirit:* the David ends by saying reality adapts to it "one layer at a time." Sixteen phases built the layers faithfully. What remains is to make the layers honest — and to amend the David where reality taught it something it did not know about itself: reflexes, retrieval clocks, and the difference between degrading gracefully and failing quietly.
