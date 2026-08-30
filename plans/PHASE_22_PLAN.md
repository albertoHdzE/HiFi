# Phase 22 Plan: Paper I — Ablation Analysis and Capstone Deliverable

**Created:** 2026-08-27 · **Status:** PLANNED
**Execution mode:** long-running, parallel with Phase 21. T1–T3 have no
dependency on Phase 21 or on further live data and start immediately.
See `PHASE_22_CONTEXT.md` for venue reasoning and the two-paper split.

**Target:** complexity-science venue (JASSS / JEDC); ACM ICAIF as secondary.
**Deliverable doubles as the capstone submission.**

---

## T1 — Manuscript skeleton and bibliography (no dependencies, start now)

- [ ] T1.1 Create `paper/` with a LaTeX skeleton and `paper/refs.bib`.
      Decide and record: LaTeX vs Markdown+pandoc. Target journal's template
      governs; check before writing prose.
- [ ] T1.2 Seed `refs.bib` with the six anchors currently absent from the whole
      repository: FinRL (ICAIF 2021), FinGPT (FinLLM @ IJCAI 2023), FinRobot
      (arXiv:2405.14767), TradingAgents (arXiv:2412.20138), AI-Trader
      (arXiv:2512.10971), LiveTradeBench (arXiv:2511.03628).
- [ ] T1.3 Migrate the 17 existing references from `doc/HIFI_DAVID.md` §20 into
      `refs.bib` with DOIs. They are currently prose-only and uncited anywhere.
- [ ] T1.4 Verify every entry resolves. A fabricated or mis-attributed citation
      is the fastest possible desk rejection, and two attributions were already
      corrected during benchmarking (FinRL never pivoted to LLMs; TradingAgents
      is not an ICLR paper).

## T2 — §2 Related work (no dependencies)

- [ ] T2.1 Survey the LLM-trading-agent line and organise it by **what varies**:
      model (AI-Trader), library (FinRL), architecture (TradingAgents, WQU
      exemplar). Establish that none manipulates a theory-derived variable and
      none has a control condition. This framing is the paper's opening move.
- [ ] T2.2 Survey the ensemble-diversity line already in `HIFI_DAVID.md` §20 —
      Page, Surowiecki, Woolley, Nemeth, Schwenk, Dietterich — and state the
      bridge: these are claims about human and statistical ensembles, untested
      on LLM agent populations under controlled conditions.
- [ ] T2.3 Survey the field's evaluation-validity controversy: leakage
      challenges to TradingAgents, LiveTradeBench's static-vs-live divergence,
      AI-Trader's data-uncontaminated framing. Position HiFi's pre-registration
      and null arm as the response.
- [ ] T2.4 State the gap in one sentence that the rest of the paper answers.

## T3 — §3 Methodology (no dependencies on live data)

- [ ] T3.1 Agent population, the five voters and the contrarian **reviewer**.
      State explicitly that the contrarian does not vote — an internal analysis
      once mis-scored this and the error was published to the bitácora before
      correction. The manuscript states the design once, correctly.
- [ ] T3.2 The four conditions (parallel / full / homogeneous / no-memory) and
      the four live arms (A/B/C/D), with the null arm's role made explicit.
- [ ] T3.3 Metric definitions with numbered equations: herding coefficient κ,
      disagreement entropy, IC, and the buy-strength encoding. Every symbol
      named in words. `src/hifi/simulation/metrics.py` is the source of truth;
      the equations must be derived from the code, not from memory.
- [ ] T3.4 Effective sample size: state how cross-sectional correlation and
      overlapping windows are handled. Phase 15's p-values ignored this; the
      correction is a methodological contribution and is written as one.
- [ ] T3.5 Pre-registration and amendment history as a numbered subsection, not
      a footnote.

## T4 — §4 Results (depends on the Phase 15 re-run)

- [ ] T4.1 Blocked on: DJ-128 redesign (populate or drop the memory contrast;
      interleave conditions, which were time-confounded; `clear_run` per
      condition) and the B7–B10 leakage cluster. Both are Phase 21-deferred —
      confirm they are scheduled before this task opens.
- [ ] T4.2 Re-run the walk-forward on repaired data. Primary output: does
      topology change measured diversity, and does diversity predict IC.
- [ ] T4.3 Every figure passes `datasaurus` G1/G1b: render the objects at full
      length, not summary statistics about them. No claim of agreement or
      difference without an elementwise comparison and its symmetric difference.
- [ ] T4.4 Report each result against its null. The 43.2% disagreement between
      two configurationally identical conditions (DJ-128) is the design's
      empirical noise floor and is the reference distribution for every
      divergence claim in this section.

## T5 — §5 The retraction section

- [ ] T5.1 Write Phase 15's retraction as a **methods contribution**, not an
      apology: reported IC = +0.0642 (p = 0.0019), which decomposed to −0.1377
      on the 16 data-bearing tickers against +0.0669 on the 83 blind ones.
- [ ] T5.2 Generalise to the defect taxonomy: silent data starvation (DJ-120),
      constraint self-blinding (DJ-121/122/131), rejected artifacts left wired
      (DJ-124, and its second habitat in DJ-131 §1.4), silent misconfiguration
      (D22, T6.6). Each with the instrument that would have caught it earlier.
- [ ] T5.3 State the general lesson plainly: **agents render missing evidence as
      conviction.** A blinded agent does not report an outage, it reports a
      confident opinion. That is the transferable finding and the reason this
      section exists.

## T6 — Submission

- [ ] T6.1 Investigate the Registered Report route before §4 is drafted
      (`PHASE_22_CONTEXT.md` §3). If available at a suitable venue, restructure:
      Stage 1 is submittable from T1–T3 alone.
- [ ] T6.2 §1 Introduction and §6 Discussion, written last.
- [ ] T6.3 Limitations section from `PHASE_22_CONTEXT.md` §7 — single market,
      single model family, effective n, contaminated windows, model provenance.
      Named by us, in the paper, not left for a reviewer.
- [ ] T6.4 Internal adversarial read against the same standard applied to the
      WQU exemplar: does any table contradict any claim in the abstract? That
      exemplar's own ablation showed +0.000 for its headline novelty while the
      abstract claimed superiority. Run that check on ourselves before submitting.
- [ ] T6.5 Capstone submission; then journal submission.

---

## Acceptance

1. `paper/refs.bib` resolves, every entry verified, no fabricated attributions.
2. §1–§3 complete and readable end-to-end without the results section.
3. Every number in the manuscript passes the `datasaurus` gates, and every claim
   of difference carries its reference distribution in the same sentence.
4. No performance claim from the live arms appears anywhere in Paper I.
5. The retraction section is present and leads the methods contribution.

## Non-goals

Live economic results (Paper II, Phase 23); open-source release (Phase 23);
any claim from the live record before OSF amendment 002 is filed.
