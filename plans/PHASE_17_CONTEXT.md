# Phase 17: Ablation Studies + Capstone Deliverable
## Context and Pre-Phase Decisions

**Gathered:** 2026-06-16
**Status:** NOT STARTED — awaits Phase 15 results + Phase 16 (strongly preferred)
**Depends on:** Phase 15 (walk-forward results for ablation baseline), Phase 14 (full pipeline)

---

## Why This Phase Has Two Streams

### Stream 1: Ablation Studies (scientific completeness)

The David (§15) requires ablation before any publication claim is defensible. An ablation
answers: "How much does each component contribute? Would removing it degrade performance,
and by how much?" Without ablation, a claim like "our ensemble achieves IC=0.07" could
be entirely due to the data pipeline, not the ensemble architecture.

Phase 14-16 built all the prerequisites:
- Phase 15 provides the baseline (full pipeline IC/IR/Sharpe on held-out 2022-2023)
- Phase 14 built the components that can be individually removed
- Phase 16 provides live evidence that supplements the ablation

### Stream 2: WQU MScFE Capstone Submission

WQU requires a capstone deliverable demonstrating a working, evaluated financial
intelligence system. This is non-negotiable for graduation. The capstone is a SUBSET
of the scientific work — it showcases the system honestly without requiring
publication-quality rigor in every section.

---

## DJ-101: Ablation Study Design

**Decision:** Re-run held-out 2022-2023 walk-forward with each component removed.
The full pipeline result (from Phase 15) is the baseline. Each ablation condition
changes exactly ONE component.

| Condition | What changes | Scientific question answered |
|---|---|---|
| Full pipeline | Baseline (Phase 15 result) | — |
| Remove Technical FT | technical_v2 → base qwen2.5-coder-32b | Is the fine-tuned Technical adapter contributing? |
| Remove sequential debate | run_ensemble() parallel mode | Does inter-agent context sharing improve IC? (OQ-P14-03) |
| Remove episodic RAG | No episodic prefix injected | Does outcome-labeled memory improve decisions? (OQ-P14-04) |
| Remove Contrarian | 5-agent subset (no contrarian) | Does the devil's advocate role add IC or reduce herding? |
| Remove verification | No HR/GR/SGR gate | Does the verification layer prevent harmful decisions? |
| Homogeneous ensemble | Phase 13 model config (Qwen-dominant) | Page theorem test: diversity effect on IC |

The last condition (homogeneous) is the most important. It directly answers:
"Is the 5-organization diversity responsible for the performance improvement, or would
any well-tuned 6-agent ensemble achieve the same IC?"

**Reporting:** Each ablation reports IC delta, IR delta, and Sharpe delta vs. full pipeline.
If delta < 0: component contributes positively. If delta ≈ 0: component is neutral.
If delta > 0: component hurts performance (examine why).

---

## DJ-102: Capstone Structure and Strategy

**WQU format:** Jupyter notebook or formal report. Format confirmed with WQU advisor
at Phase 17 start (requirements may vary by cohort year).

**Narrative strategy:** The capstone tells a story of empirical discovery, not just
engineering. The arc:

1. Problem: Financial decision-making under uncertainty. LLMs can reason about markets
   but hallucinate, herd, and agree too readily when architecturally similar.

2. Hypothesis: A heterogeneous ensemble (Page's diversity theorem applied to LLMs)
   will outperform a homogeneous one. We test this with 21 years of data.

3. System: HiFi — deterministic financial engines (MCP) + heterogeneous LLM agents
   + collective decision aggregation + full verification layer + drift monitoring.

4. Evidence: Phase 15 IC on 100 stocks, 5 regimes. Phase 15 ablation isolating
   each component's contribution. Phase 16 live paper trading confirming signal quality.

5. Conclusion: Honest assessment — what worked, what failed, what this means for
   the field, what remains open.

The complexity science lens (emergence, diversity, collective intelligence) is the
epistemological frame that elevates this from "we built a trading bot" to
"we tested a fundamental claim about heterogeneous agent systems."

---

## Open Questions

| ID | Question | Resolution |
|---|---|---|
| OQ-P17-01 | Does removing diversity (homogeneous ablation) significantly degrade IC? | Phase 17 ablation |
| OQ-P17-02 | Which single component contributes most to IC? | Ablation ranking |
| OQ-P17-03 | Does the verification layer reduce harmful decisions (high confidence, wrong direction)? | Ablation |
