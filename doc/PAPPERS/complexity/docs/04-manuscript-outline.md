# Paper I — Manuscript Outline

**Version:** 0.1 — revisable at any time (unlike doc 02, which is immutable, and
doc 03, which is frozen before analysis).
**Created:** 2026-09-01
**Governed by:** `02-charter.md` · **Evidence rules:** `03-statistical-analysis-plan.md`
**Target:** JASSS (Journal of Artificial Societies and Social Simulation), or
JEDC if the framing leans economic. Secondary: ACM ICAIF.

---

## 0. How to use this document

This is the argument skeleton, not a template to fill. Each section below states
**the one thing that section must accomplish**. If a paragraph does not serve its
section's job, it is cut regardless of how much work it represents.

Two rules from the charter govern every section:

- **The abstraction test** (charter §1.3) — would this still matter if every LLM
  were replaced tomorrow? If no, cut it.
- **Adversary before advocate** (charter §5.1) — the strongest objection to each
  result is written in the analysis log *before* the result text, and the text
  answers or concedes it.

**Writing order is not section order.** See §14.

---

## 1. Title

Requirements: scientifically meaningful, mechanism-forward, no marketing
register ("revolutionary", "autonomous", "superintelligent", "next-generation",
"high-performance").

Working candidates — the title is chosen last, after the result is known:

- *Bounded Diversity: Information Partitioning, Not Model Heterogeneity, Governs
  Independence in LLM Collectives*
- *Do Machine Crowds Have Wisdom? Effective Independence in Populations of Large
  Language Models*
- *The Substrate Floor: Why Architectural Diversity Does Not Buy Decision
  Independence*

Reject any title containing "trading system", "outperforms", or a model name.

---

## 2. Abstract

**Do not write until the evidence is mature.** Then, in this order and no other:

1. The gap — collective-intelligence results were derived for human and
   statistical ensembles; LLM populations share a pretraining substrate and may
   not inherit them.
2. The design — 2 × 2 factorial over model heterogeneity and information
   partitioning, interleaved, pre-registered, with a replicate cell measuring the
   collective's own reproducibility floor.
3. The primary result — θ, with CI, against the floor δ₀.
4. The contribution — mechanism, plus a protocol for making such claims honestly.
5. The limits — one market, one model family group, one cycle per day, and the
   power statement.

The floor δ₀ appears in the abstract (SAP §3.3). So does the power statement if
the study is underpowered. Non-negotiable.

---

## 3. §1 Introduction

**Job: establish a research gap, not a software gap.**

The move, in five beats:

1. Collective intelligence has a strong theoretical core — Page's identity makes
   the benefit of diversity a matter of algebra, not luck.
2. That core is now being applied, implicitly, to populations of LLM agents:
   multi-agent systems are built on the premise that more, and more varied,
   agents deliberate better.
3. But LLM agents are not human crowds. They are drawn from overlapping
   pretraining corpora. **Their errors may be correlated by construction**, in a
   way no result in the human-crowds literature anticipates.
4. Nobody has tested this, because nobody in the applied literature separates
   model diversity from information diversity, and nobody runs a control.
5. We do, in a live financial decision environment, and here is what we find.

State early and plainly, to disarm the first reviewer objection: **we are not
testing Page's theorem.** The identity cannot fail. We are measuring whether its
diversity term is reachable in this substrate.

End with: hypotheses (tagged confirmatory/exploratory), principal result,
contribution.

---

## 4. §2 Related literature

**Job: establish the intellectual problem, and prove we know the field.**

Three strands, in this order:

**2.1 Collective intelligence and ensemble diversity.** Page, Surowiecki,
Woolley, Nemeth, Schwenk, Dietterich, Hong & Page. State the bridge explicitly:
these are results about *human* and *statistical* ensembles. The transfer to LLM
populations is an assumption the applied field makes silently.

**2.2 LLM agents in financial decision-making.** FinRL (ICAIF 2021), FinGPT
(FinLLM @ IJCAI 2023), FinRobot (arXiv:2405.14767), TradingAgents
(arXiv:2412.20138), AI-Trader (arXiv:2512.10971), LiveTradeBench
(arXiv:2511.03628), FinMem. Organise by **what varies**: the model, the library,
the architecture. The framing sentence: *every one of these is a leaderboard;
none manipulates a theory-derived variable, none has a control condition, none
pre-registers.* This is the paper's opening move and it must be accurate.

Two corrections already made during benchmarking, never to be reintroduced:
FinRL never pivoted to LLMs; TradingAgents is not an ICLR paper. Also
never reintroduce the withdrawn claim that live execution is our
differentiator — AI-Trader explicitly claims a live, data-uncontaminated
benchmark.

**2.3 Evaluation validity in this field.** The leakage challenges to
TradingAgents; LiveTradeBench's finding that static-benchmark winners
underperform live; AI-Trader's uncontaminated framing. Position pre-registration,
the null arm and the label-free primary endpoint as the response.

Close with the gap in **one sentence** that the rest of the paper answers.

Discipline: no citation for a convenient sentence; every entry resolves to a DOI
before it enters `refs.bib`. Seventeen references currently sit in
`HIFI_DAVID.md` §20 as prose with no DOIs and are cited nowhere — migrate and
verify them.

---

## 5. §3 Conceptual framework

**Job: define independence precisely enough that the measurement is forced.**

This is likely the most important section of the paper for a complexity
audience, and the one document 01 gestured at without delivering.

- The chain: information → agents → decisions → correlation → effective
  independence → collective quality.
- The two-source decomposition of independence: **model** vs **information**.
  Give the argument for why they can dissociate, and why shared pretraining makes
  the first weak.
- Derive n_eff (participation ratio) from the identity's diversity term rather
  than asserting it: show that what the identity rewards is precisely low
  cross-agent error correlation, and that n_eff is its scale-free summary.
- State the substrate-floor hypothesis formally.
- Predictions, written before results: θ > 0 under H-PRIMARY; θ ≤ 0 falsifies.

Equations numbered. Every symbol named in words at first use. All definitions
derived from `src/hifi/simulation/metrics.py` and the SAP, **not from memory**.

---

## 6. §4 The experimental platform

**Job: only what a reader needs to believe the manipulation was real.**

Include: the five voting agents and their evidence channels; the contrarian as a
**reviewer that does not vote** (state once, correctly); the model assignment;
aggregation; the retrieval and memory components *only* insofar as they are
sharing channels; the decision record schema.

Exclude, per charter §5.3: MCP, LangChain, LangGraph, LanceDB, LM Studio,
Docker, LoRA, Langfuse, vendor tooling. A reader must be able to reimplement the
*experiment* in a different stack.

One architecture figure. Not three.

---

## 7. §5 Data

Universe and membership; price data and its adjustment basis; fundamentals;
macro; text; **information availability dates**; preprocessing; labels and
horizons where used.

Disclose here, not in limitations: the point-in-time, survivorship and
corporate-action violations found on 2026-08-23, and the fact that the primary
endpoint is label-free and therefore does not depend on them. This is a strength
argued openly, not a weakness hidden.

---

## 8. §6 Experimental design

**Job: a skeptical reviewer reconstructs the entire logic from this section
alone.**

The 2 × 2 + replicate; operational definitions of both manipulations;
interleaved randomisation and why (the DJ-128 time confound, named); the seven
gates including the manipulation check; unit of analysis and effective sample
size; the block bootstrap and permutation test; the power analysis and its
verdict; the decision rule, quoted from SAP §7.1; the pre-registration and its
amendment history as a **numbered subsection, not a footnote**.

This section is where the paper earns its credibility. Write it long.

---

## 9. §7 Results

Order fixed in advance, and it is not "best first":

1. **Gates.** All seven, with outcomes. A paper that reports its negative
   controls before its findings is read differently.
2. **The floor δ₀.** Before any effect, the size of the effect we cannot resolve.
3. **Primary: θ**, with CI, permutation p, and the decision-rule verdict stated
   in words.
4. Cell-level n_eff, the 2 × 2 table.
5. Secondary label-free endpoints S1–S5 (FDR-corrected and raw).
6. Secondary label-dependent L1–L3, with the validity caveat in the same table.
7. Robustness (SAP §8).

Negative results are not buried. If θ is inconclusive, the section says
"inconclusive" in its first sentence.

---

## 10. §8 Mechanism

**Job: answer why — or state that we cannot.**

- Where does the correlation come from? Decompose by agent pair, by evidence
  channel, by regime (exploratory).
- The synthetic-environment check (SAP §7.4): does n_eff recover a correlation
  floor that we imposed by construction? For this venue this may be the most
  persuasive single result in the paper.
- **The failure modes as mechanism, not apology** (charter §3.2): agents deprived
  of evidence emit maximum confidence rather than abstaining; agents holding no
  real text are irreproducible across runs while agents holding text are
  byte-identical. Silent degradation that is indistinguishable from signal is a
  general property of machine collectives and this is where it is reported.

If the mechanism is unsupported, say so here in one sentence and stop.

---

## 11. §9 Robustness and falsification

**Job: show we tried to destroy the finding.**

Alternative periods, horizons, universes, correlation measures, block lengths,
leave-one-out at date and agent level, the serving-stack covariate, the negative
controls, the synthetic placebo. Report each outcome, including the ones that
weaken the result.

Then, explicitly: **what did not survive**, and what that costs the claim.

---

## 12. §10 Discussion and §11 Conclusion

**Discussion.** What the findings mean; what they explicitly do **not** mean;
relation to the collective-intelligence literature; implications for multi-agent
AI design (if independence must be bought with information architecture rather
than model choice, most current systems are buying the wrong thing);
implications for financial decision systems; limitations stated as a list, not
diffused into prose.

Standing limitations to name, not hide: one market; one model-family group; one
decision cycle per day; effective sample size far below nominal; local model
pretraining cutoffs as a leakage concern for any historical evaluation, against
which live forward decisions from 2026-08-24 are immune by construction — keep
that distinction explicit, it is the field's live controversy.

**Conclusion.** Answer the question directly, in the first sentence. Claim
nothing beyond the evidence. One forward pointer to Paper II.

---

## 13. Figures and tables

Every figure answers a question. Figures that merely look impressive are cut.

| # | Figure | Question it answers |
|---|---|---|
| F1 | Conceptual framework: information → agents → correlation → n_eff | What is the causal chain being tested? |
| F2 | Experimental design: the 2 × 2 + replicate, with interleaving | What was manipulated, and against what control? |
| F3 | **n_eff by cell, with δ₀ shown as a shaded band** | Is any effect larger than our own noise? *(the paper's central figure)* |
| F4 | Agent correlation matrices, one per cell | Where does the correlation live? |
| F5 | θ with bootstrap distribution and permutation null | Is the primary contrast real? |
| F6 | Synthetic recovery: imposed floor vs measured n_eff | Does the instrument measure what it claims? |
| F7 | Reproducibility vs evidence availability, per agent | Why is the floor there? |
| F8 | Leave-one-agent-out contribution to n_eff | Is one agent doing all the work? |

| # | Table | |
|---|---|---|
| T1 | Agent, model, evidence channel, voting status | |
| T2 | Cell definitions and realised sample | |
| T3 | Gate outcomes (all seven) | |
| T4 | Primary and secondary endpoints, raw and FDR-corrected | |
| T5 | Robustness matrix | |
| T6 | Pre-registration: registered element → status → amendment | |

T3 and T6 are unusual in this literature. They are the point.

---

## 14. Writing order

Sections that depend on no data are written now, in parallel with Phase 21:

1. §2 Related literature — and `refs.bib` with every DOI verified
2. §3 Conceptual framework — the n_eff derivation
3. §6 Experimental design — from the SAP, before the freeze
4. §4 Platform, §5 Data

Then: freeze the SAP → file OSF amendment 002 → run → gates → §7 → §8 → §9 →
§10/§11 → §1 → title → abstract.

The introduction and abstract are written **last**, from the result. Writing them
first is how retrospective storytelling gets in (charter §2.6).

---

## 15. Standing prohibitions

- No claim that Page's theorem was "confirmed" or "tested".
- No economic performance claim (Paper II).
- No p-value computed under an i.i.d. assumption.
- No exploratory result in the abstract, and none phrased as a test.
- No result reported without δ₀ available for comparison.
- No technology in the title, abstract, or conclusion.
- No citation that has not been resolved to a DOI.
