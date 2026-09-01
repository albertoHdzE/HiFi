# Paper I — Research Charter

**Status:** IMMUTABLE once ratified. Supersedes `01-initial.prompt.txt`.
**Created:** 2026-09-01
**Venue:** complexity-science primary (JASSS; JEDC if the framing leans economic).
Secondary, only if the mechanism result is strong: ACM ICAIF.
**Companion documents:** `03-statistical-analysis-plan.md` (versioned, frozen
before analysis), `04-manuscript-outline.md` (revisable).

---

## 0. What this document is

The charter fixes the epistemics of the study: the question, the standard of
evidence, and the conditions under which we abandon the claim. It is written to
be **unchangeable by results**. If a result makes a rule in this document
inconvenient, that is the rule working.

The Statistical Analysis Plan (SAP, doc 03) is the mutable-but-versioned
instrument: hypotheses, endpoints, tests, thresholds. Its edit history, with
every edit dated relative to data collection, *is* the integrity evidence of
this programme. The manuscript outline (doc 04) is freely revisable.

Three documents, three lifetimes. Do not merge them.

### 0.1 Relationship to `01-initial.prompt.txt`

Document 01 was drafted against **Management Science**. That target is
withdrawn: it is an economics/OR venue whose reviewers require causal
identification of an economic effect, and the study we can actually power is a
mechanism study. Document 01's epistemic sections (§2 do-not-protect-the-
hypothesis, §16 alternative explanations, §17 negative controls, §27 what
uncertainty does this remove, §29 no retrospective storytelling) are absorbed
here almost intact and remain binding. Its experimental matrix (§5), its six
co-equal hypotheses (§4), and its Management Science adaptation phase (§26.I)
are withdrawn.

---

## 1. The question

> **Do results on collective intelligence, derived for human and statistical
> ensembles, transfer to populations of large language models — and if not, what
> is the binding constraint?**

The financial market is the empirical environment, not the subject. It is chosen
because it returns unforgiving, timestamped ground truth on every decision and
because it forbids the experimenter from grading their own homework.

### 1.1 Why the obvious framing is wrong

Page's diversity prediction theorem is an **algebraic identity**:

> collective squared error = average individual squared error − prediction diversity

It cannot fail. An experiment reporting that it "holds" reports arithmetic, and
the first reviewer will say so. Any framing of this paper as *"we confirmed
Page's theorem on LLM agents"* is a desk rejection and is prohibited by this
charter.

The empirical content is entirely in the second term. The identity guarantees
that diversity helps; it says nothing about **how much diversity an LLM
population can attain**. That is the open question, and it is ours.

### 1.2 The central claim under test

> **H-PRIMARY.** In collectives of large language models, effective decision
> independence is bounded above by a substrate-induced floor on error
> correlation. Manipulating *information access* moves effective independence;
> manipulating *model identity* does not, because distinct model families share
> overlapping pretraining corpora and therefore share errors.

The distinction the field elides — and which document 01 correctly identified as
its C7 without recognising it as the contribution — is:

| | |
|---|---|
| **Model diversity** | different weights, different families |
| **Information diversity** | different evidence reaching each agent |

Every comparator system (FinRL, FinGPT, FinRobot, TradingAgents, AI-Trader) buys
the first and assumes it purchases the second. H-PRIMARY says it does not. If
true, this is a property of machine collectives with **no analogue in the human
crowds literature** — Surowiecki's, Page's and Woolley's crowds do not share a
pretraining corpus — and it is a genuine contribution to complexity science
rather than to a trading leaderboard.

If false, the negative result is equally publishable and equally interesting:
architectural heterogeneity would then be a sufficient route to independence,
which the ensemble-learning literature would not have predicted for models
trained on a shared web corpus.

### 1.3 The abstraction test

Document 01 §30 posed the right criterion and it is retained verbatim as the
charter's acceptance test for the contribution:

> *If every LLM in HiFi were replaced tomorrow, would the central finding still
> matter?*

H-PRIMARY passes. "Our ensemble earned X%" does not. Any draft section that
fails this test is cut.

---

## 2. Standard of evidence

### 2.1 Do not protect the hypothesis

Retained from document 01 §2, unchanged and binding.

The purpose is not to show that HiFi works. It is to find out whether the
hypotheses survive. It is forbidden to adjust datasets, evaluation windows,
agent selection, aggregation rules, thresholds, prompts, labels, model selection
or evaluation procedures in order to preserve a desired result; to cherry-pick
periods; to drop unfavourable runs; or to re-run until a result appears. Every
material methodological change is documented in the bitácora and, where it
touches a registered element, in an OSF amendment.

The programme must remain capable of emitting the sentence *"our earlier result
was wrong."* It has done so once already, at cost, and that retraction is this
project's strongest credential.

### 2.2 Confirmatory and exploratory are different kinds of claim

Every hypothesis in the SAP carries exactly one tag:

- **CONFIRMATORY** — specified, with its test and threshold, in the OSF
  registration or a dated amendment **before** the data it is tested on existed.
- **EXPLORATORY** — everything else, including everything suggested by looking
  at data.

Exploratory findings may be reported, may be interesting, and may motivate Paper
II. They may never be described using the language of hypothesis testing, and
they never carry a p-value in the abstract. A confirmatory claim whose amendment
post-dates the data is reclassified as exploratory, automatically and without
negotiation.

### 2.3 One primary endpoint

The study has **one** primary endpoint and one primary contrast, fixed in the
SAP before analysis. Everything else is secondary or exploratory and is reported
with the multiplicity correction stated in the SAP.

Document 01 listed six co-equal hypotheses and seventeen performance measures.
That design has no spine and cannot survive a multiple-comparisons objection.
Withdrawn.

### 2.4 Negative controls are mandatory, not optional

A scientific instrument must fail when it should. Before any positive result is
believed, the design must demonstrate a null result under conditions where no
effect can exist. The SAP specifies these as gates, not as robustness checks:
they run first, and a positive result on a null condition halts the analysis.

### 2.5 The noise floor is a denominator, not a footnote

DJ-128 established, on the archived Phase 15 corpus, that two configurationally
identical conditions disagreed on **1,015 of 2,352 collective decisions
(43.2%)**, with a 35.7 pp swing in Sell rate and a 14.2 pp swing in herding,
because the conditions ran in disjoint wall-clock windows across model-server
reloads.

Consequences, binding:

1. **No condition contrast smaller than the measured floor is interpretable.**
2. The floor is measured **in-design**, by a replicate cell, not reconstructed
   afterwards from an accident.
3. Conditions are **interleaved** at the decision level. Running one condition
   to completion and then the next confounds condition with time, and did.

### 2.6 No retrospective storytelling

Retained from document 01 §29. The manuscript must not present the final
hypothesis as though it were known at the start. The Phase 15 retraction, the
defect record (DJ-120 → DJ-133), and the evolution of the question are reported
in the order they occurred.

---

## 3. Scope

### 3.1 In scope for Paper I

The mechanism, measured **at the level of the decision record**: agent outputs,
their correlation structure, effective independence, herding, disagreement
entropy, and how these respond to controlled manipulation of model identity and
information access.

This is a deliberate and load-bearing choice: **the mechanism metrics require no
forward-return labels.** They are computed from what the agents said, not from
what the market did. Therefore the leakage, survivorship, purge/embargo and
corporate-action repairs catalogued as C1–C3 and C8 in the 2026-08-23 evaluation
are **not blocking for the primary endpoint**. They are blocking only for
label-dependent secondary endpoints (IC), which are reported as secondary and
carry the disclosure that their evaluation protocol is under repair.

This single observation removes several months of infrastructure from the
critical path without weakening the primary claim. It was the largest strategic
error in document 01, which placed the full leakage-repair programme upstream of
everything.

### 3.2 In scope, as findings rather than housekeeping

The defect record is data. Two items in particular are results, not apologies:

- **Silent degradation renders as conviction.** Under DJ-120, agents deprived of
  evidence emitted maximum-confidence decisions (`Sell` @ 1.0) rather than
  abstaining. A collective whose failure mode is indistinguishable from signal is
  a finding about machine collectives, and it generalises far beyond finance.
- **Free-running agents under evidence-free prompts.** DJ-128 showed the three
  agents whose only evidence source had failed were the three that were
  irreproducible across runs (byte-identical output rates of 0.18, 0.02, 0.01)
  while the two holding real text were reproducible (1.00, 0.97). Near-tied
  logits let serving-stack drift decide the label. This is the mechanism of the
  noise floor and belongs in the results.

### 3.3 Out of scope for Paper I

- Any economic performance claim. Returns, Sharpe, drawdown and the equity curves
  belong to **Paper II** (finance venue, Phase 23), which cites Paper I for
  method. The live sample cannot answer *"did it make money?"* at the sample size
  available, and asking a question we cannot power is how good projects die.
- Live-arm results beyond a descriptive protocol section.
- Open-source release (Phase 23).
- Regime conditioning, multi-horizon analysis, aggregation-mechanism comparison,
  synthetic-market experiments beyond the mechanism-isolation role assigned in
  the SAP. These are legitimate and are **deferred**, not abandoned.

### 3.4 Registered Report route

Investigate before drafting §4 of the manuscript. Several venues peer-review the
design before data collection and grant in-principle acceptance independent of
outcome. The OSF pre-registration is most of a Stage 1 submission, and this route
makes the confirmatory/exploratory boundary a matter of record rather than of
our own assertion. If a complexity venue offers it, take it.

---

## 4. Falsification

The charter requires that the conditions for abandoning the claim be written
before the claim is tested. The SAP fixes the numeric thresholds; the charter
fixes what kind of evidence counts.

**H-PRIMARY is rejected if**, under the pre-specified test, the information
manipulation and the model manipulation move effective decision independence by
statistically indistinguishable amounts, or if the model manipulation moves it
by more.

**H-PRIMARY is unsupported (not rejected) if** the design lacks the power to
distinguish those two effects at the pre-specified minimum effect size. This
outcome is declared as an underpowered study, reported as such, and does not
become an exploratory positive.

**The study is void if** any negative-control gate in SAP §6 fires.

We commit in advance to reporting each of these outcomes with the same prominence
as a positive result.

---

## 5. Working method

### 5.1 Adversary before advocate

Before any result is written into the manuscript, the strongest objection a
hostile reviewer would raise against it is drafted first, in writing, in the
analysis log. The results text then answers that objection or concedes it. This
is a procedure with an artefact, not an attitude.

The standing objection catalogue — asked of every major claim, per document 01
§16 — is: more agents rather than better agents; one dominant agent; leakage;
survivorship; shared information; omitted transaction costs; model selection;
favourable regime; weak baseline; evaluation set used during development.

### 5.2 Every implementation task names its uncertainty

Retained from document 01 §27. No task enters the plan without answering *what
scientific uncertainty does this remove?* "The David specifies seven agents" is
not an answer. "We cannot distinguish information diversity from model diversity
without a partitioned-information condition" is.

### 5.3 The engineering is subordinate

The manuscript is not a catalogue of technologies. MCP, LangChain, LangGraph,
LanceDB, GraphRAG, LM Studio, Docker, LoRA, Langfuse and specific model
identifiers appear only where they bear on the scientific question. The
scientific abstraction is:

> heterogeneous decision-makers × information sets × independence × aggregation
> × environment

The implementation is evidence that these conditions can be instantiated and
controlled. It is not the contribution.

### 5.4 Citation discipline

Retained from document 01 §23. No invented citations; no paper cited for a
convenient sentence; every entry resolves to a DOI or a stable identifier before
it enters `refs.bib`. Two attributions were already corrected during
benchmarking (FinRL never pivoted to LLMs; TradingAgents is not an ICLR paper) —
a third such error found by a reviewer ends the submission.

Prose must keep visibly separate: established literature · our measurement ·
our interpretation · our hypothesis · implementation detail · speculation.

### 5.5 The decision journal continues

The bitácora and DJ numbering continue through the writing of this paper.
Hypotheses, failed analyses, methodological changes, retractions and negative
results are recorded as they happen. History is never rewritten.

---

## 6. What would make this paper strong

Not: *"our AI trading system beat the market."* Fragile, and uninteresting even
if true.

The target contribution, at the conceptual level to aim for:

> **Controlled evidence that the collective-intelligence advantage in LLM
> populations is governed by information partitioning rather than model
> heterogeneity, that the attainable independence is bounded by a measurable
> substrate floor, and a protocol — pre-registered, replicate-controlled,
> interleaved — under which such claims can be made honestly in a domain where
> almost none currently are.**

If the evidence supports a different or more interesting conclusion, we follow
the evidence and say when we changed our minds.

---

## 7. Ratification

This charter takes effect when Alberto Espinosa ratifies it and the SAP v1.0 is
frozen. From that point, changes to §§1–4 require a dated amendment recorded in
the bitácora and, where they touch a registered element, on OSF.

- [ ] Charter ratified — date:
- [ ] SAP v1.0 frozen — date:
- [ ] OSF amendment 002 filed (primary endpoint re-specification) — date:
