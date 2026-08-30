# Phase 22: Paper I — Ablation Analysis and Capstone Deliverable
## Context and Pre-Phase Decisions

**Gathered:** 2026-08-27
**Status:** PLANNED — starts in parallel with Phase 21, does not wait for it
**Supersedes:** `HIFI_PROTOCOL_V1.md` Phase 17 ("Ablation Studies + Capstone
Deliverable"), never started, no plan file. See `PHASE_21_CONTEXT.md` §6 for the
numbering remap.
**Origin:** Owner decision 2026-08-27 after benchmarking HiFi against a WQU
capstone exemplar, `HKUDS/AI-Trader` (arXiv:2512.10971) and the AI4Finance
product line.

---

## 1. What the benchmarking established

Three comparators were examined against the question *"are we at the level to
publish in a well-ranked journal?"*

| Artifact | What it is | Peer review |
|---|---|---|
| WQU capstone exemplar | student degree deliverable | none |
| HKUDS/AI-Trader | live LLM trading benchmark, 3 markets, 6 LLMs | **arXiv preprint only** |
| AI4Finance line (FinRL / FinGPT / FinRobot) | 45k+ aggregate stars | workshops and symposia; FinRobot arXiv-only |

**None is in a well-ranked journal.** The premise of the question was false, and
that is the useful finding: matching this field's level yields a preprint and a
workshop slot. A journal requires a different axis.

Two corrections to earlier internal assessments, recorded so they are not
repeated:

1. **Live execution is not a HiFi differentiator.** AI-Trader is explicitly *"the
   first fully-automated, live, and data-uncontaminated evaluation benchmark"*.
   An earlier claim in this project's favour was wrong and must not appear in the
   manuscript.
2. **Scope is a HiFi weakness.** AI-Trader covers three markets and six models;
   HiFi covers one market and one model family.

## 2. What HiFi actually has that they do not

Every comparator is a **leaderboard**: it varies the model, or the library, or
the architecture, and reports which earned more. None manipulates a variable
derived from a theory; none has a control condition; none pre-registers.

HiFi holds the model fixed and manipulates the **organisation of the agents**,
against a live equal-weight null (arm C), testing a prediction imported from
outside finance (Page's diversity-prediction theorem).

That is an experiment. The others are demonstrations. The distinction is the
paper's entire contribution and must be the opening move of the manuscript, not
a paragraph in the discussion.

Supporting assets no comparator holds: a pre-registration with amendments; a
documented defect record (DJ-120 → DJ-132); and a **self-retracted headline
result** (Phase 15), which is the strongest available evidence that this
programme's positive claims can be trusted.

## 3. Venue decision

**Primary: a complexity-science venue** (JASSS, or JEDC if the framing leans
economic).

The reasoning is not preference, it is power. A finance journal asks *did it make
money?* — a question the live sample cannot answer at n ≈ 125 correlated daily
observations across four arms. A complexity venue asks *does the mechanism behave
as the theory predicts?* — measured **per decision**, which is the sample HiFi
actually has, and the question Page's theorem actually speaks to.

Secondary target if the mechanism result is strong: ACM ICAIF, this field's home
venue and FinRL's own.

**Registered Report route to investigate first.** Some journals peer-review the
design before data collection and grant in-principle acceptance independent of
outcome (PLOS ONE, Royal Society Open Science, Nature Human Behaviour run this
format; complexity and finance venues to be checked). HiFi's OSF pre-registration
is most of a Stage 1 submission, and this route would make Phase 21's DJ-132
timing concern moot by construction. Investigate before drafting §4.

## 4. Two papers, not one paper twice

The same study cannot be published twice, and rewriting the literature review
does not make a second paper. The split must be by contribution:

- **Paper I (this phase, complexity venue).** The protocol, the diversity
  measurements, the Phase 15 retraction and the defect taxonomy.
  Contribution: *how to run this experiment honestly, and what the mechanism
  does.*
- **Paper II (Phase 23, finance venue).** The live economic outcome, citing
  Paper I for method. Contribution: *what happened when we ran it.*

Paper I must be declared when Paper II is submitted.

## 5. The decisive scheduling fact

**Paper I does not need the live experiment.** Herding κ, disagreement entropy,
and the topology → diversity relationship are computed from decision records, not
from equity curves. The Phase 15 corpus already holds 9,505 agent records over
2,352 (date, ticker) pairs.

Live trading is required only for the economic claim, which belongs to Paper II.
Paper I's results section can therefore be produced from data already collected,
once the Phase 15 re-run is redesigned per DJ-128.

This is why Phase 22 starts now and runs in parallel with Phase 21 rather than
after it.

## 6. The gap being closed

`doc/HIFI_DAVID.md` §20 lists 17 references — Page, Surowiecki, Woolley, Nemeth,
Schwenk, Holland, Arthur — and **zero LLM-trading citations**. A grep of `doc/`
and `plans/` returns **no DOIs at all**. FinRL, TradingAgents, AI-Trader, FinGPT,
FinMem appear nowhere in the repository.

The related-work section does not exist, and it is the section a reviewer reads
first to decide whether the authors know the field.

## 7. Standing risks to name in the manuscript, not hide

- Single market, single model family, one cycle per day.
- Effective sample size far below nominal under cross-sectional correlation —
  the Phase 15 error, which must be corrected in the analysis and disclosed.
- Arm A's genesis-window record carries the DJ-131 cash drag; the contaminated
  window is bounded and must be stated.
- Local model provenance: qwen2.5's pretraining cutoff is a leakage concern for
  any historical evaluation, though live forward decisions from 2026-08-24 are
  leakage-immune by construction. Paper I must keep this distinction explicit —
  it is the field's live controversy (TradingAgents was challenged on exactly
  this, and LiveTradeBench found static-benchmark winners underperform live).

## 8. Non-goals

- No results from the live arms in Paper I beyond a descriptive protocol section.
- No performance claim of any kind until DJ-132's endpoint is filed and the
  contaminated windows are excluded.
- No open-source release (Phase 23).
