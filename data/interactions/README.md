# Dataset Family E — Agent Interaction Records

**Phase:** 13 (P13-E7, DJ-079)
**Status:** Populated from Phase 12 factorial evaluation (120 EnsembleOutput records)
**Schema version:** EnsembleOutput v1 (hifi.collective.schemas.EnsembleOutput)

---

## Overview

Dataset Family E captures every multi-agent interaction: individual agent signals,
aggregated ensemble decisions, debate transcripts, and voting method comparisons.
It is the primary record of how the ensemble behaves across conditions and dates.

Per David §8.5–8.6 and DJ-079, this family provides:
- Raw material for OQ-D03 (debate participation rate)
- Training context for adaptive aggregation (Phase 14+)
- Replication artifacts for the WQU capstone and publication

---

## Directory Structure

```
data/interactions/
├── README.md            — this file
└── debate/              — debate transcripts (DebateTranscript schema, conditions C/D)
    └── (populated by Phase 13 E2 evaluation runs)
```

The Phase 12 factorial EnsembleOutput records (conditions A–D, 120 files) live in
`data/evaluation/phase12/` and are the primary Dataset Family E source. This
directory (`data/interactions/`) will hold additional interaction records generated
in Phase 13 and beyond.

---

## Schema: EnsembleOutput

Each file is a JSON-serialised `hifi.collective.schemas.EnsembleOutput` object.

```
{
  "ticker":              str,           # e.g. "AAPL"
  "as_of_date":          str,           # ISO 8601, e.g. "2020-03-31"
  "fundamental_analysis": FundamentalAnalysis | null,
  "technical_analysis":   TechnicalAnalysis | null,
  "risk_analysis":        RiskAnalysis | null,
  "macro_analysis":       MacroAnalysis | null,
  "sentiment_analysis":   SentimentAnalysis | null,
  "contrarian_analysis":  ContrarianAnalysis | null,
  "ensemble_decision": {
    "collective_decision":    "Buy" | "Hold" | "Sell",
    "collective_confidence":  float,            # [0, 1]
    "disagreement_entropy":   float,            # Shannon entropy [0, log2(3)]
    "vote_counts":            {"Buy": int, "Hold": int, "Sell": int},
    "herding_coefficient":    float             # [0, 1]
  },
  "signals":             list[AgentSignal],    # valid (non-null) voting signals
  "aggregation_method":  str,                  # "confidence_weighted" (default)
  "method_comparison":   dict[str, EnsembleDecision],  # all 4 methods
  "debate_transcript":   DebateTranscript | null,
  "latency_ms":          float
}
```

### AgentSignal schema

```
{
  "ticker":      str,
  "as_of_date":  str,
  "decision":    "Buy" | "Hold" | "Sell",
  "confidence":  float,   # [0, 1] — agent's self-assessed certainty
  "rationale":   str,     # 2-4 sentences citing specific data
  "key_concern": str,     # primary risk the agent identifies
  "data_gaps":   list[str],
  "call_ids":    list[str],  # SHA-256 prefix IDs of MCP tool calls
  "model_id":    str,        # LM Studio model identifier
  "agent_type":  str         # "fundamental"|"technical"|"risk"|"macro"|"sentiment"
}
```

### DebateTranscript schema (conditions C and D only)

```
{
  "ticker":            str,
  "as_of_date":        str,
  "round_number":      int,         # 1-indexed
  "initial_signals":   list[AgentSignal],
  "revised_signals":   list[AgentSignal],
  "turns":             list[DebateTurn],
  "vote_delta":        "converged" | "diverged" | "unchanged",
  "debate_skipped":    bool         # True when initial vote is unanimous
}
```

---

## Dataset Family E: Factorial Conditions (Phase 12)

| Condition | Config | Files | Location |
|---|---|---|---|
| A | Base models, no debate | 30 | data/evaluation/phase12/A_*.json |
| B | Fine-tuned models, no debate | 30 | data/evaluation/phase12/B_*.json |
| C | Base models, debate | 30 | data/evaluation/phase12/C_*.json |
| D | Fine-tuned models, debate | 30 | data/evaluation/phase12/D_*.json |

**Tickers:** AAPL, JPM, XOM
**Dates:** 10 quarterly dates 2020-03-31 through 2022-06-30
**Total records:** 120 EnsembleOutput files
**Summary:** data/evaluation/phase12/factorial_summary.json

---

## Dataset Family E: Future Additions

| Source | Expected Phase | Location |
|---|---|---|
| Memory evaluation (OQ-M03) | Phase 13 | data/interactions/ |
| Multi-round debate transcripts (OQ-D04) | Phase 13 | data/interactions/debate/ |
| Paper trading interactions | Phase 14 | data/interactions/live/ |

---

## Lineage

- **Phase 9:** Collective decision engine — EnsembleOutput schema defined
- **Phase 10:** 30-date bootstrap evaluation — performance_history records
- **Phase 12:** 120-record factorial — first systematic Dataset Family E population
- **Phase 13:** Memory + multi-round debate records (this phase)
- **Phase 16:** Public release with dataset cards (Hugging Face)
