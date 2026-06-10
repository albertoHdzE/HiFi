# Phase 5: Verification Layer

**Status:** PLANNED

| Epic | Title | Status |
|---|---|---|
| P5-E1 | Verification schemas (interface-first) | PLANNED |
| P5-E2 | Claim extractor | PLANNED |
| P5-E3 | Agent-level verifier | PLANNED |
| P5-E4 | Cross-agent contradiction detector | PLANNED |
| P5-E5 | Ensemble verifier + disagreement trigger | PLANNED |
| P5-E6 | Baseline measurement + holistic test | PLANNED |

**David Sections:** §13 (Verification and Hallucination Control, full section), §4.3 (Verifiability)
**Learning Guide Topics:** 2.3 (Hallucination Detection and Mitigation — deep dive)
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 5

---

## Governing Philosophy for This Phase

Phase 5 is the epistemic foundation of HiFi.

Every previous phase has produced outputs that are structurally valid: agents return well-formed
JSON, decisions are one of Buy/Hold/Sell, confidence is in [0,1]. But structural validity says
nothing about truth. An agent can produce perfectly formatted JSON whose rationale is entirely
fabricated. At Phase 4's scale — two agents, three tickers, no forward-testing — this is an
acceptable research condition. At Phase 8's scale — six agents, continuous market coverage,
collective decisions that inform real portfolio construction — it is not.

Phase 5 answers the question that structural validity cannot: when an agent says "RSI of 48.0
indicates neutral momentum", is 48.0 actually what the RSI computation returned? When it says
"Sharpe of 0.82 is moderate", did the risk engine actually compute 0.82? If the answer is no
to either question, the rationale is partially fabricated regardless of how coherent it sounds.

This matters for two reasons beyond correctness. First, a fabricated number cited as evidence
for a collective decision corrupts the confidence-weighted vote: the agent is claiming higher
knowledge than it actually has. Second, cross-agent contradictions — where the Fundamental Agent
and Technical Agent cite conflicting numerical realities — are only detectable if individual
claims can be traced to specific tool outputs. Without verification, a contradiction is invisible.

**Why Phase 5 precedes Phases 6 and 7.** Observability (Phase 6) and RAG (Phase 7) both improve
the quality of agent outputs. But improving quality without being able to measure it is speculation.
Phase 5 establishes the measurement infrastructure — hallucination rate, grounding rate,
contradiction rate — that makes Phase 7 and Phase 8 improvements empirically verifiable. You
cannot measure improvement in trustworthiness without first defining what trustworthy means.

**The call_id design pays off here.** Phases 2 and 3 embedded a `call_id` (12-char SHA-256 prefix
of the serialised tool inputs) in every MCP tool result and instructed the agents to collect these
IDs in `AgentSignal.call_ids`. The cost of this design was near zero: one line per tool call, one
field in the schema. The payoff is exact traceability: every numerical claim in a rationale can be
linked to the specific tool execution that produced it. Phase 5 is that payoff.

**What Phase 5 is NOT:**

Phase 5 does not suppress agent signals based on hallucination rate. It does not penalise agents
or route around them. It measures. The output of Phase 5 is a `VerificationReport` alongside
every `AgentSignal` — a permanent audit trail that says exactly which claims were grounded, which
were hallucinated, and which could not be resolved. Suppression logic belongs to Phase 9 (Collective
Decision Engine), where verification status is one input to the aggregation function. Phase 5
generates the data; Phase 9 acts on it.

---

## Background: The Phase 3 Approximation and Why It Is Insufficient

Phase 3 built `baseline_metrics.count_hallucinated_numbers`. Its algorithm: extract every number
from the rationale; check whether that number appears (within 1% relative tolerance) in the flat
union of all tool results. If not, count it as a hallucination candidate.

This is wrong in three ways that Phase 5 corrects:

1. **No field linkage.** The Phase 3 checker asks "does 48.0 appear anywhere in the tool results?"
   If the rationale says "RSI of 48.0" but 48.0 also happens to appear as a `net_margin` value in
   the financial ratios, the Phase 3 checker marks it as verified — even though the claim about RSI
   was never checked at all. Phase 5 links each claim to a specific field and a specific tool.

2. **No directionality.** The Phase 3 checker is symmetric: it treats "P/E of 28.3" and "P/E of
   99.9" identically as long as one of them appears somewhere in the tool results. Phase 5 checks
   the claimed value against the correct field's actual value.

3. **No cross-agent coverage.** The Phase 3 checker operates on FundamentalAnalysis only.
   Phase 5 operates on both agents and can detect when the Technical Agent cites a field it was
   never given access to (a hallucination of fundamental data) or when the Fundamental Agent
   cites a price-derived metric it never received (a hallucination of technical data).

The Phase 3 baseline result — 0 hallucinated numbers, 0 data gaps acknowledged — should be read as
"0 numbers appeared in the rationale that were entirely absent from the tool results" and nothing
stronger. Phase 5 will likely reveal a different picture when field-specific verification is applied.

---

## Key Decisions To Make in This Phase

**DJ-019: Claim extraction method — regex + alias table vs. LLM-based extractor**

Two architectures are possible:

1. **Regex + alias table:** Pattern-match the rationale for `(field_reference) of (numeric_value)`
   constructions. Map field references ("RSI", "P/E", "Sharpe", "fed funds rate") to canonical MCP
   field names via a lookup table. This approach is deterministic, testable with exact expected
   outputs, and has zero inference cost.

2. **LLM-based extractor:** Send the rationale to a small local model with an instruction to extract
   all numerical claims as structured JSON. This approach handles paraphrased references ("the
   14-day momentum indicator") that the alias table would miss.

**The case for regex at Phase 5:** The agents were designed with a specific citation format: prompts
instruct them to "cite specific values from the data (e.g., 'RSI of 42.1', 'ATR of 3.82', 'Sharpe
of 1.24')". The `technical_v1` and `fundamental_v1` prompts both use the `field_name of value`
pattern explicitly. Regex is the right tool when the text was generated against a known format
specification. An LLM extractor is needed when text is uncontrolled; HiFi's text is controlled
by design. Using an LLM to verify an LLM also introduces a second source of inference error in
the verification chain.

Decision to record as DJ-019 after P5-E2-T1 (once the alias table coverage is measured on the
Phase 3 and Phase 4 baselines). If coverage is below 80%, revisit the LLM extractor option.

**DJ-020: Verification tolerance for continuous values**

All MCP engine computations are deterministic: given the same Parquet inputs and the same date,
the risk engine always returns the same Sharpe ratio. This means verification tolerance is not
about computation variance — it is about representation variance (the agent rounds 0.8234 to "0.82"
in the rationale text).

Proposed: 1% relative tolerance for values above 1.0; absolute tolerance of 0.01 for values below
1.0. This matches the Phase 3 approximation and is consistent with how financial ratios are
typically cited (two decimal places).

Decision to record as DJ-020 after P5-E3-T3 (after running against actual baselines).

**DJ-021: Hallucination rate threshold for signal flagging**

Phase 5 measures; Phase 9 acts. But Phase 5 should define the threshold constants that Phase 9
will use, so they are established from empirical data rather than guessed at design time.

Proposed threshold: HR > 0.25 (more than 25% of verifiable claims are hallucinated) triggers a
warning flag in the VerificationReport. Phase 9 can use this flag as an input to the aggregation
function. The 0.25 threshold is provisional — it will be revised after the Phase 5 baseline run
reveals the actual distribution.

Decision to record as DJ-021 after P5-E6-T1 (after baseline verification run).

---

## Epic P5-E1: Verification Schemas

**Objective:** Define the typed output contract for the verification layer before writing any
extraction or checking logic. The schemas are the interface between Phase 5 (which produces
verification reports) and Phase 9 (which consumes them in the aggregation function).

**NumericalClaim — one extracted claim from a rationale:**

```python
class NumericalClaim(BaseModel):
    field_alias: str          # exact string found in text ("RSI", "P/E", "Sharpe")
    canonical_field: str | None  # mapped MCP field name; None if alias not in table
    value: float              # numeric value as parsed from text
    context_snippet: str      # surrounding text (±30 chars) for human audit
```

**VerificationResult — one claim checked against tool results:**

```python
class VerificationResult(BaseModel):
    claim: NumericalClaim
    status: Literal["verified", "hallucinated", "unresolvable"]
    # verified: canonical_field found in tool results, value matches within tolerance
    # hallucinated: canonical_field found in tool results, value does NOT match
    # unresolvable: canonical_field is None (alias not in lookup table)
    tool_value: float | None      # actual value from tool results (None if unresolvable)
    tool_field: str | None        # which tool result contained the match
    call_id_cited: bool           # was the relevant call_id in signal.call_ids?
    tolerance_used: float         # tolerance applied for this check
```

Note: a claim is only `hallucinated` if the field was found in the tool results but the value
does not match. If the field is absent from tool results (not computed), status is `unresolvable`.
This distinction matters: an agent claiming "GDP growth of 2.1%" when GDP data was unavailable is
unresolvable (the tool returned None), not hallucinated. Phase 5 does not penalise agents for
citing unavailable data — Phase 3's `data_gap_acknowledged` already handles that case.

**Contradiction — a field cited by both agents with incompatible values:**

```python
class Contradiction(BaseModel):
    field: str               # canonical MCP field name
    fundamental_claim: NumericalClaim
    technical_claim: NumericalClaim
    # Both agents cited the same field with values that differ beyond tolerance.
    # With orthogonal information domains (Phase 4), this is rare; it becomes
    # relevant at Phase 8 when agents share overlapping data access.
```

**AgentVerificationReport — all claims for one agent on one ticker:**

```python
class AgentVerificationReport(BaseModel):
    ticker: str
    as_of_date: str
    agent_type: str               # "fundamental" | "technical"
    prompt_version: str
    n_claims: int                 # total claims extracted
    n_verified: int               # claims with status="verified"
    n_hallucinated: int           # claims with status="hallucinated"
    n_unresolvable: int           # claims with status="unresolvable"
    hallucination_rate: float     # n_hallucinated / (n_claims - n_unresolvable)
    grounding_rate: float         # fraction of verified claims with call_id_cited=True
    flag_high_hr: bool            # True if hallucination_rate > DJ-021 threshold
    results: list[VerificationResult]
```

`hallucination_rate` excludes unresolvable claims from the denominator. An agent citing
fields that are not in the lookup table is not hallucinating — it is using language
the extractor does not recognise. Penalising this would be a measurement error.

`grounding_rate` measures a subtler property: did the agent cite the call_id for the tool
whose output it used? An agent can be factually correct (hallucination_rate = 0) but poorly
grounded (it cited the right numbers but did not include the corresponding call_ids in its
signal). Grounding is the audit trail; hallucination is the factual accuracy.

**EnsembleVerificationReport — wraps both agents:**

```python
class EnsembleVerificationReport(BaseModel):
    ticker: str
    as_of_date: str
    fundamental_report: AgentVerificationReport
    technical_report: AgentVerificationReport
    n_contradictions: int
    contradictions: list[Contradiction]
    triggered_by_disagreement: bool  # True when ensemble entropy > 0 triggered this
    total_claims: int
    total_hallucinated: int
    ensemble_hallucination_rate: float   # across both agents
```

| Ticket | Description | Status |
|---|---|---|
| P5-E1-T1 | Define NumericalClaim, VerificationResult, Contradiction in src/hifi/verification/schemas.py | PLANNED |
| P5-E1-T2 | Define AgentVerificationReport with all metric fields | PLANNED |
| P5-E1-T3 | Define EnsembleVerificationReport | PLANNED |
| P5-E1-T4 | Unit tests: AgentVerificationReport computes hallucination_rate correctly (edge cases: 0 claims, all unresolvable, all verified, all hallucinated) | PLANNED |
| P5-E1-T5 | Unit tests: EnsembleVerificationReport serialises to JSON-safe dict | PLANNED |

**Files to create:**
- `src/hifi/verification/__init__.py`
- `src/hifi/verification/schemas.py`

---

## Epic P5-E2: Claim Extractor

**Objective:** Extract all numerical claims from a rationale string as a list of NumericalClaim
objects. The extractor must be deterministic, fully testable, and operate without a live model.

**Algorithm:**

1. Apply regex to find `(field_reference) of (numeric_value)` patterns. Primary pattern:
   `([A-Za-z][A-Za-z0-9_/\s\-]{1,30})\s+of\s+([-]?\d+\.?\d*)` (case-insensitive).
   Also match `(numeric_value)%` patterns preceded by a field reference.

2. For each match: record `field_alias` (raw text), `value` (parsed float), and
   `context_snippet` (surrounding ±40 characters).

3. Map `field_alias` through the `FIELD_ALIAS_TABLE` to `canonical_field`. If not found,
   `canonical_field = None`.

**FIELD_ALIAS_TABLE (initial version — extended as new patterns are observed):**

Fundamental domain:
- "P/E", "PE", "p/e ratio", "price-to-earnings" → `pe`
- "P/B", "PB", "price-to-book" → `pb`
- "P/S", "price-to-sales" → `ps`
- "ROE", "return on equity" → `roe`
- "ROA", "return on assets" → `roa`
- "debt/equity", "debt-to-equity", "D/E" → `debt_equity`
- "net margin" → `net_margin`
- "P/E percentile", "PE percentile" → `pe_1y_percentile`
- "fed funds", "federal funds", "FEDFUNDS" → `fed_funds_rate`
- "CPI", "inflation" → `cpi_yoy`

Technical domain:
- "RSI" → `rsi`
- "SMA" → `sma`
- "EMA" → `ema`
- "MACD", "macd" → `macd`
- "MACD signal", "signal line" → `macd_signal`
- "MACD histogram", "histogram" → `macd_hist`
- "Bollinger upper", "BB upper", "bb_upper" → `bb_upper`
- "Bollinger lower", "BB lower", "bb_lower" → `bb_lower`
- "ATR", "average true range" → `atr`
- "hist vol 20", "20-day vol", "20d vol" → `hist_vol_20d`
- "hist vol 60", "60-day vol" → `hist_vol_60d`
- "hist vol 252", "annual vol", "252d vol" → `hist_vol_252d`
- "beta" → `beta`
- "max drawdown", "drawdown" → `max_drawdown_252d`
- "Sharpe", "Sharpe ratio" → `sharpe_252d`
- "VaR", "value at risk" → `var_95_20d`

The table is a living artefact. Every time a baseline run reveals a new field reference pattern
not currently covered, the table is extended and the unresolvable count in the baseline report
is the measure of how complete it is.

**Coverage goal:** Unresolvable rate below 10% on the Phase 3 and Phase 4 baselines. If above
10%, the extractor is missing common patterns and the table needs extension before Phase 5
is considered complete.

| Ticket | Description | Status |
|---|---|---|
| P5-E2-T1 | Implement FIELD_ALIAS_TABLE and extract_numerical_claims() in src/hifi/verification/extractor.py | PLANNED |
| P5-E2-T2 | Unit test: "RSI of 48.0" → NumericalClaim(field_alias="RSI", canonical_field="rsi", value=48.0) | PLANNED |
| P5-E2-T3 | Unit test: "P/E of 28.3 is below" → pe, 28.3 | PLANNED |
| P5-E2-T4 | Unit test: "fed funds rate of 4.1" → fed_funds_rate, 4.1 | PLANNED |
| P5-E2-T5 | Unit test: unknown alias "momentum index of 3.4" → NumericalClaim with canonical_field=None | PLANNED |
| P5-E2-T6 | Unit test: multiple claims in one rationale string → correct list length and values | PLANNED |
| P5-E2-T7 | Unit test: rationale with no numerical claims → empty list, no exception | PLANNED |
| P5-E2-T8 | Measure alias table coverage on Phase 3 and Phase 4 baseline fixtures; record unresolvable rate; record DJ-019 | PLANNED |

**Files to create:**
- `src/hifi/verification/extractor.py`

---

## Epic P5-E3: Agent-Level Verifier

**Objective:** For each NumericalClaim produced by the extractor, look up the corresponding
value in the agent's tool results and determine whether the claim is verified, hallucinated,
or unresolvable. Compute per-agent metrics.

**Verification algorithm for one claim:**

1. If `claim.canonical_field` is None: status = "unresolvable". Done.
2. Search all tool results (passed as a flat dict via `analysis.tool_results_flat()`) for the
   `canonical_field` key.
3. If the field is absent or its value is None: status = "unresolvable" (data was unavailable
   when the tool ran; agent may or may not have hallucinated — cannot determine).
4. If the field exists with a non-None value: apply tolerance check.
   - Relative tolerance (1%) if `abs(tool_value) > 1.0`
   - Absolute tolerance (0.01) if `abs(tool_value) <= 1.0`
   - If within tolerance: status = "verified"
   - If outside tolerance: status = "hallucinated"
5. Set `tool_value`, `tool_field`, and `call_id_cited` (True if the call_id of the tool
   result containing this field appears in `signal.call_ids`).

**Tool result lookup.** The verifier operates on the flat merged tool results from
`analysis.tool_results_flat()`. This dict contains all field-value pairs from all tool calls.
The `call_id_cited` check requires knowing which tool each field came from — this requires
the unflattened tool results. The verifier receives the full analysis object, not just the
flat dict, to enable this attribution.

**verify_agent() function:**

```python
def verify_agent(
    analysis: FundamentalAnalysis | TechnicalAnalysis,
) -> AgentVerificationReport:
```

Steps:
1. If `analysis.signal` is None: return an empty report (no rationale to verify).
2. Extract claims from `analysis.signal.rationale`.
3. Verify each claim against `analysis.tool_results_flat()` and raw tool dicts.
4. Compute metrics: n_claims, n_verified, n_hallucinated, n_unresolvable, hallucination_rate,
   grounding_rate, flag_high_hr (DJ-021 threshold).
5. Return AgentVerificationReport.

| Ticket | Description | Status |
|---|---|---|
| P5-E3-T1 | Implement verify_claim() in src/hifi/verification/verifier.py | PLANNED |
| P5-E3-T2 | Unit test: claim matches tool result within 1% → status="verified", call_id_cited correct | PLANNED |
| P5-E3-T3 | Unit test: claim outside tolerance → status="hallucinated"; record DJ-020 | PLANNED |
| P5-E3-T4 | Unit test: claim for absent field → status="unresolvable" | PLANNED |
| P5-E3-T5 | Unit test: claim for None-valued field → status="unresolvable" (not hallucinated) | PLANNED |
| P5-E3-T6 | Unit test: verify_agent() with signal=None → empty report, no exception | PLANNED |
| P5-E3-T7 | Unit test: verify_agent() on a FundamentalAnalysis with injected known claim → correct AgentVerificationReport | PLANNED |
| P5-E3-T8 | Unit test: verify_agent() on a TechnicalAnalysis with injected known claim | PLANNED |
| P5-E3-T9 | Unit test: hallucination_rate = 0.0 when all claims verified | PLANNED |
| P5-E3-T10 | Unit test: grounding_rate = 0.0 when call_ids are empty even if all claims verified | PLANNED |

**Files to create:**
- `src/hifi/verification/verifier.py` (verify_claim, verify_agent functions)

---

## Epic P5-E4: Cross-Agent Contradiction Detector

**Objective:** Given verification reports for both agents, identify any fields where both agents
made a claim about the same field with incompatible values. Produce a list of Contradiction objects.

**Context.** With Phase 4's orthogonal information domains (Fundamental Agent receives only
fundamentals/macro, Technical Agent receives only price-derived data), true contradictions on the
same field are structurally rare. The Fundamental Agent cannot access RSI; the Technical Agent
cannot access P/E. However, two classes of contradiction ARE possible and important:

1. **Domain crossing.** The Technical Agent cites "P/E of 28.3" in its rationale. Since P/E was
   never in its tool results, this is a hallucination of fundamental data. The extractor will flag
   this as hallucinated (P/E not found in technical_indicators or risk_metrics). If the Fundamental
   Agent also cited P/E as 28.3, there is no contradiction — the technical agent simply hallucinated
   a number it had no access to. If the values differ, it reveals the technical agent invented a
   number without grounding.

2. **Shared fields via Phase 8 expansion.** At Phase 8, multiple agents will have overlapping data
   access (e.g., a Risk Agent and a Technical Agent both receive risk_metrics). Contradictions on
   shared fields then represent genuine disagreements about the same data, which may indicate a
   tool error or a parsing difference.

The contradiction detector is designed for Phase 8 utility but is scaffolded now to establish the
schema and measurement infrastructure before it is needed at full scale.

**Algorithm:**

1. Collect verified and hallucinated claims from both reports (exclude unresolvable).
2. Build a dict: `{canonical_field: [(agent_type, value), ...]}` across both agents.
3. For any field with entries from both agents: check if values differ beyond tolerance.
4. If they differ: create a Contradiction object.

| Ticket | Description | Status |
|---|---|---|
| P5-E4-T1 | Implement detect_contradictions() in src/hifi/verification/verifier.py | PLANNED |
| P5-E4-T2 | Unit test: same field, matching values → no contradiction | PLANNED |
| P5-E4-T3 | Unit test: same field, differing values → Contradiction object created | PLANNED |
| P5-E4-T4 | Unit test: orthogonal fields (no shared claims) → empty contradiction list | PLANNED |
| P5-E4-T5 | Unit test: technical agent cites pe (domain crossing hallucination) → hallucinated in technical report, contradiction if value differs from fundamental report | PLANNED |

---

## Epic P5-E5: Ensemble Verifier + Disagreement Trigger

**Objective:** Wire both agent verifiers into an ensemble-level function that produces an
EnsembleVerificationReport. Implement the disagreement trigger: when `ensemble_decision.disagreement_entropy > 0`,
verification is automatically performed and the result is included in the output.

**verify_ensemble() function:**

```python
def verify_ensemble(
    output: EnsembleOutput,
    always_verify: bool = False,
) -> EnsembleVerificationReport:
```

- If `always_verify=False`: only run when `output.ensemble_decision.disagreement_entropy > 0`
  (agents disagreed) or when either agent's signal is non-None. In practice, Phase 5 should
  always verify — the flag is for Phase 9's performance-sensitive path.
- Run `verify_agent` for both agents.
- Run `detect_contradictions` on both reports.
- Compute ensemble-level metrics.
- Set `triggered_by_disagreement` based on entropy.

**Design note on the trigger.** The disagreement trigger embeds the Phase 4 diversity insight
into Phase 5's architecture: when agents disagree, their factual claims are MORE likely to
conflict, and the confidence-weighted winner may be winning based on a hallucinated rationale.
Verification is most valuable precisely when the collective decision is most uncertain. A system
that verifies only on disagreement captures most of the risk at a fraction of the verification cost.
At Phase 5, we verify all outputs to establish baselines. At Phase 9, the trigger condition becomes
a performance optimisation.

| Ticket | Description | Status |
|---|---|---|
| P5-E5-T1 | Implement verify_ensemble() in src/hifi/verification/verifier.py | PLANNED |
| P5-E5-T2 | Unit test: verify_ensemble() with both agents non-None → EnsembleVerificationReport with two agent reports | PLANNED |
| P5-E5-T3 | Unit test: triggered_by_disagreement=True when entropy > 0 | PLANNED |
| P5-E5-T4 | Unit test: triggered_by_disagreement=False when entropy = 0 | PLANNED |
| P5-E5-T5 | Unit test: ensemble_hallucination_rate correct when agents have different individual rates | PLANNED |
| P5-E5-T6 | Unit test: EnsembleVerificationReport serialises to JSON-safe dict | PLANNED |

---

## Epic P5-E6: Baseline Measurement and Holistic Test

**Objective:** Run the verifier against the Phase 3 and Phase 4 baseline fixtures. Record the
first empirical hallucination rates, grounding rates, and unresolvable rates under the Phase 5
precision measurement. Establish these as the floor against which Phase 7 (RAG) improvements
will be measured.

**compute_verification_metrics() — aggregate over multiple reports:**

```python
def compute_verification_metrics(
    reports: dict[str, AgentVerificationReport],
) -> dict:
```

Returns: mean_hallucination_rate, mean_grounding_rate, mean_unresolvable_rate, n_reports,
total_claims, total_hallucinated, total_unresolvable, alias_table_coverage
(= 1 - mean_unresolvable_rate over resolvable + unresolvable).

**Baseline runner.** A script `scripts/run_phase5_verification.py` loads the existing
`phase4_ensemble.json` fixture (no live LLM required — verification operates on already-recorded
outputs) and produces `tests/fixtures/baseline/phase5_verification.json` with the structure:

```json
{
  "metadata": { "phase": "5", "verified_from": "phase4_ensemble.json", "run_date": "..." },
  "reports": {
    "AAPL": { EnsembleVerificationReport dict },
    "JPM":  { ... },
    "XOM":  { ... }
  },
  "metrics": {
    "fundamental": { mean_hr, mean_gr, ... },
    "technical":   { mean_hr, mean_gr, ... },
    "ensemble":    { ... }
  }
}
```

Unlike the Phase 3 and Phase 4 baseline scripts, Phase 5's baseline runner requires NO live
LLM instance. Verification operates entirely on already-produced outputs. The script can
be run at any time without LM Studio.

**Holistic test.** The test `tests/holistic/test_phase5_verification_pipeline.py` runs
`verify_ensemble` on a synthetic EnsembleOutput built from the existing Phase 4 parquet
fixtures and two monkeypatched LLMs. It asserts:

1. Both agent reports are produced with correct agent_type
2. A known-correct claim in the stubbed rationale is marked verified
3. A known-incorrect number in the stubbed rationale is marked hallucinated
4. A known-unknown alias in the stubbed rationale is marked unresolvable
5. EnsembleVerificationReport is JSON-safe
6. Phase 4 regression: run_ensemble still produces valid EnsembleOutput

| Ticket | Description | Status |
|---|---|---|
| P5-E6-T1 | Implement compute_verification_metrics() in src/hifi/verification/metrics.py | PLANNED |
| P5-E6-T2 | Write scripts/run_phase5_verification.py; loads phase4_ensemble.json; no LLM required | PLANNED |
| P5-E6-T3 | Run baseline; save phase5_verification.json; record DJ-019 alias coverage and DJ-021 threshold | PLANNED |
| P5-E6-T4 | Unit tests: test_phase5_baseline.py — skip if fixture absent; structure and threshold checks | PLANNED |
| P5-E6-T5 | Holistic test: verify_ensemble with stub LLMs + fixtures; known-correct claim verified; known-wrong claim hallucinated | PLANNED |
| P5-E6-T6 | Holistic test: Phase 4 regression — run_ensemble still produces valid EnsembleOutput | PLANNED |

**Files to create:**
- `src/hifi/verification/metrics.py`
- `scripts/run_phase5_verification.py`
- `tests/fixtures/baseline/phase5_verification.json` (generated by script)
- `tests/unit/test_verification_schemas.py`
- `tests/unit/test_claim_extractor.py`
- `tests/unit/test_verifier.py`
- `tests/unit/test_contradiction_detector.py`
- `tests/unit/test_phase5_baseline.py`
- `tests/integration/test_verify_agent.py`
- `tests/integration/test_verify_ensemble.py`
- `tests/holistic/test_phase5_verification_pipeline.py`

---

## Epic Dependency Graph

```
P5-E1 (Schemas)
    |
    +------------------+
    |                  |
P5-E2 (Extractor)  (existing Phase 4 output objects)
    |                  |
    v                  |
P5-E3 (Agent Verifier) <------+
    |
    v
P5-E4 (Contradiction Detector)
    |
    v
P5-E5 (Ensemble Verifier)
    |
    v
P5-E6 (Baseline + Holistic)
```

E1 is the interface contract for all subsequent epics. E2 and E4 are independent of each other
once E1 is done. E3 depends on E1 and E2. E4 depends on E3. E5 depends on E3 and E4. E6
depends on E5.

---

## New Dependencies

No new Python packages required for the core verification logic. All extractor and verifier code
uses stdlib (`re`, `math`) and the existing Pydantic schemas.

If DJ-019 is revisited and an LLM-based extractor is chosen, `langchain` is already available.

---

## Phase 5 Quality Gates

| Gate | Criterion | Measured By |
|---|---|---|
| All unit tests pass | pytest tests/unit/, 0 failures | Manual run |
| All integration tests pass | pytest tests/integration/, 0 failures | Manual run |
| Holistic test passes | pytest tests/holistic/test_phase5_verification_pipeline.py | Manual run |
| Phase 4 regression | pytest tests/holistic/test_phase4_ensemble_pipeline.py still passes | Manual run |
| Alias table coverage | Unresolvable rate < 10% on Phase 4 baseline | phase5_verification.json |
| Baseline documented | phase5_verification.json committed with HR, GR, unresolvable rates | Fixture file |
| DJ-019 recorded | Alias table coverage measured and decision documented | phase5_verification.json |
| DJ-021 recorded | HR threshold set from empirical distribution | phase5_verification.json |
| No live LLM required | Baseline script runs with no LM Studio | Manual check |
| Lint clean | ruff check src/ tests/ scripts/, 0 errors | Manual run |

---

## Commit Strategy

| Commit | Epic | Key Files |
|---|---|---|
| Phase 5 / E1: Verification schemas | P5-E1 | verification/schemas.py, tests/unit/test_verification_schemas.py |
| Phase 5 / E2: Claim extractor | P5-E2 | verification/extractor.py, tests/unit/test_claim_extractor.py |
| Phase 5 / E3: Agent verifier | P5-E3 | verification/verifier.py (verify_claim, verify_agent), tests/unit/test_verifier.py, tests/integration/test_verify_agent.py |
| Phase 5 / E4: Contradiction detector | P5-E4 | verification/verifier.py (detect_contradictions), tests/unit/test_contradiction_detector.py |
| Phase 5 / E5: Ensemble verifier | P5-E5 | verification/verifier.py (verify_ensemble), tests/integration/test_verify_ensemble.py |
| Phase 5 / E6: Baseline and holistic | P5-E6 | verification/metrics.py, scripts/run_phase5_verification.py, tests/holistic/test_phase5_verification_pipeline.py |

---

## Open Questions This Phase Will Answer

**OQ-P5-01: What is the actual hallucination rate of the Phase 3 and Phase 4 agents under
field-specific verification?**
The Phase 3 blunt checker reported 0 hallucinations. Phase 5's field-specific check will almost
certainly reveal a different number — either true hallucinations (wrong values cited) or domain
crossings (fields cited that were not in the agent's tool results). This is the first honest
measurement of agent factual reliability.

**OQ-P5-02: What fraction of rationale claims cannot be resolved by the alias table?**
Determines whether regex + alias table (DJ-019) is sufficient or whether an LLM extractor is
needed. If unresolvable rate is consistently above 10% across tickers, the agents are using
field references the alias table does not cover. This reveals whether the prompt's citation
format instruction is being followed.

**OQ-P5-03: Is the grounding rate (call_id_cited) a meaningful signal?**
An agent can produce correct numbers with 0% hallucination rate but cite none of the call_ids.
Does low grounding correlate with any other quality signal? Phase 5 measures both; Phase 10
will test whether grounding predicts directional accuracy.

**OQ-P5-04: Do the two Phase 4 agents ever contradict each other on a shared field?**
With orthogonal domains, the expected answer is no. Confirming zero contradictions validates
the information restriction design. If contradictions appear, it means one agent is hallucinating
values from the other's domain — a specific failure mode with a specific architectural fix.

---

## Connections to Earlier and Later Phases

**Depends on Phase 4:**
- EnsembleOutput and AgentVerificationReport share the analysis objects as input
- `call_ids` from every AgentSignal are the Phase 2 design hook that makes verification possible
- The disagreement_entropy trigger comes from EnsembleDecision (Phase 4)

**Phase 7 (RAG) depends on Phase 5:**
- RAG reduces hallucinations by grounding agents in retrieved knowledge documents
- Improvement is measured as reduction in hallucination_rate between Phase 5 baseline and
  Phase 7 post-RAG run, using identical verification infrastructure
- Without Phase 5's baseline, the effect of RAG cannot be quantified

**Phase 9 (Collective Decision Engine) depends on Phase 5:**
- flag_high_hr in AgentVerificationReport is an input to Phase 9's aggregation function
- An agent with HR > DJ-021 threshold has its confidence downweighted in the collective vote
- Contradiction detection feeds Phase 9's contrarian agent trigger

**Phase 10 (Evaluation) uses Phase 5's baselines:**
- phase5_verification.json is the hallucination floor for all quality improvement measurements
- The HR, GR, and unresolvable rates become standard columns in the evaluation table
- Phase 10 can correlate grounding_rate with directional accuracy to test whether
  better-grounded agents produce better financial predictions
