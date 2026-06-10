"""
Verification layer output schemas (P5-E1).

These schemas define the typed output contract between Phase 5 (which measures
hallucination) and Phase 9 (which acts on the measurements in the aggregation
function). Every field is derived from or consistent with the raw verification
results stored in AgentVerificationReport.results.

Design decisions
----------------
NumericalClaim
    One extracted claim from a rationale: the raw alias text found in the
    response, the canonical MCP field name it maps to (or None if the alias
    table does not recognise it), the numeric value, and a context snippet for
    human audit. Produced by the extractor (P5-E2).

VerificationResult
    One claim verified against the agent's tool results. Status is one of
    three values: "verified" (matched within tolerance), "hallucinated"
    (field found in tool results but value does not match), or "unresolvable"
    (field absent from tool results, or canonical_field is None). The
    "unresolvable" status is not a hallucination signal -- it records
    a measurement gap, not a factual error.

AgentVerificationReport
    All claims for one agent on one ticker, with aggregate metrics. Metrics
    (hallucination_rate, grounding_rate, flag_high_hr) are auto-computed from
    the raw results list by model_validator so the schema is always internally
    consistent whether constructed from code or deserialised from JSON.

    hallucination_rate excludes unresolvable claims from the denominator:
        HR = n_hallucinated / (n_claims - n_unresolvable)
    An agent citing fields not in the alias table is not hallucinating --
    it is using language the extractor does not recognise.

    grounding_rate measures audit-trail quality independently of factual
    accuracy: the fraction of verified claims for which the relevant call_id
    appears in signal.call_ids. An agent can have HR=0 (all correct) and
    GR=0 (no call_ids cited), revealing a structural gap in the audit trail.

EnsembleVerificationReport
    Wraps both agent reports with cross-agent contradictions and ensemble-
    level metrics. triggered_by_disagreement records whether disagreement
    entropy > 0 at the time of verification -- the disagreement trigger hook
    for Phase 9's performance-sensitive path.

_HR_FLAG_THRESHOLD
    Provisional 0.25 threshold (DJ-021). Revised after the Phase 5 baseline
    run (P5-E6-T3) once the empirical HR distribution is known.

David reference: §13 (Verification and Hallucination Control), §4.3
(Verifiability).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# Threshold constant (DJ-021 -- revised at P5-E6-T3)
# ---------------------------------------------------------------------------

_HR_FLAG_THRESHOLD: float = 0.25
"""Hallucination-rate threshold above which flag_high_hr is set to True.

Provisional value: 0.25 (more than 25 % of verifiable claims are fabricated).
This will be revised after the Phase 5 baseline run reveals the actual HR
distribution across the Phase 3 and Phase 4 baselines (DJ-021).
"""


# ---------------------------------------------------------------------------
# P5-E1-T1: NumericalClaim
# ---------------------------------------------------------------------------


class NumericalClaim(BaseModel):
    """
    One numerical claim extracted from an agent rationale (P5-E2).

    field_alias is the exact alias string found in the rationale text
    (e.g. "RSI", "P/E", "Sharpe ratio"). canonical_field is the MCP field
    name it maps to via the FIELD_ALIAS_TABLE, or None if the alias is not
    in the table. value is the parsed float. context_snippet is the
    surrounding text (approximately +/-40 characters) for human audit.
    """

    field_alias: str
    canonical_field: str | None
    value: float
    context_snippet: str


# ---------------------------------------------------------------------------
# P5-E1-T1: VerificationResult
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """
    One claim checked against the agent's tool results (P5-E3).

    Status semantics
    ----------------
    "verified"
        canonical_field was found in the tool results with a non-None value,
        and the claimed value matches the tool value within tolerance.
    "hallucinated"
        canonical_field was found in the tool results with a non-None value,
        but the claimed value does NOT match within tolerance. The agent
        fabricated or misrepresented a value it had access to.
    "unresolvable"
        Either canonical_field is None (alias not in the lookup table), or
        the field was absent from all tool results, or its tool value was
        None (data unavailable). Unresolvable claims do not contribute to
        hallucination_rate -- they represent measurement gaps.

    call_id_cited
        True if the call_id of the tool result that contains canonical_field
        appears in signal.call_ids. An agent can be fully verified (HR=0)
        yet have call_id_cited=False for all claims, indicating it produced
        correct numbers without citing the audit trail.

    tolerance_used
        The tolerance applied for this specific check. 0.0 for unresolvable
        claims (no tolerance check was performed).
    """

    claim: NumericalClaim
    status: Literal["verified", "hallucinated", "unresolvable"]
    tool_value: float | None
    tool_field: str | None
    call_id_cited: bool
    tolerance_used: float


# ---------------------------------------------------------------------------
# P5-E1-T1: Contradiction
# ---------------------------------------------------------------------------


class Contradiction(BaseModel):
    """
    A field cited by both agents with values that differ beyond tolerance.

    With Phase 4's orthogonal information domains (Fundamental Agent sees
    fundamentals/macro; Technical Agent sees price-derived data only),
    contradictions on the same field are structurally rare. They become
    routine at Phase 8 when agents share overlapping data access.

    A Contradiction is produced by detect_contradictions() (P5-E4) when
    both agents cited the same canonical_field with incompatible values.
    Both claims may be verified, hallucinated, or mixed -- the Contradiction
    records the divergence regardless of individual claim status.

    field : str
        The canonical MCP field name on which both agents disagreed.
    fundamental_claim : NumericalClaim
        The Fundamental Agent's claim about this field.
    technical_claim : NumericalClaim
        The Technical Agent's claim about this field.
    """

    field: str
    fundamental_claim: NumericalClaim
    technical_claim: NumericalClaim


# ---------------------------------------------------------------------------
# P5-E1-T2: AgentVerificationReport
# ---------------------------------------------------------------------------


class AgentVerificationReport(BaseModel):
    """
    All verified/hallucinated/unresolvable claims for one agent on one ticker.

    Derived metrics (n_claims, n_verified, n_hallucinated, n_unresolvable,
    hallucination_rate, grounding_rate, flag_high_hr) are auto-computed from
    results by model_validator in mode="after". They are stored as explicit
    fields so the report is fully serialisable to JSON.

    hallucination_rate formula
    --------------------------
        HR = n_hallucinated / (n_claims - n_unresolvable)
        Returns 0.0 if (n_claims - n_unresolvable) == 0 (nothing resolvable).

    grounding_rate formula
    ----------------------
        GR = count(verified AND call_id_cited) / n_verified
        Returns 0.0 if n_verified == 0 (no verified claims to ground).
    """

    ticker: str
    as_of_date: str
    agent_type: str
    prompt_version: str
    results: list[VerificationResult]

    # Derived -- auto-computed by model_validator; provide defaults for
    # Pydantic field initialisation ordering.
    n_claims: int = 0
    n_verified: int = 0
    n_hallucinated: int = 0
    n_unresolvable: int = 0
    hallucination_rate: float = 0.0
    grounding_rate: float = 0.0
    flag_high_hr: bool = False

    @model_validator(mode="after")
    def _compute_metrics(self) -> AgentVerificationReport:
        n_v = sum(1 for r in self.results if r.status == "verified")
        n_h = sum(1 for r in self.results if r.status == "hallucinated")
        n_u = sum(1 for r in self.results if r.status == "unresolvable")
        n = len(self.results)

        resolvable = n - n_u
        hr = n_h / resolvable if resolvable > 0 else 0.0
        gr = (
            sum(1 for r in self.results if r.status == "verified" and r.call_id_cited)
            / n_v
            if n_v > 0
            else 0.0
        )

        self.n_claims = n
        self.n_verified = n_v
        self.n_hallucinated = n_h
        self.n_unresolvable = n_u
        self.hallucination_rate = round(hr, 6)
        self.grounding_rate = round(gr, 6)
        self.flag_high_hr = hr > _HR_FLAG_THRESHOLD

        return self


# ---------------------------------------------------------------------------
# P5-E1-T3: EnsembleVerificationReport
# ---------------------------------------------------------------------------


class EnsembleVerificationReport(BaseModel):
    """
    Verification report for both agents on one ticker (P5-E5).

    Wraps fundamental_report and technical_report with cross-agent
    contradiction detection results and ensemble-level metrics. Derived
    fields (n_contradictions, total_claims, total_hallucinated,
    ensemble_hallucination_rate) are auto-computed by model_validator.

    triggered_by_disagreement
        True when ensemble_decision.disagreement_entropy > 0 at the time
        verify_ensemble() was called. Records whether this verification was
        initiated by the disagreement trigger (Phase 9 optimisation hook).

    ensemble_hallucination_rate
        Combined HR across both agents:
            EHR = (fund_n_hallucinated + tech_n_hallucinated)
                  / (fund_resolvable + tech_resolvable)
        where resolvable = n_claims - n_unresolvable for each agent.
        Returns 0.0 if total resolvable is 0.
    """

    ticker: str
    as_of_date: str
    fundamental_report: AgentVerificationReport
    technical_report: AgentVerificationReport
    contradictions: list[Contradiction]
    triggered_by_disagreement: bool

    # Derived -- auto-computed by model_validator.
    n_contradictions: int = 0
    total_claims: int = 0
    total_hallucinated: int = 0
    ensemble_hallucination_rate: float = 0.0

    @model_validator(mode="after")
    def _compute_ensemble_metrics(self) -> EnsembleVerificationReport:
        fr = self.fundamental_report
        tr = self.technical_report

        f_resolvable = fr.n_claims - fr.n_unresolvable
        t_resolvable = tr.n_claims - tr.n_unresolvable
        total_resolvable = f_resolvable + t_resolvable
        total_hallucinated = fr.n_hallucinated + tr.n_hallucinated

        ehr = total_hallucinated / total_resolvable if total_resolvable > 0 else 0.0

        self.n_contradictions = len(self.contradictions)
        self.total_claims = fr.n_claims + tr.n_claims
        self.total_hallucinated = total_hallucinated
        self.ensemble_hallucination_rate = round(ehr, 6)

        return self
