"""
Verification functions for the HiFi verification layer (P5-E3, P5-E4, P5-E5).

Four public functions, each building on the previous:

    verify_claim()      -- P5-E3: check one NumericalClaim against tool results
    verify_agent()      -- P5-E3: verify all claims in one agent's rationale
    detect_contradictions() -- P5-E4: find fields cited by both agents with
                               incompatible values
    verify_ensemble()   -- P5-E5: wire both agents + contradiction detection
                           into an EnsembleVerificationReport

Tolerance conventions (DJ-020)
--------------------------------
Values above 1.0 use 1 % relative tolerance:
    tolerance = 0.01 * abs(tool_value)
Values at or below 1.0 use an absolute tolerance of 0.01:
    tolerance = 0.01
This matches the Phase 3 approximation and represents typical two-decimal-
place citation precision (e.g. Sharpe of 0.82, beta of 0.95). Revised after
the Phase 5 baseline run (DJ-020, P5-E3-T3).

call_id attribution
-------------------
Each tool result dict returned by the MCP server contains a "call_id" key
(12-char SHA-256 prefix of the serialised tool inputs). verify_claim checks
whether the call_id of the sub-dict that contains the canonical_field appears
in signal.call_ids. An agent that cites correct numbers but omits call_ids
gets grounding_rate=0.0 -- the audit trail hook is missing even if the facts
are right.

Unresolvable vs hallucinated
----------------------------
A claim is "unresolvable" when:
  - canonical_field is None (alias not in FIELD_ALIAS_TABLE), OR
  - the field is absent from all tool result dicts, OR
  - the field exists in a tool result dict but its value is None (data was
    unavailable at computation time).
A claim is "hallucinated" only when the field exists with a non-None value
and the claimed value differs from the tool value beyond tolerance. Penalising
agents for citing fields that were unavailable would be a measurement error.
"""

from __future__ import annotations

import difflib
import re

from hifi.agents.schemas import (
    FundamentalAnalysis,
    MacroAnalysis,
    RiskAnalysis,
    SentimentAnalysis,
    TechnicalAnalysis,
)
from hifi.collective.schemas import EnsembleOutput
from hifi.verification.extractor import extract_numerical_claims
from hifi.verification.schemas import (
    AgentVerificationReport,
    Contradiction,
    EnsembleVerificationReport,
    NumericalClaim,
    SentimentGroundingReport,
    SentimentGroundingResult,
    VerificationResult,
)

# Sentinel: distinguishes "key absent" from "key present with None value".
# Both map to "unresolvable", but tool_field attribution differs (see below).
_MISSING = object()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _named_tool_results(
    analysis: FundamentalAnalysis | TechnicalAnalysis | RiskAnalysis | MacroAnalysis,
) -> list[tuple[str, dict]]:
    """
    Return (tool_name, result_dict) pairs for all MCP tool results in analysis.

    The tool_name is the attribute name on the analysis object (e.g.
    "financial_ratios", "risk_metrics"). This name is stored in
    VerificationResult.tool_field when a claim is resolved from that dict.
    """
    if isinstance(analysis, FundamentalAnalysis):
        return [
            ("financial_ratios", analysis.financial_ratios),
            ("growth_metrics", analysis.growth_metrics),
            ("valuation_context", analysis.valuation_context),
            ("macro_snapshot", analysis.macro_snapshot),
        ]
    if isinstance(analysis, RiskAnalysis):
        return [("risk_metrics", analysis.risk_metrics)]
    if isinstance(analysis, MacroAnalysis):
        return [("macro_snapshot", analysis.macro_snapshot)]
    # TechnicalAnalysis
    return [
        ("technical_indicators", analysis.technical_indicators),
        ("risk_metrics", analysis.risk_metrics),
    ]


def _tolerance(tool_value: float) -> float:
    """Return the tolerance for a given tool value (DJ-020)."""
    if abs(tool_value) > 1.0:
        return 0.01 * abs(tool_value)  # 1 % relative
    return 0.01  # absolute


# ---------------------------------------------------------------------------
# P5-E3-T1: verify_claim
# ---------------------------------------------------------------------------


def verify_claim(
    claim: NumericalClaim,
    named_tool_results: list[tuple[str, dict]],
    signal_call_ids: list[str],
) -> VerificationResult:
    """
    Check one NumericalClaim against a list of named tool result dicts.

    Parameters
    ----------
    claim : NumericalClaim
        The claim to verify (produced by extract_numerical_claims).
    named_tool_results : list[tuple[str, dict]]
        (tool_name, result_dict) pairs from _named_tool_results(). Each
        result_dict may contain a "call_id" key alongside financial fields.
    signal_call_ids : list[str]
        The call_ids stored in AgentSignal.call_ids for this analysis.

    Returns
    -------
    VerificationResult
        status="unresolvable" if canonical_field is None or the field is
        absent/None in all tool results. status="verified" or "hallucinated"
        depending on the tolerance check.
    """
    # Step 1: unresolvable if no canonical field mapping.
    if claim.canonical_field is None:
        return VerificationResult(
            claim=claim,
            status="unresolvable",
            tool_value=None,
            tool_field=None,
            call_id_cited=False,
            tolerance_used=0.0,
        )

    # Step 2: search tool result dicts for the canonical field.
    found_value = _MISSING
    found_tool_name: str | None = None
    found_call_id: str | None = None

    for tool_name, result_dict in named_tool_results:
        raw = result_dict.get(claim.canonical_field, _MISSING)
        if raw is _MISSING:
            continue
        # Field key exists. Record tool attribution even if value is None.
        found_tool_name = tool_name
        found_call_id = result_dict.get("call_id")
        if raw is not None:
            found_value = raw
        break

    # Step 3: unresolvable if field absent or value None.
    if found_value is _MISSING or found_value is None:
        return VerificationResult(
            claim=claim,
            status="unresolvable",
            tool_value=None,
            tool_field=found_tool_name,
            call_id_cited=False,
            tolerance_used=0.0,
        )

    # Step 4: tolerance check.
    try:
        numeric_tool_value = float(found_value)
    except (TypeError, ValueError):
        return VerificationResult(
            claim=claim,
            status="unresolvable",
            tool_value=None,
            tool_field=found_tool_name,
            call_id_cited=False,
            tolerance_used=0.0,
        )

    tol = _tolerance(numeric_tool_value)
    matches = abs(claim.value - numeric_tool_value) <= tol
    status = "verified" if matches else "hallucinated"

    # Step 5: call_id attribution.
    call_id_cited = (
        found_call_id is not None and found_call_id in signal_call_ids
    )

    return VerificationResult(
        claim=claim,
        status=status,
        tool_value=numeric_tool_value,
        tool_field=found_tool_name,
        call_id_cited=call_id_cited,
        tolerance_used=tol,
    )


# ---------------------------------------------------------------------------
# P5-E3: verify_agent
# ---------------------------------------------------------------------------


def verify_agent(
    analysis: FundamentalAnalysis | TechnicalAnalysis | RiskAnalysis | MacroAnalysis,
) -> AgentVerificationReport:
    """
    Verify all claims in one agent's rationale against its tool results.

    Parameters
    ----------
    analysis : FundamentalAnalysis | TechnicalAnalysis | RiskAnalysis | MacroAnalysis
        The full agent analysis object. Must expose .signal and per-tool
        result dicts (see _named_tool_results).

    Returns
    -------
    AgentVerificationReport
        Empty report (n_claims=0) when analysis.signal is None (the agent
        failed to produce a signal; there is no rationale to verify).
        Otherwise, all extracted claims are verified and metrics computed.
    """
    if isinstance(analysis, FundamentalAnalysis):
        agent_type = "fundamental"
    elif isinstance(analysis, TechnicalAnalysis):
        agent_type = "technical"
    elif isinstance(analysis, RiskAnalysis):
        agent_type = "risk"
    else:  # MacroAnalysis
        agent_type = "macro"
    prompt_version = analysis.prompt_version

    ticker = ""
    as_of_date = ""

    if analysis.signal is not None:
        ticker = analysis.signal.ticker
        as_of_date = analysis.signal.as_of_date

    # Empty report when no signal (no rationale to verify).
    if analysis.signal is None:
        return AgentVerificationReport(
            ticker=ticker,
            as_of_date=as_of_date,
            agent_type=agent_type,
            prompt_version=prompt_version,
            results=[],
        )

    signal = analysis.signal
    # MacroAnalysis has a separate analysis-level rationale field in addition to
    # signal.rationale; extract claims from both to maximise field coverage.
    rationale_text = signal.rationale
    if isinstance(analysis, MacroAnalysis) and analysis.rationale:
        rationale_text = signal.rationale + " " + analysis.rationale
    claims = extract_numerical_claims(rationale_text)
    named_results = _named_tool_results(analysis)

    verification_results = [
        verify_claim(claim, named_results, signal.call_ids)
        for claim in claims
    ]

    return AgentVerificationReport(
        ticker=ticker,
        as_of_date=as_of_date,
        agent_type=agent_type,
        prompt_version=prompt_version,
        results=verification_results,
    )


# ---------------------------------------------------------------------------
# P13-E0-T4 / Phase-14 calibration (DJ-072): verify_sentiment_agent helpers
# ---------------------------------------------------------------------------

_OUTER_QUOTE_RE = re.compile(r'^["\u201c\u201d\']+|["\u201c\u201d\']+$')
_NBSP_RE = re.compile(r'[\xa0\u2009\u202f\u200b]')
_SGR_LCS_THRESHOLD = 0.85  # LCS must cover ≥ 85% of signal (Phase 14 calibration)


def _normalise_sgr(text: str) -> str:
    """
    Normalise a signal or context string for SGR matching.

    Phase 14 calibration (DJ-072): exact-substring matching was too strict for
    models that (a) use non-breaking spaces, (b) wrap verbatim quotes in outer
    quotation marks, or (c) add a short prefix word ("RSU"). This normaliser
    handles cases (a) and (b); the LCS fallback in _is_sgr_grounded handles (c).
    """
    text = _NBSP_RE.sub(" ", text)   # collapse non-breaking / narrow spaces
    text = " ".join(text.split())    # normalise multiple spaces
    text = text.lower().strip()
    text = _OUTER_QUOTE_RE.sub("", text).strip()  # strip outer quote chars
    return text


def _is_sgr_grounded(signal: str, context: str) -> bool:
    """
    Two-stage grounding check for one notable_signal (Phase 14 calibration, DJ-072).

    Stage 1 — exact normalised substring: handles non-breaking spaces and outer
    quote characters added by models when citing verbatim.

    Stage 2 — LCS fallback: if Stage 1 fails, compute the longest common
    substring (character-level, via SequenceMatcher) between normalised signal
    and normalised context. A ratio ≥ _SGR_LCS_THRESHOLD (0.85) of the signal
    length counts as grounded, catching short model-added prefixes such as
    "RSU awards..." where the context has "Awards...".
    """
    sig = _normalise_sgr(signal)
    ctx = _normalise_sgr(context)
    if not sig:
        return False
    if sig in ctx:
        return True
    matcher = difflib.SequenceMatcher(None, sig, ctx, autojunk=False)
    match = matcher.find_longest_match(0, len(sig), 0, len(ctx))
    return (match.size / len(sig)) >= _SGR_LCS_THRESHOLD


# ---------------------------------------------------------------------------
# P13-E0-T4: verify_sentiment_agent
# ---------------------------------------------------------------------------


def verify_sentiment_agent(
    analysis: SentimentAnalysis,
    retrieved_context: str,
) -> SentimentGroundingReport:
    """
    Measure Sentiment Grounding Rate (SGR) for one SentimentAnalysis.

    SentimentAnalysis has no numerical MCP tools — it operates on RAG-retrieved
    text only. SGR replaces GR for the Sentiment agent: it measures whether
    notable_signals items are verbatim substrings of retrieved_context.

    Algorithm (DJ-072): normalise both signal texts and context to lowercase
    and strip whitespace; check substring containment. Edit-distance tolerance
    is deferred to Phase 14 calibration.

    Parameters
    ----------
    analysis : SentimentAnalysis
        Full Sentiment agent output including notable_signals list.
    retrieved_context : str
        The RAG-retrieved text the agent had access to when producing its
        signal. This is the ground-truth evidence for grounding checks.

    Returns
    -------
    SentimentGroundingReport
        Empty report (n_signals=0, grounding_rate=0.0) when analysis.signal
        is None. grounding_rate=0.0 when notable_signals is empty or
        retrieved_context is empty.
    """
    ticker = ""
    as_of_date = ""
    if analysis.signal is not None:
        ticker = analysis.signal.ticker
        as_of_date = analysis.signal.as_of_date

    if analysis.signal is None:
        return SentimentGroundingReport(
            ticker=ticker,
            as_of_date=as_of_date,
            n_signals=0,
            n_grounded=0,
            grounding_rate=0.0,
            results=[],
        )

    results: list[SentimentGroundingResult] = []
    for signal_text in analysis.notable_signals:
        grounded = _is_sgr_grounded(signal_text, retrieved_context)
        results.append(
            SentimentGroundingResult(
                signal_text=signal_text,
                grounded=grounded,
                matched_chunk=_normalise_sgr(signal_text) if grounded else None,
            )
        )

    n_signals = len(results)
    n_grounded = sum(1 for r in results if r.grounded)
    grounding_rate = n_grounded / n_signals if n_signals > 0 else 0.0

    return SentimentGroundingReport(
        ticker=ticker,
        as_of_date=as_of_date,
        n_signals=n_signals,
        n_grounded=n_grounded,
        grounding_rate=round(grounding_rate, 6),
        results=results,
    )


# ---------------------------------------------------------------------------
# P5-E4: detect_contradictions
# ---------------------------------------------------------------------------


def detect_contradictions(
    fundamental_report: AgentVerificationReport,
    technical_report: AgentVerificationReport,
) -> list[Contradiction]:
    """
    Find fields cited by both agents with values that differ beyond tolerance.

    Only verified and hallucinated claims are considered (unresolvable claims
    have no confirmed tool value to compare). Tolerance uses the same DJ-020
    convention as verify_claim.

    With Phase 4's orthogonal information domains, the expected result is an
    empty list. Contradictions become relevant at Phase 8 when agents share
    overlapping data access.

    Parameters
    ----------
    fundamental_report : AgentVerificationReport
        From verify_agent on a FundamentalAnalysis.
    technical_report : AgentVerificationReport
        From verify_agent on a TechnicalAnalysis.

    Returns
    -------
    list[Contradiction]
        One Contradiction per field where both agents made resolvable claims
        with values outside tolerance of each other.
    """
    # Collect resolvable claims (verified or hallucinated) per agent.
    fund_claims: dict[str, NumericalClaim] = {}
    for r in fundamental_report.results:
        if r.status != "unresolvable" and r.claim.canonical_field is not None:
            fund_claims[r.claim.canonical_field] = r.claim

    tech_claims: dict[str, NumericalClaim] = {}
    for r in technical_report.results:
        if r.status != "unresolvable" and r.claim.canonical_field is not None:
            tech_claims[r.claim.canonical_field] = r.claim

    # Find shared fields and check for value divergence.
    contradictions: list[Contradiction] = []
    for field in set(fund_claims) & set(tech_claims):
        fc = fund_claims[field]
        tc = tech_claims[field]

        # Use the larger of the two values to set tolerance.
        larger = max(abs(fc.value), abs(tc.value))
        tol = 0.01 * larger if larger > 1.0 else 0.01

        if abs(fc.value - tc.value) > tol:
            contradictions.append(
                Contradiction(
                    field=field,
                    fundamental_claim=fc,
                    technical_claim=tc,
                )
            )

    return contradictions


# ---------------------------------------------------------------------------
# P5-E5: verify_ensemble
# ---------------------------------------------------------------------------


def verify_ensemble(
    output: EnsembleOutput,
    always_verify: bool = True,
    sentiment_context: str | None = None,
) -> EnsembleVerificationReport:
    """
    Verify both agents and detect contradictions for one EnsembleOutput.

    Parameters
    ----------
    output : EnsembleOutput
        The full Phase 4 ensemble output for one ticker.
    always_verify : bool
        If True (default for Phase 5): always run verification regardless of
        disagreement entropy. If False: only run when entropy > 0 (Phase 9
        performance optimisation hook). In Phase 5 all outputs are verified
        to establish the HR/GR baseline; the flag exists for Phase 9's
        inference-cost-sensitive path.
    sentiment_context : str | None
        RAG-retrieved text the Sentiment agent had access to. When provided
        and output.sentiment_analysis is not None, verify_sentiment_agent()
        is called and the result stored in sentiment_report (P13-E0-T5).

    Returns
    -------
    EnsembleVerificationReport
        Full report with both agent reports and contradiction list.
        triggered_by_disagreement records whether the disagreement trigger
        condition (entropy > 0) was active at verification time.
    """
    triggered = output.ensemble_decision.disagreement_entropy > 0

    if not always_verify and not triggered:
        # Phase 9 path: both agents agree and verification was not requested.
        # Return empty reports rather than skipping -- callers always receive
        # a structurally valid EnsembleVerificationReport.
        fund_report = AgentVerificationReport(
            ticker=output.ticker,
            as_of_date=output.as_of_date,
            agent_type="fundamental",
            prompt_version=output.fundamental_analysis.prompt_version,
            results=[],
        )
        tech_report = AgentVerificationReport(
            ticker=output.ticker,
            as_of_date=output.as_of_date,
            agent_type="technical",
            prompt_version=output.technical_analysis.prompt_version,
            results=[],
        )
        return EnsembleVerificationReport(
            ticker=output.ticker,
            as_of_date=output.as_of_date,
            fundamental_report=fund_report,
            technical_report=tech_report,
            contradictions=[],
            triggered_by_disagreement=triggered,
        )

    fund_report = verify_agent(output.fundamental_analysis)
    tech_report = verify_agent(output.technical_analysis)
    contradictions = detect_contradictions(fund_report, tech_report)

    sentiment_report = None
    if sentiment_context is not None and output.sentiment_analysis is not None:
        sentiment_report = verify_sentiment_agent(
            output.sentiment_analysis, sentiment_context
        )

    return EnsembleVerificationReport(
        ticker=output.ticker,
        as_of_date=output.as_of_date,
        fundamental_report=fund_report,
        technical_report=tech_report,
        contradictions=contradictions,
        triggered_by_disagreement=triggered,
        sentiment_report=sentiment_report,
    )
