"""Unit tests for the cross-agent contradiction detector (P5-E4-T1 through T5)."""

from __future__ import annotations

import pytest

from hifi.verification.schemas import (
    AgentVerificationReport,
    NumericalClaim,
    VerificationResult,
)
from hifi.verification.verifier import detect_contradictions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claim(
    alias: str,
    canonical: str | None,
    value: float,
) -> NumericalClaim:
    return NumericalClaim(
        field_alias=alias,
        canonical_field=canonical,
        value=value,
        context_snippet="...",
    )


def _verified_result(alias: str, canonical: str, value: float) -> VerificationResult:
    return VerificationResult(
        claim=_claim(alias, canonical, value),
        status="verified",
        tool_value=value,
        tool_field="some_tool",
        call_id_cited=False,
        tolerance_used=0.01,
    )


def _hallucinated_result(
    alias: str, canonical: str, claimed_value: float, tool_value: float
) -> VerificationResult:
    return VerificationResult(
        claim=_claim(alias, canonical, claimed_value),
        status="hallucinated",
        tool_value=tool_value,
        tool_field="some_tool",
        call_id_cited=False,
        tolerance_used=0.01,
    )


def _unresolvable_result(alias: str) -> VerificationResult:
    return VerificationResult(
        claim=_claim(alias, None, 99.0),
        status="unresolvable",
        tool_value=None,
        tool_field=None,
        call_id_cited=False,
        tolerance_used=0.0,
    )


def _make_report(results: list[VerificationResult], agent_type: str) -> AgentVerificationReport:
    return AgentVerificationReport(
        ticker="AAPL",
        as_of_date="2023-03-31",
        agent_type=agent_type,
        prompt_version="v1",
        results=results,
    )


# ---------------------------------------------------------------------------
# P5-E4-T2: Same field, matching values -> no contradiction
# ---------------------------------------------------------------------------


def test_no_contradiction_when_values_match():
    """Both agents cite pe=28.3 -> no contradiction."""
    fund = _make_report(
        [_verified_result("P/E", "pe", 28.3)],
        "fundamental",
    )
    tech = _make_report(
        [_verified_result("P/E", "pe", 28.3)],
        "technical",
    )
    contradictions = detect_contradictions(fund, tech)
    assert contradictions == []


def test_no_contradiction_within_tolerance():
    """Values differ by <1% for field >1.0 -> no contradiction."""
    # tool_value=28.3; agent values differ by 0.1 % -> within tolerance
    fund = _make_report(
        [_verified_result("P/E", "pe", 28.30)],
        "fundamental",
    )
    tech = _make_report(
        [_verified_result("P/E", "pe", 28.30 * 1.005)],  # 0.5% off
        "technical",
    )
    contradictions = detect_contradictions(fund, tech)
    assert contradictions == []


# ---------------------------------------------------------------------------
# P5-E4-T3: Same field, differing values -> Contradiction created
# ---------------------------------------------------------------------------


def test_contradiction_when_values_differ():
    """Both agents cite rsi but with values 42.1 vs 78.0 -> contradiction."""
    fund = _make_report(
        [_verified_result("RSI", "rsi", 42.1)],
        "fundamental",
    )
    tech = _make_report(
        [_verified_result("RSI", "rsi", 78.0)],
        "technical",
    )
    contradictions = detect_contradictions(fund, tech)
    assert len(contradictions) == 1
    c = contradictions[0]
    assert c.field == "rsi"
    assert c.fundamental_claim.value == pytest.approx(42.1)
    assert c.technical_claim.value == pytest.approx(78.0)


def test_contradiction_large_value_outside_relative_tolerance():
    """pe=28.3 vs pe=50.0 -> |28.3-50.0|=21.7 >> 0.01*50=0.5 -> contradiction."""
    fund = _make_report(
        [_verified_result("P/E", "pe", 28.3)],
        "fundamental",
    )
    tech = _make_report(
        [_verified_result("P/E", "pe", 50.0)],
        "technical",
    )
    contradictions = detect_contradictions(fund, tech)
    assert len(contradictions) == 1


def test_multiple_contradictions():
    """Two shared fields both differ -> two Contradiction objects."""
    fund = _make_report(
        [
            _verified_result("P/E", "pe", 28.3),
            _verified_result("RSI", "rsi", 42.1),
        ],
        "fundamental",
    )
    tech = _make_report(
        [
            _verified_result("P/E", "pe", 55.0),
            _verified_result("RSI", "rsi", 78.0),
        ],
        "technical",
    )
    contradictions = detect_contradictions(fund, tech)
    assert len(contradictions) == 2
    fields = {c.field for c in contradictions}
    assert fields == {"pe", "rsi"}


# ---------------------------------------------------------------------------
# P5-E4-T4: Orthogonal fields -> empty contradiction list
# ---------------------------------------------------------------------------


def test_no_contradiction_orthogonal_fields():
    """Fund cites pe; tech cites rsi. No shared fields -> no contradictions."""
    fund = _make_report(
        [_verified_result("P/E", "pe", 28.3)],
        "fundamental",
    )
    tech = _make_report(
        [_verified_result("RSI", "rsi", 42.1)],
        "technical",
    )
    contradictions = detect_contradictions(fund, tech)
    assert contradictions == []


def test_empty_reports_no_contradictions():
    fund = _make_report([], "fundamental")
    tech = _make_report([], "technical")
    assert detect_contradictions(fund, tech) == []


# ---------------------------------------------------------------------------
# P5-E4-T5: Domain crossing -- technical agent cites pe (hallucinated)
# ---------------------------------------------------------------------------


def test_domain_crossing_different_value_is_contradiction():
    """
    Technical agent cites pe=50.0 (hallucinated -- not in its tool results).
    Fundamental agent cites pe=28.3 (verified).
    Both are resolvable claims (verified/hallucinated) -> contradiction detected.
    """
    fund = _make_report(
        [_verified_result("P/E", "pe", 28.3)],
        "fundamental",
    )
    tech = _make_report(
        [_hallucinated_result("P/E", "pe", 50.0, tool_value=0.0)],
        "technical",
    )
    contradictions = detect_contradictions(fund, tech)
    assert len(contradictions) == 1
    assert contradictions[0].field == "pe"


def test_domain_crossing_same_value_no_contradiction():
    """
    Tech agent hallucinated pe=28.3 (matching fund pe=28.3).
    Values match within tolerance -> no contradiction even though it's a hallucination.
    """
    fund = _make_report(
        [_verified_result("P/E", "pe", 28.3)],
        "fundamental",
    )
    tech = _make_report(
        [_hallucinated_result("P/E", "pe", 28.3, tool_value=0.0)],
        "technical",
    )
    contradictions = detect_contradictions(fund, tech)
    assert contradictions == []


# ---------------------------------------------------------------------------
# Unresolvable claims are excluded from contradiction detection
# ---------------------------------------------------------------------------


def test_unresolvable_claims_excluded():
    """Unresolvable claims on same field do not produce a contradiction."""
    fund = _make_report(
        [_unresolvable_result("mystery")],
        "fundamental",
    )
    tech = _make_report(
        [_unresolvable_result("mystery")],
        "technical",
    )
    assert detect_contradictions(fund, tech) == []
