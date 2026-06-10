"""Unit tests for verification layer schemas (P5-E1-T4, P5-E1-T5)."""

from __future__ import annotations

import json

import pytest

from hifi.verification.schemas import (
    AgentVerificationReport,
    Contradiction,
    EnsembleVerificationReport,
    NumericalClaim,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claim(
    alias: str = "RSI", canonical: str | None = "rsi", value: float = 48.0
) -> NumericalClaim:
    return NumericalClaim(
        field_alias=alias,
        canonical_field=canonical,
        value=value,
        context_snippet="...RSI of 48.0 signals...",
    )


def _result(status: str, call_id_cited: bool = False) -> VerificationResult:
    return VerificationResult(
        claim=_claim(),
        status=status,  # type: ignore[arg-type]
        tool_value=48.0 if status == "verified" else None,
        tool_field="technical_indicators" if status != "unresolvable" else None,
        call_id_cited=call_id_cited,
        tolerance_used=0.01,
    )


def _make_report(
    results: list[VerificationResult], agent_type: str = "technical"
) -> AgentVerificationReport:
    return AgentVerificationReport(
        ticker="AAPL",
        as_of_date="2023-03-31",
        agent_type=agent_type,
        prompt_version="technical_v1",
        results=results,
    )


def _make_ensemble(
    fund_results: list[VerificationResult],
    tech_results: list[VerificationResult],
    triggered: bool = False,
) -> EnsembleVerificationReport:
    fr = _make_report(fund_results, "fundamental")
    tr = _make_report(tech_results, "technical")
    return EnsembleVerificationReport(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_report=fr,
        technical_report=tr,
        contradictions=[],
        triggered_by_disagreement=triggered,
    )


# ---------------------------------------------------------------------------
# NumericalClaim
# ---------------------------------------------------------------------------


def test_numerical_claim_basic():
    c = _claim()
    assert c.field_alias == "RSI"
    assert c.canonical_field == "rsi"
    assert c.value == 48.0


def test_numerical_claim_none_canonical():
    c = _claim(alias="mystery_factor", canonical=None)
    assert c.canonical_field is None


# ---------------------------------------------------------------------------
# AgentVerificationReport: hallucination_rate edge cases (P5-E1-T4)
# ---------------------------------------------------------------------------


def test_agent_report_zero_claims():
    r = _make_report([])
    assert r.n_claims == 0
    assert r.n_verified == 0
    assert r.n_hallucinated == 0
    assert r.n_unresolvable == 0
    assert r.hallucination_rate == 0.0
    assert r.grounding_rate == 0.0
    assert r.flag_high_hr is False


def test_agent_report_all_unresolvable():
    results = [_result("unresolvable"), _result("unresolvable"), _result("unresolvable")]
    r = _make_report(results)
    assert r.n_unresolvable == 3
    assert r.n_claims == 3
    # hallucination_rate: denominator = n_claims - n_unresolvable = 0 -> 0.0
    assert r.hallucination_rate == 0.0
    assert r.flag_high_hr is False


def test_agent_report_all_verified():
    results = [_result("verified"), _result("verified")]
    r = _make_report(results)
    assert r.n_verified == 2
    assert r.n_hallucinated == 0
    assert r.hallucination_rate == 0.0
    assert r.flag_high_hr is False


def test_agent_report_all_hallucinated():
    results = [_result("hallucinated"), _result("hallucinated")]
    r = _make_report(results)
    assert r.n_hallucinated == 2
    # hallucination_rate = 2 / (2 - 0) = 1.0
    assert r.hallucination_rate == pytest.approx(1.0)
    assert r.flag_high_hr is True


def test_agent_report_mixed():
    # 2 verified, 1 hallucinated, 1 unresolvable
    results = [
        _result("verified"),
        _result("verified"),
        _result("hallucinated"),
        _result("unresolvable"),
    ]
    r = _make_report(results)
    assert r.n_claims == 4
    assert r.n_verified == 2
    assert r.n_hallucinated == 1
    assert r.n_unresolvable == 1
    # resolvable = 4 - 1 = 3; HR = 1/3
    assert r.hallucination_rate == pytest.approx(1 / 3, rel=1e-5)
    # 1/3 = 0.333 > _HR_FLAG_THRESHOLD = 0.25 -> flag_high_hr=True
    assert r.flag_high_hr is True


def test_agent_report_hr_just_below_threshold():
    # Create exactly n_hallucinated/resolvable = 0.24 < _HR_FLAG_THRESHOLD = 0.25
    # 24 hallucinated, 76 verified -> HR = 24/100 = 0.24
    results = [_result("hallucinated")] * 24 + [_result("verified")] * 76
    r = _make_report(results)
    assert r.hallucination_rate == pytest.approx(0.24, rel=1e-5)
    assert r.flag_high_hr is False


def test_agent_report_hr_just_above_threshold():
    # 26 hallucinated, 74 verified -> HR = 26/100 = 0.26 > 0.25
    results = [_result("hallucinated")] * 26 + [_result("verified")] * 74
    r = _make_report(results)
    assert r.flag_high_hr is True


# ---------------------------------------------------------------------------
# AgentVerificationReport: grounding_rate
# ---------------------------------------------------------------------------


def test_agent_report_grounding_rate_all_cited():
    results = [_result("verified", call_id_cited=True)] * 3
    r = _make_report(results)
    assert r.grounding_rate == pytest.approx(1.0)


def test_agent_report_grounding_rate_none_cited():
    results = [_result("verified", call_id_cited=False)] * 3
    r = _make_report(results)
    # All verified but none cited -> grounding_rate = 0.0
    assert r.grounding_rate == pytest.approx(0.0)


def test_agent_report_grounding_rate_partial():
    results = [
        _result("verified", call_id_cited=True),
        _result("verified", call_id_cited=False),
    ]
    r = _make_report(results)
    assert r.grounding_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# EnsembleVerificationReport metrics (P5-E1-T5)
# ---------------------------------------------------------------------------


def test_ensemble_report_total_claims():
    fund_results = [_result("verified"), _result("hallucinated")]
    tech_results = [_result("unresolvable"), _result("verified")]
    ens = _make_ensemble(fund_results, tech_results)
    assert ens.total_claims == 4
    assert ens.total_hallucinated == 1


def test_ensemble_report_ensemble_hr():
    # fund: 1 hallucinated / 2 resolvable; tech: 0 hallucinated / 1 resolvable (1 unresolvable)
    fund_results = [_result("hallucinated"), _result("verified")]
    tech_results = [_result("verified"), _result("unresolvable")]
    ens = _make_ensemble(fund_results, tech_results)
    # total_hallucinated = 1; total_resolvable = 2 + 1 = 3
    assert ens.ensemble_hallucination_rate == pytest.approx(1 / 3, rel=1e-5)


def test_ensemble_report_zero_resolvable():
    fund_results = [_result("unresolvable")]
    tech_results = [_result("unresolvable")]
    ens = _make_ensemble(fund_results, tech_results)
    assert ens.ensemble_hallucination_rate == 0.0


def test_ensemble_report_n_contradictions():
    fr = _make_report([])
    tr = _make_report([])
    c = Contradiction(
        field="rsi",
        fundamental_claim=_claim("RSI", "rsi", 42.0),
        technical_claim=_claim("RSI", "rsi", 78.0),
    )
    ens = EnsembleVerificationReport(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_report=fr,
        technical_report=tr,
        contradictions=[c],
        triggered_by_disagreement=False,
    )
    assert ens.n_contradictions == 1


def test_ensemble_report_json_serialisable():
    ens = _make_ensemble(
        [_result("verified"), _result("hallucinated")],
        [_result("unresolvable")],
        triggered=True,
    )
    dumped = json.dumps(ens.model_dump())
    loaded = json.loads(dumped)
    assert loaded["ticker"] == "AAPL"
    assert loaded["triggered_by_disagreement"] is True
    assert "fundamental_report" in loaded
    assert "technical_report" in loaded


def test_ensemble_report_triggered_by_disagreement_stored():
    ens_true = _make_ensemble([], [], triggered=True)
    ens_false = _make_ensemble([], [], triggered=False)
    assert ens_true.triggered_by_disagreement is True
    assert ens_false.triggered_by_disagreement is False
