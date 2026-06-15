"""Unit tests for MacroAnalysis verification (P13-E0-T2)."""

from __future__ import annotations

import pytest

from hifi.agents.schemas import AgentSignal, MacroAnalysis
from hifi.verification.verifier import verify_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_macro_analysis(
    signal_rationale: str = "Macro backdrop neutral.",
    analysis_rationale: str = "Standard macro rationale.",
    call_ids: list[str] | None = None,
    macro_snapshot: dict | None = None,
) -> MacroAnalysis:
    signal = AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Hold",
        confidence=0.60,
        rationale=signal_rationale,
        key_concern="Inflation risk.",
        call_ids=call_ids or [],
        model_id="test-macro-model",
        agent_type="macro",
    )
    return MacroAnalysis(
        signal=signal,
        regime_assessment="Tightening cycle.",
        rationale=analysis_rationale,
        macro_snapshot=macro_snapshot
        or {
            "fed_funds_rate": 4.75,
            "cpi_yoy": 5.0,
            "unemployment_rate": 3.5,
            "vix": 18.5,
            "yield_10y": 3.96,
            "yield_2y": 4.60,
            "yield_curve_slope": -0.64,
            "gdp_growth": 2.1,
            "call_id": "macro001",
        },
        prompt_version="macro_v1",
    )


# ---------------------------------------------------------------------------
# E0-T2: MacroAnalysis returns AgentVerificationReport with agent_type="macro"
# ---------------------------------------------------------------------------


def test_verify_agent_macro_returns_report():
    analysis = _make_macro_analysis()
    report = verify_agent(analysis)
    assert report.agent_type == "macro"
    assert report.ticker == "AAPL"
    assert report.as_of_date == "2023-03-31"
    assert report.prompt_version == "macro_v1"


def test_verify_agent_macro_signal_none_empty_report():
    analysis = MacroAnalysis(
        signal=None,
        regime_assessment="Unknown.",
        rationale="No data.",
        macro_snapshot={},
        prompt_version="macro_v1",
    )
    report = verify_agent(analysis)
    assert report.agent_type == "macro"
    assert report.n_claims == 0
    assert report.results == []


# ---------------------------------------------------------------------------
# E0-T2: Field resolution for known macro_snapshot keys
# ---------------------------------------------------------------------------


def test_verify_agent_macro_fed_funds_verified():
    """fed funds rate of 4.75 in signal rationale -> verified."""
    analysis = _make_macro_analysis(
        signal_rationale="Fed funds rate of 4.75 is restrictive.",
        call_ids=["macro001"],
        macro_snapshot={"fed_funds_rate": 4.75, "call_id": "macro001"},
    )
    report = verify_agent(analysis)
    verified = [
        r
        for r in report.results
        if r.status == "verified" and r.claim.canonical_field == "fed_funds_rate"
    ]
    assert len(verified) >= 1
    assert verified[0].call_id_cited is True
    assert verified[0].tool_field == "macro_snapshot"


def test_verify_agent_macro_vix_verified():
    """VIX of 18.5 in rationale -> verified against vix field."""
    analysis = _make_macro_analysis(
        signal_rationale="VIX of 18.5 indicates moderate fear.",
        call_ids=["macro001"],
        macro_snapshot={"vix": 18.5, "call_id": "macro001"},
    )
    report = verify_agent(analysis)
    verified = [
        r
        for r in report.results
        if r.status == "verified" and r.claim.canonical_field == "vix"
    ]
    assert len(verified) >= 1


def test_verify_agent_macro_analysis_rationale_claims_extracted():
    """Claims in analysis.rationale (not signal.rationale) are also extracted."""
    # Signal rationale is generic; VIX claim is in analysis.rationale only.
    analysis = _make_macro_analysis(
        signal_rationale="Macro backdrop neutral.",
        analysis_rationale="VIX of 18.5 suggests calm markets.",
        call_ids=["macro001"],
        macro_snapshot={"vix": 18.5, "call_id": "macro001"},
    )
    report = verify_agent(analysis)
    vix_results = [r for r in report.results if r.claim.canonical_field == "vix"]
    assert len(vix_results) >= 1
    assert vix_results[0].status == "verified"


def test_verify_agent_macro_dual_rationale_deduplication():
    """Same claim in both rationales appears twice — both verified."""
    analysis = _make_macro_analysis(
        signal_rationale="VIX of 18.5 in signal.",
        analysis_rationale="VIX of 18.5 in analysis.",
        call_ids=["macro001"],
        macro_snapshot={"vix": 18.5, "call_id": "macro001"},
    )
    report = verify_agent(analysis)
    vix_results = [r for r in report.results if r.claim.canonical_field == "vix"]
    # Two VIX claims extracted (one per rationale text), both verified
    assert len(vix_results) == 2
    assert all(r.status == "verified" for r in vix_results)


def test_verify_agent_macro_hallucinated_cpi():
    """CPI of 99.0 but tool has cpi_yoy=5.0 -> hallucinated."""
    analysis = _make_macro_analysis(
        signal_rationale="CPI of 99.0 is extreme.",
        call_ids=["macro001"],
        macro_snapshot={"cpi_yoy": 5.0, "call_id": "macro001"},
    )
    report = verify_agent(analysis)
    hallucinated = [r for r in report.results if r.status == "hallucinated"]
    assert len(hallucinated) >= 1


def test_verify_agent_macro_zero_hr():
    """Fed funds and VIX both correct -> hallucination_rate=0.0."""
    analysis = _make_macro_analysis(
        signal_rationale="Fed funds rate of 4.75 and VIX of 18.5 are key indicators.",
        analysis_rationale="Standard analysis.",
        call_ids=["macro001"],
        macro_snapshot={
            "fed_funds_rate": 4.75,
            "vix": 18.5,
            "call_id": "macro001",
        },
    )
    report = verify_agent(analysis)
    assert report.hallucination_rate == pytest.approx(0.0)
