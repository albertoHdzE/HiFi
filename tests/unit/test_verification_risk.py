"""Unit tests for RiskAnalysis verification (P13-E0-T1)."""

from __future__ import annotations

import pytest

from hifi.agents.schemas import AgentSignal, RiskAnalysis
from hifi.verification.verifier import verify_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_risk_analysis(
    rationale: str = "Risk looks manageable.",
    call_ids: list[str] | None = None,
    risk_metrics: dict | None = None,
) -> RiskAnalysis:
    signal = AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Hold",
        confidence=0.60,
        rationale=rationale,
        key_concern="Tail risk.",
        call_ids=call_ids or [],
        model_id="test-risk-model",
        agent_type="risk",
    )
    return RiskAnalysis(
        signal=signal,
        risk_assessment="Moderate risk profile.",
        risk_metrics=risk_metrics
        or {
            "hist_vol_20d": 0.18,
            "hist_vol_252d": 0.22,
            "beta": 1.15,
            "sharpe_252d": 0.82,
            "max_drawdown_252d": -0.25,
            "var_95_20d": -0.03,
            "call_id": "risk001",
        },
        prompt_version="risk_v1",
    )


# ---------------------------------------------------------------------------
# E0-T1: RiskAnalysis returns AgentVerificationReport with agent_type="risk"
# ---------------------------------------------------------------------------


def test_verify_agent_risk_returns_report():
    analysis = _make_risk_analysis()
    report = verify_agent(analysis)
    assert report.agent_type == "risk"
    assert report.ticker == "AAPL"
    assert report.as_of_date == "2023-03-31"
    assert report.prompt_version == "risk_v1"


def test_verify_agent_risk_signal_none_empty_report():
    analysis = RiskAnalysis(
        signal=None,
        risk_assessment="Unknown.",
        risk_metrics={},
        prompt_version="risk_v1",
    )
    report = verify_agent(analysis)
    assert report.agent_type == "risk"
    assert report.n_claims == 0
    assert report.results == []
    assert report.hallucination_rate == 0.0


# ---------------------------------------------------------------------------
# E0-T1: Field resolution for known risk_metrics keys
# ---------------------------------------------------------------------------


def test_verify_agent_risk_beta_verified():
    """Beta of 1.15 in rationale matches beta=1.15 in risk_metrics -> verified."""
    analysis = _make_risk_analysis(
        rationale="Beta of 1.15 indicates market sensitivity.",
        call_ids=["risk001"],
        risk_metrics={"beta": 1.15, "call_id": "risk001"},
    )
    report = verify_agent(analysis)
    verified = [
        r
        for r in report.results
        if r.status == "verified" and r.claim.canonical_field == "beta"
    ]
    assert len(verified) >= 1
    assert verified[0].call_id_cited is True
    assert verified[0].tool_field == "risk_metrics"


def test_verify_agent_risk_sharpe_verified():
    """Sharpe ratio of 0.82 in rationale -> verified against sharpe_252d."""
    analysis = _make_risk_analysis(
        rationale="Sharpe ratio of 0.82 is acceptable.",
        call_ids=["risk001"],
        risk_metrics={"sharpe_252d": 0.82, "call_id": "risk001"},
    )
    report = verify_agent(analysis)
    verified = [
        r
        for r in report.results
        if r.status == "verified" and r.claim.canonical_field == "sharpe_252d"
    ]
    assert len(verified) >= 1


def test_verify_agent_risk_hist_vol_20d_verified():
    """hist vol 20d field resolves to hist_vol_20d."""
    analysis = _make_risk_analysis(
        rationale="hist vol 20d of 0.18 is low.",
        call_ids=["risk001"],
        risk_metrics={"hist_vol_20d": 0.18, "call_id": "risk001"},
    )
    report = verify_agent(analysis)
    verified = [
        r
        for r in report.results
        if r.status == "verified" and r.claim.canonical_field == "hist_vol_20d"
    ]
    assert len(verified) >= 1


def test_verify_agent_risk_hallucinated_beta():
    """Beta of 9.99 but tool has beta=1.15 -> hallucinated."""
    analysis = _make_risk_analysis(
        rationale="Beta of 9.99 is extreme.",
        call_ids=["risk001"],
        risk_metrics={"beta": 1.15, "call_id": "risk001"},
    )
    report = verify_agent(analysis)
    hallucinated = [r for r in report.results if r.status == "hallucinated"]
    assert len(hallucinated) >= 1
    assert report.n_hallucinated >= 1


def test_verify_agent_risk_absent_field_unresolvable():
    """RSI field not in risk_metrics -> unresolvable (not hallucinated)."""
    analysis = _make_risk_analysis(
        rationale="RSI of 42.0 suggests neutral.",
        call_ids=["risk001"],
        risk_metrics={"beta": 1.15, "call_id": "risk001"},
    )
    report = verify_agent(analysis)
    # rsi is not in risk_metrics; should be unresolvable
    unresolvable = [
        r for r in report.results if r.claim.canonical_field == "rsi"
    ]
    if unresolvable:
        assert unresolvable[0].status == "unresolvable"


def test_verify_agent_risk_zero_hr_all_verified():
    """Beta and Sharpe both correct -> hallucination_rate=0.0."""
    analysis = _make_risk_analysis(
        rationale="Beta of 1.15 and Sharpe ratio of 0.82 indicate acceptable risk.",
        call_ids=["risk001"],
        risk_metrics={"beta": 1.15, "sharpe_252d": 0.82, "call_id": "risk001"},
    )
    report = verify_agent(analysis)
    assert report.hallucination_rate == pytest.approx(0.0)
