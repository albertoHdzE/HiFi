"""Unit tests for verify_claim and verify_agent (P5-E3-T1 through T10)."""

from __future__ import annotations

import pytest

from hifi.agents.schemas import AgentSignal, FundamentalAnalysis, TechnicalAnalysis
from hifi.verification.schemas import NumericalClaim
from hifi.verification.verifier import verify_agent, verify_claim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claim(
    alias: str,
    canonical: str | None,
    value: float,
    snippet: str = "...",
) -> NumericalClaim:
    return NumericalClaim(
        field_alias=alias,
        canonical_field=canonical,
        value=value,
        context_snippet=snippet,
    )


def _named_results(
    fields: dict[str, object], call_id: str, tool_name: str = "tool"
) -> list[tuple[str, dict]]:
    """Build a single-item named_tool_results list."""
    d = dict(fields)
    d["call_id"] = call_id
    return [(tool_name, d)]


def _make_fund_analysis(
    rationale: str = "Holds steady.",
    call_ids: list[str] | None = None,
    **tool_overrides: object,
) -> FundamentalAnalysis:
    signal = AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Hold",
        confidence=0.65,
        rationale=rationale,
        key_concern="Rate risk.",
        call_ids=call_ids or [],
        model_id="test-model",
        agent_type="fundamental",
    )
    defaults: dict = {
        "financial_ratios": {"pe": 28.3, "roe": 0.24, "call_id": "abc123"},
        "growth_metrics": {"net_margin": 0.25, "call_id": "def456"},
        "valuation_context": {"pe_1y_percentile": 0.6, "call_id": "ghi789"},
        "macro_snapshot": {"fed_funds_rate": 4.75, "call_id": "jkl012"},
    }
    defaults.update(tool_overrides)
    return FundamentalAnalysis(
        signal=signal,
        prompt_version="fundamental_v1",
        **defaults,
    )


def _make_tech_analysis(
    rationale: str = "Neutral.",
    call_ids: list[str] | None = None,
    **tool_overrides: object,
) -> TechnicalAnalysis:
    signal = AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Hold",
        confidence=0.65,
        rationale=rationale,
        key_concern="Volatility risk.",
        call_ids=call_ids or [],
        model_id="test-tech-model",
        agent_type="technical",
    )
    defaults: dict = {
        "technical_indicators": {"rsi": 42.1, "sma": 158.0, "call_id": "tech001"},
        "risk_metrics": {"sharpe_252d": 0.82, "hist_vol_252d": 0.25, "call_id": "tech002"},
    }
    defaults.update(tool_overrides)
    return TechnicalAnalysis(
        signal=signal,
        prompt_version="technical_v1",
        **defaults,
    )


# ---------------------------------------------------------------------------
# P5-E3-T2: Verified claim with call_id_cited=True
# ---------------------------------------------------------------------------


def test_verify_claim_exact_match_verified():
    claim = _claim("RSI", "rsi", 42.1)
    named = _named_results({"rsi": 42.1}, "tech001", "technical_indicators")
    result = verify_claim(claim, named, ["tech001"])
    assert result.status == "verified"
    assert result.tool_value == pytest.approx(42.1)
    assert result.call_id_cited is True
    assert result.tool_field == "technical_indicators"


def test_verify_claim_within_relative_tolerance():
    # tool_value=28.3, claimed=28.3 * 1.005 (within 1% relative tolerance)
    tool_val = 28.3
    claimed = tool_val * 1.005  # 0.5% off -> verified
    claim = _claim("P/E", "pe", claimed)
    named = _named_results({"pe": tool_val}, "abc123", "financial_ratios")
    result = verify_claim(claim, named, ["abc123"])
    assert result.status == "verified"
    assert result.call_id_cited is True


def test_verify_claim_call_id_not_in_signal():
    """Claim is verified but call_id not in signal.call_ids -> call_id_cited=False."""
    claim = _claim("RSI", "rsi", 42.1)
    named = _named_results({"rsi": 42.1}, "tech001", "technical_indicators")
    result = verify_claim(claim, named, ["other_id"])  # wrong call_id
    assert result.status == "verified"
    assert result.call_id_cited is False


# ---------------------------------------------------------------------------
# P5-E3-T3: Hallucinated claim (DJ-020)
# ---------------------------------------------------------------------------


def test_verify_claim_outside_relative_tolerance_hallucinated():
    # tool_value=28.3, claimed=50.0 -> far outside 1% -> hallucinated
    claim = _claim("P/E", "pe", 50.0)
    named = _named_results({"pe": 28.3}, "abc123", "financial_ratios")
    result = verify_claim(claim, named, ["abc123"])
    assert result.status == "hallucinated"
    assert result.tool_value == pytest.approx(28.3)


def test_verify_claim_hallucinated_small_value():
    # tool_value=0.82 (<=1.0), absolute tol=0.01, claimed=0.95 -> |0.95-0.82|=0.13 > 0.01
    claim = _claim("Sharpe", "sharpe_252d", 0.95)
    named = _named_results({"sharpe_252d": 0.82}, "tech002", "risk_metrics")
    result = verify_claim(claim, named, ["tech002"])
    assert result.status == "hallucinated"


def test_verify_claim_tolerance_boundary_exact():
    # tool_value=10.0, tol=0.01*10=0.10
    # claimed=10.10 -> |0.10| == 0.10 -> exactly on boundary -> verified
    claim = _claim("SMA", "sma", 10.10)
    named = _named_results({"sma": 10.0}, "t01", "technical_indicators")
    result = verify_claim(claim, named, [])
    assert result.status == "verified"


def test_verify_claim_tolerance_boundary_just_over():
    # tool_value=10.0, tol=0.10, claimed=10.101 -> |0.101| > 0.10 -> hallucinated
    claim = _claim("SMA", "sma", 10.101)
    named = _named_results({"sma": 10.0}, "t01", "technical_indicators")
    result = verify_claim(claim, named, [])
    assert result.status == "hallucinated"


# ---------------------------------------------------------------------------
# P5-E3-T4: Absent field -> unresolvable
# ---------------------------------------------------------------------------


def test_verify_claim_absent_field_unresolvable():
    claim = _claim("RSI", "rsi", 42.1)
    # rsi not in the dict
    named = _named_results({"sma": 158.0}, "tech001", "technical_indicators")
    result = verify_claim(claim, named, ["tech001"])
    assert result.status == "unresolvable"
    assert result.tool_value is None


# ---------------------------------------------------------------------------
# P5-E3-T5: None-valued field -> unresolvable (not hallucinated)
# ---------------------------------------------------------------------------


def test_verify_claim_none_value_unresolvable():
    """Field present but value is None -> unresolvable, not hallucinated."""
    claim = _claim("RSI", "rsi", 42.1)
    named = _named_results({"rsi": None}, "tech001", "technical_indicators")
    result = verify_claim(claim, named, ["tech001"])
    assert result.status == "unresolvable"
    assert result.tool_value is None
    # tool_field is set because the key was found (even though value is None)
    assert result.tool_field == "technical_indicators"


# ---------------------------------------------------------------------------
# P5-E3-T4 (canonical=None case)
# ---------------------------------------------------------------------------


def test_verify_claim_no_canonical_field_unresolvable():
    """canonical_field=None (unknown alias) -> immediately unresolvable."""
    claim = _claim("mystery_metric", None, 3.4)
    named = _named_results({"rsi": 42.1}, "tech001")
    result = verify_claim(claim, named, ["tech001"])
    assert result.status == "unresolvable"
    assert result.tool_field is None
    assert result.tolerance_used == 0.0


# ---------------------------------------------------------------------------
# P5-E3-T6: verify_agent with signal=None -> empty report
# ---------------------------------------------------------------------------


def test_verify_agent_signal_none_returns_empty():
    analysis = FundamentalAnalysis(
        signal=None,
        financial_ratios={},
        growth_metrics={},
        valuation_context={},
        macro_snapshot={},
        prompt_version="fundamental_v1",
    )
    report = verify_agent(analysis)
    assert report.n_claims == 0
    assert report.n_verified == 0
    assert report.hallucination_rate == 0.0
    assert report.results == []
    assert report.agent_type == "fundamental"


# ---------------------------------------------------------------------------
# P5-E3-T7: verify_agent on FundamentalAnalysis with injected known claim
# ---------------------------------------------------------------------------


def test_verify_agent_fundamental_known_verified_claim():
    """P/E of 28.3 in rationale matches pe=28.3 in tool results -> verified."""
    analysis = _make_fund_analysis(
        rationale="P/E of 28.3 is below sector average.",
        call_ids=["abc123"],
        financial_ratios={"pe": 28.3, "roe": 0.24, "call_id": "abc123"},
    )
    report = verify_agent(analysis)
    assert report.agent_type == "fundamental"
    verified = [
        r for r in report.results
        if r.status == "verified" and r.claim.canonical_field == "pe"
    ]
    assert len(verified) >= 1
    assert verified[0].call_id_cited is True


def test_verify_agent_fundamental_known_hallucinated_claim():
    """P/E of 999.0 in rationale but tool has pe=28.3 -> hallucinated."""
    analysis = _make_fund_analysis(
        rationale="P/E of 999.0 seems extreme.",
        call_ids=["abc123"],
        financial_ratios={"pe": 28.3, "call_id": "abc123"},
    )
    report = verify_agent(analysis)
    hallucinated = [r for r in report.results if r.status == "hallucinated"]
    assert len(hallucinated) >= 1
    assert report.n_hallucinated >= 1


# ---------------------------------------------------------------------------
# P5-E3-T8: verify_agent on TechnicalAnalysis with injected known claim
# ---------------------------------------------------------------------------


def test_verify_agent_technical_known_verified_claim():
    """RSI of 42.1 in rationale matches rsi=42.1 in tool results -> verified."""
    analysis = _make_tech_analysis(
        rationale="RSI of 42.1 signals recovery.",
        call_ids=["tech001"],
        technical_indicators={"rsi": 42.1, "sma": 158.0, "call_id": "tech001"},
    )
    report = verify_agent(analysis)
    assert report.agent_type == "technical"
    verified = [
        r for r in report.results
        if r.status == "verified" and r.claim.canonical_field == "rsi"
    ]
    assert len(verified) >= 1


# ---------------------------------------------------------------------------
# P5-E3-T9: hallucination_rate=0.0 when all verified
# ---------------------------------------------------------------------------


def test_verify_agent_zero_hr_all_verified():
    analysis = _make_fund_analysis(
        rationale="P/E of 28.3 is solid. ROE of 0.24 is strong.",
        call_ids=["abc123"],
        financial_ratios={"pe": 28.3, "roe": 0.24, "call_id": "abc123"},
    )
    report = verify_agent(analysis)
    assert report.hallucination_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# P5-E3-T10: grounding_rate=0.0 when call_ids empty even if all verified
# ---------------------------------------------------------------------------


def test_verify_agent_zero_grounding_when_no_call_ids():
    """Verified claims but signal.call_ids=[] -> call_id_cited=False for all."""
    analysis = _make_fund_analysis(
        rationale="P/E of 28.3 is solid.",
        call_ids=[],  # no call_ids in signal
        financial_ratios={"pe": 28.3, "call_id": "abc123"},
    )
    report = verify_agent(analysis)
    # P/E claim may be verified (value matches) but call_id not cited
    [
        r for r in report.results if r.status == "verified" and not r.call_id_cited
    ]
    if report.n_verified > 0:
        assert report.grounding_rate == pytest.approx(0.0)
