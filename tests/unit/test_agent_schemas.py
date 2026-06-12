"""Unit tests for agent output schemas (P3-E2, P8-E1)."""

import json

import pytest
from pydantic import ValidationError

from hifi.agents.schemas import (
    AgentSignal,
    ContrarianAnalysis,
    FundamentalAnalysis,
    MacroAnalysis,
    RiskAnalysis,
    SentimentAnalysis,
)

# ---------------------------------------------------------------------------
# AgentSignal
# ---------------------------------------------------------------------------


def _valid_signal(**overrides) -> AgentSignal:
    defaults = dict(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Buy",
        confidence=0.75,
        rationale="P/E of 28.3 is reasonable given ROE of 0.24 and macro tailwinds.",
        key_concern="High debt/equity ratio of 1.8 limits financial flexibility.",
        data_gaps=[],
        call_ids=["abc123def456"],
        model_id="qwen2.5-coder-32b-instruct-mlx",
        agent_type="fundamental",
    )
    defaults.update(overrides)
    return AgentSignal(**defaults)


def test_agent_signal_valid_buy():
    sig = _valid_signal(decision="Buy")
    assert sig.decision == "Buy"
    assert sig.confidence == 0.75


def test_agent_signal_valid_hold():
    sig = _valid_signal(decision="Hold", confidence=0.5)
    assert sig.decision == "Hold"


def test_agent_signal_valid_sell():
    sig = _valid_signal(decision="Sell", confidence=0.3)
    assert sig.decision == "Sell"


def test_agent_signal_confidence_boundary_zero():
    sig = _valid_signal(confidence=0.0)
    assert sig.confidence == 0.0


def test_agent_signal_confidence_boundary_one():
    sig = _valid_signal(confidence=1.0)
    assert sig.confidence == 1.0


def test_agent_signal_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        _valid_signal(confidence=-0.01)


def test_agent_signal_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        _valid_signal(confidence=1.01)


def test_agent_signal_rejects_invalid_decision():
    with pytest.raises(ValidationError):
        _valid_signal(decision="Maybe")


def test_agent_signal_rejects_empty_rationale():
    with pytest.raises(ValidationError):
        _valid_signal(rationale="   ")


def test_agent_signal_rejects_empty_key_concern():
    with pytest.raises(ValidationError):
        _valid_signal(key_concern="")


def test_agent_signal_empty_data_gaps_and_call_ids():
    sig = _valid_signal(data_gaps=[], call_ids=[])
    assert sig.data_gaps == []
    assert sig.call_ids == []


def test_agent_signal_data_gaps_populated():
    sig = _valid_signal(data_gaps=["pe", "revenue_growth_yoy"])
    assert "pe" in sig.data_gaps


def test_agent_signal_json_serialisable():
    sig = _valid_signal()
    dumped = json.dumps(sig.model_dump())
    loaded = json.loads(dumped)
    assert loaded["decision"] == "Buy"
    assert loaded["confidence"] == 0.75


# ---------------------------------------------------------------------------
# FundamentalAnalysis
# ---------------------------------------------------------------------------


def _valid_analysis(**overrides) -> FundamentalAnalysis:
    sig = _valid_signal()
    defaults = dict(
        signal=sig,
        financial_ratios={"pe": 28.3, "roe": 0.24, "call_id": "abc123"},
        growth_metrics={"net_margin": 0.25, "call_id": "def456"},
        valuation_context={"pe_1y_percentile": 0.6, "call_id": "ghi789"},
        macro_snapshot={"fed_funds_rate": 4.75, "call_id": "jkl012"},
        prompt_version="fundamental_v1",
        latency_ms=4200.0,
    )
    defaults.update(overrides)
    return FundamentalAnalysis(**defaults)


def test_fundamental_analysis_valid():
    a = _valid_analysis()
    assert a.signal.decision == "Buy"
    assert a.prompt_version == "fundamental_v1"
    assert a.latency_ms == 4200.0


def test_fundamental_analysis_latency_optional():
    a = _valid_analysis(latency_ms=None)
    assert a.latency_ms is None


def test_fundamental_analysis_json_serialisable():
    a = _valid_analysis()
    dumped = json.dumps(a.model_dump())
    loaded = json.loads(dumped)
    assert loaded["signal"]["decision"] == "Buy"
    assert loaded["prompt_version"] == "fundamental_v1"


def test_fundamental_analysis_tool_results_flat():
    a = _valid_analysis()
    flat = a.tool_results_flat()
    assert flat["pe"] == 28.3
    assert flat["fed_funds_rate"] == 4.75
    assert flat["net_margin"] == 0.25


def test_fundamental_analysis_signal_none_is_allowed():
    # A failed analysis (parse error) stores signal=None
    a = FundamentalAnalysis(
        signal=None,
        financial_ratios={},
        growth_metrics={},
        valuation_context={},
        macro_snapshot={},
        prompt_version="fundamental_v1",
        latency_ms=None,
    )
    assert a.signal is None


# ---------------------------------------------------------------------------
# RiskAnalysis (P8-E1)
# ---------------------------------------------------------------------------


def _valid_risk_signal() -> AgentSignal:
    return AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Hold",
        confidence=0.60,
        rationale="hist_vol_20d of 0.22 is moderate. Sharpe of 0.82 is acceptable.",
        key_concern="max_drawdown_252d of -0.28 is a material tail risk.",
        call_ids=["abc123def456"],
        model_id="google/gemma-3-4b",
        agent_type="risk",
    )


def test_risk_analysis_valid():
    ra = RiskAnalysis(
        signal=_valid_risk_signal(),
        risk_assessment="Moderate risk profile. Volatility is within typical bounds.",
        recommended_position_size=0.05,
        risk_metrics={"hist_vol_20d": 0.22, "sharpe_252d": 0.82, "call_id": "abc123"},
        prompt_version="risk_v1",
        latency_ms=500.0,
    )
    assert ra.signal.decision == "Hold"
    assert ra.recommended_position_size == pytest.approx(0.05)
    assert ra.prompt_version == "risk_v1"


def test_risk_analysis_signal_none_allowed():
    ra = RiskAnalysis(
        signal=None,
        risk_assessment="Parse failed.",
        risk_metrics={},
        prompt_version="risk_v1",
    )
    assert ra.signal is None


def test_risk_analysis_position_size_optional():
    ra = RiskAnalysis(
        signal=_valid_risk_signal(),
        risk_assessment="Moderate.",
        recommended_position_size=None,
        prompt_version="risk_v1",
    )
    assert ra.recommended_position_size is None


def test_risk_analysis_json_serialisable():
    ra = RiskAnalysis(
        signal=_valid_risk_signal(),
        risk_assessment="Moderate risk.",
        recommended_position_size=0.05,
        prompt_version="risk_v1",
    )
    json.dumps(ra.model_dump())


def test_risk_analysis_tool_results_flat():
    ra = RiskAnalysis(
        signal=_valid_risk_signal(),
        risk_assessment="ok",
        risk_metrics={"hist_vol_20d": 0.22, "sharpe_252d": 0.82},
        prompt_version="risk_v1",
    )
    flat = ra.tool_results_flat()
    assert flat["hist_vol_20d"] == pytest.approx(0.22)


# ---------------------------------------------------------------------------
# MacroAnalysis (P8-E1)
# ---------------------------------------------------------------------------


def _valid_macro_signal() -> AgentSignal:
    return AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Hold",
        confidence=0.55,
        rationale="Fed funds at 4.75% signals tight monetary conditions.",
        key_concern="Yield curve inverted; recession risk elevated.",
        call_ids=["abc123def456"],
        model_id="qwen3.5-27b-distilled",
        agent_type="macro",
    )


def test_macro_analysis_valid():
    ma = MacroAnalysis(
        signal=_valid_macro_signal(),
        regime_assessment="Late-cycle tightening",
        rationale="Fed at 4.75% with inverted yield curve suggests late-cycle.",
        macro_snapshot={"fed_funds_rate": 4.75, "cpi_yoy": 0.05, "call_id": "abc"},
        prompt_version="macro_v1",
        latency_ms=800.0,
    )
    assert ma.regime_assessment == "Late-cycle tightening"
    assert ma.prompt_version == "macro_v1"


def test_macro_analysis_signal_none_allowed():
    ma = MacroAnalysis(
        signal=None,
        regime_assessment="Unknown",
        rationale="Parse failed.",
        prompt_version="macro_v1",
    )
    assert ma.signal is None


def test_macro_analysis_json_serialisable():
    ma = MacroAnalysis(
        signal=_valid_macro_signal(),
        regime_assessment="Late-cycle",
        rationale="Rates elevated.",
        prompt_version="macro_v1",
    )
    json.dumps(ma.model_dump())


def test_macro_analysis_tool_results_flat():
    ma = MacroAnalysis(
        signal=_valid_macro_signal(),
        regime_assessment="ok",
        rationale="ok",
        macro_snapshot={"fed_funds_rate": 4.75},
        prompt_version="macro_v1",
    )
    flat = ma.tool_results_flat()
    assert flat["fed_funds_rate"] == pytest.approx(4.75)


# ---------------------------------------------------------------------------
# SentimentAnalysis (P8-E1)
# ---------------------------------------------------------------------------


def _valid_sentiment_signal() -> AgentSignal:
    return AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Buy",
        confidence=0.65,
        rationale="MD&A tone is cautiously optimistic; services growth highlighted.",
        key_concern="Management cited FX headwinds as a risk to international revenue.",
        call_ids=[],
        model_id="qwen2.5-coder-32b-instruct-mlx",
        agent_type="sentiment",
    )


def test_sentiment_analysis_valid():
    sa = SentimentAnalysis(
        signal=_valid_sentiment_signal(),
        sentiment_summary="Positive management tone with selective risk disclosures.",
        notable_signals=["services revenue grew 5% YoY", "FX headwinds mentioned"],
        prompt_version="sentiment_v1",
        latency_ms=600.0,
    )
    assert sa.sentiment_summary.startswith("Positive")
    assert len(sa.notable_signals) == 2
    assert sa.prompt_version == "sentiment_v1"


def test_sentiment_analysis_default_signal_hold():
    """Fail-open default: decision=Hold, confidence=0.0."""
    from hifi.agents.schemas import AgentSignal

    default_sig = AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Hold",
        confidence=0.0,
        rationale="Insufficient SEC filing data for sentiment analysis.",
        key_concern="No qualitative context available.",
        call_ids=[],
        model_id="sentiment-default",
        agent_type="sentiment",
    )
    sa = SentimentAnalysis(
        signal=default_sig,
        sentiment_summary="Insufficient Data",
        notable_signals=[],
        prompt_version="sentiment_v1",
    )
    assert sa.signal.decision == "Hold"
    assert sa.signal.confidence == 0.0


def test_sentiment_analysis_signal_none_allowed():
    sa = SentimentAnalysis(
        signal=None,
        sentiment_summary="Parse failed.",
        prompt_version="sentiment_v1",
    )
    assert sa.signal is None


def test_sentiment_analysis_json_serialisable():
    sa = SentimentAnalysis(
        signal=_valid_sentiment_signal(),
        sentiment_summary="Positive.",
        notable_signals=["services growth"],
        prompt_version="sentiment_v1",
    )
    json.dumps(sa.model_dump())


# ---------------------------------------------------------------------------
# ContrarianAnalysis (P8-E1)
# ---------------------------------------------------------------------------


def test_contrarian_analysis_valid():
    ca = ContrarianAnalysis(
        alternative_thesis="Despite positive signals, services growth may decelerate.",
        risk_scenario="Macro tightening reduces consumer discretionary spend by 15%.",
        counterargument="The Buy consensus ignores the yield curve inversion signal.",
        confidence=0.45,
        prompt_version="contrarian_v1",
        latency_ms=1200.0,
    )
    assert ca.confidence == pytest.approx(0.45)
    assert ca.prompt_version == "contrarian_v1"


def test_contrarian_analysis_no_agent_signal_field():
    """ContrarianAnalysis has no signal field — it does not vote."""
    ca = ContrarianAnalysis(
        alternative_thesis="Bear case.",
        risk_scenario="Recession within 12 months.",
        counterargument="Consensus ignores macro risk.",
        confidence=0.40,
        prompt_version="contrarian_v1",
    )
    assert not hasattr(ca, "decision")
    assert not hasattr(ca, "signal")


def test_contrarian_analysis_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        ContrarianAnalysis(
            alternative_thesis="Bear case.",
            risk_scenario="Recession.",
            counterargument="Macro risk ignored.",
            confidence=1.5,
            prompt_version="contrarian_v1",
        )


def test_contrarian_analysis_rejects_empty_thesis():
    with pytest.raises(ValidationError):
        ContrarianAnalysis(
            alternative_thesis="   ",
            risk_scenario="Recession.",
            counterargument="Macro risk ignored.",
            confidence=0.40,
            prompt_version="contrarian_v1",
        )


def test_contrarian_analysis_json_serialisable():
    ca = ContrarianAnalysis(
        alternative_thesis="Bear case.",
        risk_scenario="Recession within 12 months.",
        counterargument="Consensus ignores macro risk.",
        confidence=0.40,
        prompt_version="contrarian_v1",
    )
    dumped = json.dumps(ca.model_dump())
    loaded = json.loads(dumped)
    assert loaded["confidence"] == pytest.approx(0.40)


def test_contrarian_analysis_latency_optional():
    ca = ContrarianAnalysis(
        alternative_thesis="Bear case.",
        risk_scenario="Recession.",
        counterargument="Macro risk ignored.",
        confidence=0.40,
        prompt_version="contrarian_v1",
        latency_ms=None,
    )
    assert ca.latency_ms is None
