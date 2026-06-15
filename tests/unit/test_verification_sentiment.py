"""Unit tests for SentimentGroundingReport schemas and verify_sentiment_agent (P13-E0-T3/T4/T5)."""

from __future__ import annotations

import pytest

from hifi.agents.schemas import (
    AgentSignal,
    FundamentalAnalysis,
    SentimentAnalysis,
    TechnicalAnalysis,
)
from hifi.collective.schemas import EnsembleDecision, EnsembleOutput
from hifi.verification.schemas import SentimentGroundingReport, SentimentGroundingResult
from hifi.verification.verifier import verify_ensemble, verify_sentiment_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sentiment(
    notable_signals: list[str] | None = None,
    signal_rationale: str = "Neutral sentiment.",
    has_signal: bool = True,
) -> SentimentAnalysis:
    signal = (
        AgentSignal(
            ticker="AAPL",
            as_of_date="2023-03-31",
            decision="Hold",
            confidence=0.5,
            rationale=signal_rationale,
            key_concern="Tone ambiguous.",
            call_ids=[],
            model_id="test-sentiment-model",
            agent_type="sentiment",
        )
        if has_signal
        else None
    )
    return SentimentAnalysis(
        signal=signal,
        sentiment_summary="Neutral tone.",
        notable_signals=notable_signals or [],
        prompt_version="sentiment_v1",
    )


def _minimal_ensemble_output(
    sentiment_analysis: SentimentAnalysis | None = None,
) -> EnsembleOutput:
    """Build a minimal EnsembleOutput (Fund + Tech only, optional Sentiment)."""

    def _signal(agent_type: str) -> AgentSignal:
        return AgentSignal(
            ticker="AAPL",
            as_of_date="2023-03-31",
            decision="Hold",
            confidence=0.5,
            rationale="Holds steady.",
            key_concern="Rate risk.",
            call_ids=[],
            model_id="test-model",
            agent_type=agent_type,
        )

    fund = FundamentalAnalysis(
        signal=_signal("fundamental"),
        financial_ratios={"pe": 28.3, "call_id": "abc"},
        growth_metrics={"net_margin": 0.25, "call_id": "def"},
        valuation_context={"pe_1y_percentile": 0.6, "call_id": "ghi"},
        macro_snapshot={"fed_funds_rate": 4.75, "call_id": "jkl"},
        prompt_version="fundamental_v1",
    )
    tech = TechnicalAnalysis(
        signal=_signal("technical"),
        technical_indicators={"rsi": 42.1, "call_id": "tech001"},
        risk_metrics={"sharpe_252d": 0.82, "call_id": "tech002"},
        prompt_version="technical_v1",
    )
    decision = EnsembleDecision(
        collective_decision="Hold",
        collective_confidence=0.6,
        n_valid_signals=2,
        agreement=True,
        disagreement_entropy=0.0,
        opinion_dispersion=0.0,
        agent_decisions=["Hold", "Hold"],
        agent_confidences=[0.5, 0.5],
        winning_score=1.0,
        total_score=1.0,
    )
    return EnsembleOutput(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_analysis=fund,
        technical_analysis=tech,
        ensemble_decision=decision,
        latency_ms=100.0,
        sentiment_analysis=sentiment_analysis,
    )


# ---------------------------------------------------------------------------
# E0-T3: Schema validation
# ---------------------------------------------------------------------------


def test_sentiment_grounding_result_grounded():
    r = SentimentGroundingResult(
        signal_text="management expressed confidence in revenue guidance",
        grounded=True,
        matched_chunk="management expressed confidence in revenue guidance",
    )
    assert r.grounded is True
    assert r.matched_chunk is not None


def test_sentiment_grounding_result_ungrounded():
    r = SentimentGroundingResult(
        signal_text="claims fabricated",
        grounded=False,
        matched_chunk=None,
    )
    assert r.grounded is False
    assert r.matched_chunk is None


def test_sentiment_grounding_result_matched_chunk_optional():
    """matched_chunk defaults to None — field is optional."""
    r = SentimentGroundingResult(signal_text="foo", grounded=False)
    assert r.matched_chunk is None


def test_sentiment_grounding_report_basic():
    report = SentimentGroundingReport(
        ticker="AAPL",
        as_of_date="2023-03-31",
        n_signals=2,
        n_grounded=1,
        grounding_rate=0.5,
        results=[
            SentimentGroundingResult(
                signal_text="foo", grounded=True, matched_chunk="foo"
            ),
            SentimentGroundingResult(
                signal_text="bar", grounded=False, matched_chunk=None
            ),
        ],
    )
    assert report.n_signals == 2
    assert report.n_grounded == 1
    assert report.grounding_rate == pytest.approx(0.5)
    assert report.ticker == "AAPL"


def test_sentiment_grounding_report_zero_signals():
    """n_signals=0 -> grounding_rate=0.0 (no ZeroDivisionError guard needed in schema)."""
    report = SentimentGroundingReport(
        ticker="AAPL",
        as_of_date="2023-03-31",
        n_signals=0,
        n_grounded=0,
        grounding_rate=0.0,
        results=[],
    )
    assert report.grounding_rate == 0.0
    assert report.results == []


# ---------------------------------------------------------------------------
# E0-T4: verify_sentiment_agent edge cases
# ---------------------------------------------------------------------------


def test_verify_sentiment_signal_none_empty_report():
    """analysis.signal is None -> empty report, n_signals=0."""
    analysis = _make_sentiment(has_signal=False)
    report = verify_sentiment_agent(analysis, "any context")
    assert report.n_signals == 0
    assert report.n_grounded == 0
    assert report.grounding_rate == 0.0
    assert report.results == []


def test_verify_sentiment_empty_notable_signals():
    """notable_signals=[] -> n_signals=0, grounding_rate=0.0."""
    analysis = _make_sentiment(notable_signals=[])
    report = verify_sentiment_agent(analysis, "lots of context here")
    assert report.n_signals == 0
    assert report.n_grounded == 0
    assert report.grounding_rate == 0.0


def test_verify_sentiment_all_grounded():
    """All signals found verbatim in context -> grounding_rate=1.0."""
    context = (
        "Management expressed confidence in revenue guidance "
        "and expects strong growth."
    )
    signals = [
        "management expressed confidence in revenue guidance",
        "expects strong growth",
    ]
    analysis = _make_sentiment(notable_signals=signals)
    report = verify_sentiment_agent(analysis, context)
    assert report.n_signals == 2
    assert report.n_grounded == 2
    assert report.grounding_rate == pytest.approx(1.0)
    for r in report.results:
        assert r.grounded is True
        assert r.matched_chunk is not None


def test_verify_sentiment_none_grounded():
    """No signal found in context -> grounding_rate=0.0."""
    context = "Completely different text with no overlap."
    signals = ["management expressed confidence", "expects strong growth"]
    analysis = _make_sentiment(notable_signals=signals)
    report = verify_sentiment_agent(analysis, context)
    assert report.n_grounded == 0
    assert report.grounding_rate == 0.0
    for r in report.results:
        assert r.grounded is False
        assert r.matched_chunk is None


def test_verify_sentiment_partial_grounding():
    """One of two signals grounded -> grounding_rate=0.5."""
    context = "Revenue guidance was positive."
    signals = ["revenue guidance was positive", "fabricated claim not in context"]
    analysis = _make_sentiment(notable_signals=signals)
    report = verify_sentiment_agent(analysis, context)
    assert report.n_signals == 2
    assert report.n_grounded == 1
    assert report.grounding_rate == pytest.approx(0.5)


def test_verify_sentiment_empty_context():
    """Empty retrieved_context -> all signals ungrounded."""
    signals = ["some management statement", "another claim"]
    analysis = _make_sentiment(notable_signals=signals)
    report = verify_sentiment_agent(analysis, "")
    assert report.n_grounded == 0
    assert report.grounding_rate == 0.0


def test_verify_sentiment_case_insensitive():
    """Normalisation is lowercase: UPPER context matches lower signal."""
    context = "MANAGEMENT EXPRESSED CONFIDENCE IN REVENUE GUIDANCE"
    signals = ["management expressed confidence in revenue guidance"]
    analysis = _make_sentiment(notable_signals=signals)
    report = verify_sentiment_agent(analysis, context)
    assert report.n_grounded == 1
    assert report.grounding_rate == pytest.approx(1.0)


def test_verify_sentiment_ticker_and_date():
    """Report carries ticker and as_of_date from signal."""
    analysis = _make_sentiment(notable_signals=["test signal"])
    report = verify_sentiment_agent(analysis, "test signal found here")
    assert report.ticker == "AAPL"
    assert report.as_of_date == "2023-03-31"


def test_verify_sentiment_matched_chunk_is_normalised_signal():
    """matched_chunk equals the normalised (lowercase) form of the signal."""
    context = "Revenue guidance was positive."
    signals = ["Revenue Guidance Was Positive"]
    analysis = _make_sentiment(notable_signals=signals)
    report = verify_sentiment_agent(analysis, context)
    assert report.n_grounded == 1
    assert report.results[0].matched_chunk == "revenue guidance was positive"


# ---------------------------------------------------------------------------
# E0-T5: verify_ensemble backward compatibility and sentiment extension
# ---------------------------------------------------------------------------


def test_verify_ensemble_backward_compat_no_sentiment():
    """Existing call (no sentiment_context) -> sentiment_report=None."""
    output = _minimal_ensemble_output()
    report = verify_ensemble(output, always_verify=True)
    assert report.sentiment_report is None
    assert report.ticker == "AAPL"
    assert report.fundamental_report is not None
    assert report.technical_report is not None


def test_verify_ensemble_sentiment_context_no_sentiment_analysis():
    """sentiment_context provided but output.sentiment_analysis=None -> None."""
    output = _minimal_ensemble_output(sentiment_analysis=None)
    report = verify_ensemble(output, always_verify=True, sentiment_context="some context")
    assert report.sentiment_report is None


def test_verify_ensemble_with_sentiment_populates_report():
    """sentiment_context + sentiment_analysis present -> sentiment_report populated."""
    context = "Revenue guidance was positive and growth is expected."
    sentiment = SentimentAnalysis(
        signal=AgentSignal(
            ticker="AAPL",
            as_of_date="2023-03-31",
            decision="Buy",
            confidence=0.7,
            rationale="Positive tone.",
            key_concern="None identified.",
            call_ids=[],
            model_id="test-sentiment-model",
            agent_type="sentiment",
        ),
        sentiment_summary="Positive.",
        notable_signals=["revenue guidance was positive", "growth is expected"],
        prompt_version="sentiment_v1",
    )
    output = _minimal_ensemble_output(sentiment_analysis=sentiment)
    report = verify_ensemble(output, always_verify=True, sentiment_context=context)
    assert report.sentiment_report is not None
    assert report.sentiment_report.n_signals == 2
    assert report.sentiment_report.n_grounded == 2
    assert report.sentiment_report.grounding_rate == pytest.approx(1.0)


def test_verify_ensemble_sentiment_report_json_serialisable():
    """EnsembleVerificationReport with sentiment_report serialises cleanly."""
    import json

    context = "management tone was positive."
    sentiment = SentimentAnalysis(
        signal=AgentSignal(
            ticker="AAPL",
            as_of_date="2023-03-31",
            decision="Buy",
            confidence=0.6,
            rationale="Good tone.",
            key_concern="None.",
            call_ids=[],
            model_id="test",
            agent_type="sentiment",
        ),
        sentiment_summary="Positive.",
        notable_signals=["management tone was positive"],
        prompt_version="sentiment_v1",
    )
    output = _minimal_ensemble_output(sentiment_analysis=sentiment)
    report = verify_ensemble(output, always_verify=True, sentiment_context=context)
    dumped = json.dumps(report.model_dump())
    loaded = json.loads(dumped)
    assert "sentiment_report" in loaded
    assert loaded["sentiment_report"]["grounding_rate"] == pytest.approx(1.0)


def test_verify_ensemble_existing_fields_unchanged():
    """Adding sentiment_report does not change existing EnsembleVerificationReport fields."""
    output = _minimal_ensemble_output()
    report = verify_ensemble(output, always_verify=True)
    assert hasattr(report, "fundamental_report")
    assert hasattr(report, "technical_report")
    assert hasattr(report, "contradictions")
    assert hasattr(report, "triggered_by_disagreement")
    assert hasattr(report, "n_contradictions")
    assert hasattr(report, "total_claims")
    assert hasattr(report, "ensemble_hallucination_rate")
