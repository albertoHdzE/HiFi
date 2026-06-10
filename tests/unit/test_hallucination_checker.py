"""Unit tests for the baseline hallucination checker and metrics (P3-E4)."""

import pytest

from hifi.agents.baseline_metrics import (
    call_id_coverage,
    compute_metrics,
    count_hallucinated_numbers,
    data_gap_acknowledged,
    extract_numbers,
    number_in_tool_results,
)
from hifi.agents.schemas import AgentSignal, FundamentalAnalysis


def _make_analysis(
    rationale: str = "P/E of 28.3 is reasonable.",
    key_concern: str = "High debt.",
    data_gaps: list | None = None,
    call_ids: list | None = None,
    financial_ratios: dict | None = None,
    growth_metrics: dict | None = None,
    valuation_context: dict | None = None,
    macro_snapshot: dict | None = None,
    latency_ms: float = 1000.0,
) -> FundamentalAnalysis:
    sig = AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Hold",
        confidence=0.6,
        rationale=rationale,
        key_concern=key_concern,
        data_gaps=data_gaps or [],
        call_ids=call_ids or [],
        model_id="qwen2.5-coder-32b-instruct-mlx",
        agent_type="fundamental",
    )
    return FundamentalAnalysis(
        signal=sig,
        financial_ratios=financial_ratios or {"pe": 28.3, "roe": 0.24, "call_id": "aaa"},
        growth_metrics=growth_metrics or {"net_margin": 0.25, "call_id": "bbb"},
        valuation_context=valuation_context or {"pe_1y_percentile": 0.6, "call_id": "ccc"},
        macro_snapshot=macro_snapshot or {"fed_funds_rate": 4.75, "call_id": "ddd"},
        prompt_version="fundamental_v1",
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# extract_numbers
# ---------------------------------------------------------------------------


def test_extract_numbers_simple():
    assert extract_numbers("P/E of 28.3 and ROE of 0.24") == [28.3, 0.24]


def test_extract_numbers_negative():
    nums = extract_numbers("drawdown of -15.2%")
    assert -15.2 in nums


def test_extract_numbers_empty():
    assert extract_numbers("no numbers here") == []


def test_extract_numbers_integer():
    nums = extract_numbers("up 5 percent")
    assert 5.0 in nums


# ---------------------------------------------------------------------------
# number_in_tool_results
# ---------------------------------------------------------------------------


def test_number_found_exact():
    assert number_in_tool_results(28.3, {"pe": 28.3})


def test_number_found_within_tolerance():
    assert number_in_tool_results(28.29, {"pe": 28.3})


def test_number_not_found():
    assert not number_in_tool_results(99.9, {"pe": 28.3, "roe": 0.24})


def test_number_zero_match():
    assert number_in_tool_results(0.0, {"x": 0.0})


def test_number_in_nested_dict():
    assert number_in_tool_results(4.75, {"macro": {"fed_funds_rate": 4.75}})


# ---------------------------------------------------------------------------
# count_hallucinated_numbers
# ---------------------------------------------------------------------------


def test_no_hallucinations_when_all_numbers_in_results():
    # rationale cites 28.3 which is in financial_ratios
    a = _make_analysis(rationale="P/E of 28.3 and ROE of 0.24 look solid.")
    assert count_hallucinated_numbers(a) == 0


def test_hallucination_detected_for_invented_number():
    # 99.99 does not appear in any tool result
    a = _make_analysis(rationale="P/E of 99.99 seems very high.")
    assert count_hallucinated_numbers(a) >= 1


def test_small_integers_not_flagged():
    # "1" and "2" should not be flagged as hallucinations
    a = _make_analysis(rationale="The company has 1 key risk and 2 advantages.")
    assert count_hallucinated_numbers(a) == 0


def test_hallucination_count_zero_when_signal_none():
    a = FundamentalAnalysis(
        signal=None,
        financial_ratios={}, growth_metrics={},
        valuation_context={}, macro_snapshot={},
        prompt_version="v1", latency_ms=None,
    )
    assert count_hallucinated_numbers(a) == 0


# ---------------------------------------------------------------------------
# data_gap_acknowledged
# ---------------------------------------------------------------------------


def test_data_gap_acknowledged_when_phrase_in_rationale():
    a = _make_analysis(
        data_gaps=["revenue_growth_yoy"],
        rationale="revenue_growth_yoy is unavailable for this snapshot.",
    )
    assert data_gap_acknowledged(a)


def test_data_gap_not_acknowledged_when_field_missing_from_rationale():
    a = _make_analysis(
        data_gaps=["revenue_growth_yoy"],
        rationale="The P/E ratio of 28.3 looks reasonable to me.",
    )
    assert not data_gap_acknowledged(a)


def test_no_data_gaps_returns_true():
    a = _make_analysis(data_gaps=[])
    assert data_gap_acknowledged(a)


# ---------------------------------------------------------------------------
# call_id_coverage
# ---------------------------------------------------------------------------


def test_full_coverage_when_all_call_ids_cited():
    a = _make_analysis(call_ids=["aaa", "bbb", "ccc", "ddd"])
    # 4 call_ids, 4 tool results each with call_id -> coverage = 1.0
    assert call_id_coverage(a) == 1.0


def test_partial_coverage():
    a = _make_analysis(call_ids=["aaa", "bbb"])  # cited 2 of 4
    assert call_id_coverage(a) == pytest.approx(0.5)


def test_zero_coverage_when_no_call_ids():
    a = _make_analysis(call_ids=[])
    assert call_id_coverage(a) == 0.0


def test_zero_coverage_when_signal_none():
    a = FundamentalAnalysis(
        signal=None,
        financial_ratios={}, growth_metrics={},
        valuation_context={}, macro_snapshot={},
        prompt_version="v1", latency_ms=None,
    )
    assert call_id_coverage(a) == 0.0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_full_compliance():
    analyses = {
        "AAPL": _make_analysis(call_ids=["aaa", "bbb", "ccc", "ddd"], latency_ms=3000.0),
        "JPM": _make_analysis(call_ids=["aaa", "bbb", "ccc", "ddd"], latency_ms=4000.0),
    }
    m = compute_metrics(analyses)
    assert m["compliance_rate"] == 1.0
    assert m["n_analyses"] == 2
    assert m["mean_latency_ms"] == pytest.approx(3500.0)


def test_compute_metrics_empty():
    m = compute_metrics({})
    assert m["compliance_rate"] == 0.0
    assert m["n_analyses"] == 0


def test_compute_metrics_partial_compliance():
    a_valid = _make_analysis()
    a_failed = FundamentalAnalysis(
        signal=None,
        financial_ratios={}, growth_metrics={},
        valuation_context={}, macro_snapshot={},
        prompt_version="v1", latency_ms=None,
    )
    m = compute_metrics({"AAPL": a_valid, "JPM": a_failed})
    assert m["compliance_rate"] == pytest.approx(0.5)
