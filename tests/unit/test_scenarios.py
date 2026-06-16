"""Tests for ScenarioEvaluator schema (P13-E6-T1, DJ-078)."""

import pytest

from hifi.collective.scenarios import (
    PHASE13_SCENARIOS,
    ScenarioDefinition,
    ScenarioResult,
    _check_alignment,
)

# ---------------------------------------------------------------------------
# ScenarioDefinition
# ---------------------------------------------------------------------------

def test_scenario_definition_valid():
    s = ScenarioDefinition(
        scenario_id="F-001",
        ticker="AAPL",
        as_of_date="2020-03-16",
        event_description="COVID crash",
        expected_direction="Risk-Off",
        regime="crash",
    )
    assert s.scenario_id == "F-001"
    assert s.expected_direction == "Risk-Off"


def test_scenario_definition_rejects_invalid_direction():
    with pytest.raises((ValueError, Exception)):  # noqa: B017
        ScenarioDefinition(
            scenario_id="F-X",
            ticker="AAPL",
            as_of_date="2020-03-16",
            event_description="test",
            expected_direction="Strong Sell",
            regime="test",
        )


# ---------------------------------------------------------------------------
# ScenarioResult
# ---------------------------------------------------------------------------

def test_scenario_result_aligned():
    r = ScenarioResult(
        scenario_id="F-001",
        ticker="AAPL",
        as_of_date="2020-03-16",
        collective_decision="Sell",
        expected_direction="Risk-Off",
        aligned=True,
        ensemble_output={},
    )
    assert r.aligned is True


def test_scenario_result_serialisable():
    r = ScenarioResult(
        scenario_id="F-003",
        ticker="AAPL",
        as_of_date="2023-02-02",
        collective_decision="Buy",
        expected_direction="Buy",
        aligned=True,
        ensemble_output={"ticker": "AAPL"},
    )
    d = r.model_dump()
    assert d["aligned"] is True
    assert d["ensemble_output"]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# _check_alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decision,expected,aligned", [
    ("Sell",  "Risk-Off", True),
    ("Hold",  "Risk-Off", True),
    ("Buy",   "Risk-Off", False),
    ("Buy",   "Buy",      True),
    ("Hold",  "Buy",      False),
    ("Sell",  "Sell",     True),
    ("Buy",   "Sell",     False),
    (None,    "Buy",      False),
    (None,    "Risk-Off", False),
])
def test_check_alignment(decision, expected, aligned):
    assert _check_alignment(decision, expected) is aligned


# ---------------------------------------------------------------------------
# PHASE13_SCENARIOS catalogue
# ---------------------------------------------------------------------------

def test_phase13_scenarios_non_empty():
    assert len(PHASE13_SCENARIOS) >= 3


def test_phase13_scenarios_unique_ids():
    ids = [s.scenario_id for s in PHASE13_SCENARIOS]
    assert len(ids) == len(set(ids))


def test_phase13_scenarios_valid_tickers():
    tickers = {s.ticker for s in PHASE13_SCENARIOS}
    assert tickers.issubset({"AAPL", "JPM", "XOM"})


def test_f001_crash_regime():
    f001 = next(s for s in PHASE13_SCENARIOS if s.scenario_id == "F-001")
    assert f001.regime == "crash"
    assert f001.as_of_date == "2020-03-16"


def test_f003_buy_direction():
    f003 = next(s for s in PHASE13_SCENARIOS if s.scenario_id == "F-003")
    assert f003.expected_direction == "Buy"
    assert f003.ticker == "AAPL"
