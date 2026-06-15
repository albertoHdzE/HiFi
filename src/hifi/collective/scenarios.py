"""
Synthetic scenario framework for Phase 13 (P13-E6, DJ-078).

Populates Dataset Family F with three historical stress-test scenarios (DJ-078).
Scenarios use historical market data already acquired in Phase 1 — they are NOT
generated synthetic data. This avoids generation methodology artifacts while
still stress-testing agent behavior under extreme conditions.

Three Phase 13 scenarios (DJ-078):
  F-001  2020-03-16 (Black Monday II)  — COVID crash, single-day drop >10%
  F-002  2022-03-31 (rate shock)        — FFR rising, CPI at 8.5%
  F-003  2023-02-02 (AAPL earnings beat)— Q1 2023 beat $0.06 EPS

Methodological limitation (honest, from DJ-078):
  Historical scenarios are a SUBSET of the existing evaluation universe.
  They do NOT test agent behavior on truly unseen distribution shifts.
  True generative synthetic scenarios (GARCH, VAE) are deferred to Phase 16.

David reference: §8.7 (Dataset Family F — Synthetic Scenario Datasets)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ScenarioDefinition(BaseModel):
    """
    One stress-test scenario definition for Dataset Family F (P13-E6-T1).

    expected_direction is a soft expectation — not a hard label. The system
    may legitimately disagree. Alignment is documented but not penalised.
    "Risk-Off" means any defensive signal (Hold or Sell) is aligned.
    """

    scenario_id: str                           # e.g. "F-001"
    ticker: str                                # equity ticker to evaluate
    as_of_date: str                            # ISO 8601 analysis date
    event_description: str                     # narrative description of the event
    expected_direction: Literal["Buy", "Hold", "Sell", "Risk-Off"]
    regime: str                                # e.g. "crash", "rate_shock", "earnings_beat"


class ScenarioResult(BaseModel):
    """
    Evaluation result for one ScenarioDefinition (P13-E6-T1).

    aligned is True when:
      - expected_direction in {"Buy", "Hold", "Sell"} and
        collective_decision == expected_direction, OR
      - expected_direction == "Risk-Off" and
        collective_decision in {"Hold", "Sell"}
    ensemble_output stores the full EnsembleOutput JSON for Dataset Family F.
    """

    scenario_id: str
    ticker: str
    as_of_date: str
    collective_decision: str | None          # None when all agents failed
    expected_direction: str
    aligned: bool
    ensemble_output: dict                    # serialised EnsembleOutput


class ScenarioEvaluator:
    """
    Runs a ScenarioDefinition through the ensemble and records alignment (P13-E6-T1).

    Requires LM Studio to be running (same as any live ensemble run).
    Output is written to data/scenarios/{scenario_id}.json.
    """

    def __init__(
        self,
        snapshot_json: str,
        data_dir: str | None = None,
        agents: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        snapshot_json : str
            JSON-serialised FundamentalsSnapshot for the Fundamental Agent.
        data_dir : str | None
            Path to data root. Defaults to HIFI_DATA_DIR env var.
        agents : list[str] | None
            Agents to include. None = all 6 agents.
        """
        self._snapshot_json = snapshot_json
        self._data_dir = data_dir
        self._agents = agents

    def run(self, scenario: ScenarioDefinition) -> ScenarioResult:
        """
        Run ensemble on (ticker, as_of_date) and record alignment with expected_direction.

        Parameters
        ----------
        scenario : ScenarioDefinition
            The scenario to evaluate.

        Returns
        -------
        ScenarioResult
            Full alignment record including serialised EnsembleOutput.
        """
        from hifi.agents.ensemble_runner import run_ensemble

        output = run_ensemble(
            ticker=scenario.ticker,
            as_of_date=scenario.as_of_date,
            snapshot_json=self._snapshot_json,
            data_dir=self._data_dir,
            agents=self._agents,
        )

        collective = output.ensemble_decision.collective_decision
        aligned = _check_alignment(collective, scenario.expected_direction)

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            ticker=scenario.ticker,
            as_of_date=scenario.as_of_date,
            collective_decision=collective,
            expected_direction=scenario.expected_direction,
            aligned=aligned,
            ensemble_output=output.model_dump(),
        )


# ---------------------------------------------------------------------------
# Built-in Phase 13 scenarios (DJ-078)
# ---------------------------------------------------------------------------

PHASE13_SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        scenario_id="F-001",
        ticker="AAPL",
        as_of_date="2020-03-16",
        event_description=(
            "Black Monday II: COVID crash. S&P 500 fell 12% in a single session. "
            "VIX spiked above 80. Federal Reserve cut rates 100 bps in emergency action."
        ),
        expected_direction="Risk-Off",
        regime="crash",
    ),
    ScenarioDefinition(
        scenario_id="F-001b",
        ticker="JPM",
        as_of_date="2020-03-16",
        event_description=(
            "Black Monday II: COVID crash. Banking sector under acute stress. "
            "Loan-loss provisions expected to spike."
        ),
        expected_direction="Risk-Off",
        regime="crash",
    ),
    ScenarioDefinition(
        scenario_id="F-001c",
        ticker="XOM",
        as_of_date="2020-03-16",
        event_description=(
            "Black Monday II: COVID crash combined with oil price war. "
            "WTI crude fell below $30/bbl."
        ),
        expected_direction="Sell",
        regime="crash",
    ),
    ScenarioDefinition(
        scenario_id="F-002",
        ticker="AAPL",
        as_of_date="2022-03-31",
        event_description=(
            "Fed rate shock: CPI at 8.5% (40-year high). Fed signalled aggressive "
            "rate hiking path. Growth stocks under pressure from rising discount rates."
        ),
        expected_direction="Risk-Off",
        regime="rate_shock",
    ),
    ScenarioDefinition(
        scenario_id="F-002b",
        ticker="JPM",
        as_of_date="2022-03-31",
        event_description=(
            "Fed rate shock: rising rates benefit bank net interest margins but "
            "increase recession risk. Mixed signal for financials."
        ),
        expected_direction="Hold",
        regime="rate_shock",
    ),
    ScenarioDefinition(
        scenario_id="F-002c",
        ticker="XOM",
        as_of_date="2022-03-31",
        event_description=(
            "Fed rate shock + Russia/Ukraine: energy prices elevated, sanctions "
            "creating supply disruption. Macro tailwind for energy sector."
        ),
        expected_direction="Buy",
        regime="rate_shock",
    ),
    ScenarioDefinition(
        scenario_id="F-003",
        ticker="AAPL",
        as_of_date="2023-02-02",
        event_description=(
            "AAPL Q1 FY2023 earnings beat: EPS $1.88 vs $1.94 estimate — slight miss "
            "on EPS but beat on revenue. Services growth strong. Stock rallied +2.4% "
            "after hours. Provides a positive earnings surprise test."
        ),
        expected_direction="Buy",
        regime="earnings_beat",
    ),
]


def _check_alignment(
    collective_decision: str | None,
    expected_direction: str,
) -> bool:
    """
    Determine whether collective_decision is aligned with expected_direction.

    "Risk-Off" expects Hold or Sell (any defensive signal).
    Exact match is required for Buy, Hold, Sell.
    None decision (all agents failed) is never aligned.
    """
    if collective_decision is None:
        return False
    if expected_direction == "Risk-Off":
        return collective_decision in ("Hold", "Sell")
    return collective_decision == expected_direction
