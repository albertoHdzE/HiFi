"""DJ-133a: the fundamental agent must receive fundamentals, point-in-time.

For the entire live record the fundamental agent ran on ``build_minimal_snapshot``
-- a Phase 15 walk-forward scaffold whose every financial field is None by
design -- because ``run_agent_pass`` was called without ``snapshot_json`` and the
fallback was silent. Measured 2026-08-24..27: pe/pb/ps/ev_ebitda/roe/roa absent
on 97/97 tickers all four days, and the agent voted Hold on 97/97 for three
consecutive days while real statements sat in data/fundamentals/.

Two properties are asserted here, and the second is the one that makes the
first durable:

1. A snapshot built from real statements yields real ratios.
2. Nothing is ever read before it was filed. The gate is the EDGAR filingDate,
   never the fiscal period end -- AEP's quarter ending 2026-03-31 was not
   public until 2026-05-05, so a period_end gate leaks five weeks.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from hifi.simulation.snapshot import (
    _PERIOD_MATCH_TOLERANCE_DAYS,
    _match_periods,
    build_minimal_snapshot,
    build_pointintime_snapshot,
)


@pytest.fixture
def statements(tmp_path):
    """Four clean quarters plus the trailing all-NaN placeholder the source emits."""
    periods = pd.to_datetime([
        "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31",
    ])
    df = pd.DataFrame(
        {
            "Total Revenue": [100.0, 110.0, 120.0, 130.0, float("nan")],
            "Net Income": [10.0, 11.0, 12.0, 13.0, float("nan")],
            "Total Assets": [1000.0, 1010.0, 1020.0, 1030.0, float("nan")],
            "Total Debt": [400.0, 410.0, 420.0, 430.0, float("nan")],
            "Stockholders Equity": [500.0, 510.0, 520.0, 530.0, float("nan")],
            "Diluted EPS": [1.0, 1.1, 1.2, 1.3, float("nan")],
            "Ordinary Shares Number": [10.0, 10.0, 10.0, 10.0, float("nan")],
        },
        index=periods,
    )
    d = tmp_path / "fundamentals" / "TEST"
    d.mkdir(parents=True)
    df.to_parquet(d / "quarterly.parquet")

    cal = pd.DataFrame({
        "ticker": ["TEST"] * 5,
        "form": ["10-Q"] * 5,
        "period_end": periods,
        # Each filed ~35 days after the period end.
        "filing_date": periods + pd.Timedelta(days=35),
        "accession": [""] * 5,
    })
    cal.to_parquet(tmp_path / "fundamentals" / "filing_calendar.parquet", index=False)

    mkt = tmp_path / "market" / "TEST"
    mkt.mkdir(parents=True)
    bars = pd.DataFrame(
        {"Close": [50.0, 52.0]},
        index=pd.to_datetime(["2026-05-28", "2026-05-29"]),
    )
    bars.index.name = "Date"
    bars.to_parquet(mkt / "ohlcv.parquet")
    return tmp_path


class TestRatiosBecomeComputable:
    def test_snapshot_carries_real_financials(self, statements):
        raw = build_pointintime_snapshot("TEST", "2026-05-29", data_dir=statements)
        assert raw is not None
        d = json.loads(raw)
        assert d["revenue"] is not None, (
            "the whole defect: an agent that cannot see revenue cannot value anything"
        )
        assert d["total_equity"] is not None
        assert d["source"] == "edgar_pointintime"

    def test_flows_are_trailing_twelve_months_not_one_quarter(self, statements):
        d = json.loads(build_pointintime_snapshot("TEST", "2026-05-29", data_dir=statements))
        # The four reported quarters, skipping the NaN placeholder.
        assert d["revenue"] == pytest.approx(100 + 110 + 120 + 130)
        assert d["net_income"] == pytest.approx(10 + 11 + 12 + 13)

    def test_stocks_are_a_single_period_not_summed(self, statements):
        d = json.loads(build_pointintime_snapshot("TEST", "2026-05-29", data_dir=statements))
        assert d["total_equity"] == pytest.approx(530.0)
        assert d["total_assets"] == pytest.approx(1030.0)

    def test_placeholder_row_does_not_blank_the_snapshot(self, statements):
        """Oracle and Medtronic lost their entire balance sheet to this row."""
        d = json.loads(build_pointintime_snapshot("TEST", "2026-05-29", data_dir=statements))
        assert d["total_equity"] is not None
        assert d["market_cap"] == pytest.approx(52.0 * 10.0)

    def test_ratios_actually_compute(self, statements):
        from hifi.data.schemas import FundamentalsSnapshot
        from hifi.engines.fundamental import compute_financial_ratios

        raw = build_pointintime_snapshot("TEST", "2026-05-29", data_dir=statements)
        snap = FundamentalsSnapshot.model_validate_json(raw)
        r = compute_financial_ratios(snap, 52.0)
        for field in ("pe", "pb", "ps", "roe", "roa", "debt_equity"):
            assert getattr(r, field) is not None, f"{field} is None -- DJ-133a again"


class TestNothingIsReadBeforeItWasFiled:
    def test_period_end_is_not_the_gate(self, statements):
        """One day after the period closes, nothing has been filed."""
        assert build_pointintime_snapshot("TEST", "2025-04-01", data_dir=statements) is None

    def test_snapshot_uses_only_filed_periods(self, statements):
        # On 2026-02-10 the 2025-12-31 quarter (filed 2026-02-04) is public,
        # but 2026-03-31 has not even ended.
        d = json.loads(build_pointintime_snapshot("TEST", "2026-02-10", data_dir=statements))
        assert d["period_end"] == "2025-12-31"
        assert d["provenance"]["parameters"]["filing_date"] == "2026-02-04"

    def test_the_day_before_filing_excludes_the_period(self, statements):
        d = json.loads(build_pointintime_snapshot("TEST", "2026-02-03", data_dir=statements))
        assert d["period_end"] == "2025-09-30", (
            "a filing must not be visible on the day before it was filed"
        )

    def test_missing_calendar_returns_none_rather_than_an_empty_snapshot(self, statements):
        (statements / "fundamentals" / "filing_calendar.parquet").unlink()
        assert build_pointintime_snapshot("TEST", "2026-05-29", data_dir=statements) is None


class TestFiscalCalendarsThatAreNotCalendarQuarters:
    """Apple closes Q2 on 2026-03-28 and PepsiCo on 2026-03-21 while the local
    statements say 2026-03-31. An exact-match join drops exactly those firms."""

    def _published(self, period_ends):
        idx = pd.to_datetime(period_ends)
        return pd.DataFrame({
            "ticker": ["X"] * len(idx),
            "period_end": idx,
            "filing_date": idx + pd.Timedelta(days=30),
        })

    def test_apple_style_three_day_offset_matches(self):
        out = _match_periods(
            [pd.Timestamp("2026-03-31")], self._published(["2026-03-28"])
        )
        assert out, "a 3-day fiscal offset must not blind the ticker"

    def test_retail_445_seventeen_day_offset_matches(self):
        out = _match_periods(
            [pd.Timestamp("2026-06-30")], self._published(["2026-06-13"])
        )
        assert out, "PepsiCo/Costco 4-4-5 calendars drift ~2.5 weeks"

    def test_adjacent_quarters_are_never_confused(self):
        """The tolerance must stay far below the ~91-day quarter spacing."""
        assert _PERIOD_MATCH_TOLERANCE_DAYS < 45
        out = _match_periods(
            [pd.Timestamp("2026-06-30")], self._published(["2026-03-31", "2026-06-30"])
        )
        assert out[pd.Timestamp("2026-06-30")] == pd.Timestamp("2026-07-30")

    def test_nearest_period_wins_not_latest_filing(self):
        pub = self._published(["2026-06-13", "2026-06-30"])
        out = _match_periods([pd.Timestamp("2026-06-30")], pub)
        assert out[pd.Timestamp("2026-06-30")] == pd.Timestamp("2026-07-30")


class TestTheScaffoldIsStillHonestlyEmpty:
    """build_minimal_snapshot is retained for the walk-forward harness. It must
    keep advertising that it carries nothing, so it can never again be mistaken
    for a data source."""

    def test_minimal_snapshot_declares_its_emptiness(self):
        d = json.loads(build_minimal_snapshot("AAPL", "2022-01-31"))
        assert d["source"] == "walk_forward_eval"
        for field in ("revenue", "net_income", "total_assets", "total_equity", "eps"):
            assert d[field] is None

    def test_the_two_builders_are_distinguishable_in_the_record(self, statements):
        real = json.loads(build_pointintime_snapshot("TEST", "2026-05-29", data_dir=statements))
        empty = json.loads(build_minimal_snapshot("TEST", "2026-05-29"))
        assert real["source"] != empty["source"], (
            "provenance must separate them: DJ-133a survived two months because "
            "a blind agent looked exactly like a working one"
        )
