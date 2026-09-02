"""The offline sweep's pipeline and status modes — 0% covered before this.

This is the path the Phase 15 re-run will use. That re-run matters: the original
Phase 15 result is retracted (signals were generated while 83 of 99 tickers had
no data), so whatever this code produces next is what the Page-theorem claim
will rest on. Untested was the wrong state for it.

The property that carries the most weight is checkpoint-resume. A sweep runs for
hours and gets interrupted; if resume were wrong in the silent direction — a
stale portfolio.json accepted as done, or a completed date recomputed against a
changed store — the output would be a mixture of two runs with nothing marking
the boundary.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from hifi.live import paths, walkforward


@pytest.fixture
def store(tmp_path):
    return tmp_path


def _write_bars(root, ticker, dates, closes):
    p = root / "market" / ticker / "ohlcv.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"close": closes}, index=pd.to_datetime(dates)).to_parquet(p)


def _write_ensemble(out_dir, condition, date, ticker, decision, confidence=0.8):
    p = paths._ensemble_path(str(out_dir), condition, date, ticker)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "ticker": ticker,
        "ensemble_decision": {"collective_decision": decision,
                              "collective_confidence": confidence},
    }))


class TestLoadOhlcv:
    def test_it_returns_dated_closes_per_ticker(self, store):
        _write_bars(store, "AAPL", ["2026-08-27", "2026-08-28"], [199.0, 201.0])
        got = walkforward._load_ohlcv(str(store), ["AAPL"], "2026-08-31")
        assert got["AAPL"][-1] == {"date": "2026-08-28", "close": 201.0}

    def test_bars_after_the_as_of_date_are_excluded(self, store):
        """Point-in-time: the sweep must not see the future it is predicting."""
        _write_bars(store, "AAPL", ["2026-08-27", "2026-09-15"], [199.0, 250.0])
        got = walkforward._load_ohlcv(str(store), ["AAPL"], "2026-08-31")
        assert [r["date"] for r in got["AAPL"]] == ["2026-08-27"]

    def test_it_keeps_only_the_trailing_window(self, store):
        dates = pd.bdate_range("2025-01-01", periods=200).strftime("%Y-%m-%d").tolist()
        _write_bars(store, "AAPL", dates, [100.0] * 200)
        got = walkforward._load_ohlcv(str(store), ["AAPL"], "2026-12-31")
        assert len(got["AAPL"]) == 90

    def test_a_missing_ticker_is_omitted(self, store):
        assert walkforward._load_ohlcv(str(store), ["NOPE"], "2026-08-31") == {}

    def test_a_ticker_with_no_bars_before_the_date_is_omitted(self, store):
        _write_bars(store, "NEW", ["2026-09-15"], [10.0])
        assert walkforward._load_ohlcv(str(store), ["NEW"], "2026-08-31") == {}

    def test_column_case_is_normalised(self, store):
        p = store / "market" / "X" / "ohlcv.parquet"
        p.parent.mkdir(parents=True)
        pd.DataFrame({"Close": [50.0]}, index=pd.to_datetime(["2026-08-28"])
                     ).to_parquet(p)
        assert walkforward._load_ohlcv(str(store), ["X"], "2026-08-31")["X"]

    def test_a_corrupt_store_degrades_to_empty_rather_than_raising(self, store,
                                                                   caplog):
        bad = store / "market" / "BAD" / "ohlcv.parquet"
        bad.parent.mkdir(parents=True)
        bad.write_text("corrupt")
        assert walkforward._load_ohlcv(str(store), ["BAD"], "2026-08-31") == {}
        assert "OHLCV load error" in caplog.text


class TestPipelineMode:
    def _setup(self, store, condition="full", date="2026-08-31"):
        for t, d in [("AAPL", "Buy"), ("MSFT", "Buy"), ("JPM", "Hold")]:
            _write_ensemble(store / "wf", condition, date, t, d)
            _write_bars(store, t, ["2026-08-28"], [100.0])
        return ["AAPL", "MSFT", "JPM"]

    def test_it_writes_one_portfolio_per_date(self, store):
        tickers = self._setup(store)
        counts = walkforward.run_pipeline_mode(
            condition="full", dates=["2026-08-31"], tickers=tickers,
            data_dir=str(store), output_dir=str(store / "wf"),
            dry_run=False, quiet=True)
        assert counts["done"] == 1 and counts["fail"] == 0
        assert paths._portfolio_path(str(store / "wf"), "full", "2026-08-31").exists()

    def test_checkpoint_resume_skips_a_completed_date(self, store):
        tickers = self._setup(store)
        port = paths._portfolio_path(str(store / "wf"), "full", "2026-08-31")
        port.parent.mkdir(parents=True, exist_ok=True)
        port.write_text('{"already": "done"}')

        counts = walkforward.run_pipeline_mode(
            condition="full", dates=["2026-08-31"], tickers=tickers,
            data_dir=str(store), output_dir=str(store / "wf"),
            dry_run=False, quiet=True)
        assert counts == {"done": 0, "skip": 1, "fail": 0}
        assert json.loads(port.read_text()) == {"already": "done"}, (
            "a completed date was recomputed; the output would mix two runs"
        )

    def test_a_dry_run_writes_nothing(self, store):
        tickers = self._setup(store)
        counts = walkforward.run_pipeline_mode(
            condition="full", dates=["2026-08-31"], tickers=tickers,
            data_dir=str(store), output_dir=str(store / "wf"),
            dry_run=True, quiet=True)
        assert counts["done"] == 1
        assert not paths._portfolio_path(str(store / "wf"), "full",
                                         "2026-08-31").exists()

    def test_a_date_with_no_ensembles_is_a_failure_not_a_silent_skip(self, store):
        """An empty portfolio would be indistinguishable from a flat decision."""
        counts = walkforward.run_pipeline_mode(
            condition="full", dates=["2026-08-31"], tickers=["AAPL"],
            data_dir=str(store), output_dir=str(store / "wf"),
            dry_run=False, quiet=True)
        assert counts["fail"] == 1 and counts["done"] == 0

    def test_a_corrupt_ensemble_json_is_skipped_not_fatal(self, store, caplog):
        tickers = self._setup(store)
        bad = paths._ensemble_path(str(store / "wf"), "full", "2026-08-31", "MSFT")
        bad.write_text("{ not json")
        counts = walkforward.run_pipeline_mode(
            condition="full", dates=["2026-08-31"], tickers=tickers,
            data_dir=str(store), output_dir=str(store / "wf"),
            dry_run=False, quiet=True)
        assert counts["done"] == 1
        assert "Could not read ensemble JSON" in caplog.text

    def test_a_missing_decision_defaults_to_hold_not_to_buy(self, store):
        # The safe default: absence must never become a position.
        _write_ensemble(store / "wf", "full", "2026-08-31", "AAPL", None)
        _write_bars(store, "AAPL", ["2026-08-28"], [100.0])
        with patch("hifi.simulation.pipeline.run_pipeline") as rp:
            rp.return_value = type("S", (), {"orders": []})()
            walkforward.run_pipeline_mode(
                condition="full", dates=["2026-08-31"], tickers=["AAPL"],
                data_dir=str(store), output_dir=str(store / "wf"),
                dry_run=False, quiet=True)
        assert rp.call_args.args[0][0]["decision"] == "Hold"

    def test_each_date_is_isolated_from_the_others(self, store):
        tickers = self._setup(store, date="2026-08-31")
        self._setup(store, date="2026-07-31")
        counts = walkforward.run_pipeline_mode(
            condition="full", dates=["2026-07-31", "2026-08-31"], tickers=tickers,
            data_dir=str(store), output_dir=str(store / "wf"),
            dry_run=False, quiet=True)
        assert counts["done"] == 2

    def test_every_arm_starts_from_the_same_capital(self):
        # The sweep compares conditions; a differing capital base would make
        # the comparison meaningless.
        assert walkforward._CAPITAL == 500_000.0


class TestStatusMode:
    def test_it_reports_progress_without_running_anything(self, store, capsys):
        _write_ensemble(store / "wf", "full", "2026-08-31", "AAPL", "Buy")
        walkforward.run_status_mode(
            conditions=["full", "parallel"], dates=["2026-08-31"],
            tickers=["AAPL", "MSFT"], data_dir=str(store),
            output_dir=str(store / "wf"))
        out = capsys.readouterr().out
        assert "full" in out and "parallel" in out

    def test_an_empty_store_reports_zero_rather_than_failing(self, store, capsys):
        walkforward.run_status_mode(
            conditions=["full"], dates=["2026-08-31"], tickers=["AAPL"],
            data_dir=str(store), output_dir=str(store / "wf"))
        assert capsys.readouterr().out
