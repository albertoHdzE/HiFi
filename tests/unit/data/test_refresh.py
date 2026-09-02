"""The refresh path: merge, never overwrite; verify, never assume.

0% covered when written (DJ-135), which is the wrong state for the module whose
entire reason to exist is preventing two specific data losses:

* ``acquire_fundamentals`` writes ``combined.to_parquet(...)`` and yfinance
  serves only five to seven quarters, so a plain re-run buys the newest quarter
  at the cost of the oldest ones. The merge here is what stops that.
* Writing a macro parquet with ``df.to_parquet()`` strips the schema metadata
  ``read_macro`` requires. All seven series became unreadable, ``_load_all_macro``
  swallowed the per-file exception and returned ``{}``, and the macro agent voted
  Hold on 193 of 194 passes (DJ-133c). The round-trip check is what stops that.

Both failures were silent at write time. Both tests below plant the failure and
confirm it is caught.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hifi.data import refresh


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    """Keep the dataset registry out of the real data/ tree.

    ``_register`` constructs ``DatasetRegistry()`` with its default path, which
    resolves to the repository's own data/registry.json. Without this, running
    these tests writes provenance entries whose file_path points into
    /tmp/pytest-of-... — polluting the reproducibility record with rows that
    describe files no longer on disk.

    Patching ``versioning._DEFAULT_REGISTRY_PATH`` does NOT work: it is bound
    as a default argument at class-definition time. The redirection has to
    happen where the class is used.
    """
    import functools

    from hifi.data.versioning import DatasetRegistry

    monkeypatch.setattr(
        refresh, "DatasetRegistry",
        functools.partial(DatasetRegistry, path=tmp_path / "registry.json"))
    return tmp_path / "registry.json"


def _quarterly(periods: list[str], revenue_base: float = 100.0) -> pd.DataFrame:
    idx = pd.to_datetime(periods)
    return pd.DataFrame(
        {"revenue": [revenue_base + i for i in range(len(periods))],
         "net_income": [10.0 + i for i in range(len(periods))]},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Fundamentals: the union rule
# ---------------------------------------------------------------------------


class TestFundamentalsMergeNeverLosesHistory:
    def test_a_short_fresh_response_does_not_truncate_the_store(self, tmp_path):
        """The literal failure this module exists to prevent.

        Local store holds eight quarters; yfinance returns only the newest
        three. A plain overwrite would delete five quarters of history that the
        TTM ratios and the walk-forward both depend on.
        """
        old = _quarterly(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                          "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"])
        path = tmp_path / "fundamentals" / "AAPL" / "quarterly.parquet"
        path.parent.mkdir(parents=True)
        old.to_parquet(path)

        fresh = _quarterly(["2025-09-30", "2025-12-31", "2026-03-31"], revenue_base=500.0)
        with patch.object(refresh, "_fetch", return_value=fresh):
            report = refresh.refresh_ticker("AAPL", tmp_path, quiet=True)

        merged = pd.read_parquet(path)
        assert len(merged) == 9, f"history was truncated to {len(merged)} quarters"
        assert report["added"] == ["2026-03-31"]
        assert len(report["preserved_only_locally"]) == 6

    def test_fresh_values_win_on_overlap_so_restatements_propagate(self, tmp_path):
        old = _quarterly(["2025-12-31"], revenue_base=100.0)
        path = tmp_path / "fundamentals" / "X" / "quarterly.parquet"
        path.parent.mkdir(parents=True)
        old.to_parquet(path)

        restated = _quarterly(["2025-12-31"], revenue_base=999.0)
        with patch.object(refresh, "_fetch", return_value=restated):
            refresh.refresh_ticker("X", tmp_path, quiet=True)

        assert pd.read_parquet(path)["revenue"].iloc[0] == 999.0, (
            "a restatement did not propagate; the store holds a superseded value"
        )

    def test_a_newly_reported_line_item_is_not_dropped(self, tmp_path):
        old = _quarterly(["2025-12-31"])
        path = tmp_path / "fundamentals" / "X" / "quarterly.parquet"
        path.parent.mkdir(parents=True)
        old.to_parquet(path)

        fresh = old.copy()
        fresh["ebitda"] = [42.0]
        with patch.object(refresh, "_fetch", return_value=fresh):
            refresh.refresh_ticker("X", tmp_path, quiet=True)

        assert "ebitda" in pd.read_parquet(path).columns

    def test_the_result_is_sorted_by_period(self, tmp_path):
        old = _quarterly(["2025-12-31"])
        path = tmp_path / "fundamentals" / "X" / "quarterly.parquet"
        path.parent.mkdir(parents=True)
        old.to_parquet(path)
        with patch.object(refresh, "_fetch",
                          return_value=_quarterly(["2024-06-30", "2026-03-31"])):
            refresh.refresh_ticker("X", tmp_path, quiet=True)
        assert pd.read_parquet(path).index.is_monotonic_increasing

    def test_first_ever_fetch_writes_everything(self, tmp_path):
        with patch.object(refresh, "_fetch",
                          return_value=_quarterly(["2025-12-31", "2026-03-31"])):
            report = refresh.refresh_ticker("NEW", tmp_path, quiet=True)
        assert report["status"] == "ok"
        assert report["periods_before"] == 0
        assert report["periods_after"] == 2


class TestFundamentalsRefusesToClobber:
    def test_an_unreadable_existing_file_is_not_overwritten(self, tmp_path):
        path = tmp_path / "fundamentals" / "X" / "quarterly.parquet"
        path.parent.mkdir(parents=True)
        path.write_text("corrupt")
        with patch.object(refresh, "_fetch", return_value=_quarterly(["2026-03-31"])):
            report = refresh.refresh_ticker("X", tmp_path, quiet=True)
        assert report["status"] == "unreadable"
        assert path.read_text() == "corrupt", (
            "an unreadable file was replaced; whatever it held is now gone and "
            "unrecoverable"
        )

    def test_a_failed_fetch_leaves_the_store_untouched(self, tmp_path):
        old = _quarterly(["2025-12-31"])
        path = tmp_path / "fundamentals" / "X" / "quarterly.parquet"
        path.parent.mkdir(parents=True)
        old.to_parquet(path)
        with patch.object(refresh, "_fetch", side_effect=RuntimeError("429")):
            report = refresh.refresh_ticker("X", tmp_path, quiet=True)
        assert report["status"] == "fetch_failed"
        assert len(pd.read_parquet(path)) == 1

    def test_an_empty_response_is_not_treated_as_deletion(self, tmp_path):
        old = _quarterly(["2025-12-31"])
        path = tmp_path / "fundamentals" / "X" / "quarterly.parquet"
        path.parent.mkdir(parents=True)
        old.to_parquet(path)
        with patch.object(refresh, "_fetch", return_value=pd.DataFrame()):
            report = refresh.refresh_ticker("X", tmp_path, quiet=True)
        assert report["status"] == "empty_response"
        assert len(pd.read_parquet(path)) == 1


# ---------------------------------------------------------------------------
# Macro: the round-trip guard
# ---------------------------------------------------------------------------


class _Fred:
    def __init__(self, series: pd.Series):
        self._series = series

    def get_series(self, series_id):
        return self._series


def _fred_series(dates: list[str], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates))


class TestMacroRoundTripGuard:
    def test_a_written_series_is_readable_back(self, tmp_path):
        fred = _Fred(_fred_series(["2026-07-01", "2026-08-01"], [4.3, 4.4]))
        report = refresh.refresh_series("GS10", tmp_path, fred, quiet=True)
        assert report["status"] == "ok"

        from hifi.data.storage import read_macro
        back = read_macro(tmp_path / "macro" / "GS10.parquet")
        assert back.series_id == "GS10"
        assert len(back.observations) == 2

    def test_a_metadata_stripping_write_is_caught(self, tmp_path):
        """DJ-133c, planted.

        ``df.to_parquet()`` drops the schema metadata ``read_macro`` requires.
        At write time nothing complains; the failure surfaced three days later
        as the macro agent voting Hold on 193 of 194 passes. The refresh must
        refuse to report success.
        """
        def _bad_write(dataset, path):
            pd.DataFrame({"date": [o.date for o in dataset.observations],
                          "value": [o.value for o in dataset.observations]}
                         ).to_parquet(path)

        fred = _Fred(_fred_series(["2026-08-01"], [4.4]))
        with patch("hifi.data.storage.write_macro", side_effect=_bad_write):
            report = refresh.refresh_series("GS10", tmp_path, fred, quiet=True)

        assert report["status"] == "unreadable_after_write", (
            "a write that strips schema metadata reported success; this is the "
            "exact defect that blinded the macro agent"
        )

    def test_a_truncating_write_is_caught(self, tmp_path):
        # Readable, right series_id, wrong number of observations.
        from hifi.data.storage import write_macro

        def _truncate(dataset, path):
            dataset.observations = dataset.observations[:1]
            write_macro(dataset, path)

        fred = _Fred(_fred_series(["2026-07-01", "2026-08-01"], [4.3, 4.4]))
        with patch("hifi.data.storage.write_macro", side_effect=_truncate):
            report = refresh.refresh_series("GS10", tmp_path, fred, quiet=True)
        assert report["status"] == "unreadable_after_write"


class TestMacroMerge:
    def test_history_is_unioned_and_revisions_win(self, tmp_path):
        """FRED revises published history in place; both rules matter."""
        fred_old = _Fred(_fred_series(["2026-06-01", "2026-07-01"], [4.0, 4.1]))
        refresh.refresh_series("GS10", tmp_path, fred_old, quiet=True)

        # July revised, August added, June no longer served.
        fred_new = _Fred(_fred_series(["2026-07-01", "2026-08-01"], [4.15, 4.4]))
        report = refresh.refresh_series("GS10", tmp_path, fred_new, quiet=True)

        from hifi.data.storage import read_macro
        obs = {str(o.date): o.value for o in
               read_macro(tmp_path / "macro" / "GS10.parquet").observations}
        assert set(obs) == {"2026-06-01", "2026-07-01", "2026-08-01"}
        assert obs["2026-06-01"] == 4.0, "history dropped when FRED stopped serving it"
        assert obs["2026-07-01"] == 4.15, "a revision did not propagate"
        assert report["was"] == "2026-07-01" and report["now"] == "2026-08-01"

    def test_nan_observations_are_dropped(self, tmp_path):
        fred = _Fred(_fred_series(["2026-06-01", "2026-07-01"], [float("nan"), 4.1]))
        refresh.refresh_series("GS10", tmp_path, fred, quiet=True)
        from hifi.data.storage import read_macro
        assert len(read_macro(tmp_path / "macro" / "GS10.parquet").observations) == 1

    def test_a_failed_fred_call_leaves_the_file_alone(self, tmp_path):
        fred = _Fred(_fred_series(["2026-07-01"], [4.1]))
        refresh.refresh_series("GS10", tmp_path, fred, quiet=True)
        broken = MagicMock()
        broken.get_series.side_effect = RuntimeError("api key")
        report = refresh.refresh_series("GS10", tmp_path, broken, quiet=True)
        assert report["status"] == "fetch_failed"
        from hifi.data.storage import read_macro
        assert len(read_macro(tmp_path / "macro" / "GS10.parquet").observations) == 1

    def test_an_unreadable_existing_file_is_not_clobbered(self, tmp_path):
        path = tmp_path / "macro" / "GS10.parquet"
        path.parent.mkdir(parents=True)
        path.write_text("corrupt")
        report = refresh.refresh_series(
            "GS10", tmp_path, _Fred(_fred_series(["2026-07-01"], [4.1])), quiet=True)
        assert report["status"] == "unreadable"
        assert path.read_text() == "corrupt"

    def test_every_agent_read_series_is_declared(self):
        # These are the series compute_macro_snapshot actually reads. A series
        # absent here has no refresh path at all — which is how VIXCLS, a daily
        # series, went 17 days stale while the agent quoted it as current.
        assert set(refresh.SERIES) >= {
            "VIXCLS", "GS10", "GS2", "FEDFUNDS", "CPIAUCSL", "UNRATE",
            "A191RL1Q225SBEA"}
        assert refresh.SERIES["VIXCLS"] == "daily"


# ---------------------------------------------------------------------------
# Provenance and quality
# ---------------------------------------------------------------------------


class TestRegistryIsWrittenButNeverBlocking:
    def test_a_refresh_records_a_content_hash(self, tmp_path, _isolated_registry):
        """§4.5 reproducibility: a hash says what is in a file, not when it was
        touched. FRED revises history in place, so two refreshes that "changed
        nothing" but hash differently are the only way to notice."""
        fred = _Fred(_fred_series(["2026-07-01"], [4.1]))
        refresh.refresh_series("GS10", tmp_path, fred, quiet=True)

        from hifi.data.versioning import DatasetRegistry
        entry = DatasetRegistry(path=_isolated_registry).lookup("macro/GS10")
        assert entry is not None and entry.content_hash

    def test_an_unwritable_registry_does_not_fail_the_refresh(self, tmp_path, caplog):
        """The registry is an audit trail, not a precondition. Losing an entry
        is a gap in the record; refusing to refresh is a gap in the data."""
        import logging

        fred = _Fred(_fred_series(["2026-07-01"], [4.1]))
        # Patched on hifi.data.refresh, not hifi.data.versioning: the name is
        # bound at import, so patching the source module has no effect.
        with caplog.at_level(logging.WARNING, logger="hifi.data.refresh"), \
             patch.object(refresh, "DatasetRegistry",
                          side_effect=OSError("read-only filesystem")):
            report = refresh.refresh_series("GS10", tmp_path, fred, quiet=True)
        assert report["status"] == "ok"
        assert "registry" in caplog.text.lower()


class TestOhlcvQualityThreshold:
    def test_the_threshold_is_below_the_holiday_floor(self):
        """0.98 is unreachable and would fire on every ticker, every refresh.

        Completeness counts weekdays without subtracting market holidays.
        Measured on AAPL, 2004-01-02 to 2026-09-01: 5,913 weekdays, 5,702 bars,
        211 missing — 9.31/year against 9-10 US market holidays per year, zero
        gaps detected. A threshold that fires on healthy data teaches the reader
        to ignore it, which is how DJ-120 stayed invisible for a month.
        """
        assert refresh.MIN_COMPLETENESS <= 0.96
        assert refresh.MIN_COMPLETENESS >= 0.90, "so loose it would miss a real outage"

    def test_a_ticker_with_a_real_gap_is_flagged(self, tmp_path):
        from hifi.data.schemas import OHLCVBar, OHLCVDataset, ProvenanceRecord

        now = datetime.now(UTC)
        # Two months of weekdays with a five-week hole in the middle.
        days = pd.bdate_range("2026-01-01", "2026-03-31")
        keep = [d for d in days if not (pd.Timestamp("2026-02-01") <= d
                                        <= pd.Timestamp("2026-03-07"))]
        bars = [OHLCVBar(ticker="GAP", date=d.date(), open=10, high=11, low=9,
                         close=10, volume=1000) for d in keep]
        ds = OHLCVDataset(ticker="GAP", bars=bars, source="test", fetched_at=now,
                          date_from=bars[0].date, date_to=bars[-1].date,
                          provenance=ProvenanceRecord(source="test", fetched_at=now))
        with patch("hifi.mcp.financial_server._load_ohlcv", return_value=ds):
            poor = refresh.check_ohlcv_quality(["GAP"], tmp_path, quiet=True)
        assert poor and poor[0]["ticker"] == "GAP"
        assert poor[0]["gaps"] >= 1

    def test_an_unreadable_ticker_is_reported_with_its_error(self, tmp_path):
        with patch("hifi.mcp.financial_server._load_ohlcv",
                   side_effect=FileNotFoundError("no parquet")):
            poor = refresh.check_ohlcv_quality(["MISSING"], tmp_path, quiet=True)
        assert poor[0]["completeness"] is None
        assert "no parquet" in poor[0]["error"]


class TestAgainstTheLiveStore:
    def test_the_universe_passes_its_own_quality_gate(self):
        """Regression guard for the whole OHLCV store."""
        if not Path("data/market").exists():
            pytest.skip("no market store in this checkout")
        from hifi.data.universe import PHASE14_UNIVERSE

        tickers = [e["ticker"] for e in PHASE14_UNIVERSE]
        poor = refresh.check_ohlcv_quality(tickers, Path("data"), quiet=True)
        assert not poor, f"{len(poor)} ticker(s) below the quality floor: {poor[:5]}"


class TestFetchAndEnv:
    """The two helpers the rest of the module leans on."""

    def _yf(self, frames: dict[str, pd.DataFrame]):
        ticker = MagicMock()
        for attr in ("quarterly_income_stmt", "quarterly_balance_sheet",
                     "quarterly_cashflow"):
            setattr(ticker, attr, frames.get(attr, pd.DataFrame()))
        return MagicMock(Ticker=MagicMock(return_value=ticker))

    def test_statements_are_transposed_to_period_rows(self):
        # yfinance returns fields as rows and periods as columns.
        stmt = pd.DataFrame({pd.Timestamp("2026-03-31"): [100.0],
                             pd.Timestamp("2025-12-31"): [90.0]},
                            index=["Total Revenue"])
        with patch.dict("sys.modules", {"yfinance": self._yf(
                {"quarterly_income_stmt": stmt})}):
            got = refresh._fetch("X")
        assert list(got.index) == [pd.Timestamp("2025-12-31"), pd.Timestamp("2026-03-31")]
        assert "Total Revenue" in got.columns

    def test_a_line_item_shared_by_two_statements_is_kept_once(self):
        shared = pd.DataFrame({pd.Timestamp("2026-03-31"): [5.0]}, index=["Net Income"])
        with patch.dict("sys.modules", {"yfinance": self._yf({
                "quarterly_income_stmt": shared, "quarterly_cashflow": shared})}):
            got = refresh._fetch("X")
        assert list(got.columns).count("Net Income") == 1, (
            "a duplicated column label causes a reindex explosion on merge"
        )

    def test_no_statements_returns_none_not_an_empty_frame(self):
        with patch.dict("sys.modules", {"yfinance": self._yf({})}):
            assert refresh._fetch("X") is None

    def test_load_env_reads_the_key_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        (tmp_path / ".env").write_text("OTHER=1\nFRED_API_KEY=abc123\n")
        refresh._load_env(tmp_path)
        import os
        assert os.environ["FRED_API_KEY"] == "abc123"

    def test_load_env_does_not_override_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "from-shell")
        (tmp_path / ".env").write_text("FRED_API_KEY=from-file\n")
        refresh._load_env(tmp_path)
        import os
        assert os.environ["FRED_API_KEY"] == "from-shell"

    def test_a_missing_env_file_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        refresh._load_env(tmp_path)  # must not raise
