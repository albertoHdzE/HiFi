"""Tests for the canonical OHLCV store resolver (DJ-120).

These are regression tests for the data-starvation defect: five call sites
globbed the legacy flat fixture pattern, so 83 of 98 tickers resolved to
nothing and the MCP tools reported TICKER_NOT_FOUND. The agents read that
absence as bearish evidence rather than as an error, so the bug presented as a
plausible trading signal. The invariant worth protecting is therefore not just
"nested wins" but "a ticker present in the canonical store is never reported
missing".
"""

from __future__ import annotations

import pandas as pd
import pytest

from hifi.data.market_store import coverage_report, market_dir, resolve_ohlcv_path


def _write_nested(root, ticker, last="2026-08-13", rows=5):
    d = root / "market" / ticker
    d.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(end=last, periods=rows, freq="D", name="Date")
    pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100.0},
        index=idx,
    ).to_parquet(d / "ohlcv.parquet")
    return d / "ohlcv.parquet"


def _write_flat(root, ticker, tag="2016-01-01_2023-06-30", rows=5):
    d = root / "market"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{ticker}_{tag}.parquet"
    pd.DataFrame({
        "Date": pd.date_range("2023-06-26", periods=rows, freq="D"),
        "Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5,
        "Adj Close": 1.5, "Volume": 100.0,
    }).to_parquet(path)
    return path


class TestResolveOhlcvPath:
    def test_resolves_nested_layout(self, tmp_path):
        expected = _write_nested(tmp_path, "ACN")
        assert resolve_ohlcv_path("ACN", tmp_path) == expected

    def test_falls_back_to_flat_fixture(self, tmp_path):
        expected = _write_flat(tmp_path, "AAPL")
        assert resolve_ohlcv_path("AAPL", tmp_path) == expected

    def test_prefers_nested_over_flat(self, tmp_path):
        """The core regression: the stale flat fixture must never shadow the
        canonical store. NVDA had both; the flat one stopped at 2023-06-30."""
        nested = _write_nested(tmp_path, "NVDA")
        _write_flat(tmp_path, "NVDA")
        assert resolve_ohlcv_path("NVDA", tmp_path) == nested

    def test_raises_when_ticker_absent(self, tmp_path):
        (tmp_path / "market").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="NOPE"):
            resolve_ohlcv_path("NOPE", tmp_path)

    def test_honours_hifi_data_dir(self, tmp_path, monkeypatch):
        _write_nested(tmp_path, "MSFT")
        monkeypatch.setenv("HIFI_DATA_DIR", str(tmp_path))
        assert market_dir() == tmp_path / "market"
        assert resolve_ohlcv_path("MSFT").exists()


class TestCoverageReport:
    def test_reports_layout_rows_and_last_date(self, tmp_path):
        _write_nested(tmp_path, "ACN", last="2026-08-13", rows=7)
        _write_flat(tmp_path, "AAPL")
        rep = coverage_report(["ACN", "AAPL", "GONE"], tmp_path)

        assert rep["ACN"] == {
            "found": True, "layout": "nested", "rows": 7, "last_date": "2026-08-13",
        }
        assert rep["AAPL"]["layout"] == "flat-legacy"
        assert rep["AAPL"]["last_date"] == "2023-06-30"
        assert rep["GONE"]["found"] is False

    def test_surfaces_a_starved_universe(self, tmp_path):
        """The panel exists so a mostly-dark universe is visible before agents
        run on it, instead of being inferred weeks later from bearish signals."""
        _write_nested(tmp_path, "AAPL")
        universe = ["AAPL"] + [f"DARK{i}" for i in range(10)]
        rep = coverage_report(universe, tmp_path)
        assert sum(r["found"] for r in rep.values()) == 1


class TestFinancialServerIntegration:
    def test_load_ohlcv_reads_nested_store(self, tmp_path, monkeypatch):
        """financial_server._load_ohlcv must parse the nested layout, which has
        a DatetimeIndex and no 'Adj Close' column."""
        _write_nested(tmp_path, "ACN", last="2026-08-13", rows=4)
        monkeypatch.setenv("HIFI_DATA_DIR", str(tmp_path))

        from hifi.mcp import financial_server as fs

        ds = fs._load_ohlcv("ACN")
        assert len(ds.bars) == 4
        assert str(ds.date_to) == "2026-08-13"
        # Provenance must not claim these real bars are a test fixture.
        assert ds.source == "market_store"

    def test_load_ohlcv_still_reads_flat_fixture(self, tmp_path, monkeypatch):
        _write_flat(tmp_path, "AAPL")
        monkeypatch.setenv("HIFI_DATA_DIR", str(tmp_path))

        from hifi.mcp import financial_server as fs

        ds = fs._load_ohlcv("AAPL")
        assert len(ds.bars) == 5
        assert ds.source == "fixture"


class TestIndicatorsServerIntegration:
    def test_load_ohlcv_df_prefers_nested(self, tmp_path, monkeypatch):
        """indicators_server duplicates the resolution logic (pandas 1.5.3
        constraint); it must stay in step with market_store."""
        _write_nested(tmp_path, "NVDA", last="2026-08-13", rows=6)
        _write_flat(tmp_path, "NVDA")
        monkeypatch.setenv("HIFI_DATA_DIR", str(tmp_path))

        from hifi.mcp import indicators_server as ind

        df = ind._load_ohlcv_df("NVDA")
        assert len(df) == 6
        assert str(df["date"].max().date()) == "2026-08-13"
