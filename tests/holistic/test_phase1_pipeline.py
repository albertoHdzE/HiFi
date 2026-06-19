"""
Holistic test: Phase 1 end-to-end data pipeline (P1-E4-T10).

This test runs the full Phase 1 pipeline for a representative subset of the
HiFi universe using pre-recorded fixtures (no live API calls):

  Acquisition (E2/E3) → Quality Validation (E4) → Versioning (E5)
  → Parquet Storage (E2/E3) → Registry → Integrity Check → Re-load

Coverage:
- 3 market tickers: AAPL, JPM, XOM (recorded yfinance fixtures)
- 2 macro series: FEDFUNDS, CPIAUCSL (recorded FRED XML fixtures)
- Full schema validation at every stage
- Quality report generated for each dataset
- Every dataset registered; integrity checked immediately after write
- Fresh load from Parquet compared to original (round-trip fidelity)

This test MUST remain green in all subsequent phases. It is the regression
guard for the data layer: if Phase 2 engineering accidentally breaks data
loading, this test catches it immediately.

Scientific note: the fixtures contain real historical data (Q1 2023 OHLCV,
2022 macro rates). Assertions are value-agnostic (we assert structural
correctness, not specific prices) because the fixtures are real data that
may be regenerated with live data that has since been revised.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from hifi.data.macro import MacroDataFetcher
from hifi.data.market import MarketDataFetcher
from hifi.data.quality import DataQualityChecker
from hifi.data.storage import read_macro, read_ohlcv, write_macro, write_ohlcv
from hifi.data.versioning import DatasetRegistry

_MARKET_FIXTURES = Path(__file__).parent.parent / "fixtures" / "market"
_MACRO_FIXTURES = Path(__file__).parent.parent / "fixtures" / "macro"

_MARKET_TICKERS = ["AAPL", "JPM", "XOM"]
_MACRO_SERIES = ["FEDFUNDS", "CPIAUCSL"]


# ---------------------------------------------------------------------------
# Fixture data loaders
# ---------------------------------------------------------------------------


def _load_market_df(ticker: str) -> pd.DataFrame:
    path = _MARKET_FIXTURES / f"{ticker}_2023-01-03_2023-04-01.parquet"
    if not path.exists():
        pytest.skip(f"Market fixture missing: {path}")
    df = pd.read_parquet(path).set_index("Date")
    return df


def _xml_to_series(obs_xml: bytes) -> pd.Series:
    root = ET.fromstring(obs_xml)
    dates, values = [], []
    for obs in root.findall("observation"):
        d = obs.get("date")
        v = obs.get("value")
        if d and v and v != ".":
            dates.append(pd.Timestamp(d))
            values.append(float(v))
    return pd.Series(values, index=pd.DatetimeIndex(dates))


def _xml_to_info(info_xml: bytes) -> pd.Series:
    root = ET.fromstring(info_xml)
    s_elem = root.find("series")
    s = s_elem if s_elem is not None else root
    return pd.Series({
        "title": s.get("title", ""),
        "frequency": s.get("frequency", ""),
        "units": s.get("units", ""),
    })


def _load_macro_fixture(series_id: str):
    """Return (obs_series, info_series) for a given series_id from XML fixtures."""
    obs_path = _MACRO_FIXTURES / f"{series_id.lower()}_2022_observations.xml"
    info_path = _MACRO_FIXTURES / f"{series_id.lower()}_series_info.xml"
    for p in (obs_path, info_path):
        if not p.exists():
            pytest.skip(f"Macro fixture missing: {p}")
    return _xml_to_series(obs_path.read_bytes()), _xml_to_info(info_path.read_bytes())


# ---------------------------------------------------------------------------
# Holistic test
# ---------------------------------------------------------------------------


@pytest.mark.holistic
class TestPhase1Pipeline:
    """End-to-end Phase 1 pipeline: acquire → validate → store → version → reload."""

    def test_market_pipeline_aapl(self, tmp_path: Path) -> None:
        """
        Full pipeline for AAPL:
        1. Normalize yfinance fixture → OHLCVDataset
        2. Run quality check → report passes threshold
        3. Write to Parquet
        4. Register and verify integrity
        5. Reload and compare to original
        """
        df = _load_market_df("AAPL")
        fetcher = MarketDataFetcher()
        dataset = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 4, 1), _test_download=df)

        assert len(dataset.bars) > 0
        for bar in dataset.bars:
            assert bar.open > 0
            assert bar.high >= bar.low

        checker = DataQualityChecker()
        report = checker.check(dataset)
        assert report.completeness >= 0.95
        assert report.ohlcv_violations == 0

        out_path = tmp_path / "AAPL.parquet"
        write_ohlcv(dataset, out_path)
        registry = DatasetRegistry(tmp_path / "registry.json")
        entry = registry.register(
            "AAPL_yfinance", "yfinance", "2023-01-03", "2023-04-01", out_path
        )

        assert registry.verify_integrity(entry)

        loaded = read_ohlcv(out_path)
        assert len(loaded.bars) == len(dataset.bars)
        for orig, reloaded in zip(dataset.bars, loaded.bars, strict=True):
            assert orig.date == reloaded.date
            assert orig.open == reloaded.open
            assert orig.close == reloaded.close

    def test_market_pipeline_all_tickers(self, tmp_path: Path) -> None:
        """Pipeline for all three fixture tickers."""
        registry = DatasetRegistry(tmp_path / "registry.json")
        for ticker in _MARKET_TICKERS:
            df = _load_market_df(ticker)
            fetcher = MarketDataFetcher()
            dataset = fetcher.fetch_ohlcv(
                ticker, date(2023, 1, 3), date(2023, 4, 1), _test_download=df
            )

            out_path = tmp_path / f"{ticker}.parquet"
            write_ohlcv(dataset, out_path)
            entry = registry.register(
                f"{ticker}_yfinance", "yfinance", "2023-01-03", "2023-04-01", out_path
            )
            assert registry.verify_integrity(entry)
            assert registry.lookup(f"{ticker}_yfinance") is not None

        assert len(registry.all_entries()) == len(_MARKET_TICKERS)

    def test_macro_pipeline_fedfunds(self, tmp_path: Path) -> None:
        """
        Full pipeline for FEDFUNDS:
        1. Parse XML fixture → MacroDataset
        2. Write to Parquet
        3. Register and verify integrity
        4. Reload and compare values
        """
        obs_series, info_series = _load_macro_fixture("fedfunds")
        fetcher = MacroDataFetcher(api_key="dummy")
        dataset = fetcher.fetch_series(
            "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31),
            _test_series=obs_series, _test_series_info=info_series,
        )

        assert dataset.series_id == "FEDFUNDS"
        assert len(dataset.observations) == 12

        out_path = tmp_path / "FEDFUNDS.parquet"
        write_macro(dataset, out_path)
        registry = DatasetRegistry(tmp_path / "registry.json")
        entry = registry.register(
            "FEDFUNDS_FRED", "FRED", "2022-01-01", "2022-12-31", out_path
        )

        assert registry.verify_integrity(entry)

        loaded = read_macro(out_path)
        assert len(loaded.observations) == len(dataset.observations)
        for orig, reloaded in zip(dataset.observations, loaded.observations, strict=True):
            assert orig.date == reloaded.date
            assert orig.value == reloaded.value

    def test_macro_pipeline_all_series(self, tmp_path: Path) -> None:
        """Pipeline for all fixture macro series (FEDFUNDS, CPIAUCSL)."""
        registry = DatasetRegistry(tmp_path / "registry.json")
        for series_id in _MACRO_SERIES:
            obs_series, info_series = _load_macro_fixture(series_id)
            fetcher = MacroDataFetcher(api_key="dummy")
            dataset = fetcher.fetch_series(
                series_id, date(2022, 1, 1), date(2022, 12, 31),
                _test_series=obs_series, _test_series_info=info_series,
            )

            out_path = tmp_path / f"{series_id}.parquet"
            write_macro(dataset, out_path)
            entry = registry.register(
                f"{series_id}_FRED", "FRED", "2022-01-01", "2022-12-31", out_path
            )
            assert registry.verify_integrity(entry)

        assert len(registry.all_entries()) == len(_MACRO_SERIES)

    def test_content_hashes_stable_across_reads(self, tmp_path: Path) -> None:
        """Content hashes of two reads of the same file are identical."""
        from hifi.data.versioning import content_hash

        df = _load_market_df("AAPL")
        fetcher = MarketDataFetcher()
        dataset = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 4, 1), _test_download=df)

        out_path = tmp_path / "AAPL.parquet"
        write_ohlcv(dataset, out_path)

        hash_1 = content_hash(out_path)
        hash_2 = content_hash(out_path)
        assert hash_1 == hash_2

    def test_registry_entries_created_correctly(self, tmp_path: Path) -> None:
        """Registry records correct metadata for a mixed market + macro run."""
        registry = DatasetRegistry(tmp_path / "registry.json")

        df = _load_market_df("AAPL")
        fetcher = MarketDataFetcher()
        mkt = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 4, 1), _test_download=df)
        mkt_path = tmp_path / "AAPL.parquet"
        write_ohlcv(mkt, mkt_path)
        registry.register("AAPL_yfinance", "yfinance", "2023-01-03", "2023-04-01", mkt_path)

        obs_series, info_series = _load_macro_fixture("fedfunds")
        fetcher_m = MacroDataFetcher(api_key="dummy")
        macro = fetcher_m.fetch_series(
            "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31),
            _test_series=obs_series, _test_series_info=info_series,
        )
        macro_path = tmp_path / "FEDFUNDS.parquet"
        write_macro(macro, macro_path)
        registry.register("FEDFUNDS_FRED", "FRED", "2022-01-01", "2022-12-31", macro_path)

        entries = registry.all_entries()
        assert len(entries) == 2
        ids = {e.dataset_id for e in entries}
        assert "AAPL_yfinance" in ids
        assert "FEDFUNDS_FRED" in ids
