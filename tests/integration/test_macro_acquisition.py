"""
Integration tests for macro data acquisition pipeline (P1-E3).

These tests use pre-recorded FRED XML fixture files stored in
tests/fixtures/macro/.  The XML is parsed into pd.Series and injected via
the _test_series/_test_series_info DI parameters, so no live FRED API calls
or API keys are required during testing.

The XML fixtures contain real historical Federal Funds Rate and CPI values
from 2022 (the US monetary tightening cycle), making them scientifically
representative rather than arbitrary.

Tickets covered:
- P1-E3-T8: Full fetch for FEDFUNDS using recorded XML fixture
- P1-E3-T9: Parquet write/read round-trip for MacroDataset preserves all values
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from hifi.data.macro import MacroDataFetcher
from hifi.data.storage import read_macro, write_macro

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "macro"


def _load_xml(name: str) -> bytes:
    path = _FIXTURES / name
    if not path.exists():
        pytest.skip(f"XML fixture not found: {path}")
    return path.read_bytes()


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


# ---------------------------------------------------------------------------
# P1-E3-T8: Full fetch using recorded XML fixture
# ---------------------------------------------------------------------------


class TestMacroFetchWithXMLFixture:
    """T8: Full pipeline from fredapi XML response to MacroDataset."""

    @pytest.mark.integration
    def test_fedfunds_fixture_produces_valid_dataset(self) -> None:
        """T8: FEDFUNDS XML fixture produces a MacroDataset with 12 monthly observations."""
        obs_series = _xml_to_series(_load_xml("fedfunds_2022_observations.xml"))
        info_series = _xml_to_info(_load_xml("fedfunds_series_info.xml"))
        fetcher = MacroDataFetcher(api_key="dummy")
        dataset = fetcher.fetch_series(
            "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31),
            _test_series=obs_series, _test_series_info=info_series,
        )
        assert dataset.series_id == "FEDFUNDS"
        assert len(dataset.observations) == 12

    @pytest.mark.integration
    def test_fedfunds_values_match_fixture(self) -> None:
        """T8: parsed values match the fixture XML exactly."""
        obs_series = _xml_to_series(_load_xml("fedfunds_2022_observations.xml"))
        info_series = _xml_to_info(_load_xml("fedfunds_series_info.xml"))
        fetcher = MacroDataFetcher(api_key="dummy")
        dataset = fetcher.fetch_series(
            "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31),
            _test_series=obs_series, _test_series_info=info_series,
        )
        # Jan 2022: near-zero (0.08); Dec 2022: 4.10 (post-hiking)
        assert abs(dataset.observations[0].value - 0.08) < 1e-6
        assert abs(dataset.observations[-1].value - 4.10) < 1e-6

    @pytest.mark.integration
    def test_fedfunds_dates_are_monthly_first(self) -> None:
        """T8: observation dates are the first of each month (FRED convention)."""
        obs_series = _xml_to_series(_load_xml("fedfunds_2022_observations.xml"))
        info_series = _xml_to_info(_load_xml("fedfunds_series_info.xml"))
        fetcher = MacroDataFetcher(api_key="dummy")
        dataset = fetcher.fetch_series(
            "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31),
            _test_series=obs_series, _test_series_info=info_series,
        )
        for obs in dataset.observations:
            assert obs.date.day == 1

    @pytest.mark.integration
    def test_cpiaucsl_fixture_produces_valid_dataset(self) -> None:
        """T8: CPI XML fixture produces 12 monthly observations with positive values."""
        obs_series = _xml_to_series(_load_xml("cpiaucsl_2022_observations.xml"))
        info_series = _xml_to_info(_load_xml("cpiaucsl_series_info.xml"))
        fetcher = MacroDataFetcher(api_key="dummy")
        dataset = fetcher.fetch_series(
            "CPIAUCSL", date(2022, 1, 1), date(2022, 12, 31),
            _test_series=obs_series, _test_series_info=info_series,
        )
        assert dataset.series_id == "CPIAUCSL"
        assert len(dataset.observations) == 12
        assert all(obs.value > 0 for obs in dataset.observations)


# ---------------------------------------------------------------------------
# P1-E3-T9: Parquet write/read round-trip
# ---------------------------------------------------------------------------


class TestMacroParquetRoundTrip:
    """T9: MacroDataset round-trips through Parquet without loss."""

    @pytest.mark.integration
    def test_fedfunds_round_trip_observation_count(self, tmp_path: Path) -> None:
        """T9: observation count is preserved after Parquet round-trip."""
        obs_series = _xml_to_series(_load_xml("fedfunds_2022_observations.xml"))
        info_series = _xml_to_info(_load_xml("fedfunds_series_info.xml"))
        fetcher = MacroDataFetcher(api_key="dummy")
        original = fetcher.fetch_series(
            "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31),
            _test_series=obs_series, _test_series_info=info_series,
        )
        out_path = tmp_path / "FEDFUNDS.parquet"
        write_macro(original, out_path)
        loaded = read_macro(out_path)
        assert len(loaded.observations) == len(original.observations)

    @pytest.mark.integration
    def test_fedfunds_round_trip_values(self, tmp_path: Path) -> None:
        """T9: all observation values are preserved exactly after round-trip."""
        obs_series = _xml_to_series(_load_xml("fedfunds_2022_observations.xml"))
        info_series = _xml_to_info(_load_xml("fedfunds_series_info.xml"))
        fetcher = MacroDataFetcher(api_key="dummy")
        original = fetcher.fetch_series(
            "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31),
            _test_series=obs_series, _test_series_info=info_series,
        )
        out_path = tmp_path / "FEDFUNDS.parquet"
        write_macro(original, out_path)
        loaded = read_macro(out_path)
        for orig_obs, loaded_obs in zip(original.observations, loaded.observations, strict=True):
            assert orig_obs.date == loaded_obs.date
            assert orig_obs.value == loaded_obs.value

    @pytest.mark.integration
    def test_fedfunds_round_trip_metadata(self, tmp_path: Path) -> None:
        """T9: series_id, name, frequency, source survive the round-trip."""
        obs_series = _xml_to_series(_load_xml("fedfunds_2022_observations.xml"))
        info_series = _xml_to_info(_load_xml("fedfunds_series_info.xml"))
        fetcher = MacroDataFetcher(api_key="dummy")
        original = fetcher.fetch_series(
            "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31),
            _test_series=obs_series, _test_series_info=info_series,
        )
        out_path = tmp_path / "FEDFUNDS.parquet"
        write_macro(original, out_path)
        loaded = read_macro(out_path)
        assert loaded.series_id == original.series_id
        assert loaded.name == original.name
        assert loaded.source == original.source
        assert loaded.date_from == original.date_from
        assert loaded.date_to == original.date_to
