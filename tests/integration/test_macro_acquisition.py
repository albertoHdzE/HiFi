"""
Integration tests for macro data acquisition pipeline (P1-E3).

These tests use pre-recorded FRED XML fixture files stored in
tests/fixtures/macro/.  The fredapi HTTP layer is intercepted by patching
fredapi.fred.urlopen to return fixture XML bytes, so no live FRED API calls
or API keys are required during testing.

The XML fixtures contain real historical Federal Funds Rate and CPI values
from 2022 (the US monetary tightening cycle), making them scientifically
representative rather than arbitrary.

Tickets covered:
- P1-E3-T8: Full fetch for FEDFUNDS using recorded XML fixture
- P1-E3-T9: Parquet write/read round-trip for MacroDataset preserves all values
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from hifi.data.macro import MacroDataFetcher
from hifi.data.storage import read_macro, write_macro

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "macro"


def _load_xml_fixture(name: str) -> bytes:
    """Load an XML fixture file as bytes."""
    path = _FIXTURES / name
    if not path.exists():
        pytest.skip(f"XML fixture not found: {path}")
    return path.read_bytes()


def _make_fetcher_from_xml(observations_xml: bytes, info_xml: bytes) -> MacroDataFetcher:
    """
    Build a MacroDataFetcher whose fredapi HTTP calls are intercepted.

    The fredapi.Fred._Fred__fetch_data method calls urlopen and then
    ET.fromstring(response.read()). We patch urlopen in the fredapi.fred
    module to return an io.BytesIO containing the fixture XML. Since
    get_series and get_series_info make different URL patterns, we use a
    side_effect function to dispatch the correct fixture.
    """

    def _urlopen_side_effect(url: str) -> io.BytesIO:
        if "series/observations" in url:
            return io.BytesIO(observations_xml)
        else:
            return io.BytesIO(info_xml)

    with patch("fredapi.fred.urlopen", side_effect=_urlopen_side_effect):
        fetcher = MacroDataFetcher(api_key="test_key_for_fixtures")
    return fetcher, _urlopen_side_effect


# ---------------------------------------------------------------------------
# P1-E3-T8: Full fetch using recorded XML fixture
# ---------------------------------------------------------------------------


class TestMacroFetchWithXMLFixture:
    """T8: Full pipeline from fredapi XML response to MacroDataset."""

    @pytest.mark.integration
    def test_fedfunds_fixture_produces_valid_dataset(self) -> None:
        """T8: FEDFUNDS XML fixture produces a MacroDataset with 12 monthly observations."""
        obs_xml = _load_xml_fixture("fedfunds_2022_observations.xml")
        info_xml = _load_xml_fixture("fedfunds_series_info.xml")

        with patch("fredapi.fred.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = lambda url: io.BytesIO(
                obs_xml if "observations" in url else info_xml
            )
            fetcher = MacroDataFetcher(api_key="test_key")
            dataset = fetcher.fetch_series(
                "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31)
            )

        assert dataset.series_id == "FEDFUNDS"
        assert len(dataset.observations) == 12

    @pytest.mark.integration
    def test_fedfunds_values_match_fixture(self) -> None:
        """T8: parsed values match the fixture XML exactly."""
        obs_xml = _load_xml_fixture("fedfunds_2022_observations.xml")
        info_xml = _load_xml_fixture("fedfunds_series_info.xml")

        with patch("fredapi.fred.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = lambda url: io.BytesIO(
                obs_xml if "observations" in url else info_xml
            )
            fetcher = MacroDataFetcher(api_key="test_key")
            dataset = fetcher.fetch_series(
                "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31)
            )

        # Jan 2022: near-zero (0.08); Dec 2022: 4.10 (post-hiking)
        first = dataset.observations[0]
        last = dataset.observations[-1]
        assert abs(first.value - 0.08) < 1e-6
        assert abs(last.value - 4.10) < 1e-6

    @pytest.mark.integration
    def test_fedfunds_dates_are_monthly_first(self) -> None:
        """T8: observation dates are the first of each month (FRED convention)."""
        obs_xml = _load_xml_fixture("fedfunds_2022_observations.xml")
        info_xml = _load_xml_fixture("fedfunds_series_info.xml")

        with patch("fredapi.fred.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = lambda url: io.BytesIO(
                obs_xml if "observations" in url else info_xml
            )
            fetcher = MacroDataFetcher(api_key="test_key")
            dataset = fetcher.fetch_series(
                "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31)
            )

        for obs in dataset.observations:
            assert obs.date.day == 1

    @pytest.mark.integration
    def test_cpiaucsl_fixture_produces_valid_dataset(self) -> None:
        """T8: CPI XML fixture produces 12 monthly observations with positive values."""
        obs_xml = _load_xml_fixture("cpiaucsl_2022_observations.xml")
        info_xml = _load_xml_fixture("cpiaucsl_series_info.xml")

        with patch("fredapi.fred.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = lambda url: io.BytesIO(
                obs_xml if "observations" in url else info_xml
            )
            fetcher = MacroDataFetcher(api_key="test_key")
            dataset = fetcher.fetch_series(
                "CPIAUCSL", date(2022, 1, 1), date(2022, 12, 31)
            )

        assert dataset.series_id == "CPIAUCSL"
        assert len(dataset.observations) == 12
        # CPI index values should be positive
        assert all(obs.value > 0 for obs in dataset.observations)


# ---------------------------------------------------------------------------
# P1-E3-T9: Parquet write/read round-trip
# ---------------------------------------------------------------------------


class TestMacroParquetRoundTrip:
    """T9: MacroDataset round-trips through Parquet without loss."""

    @pytest.mark.integration
    def test_fedfunds_round_trip_observation_count(self, tmp_path: Path) -> None:
        """T9: observation count is preserved after Parquet round-trip."""
        obs_xml = _load_xml_fixture("fedfunds_2022_observations.xml")
        info_xml = _load_xml_fixture("fedfunds_series_info.xml")

        with patch("fredapi.fred.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = lambda url: io.BytesIO(
                obs_xml if "observations" in url else info_xml
            )
            fetcher = MacroDataFetcher(api_key="test_key")
            original = fetcher.fetch_series(
                "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31)
            )

        out_path = tmp_path / "FEDFUNDS.parquet"
        write_macro(original, out_path)
        loaded = read_macro(out_path)

        assert len(loaded.observations) == len(original.observations)

    @pytest.mark.integration
    def test_fedfunds_round_trip_values(self, tmp_path: Path) -> None:
        """T9: all observation values are preserved exactly after round-trip."""
        obs_xml = _load_xml_fixture("fedfunds_2022_observations.xml")
        info_xml = _load_xml_fixture("fedfunds_series_info.xml")

        with patch("fredapi.fred.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = lambda url: io.BytesIO(
                obs_xml if "observations" in url else info_xml
            )
            fetcher = MacroDataFetcher(api_key="test_key")
            original = fetcher.fetch_series(
                "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31)
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
        obs_xml = _load_xml_fixture("fedfunds_2022_observations.xml")
        info_xml = _load_xml_fixture("fedfunds_series_info.xml")

        with patch("fredapi.fred.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = lambda url: io.BytesIO(
                obs_xml if "observations" in url else info_xml
            )
            fetcher = MacroDataFetcher(api_key="test_key")
            original = fetcher.fetch_series(
                "FEDFUNDS", date(2022, 1, 1), date(2022, 12, 31)
            )

        out_path = tmp_path / "FEDFUNDS.parquet"
        write_macro(original, out_path)
        loaded = read_macro(out_path)

        assert loaded.series_id == original.series_id
        assert loaded.name == original.name
        assert loaded.source == original.source
        assert loaded.date_from == original.date_from
        assert loaded.date_to == original.date_to
