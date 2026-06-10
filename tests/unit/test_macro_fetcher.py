"""
Unit tests for MacroDataFetcher and forward_fill_to_daily (P1-E3).

Tests use synthetic pandas Series that match the fredapi output format.
No live FRED API calls are made: _get_series and _get_series_info are patched
to return pre-constructed data.

Tickets covered:
- P1-E3-T5: Fetcher normalises fredapi Series output to MacroDataset schema
- P1-E3-T6: forward_fill_to_daily is deterministic and does not bleed future values
- P1-E3-T7: Fetcher attaches correct provenance metadata
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from hifi.data.macro import MacroDataFetcher, forward_fill_to_daily
from hifi.data.schemas import MacroDataset, MacroIndicator

# ---------------------------------------------------------------------------
# Helpers: build a MacroDataFetcher without requiring a FRED API key
# ---------------------------------------------------------------------------


def _make_fetcher() -> MacroDataFetcher:
    """
    Create a MacroDataFetcher bypassing the fredapi constructor.

    The fredapi.Fred constructor requires an API key.  We bypass it by
    patching the constructor call so unit tests run without credentials.
    """
    with patch("hifi.data.macro.MacroDataFetcher.__init__", return_value=None):
        fetcher = MacroDataFetcher.__new__(MacroDataFetcher)
        fetcher._fred = MagicMock()
        fetcher._source = "FRED"
    return fetcher


def _make_fred_series(dates: list[date], values: list[float]) -> pd.Series:
    """Build a synthetic fredapi-style Series (DatetimeIndex of Timestamps)."""
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.Series(values, index=index)


def _make_series_info(
    title: str = "Federal Funds Effective Rate",
    frequency: str = "Monthly",
    units: str = "Percent",
) -> pd.Series:
    """Build a synthetic fredapi get_series_info result."""
    return pd.Series(
        {
            "id": "FEDFUNDS",
            "title": title,
            "frequency": frequency,
            "frequency_short": "M",
            "units": units,
            "units_short": "%",
            "observation_start": "1954-07-01",
            "observation_end": "2022-12-01",
        }
    )


# ---------------------------------------------------------------------------
# P1-E3-T5: normalisation to MacroDataset
# ---------------------------------------------------------------------------


class TestMacroDataFetcherNormalisation:
    """T5: fredapi output is normalised to MacroDataset."""

    def test_returns_macro_dataset(self) -> None:
        """T5: fetch_series returns a MacroDataset instance."""
        fetcher = _make_fetcher()
        raw = _make_fred_series(
            [date(2022, 1, 1), date(2022, 2, 1)],
            [0.08, 0.08],
        )
        info = _make_series_info()
        fetcher._fred.get_series.return_value = raw
        fetcher._fred.get_series_info.return_value = info
        with patch.object(fetcher, "_get_series", return_value=raw), \
             patch.object(fetcher, "_get_series_info", return_value=info):
            result = fetcher.fetch_series("FEDFUNDS", date(2022, 1, 1), date(2022, 3, 1))
        assert isinstance(result, MacroDataset)

    def test_correct_series_id(self) -> None:
        """T5: series_id in the dataset matches the requested series."""
        fetcher = _make_fetcher()
        raw = _make_fred_series([date(2022, 1, 1)], [0.08])
        info = _make_series_info()
        with patch.object(fetcher, "_get_series", return_value=raw), \
             patch.object(fetcher, "_get_series_info", return_value=info):
            result = fetcher.fetch_series("FEDFUNDS", date(2022, 1, 1), date(2022, 2, 1))
        assert result.series_id == "FEDFUNDS"

    def test_correct_observation_count(self) -> None:
        """T5: number of observations equals the number of rows in the Series."""
        fetcher = _make_fetcher()
        raw = _make_fred_series(
            [date(2022, 1, 1), date(2022, 2, 1), date(2022, 3, 1)],
            [0.08, 0.08, 0.20],
        )
        info = _make_series_info()
        with patch.object(fetcher, "_get_series", return_value=raw), \
             patch.object(fetcher, "_get_series_info", return_value=info):
            result = fetcher.fetch_series("FEDFUNDS", date(2022, 1, 1), date(2022, 4, 1))
        assert len(result.observations) == 3

    def test_values_normalised_correctly(self) -> None:
        """T5: observation values match the synthetic Series."""
        fetcher = _make_fetcher()
        raw = _make_fred_series(
            [date(2022, 1, 1), date(2022, 3, 1)],
            [0.08, 0.20],
        )
        info = _make_series_info()
        with patch.object(fetcher, "_get_series", return_value=raw), \
             patch.object(fetcher, "_get_series_info", return_value=info):
            result = fetcher.fetch_series("FEDFUNDS", date(2022, 1, 1), date(2022, 4, 1))
        assert abs(result.observations[0].value - 0.08) < 1e-9
        assert abs(result.observations[1].value - 0.20) < 1e-9

    def test_observation_dates_are_date_objects(self) -> None:
        """T5: observation dates are datetime.date, not Timestamps."""
        fetcher = _make_fetcher()
        raw = _make_fred_series([date(2022, 1, 1)], [0.08])
        info = _make_series_info()
        with patch.object(fetcher, "_get_series", return_value=raw), \
             patch.object(fetcher, "_get_series_info", return_value=info):
            result = fetcher.fetch_series("FEDFUNDS", date(2022, 1, 1), date(2022, 2, 1))
        assert isinstance(result.observations[0].date, date)

    def test_negative_values_stored(self) -> None:
        """T5: negative values (e.g. negative real rates) are stored as-is."""
        fetcher = _make_fetcher()
        raw = _make_fred_series([date(2021, 1, 1)], [-0.5])
        info = _make_series_info(title="Real Fed Funds Rate", units="Percent")
        with patch.object(fetcher, "_get_series", return_value=raw), \
             patch.object(fetcher, "_get_series_info", return_value=info):
            result = fetcher.fetch_series("RFEDTARMD", date(2021, 1, 1), date(2021, 2, 1))
        assert result.observations[0].value == -0.5

    def test_metadata_from_series_info(self) -> None:
        """T5: name, frequency, and unit are populated from series_info."""
        fetcher = _make_fetcher()
        raw = _make_fred_series([date(2022, 1, 1)], [0.08])
        info = _make_series_info(
            title="Federal Funds Effective Rate",
            frequency="Monthly",
            units="Percent",
        )
        with patch.object(fetcher, "_get_series", return_value=raw), \
             patch.object(fetcher, "_get_series_info", return_value=info):
            result = fetcher.fetch_series("FEDFUNDS", date(2022, 1, 1), date(2022, 2, 1))
        assert result.name == "Federal Funds Effective Rate"
        assert result.frequency == "Monthly"
        assert result.unit == "Percent"


# ---------------------------------------------------------------------------
# P1-E3-T6: forward_fill_to_daily determinism and no future bleed
# ---------------------------------------------------------------------------


class TestForwardFillToDaily:
    """T6: forward-fill is deterministic and point-in-time safe."""

    def _make_macro_dataset(
        self, observations: list[tuple[date, float]]
    ) -> MacroDataset:
        """Build a minimal MacroDataset from a list of (date, value) pairs."""
        from datetime import UTC, datetime

        from hifi.data.schemas import ProvenanceRecord

        obs = [MacroIndicator(series_id="FEDFUNDS", date=d, value=v) for d, v in observations]
        prov = ProvenanceRecord(source="FRED", fetched_at=datetime(2022, 1, 1, tzinfo=UTC))
        return MacroDataset(
            series_id="FEDFUNDS",
            name="Federal Funds Effective Rate",
            frequency="monthly",
            unit="percent",
            observations=obs,
            source="FRED",
            fetched_at=datetime(2022, 1, 1, tzinfo=UTC),
            date_from=date(2022, 1, 1),
            date_to=date(2022, 3, 31),
            provenance=prov,
        )

    def test_daily_range_filled(self) -> None:
        """T6: every day in date_range has an entry in the output."""
        ds = self._make_macro_dataset(
            [(date(2022, 1, 1), 0.08), (date(2022, 2, 1), 0.08)]
        )
        day_range = [date(2022, 1, 1), date(2022, 1, 15), date(2022, 2, 1), date(2022, 2, 15)]
        result = forward_fill_to_daily(ds, day_range)
        # All days on or after first observation should be present
        assert set(result.keys()) == set(day_range)

    def test_forward_fill_carries_value(self) -> None:
        """T6: a day between two monthly observations carries the earlier value."""
        ds = self._make_macro_dataset(
            [(date(2022, 1, 1), 0.08), (date(2022, 2, 1), 0.20)]
        )
        day_range = [date(2022, 1, 15)]
        result = forward_fill_to_daily(ds, day_range)
        # Jan 15 is between Jan 1 (0.08) and Feb 1 (0.20)
        # Point-in-time rule: carry Jan 1 value forward
        assert abs(result[date(2022, 1, 15)] - 0.08) < 1e-9

    def test_no_future_bleed(self) -> None:
        """T6: a date before any observation is excluded from results (no look-ahead)."""
        ds = self._make_macro_dataset([(date(2022, 2, 1), 0.20)])
        # Jan 15 is before the first observation
        day_range = [date(2022, 1, 15), date(2022, 2, 1), date(2022, 2, 15)]
        result = forward_fill_to_daily(ds, day_range)
        assert date(2022, 1, 15) not in result
        assert date(2022, 2, 1) in result

    def test_publication_date_value_is_same(self) -> None:
        """T6: on the publication date itself, the published value is returned."""
        ds = self._make_macro_dataset(
            [(date(2022, 1, 1), 0.08), (date(2022, 2, 1), 0.20)]
        )
        result = forward_fill_to_daily(ds, [date(2022, 2, 1)])
        assert abs(result[date(2022, 2, 1)] - 0.20) < 1e-9

    def test_is_deterministic(self) -> None:
        """T6: calling forward_fill_to_daily twice on the same inputs gives the same result."""
        ds = self._make_macro_dataset(
            [(date(2022, 1, 1), 0.08), (date(2022, 2, 1), 0.20)]
        )
        day_range = [date(2022, 1, 1), date(2022, 1, 15), date(2022, 2, 1)]
        result_a = forward_fill_to_daily(ds, day_range)
        result_b = forward_fill_to_daily(ds, day_range)
        assert result_a == result_b

    def test_empty_dataset_returns_empty_dict(self) -> None:
        """T6: empty observations produce an empty output, not an error."""
        ds = self._make_macro_dataset([])
        result = forward_fill_to_daily(ds, [date(2022, 1, 1)])
        assert result == {}

    def test_nan_observations_excluded(self) -> None:
        """T6: NaN observations are excluded from forward-fill (not propagated)."""

        ds = self._make_macro_dataset(
            [(date(2022, 1, 1), float("nan")), (date(2022, 2, 1), 0.20)]
        )
        result = forward_fill_to_daily(ds, [date(2022, 1, 15)])
        # Jan 15 is before the first non-NaN value (Feb 1), so excluded
        assert date(2022, 1, 15) not in result


# ---------------------------------------------------------------------------
# P1-E3-T7: provenance metadata
# ---------------------------------------------------------------------------


class TestMacroDataFetcherProvenance:
    """T7: provenance record is attached with correct request parameters."""

    def test_provenance_source_is_fred(self) -> None:
        """T7: provenance.source matches the fetcher's source label."""
        fetcher = _make_fetcher()
        raw = _make_fred_series([date(2022, 1, 1)], [0.08])
        info = _make_series_info()
        with patch.object(fetcher, "_get_series", return_value=raw), \
             patch.object(fetcher, "_get_series_info", return_value=info):
            result = fetcher.fetch_series("FEDFUNDS", date(2022, 1, 1), date(2022, 2, 1))
        assert result.provenance.source == "FRED"

    def test_provenance_parameters_contain_series_id(self) -> None:
        """T7: provenance.parameters records the series_id."""
        fetcher = _make_fetcher()
        raw = _make_fred_series([date(2022, 1, 1)], [0.08])
        info = _make_series_info()
        with patch.object(fetcher, "_get_series", return_value=raw), \
             patch.object(fetcher, "_get_series_info", return_value=info):
            result = fetcher.fetch_series("FEDFUNDS", date(2022, 1, 1), date(2022, 2, 1))
        assert result.provenance.parameters["series_id"] == "FEDFUNDS"

    def test_provenance_parameters_contain_date_range(self) -> None:
        """T7: provenance.parameters records start and end dates."""
        fetcher = _make_fetcher()
        raw = _make_fred_series([date(2022, 1, 1)], [0.08])
        info = _make_series_info()
        with patch.object(fetcher, "_get_series", return_value=raw), \
             patch.object(fetcher, "_get_series_info", return_value=info):
            result = fetcher.fetch_series("FEDFUNDS", date(2022, 1, 1), date(2022, 2, 1))
        assert result.provenance.parameters["start"] == "2022-01-01"
        assert result.provenance.parameters["end"] == "2022-02-01"
