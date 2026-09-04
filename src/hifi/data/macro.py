"""
Macroeconomic data acquisition from the Federal Reserve Economic Data (FRED).

Provides MacroDataFetcher which downloads time series from FRED via the fredapi
library and normalises them to MacroDataset schemas.

The 7 series fetched for Phase 1 (David section 10.2, Macro Agent context):
  FEDFUNDS   - Federal Funds Effective Rate (monthly, %)
  CPIAUCSL   - Consumer Price Index, All Items (monthly, index)
  UNRATE     - Unemployment Rate (monthly, %)
  GS10       - 10-Year Treasury Constant Maturity Rate (monthly, %)
  GS2        - 2-Year Treasury Constant Maturity Rate (monthly, %)
  VIXCLS     - CBOE Volatility Index (daily, index)
  A191RL1Q225SBEA - Real GDP Growth Rate (quarterly, %)

Design decisions:
- Raw data is stored at native FRED publication frequency (monthly, quarterly,
  daily). Forward-filling to daily is a separate, explicit operation via
  forward_fill_to_daily() -- it is not baked into the stored dataset. This
  keeps the raw data pure and the transformation testable in isolation.
- FRED returns '.' for missing values; fredapi converts these to NaN. We keep
  NaN observations in the dataset as-is so the quality checker can measure them.
- A FRED API key must be available either via the FRED_API_KEY environment
  variable or passed explicitly to MacroDataFetcher. The fetcher raises
  ValueError at construction time if no key is found (same behaviour as fredapi).
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, date, datetime
from typing import cast

import pandas as pd

from hifi.data.schemas import MacroDataset, MacroIndicator, ProvenanceRecord

logger = logging.getLogger(__name__)

# Human-readable metadata for each series we track
SERIES_METADATA: dict[str, dict[str, str]] = {
    "FEDFUNDS": {
        "name": "Federal Funds Effective Rate",
        "frequency": "monthly",
        "unit": "percent",
    },
    "CPIAUCSL": {
        "name": "Consumer Price Index for All Urban Consumers: All Items",
        "frequency": "monthly",
        "unit": "index_1982_84_100",
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "frequency": "monthly",
        "unit": "percent",
    },
    "GS10": {
        "name": "10-Year Treasury Constant Maturity Rate",
        "frequency": "monthly",
        "unit": "percent",
    },
    "GS2": {
        "name": "2-Year Treasury Constant Maturity Rate",
        "frequency": "monthly",
        "unit": "percent",
    },
    "VIXCLS": {
        "name": "CBOE Volatility Index: VIX",
        "frequency": "daily",
        "unit": "index",
    },
    "A191RL1Q225SBEA": {
        "name": "Real Gross Domestic Product",
        "frequency": "quarterly",
        "unit": "percent_change",
    },
}


def _now_utc() -> datetime:
    return datetime.now(UTC)


class MacroDataFetcher:
    """
    Downloads macroeconomic time series from FRED and normalises to MacroDataset.

    Parameters
    ----------
    api_key : str | None
        FRED API key. If None, the FRED_API_KEY environment variable is used.
        Raises ValueError at construction if no key is found anywhere.
    source : str
        Source label attached to provenance records (default "FRED").
    """

    def __init__(self, api_key: str | None = None, source: str = "FRED") -> None:
        from fredapi import Fred

        self._fred = Fred(api_key=api_key)
        self._source = source

    def fetch_series(
        self,
        series_id: str,
        start: date,
        end: date,
        _test_series: pd.Series | None = None,
        _test_series_info: pd.Series | None = None,
    ) -> MacroDataset:
        """
        Download a FRED series at its native publication frequency.

        Parameters
        ----------
        series_id : str
            FRED series identifier (e.g. "FEDFUNDS").
        start : date
            Earliest observation date to include.
        end : date
            Latest observation date to include.

        Returns
        -------
        MacroDataset
            Dataset at native frequency. NaN values (FRED missing marker '.')
            are included as NaN observations so the quality layer can count them.
        """
        fetched_at = _now_utc()
        start_str = start.isoformat()
        end_str = end.isoformat()

        raw_series = (
            _test_series if _test_series is not None
            else self._get_series(series_id, start_str, end_str)
        )
        series_info = (
            _test_series_info if _test_series_info is not None
            else self._get_series_info(series_id)
        )

        observations = self._normalise(series_id, raw_series)

        meta = SERIES_METADATA.get(series_id, {})
        name = series_info.get("title") or meta.get("name", series_id)
        frequency = series_info.get("frequency") or meta.get("frequency", "unknown")
        unit = series_info.get("units") or meta.get("unit", "unknown")

        provenance = ProvenanceRecord(
            source=self._source,
            fetched_at=fetched_at,
            parameters={
                "series_id": series_id,
                "start": start_str,
                "end": end_str,
            },
        )

        return MacroDataset(
            series_id=series_id,
            name=name,
            frequency=frequency,
            unit=unit,
            observations=observations,
            source=self._source,
            fetched_at=fetched_at,
            date_from=start,
            date_to=end,
            provenance=provenance,
        )

    def _get_series(
        self, series_id: str, start: str, end: str
    ) -> pd.Series:
        """Issue the FRED series request. Isolated for patching in tests."""
        # fredapi ships no stubs; cast keeps the declared contract explicit.
        return cast("pd.Series", self._fred.get_series(
            series_id,
            observation_start=start,
            observation_end=end,
        ))

    def _get_series_info(self, series_id: str) -> pd.Series:
        """Issue the FRED series info request. Isolated for patching in tests."""
        return cast("pd.Series", self._fred.get_series_info(series_id))

    def _normalise(
        self, series_id: str, raw: pd.Series
    ) -> list[MacroIndicator]:
        """Convert a raw fredapi Series to a list of MacroIndicator objects."""
        observations: list[MacroIndicator] = []
        for ts, value in raw.items():
            obs_date: date = (ts.date() if isinstance(ts, pd.Timestamp)
                              else pd.Timestamp(str(ts)).date())

            if not isinstance(value, float):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = float("nan")

            observations.append(
                MacroIndicator(series_id=series_id, date=obs_date, value=value)
            )
        return observations


def forward_fill_to_daily(
    dataset: MacroDataset,
    date_range: list[date],
) -> dict[date, float]:
    """
    Forward-fill a MacroDataset's observations to cover every date in date_range.

    This implements the point-in-time safe alignment needed to combine macro
    indicators with daily OHLCV data. The rule is: on any given day, an agent
    sees the most recently published value, not a future revision.

    Documented assumption: we carry the last published value forward until the
    next publication date. This means GDP (quarterly) will show the same value
    for approximately 90 days. This is correct -- an agent in March 2022 knows
    the Q4 2021 GDP figure but not the Q1 2022 figure.

    Parameters
    ----------
    dataset : MacroDataset
        Source dataset at native (non-daily) frequency.
    date_range : list[date]
        Target dates to align to (typically all US trading days in a period).

    Returns
    -------
    dict[date, float]
        Mapping from each date in date_range to the forward-filled value.
        Dates before the first observation have no entry (not included in the
        output rather than filled with NaN to avoid silent look-ahead).
    """
    if not dataset.observations:
        return {}

    # Build a sorted list of (obs_date, value) pairs
    obs_sorted = sorted(
        [(obs.date, obs.value) for obs in dataset.observations if not math.isnan(obs.value)],
        key=lambda x: x[0],
    )

    if not obs_sorted:
        return {}

    result: dict[date, float] = {}
    obs_idx = 0

    for target_date in sorted(date_range):
        # Advance obs_idx to the last observation that is <= target_date
        while (
            obs_idx + 1 < len(obs_sorted)
            and obs_sorted[obs_idx + 1][0] <= target_date
        ):
            obs_idx += 1

        # If target_date is before the first observation, skip it
        if obs_sorted[obs_idx][0] > target_date:
            continue

        result[target_date] = obs_sorted[obs_idx][1]

    return result
