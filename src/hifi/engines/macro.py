"""
Macro snapshot engine for HiFi (P2-E5).

Pure function that extracts a point-in-time cross-section of macroeconomic
indicators from a collection of MacroDatasets. No I/O, no side effects.

Series-to-field mappings (FRED series IDs):
  FEDFUNDS         -> fed_funds_rate   (monthly, %)
  CPIAUCSL         -> cpi_yoy          (monthly index; YoY % computed here)
  UNRATE           -> unemployment_rate (monthly, %)
  GS10             -> yield_10y        (monthly, %)
  GS2              -> yield_2y         (monthly, %)
  GS10 - GS2       -> yield_curve_slope (derived; negative = inverted)
  VIXCLS           -> vix              (daily, index)
  A191RL1Q225SBEA  -> gdp_growth       (quarterly, % — already a rate)

CPI YoY computation:
  CPIAUCSL is a raw price index (base year 1982-84 = 100). The field
  cpi_yoy is the year-over-year percentage change:
      cpi_yoy = (CPI_t - CPI_t-12m) / CPI_t-12m * 100
  The prior-year anchor is forward-filled from the observation closest
  to (as_of_date - 365 days). Returns None when the 12-month-prior
  observation is unavailable (typically dates before the dataset start).

Forward-filling semantics:
  All other series are forward-filled using the existing forward_fill_to_daily()
  function from the data.macro module. This ensures point-in-time correctness:
  on a given date the agent sees the most recently published value, not a
  future revision. GDP (quarterly) will show the same value for ~90 days,
  which correctly reflects the information available to a real investor.

David reference: §4.1 Deterministic-First — given the same datasets and
date, compute_macro_snapshot() returns the same result across all runs.
"""

from __future__ import annotations

from datetime import date, timedelta

from hifi.data.macro import forward_fill_to_daily
from hifi.data.schemas import MacroDataset
from hifi.engines.types import MacroSnapshotResult

# Mapping from MacroSnapshotResult field name to FRED series ID.
# Direct fields: forward-fill and assign.
_DIRECT_FIELDS: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "unemployment_rate": "UNRATE",
    "yield_10y": "GS10",
    "yield_2y": "GS2",
    "vix": "VIXCLS",
    "gdp_growth": "A191RL1Q225SBEA",
}

_CPI_SERIES_ID = "CPIAUCSL"


def _point_in_time(dataset: MacroDataset, as_of: date) -> float | None:
    """
    Return the forward-filled value for a single date from a dataset.

    Wraps forward_fill_to_daily() for a single-date query. Returns None
    when no observation on or before as_of exists.
    """
    result = forward_fill_to_daily(dataset, [as_of])
    return result.get(as_of)


def compute_macro_snapshot(
    datasets: dict[str, MacroDataset],
    as_of_date: date,
) -> MacroSnapshotResult:
    """
    Extract a point-in-time macro cross-section from a collection of datasets.

    Parameters
    ----------
    datasets : dict[str, MacroDataset]
        Mapping from FRED series_id to MacroDataset. Keys must match the
        FRED series IDs listed in the module-level mapping.
    as_of_date : date
        Analysis date. All values are the most recent published observation
        on or before this date.

    Returns
    -------
    MacroSnapshotResult
        Fields are None when no published value is available on or before
        as_of_date for that series.
    """
    fields: dict[str, float | None] = {}

    # Direct forward-fill fields
    for field_name, series_id in _DIRECT_FIELDS.items():
        ds = datasets.get(series_id)
        fields[field_name] = _point_in_time(ds, as_of_date) if ds is not None else None

    # Yield curve slope = GS10 - GS2 (10-2 spread)
    y10 = fields.get("yield_10y")
    y2 = fields.get("yield_2y")
    fields["yield_curve_slope"] = (
        (y10 - y2) if (y10 is not None and y2 is not None) else None
    )

    # CPI YoY: (CPI_t - CPI_{t-12m}) / CPI_{t-12m} * 100
    cpi_ds = datasets.get(_CPI_SERIES_ID)
    if cpi_ds is not None:
        cpi_now = _point_in_time(cpi_ds, as_of_date)
        cpi_prev = _point_in_time(cpi_ds, as_of_date - timedelta(days=365))
        if cpi_now is not None and cpi_prev is not None and cpi_prev != 0.0:
            fields["cpi_yoy"] = (cpi_now - cpi_prev) / cpi_prev * 100.0
        else:
            fields["cpi_yoy"] = None
    else:
        fields["cpi_yoy"] = None

    return MacroSnapshotResult(**fields)
