"""
Walk-forward period schedule for Phase 15 (DJ-095).

Period definitions (from DJ-095 decision):

  Period          Dates           Role
  -----------     -----------     ------------------------------------------
  training        2004-2019       Calibration, EDGAR corpus, bootstrap labels
  validation      2020-2021       COVID regime; no re-fitting after this point
  held_out_test   2022-2023       Primary scientific result (rate-shock regime)
  walk_forward    2024-2025       Sequential monthly; causal order enforced

Evaluation frequency: monthly (last calendar day of each month).
Parallelism: tickers within a date are independent; dates are strictly sequential.
"""

from __future__ import annotations

import calendar
from datetime import date
from enum import Enum


# noqa comment, not a StrEnum: converting would change str() and every
# f-string of a member from "WalkForwardPeriod.TRAINING" to "training".
# Period names appear in log lines and CLI echoes on the walk-forward path, so
# that is a behaviour change, not a modernisation.
class WalkForwardPeriod(str, Enum):  # noqa: UP042
    TRAINING = "training"
    VALIDATION = "validation"
    HELD_OUT_TEST = "held_out_test"
    WALK_FORWARD = "walk_forward"


PERIOD_BOUNDARIES: dict[WalkForwardPeriod, tuple[str, str]] = {
    WalkForwardPeriod.TRAINING:      ("2004-01-01", "2019-12-31"),
    WalkForwardPeriod.VALIDATION:    ("2020-01-01", "2021-12-31"),
    WalkForwardPeriod.HELD_OUT_TEST: ("2022-01-01", "2023-12-31"),
    WalkForwardPeriod.WALK_FORWARD:  ("2024-01-01", "2025-12-31"),
}

# Canonical period names accepted as CLI --period argument
PERIOD_ALIASES: dict[str, WalkForwardPeriod] = {
    "training":       WalkForwardPeriod.TRAINING,
    "validation":     WalkForwardPeriod.VALIDATION,
    "held-out-test":  WalkForwardPeriod.HELD_OUT_TEST,
    "held_out_test":  WalkForwardPeriod.HELD_OUT_TEST,
    "walk-forward":   WalkForwardPeriod.WALK_FORWARD,
    "walk_forward":   WalkForwardPeriod.WALK_FORWARD,
}


def generate_month_ends(start_date: str, end_date: str) -> list[str]:
    """
    Return ISO 8601 month-end dates from start_date to end_date (both inclusive).

    Each returned date is the last calendar day of its month.  The first
    returned date is the last day of the month that contains start_date,
    provided that day is >= start_date.

    Parameters
    ----------
    start_date : str
        ISO 8601 start date, e.g. "2022-01-01".
    end_date : str
        ISO 8601 end date (inclusive), e.g. "2023-12-31".

    Returns
    -------
    list[str]
        Month-end dates in ascending chronological order.

    Examples
    --------
    >>> generate_month_ends("2022-01-01", "2022-03-31")
    ['2022-01-31', '2022-02-28', '2022-03-31']
    """
    t0 = date.fromisoformat(start_date)
    t1 = date.fromisoformat(end_date)

    result: list[str] = []
    year, month = t0.year, t0.month

    while True:
        last_day = calendar.monthrange(year, month)[1]
        end_of_month = date(year, month, last_day)
        if end_of_month > t1:
            break
        if end_of_month >= t0:
            result.append(end_of_month.isoformat())
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return result


def classify_period(date_str: str) -> WalkForwardPeriod:
    """
    Return the walk-forward period that a date belongs to.

    Parameters
    ----------
    date_str : str
        ISO 8601 date string, e.g. "2022-06-30".

    Returns
    -------
    WalkForwardPeriod

    Raises
    ------
    ValueError
        If date_str is outside all defined periods (before 2004 or after 2025).
    """
    for period, (start, end) in PERIOD_BOUNDARIES.items():
        if start <= date_str <= end:
            return period
    raise ValueError(
        f"Date {date_str!r} is outside all defined walk-forward periods (2004-2025)"
    )


def get_period_dates(period: WalkForwardPeriod | str) -> list[str]:
    """
    Return all month-end evaluation dates for the given period.

    Parameters
    ----------
    period : WalkForwardPeriod | str
        Period enum value or string alias (e.g. "held-out-test", "held_out_test").

    Returns
    -------
    list[str]
        Month-end dates in ascending order.
    """
    if isinstance(period, str):
        period = PERIOD_ALIASES[period] if period in PERIOD_ALIASES else WalkForwardPeriod(period)
    start, end = PERIOD_BOUNDARIES[period]
    return generate_month_ends(start, end)


def get_multi_period_dates(periods: list[WalkForwardPeriod | str]) -> list[str]:
    """
    Return deduplicated, sorted month-end dates across multiple periods.

    Parameters
    ----------
    periods : list[WalkForwardPeriod | str]
        Periods to combine (e.g. ["validation", "held_out_test"]).

    Returns
    -------
    list[str]
        Sorted, deduplicated month-end dates.
    """
    all_dates: set[str] = set()
    for p in periods:
        all_dates.update(get_period_dates(p))
    return sorted(all_dates)
