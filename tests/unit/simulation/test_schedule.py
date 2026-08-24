"""
Unit tests for hifi.simulation.schedule (Phase 15 walk-forward date schedule).

Tests cover:
- generate_month_ends: correctness, leap years, edge cases
- classify_period: all four periods, boundary dates, out-of-range error
- get_period_dates: known counts for each period
- get_multi_period_dates: deduplication and ordering
"""

from __future__ import annotations

import pytest

from hifi.simulation.schedule import (
    WalkForwardPeriod,
    classify_period,
    generate_month_ends,
    get_multi_period_dates,
    get_period_dates,
)

# ---------------------------------------------------------------------------
# generate_month_ends
# ---------------------------------------------------------------------------


def test_generate_month_ends_basic():
    dates = generate_month_ends("2022-01-01", "2022-03-31")
    assert dates == ["2022-01-31", "2022-02-28", "2022-03-31"]


def test_generate_month_ends_leap_year():
    dates = generate_month_ends("2020-02-01", "2020-02-29")
    assert dates == ["2020-02-29"]


def test_generate_month_ends_non_leap_year():
    dates = generate_month_ends("2021-02-01", "2021-02-28")
    assert dates == ["2021-02-28"]


def test_generate_month_ends_single_month():
    dates = generate_month_ends("2022-06-01", "2022-06-30")
    assert dates == ["2022-06-30"]


def test_generate_month_ends_start_on_last_day():
    """start_date is the last day of the month → that day is included."""
    dates = generate_month_ends("2022-01-31", "2022-03-31")
    assert dates[0] == "2022-01-31"
    assert "2022-02-28" in dates
    assert "2022-03-31" in dates


def test_generate_month_ends_start_after_end():
    """start_date after end_date → empty list."""
    dates = generate_month_ends("2022-03-01", "2022-01-31")
    assert dates == []


def test_generate_month_ends_same_month_mid():
    """start and end in the same month: returns that month-end."""
    dates = generate_month_ends("2022-06-15", "2022-06-30")
    assert dates == ["2022-06-30"]


def test_generate_month_ends_crosses_year():
    dates = generate_month_ends("2021-11-01", "2022-02-28")
    assert dates == ["2021-11-30", "2021-12-31", "2022-01-31", "2022-02-28"]


def test_generate_month_ends_sorted_ascending():
    dates = generate_month_ends("2022-01-01", "2022-12-31")
    assert dates == sorted(dates)


def test_generate_month_ends_count():
    """2022 has 12 month-ends."""
    dates = generate_month_ends("2022-01-01", "2022-12-31")
    assert len(dates) == 12


def test_generate_month_ends_held_out_count():
    """Held-out 2022-2023 → 24 monthly dates."""
    dates = generate_month_ends("2022-01-01", "2023-12-31")
    assert len(dates) == 24


# ---------------------------------------------------------------------------
# classify_period
# ---------------------------------------------------------------------------


def test_classify_period_training():
    assert classify_period("2010-06-30") == WalkForwardPeriod.TRAINING


def test_classify_period_training_start():
    assert classify_period("2004-01-01") == WalkForwardPeriod.TRAINING


def test_classify_period_training_end():
    assert classify_period("2019-12-31") == WalkForwardPeriod.TRAINING


def test_classify_period_validation():
    assert classify_period("2021-03-31") == WalkForwardPeriod.VALIDATION


def test_classify_period_validation_start():
    assert classify_period("2020-01-01") == WalkForwardPeriod.VALIDATION


def test_classify_period_validation_end():
    assert classify_period("2021-12-31") == WalkForwardPeriod.VALIDATION


def test_classify_period_held_out_test():
    assert classify_period("2022-06-30") == WalkForwardPeriod.HELD_OUT_TEST


def test_classify_period_held_out_start():
    assert classify_period("2022-01-01") == WalkForwardPeriod.HELD_OUT_TEST


def test_classify_period_held_out_end():
    assert classify_period("2023-12-31") == WalkForwardPeriod.HELD_OUT_TEST


def test_classify_period_walk_forward():
    assert classify_period("2024-06-30") == WalkForwardPeriod.WALK_FORWARD


def test_classify_period_walk_forward_start():
    assert classify_period("2024-01-01") == WalkForwardPeriod.WALK_FORWARD


def test_classify_period_walk_forward_end():
    assert classify_period("2025-12-31") == WalkForwardPeriod.WALK_FORWARD


def test_classify_period_out_of_range_early():
    with pytest.raises(ValueError, match="outside all defined"):
        classify_period("2003-12-31")


def test_classify_period_out_of_range_late():
    with pytest.raises(ValueError, match="outside all defined"):
        classify_period("2026-01-01")


# ---------------------------------------------------------------------------
# get_period_dates
# ---------------------------------------------------------------------------


def test_get_period_dates_held_out_count():
    """Held-out test period 2022-2023 → 24 evaluation dates."""
    dates = get_period_dates(WalkForwardPeriod.HELD_OUT_TEST)
    assert len(dates) == 24


def test_get_period_dates_walk_forward_count():
    """Walk-forward period 2024-2025 → 24 evaluation dates."""
    dates = get_period_dates(WalkForwardPeriod.WALK_FORWARD)
    assert len(dates) == 24


def test_get_period_dates_validation_count():
    """Validation period 2020-2021 → 24 evaluation dates."""
    dates = get_period_dates(WalkForwardPeriod.VALIDATION)
    assert len(dates) == 24


def test_get_period_dates_accepts_string_alias():
    dates_enum = get_period_dates(WalkForwardPeriod.HELD_OUT_TEST)
    dates_str_hyphen = get_period_dates("held-out-test")
    dates_str_underscore = get_period_dates("held_out_test")
    assert dates_enum == dates_str_hyphen == dates_str_underscore


def test_get_period_dates_all_in_range():
    """Every date returned by get_period_dates classifies to the correct period."""
    for period in WalkForwardPeriod:
        for d in get_period_dates(period):
            assert classify_period(d) == period


def test_get_period_dates_ordered():
    for period in WalkForwardPeriod:
        dates = get_period_dates(period)
        assert dates == sorted(dates)


def test_get_period_dates_all_month_ends():
    """Every date returned is the last day of its month."""
    import calendar
    from datetime import date

    for period in [WalkForwardPeriod.HELD_OUT_TEST, WalkForwardPeriod.VALIDATION]:
        for d_str in get_period_dates(period):
            d = date.fromisoformat(d_str)
            last_day = calendar.monthrange(d.year, d.month)[1]
            assert d.day == last_day, f"{d_str} is not a month-end"


# ---------------------------------------------------------------------------
# get_multi_period_dates
# ---------------------------------------------------------------------------


def test_get_multi_period_dates_deduplicates():
    """No duplicates across overlapping call."""
    combined = get_multi_period_dates(
        [WalkForwardPeriod.VALIDATION, WalkForwardPeriod.HELD_OUT_TEST]
    )
    assert len(combined) == len(set(combined))


def test_get_multi_period_dates_sorted():
    combined = get_multi_period_dates(
        [WalkForwardPeriod.HELD_OUT_TEST, WalkForwardPeriod.VALIDATION]
    )
    assert combined == sorted(combined)


def test_get_multi_period_dates_total_count():
    combined = get_multi_period_dates(
        [WalkForwardPeriod.VALIDATION, WalkForwardPeriod.HELD_OUT_TEST]
    )
    assert len(combined) == 48  # 24 + 24, no overlap


def test_get_multi_period_dates_accepts_strings():
    combined = get_multi_period_dates(["validation", "held-out-test"])
    assert len(combined) == 48
