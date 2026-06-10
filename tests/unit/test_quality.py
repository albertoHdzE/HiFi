"""
Unit tests for DataQualityChecker and QualityReport (P1-E4).

Tests inject synthetic OHLCVDatasets with known defects to verify that
each check fires precisely when it should and does not fire on clean data.
No file I/O or live API calls in these tests.

Tickets covered:
- P1-E4-T8: Quality checker detects known defects in synthetic bad data
- P1-E4-T9: Quality checker passes clean synthetic data
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from hifi.data.quality import (
    DataQualityChecker,
)
from hifi.data.schemas import OHLCVBar, OHLCVDataset, ProvenanceRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FETCH_TIME = datetime(2023, 6, 1, 12, 0, 0, tzinfo=UTC)


def _prov() -> ProvenanceRecord:
    return ProvenanceRecord(source="test", fetched_at=_FETCH_TIME)


def _bar(
    d: date,
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 98.0,
    close: float = 103.0,
    volume: float = 1_000_000.0,
    adj_close: float | None = 103.0,
    ticker: str = "AAPL",
) -> OHLCVBar:
    return OHLCVBar(
        ticker=ticker,
        date=d,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        adjusted_close=adj_close,
    )


def _dataset(
    bars: list[OHLCVBar],
    date_from: date,
    date_to: date,
    ticker: str = "AAPL",
) -> OHLCVDataset:
    return OHLCVDataset(
        ticker=ticker,
        bars=bars,
        source="test",
        fetched_at=_FETCH_TIME,
        date_from=date_from,
        date_to=date_to,
        provenance=_prov(),
    )


# A clean 5-bar dataset covering one trading week: Jan 9-13, 2023 (Mon-Fri)
_WEEK_DATES = [
    date(2023, 1, 9),
    date(2023, 1, 10),
    date(2023, 1, 11),
    date(2023, 1, 12),
    date(2023, 1, 13),
]


def _clean_week() -> OHLCVDataset:
    """One complete trading week with 5 clean bars."""
    bars = [_bar(d) for d in _WEEK_DATES]
    return _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 14))


# ---------------------------------------------------------------------------
# P1-E4-T9: Clean data passes
# ---------------------------------------------------------------------------


class TestCleanDataPasses:
    """T9: Quality checker reports no defects for clean synthetic data."""

    def test_clean_week_passes_threshold(self) -> None:
        """T9: Five bars covering a full trading week reach >=98% completeness."""
        checker = DataQualityChecker()
        report = checker.check(_clean_week())
        assert report.passes_threshold

    def test_clean_week_completeness_is_one(self) -> None:
        """T9: All 5 weekday slots filled means completeness = 1.0."""
        checker = DataQualityChecker()
        report = checker.check(_clean_week())
        assert report.completeness == 1.0

    def test_clean_week_no_anomalies(self) -> None:
        """T9: No price anomalies in clean data."""
        checker = DataQualityChecker()
        report = checker.check(_clean_week())
        assert report.anomaly_count == 0

    def test_clean_week_no_gaps(self) -> None:
        """T9: No gaps in a fully populated week."""
        checker = DataQualityChecker()
        report = checker.check(_clean_week())
        assert report.gap_count == 0

    def test_clean_week_no_ohlcv_violations(self) -> None:
        """T9: All bars satisfy H >= max(O,C) and L <= min(O,C)."""
        checker = DataQualityChecker()
        report = checker.check(_clean_week())
        assert report.ohlcv_violations == 0

    def test_returns_correct_ticker(self) -> None:
        """T9: Report records the correct ticker symbol."""
        checker = DataQualityChecker()
        report = checker.check(_clean_week())
        assert report.ticker == "AAPL"

    def test_returns_correct_bar_count(self) -> None:
        """T9: total_bars in report equals the input bar count."""
        checker = DataQualityChecker()
        report = checker.check(_clean_week())
        assert report.total_bars == 5


# ---------------------------------------------------------------------------
# P1-E4-T8: Defect detection
# ---------------------------------------------------------------------------


class TestDefectDetection:
    """T8: Quality checker detects every injected defect."""

    # --- Completeness ---

    def test_missing_bars_reduces_completeness(self) -> None:
        """T8: Only 3 of 5 bars present → completeness = 0.6 < threshold."""
        bars = [_bar(d) for d in _WEEK_DATES[:3]]  # 3 of 5 days
        ds = _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 14))
        checker = DataQualityChecker()
        report = checker.check(ds)
        assert abs(report.completeness - 0.6) < 1e-9
        assert not report.passes_threshold

    def test_empty_dataset_completeness_is_zero(self) -> None:
        """T8: Zero bars over a period with expected bars gives completeness=0."""
        ds = _dataset([], date_from=date(2023, 1, 9), date_to=date(2023, 1, 14))
        checker = DataQualityChecker()
        report = checker.check(ds)
        assert report.completeness == 0.0
        assert not report.passes_threshold

    # --- Gap detection ---

    def test_large_gap_detected(self) -> None:
        """T8: A 2-week gap (10 weekdays) between two clusters is detected."""
        # Jan 9 bar, then nothing for 2 weeks, then Jan 30 bar
        bars = [_bar(date(2023, 1, 9)), _bar(date(2023, 1, 30))]
        ds = _dataset(
            bars,
            date_from=date(2023, 1, 9),
            date_to=date(2023, 1, 31),
        )
        checker = DataQualityChecker(min_gap_days=5)
        report = checker.check(ds)
        assert report.gap_count >= 1
        assert report.gaps[0].duration_days >= 10

    def test_short_gap_not_flagged(self) -> None:
        """T8: A 1-day gap (e.g., holiday) does not create a Gap record."""
        # Mon, Tue missing, Wed, Thu, Fri
        bars = [
            _bar(date(2023, 1, 9)),   # Mon
            # Jan 10 (Tue) missing -- 1 day gap
            _bar(date(2023, 1, 11)),  # Wed
            _bar(date(2023, 1, 12)),
            _bar(date(2023, 1, 13)),
        ]
        ds = _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 14))
        checker = DataQualityChecker(min_gap_days=5)
        report = checker.check(ds)
        assert report.gap_count == 0

    # --- Large single-day move ---

    def test_large_move_detected(self) -> None:
        """T8: A 60% single-day close jump is flagged as large_move."""
        bars = [
            _bar(date(2023, 1, 9), close=100.0),
            _bar(date(2023, 1, 10), open_=155.0, high=165.0, low=150.0, close=160.0),  # 60% jump
        ]
        ds = _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 11))
        checker = DataQualityChecker()
        report = checker.check(ds)
        large_moves = [a for a in report.price_anomalies if a.anomaly_type == "large_move"]
        assert len(large_moves) == 1
        assert large_moves[0].date == date(2023, 1, 10)

    def test_small_move_not_flagged(self) -> None:
        """T8: A 5% daily move is normal and not flagged."""
        bars = [
            _bar(date(2023, 1, 9), close=100.0),
            _bar(date(2023, 1, 10), close=105.0),
        ]
        ds = _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 11))
        checker = DataQualityChecker()
        report = checker.check(ds)
        large_moves = [a for a in report.price_anomalies if a.anomaly_type == "large_move"]
        assert len(large_moves) == 0

    def test_large_move_downward_detected(self) -> None:
        """T8: A 55% downward move is also flagged."""
        bars = [
            _bar(date(2023, 1, 9), close=100.0),
            _bar(date(2023, 1, 10), open_=48.0, high=50.0, low=40.0, close=45.0),  # 55% drop
        ]
        ds = _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 11))
        checker = DataQualityChecker()
        report = checker.check(ds)
        large_moves = [a for a in report.price_anomalies if a.anomaly_type == "large_move"]
        assert len(large_moves) == 1

    # --- Zero volume ---

    def test_zero_volume_detected(self) -> None:
        """T8: A bar with volume=0 is flagged as zero_volume anomaly."""
        bars = [_bar(date(2023, 1, 9), volume=0.0)]
        ds = _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 10))
        checker = DataQualityChecker()
        report = checker.check(ds)
        zero_vol = [a for a in report.price_anomalies if a.anomaly_type == "zero_volume"]
        assert len(zero_vol) == 1

    def test_nonzero_volume_not_flagged(self) -> None:
        """T8: Normal volume does not trigger any anomaly."""
        bars = [_bar(date(2023, 1, 9), volume=1_000_000.0)]
        ds = _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 10))
        checker = DataQualityChecker()
        report = checker.check(ds)
        zero_vol = [a for a in report.price_anomalies if a.anomaly_type == "zero_volume"]
        assert len(zero_vol) == 0

    # --- Corporate action consistency ---

    def test_corporate_action_ratio_jump_detected(self) -> None:
        """T8: A 10x jump in close/adj_close ratio is flagged as corp_action."""
        bars = [
            _bar(date(2023, 1, 9), close=100.0, adj_close=100.0),  # ratio = 1.0
            _bar(date(2023, 1, 10), close=100.0, adj_close=10.0),   # ratio = 10.0 (10x jump)
        ]
        ds = _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 11))
        checker = DataQualityChecker()
        report = checker.check(ds)
        corp = [a for a in report.price_anomalies if a.anomaly_type == "corp_action"]
        assert len(corp) == 1

    def test_stable_ratio_not_flagged(self) -> None:
        """T8: A close/adj_close ratio that changes by less than 5x is not flagged."""
        bars = [
            _bar(date(2023, 1, 9), close=100.0, adj_close=95.0),
            _bar(date(2023, 1, 10), close=103.0, adj_close=97.0),
        ]
        ds = _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 11))
        checker = DataQualityChecker()
        report = checker.check(ds)
        corp = [a for a in report.price_anomalies if a.anomaly_type == "corp_action"]
        assert len(corp) == 0

    # --- Multiple defects simultaneously ---

    def test_multiple_defects_all_detected(self) -> None:
        """T8: Multiple injected defects are all detected and reported."""
        bars = [
            _bar(date(2023, 1, 9), close=100.0, volume=0.0),  # zero volume
            _bar(date(2023, 1, 10), open_=170.0, high=185.0,
                 low=165.0, close=180.0, volume=500_000.0),  # large move
        ]
        ds = _dataset(bars, date_from=date(2023, 1, 9), date_to=date(2023, 1, 11))
        checker = DataQualityChecker()
        report = checker.check(ds)
        assert any(a.anomaly_type == "zero_volume" for a in report.price_anomalies)
        assert any(a.anomaly_type == "large_move" for a in report.price_anomalies)
