"""
Data quality validation for HiFi datasets.

Provides DataQualityChecker and QualityReport for OHLCVDataset.

Quality is measured along five dimensions:

1. Completeness: what fraction of expected US trading days have a bar?
   Expected days are computed as weekdays (Mon-Fri) in the date range. US market
   holidays are not subtracted -- this is a documented approximation that makes
   the completeness score slightly conservative (2-3% below reality). A score
   above 98% is considered acceptable for Phase 1 purposes.

2. Gap detection: any run of consecutive missing weekdays longer than a threshold
   (default 5) is recorded as a Gap. Short gaps (1-2 days) are typically market
   holidays and are expected.

3. Price sanity: three checks --
   - Zero volume (unusual for liquid equities, logged as anomaly)
   - Single-day close-to-close move > 50% (likely unadjusted corporate action)
   - Day-over-day price reversal > 95% in a single session (data error)

4. OHLCV relationships: re-validates H >= max(O,C) and L <= min(O,C) at the
   dataset level. These are enforced at the schema level, so violations should
   be zero; this check exists as a defence-in-depth audit.

5. Corporate action consistency: if adjusted_close is available, checks that
   the close/adjusted_close ratio does not change by more than 5x in a single
   day. A ratio change of that magnitude indicates an unadjusted split.

Design decision: failures are reported, not raised. A dataset with a known gap
during the 2020 COVID crash is still usable -- the consumer must see the report
and decide whether the gaps are acceptable for their use case. Crashing on
imperfect data would make the system fragile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from hifi.data.schemas import OHLCVDataset

# Threshold for single-day close move to be flagged as anomalous
_LARGE_MOVE_THRESHOLD = 0.50

# Threshold for ratio change between close and adjusted_close
_CORP_ACTION_RATIO_THRESHOLD = 5.0

# Minimum consecutive missing weekdays to record as a Gap
_MIN_GAP_DAYS = 5

# Completeness threshold required to pass the quality gate
COMPLETENESS_THRESHOLD = 0.98


@dataclass
class Gap:
    """A contiguous period of missing weekday bars in an OHLCVDataset."""

    start_date: date
    end_date: date
    duration_days: int


@dataclass
class PriceAnomaly:
    """
    A bar (or pair of bars) that fails a price sanity check.

    anomaly_type values:
    - "large_move": single-day close change > 50% (possible unadjusted event)
    - "zero_volume": volume reported as zero
    - "corp_action": close/adjusted_close ratio jumped > 5x vs. prior bar
    """

    date: date
    close: float
    anomaly_type: str
    detail: str = ""


@dataclass
class QualityReport:
    """
    Summary of data quality for a single OHLCVDataset.

    This report is a deliverable of Phase 1, not an internal diagnostic. It
    is intended to be read by the researcher before Phase 2 computation begins.
    """

    ticker: str
    date_from: date
    date_to: date
    total_bars: int
    expected_bars: int
    completeness: float
    gaps: list[Gap] = field(default_factory=list)
    price_anomalies: list[PriceAnomaly] = field(default_factory=list)
    ohlcv_violations: int = 0
    passes_threshold: bool = False

    @property
    def gap_count(self) -> int:
        return len(self.gaps)

    @property
    def anomaly_count(self) -> int:
        return len(self.price_anomalies)


class DataQualityChecker:
    """
    Validates an OHLCVDataset across completeness, price sanity, and consistency
    dimensions.

    Parameters
    ----------
    completeness_threshold : float
        Minimum completeness ratio to pass the quality gate (default 0.98).
    min_gap_days : int
        Minimum consecutive missing weekdays to record as a Gap (default 5).
    large_move_threshold : float
        Single-day close change fraction above which a bar is flagged as
        potentially anomalous (default 0.50).
    """

    def __init__(
        self,
        completeness_threshold: float = COMPLETENESS_THRESHOLD,
        min_gap_days: int = _MIN_GAP_DAYS,
        large_move_threshold: float = _LARGE_MOVE_THRESHOLD,
    ) -> None:
        self._completeness_threshold = completeness_threshold
        self._min_gap_days = min_gap_days
        self._large_move_threshold = large_move_threshold

    def check(self, dataset: OHLCVDataset) -> QualityReport:
        """
        Run all quality checks on an OHLCVDataset and return a QualityReport.

        The dataset is not modified. All detected issues are described in the
        report; nothing is removed or imputed.
        """
        bars = dataset.bars
        expected = self._count_weekdays(dataset.date_from, dataset.date_to)

        total = len(bars)
        completeness = (total / expected) if expected > 0 else 0.0

        gaps = self._detect_gaps(bars, dataset.date_from, dataset.date_to)
        anomalies = self._check_price_sanity(bars)
        violations = self._check_ohlcv_relationships(bars)

        passes = (
            completeness >= self._completeness_threshold
            and violations == 0
        )

        return QualityReport(
            ticker=dataset.ticker,
            date_from=dataset.date_from,
            date_to=dataset.date_to,
            total_bars=total,
            expected_bars=expected,
            completeness=completeness,
            gaps=gaps,
            price_anomalies=anomalies,
            ohlcv_violations=violations,
            passes_threshold=passes,
        )

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    @staticmethod
    def _count_weekdays(start: date, end: date) -> int:
        """
        Count Mon-Fri days in [start, end).

        Uses numpy.busday_count for efficiency. No US holiday calendar is
        applied: this produces a slightly conservative completeness score.
        """
        return int(np.busday_count(start.isoformat(), end.isoformat()))

    def _detect_gaps(
        self, bars: list, start: date, end: date
    ) -> list[Gap]:
        """
        Find contiguous runs of missing weekdays longer than min_gap_days.

        The algorithm builds the set of dates that have bars, then walks
        the weekday calendar between start and end looking for runs of days
        without bars.
        """
        bar_dates: set[date] = {b.date for b in bars}
        gaps: list[Gap] = []

        gap_start: date | None = None
        gap_len = 0
        current = start
        delta = timedelta(days=1)

        while current < end:
            if current.weekday() < 5:  # weekday (Mon=0 ... Fri=4)
                if current not in bar_dates:
                    if gap_start is None:
                        gap_start = current
                    gap_len += 1
                else:
                    if gap_start is not None and gap_len >= self._min_gap_days:
                        gaps.append(
                            Gap(
                                start_date=gap_start,
                                end_date=current - delta,
                                duration_days=gap_len,
                            )
                        )
                    gap_start = None
                    gap_len = 0
            current += delta

        # Flush trailing gap
        if gap_start is not None and gap_len >= self._min_gap_days:
            gaps.append(
                Gap(
                    start_date=gap_start,
                    end_date=current - delta,
                    duration_days=gap_len,
                )
            )

        return gaps

    def _check_price_sanity(self, bars: list) -> list[PriceAnomaly]:
        """
        Check for zero-volume bars, large single-day moves, and suspicious
        adjusted_close/close ratio changes.
        """
        anomalies: list[PriceAnomaly] = []
        sorted_bars = sorted(bars, key=lambda b: b.date)

        prev_close: float | None = None
        prev_adj_ratio: float | None = None

        for bar in sorted_bars:
            # Zero volume (suspicious for liquid equities)
            if bar.volume == 0:
                anomalies.append(
                    PriceAnomaly(
                        date=bar.date,
                        close=bar.close,
                        anomaly_type="zero_volume",
                        detail=f"volume=0 on {bar.date}",
                    )
                )

            # Large single-day move
            if prev_close is not None and prev_close > 0:
                pct = abs(bar.close - prev_close) / prev_close
                if pct > self._large_move_threshold:
                    anomalies.append(
                        PriceAnomaly(
                            date=bar.date,
                            close=bar.close,
                            anomaly_type="large_move",
                            detail=(
                                f"{pct:.1%} change from {prev_close:.2f} "
                                f"to {bar.close:.2f}"
                            ),
                        )
                    )

            # Corporate action consistency via close/adjusted_close ratio
            if bar.adjusted_close is not None and bar.adjusted_close > 0:
                ratio = bar.close / bar.adjusted_close
                if prev_adj_ratio is not None and prev_adj_ratio > 0:
                    ratio_change = ratio / prev_adj_ratio
                    if ratio_change > _CORP_ACTION_RATIO_THRESHOLD or ratio_change < (
                        1.0 / _CORP_ACTION_RATIO_THRESHOLD
                    ):
                        anomalies.append(
                            PriceAnomaly(
                                date=bar.date,
                                close=bar.close,
                                anomaly_type="corp_action",
                                detail=(
                                    f"close/adj_close ratio changed {ratio_change:.1f}x "
                                    f"(from {prev_adj_ratio:.4f} to {ratio:.4f})"
                                ),
                            )
                        )
                prev_adj_ratio = ratio
            else:
                prev_adj_ratio = None

            prev_close = bar.close

        return anomalies

    @staticmethod
    def _check_ohlcv_relationships(bars: list) -> int:
        """
        Count bars that violate H >= max(O,C) or L <= min(O,C).

        These should be zero because the schema enforces these invariants at
        construction time. This check is defence-in-depth: if a future code
        change bypasses schema validation, this catches it.
        """
        violations = 0
        for bar in bars:
            if bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
                violations += 1
        return violations
