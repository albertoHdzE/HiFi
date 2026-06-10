"""
Unit tests for MarketDataFetcher and FundamentalsFetcher (P1-E2).

Tests use synthetic DataFrames that match the yfinance output format.
No live API calls are made: _download and _get_info are patched to return
pre-constructed pandas DataFrames.

Tickets covered:
- P1-E2-T5: Fetcher normalises yfinance output to OHLCVDataset schema
- P1-E2-T6: Fetcher handles missing data gracefully (NaN bars dropped, logged)
- P1-E2-T7: Fetcher attaches correct provenance metadata
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

from hifi.data.market import MarketDataFetcher
from hifi.data.schemas import OHLCVDataset

# ---------------------------------------------------------------------------
# Helpers: synthetic DataFrames matching yfinance output format
# ---------------------------------------------------------------------------


def _make_yfinance_df(
    tickers_dates: list[tuple[str, date, float, float, float, float, float, float | None]],
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Build a synthetic DataFrame matching yfinance Ticker.history() output.

    Columns: Open, High, Low, Close, Adj Close, Volume
    Index: DatetimeIndex with the given timezone.

    Parameters match (date, open, high, low, close, volume, adj_close).
    adj_close=None produces NaN in the Adj Close column.
    """
    rows = []
    for _, _d, o, h, lv, c, vol, adj in tickers_dates:
        rows.append(
            {
                "Open": o,
                "High": h,
                "Low": lv,
                "Close": c,
                "Adj Close": adj if adj is not None else float("nan"),
                "Volume": vol,
            }
        )
    index = pd.DatetimeIndex(
        [pd.Timestamp(d).tz_localize(tz) for _, d, *_ in tickers_dates],
        name="Date",
    )
    return pd.DataFrame(rows, index=index)


def _clean_rows() -> list[tuple]:
    """Three valid bars for AAPL: Jan 3-5, 2023."""
    return [
        ("AAPL", date(2023, 1, 3), 130.28, 130.90, 124.17, 125.07, 112_117_500.0, 122.98),
        ("AAPL", date(2023, 1, 4), 126.89, 128.66, 125.08, 126.36, 89_113_600.0, 124.25),
        ("AAPL", date(2023, 1, 5), 127.13, 127.77, 124.76, 125.02, 80_962_700.0, 122.93),
    ]


# ---------------------------------------------------------------------------
# P1-E2-T5: normalisation to OHLCVDataset
# ---------------------------------------------------------------------------


class TestMarketDataFetcherNormalisation:
    """T5: yfinance DataFrame is normalised to a valid OHLCVDataset."""

    def test_returns_ohlcv_dataset(self) -> None:
        """T5: fetch_ohlcv returns an OHLCVDataset instance."""
        df = _make_yfinance_df(_clean_rows())
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        assert isinstance(result, OHLCVDataset)

    def test_correct_ticker(self) -> None:
        """T5: ticker in the returned dataset matches the requested ticker."""
        df = _make_yfinance_df(_clean_rows())
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        assert result.ticker == "AAPL"

    def test_correct_bar_count(self) -> None:
        """T5: number of bars equals the number of rows in the yfinance DataFrame."""
        df = _make_yfinance_df(_clean_rows())
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        assert len(result.bars) == 3

    def test_prices_normalised_correctly(self) -> None:
        """T5: Open/High/Low/Close/AdjClose values match the synthetic DataFrame."""
        rows = _clean_rows()
        df = _make_yfinance_df(rows)
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        bar = result.bars[0]
        assert abs(bar.open - 130.28) < 1e-4
        assert abs(bar.high - 130.90) < 1e-4
        assert abs(bar.low - 124.17) < 1e-4
        assert abs(bar.close - 125.07) < 1e-4
        assert bar.adjusted_close is not None
        assert abs(bar.adjusted_close - 122.98) < 1e-4

    def test_dates_are_date_objects(self) -> None:
        """T5: bar.date is a datetime.date, not a Timestamp or string."""
        df = _make_yfinance_df(_clean_rows())
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        assert all(isinstance(b.date, date) for b in result.bars)

    def test_dates_stripped_of_timezone(self) -> None:
        """T5: bar dates are date-only, not timezone-aware datetimes."""
        df = _make_yfinance_df(_clean_rows())
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        assert result.bars[0].date == date(2023, 1, 3)

    def test_empty_dataframe_produces_empty_dataset(self) -> None:
        """T5: empty yfinance response produces OHLCVDataset with no bars."""
        df = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            index=pd.DatetimeIndex([], name="Date"),
        )
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        assert result.bars == []

    def test_ohlcv_invariants_hold_for_all_bars(self) -> None:
        """T5: all returned bars satisfy OHLCV schema constraints (high >= low, etc.)."""
        df = _make_yfinance_df(_clean_rows())
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        for bar in result.bars:
            assert bar.high >= bar.open
            assert bar.high >= bar.close
            assert bar.low <= bar.open
            assert bar.low <= bar.close


# ---------------------------------------------------------------------------
# P1-E2-T6: missing data handling
# ---------------------------------------------------------------------------


class TestMarketDataFetcherMissingData:
    """T6: bars with NaN OHLCV are dropped; dataset still valid."""

    def test_nan_open_bar_is_dropped(self) -> None:
        """T6: a bar with NaN Open is excluded from the dataset."""
        rows = [
            ("AAPL", date(2023, 1, 3), float("nan"), 130.90, 124.17, 125.07, 112_117_500.0, 122.98),
            ("AAPL", date(2023, 1, 4), 126.89, 128.66, 125.08, 126.36, 89_113_600.0, 124.25),
        ]
        df = _make_yfinance_df(rows)
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 5))
        assert len(result.bars) == 1
        assert result.bars[0].date == date(2023, 1, 4)

    def test_nan_close_bar_is_dropped(self) -> None:
        """T6: a bar with NaN Close is excluded."""
        rows = [
            ("AAPL", date(2023, 1, 3), 130.28, 130.90, 124.17, float("nan"), 112_117_500.0, None),
            ("AAPL", date(2023, 1, 4), 126.89, 128.66, 125.08, 126.36, 89_113_600.0, 124.25),
        ]
        df = _make_yfinance_df(rows)
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 5))
        assert len(result.bars) == 1

    def test_nan_adjusted_close_is_kept_as_none(self) -> None:
        """T6: NaN Adj Close is converted to None; the bar is not dropped."""
        rows = [
            ("AAPL", date(2023, 1, 3), 130.28, 130.90, 124.17, 125.07, 112_117_500.0, None),
        ]
        df = _make_yfinance_df(rows)
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 4))
        assert len(result.bars) == 1
        assert result.bars[0].adjusted_close is None

    def test_all_nan_rows_produce_empty_dataset(self) -> None:
        """T6: if every row has NaN prices, the result has zero bars."""
        rows = [
            ("AAPL", date(2023, 1, 3),
             float("nan"), float("nan"), float("nan"), float("nan"), 0.0, None),
        ]
        df = _make_yfinance_df(rows)
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 4))
        assert result.bars == []


# ---------------------------------------------------------------------------
# P1-E2-T7: provenance metadata
# ---------------------------------------------------------------------------


class TestMarketDataFetcherProvenance:
    """T7: provenance record is attached and contains correct request parameters."""

    def test_provenance_source_is_yfinance(self) -> None:
        """T7: provenance.source matches the fetcher's source label."""
        df = _make_yfinance_df(_clean_rows())
        fetcher = MarketDataFetcher(source="yfinance")
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        assert result.provenance.source == "yfinance"

    def test_provenance_parameters_contain_ticker(self) -> None:
        """T7: provenance.parameters records the ticker that was requested."""
        df = _make_yfinance_df(_clean_rows())
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        assert result.provenance.parameters["ticker"] == "AAPL"

    def test_provenance_parameters_contain_date_range(self) -> None:
        """T7: provenance.parameters records start and end dates."""
        df = _make_yfinance_df(_clean_rows())
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        assert result.provenance.parameters["start"] == "2023-01-03"
        assert result.provenance.parameters["end"] == "2023-01-06"

    def test_provenance_fetched_at_is_utc(self) -> None:
        """T7: provenance.fetched_at is timezone-aware (UTC)."""
        df = _make_yfinance_df(_clean_rows())
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            result = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 6))
        assert result.provenance.fetched_at.tzinfo is not None
