"""
Integration tests for market data acquisition pipeline (P1-E2).

These tests use pre-recorded yfinance fixture DataFrames (captured once
from the live API and stored in tests/fixtures/market/).  No live API calls
are made during testing.

Tickets covered:
- P1-E2-T8: Full fetch for one ticker using recorded fixture
- P1-E2-T9: Parquet write/read round-trip preserves all values exactly
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from hifi.data.market import MarketDataFetcher
from hifi.data.storage import read_ohlcv, write_ohlcv

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "market"


def _load_fixture_as_yfinance_df(ticker: str) -> pd.DataFrame:
    """
    Load a pre-recorded fixture parquet and reformat it as a yfinance-style DataFrame.

    The fixture was saved with Date as a column (index reset).  We restore the
    DatetimeIndex with tz-naive timestamps (yfinance tz is stripped in the
    fixture) so the normaliser treats it correctly.
    """
    path = _FIXTURES / f"{ticker}_2023-01-03_2023-04-01.parquet"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")

    df = pd.read_parquet(path)
    df = df.set_index("Date")
    # Restore column names to match yfinance output format
    df = df.rename(
        columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adj Close",
            "Volume": "Volume",
        }
    )
    return df


# ---------------------------------------------------------------------------
# P1-E2-T8: Full fetch using recorded fixture
# ---------------------------------------------------------------------------


class TestMarketFetchWithRecordedFixture:
    """T8: Full fetch pipeline using pre-captured yfinance data."""

    @pytest.mark.integration
    def test_aapl_fixture_produces_valid_dataset(self) -> None:
        """T8: AAPL fixture normalises to a valid OHLCVDataset with expected bar count."""
        df = _load_fixture_as_yfinance_df("AAPL")
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            dataset = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 4, 1))

        assert dataset.ticker == "AAPL"
        # The fixture contains 62 trading days (Jan 3 - Mar 31, 2023)
        assert len(dataset.bars) == 62
        # All bars have positive prices
        for bar in dataset.bars:
            assert bar.open > 0
            assert bar.high >= bar.low
            assert bar.close > 0

    @pytest.mark.integration
    def test_jpm_fixture_produces_valid_dataset(self) -> None:
        """T8: JPM fixture normalises correctly."""
        df = _load_fixture_as_yfinance_df("JPM")
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            dataset = fetcher.fetch_ohlcv("JPM", date(2023, 1, 3), date(2023, 4, 1))
        assert dataset.ticker == "JPM"
        assert len(dataset.bars) == 62

    @pytest.mark.integration
    def test_xom_fixture_produces_valid_dataset(self) -> None:
        """T8: XOM fixture normalises correctly."""
        df = _load_fixture_as_yfinance_df("XOM")
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            dataset = fetcher.fetch_ohlcv("XOM", date(2023, 1, 3), date(2023, 4, 1))
        assert dataset.ticker == "XOM"
        assert len(dataset.bars) == 62


# ---------------------------------------------------------------------------
# P1-E2-T9: Parquet write/read round-trip
# ---------------------------------------------------------------------------


class TestParquetRoundTrip:
    """T9: writing an OHLCVDataset to Parquet and reading it back is lossless."""

    @pytest.mark.integration
    def test_aapl_round_trip_bar_count(self, tmp_path: Path) -> None:
        """T9: number of bars is preserved after Parquet round-trip."""
        df = _load_fixture_as_yfinance_df("AAPL")
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            original = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 4, 1))

        out_path = tmp_path / "AAPL.parquet"
        write_ohlcv(original, out_path)
        loaded = read_ohlcv(out_path)

        assert len(loaded.bars) == len(original.bars)

    @pytest.mark.integration
    def test_aapl_round_trip_price_precision(self, tmp_path: Path) -> None:
        """T9: OHLCV prices are preserved to float64 precision after round-trip."""
        df = _load_fixture_as_yfinance_df("AAPL")
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            original = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 4, 1))

        out_path = tmp_path / "AAPL.parquet"
        write_ohlcv(original, out_path)
        loaded = read_ohlcv(out_path)

        for orig_bar, loaded_bar in zip(original.bars, loaded.bars, strict=True):
            assert orig_bar.date == loaded_bar.date
            assert orig_bar.open == loaded_bar.open
            assert orig_bar.high == loaded_bar.high
            assert orig_bar.low == loaded_bar.low
            assert orig_bar.close == loaded_bar.close
            assert orig_bar.volume == loaded_bar.volume

    @pytest.mark.integration
    def test_aapl_round_trip_adjusted_close(self, tmp_path: Path) -> None:
        """T9: adjusted_close is preserved (including None values) after round-trip."""
        df = _load_fixture_as_yfinance_df("AAPL")
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            original = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 4, 1))

        out_path = tmp_path / "AAPL.parquet"
        write_ohlcv(original, out_path)
        loaded = read_ohlcv(out_path)

        for orig_bar, loaded_bar in zip(original.bars, loaded.bars, strict=True):
            if orig_bar.adjusted_close is None:
                assert loaded_bar.adjusted_close is None
            else:
                assert loaded_bar.adjusted_close is not None
                assert abs(orig_bar.adjusted_close - loaded_bar.adjusted_close) < 1e-9

    @pytest.mark.integration
    def test_round_trip_metadata_preserved(self, tmp_path: Path) -> None:
        """T9: ticker, source, date range, and provenance survive the round-trip."""
        df = _load_fixture_as_yfinance_df("AAPL")
        fetcher = MarketDataFetcher()
        with patch.object(fetcher, "_download", return_value=df):
            original = fetcher.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 4, 1))

        out_path = tmp_path / "AAPL.parquet"
        write_ohlcv(original, out_path)
        loaded = read_ohlcv(out_path)

        assert loaded.ticker == original.ticker
        assert loaded.source == original.source
        assert loaded.date_from == original.date_from
        assert loaded.date_to == original.date_to
        assert loaded.provenance.source == original.provenance.source
        assert loaded.provenance.parameters == original.provenance.parameters

    @pytest.mark.integration
    def test_empty_dataset_round_trip(self, tmp_path: Path) -> None:
        """T9: empty dataset (no bars) round-trips without error."""
        from datetime import UTC, datetime

        from hifi.data.schemas import OHLCVDataset, ProvenanceRecord

        prov = ProvenanceRecord(
            source="yfinance",
            fetched_at=datetime(2023, 1, 1, tzinfo=UTC),
            parameters={"ticker": "AAPL"},
        )
        empty = OHLCVDataset(
            ticker="AAPL",
            bars=[],
            source="yfinance",
            fetched_at=datetime(2023, 1, 1, tzinfo=UTC),
            date_from=date(2023, 1, 3),
            date_to=date(2023, 1, 6),
            provenance=prov,
        )
        out_path = tmp_path / "empty.parquet"
        write_ohlcv(empty, out_path)
        loaded = read_ohlcv(out_path)
        assert loaded.bars == []
        assert loaded.ticker == "AAPL"
