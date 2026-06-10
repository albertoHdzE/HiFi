"""
Unit tests for HiFi data schemas (P1-E1).

Tickets covered:
- P1-E1-T6: Schema validation accepts valid data for every model
- P1-E1-T7: Schema validation rejects invalid data (negative price, missing
             required field, OHLCV relationship violations, ticker mismatch)
- P1-E1-T8: ProvenanceRecord.compute_signature() is deterministic and
             distinguishes different parameter sets
"""

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from hifi.data.schemas import (
    FundamentalsSnapshot,
    MacroDataset,
    MacroIndicator,
    OHLCVBar,
    OHLCVDataset,
    ProvenanceRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FETCH_TIME = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
_DATE = date(2023, 6, 1)


def _provenance(**overrides: object) -> ProvenanceRecord:
    kwargs: dict = {
        "source": "yfinance",
        "fetched_at": _FETCH_TIME,
        "parameters": {"ticker": "AAPL", "start": "2023-01-01", "end": "2023-12-31"},
    }
    kwargs.update(overrides)
    return ProvenanceRecord(**kwargs)


def _bar(**overrides: object) -> OHLCVBar:
    kwargs: dict = {
        "ticker": "AAPL",
        "date": _DATE,
        "open": 100.0,
        "high": 105.0,
        "low": 98.0,
        "close": 103.0,
        "volume": 1_000_000.0,
    }
    kwargs.update(overrides)
    return OHLCVBar(**kwargs)


def _dataset(bars: list[OHLCVBar] | None = None, **overrides: object) -> OHLCVDataset:
    if bars is None:
        bars = [_bar()]
    kwargs: dict = {
        "ticker": "AAPL",
        "bars": bars,
        "source": "yfinance",
        "fetched_at": _FETCH_TIME,
        "date_from": date(2023, 1, 1),
        "date_to": date(2023, 12, 31),
        "provenance": _provenance(),
    }
    kwargs.update(overrides)
    return OHLCVDataset(**kwargs)


# ---------------------------------------------------------------------------
# P1-E1-T8: ProvenanceRecord signature determinism
# ---------------------------------------------------------------------------


class TestProvenanceRecord:
    """T6, T8: valid construction and signature stability."""

    def test_valid_construction(self) -> None:
        """T6: ProvenanceRecord with all required fields is accepted."""
        prov = _provenance()
        assert prov.source == "yfinance"
        assert prov.content_hash is None

    def test_content_hash_defaults_to_none(self) -> None:
        """T6: content_hash is absent at construction time."""
        prov = _provenance()
        assert prov.content_hash is None

    def test_content_hash_can_be_set(self) -> None:
        """T6: content_hash can be populated after construction."""
        prov = ProvenanceRecord(
            source="yfinance",
            fetched_at=_FETCH_TIME,
            content_hash="abc123",
        )
        assert prov.content_hash == "abc123"

    def test_signature_is_deterministic(self) -> None:
        """T8: calling compute_signature() twice on the same record returns the same digest."""
        prov = _provenance()
        assert prov.compute_signature() == prov.compute_signature()

    def test_signature_is_stable_across_instances(self) -> None:
        """T8: two records with identical fields produce identical signatures."""
        prov_a = _provenance()
        prov_b = _provenance()
        assert prov_a.compute_signature() == prov_b.compute_signature()

    def test_signature_differs_for_different_parameters(self) -> None:
        """T8: changing parameters produces a different signature."""
        prov_a = _provenance(parameters={"ticker": "AAPL"})
        prov_b = _provenance(parameters={"ticker": "MSFT"})
        assert prov_a.compute_signature() != prov_b.compute_signature()

    def test_signature_differs_for_different_sources(self) -> None:
        """T8: changing source produces a different signature."""
        prov_a = _provenance(source="yfinance")
        prov_b = _provenance(source="alpaca")
        assert prov_a.compute_signature() != prov_b.compute_signature()

    def test_signature_is_hex_string(self) -> None:
        """T8: signature is a 64-character hex string (SHA-256)."""
        sig = _provenance().compute_signature()
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)


# ---------------------------------------------------------------------------
# P1-E1-T6 and T7: OHLCVBar
# ---------------------------------------------------------------------------


class TestOHLCVBar:
    """T6: valid bars accepted; T7: invalid bars rejected."""

    def test_valid_bar_accepted(self) -> None:
        """T6: a well-formed bar with all required fields is accepted."""
        bar = _bar()
        assert bar.ticker == "AAPL"
        assert bar.high == 105.0

    def test_open_equal_to_high_and_low_accepted(self) -> None:
        """T6: flat bar (O=H=L=C) is valid (e.g., halted stock)."""
        bar = _bar(open=100.0, high=100.0, low=100.0, close=100.0)
        assert bar.open == 100.0

    def test_adjusted_close_accepted(self) -> None:
        """T6: optional adjusted_close is accepted when positive."""
        bar = _bar(adjusted_close=102.5)
        assert bar.adjusted_close == 102.5

    def test_adjusted_close_none_accepted(self) -> None:
        """T6: adjusted_close defaults to None when omitted."""
        bar = _bar()
        assert bar.adjusted_close is None

    def test_zero_volume_accepted(self) -> None:
        """T6: zero volume is valid (non-negative constraint)."""
        bar = _bar(volume=0.0)
        assert bar.volume == 0.0

    # --- T7: rejection tests ---

    def test_negative_open_rejected(self) -> None:
        """T7: negative open price raises ValidationError."""
        with pytest.raises(ValidationError):
            _bar(open=-1.0)

    def test_zero_open_rejected(self) -> None:
        """T7: zero open price raises ValidationError."""
        with pytest.raises(ValidationError):
            _bar(open=0.0)

    def test_negative_close_rejected(self) -> None:
        """T7: negative close raises ValidationError."""
        with pytest.raises(ValidationError):
            _bar(close=-1.0)

    def test_negative_high_rejected(self) -> None:
        """T7: negative high raises ValidationError."""
        with pytest.raises(ValidationError):
            _bar(high=-1.0)

    def test_negative_low_rejected(self) -> None:
        """T7: negative low raises ValidationError."""
        with pytest.raises(ValidationError):
            _bar(low=-1.0)

    def test_negative_volume_rejected(self) -> None:
        """T7: negative volume raises ValidationError."""
        with pytest.raises(ValidationError):
            _bar(volume=-100.0)

    def test_high_less_than_low_rejected(self) -> None:
        """T7: high < low violates OHLCV invariant."""
        with pytest.raises(ValidationError):
            _bar(high=95.0, low=100.0, open=97.0, close=97.0)

    def test_high_less_than_open_rejected(self) -> None:
        """T7: high < open violates OHLCV invariant."""
        with pytest.raises(ValidationError):
            _bar(open=110.0, high=105.0, low=98.0, close=103.0)

    def test_high_less_than_close_rejected(self) -> None:
        """T7: high < close violates OHLCV invariant."""
        with pytest.raises(ValidationError):
            _bar(open=100.0, high=105.0, low=98.0, close=108.0)

    def test_low_greater_than_open_rejected(self) -> None:
        """T7: low > open violates OHLCV invariant."""
        with pytest.raises(ValidationError):
            _bar(open=95.0, high=105.0, low=100.0, close=103.0)

    def test_low_greater_than_close_rejected(self) -> None:
        """T7: low > close violates OHLCV invariant."""
        with pytest.raises(ValidationError):
            _bar(open=100.0, high=105.0, low=100.0, close=98.0)

    def test_adjusted_close_zero_rejected(self) -> None:
        """T7: adjusted_close of zero raises ValidationError."""
        with pytest.raises(ValidationError):
            _bar(adjusted_close=0.0)

    def test_adjusted_close_negative_rejected(self) -> None:
        """T7: negative adjusted_close raises ValidationError."""
        with pytest.raises(ValidationError):
            _bar(adjusted_close=-5.0)

    def test_missing_ticker_rejected(self) -> None:
        """T7: missing required field ticker raises ValidationError."""
        with pytest.raises(ValidationError):
            OHLCVBar(
                date=_DATE,
                open=100.0,
                high=105.0,
                low=98.0,
                close=103.0,
                volume=1_000_000.0,
            )


# ---------------------------------------------------------------------------
# P1-E1-T6 and T7: OHLCVDataset
# ---------------------------------------------------------------------------


class TestOHLCVDataset:
    """T6: valid datasets accepted; T7: ticker mismatch rejected."""

    def test_valid_dataset_accepted(self) -> None:
        """T6: dataset with matching bars and provenance is accepted."""
        ds = _dataset()
        assert ds.ticker == "AAPL"
        assert len(ds.bars) == 1

    def test_empty_bars_accepted(self) -> None:
        """T6: empty bars list is valid (no data in requested window)."""
        ds = _dataset(bars=[])
        assert ds.bars == []

    def test_multiple_bars_accepted(self) -> None:
        """T6: multiple bars for the same ticker are accepted."""
        bars = [
            _bar(date=date(2023, 6, 1)),
            _bar(date=date(2023, 6, 2)),
            _bar(date=date(2023, 6, 5)),
        ]
        ds = _dataset(bars=bars)
        assert len(ds.bars) == 3

    def test_mismatched_bar_ticker_rejected(self) -> None:
        """T7: a bar with a different ticker raises ValidationError."""
        wrong_bar = _bar(ticker="MSFT")
        with pytest.raises(ValidationError):
            _dataset(bars=[wrong_bar])

    def test_missing_provenance_rejected(self) -> None:
        """T7: provenance is a required field."""
        with pytest.raises(ValidationError):
            OHLCVDataset(
                ticker="AAPL",
                bars=[],
                source="yfinance",
                fetched_at=_FETCH_TIME,
                date_from=date(2023, 1, 1),
                date_to=date(2023, 12, 31),
            )


# ---------------------------------------------------------------------------
# P1-E1-T6: FundamentalsSnapshot
# ---------------------------------------------------------------------------


class TestFundamentalsSnapshot:
    """T6: valid snapshots accepted including edge cases."""

    def test_valid_snapshot_accepted(self) -> None:
        """T6: snapshot with all fields populated is accepted."""
        snap = FundamentalsSnapshot(
            ticker="AAPL",
            period_end=date(2023, 9, 30),
            revenue=89_498_000_000.0,
            net_income=22_956_000_000.0,
            total_assets=352_583_000_000.0,
            total_liabilities=290_437_000_000.0,
            total_equity=62_146_000_000.0,
            eps=1.46,
            pe_ratio=28.5,
            market_cap=2_750_000_000_000.0,
            source="yfinance",
            fetched_at=_FETCH_TIME,
            provenance=_provenance(),
        )
        assert snap.ticker == "AAPL"
        assert snap.revenue == 89_498_000_000.0

    def test_all_financial_fields_none_accepted(self) -> None:
        """T6: all optional financial fields can be None (sparse data is valid)."""
        snap = FundamentalsSnapshot(
            ticker="AAPL",
            period_end=date(2023, 9, 30),
            source="yfinance",
            fetched_at=_FETCH_TIME,
            provenance=_provenance(),
        )
        assert snap.revenue is None
        assert snap.net_income is None

    def test_negative_net_income_accepted(self) -> None:
        """T6: negative net_income (a loss) is valid financial data."""
        snap = FundamentalsSnapshot(
            ticker="SNAP",
            period_end=date(2023, 9, 30),
            net_income=-368_000_000.0,
            source="yfinance",
            fetched_at=_FETCH_TIME,
            provenance=_provenance(parameters={"ticker": "SNAP"}),
        )
        assert snap.net_income == -368_000_000.0

    def test_negative_total_equity_accepted(self) -> None:
        """T6: negative equity (liabilities > assets) is valid (e.g., McDonald's)."""
        snap = FundamentalsSnapshot(
            ticker="MCD",
            period_end=date(2023, 9, 30),
            total_equity=-7_000_000_000.0,
            source="yfinance",
            fetched_at=_FETCH_TIME,
            provenance=_provenance(parameters={"ticker": "MCD"}),
        )
        assert snap.total_equity < 0


# ---------------------------------------------------------------------------
# P1-E1-T6 and T7: MacroIndicator and MacroDataset
# ---------------------------------------------------------------------------


class TestMacroIndicator:
    """T6: valid indicators including negative values."""

    def test_positive_value_accepted(self) -> None:
        """T6: positive macro value (e.g., 5.25% fed funds rate) is accepted."""
        obs = MacroIndicator(series_id="FEDFUNDS", date=_DATE, value=5.25)
        assert obs.value == 5.25

    def test_negative_value_accepted(self) -> None:
        """T6: negative macro value (e.g., negative real rates) is valid."""
        obs = MacroIndicator(series_id="FEDFUNDS", date=_DATE, value=-0.5)
        assert obs.value == -0.5

    def test_zero_value_accepted(self) -> None:
        """T6: zero macro value is valid."""
        obs = MacroIndicator(series_id="FEDFUNDS", date=_DATE, value=0.0)
        assert obs.value == 0.0


class TestMacroDataset:
    """T6: valid datasets accepted; T7: series_id mismatch rejected."""

    def _make_dataset(self, **overrides: object) -> MacroDataset:
        obs = [
            MacroIndicator(series_id="FEDFUNDS", date=date(2023, 1, 1), value=4.33),
            MacroIndicator(series_id="FEDFUNDS", date=date(2023, 2, 1), value=4.57),
        ]
        kwargs: dict = {
            "series_id": "FEDFUNDS",
            "name": "Federal Funds Effective Rate",
            "frequency": "monthly",
            "unit": "percent",
            "observations": obs,
            "source": "FRED",
            "fetched_at": _FETCH_TIME,
            "date_from": date(2023, 1, 1),
            "date_to": date(2023, 12, 31),
            "provenance": _provenance(
                source="FRED",
                parameters={"series_id": "FEDFUNDS", "start": "2023-01-01"},
            ),
        }
        kwargs.update(overrides)
        return MacroDataset(**kwargs)

    def test_valid_macro_dataset_accepted(self) -> None:
        """T6: well-formed MacroDataset is accepted."""
        ds = self._make_dataset()
        assert ds.series_id == "FEDFUNDS"
        assert len(ds.observations) == 2

    def test_empty_observations_accepted(self) -> None:
        """T6: empty observations list is valid."""
        ds = self._make_dataset(observations=[])
        assert ds.observations == []

    def test_mismatched_series_id_rejected(self) -> None:
        """T7: an observation with a different series_id raises ValidationError."""
        wrong_obs = MacroIndicator(
            series_id="CPIAUCSL", date=date(2023, 1, 1), value=296.8
        )
        with pytest.raises(ValidationError):
            self._make_dataset(observations=[wrong_obs])

    def test_missing_provenance_rejected(self) -> None:
        """T7: provenance is required."""
        with pytest.raises(ValidationError):
            MacroDataset(
                series_id="FEDFUNDS",
                name="Federal Funds Rate",
                frequency="monthly",
                unit="percent",
                observations=[],
                fetched_at=_FETCH_TIME,
                date_from=date(2023, 1, 1),
                date_to=date(2023, 12, 31),
            )
