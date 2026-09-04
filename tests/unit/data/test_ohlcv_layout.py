"""What the OHLCV store looks like on disk, and that every reader agrees (DJ-141).

``data/market/<TICKER>/ohlcv.parquet`` is read from seven places, each of which
re-implemented the normalisation, and they did not agree about where the date
lives. Three contracts:

  * date in the **index**, reset to a column   — live/cycle, riskbudget_strategy,
                                                 execution/market_data
  * date in the **index**, left there          — live/walkforward,
                                                 live/market._last_completed_session
  * neither; ``.iloc[-1]`` with no sort        — live/market._latest_prices
  * both shapes handled explicitly             — mcp/financial_server

All seven work against today's file, each for a different reason, which means
six of them work by accident. The accidents are not equal in cost:

  * ``_last_completed_session`` takes ``read_parquet(path).index.max()`` and
    stamps every arm's decision with it. A date moved into a column makes that a
    RangeIndex, ``idx.max()`` an integer, and ``pd.Timestamp(5712)`` the first
    of January 1970 — for the whole cycle, silently.
  * ``_latest_prices`` never sorts, and its number sizes orders.
  * ``walkforward``'s ``df.index <= as_of_date`` on a RangeIndex returns
    everything or nothing with no error.

This module pins the layout so that a change to the writer fails here rather
than in the record. It is deliberately written against the real store: a
fixture-only test would pin the fixture.

DJ-120 is the precedent. Five call sites independently globbed the flat pattern,
saw 83 of 98 tickers as missing, and the agents rendered the gap as conviction.
The fix centralised *path* resolution in ``hifi.data.market_store``; frame
normalisation stayed scattered. Same defect, one layer up.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hifi.data.market_store import load_ohlcv_frame

_REPO = Path(__file__).resolve().parents[3]
_TICKER = "AAPL"
_CANONICAL = _REPO / "data" / "market" / _TICKER / "ohlcv.parquet"

pytestmark = pytest.mark.skipif(
    not _CANONICAL.exists(),
    reason="no canonical OHLCV store in this checkout",
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_parquet(_CANONICAL)


class TestTheShapeOnDisk:
    """The contract the seven readers are, between them, assuming."""

    def test_the_date_is_a_datetime_index_named_date(self, raw):
        assert isinstance(raw.index, pd.DatetimeIndex), (
            "the date left the index; _last_completed_session would read a "
            "RangeIndex and date the whole cycle 1970-01-01"
        )
        assert raw.index.name == "Date"

    def test_the_ohlcv_columns_are_capitalised(self, raw):
        assert {"Open", "High", "Low", "Close", "Volume"} <= set(raw.columns)

    def test_the_date_is_not_also_a_column(self, raw):
        # If it were both, reset_index() would collide.
        assert "Date" not in raw.columns and "date" not in raw.columns

    def test_the_bars_are_sorted_ascending(self, raw):
        assert raw.index.is_monotonic_increasing, (
            "_latest_prices takes .iloc[-1] without sorting; an unsorted store "
            "hands the allocator a stale close and nothing reports it"
        )

    def test_there_is_no_hifi_metadata_block(self):
        """Why mcp/financial_server needs a fallback reader at all.

        ``hifi.data.storage.read_ohlcv`` requires the HiFi metadata block and
        raises ValueError without it. These files carry only pandas metadata, so
        that reader cannot read the canonical store.
        """
        import pyarrow.parquet as pq

        from hifi.data.storage import read_ohlcv

        keys = {k.decode() for k in (pq.read_table(_CANONICAL).schema.metadata or {})}
        assert "hifi_metadata" not in keys

        with pytest.raises(ValueError, match="metadata"):
            read_ohlcv(_CANONICAL)


class TestEveryReaderAgrees:
    """Same file, same ticker — the readers must return the same last bar."""

    @pytest.fixture(scope="class")
    def expected(self, raw) -> tuple[str, float]:
        last = raw.index.max()
        return str(pd.Timestamp(last).date()), float(raw.loc[last, "Close"])

    def test_financial_server(self, expected):
        from hifi.mcp.financial_server import _load_ohlcv

        dataset = _load_ohlcv(_TICKER)
        bar = max(dataset.bars, key=lambda b: b.date)
        assert str(bar.date) == expected[0]
        assert bar.close == pytest.approx(expected[1])

    def test_walkforward(self, expected):
        from hifi.live.walkforward import _load_ohlcv

        rows = _load_ohlcv(str(_REPO / "data"), [_TICKER], expected[0])[_TICKER]
        assert rows[-1]["date"] == expected[0]
        assert rows[-1]["close"] == pytest.approx(expected[1])

    def test_latest_prices(self, expected):
        from hifi.live.market import _latest_prices

        assert _latest_prices([_TICKER])[_TICKER] == pytest.approx(expected[1]), (
            "the price that sizes orders is not the close of the newest bar"
        )

    def test_riskbudget_point_in_time_closes(self, expected):
        from hifi.execution.riskbudget_strategy import point_in_time_closes

        closes = point_in_time_closes(_TICKER, expected[0], str(_REPO / "data"))
        assert closes, "arm D sees no price history for a ticker the others do"
        assert closes[-1] == pytest.approx(expected[1])

    def test_last_completed_session(self, expected):
        from hifi.live.market import _last_completed_session

        assert _last_completed_session([_TICKER]) == expected[0], (
            "the session that dates every arm's decision disagrees with the "
            "newest bar in the store"
        )


class TestTheDateAsAColumnIsReadIdentically:
    """The regression test. Fails before the DJ-141 consolidation.

    Nothing forces the writer to keep the date in the index — ``refresh.py`` and
    ``market_data.py`` both rewrite these files. A store written with the date as
    a column is a perfectly reasonable parquet, and today four of the seven
    readers would misread it in four different ways, none of them raising.
    """

    @pytest.fixture
    def store(self, tmp_path, raw) -> Path:
        """A store shaped exactly like the canonical one, date in a column."""
        d = tmp_path / "market" / _TICKER
        d.mkdir(parents=True)
        flat = raw.tail(400).reset_index()          # 'Date' becomes a column
        flat.to_parquet(d / "ohlcv.parquet", index=False)
        return tmp_path

    @pytest.fixture
    def expected(self, raw) -> tuple[str, float]:
        last = raw.index.max()
        return str(pd.Timestamp(last).date()), float(raw.loc[last, "Close"])

    def test_walkforward_reads_it(self, store, expected):
        from hifi.live.walkforward import _load_ohlcv

        rows = _load_ohlcv(str(store), [_TICKER], expected[0]).get(_TICKER)
        assert rows, "walkforward silently returned nothing for a valid store"
        assert rows[-1]["date"] == expected[0]
        assert rows[-1]["close"] == pytest.approx(expected[1])

    def test_riskbudget_reads_it(self, store, expected):
        from hifi.execution.riskbudget_strategy import point_in_time_closes

        closes = point_in_time_closes(_TICKER, expected[0], str(store))
        assert closes, "arm D silently returned no closes"
        assert closes[-1] == pytest.approx(expected[1])

    def test_financial_server_reads_it(self, store, expected, monkeypatch):
        monkeypatch.setenv("HIFI_DATA_DIR", str(store))
        from hifi.mcp.financial_server import _load_ohlcv

        bar = max(_load_ohlcv(_TICKER).bars, key=lambda b: b.date)
        assert str(bar.date) == expected[0]
        assert bar.close == pytest.approx(expected[1])

    def test_latest_prices_reads_it(self, store, expected, monkeypatch):
        from hifi.live import market, paths

        monkeypatch.setattr(paths, "_DATA_DIR", str(store))
        assert market._latest_prices([_TICKER])[_TICKER] == pytest.approx(expected[1])

    def test_last_completed_session_reads_it(self, store, expected, monkeypatch):
        """The one that would date the whole cycle 1970-01-01."""
        from hifi.live import market, paths

        monkeypatch.setattr(paths, "_DATA_DIR", str(store))
        assert market._last_completed_session([_TICKER]) == expected[0]


class TestLoadOhlcvFrameContract:
    """The normalised frame every reader now gets."""

    def test_columns_are_lowercase_with_a_date_column(self):
        df = load_ohlcv_frame(_TICKER, str(_REPO / "data"))
        assert list(df.columns) == [c.lower() for c in df.columns]
        assert "date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_it_is_sorted(self):
        df = load_ohlcv_frame(_TICKER, str(_REPO / "data"))
        assert df["date"].is_monotonic_increasing

    def test_an_unsorted_store_comes_back_sorted(self, tmp_path, raw):
        d = tmp_path / "market" / _TICKER
        d.mkdir(parents=True)
        raw.tail(50).sample(frac=1.0, random_state=0).to_parquet(d / "ohlcv.parquet")
        assert load_ohlcv_frame(_TICKER, tmp_path)["date"].is_monotonic_increasing

    def test_a_missing_ticker_raises_rather_than_returning_empty(self, tmp_path):
        # DJ-120: an absence that looks like data is how blindness becomes a
        # bearish signal. The caller must be forced to decide.
        with pytest.raises(FileNotFoundError):
            load_ohlcv_frame("NOSUCH", tmp_path)

    def test_a_positional_index_is_refused_not_read_as_1970(self, tmp_path):
        """The failure this function exists to make impossible."""
        d = tmp_path / "market" / "X"
        d.mkdir(parents=True)
        pd.DataFrame({"Close": [1.0, 2.0]}).to_parquet(d / "ohlcv.parquet")
        with pytest.raises(ValueError, match="1970"):
            load_ohlcv_frame("X", tmp_path)

    def test_it_honours_an_explicit_data_dir_over_the_environment(
            self, tmp_path, raw, monkeypatch):
        """hifi.live.paths deliberately ignores HIFI_DATA_DIR, so the live
        callers pass the directory explicitly and must not be overridden by a
        stray export."""
        d = tmp_path / "market" / _TICKER
        d.mkdir(parents=True)
        raw.tail(5).to_parquet(d / "ohlcv.parquet")
        monkeypatch.setenv("HIFI_DATA_DIR", "/nonexistent")
        assert len(load_ohlcv_frame(_TICKER, tmp_path)) == 5


class TestThereIsOneNormaliser:
    """No module may go back to normalising this file itself (DJ-141)."""

    _FORMER = [
        "src/hifi/live/cycle.py",
        "src/hifi/live/market.py",
        "src/hifi/live/walkforward.py",
        "src/hifi/execution/market_data.py",
        "src/hifi/execution/riskbudget_strategy.py",
        "src/hifi/mcp/financial_server.py",
    ]

    @pytest.mark.parametrize("path", _FORMER)
    def test_it_does_not_renormalise(self, path):
        code = [ln for ln in (_REPO / path).read_text().splitlines()
                if not ln.lstrip().startswith("#")]
        for marker in ("index.name.lower()", "columns.str.lower()",
                       "c.lower() for c in df.columns"):
            assert not any(marker in ln for ln in code), (
                f"{path} normalises the OHLCV frame itself again ({marker}); "
                "use hifi.data.market_store.load_ohlcv_frame"
            )

    @pytest.mark.parametrize("path", _FORMER)
    def test_it_uses_the_shared_loader(self, path):
        assert "load_ohlcv_frame" in (_REPO / path).read_text()

    def test_no_module_hard_codes_the_nested_parquet_path(self):
        """Four readers bypassed resolve_ohlcv_path and hard-coded the nested
        layout, so they saw 'no data' exactly where the others saw a legacy
        fixture — arms disagreeing about whether a ticker exists."""
        offenders = []
        for path in self._FORMER:
            for line in (_REPO / path).read_text().splitlines():
                if line.lstrip().startswith("#"):
                    continue
                if '"market"' in line and "ohlcv.parquet" in line:
                    offenders.append(f"{path}: {line.strip()[:70]}")
        assert not offenders, offenders
