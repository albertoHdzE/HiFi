"""Tests for the Phase 16 live orchestrator (control strategy, account routing)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import run_phase16_live as live  # noqa: E402

from hifi.execution.broker import Position  # noqa: E402


def _mock_executor(equity: float, cash: float, positions: dict[str, Position]):
    ex = MagicMock()
    ex.get_portfolio_value.return_value = equity
    ex.get_account_cash.return_value = cash
    ex.get_positions.return_value = positions
    return ex


class TestControlStrategy:
    @patch.object(live, "_latest_prices")
    def test_buys_missing_tickers_equal_weight(self, mock_prices):
        mock_prices.return_value = {"AAPL": 100.0, "MSFT": 200.0}
        ex = _mock_executor(equity=10_000.0, cash=10_000.0, positions={})

        orders = live.run_control_strategy(["AAPL", "MSFT"], ex, dry_run=True)

        # buffered slice = $10,000 * 0.99 / 2 = $4,950: AAPL x49.5, MSFT x24.75
        assert len(orders) == 2
        by_ticker = {o["ticker"]: o for o in orders}
        assert by_ticker["AAPL"]["qty"] == 49.5
        assert by_ticker["MSFT"]["qty"] == 24.75
        assert all(o["status"] == "dry_run" for o in orders)

    @patch.object(live, "_latest_prices")
    def test_holds_existing_positions(self, mock_prices):
        mock_prices.return_value = {"AAPL": 100.0, "MSFT": 200.0}
        held = {"AAPL": Position("AAPL", 50, 5000.0, 100.0, 0.0, "long")}
        ex = _mock_executor(equity=10_000.0, cash=5_000.0, positions=held)

        orders = live.run_control_strategy(["AAPL", "MSFT"], ex, dry_run=True)

        assert len(orders) == 1
        assert orders[0]["ticker"] == "MSFT"

    @patch.object(live, "_latest_prices")
    def test_fractional_shares_for_expensive_tickers(self, mock_prices):
        mock_prices.return_value = {"AAPL": 100.0, "EXPENSIVE": 10_000.0}
        ex = _mock_executor(equity=10_000.0, cash=10_000.0, positions={})

        orders = live.run_control_strategy(["AAPL", "EXPENSIVE"], ex, dry_run=True)

        # buffered slice = $4,950: AAPL x49.5 whole-ish, EXPENSIVE x0.495 fractional
        by_ticker = {o["ticker"]: o for o in orders}
        assert by_ticker["AAPL"]["qty"] == 49.5
        assert by_ticker["EXPENSIVE"]["qty"] == 0.495

    @patch.object(live, "_latest_prices")
    def test_stops_when_out_of_cash(self, mock_prices):
        mock_prices.return_value = {"AAPL": 100.0, "MSFT": 100.0}
        ex = _mock_executor(equity=10_000.0, cash=4_000.0, positions={})

        orders = live.run_control_strategy(["AAPL", "MSFT"], ex, dry_run=True)

        # slice = $5,000 → AAPL x50 = $5,000 > $4,000 cash → nothing placed
        assert orders == []

    @patch.object(live, "_latest_prices")
    def test_places_real_orders_when_not_dry(self, mock_prices):
        mock_prices.return_value = {"AAPL": 100.0}
        ex = _mock_executor(equity=1_000.0, cash=1_000.0, positions={})
        result = MagicMock()
        result.status = "accepted"
        result.order_id = "oid-1"
        ex.place_market_order.return_value = result

        orders = live.run_control_strategy(["AAPL"], ex, dry_run=False)

        ex.place_market_order.assert_called_once_with("AAPL", 9.9, "buy")
        assert orders[0]["status"] == "accepted"


class TestExecuteOrders:
    """Regression guard for HiFi issue #1 — allocator emits 'quantity'/'BUY'."""

    class _Snap:
        def __init__(self, orders):
            self.orders = orders

    def test_reads_allocator_quantity_field(self):
        # allocate_capital output schema: ticker, side="BUY", quantity
        snap = self._Snap([
            {"ticker": "JPM", "side": "BUY", "quantity": 14, "order_type": "MARKET"},
            {"ticker": "KO", "side": "BUY", "quantity": 61, "order_type": "MARKET"},
        ])
        ex = MagicMock()
        out = live.execute_orders(snap, ex, dry_run=True)
        assert [(o["ticker"], o["side"], o["qty"]) for o in out] == [
            ("JPM", "buy", 14), ("KO", "buy", 61),
        ]

    def test_zero_quantity_skipped(self):
        snap = self._Snap([{"ticker": "X", "side": "BUY", "quantity": 0}])
        assert live.execute_orders(snap, MagicMock(), dry_run=True) == []

    def test_places_real_orders_with_quantity(self):
        snap = self._Snap([{"ticker": "JPM", "side": "BUY", "quantity": 14}])
        ex = MagicMock()
        res = MagicMock()
        res.status = "accepted"
        res.order_id = "o1"
        res.filled_avg_price = None
        ex.place_market_order.return_value = res
        out = live.execute_orders(snap, ex, dry_run=False)
        ex.place_market_order.assert_called_once_with("JPM", 14, "buy")
        assert out[0]["status"] == "accepted"


class TestAccountRouting:
    def test_account_conditions(self):
        assert live._ACCOUNTS["A"]["condition"] == "parallel"
        assert live._ACCOUNTS["B"]["condition"] == "full"
        assert live._ACCOUNTS["C"]["condition"] == "control"
        assert live._ACCOUNTS["D"]["condition"] == "riskbudget"

    def test_missing_credentials_returns_none(self, monkeypatch):
        for suffix in live._ACCOUNTS["C"]["suffixes"]:
            monkeypatch.delenv(f"ALPACA_API_KEY{suffix}", raising=False)
            monkeypatch.delenv(f"ALPACA_SECRET{suffix}", raising=False)
        assert live.get_executor("C") is None

    @patch("hifi.execution.alpaca_executor.TradingClient")
    def test_account_a_falls_back_to_unsuffixed_keys(self, mock_client_cls, monkeypatch):
        mock_acct = MagicMock()
        mock_acct.equity = "1000.00"
        mock_acct.cash = "1000.00"
        mock_client_cls.return_value.get_account.return_value = mock_acct

        for suffix in ("_FIRST", "_A"):
            monkeypatch.delenv(f"ALPACA_API_KEY{suffix}", raising=False)
            monkeypatch.delenv(f"ALPACA_SECRET{suffix}", raising=False)
        monkeypatch.setenv("ALPACA_API_KEY", "fallback-key")
        monkeypatch.setenv("ALPACA_SECRET", "fallback-secret")

        ex = live.get_executor("A")
        assert ex is not None
        mock_client_cls.assert_called_once_with(
            api_key="fallback-key", secret_key="fallback-secret", paper=True
        )

    @patch("hifi.execution.alpaca_executor.TradingClient")
    def test_account_b_uses_second_suffix(self, mock_client_cls, monkeypatch):
        mock_acct = MagicMock()
        mock_acct.equity = "1000.00"
        mock_acct.cash = "1000.00"
        mock_client_cls.return_value.get_account.return_value = mock_acct

        monkeypatch.setenv("ALPACA_API_KEY_SECOND", "key-b")
        monkeypatch.setenv("ALPACA_SECRET_SECOND", "secret-b")

        ex = live.get_executor("B")
        assert ex is not None
        mock_client_cls.assert_called_once_with(
            api_key="key-b", secret_key="secret-b", paper=True
        )


def _breaker_executor(equity: float, positions: dict[str, Position],
                      last_equity: float | None = None):
    """Executor whose Alpaca account reports `equity` / `last_equity`."""
    ex = _mock_executor(equity=equity, cash=0.0, positions=positions)
    acct = MagicMock()
    acct.equity = str(equity)
    acct.last_equity = str(equity if last_equity is None else last_equity)
    ex.client.get_account.return_value = acct
    return ex


class TestCircuitBreakerScaling:
    """DJ-119: position loss halts on portfolio impact, not raw percentage."""

    @pytest.fixture(autouse=True)
    def _isolate_breaker_log(self, tmp_path, monkeypatch):
        # check_circuit_breakers() appends to data/live/<acct>/circuit_breakers.jsonl.
        # Without this, a test run writes synthetic rows into the real live
        # record (it did: three fake "A daily_loss -0.05" rows on 2026-07-28,
        # removed by hand). Redirect every account dir at the filesystem level
        # so no test in this class can reach data/live, whatever it patches.
        monkeypatch.setattr(live, "_account_dir", lambda a: tmp_path / a)
        self.breaker_dir = tmp_path

    def test_wide_book_single_loser_does_not_halt(self, tmp_path):
        # Equal-weight 98-name book: each name ~1% of equity, so a 21% drawdown
        # on one costs ~0.21% of the portfolio. This is the case that froze
        # account C on every run from 2026-07-22 to 2026-07-28.
        positions = {
            f"T{i}": Position(f"T{i}", 10, 1_000.0, 100.0, 0.0, "long")
            for i in range(98)
        }
        positions["T7"] = Position("T7", 10, 790.0, 100.0, -210.0, "long")
        ex = _breaker_executor(equity=100_000.0, positions=positions)

        with patch.object(live, "_log_circuit_breaker") as log:
            assert live.check_circuit_breakers("C", ex) is False

        actions = [c.kwargs["action"] for c in log.call_args_list]
        assert actions == ["flag"], "the 21% loser must still be recorded"

    def test_concentrated_loser_halts(self, tmp_path):
        # One name at 40% of equity down 20% costs 8% of the book -> halt.
        positions = {
            "BIG": Position("BIG", 400, 40_000.0, 125.0, -10_000.0, "long"),
            "OK": Position("OK", 10, 1_000.0, 100.0, 0.0, "long"),
        }
        ex = _breaker_executor(equity=100_000.0, positions=positions)

        with patch.object(live, "_log_circuit_breaker") as log:
            assert live.check_circuit_breakers("A", ex) is True

        assert log.call_args_list[0].kwargs["action"] == "halt"

    def test_portfolio_daily_loss_still_halts(self):
        ex = _breaker_executor(equity=95_000.0, positions={}, last_equity=100_000.0)
        assert live.check_circuit_breakers("A", ex) is True
        # Isolation guard: the row must land in tmp, never in data/live.
        assert (self.breaker_dir / "A" / "circuit_breakers.jsonl").exists()
        assert not (live._ROOT / "data" / "live" / "A"
                    / "circuit_breakers.jsonl").exists()

    def test_small_loss_below_flag_threshold_is_silent(self):
        positions = {"OK": Position("OK", 10, 950.0, 100.0, -50.0, "long")}
        ex = _breaker_executor(equity=100_000.0, positions=positions)

        with patch.object(live, "_log_circuit_breaker") as log:
            assert live.check_circuit_breakers("A", ex) is False
        log.assert_not_called()


class TestAlreadyDecided:
    """DJ-119: one decision cycle per account per day."""

    def _write(self, tmp_path, account, dates):
        d = tmp_path / account
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "decisions.jsonl", "w") as f:
            for dt in dates:
                f.write(f'{{"decision_date": "{dt}", "n_orders": 0}}\n')

    def test_detects_existing_date(self, tmp_path, monkeypatch):
        self._write(tmp_path, "D", ["2026-07-24", "2026-07-28"])
        monkeypatch.setattr(live, "_account_dir", lambda a: tmp_path / a)

        assert live.already_decided("D", "2026-07-28") is True
        assert live.already_decided("D", "2026-07-29") is False

    def test_missing_log_is_not_decided(self, tmp_path, monkeypatch):
        monkeypatch.setattr(live, "_account_dir", lambda a: tmp_path / a)
        assert live.already_decided("A", "2026-07-28") is False

    def test_corrupt_log_fails_open(self, tmp_path, monkeypatch):
        d = tmp_path / "A"
        d.mkdir(parents=True)
        (d / "decisions.jsonl").write_text("{not json\n")
        monkeypatch.setattr(live, "_account_dir", lambda a: tmp_path / a)
        # Fail open: a damaged log must not silently block the nightly cycle.
        assert live.already_decided("A", "2026-07-28") is False

    def test_cycle_skips_when_already_decided(self, monkeypatch):
        monkeypatch.setattr(live, "already_decided", lambda a, d: True)
        called = []
        monkeypatch.setattr(live, "get_executor", lambda a: called.append(a))

        live.run_account_cycle("D", ["AAPL"], "2026-07-28", dry_run=False, execute=True)
        assert called == [], "must not even connect to the broker"

    def test_force_overrides(self, monkeypatch):
        monkeypatch.setattr(live, "already_decided", lambda a, d: True)
        called = []
        monkeypatch.setattr(live, "get_executor", lambda a: called.append(a) or None)

        live.run_account_cycle("D", ["AAPL"], "2026-07-28", dry_run=False,
                               execute=True, force=True)
        assert called == ["D"]

    def test_dry_run_not_blocked(self, monkeypatch):
        monkeypatch.setattr(live, "already_decided", lambda a, d: True)
        called = []
        monkeypatch.setattr(live, "get_executor", lambda a: called.append(a) or None)

        live.run_account_cycle("D", ["AAPL"], "2026-07-28", dry_run=True, execute=False)
        assert called == ["D"], "dry-runs are free to repeat"


class TestArmInvariance:
    """DJ-119: the apparatus must not constrain arms as a function of breadth.

    Book width N is downstream of the treatment under test (a diverse ensemble
    spreads, a herding one concentrates), so any rule whose bite depends on N
    is confounded with the hypothesis. These tests pin that property.
    """

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(live, "_account_dir", lambda a: tmp_path / a)

    def _book(self, n: int, loser_pnl_pct: float, equity: float = 100_000.0):
        """Equal-weight, fully invested book of n names, one down loser_pnl_pct."""
        slice_value = equity / n
        book = {}
        for i in range(n):
            cost = slice_value
            book[f"T{i}"] = Position(f"T{i}", 10, cost, cost / 10, 0.0, "long")
        cost = slice_value
        mv = cost * (1 + loser_pnl_pct)
        book["T0"] = Position("T0", 10, mv, cost / 10, cost * loser_pnl_pct, "long")
        return book

    @pytest.mark.parametrize("n", [1, 5, 20, 98])
    def test_halt_decision_is_invariant_to_book_width(self, n):
        # Same *portfolio impact* (one name losing 1% of total equity) at every
        # width must give the same answer. Under the old rule this flipped from
        # no-halt at n=1 to halt at n=98 purely because of N.
        loss_frac_of_equity = 0.01
        pnl_pct = -loss_frac_of_equity * n  # weight is 1/n, so impact is constant
        if abs(pnl_pct) >= 1.0:
            pytest.skip("loss beyond total wipeout is not representable")
        ex = _breaker_executor(equity=100_000.0, positions=self._book(n, pnl_pct))
        assert live.check_circuit_breakers("A", ex) is False, (
            f"n={n}: a 1%-of-equity loss must never halt, at any book width"
        )

    def test_effective_threshold_scales_with_width(self):
        # Equal-weight book: threshold relaxes linearly in N.
        assert live.effective_halt_threshold(1, 1.0) == pytest.approx(0.10)
        assert live.effective_halt_threshold(20, 1.0) == pytest.approx(0.40)
        assert live.effective_halt_threshold(98, 1.0) == pytest.approx(1.96)

    def test_threshold_floors_at_position_loss_limit(self):
        # A concentrated book must not halt on a trivial move: the 10% floor
        # holds even though 2%/weight would be smaller.
        assert live.effective_halt_threshold(1, 1.0) == pytest.approx(0.10)

    def test_threshold_accounts_for_cash_drag(self):
        # A at 5% invested in one name: the name must move enormously to cost
        # the book 2%, which is correct — it is 5% of the portfolio.
        assert live.effective_halt_threshold(1, 0.05) == pytest.approx(0.40)

    def test_degenerate_inputs_do_not_halt(self):
        assert live.effective_halt_threshold(0, 1.0) == float("inf")
        assert live.effective_halt_threshold(10, 0.0) == float("inf")


class TestLastCompletedSession:
    """DJ-121: date the decision by the session the agents actually see."""

    def _store(self, tmp_path, ticker, dates):
        import pandas as pd
        d = tmp_path / "market" / ticker
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {"Close": [1.0] * len(dates)},
            index=pd.DatetimeIndex(pd.to_datetime(dates), name="Date"),
        )
        df.to_parquet(d / "ohlcv.parquet")

    def test_returns_newest_bar_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        self._store(tmp_path, "AAPL", ["2026-08-05", "2026-08-06", "2026-08-07"])
        assert live._last_completed_session(["AAPL"]) == "2026-08-07"

    def test_weekend_resolves_back_to_friday(self, tmp_path, monkeypatch):
        # The Sunday 2026-08-09 case: newest bar is Friday, so the decision is
        # Friday's cycle executed late — not a phantom Sunday observation.
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        self._store(tmp_path, "AAPL", ["2026-08-06", "2026-08-07"])
        assert live._last_completed_session(["AAPL"]) == "2026-08-07"

    def test_takes_max_so_one_stale_ticker_cannot_drag_it_back(self, tmp_path, monkeypatch):
        # A halted or delisted name stops updating; it must not define the date.
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        self._store(tmp_path, "AAPL", ["2026-08-07"])
        self._store(tmp_path, "HALTED", ["2026-06-01"])
        assert live._last_completed_session(["HALTED", "AAPL"]) == "2026-08-07"

    def test_holiday_is_handled_without_a_calendar(self, tmp_path, monkeypatch):
        # No bar exists for a closed holiday, so the store answers correctly
        # with no hard-coded calendar to maintain.
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        self._store(tmp_path, "AAPL", ["2026-07-02", "2026-07-06"])  # Jul 3/4 closed
        assert live._last_completed_session(["AAPL"]) == "2026-07-06"

    def test_missing_store_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        assert live._last_completed_session(["NOPE"]) is None

    def test_unreadable_parquet_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        bad = tmp_path / "market" / "BAD"
        bad.mkdir(parents=True)
        (bad / "ohlcv.parquet").write_text("not a parquet")
        self._store(tmp_path, "AAPL", ["2026-08-07"])
        assert live._last_completed_session(["BAD", "AAPL"]) == "2026-08-07"

    def test_empty_frame_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        self._store(tmp_path, "EMPTY", [])
        self._store(tmp_path, "AAPL", ["2026-08-07"])
        assert live._last_completed_session(["EMPTY", "AAPL"]) == "2026-08-07"


class TestDataCoveragePreflight:
    """The DJ-120 gate: refuse to write decision records over a starved universe.

    This blocks rather than warns because the failure it guards is not an
    outage but a plausible-looking result — agents told TICKER_NOT_FOUND
    returned "no data available -- Sell" at confidence 1.0 for a month.
    """

    @staticmethod
    def _store(tmp_path, tickers, layout="nested"):
        import pandas as pd
        for t in tickers:
            if layout == "nested":
                d = tmp_path / "market" / t
                d.mkdir(parents=True, exist_ok=True)
                idx = pd.date_range(end="2026-08-13", periods=3, freq="D", name="Date")
                pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                              "Close": 1.0, "Volume": 1.0}, index=idx
                             ).to_parquet(d / "ohlcv.parquet")
            else:
                d = tmp_path / "market"
                d.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "Date": pd.date_range("2023-06-26", periods=3, freq="D"),
                    "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                    "Adj Close": 1.0, "Volume": 1.0,
                }).to_parquet(d / f"{t}_2016-01-01_2023-06-30.parquet")

    def test_passes_on_full_coverage(self, tmp_path, monkeypatch):
        tickers = [f"T{i}" for i in range(10)]
        self._store(tmp_path, tickers)
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        assert live.check_data_coverage(tickers) is True

    def test_blocks_when_universe_is_starved(self, tmp_path, monkeypatch):
        """83/98 missing was the production state; anything below the
        threshold must abort rather than produce bearish-looking records."""
        present = [f"T{i}" for i in range(15)]
        self._store(tmp_path, present)
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        universe = present + [f"DARK{i}" for i in range(83)]
        assert live.check_data_coverage(universe) is False

    def test_blocks_on_a_single_missing_ticker_at_default_threshold(
        self, tmp_path, monkeypatch
    ):
        tickers = [f"T{i}" for i in range(10)]
        self._store(tmp_path, tickers)
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        assert live.check_data_coverage(tickers + ["MISSING"]) is False

    def test_threshold_is_configurable(self, tmp_path, monkeypatch):
        tickers = [f"T{i}" for i in range(10)]
        self._store(tmp_path, tickers)
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        assert live.check_data_coverage(tickers + ["MISSING"], min_fraction=0.9) is True

    def test_passes_but_warns_on_stale_flat_layout(self, tmp_path, monkeypatch, caplog):
        """The quieter half of DJ-120: tickers that resolved, but to 2023 bars."""
        tickers = [f"T{i}" for i in range(10)]
        self._store(tmp_path, tickers, layout="flat")
        monkeypatch.setattr(live, "_DATA_DIR", str(tmp_path))
        with caplog.at_level("WARNING"):
            assert live.check_data_coverage(tickers) is True
        assert "STALE legacy flat layout" in caplog.text
