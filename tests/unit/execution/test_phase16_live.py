"""Tests for the Phase 16 live orchestrator (control strategy, account routing)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestAccountRouting:
    def test_account_conditions(self):
        assert live._ACCOUNTS["A"]["condition"] == "parallel"
        assert live._ACCOUNTS["B"]["condition"] == "full"
        assert live._ACCOUNTS["C"]["condition"] == "control"

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
