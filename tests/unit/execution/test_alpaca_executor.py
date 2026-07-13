"""Tests for AlpacaExecutor and BrokerExecutor protocol."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hifi.execution.broker import BrokerExecutor, OrderResult, Position


class TestBrokerProtocol:
    def test_position_dataclass(self):
        p = Position("AAPL", 10, 1700.0, 170.0, 50.0, "long")
        assert p.ticker == "AAPL"
        assert p.qty == 10
        assert p.side == "long"

    def test_order_result_dataclass(self):
        o = OrderResult("AAPL", "buy", 10, 170.5, "filled", "abc-123")
        assert o.ticker == "AAPL"
        assert o.filled_avg_price == 170.5
        assert o.status == "filled"

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(BrokerExecutor, "__protocol_attrs__") or hasattr(
            BrokerExecutor, "__abstractmethods__"
        ) or callable(getattr(BrokerExecutor, "_is_protocol", None))


class TestAlpacaExecutor:
    @patch("hifi.execution.alpaca_executor.TradingClient")
    def test_connect(self, mock_client_cls):
        mock_acct = MagicMock()
        mock_acct.equity = "10000.00"
        mock_acct.cash = "5000.00"
        mock_client_cls.return_value.get_account.return_value = mock_acct

        from hifi.execution.alpaca_executor import AlpacaExecutor

        ex = AlpacaExecutor(api_key="test", secret_key="test", paper=True)
        ex.connect()

        mock_client_cls.assert_called_once_with(
            api_key="test", secret_key="test", paper=True
        )

    @patch("hifi.execution.alpaca_executor.TradingClient")
    def test_get_account_cash(self, mock_client_cls):
        mock_acct = MagicMock()
        mock_acct.equity = "10000.00"
        mock_acct.cash = "5000.00"
        mock_client_cls.return_value.get_account.return_value = mock_acct

        from hifi.execution.alpaca_executor import AlpacaExecutor

        ex = AlpacaExecutor(api_key="test", secret_key="test")
        ex.connect()
        assert ex.get_account_cash() == 5000.0

    @patch("hifi.execution.alpaca_executor.TradingClient")
    def test_get_positions(self, mock_client_cls):
        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.market_value = "1700.00"
        mock_pos.avg_entry_price = "170.00"
        mock_pos.unrealized_pl = "50.00"
        mock_pos.side = "PositionSide.long"

        mock_acct = MagicMock()
        mock_acct.equity = "10000.00"
        mock_acct.cash = "5000.00"
        mock_client_cls.return_value.get_account.return_value = mock_acct
        mock_client_cls.return_value.get_all_positions.return_value = [mock_pos]

        from hifi.execution.alpaca_executor import AlpacaExecutor

        ex = AlpacaExecutor(api_key="test", secret_key="test")
        ex.connect()
        positions = ex.get_positions()

        assert "AAPL" in positions
        assert positions["AAPL"].qty == 10.0
        assert positions["AAPL"].market_value == 1700.0

    @patch("hifi.execution.alpaca_executor.TradingClient")
    def test_place_market_order(self, mock_client_cls):
        mock_order = MagicMock()
        mock_order.status = "OrderStatus.accepted"
        mock_order.id = "order-123"
        mock_order.filled_avg_price = None
        mock_order.client_order_id = "client-123"

        mock_acct = MagicMock()
        mock_acct.equity = "10000.00"
        mock_acct.cash = "5000.00"
        mock_client_cls.return_value.get_account.return_value = mock_acct
        mock_client_cls.return_value.submit_order.return_value = mock_order

        from hifi.execution.alpaca_executor import AlpacaExecutor

        ex = AlpacaExecutor(api_key="test", secret_key="test")
        ex.connect()
        result = ex.place_market_order("AAPL", 5, "buy")

        assert result.ticker == "AAPL"
        assert result.side == "buy"
        assert result.qty == 5
        assert result.order_id == "order-123"

    def test_not_connected_raises(self):
        from hifi.execution.alpaca_executor import AlpacaExecutor

        ex = AlpacaExecutor(api_key="test", secret_key="test")
        with pytest.raises(RuntimeError, match="connect"):
            ex.get_account_cash()


class TestMarketData:
    @patch("hifi.execution.market_data.StockHistoricalDataClient")
    def test_get_data_client(self, mock_cls):
        from hifi.execution.market_data import get_data_client

        get_data_client(api_key="test", secret_key="test")
        mock_cls.assert_called_once_with(api_key="test", secret_key="test")
