"""Alpaca paper-trading executor (Phase 16, DJ-098).

Uses alpaca-py SDK. Reads credentials from environment:
  ALPACA_API_KEY, ALPACA_SECRET, ALPACA_END_POINT
"""

from __future__ import annotations

import logging
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from hifi.execution.broker import OrderResult, Position
from hifi.execution.retry import with_retry

logger = logging.getLogger(__name__)


class AlpacaExecutor:
    """BrokerExecutor implementation for Alpaca paper trading."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ) -> None:
        self._api_key = api_key or os.environ["ALPACA_API_KEY"]
        self._secret_key = secret_key or os.environ["ALPACA_SECRET"]
        self._paper = paper
        self._client: TradingClient | None = None

    @with_retry()
    def connect(self) -> None:
        self._client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )
        acct = self._client.get_account()
        logger.info(
            "Alpaca connected: paper=%s equity=$%.2f cash=$%.2f",
            self._paper, float(acct.equity), float(acct.cash),
        )

    def disconnect(self) -> None:
        self._client = None

    @property
    def client(self) -> TradingClient:
        if self._client is None:
            raise RuntimeError("Call connect() first")
        return self._client

    @with_retry()
    def get_account_cash(self) -> float:
        return float(self.client.get_account().cash)

    @with_retry()
    def get_portfolio_value(self) -> float:
        return float(self.client.get_account().equity)

    @with_retry()
    def get_account_snapshot(self) -> dict:
        """Full point-in-time account state for the equity/performance record."""
        a = self.client.get_account()
        return {
            "equity": float(a.equity),
            "last_equity": float(a.last_equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "long_market_value": float(a.long_market_value or 0.0),
        }

    @with_retry()
    def get_portfolio_history(self, period: str = "all", timeframe: str = "1D") -> dict:
        """Authoritative, close-marked daily equity curve from Alpaca.

        Returns {timestamp: [epoch...], equity: [...], profit_loss: [...],
        profit_loss_pct: [...]}. Server-computed, gap-free, no human input.
        """
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        req = GetPortfolioHistoryRequest(
            period=period, timeframe=timeframe, intraday_reporting="market_hours"
        )
        h = self.client.get_portfolio_history(req)
        return {
            "timestamp": list(h.timestamp),
            "equity": [float(x) if x is not None else None for x in h.equity],
            "profit_loss": [float(x) if x is not None else None for x in h.profit_loss],
            "profit_loss_pct": [float(x) if x is not None else None for x in h.profit_loss_pct],
        }

    @with_retry()
    def get_positions(self) -> dict[str, Position]:
        positions = self.client.get_all_positions()
        result: dict[str, Position] = {}
        for p in positions:
            result[p.symbol] = Position(
                ticker=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                unrealized_pnl=float(p.unrealized_pl),
                side=str(p.side).split(".")[-1].lower(),
            )
        return result

    @with_retry()
    def is_fractionable(self, ticker: str) -> bool:
        try:
            return bool(self.client.get_asset(ticker).fractionable)
        except Exception:
            return False

    def place_market_order(self, ticker: str, qty: float, side: str) -> OrderResult:
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(req)
        logger.info(
            "Order submitted: %s %s x%.1f → %s (id=%s)",
            side, ticker, qty, order.status, order.id,
        )
        return OrderResult(
            ticker=ticker,
            side=side.lower(),
            qty=qty,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            status=str(order.status).split(".")[-1].lower(),
            order_id=str(order.id),
            raw={"client_order_id": order.client_order_id},
        )

    def close_position(self, ticker: str) -> OrderResult:
        order = self.client.close_position(ticker)
        return OrderResult(
            ticker=ticker,
            side="sell",
            qty=float(order.qty) if order.qty else 0,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            status=str(order.status).split(".")[-1].lower(),
            order_id=str(order.id),
        )

    def close_all_positions(self) -> list[OrderResult]:
        responses = self.client.close_all_positions(cancel_orders=True)
        results = []
        for resp in responses:
            order = resp.body if hasattr(resp, "body") else resp
            results.append(OrderResult(
                ticker=str(getattr(order, "symbol", "?")),
                side="sell",
                qty=float(order.qty) if hasattr(order, "qty") and order.qty else 0,
                filled_avg_price=None,
                status=str(getattr(order, "status", "unknown")).split(".")[-1].lower(),
                order_id=str(getattr(order, "id", "")),
            ))
        return results

