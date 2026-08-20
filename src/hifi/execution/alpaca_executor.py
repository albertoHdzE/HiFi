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
        """Whether the broker will accept fractional/notional orders for ticker.

        Returns False on error, which is the safe direction — a share order
        always works — but the two cases are NOT equivalent and the caller
        cannot see the difference, so log it. Since DJ-126 this answer decides
        whether a buy is sized in dollars or shares: a transient API blip would
        silently downgrade that ticker to share sizing and reintroduce the
        overnight-gap margin problem for it alone.
        """
        try:
            return bool(self.client.get_asset(ticker).fractionable)
        except Exception as exc:
            logger.warning(
                "is_fractionable(%s) failed (%s); assuming NOT fractionable, so this "
                "order will be sized in shares rather than dollars", ticker, exc)
            return False

    def place_market_order(
        self, ticker: str, qty: float, side: str, notional: float | None = None
    ) -> OrderResult:
        """Submit a market DAY order, by share count or by dollar amount.

        ``notional`` sizes the order in dollars instead of shares (DJ-126).
        This matters because orders are sized after the close and fill at the
        next open: a share-count order spends whatever the gap decides, so on
        2026-08-18 all three pipeline arms overshot their cash budget (by 2.79%,
        4.90% and 1.44%) and ended on slight margin. A notional order spends
        exactly the amount requested whatever the open brings.

        Alpaca accepts notional only for fractionable assets, so callers must
        check ``is_fractionable`` and fall back to ``qty``. Notional is used for
        BUYs only: a notional SELL could exceed the shares actually held if the
        price gapped down, and this is a long-only book.
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        if notional is not None:
            if side.lower() != "buy":
                raise ValueError("notional orders are BUY-only (long-only book)")
            req = MarketOrderRequest(
                symbol=ticker,
                notional=round(float(notional), 2),
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        order = self.client.submit_order(req)
        logger.info(
            "Order submitted: %s %s %s → %s (id=%s)",
            side, ticker,
            f"${notional:,.2f}" if notional is not None else f"x{qty:.3f}",
            order.status, order.id,
        )
        return OrderResult(
            ticker=ticker,
            side=side.lower(),
            # Alpaca resolves a notional order to a share count at fill; until
            # then qty is None. Report what we know rather than inventing it.
            # A share order reports what was requested (the broker echoes it).
            # A notional order has no share count until it fills, so report the
            # broker's value when present and 0.0 rather than inventing one.
            qty=(qty if notional is None
                 else (float(order.qty) if order.qty else 0.0)),
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            status=str(order.status).split(".")[-1].lower(),
            order_id=str(order.id),
            raw={"client_order_id": order.client_order_id,
                 "notional": round(float(notional), 2) if notional is not None else None},
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

