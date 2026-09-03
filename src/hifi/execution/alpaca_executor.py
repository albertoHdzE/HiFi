"""Alpaca paper-trading executor (Phase 16, DJ-098).

Uses alpaca-py SDK. Reads credentials from environment:
  ALPACA_API_KEY, ALPACA_SECRET, ALPACA_END_POINT
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypeVar

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Position as AlpacaPosition
from alpaca.trading.requests import MarketOrderRequest

from hifi.execution.broker import OrderResult, Position
from hifi.execution.retry import with_retry

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _model(value: _T | dict[str, Any]) -> _T:
    """Narrow an alpaca-py response to its model, refusing the raw-dict branch.

    Every ``TradingClient`` method is typed ``Model | dict[str, Any]`` because
    the client can be constructed with ``raw_data=True``, in which case it hands
    back parsed JSON instead. HiFi never does that, so the dict branch is
    unreachable — but it is unreachable by convention, not by construction, and
    the convention lives in one place only if it is written down.

    Asserting it here rather than sprinkling ``# type: ignore`` at forty call
    sites means that if anyone ever does pass ``raw_data=True``, this raises at
    the boundary with a message naming the cause, instead of producing an
    AttributeError deep inside a nightly cycle at 21:30.
    """
    if isinstance(value, dict):
        raise TypeError(
            "alpaca-py returned raw JSON rather than a model object; the "
            "TradingClient must not be constructed with raw_data=True"
        )
    return value


def _num(value: str | float | None, field: str) -> float:
    """Convert a money field, refusing to invent a number when it is absent.

    alpaca-py types equity, cash, buying_power and the rest as ``str | None``.
    ``float(None)`` raises a TypeError naming neither the field nor the account,
    and the tempting alternative — ``float(x or 0)`` — is worse: a funded
    account reporting no equity would read as a total loss and trip the
    drawdown guard (DJ-129b) into halting an arm that is perfectly healthy.

    So: absent is an error, and the error says which field.
    """
    if value is None:
        raise ValueError(
            f"Alpaca returned no value for {field!r}; treating a missing "
            "balance as a number would misreport the account"
        )
    return float(value)


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
        acct = _model(self._client.get_account())
        logger.info(
            "Alpaca connected: paper=%s equity=$%.2f cash=$%.2f",
            self._paper, _num(acct.equity, "equity"), _num(acct.cash, "cash"),
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
        return _num(_model(self.client.get_account()).cash, "cash")

    @with_retry()
    def get_portfolio_value(self) -> float:
        return _num(_model(self.client.get_account()).equity, "equity")

    @with_retry()
    def get_account_snapshot(self) -> dict:
        """Full point-in-time account state for the equity/performance record."""
        a = _model(self.client.get_account())
        return {
            "equity": _num(a.equity, "equity"),
            "last_equity": _num(a.last_equity, "last_equity"),
            "cash": _num(a.cash, "cash"),
            "buying_power": _num(a.buying_power, "buying_power"),
            # A flat account genuinely holds nothing, so None is 0.0 here
            # and only here.
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
        h = _model(self.client.get_portfolio_history(req))
        return {
            "timestamp": list(h.timestamp),
            "equity": [float(x) if x is not None else None for x in h.equity],
            "profit_loss": [float(x) if x is not None else None for x in h.profit_loss],
            "profit_loss_pct": [float(x) if x is not None else None for x in h.profit_loss_pct],
        }

    @with_retry()
    def get_positions(self) -> dict[str, Position]:
        positions: list[AlpacaPosition] = _model(self.client.get_all_positions())
        result: dict[str, Position] = {}
        for p in positions:
            result[p.symbol] = Position(
                ticker=p.symbol,
                qty=float(p.qty),
                market_value=_num(p.market_value, "market_value"),
                avg_entry_price=_num(p.avg_entry_price, "avg_entry_price"),
                unrealized_pnl=_num(p.unrealized_pl, "unrealized_pl"),
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
            return bool(_model(self.client.get_asset(ticker)).fractionable)
        except Exception as exc:
            logger.warning(
                "is_fractionable(%s) failed (%s); assuming NOT fractionable, so this "
                "order will be sized in shares rather than dollars", ticker, exc)
            return False

    @with_retry()
    def get_client_order_ids(self, limit: int = 500) -> set[str]:
        """Client order ids visible among this account's recent orders (DJ-129a).

        Scans both open and closed recent orders. The nightly cycle uses this to
        make re-runs idempotent: a deterministic client_order_id submitted twice
        must produce one order at the broker, never two. The installed alpaca-py
        has no get_order_by_client_order_id, so the scan is local over recent
        history; a night's book is ~100 orders per account, well inside limit.

        Raises on API failure — callers treat "cannot verify" as "do not submit"
        (fail-closed), which is the only safe direction for an idempotency gate.
        """
        from alpaca.trading.requests import GetOrdersRequest

        ids: set[str] = set()
        for status in (QueryOrderStatus.OPEN, QueryOrderStatus.CLOSED):
            req = GetOrdersRequest(status=status, limit=limit)
            for order in self.client.get_orders(req):
                cid = getattr(order, "client_order_id", None)
                if cid:
                    ids.add(str(cid))
        return ids

    def place_market_order(
        self, ticker: str, qty: float, side: str, notional: float | None = None,
        client_order_id: str | None = None,
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

        ``client_order_id`` (DJ-129a) makes submission idempotent: the broker
        rejects a duplicate id, so a crash-and-rerun of the nightly cycle can
        never double-fill. Callers derive it deterministically from
        (account, decision date, ticker, side) so a re-run reproduces the same
        ids for the same intended orders.
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        def _req(**kwargs) -> MarketOrderRequest:
            if client_order_id:
                kwargs["client_order_id"] = client_order_id
            return MarketOrderRequest(**kwargs)

        if notional is not None:
            if side.lower() != "buy":
                raise ValueError("notional orders are BUY-only (long-only book)")
            req = _req(
                symbol=ticker,
                notional=round(float(notional), 2),
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            req = _req(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        order = _model(self.client.submit_order(req))
        logger.info(
            "Order submitted: %s %s %s → %s (id=%s%s)",
            side, ticker,
            f"${notional:,.2f}" if notional is not None else f"x{qty:.3f}",
            order.status, order.id,
            f", coid={client_order_id}" if client_order_id else "",
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
            raw={"client_order_id": client_order_id or order.client_order_id,
                 "notional": round(float(notional), 2) if notional is not None else None},
        )

    def close_position(self, ticker: str) -> OrderResult:
        order = _model(self.client.close_position(ticker))
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

