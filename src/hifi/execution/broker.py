"""Broker-agnostic execution protocol (Phase 16, DJ-098).

Concrete implementations: AlpacaExecutor, (future) IBKRExecutor, BinanceExecutor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Position:
    ticker: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pnl: float
    side: str  # "long" | "short"


@dataclass(frozen=True)
class OrderResult:
    ticker: str
    side: str  # "buy" | "sell"
    qty: float
    filled_avg_price: float | None
    status: str  # "filled", "partial", "rejected", "pending"
    order_id: str
    raw: dict | None = None


@runtime_checkable
class BrokerExecutor(Protocol):
    """Minimal interface every broker adapter must satisfy."""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_account_cash(self) -> float: ...
    def get_portfolio_value(self) -> float: ...
    def get_positions(self) -> dict[str, Position]: ...
    def place_market_order(self, ticker: str, qty: float, side: str) -> OrderResult: ...
    def close_position(self, ticker: str) -> OrderResult: ...
    def close_all_positions(self) -> list[OrderResult]: ...
