"""Narrowing helpers for the alpaca-py boundary (DJ-140, DJ-142).

Every alpaca-py client method is typed ``Model | dict[str, Any]``, because a
client can be constructed with ``raw_data=True`` and then hands back parsed JSON
instead of a model object. HiFi never does that, so the dict branch is
unreachable — but it is unreachable *by convention*, and a convention only lives
in one place if it is written down in one place.

Three modules cross that boundary: ``execution/alpaca_executor`` (orders,
positions, account), ``execution/market_data`` (bars) and ``data/news``
(headlines). They had the same forty type errors between them and would
otherwise have grown the same forty ``# type: ignore`` comments, which suppress
the message without answering it.
"""

from __future__ import annotations

from typing import Any, TypeVar

_T = TypeVar("_T")


def model(value: _T | dict[str, Any]) -> _T:
    """Narrow an alpaca-py response to its model, refusing the raw-dict branch.

    Raises at the boundary with a message naming the cause, rather than letting
    the mistake surface as an AttributeError deep inside a nightly cycle.
    """
    if isinstance(value, dict):
        raise TypeError(
            "alpaca-py returned raw JSON rather than a model object; the "
            "client must not be constructed with raw_data=True"
        )
    return value


def num(value: str | float | None, field: str) -> float:
    """Convert a money field, refusing to invent a number when it is absent.

    alpaca-py types equity, cash, buying_power and the rest as ``str | None``.
    ``float(None)`` raises a TypeError naming neither the field nor the account,
    and the tempting alternative — ``float(x or 0)`` — is worse: a funded
    account reporting no equity would read as a total loss and trip the drawdown
    guard (DJ-129b) into halting an arm that is perfectly healthy.

    So: absent is an error, and the error says which field.
    """
    if value is None:
        raise ValueError(
            f"Alpaca returned no value for {field!r}; treating a missing "
            "balance as a number would misreport the account"
        )
    return float(value)
