"""
Unit tests for SMOKE_UNIVERSE (E2-T1, DJ-107).

Verifies:
- Exactly 22 tickers
- All 11 GICS sectors represented
- Exactly 2 tickers per sector
- All tickers are members of PHASE14_UNIVERSE
- Schema keys match PHASE14_UNIVERSE (ticker, sector, sub_industry)
"""

from __future__ import annotations

from hifi.data.smoke_universe import _EXPECTED_SECTORS, SMOKE_UNIVERSE
from hifi.data.universe import PHASE14_UNIVERSE

_PHASE14_TICKERS: frozenset[str] = frozenset(e["ticker"] for e in PHASE14_UNIVERSE)
_PHASE14_SECTORS: frozenset[str] = frozenset(e["sector"] for e in PHASE14_UNIVERSE)


def test_smoke_universe_length():
    assert len(SMOKE_UNIVERSE) == 22


def test_smoke_universe_all_11_sectors_covered():
    sectors = {e["sector"] for e in SMOKE_UNIVERSE}
    assert sectors == _EXPECTED_SECTORS, (
        f"Missing sectors: {_EXPECTED_SECTORS - sectors}; "
        f"Extra sectors: {sectors - _EXPECTED_SECTORS}"
    )


def test_smoke_universe_exactly_two_per_sector():
    from collections import Counter
    counts = Counter(e["sector"] for e in SMOKE_UNIVERSE)
    for sector, count in counts.items():
        assert count == 2, f"Sector {sector!r} has {count} tickers (expected 2)"


def test_all_smoke_tickers_in_phase14_universe():
    for entry in SMOKE_UNIVERSE:
        ticker = entry["ticker"]
        assert ticker in _PHASE14_TICKERS, (
            f"{ticker!r} is in SMOKE_UNIVERSE but not in PHASE14_UNIVERSE"
        )


def test_smoke_universe_schema_keys():
    required_keys = {"ticker", "sector", "sub_industry"}
    for entry in SMOKE_UNIVERSE:
        assert set(entry.keys()) == required_keys, (
            f"Entry {entry!r} has unexpected keys"
        )


def test_smoke_universe_expected_tickers():
    """Spot-check the specific tickers specified in DJ-107."""
    expected = {
        "AAPL", "CRM",      # IT
        "UNH", "ABT",       # Health Care
        "JPM", "BLK",       # Financials
        "AMZN", "NKE",      # Consumer Discretionary
        "GOOGL", "DIS",     # Communication Services
        "HON", "CAT",       # Industrials
        "PG", "COST",       # Consumer Staples
        "XOM", "COP",       # Energy
        "LIN", "FCX",       # Materials
        "PLD", "AMT",       # Real Estate
        "NEE", "DUK",       # Utilities
    }
    actual = {e["ticker"] for e in SMOKE_UNIVERSE}
    assert actual == expected
