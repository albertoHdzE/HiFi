"""
Unit tests for PHASE14_UNIVERSE (E2-T1, DJ-090).

All tests are deterministic. No network, no LLMs, no external services.
"""

from __future__ import annotations

from collections import Counter

from hifi.data.universe import (
    _GICS_SECTORS,
    PHASE14_UNIVERSE,
    get_sector,
    get_sectors,
    get_sub_industry,
    get_tickers,
    get_tickers_by_sector,
)

_ALL_GICS_SECTORS = {
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Materials",
    "Real Estate",
    "Utilities",
}


# ---------------------------------------------------------------------------
# Universe size and structure
# ---------------------------------------------------------------------------


def test_universe_size_approximately_100() -> None:
    """Universe has ~100 tickers (target: 90-110)."""
    n = len(PHASE14_UNIVERSE)
    assert 90 <= n <= 110, f"Expected ~100 tickers, got {n}"


def test_all_11_gics_sectors_present() -> None:
    """All 11 GICS sectors are represented."""
    assert get_sectors() == _ALL_GICS_SECTORS


def test_minimum_8_tickers_per_sector() -> None:
    """Every sector has at least 8 tickers (plan requirement)."""
    sector_counts = Counter(entry["sector"] for entry in PHASE14_UNIVERSE)
    for sector, count in sector_counts.items():
        assert count >= 8, f"{sector} has only {count} tickers (min 8)"


def test_no_duplicate_tickers() -> None:
    """No ticker appears more than once."""
    tickers = [entry["ticker"] for entry in PHASE14_UNIVERSE]
    seen = Counter(tickers)
    duplicates = {t: c for t, c in seen.items() if c > 1}
    assert not duplicates, f"Duplicate tickers: {duplicates}"


def test_all_entries_have_required_fields() -> None:
    """Every entry has ticker, sector, and sub_industry keys."""
    for entry in PHASE14_UNIVERSE:
        assert "ticker" in entry, f"Missing ticker: {entry}"
        assert "sector" in entry, f"Missing sector: {entry}"
        assert "sub_industry" in entry, f"Missing sub_industry: {entry}"


def test_all_sectors_are_valid_gics() -> None:
    """Every sector value is one of the 11 official GICS sectors."""
    for entry in PHASE14_UNIVERSE:
        assert entry["sector"] in _ALL_GICS_SECTORS, (
            f"Unknown sector '{entry['sector']}' for {entry['ticker']}"
        )


def test_gics_sectors_constant_matches_universe() -> None:
    """_GICS_SECTORS constant matches the set of sectors in the universe."""
    assert _GICS_SECTORS == _ALL_GICS_SECTORS


# ---------------------------------------------------------------------------
# Ticker symbol format
# ---------------------------------------------------------------------------


def test_ticker_symbols_are_uppercase_strings() -> None:
    """All tickers are non-empty uppercase alphabetical strings."""
    for entry in PHASE14_UNIVERSE:
        t = entry["ticker"]
        assert t == t.upper(), f"Ticker not uppercase: {t!r}"
        assert all(c.isalpha() for c in t), f"Non-alpha characters in ticker: {t!r}"
        assert len(t) >= 1, "Empty ticker"


def test_sub_industry_fields_are_non_empty_strings() -> None:
    for entry in PHASE14_UNIVERSE:
        assert isinstance(entry["sub_industry"], str)
        assert len(entry["sub_industry"]) > 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_get_tickers_returns_all_tickers() -> None:
    tickers = get_tickers()
    assert len(tickers) == len(PHASE14_UNIVERSE)
    assert all(isinstance(t, str) for t in tickers)


def test_get_tickers_no_duplicates() -> None:
    tickers = get_tickers()
    assert len(tickers) == len(set(tickers))


def test_get_sectors_returns_all_11() -> None:
    assert get_sectors() == _ALL_GICS_SECTORS


def test_get_tickers_by_sector_information_technology() -> None:
    it_tickers = get_tickers_by_sector("Information Technology")
    assert len(it_tickers) >= 8
    assert "AAPL" in it_tickers
    assert "MSFT" in it_tickers


def test_get_tickers_by_sector_energy() -> None:
    energy_tickers = get_tickers_by_sector("Energy")
    assert len(energy_tickers) >= 8
    assert "XOM" in energy_tickers


def test_get_tickers_by_sector_financials() -> None:
    fin_tickers = get_tickers_by_sector("Financials")
    assert len(fin_tickers) >= 8
    assert "JPM" in fin_tickers


def test_get_tickers_by_sector_unknown_returns_empty() -> None:
    assert get_tickers_by_sector("Fake Sector") == []


def test_get_sector_known_tickers() -> None:
    assert get_sector("AAPL") == "Information Technology"
    assert get_sector("JPM") == "Financials"
    assert get_sector("XOM") == "Energy"
    assert get_sector("NEE") == "Utilities"
    assert get_sector("PLD") == "Real Estate"
    assert get_sector("LIN") == "Materials"


def test_get_sector_unknown_ticker_returns_none() -> None:
    assert get_sector("FAKE") is None
    assert get_sector("") is None


def test_get_sub_industry_known_ticker() -> None:
    sub = get_sub_industry("AAPL")
    assert sub is not None
    assert isinstance(sub, str)
    assert len(sub) > 0


def test_get_sub_industry_unknown_ticker_returns_none() -> None:
    assert get_sub_industry("FAKE") is None


# ---------------------------------------------------------------------------
# Spot-checks: one representative ticker per sector
# ---------------------------------------------------------------------------


def test_known_tickers_cover_all_sectors() -> None:
    """Spot-check that one expected ticker is present per sector."""
    sector_spot_checks = {
        "Information Technology": "AAPL",
        "Health Care": "JNJ",
        "Financials": "JPM",
        "Consumer Discretionary": "AMZN",
        "Communication Services": "GOOGL",
        "Industrials": "CAT",
        "Consumer Staples": "PG",
        "Energy": "XOM",
        "Materials": "LIN",
        "Real Estate": "PLD",
        "Utilities": "NEE",
    }
    all_tickers = set(get_tickers())
    for sector, expected in sector_spot_checks.items():
        assert expected in all_tickers, f"{expected} ({sector}) not in universe"


def test_phase13_tickers_still_in_universe() -> None:
    """AAPL, JPM, XOM were used in Phase 13 evals and must remain in universe."""
    all_tickers = set(get_tickers())
    for ticker in ("AAPL", "JPM", "XOM"):
        assert ticker in all_tickers, f"Phase 13 eval ticker {ticker} missing from universe"
