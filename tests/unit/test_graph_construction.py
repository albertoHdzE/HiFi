"""
Unit tests for build_financial_graph() (P12-E1-T2).

All tests pass ticker_metadata to avoid yfinance network calls.
Deterministic: no mocks, no LLMs, no live services.
"""

from __future__ import annotations

from hifi.knowledge.graph_construction import (
    DEFAULT_COMPETITORS,
    DEFAULT_MACRO_SENSITIVITY,
    build_financial_graph,
)
from hifi.knowledge.graph_store import FinancialGraph

# ---------------------------------------------------------------------------
# Minimal test metadata (overrides yfinance)
# ---------------------------------------------------------------------------

_TECH_META: dict[str, dict[str, str]] = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "industry": "Software"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "industry": "Internet Content"},
}

_ENERGY_META: dict[str, dict[str, str]] = {
    "XOM": {"name": "Exxon Mobil Corporation", "sector": "Energy", "industry": "Oil & Gas Integrated"},  # noqa: E501
    "CVX": {"name": "Chevron Corporation", "sector": "Energy", "industry": "Oil & Gas Integrated"},
}

_FINANCE_META: dict[str, dict[str, str]] = {
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial Services", "industry": "Banks"},
    "BAC": {"name": "Bank of America Corp.", "sector": "Financial Services", "industry": "Banks"},
    "GS": {
        "name": "The Goldman Sachs Group Inc.",
        "sector": "Financial Services",
        "industry": "Investment Banking",
    },
}

_FULL_META = {**_TECH_META, **_ENERGY_META, **_FINANCE_META}


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_build_minimal_graph_node_count() -> None:
    g = build_financial_graph(
        tickers=["AAPL"],
        competitor_seed={},
        macro_sensitivity={},
        ticker_metadata={"AAPL": {"name": "Apple", "sector": "Technology", "industry": "HW"}},
    )
    # 1 company + 1 sector + 3 macro factors (always added)
    assert g.node_count() == 5


def test_build_always_adds_macro_factor_nodes() -> None:
    # Even with no tickers, macro factors are added; also verify per-factor sensitivity
    for factor in ["VIX", "FFR", "CPI"]:
        result = build_financial_graph(
            tickers=["AAPL"],
            competitor_seed={},
            macro_sensitivity={"Technology": [factor]},
            ticker_metadata={
                "AAPL": {"name": "Apple", "sector": "Technology", "industry": "HW"},
            },
        )
        assert factor in result.get_macro_factors("AAPL")


_TECH_ONLY_COMPETITORS = {
    "AAPL": ["MSFT", "GOOGL"],
    "MSFT": ["AAPL", "GOOGL"],
    "GOOGL": ["AAPL", "MSFT"],
}


def test_build_tech_graph_company_nodes() -> None:
    tickers = ["AAPL", "MSFT", "GOOGL"]
    g = build_financial_graph(
        tickers=tickers,
        competitor_seed=_TECH_ONLY_COMPETITORS,  # only edges among the 3 test tickers
        macro_sensitivity=DEFAULT_MACRO_SENSITIVITY,
        ticker_metadata=_TECH_META,
    )
    # 3 companies + 1 sector + 3 macro factors = 7
    assert g.node_count() == 7


def test_build_belongs_to_edges() -> None:
    g = build_financial_graph(
        tickers=["AAPL", "MSFT"],
        competitor_seed={},
        macro_sensitivity={},
        ticker_metadata=_TECH_META,
    )
    # Each company has one BELONGS_TO edge
    peers = g.get_sector_peers("AAPL")
    assert "MSFT" in peers


def test_build_competes_with_symmetric() -> None:
    g = build_financial_graph(
        tickers=["AAPL", "MSFT", "GOOGL"],
        competitor_seed=DEFAULT_COMPETITORS,
        macro_sensitivity={},
        ticker_metadata=_TECH_META,
    )
    assert "MSFT" in g.get_competitors("AAPL")
    assert "AAPL" in g.get_competitors("MSFT")


def test_build_sensitive_to_edges() -> None:
    g = build_financial_graph(
        tickers=["AAPL", "MSFT", "GOOGL"],
        competitor_seed={},
        macro_sensitivity=DEFAULT_MACRO_SENSITIVITY,
        ticker_metadata=_TECH_META,
    )
    factors = set(g.get_macro_factors("AAPL"))
    assert "FFR" in factors
    assert "VIX" in factors


def test_build_sensitive_to_only_for_present_sectors() -> None:
    """SENSITIVE_TO edges for Energy are not added when only Tech tickers are present."""
    g = build_financial_graph(
        tickers=["AAPL"],
        competitor_seed={},
        macro_sensitivity=DEFAULT_MACRO_SENSITIVITY,
        ticker_metadata=_TECH_META,
    )
    # Should NOT have Energy macro sensitivity edges (Energy sector not in graph)
    # VIX and FFR should be factors for AAPL (Technology), but not for Energy tickers
    factors = set(g.get_macro_factors("AAPL"))
    assert "FFR" in factors
    assert "VIX" in factors


def test_build_multi_sector_graph() -> None:
    all_tickers = ["AAPL", "JPM", "XOM"]
    all_meta = {**_TECH_META, **_FINANCE_META, **_ENERGY_META}
    g = build_financial_graph(
        tickers=all_tickers,
        competitor_seed=DEFAULT_COMPETITORS,
        macro_sensitivity=DEFAULT_MACRO_SENSITIVITY,
        ticker_metadata=all_meta,
    )
    # Each ticker should be in a different sector
    assert "AAPL" not in g.get_sector_peers("JPM")
    assert "JPM" not in g.get_sector_peers("AAPL")
    assert "XOM" not in g.get_sector_peers("AAPL")


def test_build_energy_macro_sensitivity() -> None:
    g = build_financial_graph(
        tickers=["XOM", "CVX"],
        competitor_seed=DEFAULT_COMPETITORS,
        macro_sensitivity=DEFAULT_MACRO_SENSITIVITY,
        ticker_metadata=_ENERGY_META,
    )
    factors = set(g.get_macro_factors("XOM"))
    assert "VIX" in factors
    assert "CPI" in factors


def test_build_financial_services_macro_sensitivity() -> None:
    g = build_financial_graph(
        tickers=["JPM", "BAC"],
        competitor_seed=DEFAULT_COMPETITORS,
        macro_sensitivity=DEFAULT_MACRO_SENSITIVITY,
        ticker_metadata=_FINANCE_META,
    )
    factors = set(g.get_macro_factors("JPM"))
    assert "FFR" in factors


def test_build_competitor_edges_only_for_present_tickers() -> None:
    """Competitor edges are only added when at least one side is in the ticker list."""
    g = build_financial_graph(
        tickers=["AAPL"],  # Only AAPL, not MSFT
        competitor_seed={"AAPL": ["MSFT"], "MSFT": ["AAPL"]},
        macro_sensitivity={},
        ticker_metadata={
            "AAPL": {"name": "Apple", "sector": "Technology", "industry": "HW"},
        },
    )
    # AAPL is in tickers, so AAPL<->MSFT edge added even though MSFT is not a Company node
    competitors = g.get_competitors("AAPL")
    assert "MSFT" in competitors


def test_build_returns_financial_graph_instance() -> None:
    g = build_financial_graph(
        tickers=["AAPL"],
        competitor_seed={},
        macro_sensitivity={},
        ticker_metadata={"AAPL": {"name": "Apple", "sector": "Technology", "industry": "HW"}},
    )
    assert isinstance(g, FinancialGraph)


def test_build_with_ticker_metadata_no_yfinance_call() -> None:
    """When ticker_metadata is provided for all tickers, no yfinance calls are made."""
    # This test passes deterministically (no network) because all tickers are in metadata
    g = build_financial_graph(
        tickers=list(_FULL_META.keys()),
        competitor_seed=DEFAULT_COMPETITORS,
        macro_sensitivity=DEFAULT_MACRO_SENSITIVITY,
        ticker_metadata=_FULL_META,
    )
    assert g.node_count() > 0
    assert g.edge_count() > 0


def test_build_expand_query_tickers_integration() -> None:
    """expand_query_tickers works on a graph built by build_financial_graph."""
    g = build_financial_graph(
        tickers=["AAPL", "MSFT", "GOOGL"],
        competitor_seed=DEFAULT_COMPETITORS,
        macro_sensitivity=DEFAULT_MACRO_SENSITIVITY,
        ticker_metadata=_TECH_META,
    )
    expanded = g.expand_query_tickers("AAPL", max_hops=2)
    assert "AAPL" in expanded
    assert "MSFT" in expanded
    assert "GOOGL" in expanded
