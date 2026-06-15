"""
Unit tests for FinancialGraph (P12-E1-T1).

All tests use small deterministic graphs -- no yfinance, no OHLCV, no LLMs.
"""

from __future__ import annotations

from pathlib import Path

from hifi.knowledge.graph_store import FinancialGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tech_graph() -> FinancialGraph:
    """Small Technology sector graph for testing."""
    g = FinancialGraph()
    g.add_sector("Technology")
    g.add_macro_factor("FFR", "FEDFUNDS")
    g.add_macro_factor("VIX", "VIXCLS")
    g.add_company("AAPL", "Apple Inc.", "Technology", "Consumer Electronics")
    g.add_company("MSFT", "Microsoft Corporation", "Technology", "Software")
    g.add_company("GOOGL", "Alphabet Inc.", "Technology", "Internet Content")
    g.add_belongs_to("AAPL", "Technology")
    g.add_belongs_to("MSFT", "Technology")
    g.add_belongs_to("GOOGL", "Technology")
    g.add_competes_with("AAPL", "MSFT")
    g.add_competes_with("AAPL", "GOOGL")
    g.add_sensitive_to("Technology", "FFR")
    g.add_sensitive_to("Technology", "VIX")
    return g


# ---------------------------------------------------------------------------
# Node creation
# ---------------------------------------------------------------------------

def test_add_company_node() -> None:
    g = FinancialGraph()
    g.add_company("AAPL", "Apple Inc.", "Technology", "Consumer Electronics")
    assert g.node_count() == 1


def test_add_sector_node() -> None:
    g = FinancialGraph()
    g.add_sector("Technology")
    assert g.node_count() == 1


def test_add_macro_factor_node() -> None:
    g = FinancialGraph()
    g.add_macro_factor("VIX", "VIXCLS")
    assert g.node_count() == 1


def test_node_count_full_graph() -> None:
    g = _make_tech_graph()
    # 3 companies + 1 sector + 2 macro factors
    assert g.node_count() == 6


# ---------------------------------------------------------------------------
# Edge creation
# ---------------------------------------------------------------------------

def test_competes_with_is_symmetric() -> None:
    g = FinancialGraph()
    g.add_company("AAPL", "Apple", "Tech", "HW")
    g.add_company("MSFT", "Microsoft", "Tech", "SW")
    g.add_competes_with("AAPL", "MSFT")
    assert "MSFT" in g.get_competitors("AAPL")
    assert "AAPL" in g.get_competitors("MSFT")


def test_belongs_to_edge_count() -> None:
    g = FinancialGraph()
    g.add_sector("Technology")
    g.add_company("AAPL", "Apple", "Technology", "HW")
    g.add_belongs_to("AAPL", "Technology")
    # 1 BELONGS_TO edge
    assert g.edge_count() == 1


def test_sensitive_to_edge() -> None:
    g = FinancialGraph()
    g.add_sector("Technology")
    g.add_macro_factor("FFR", "FEDFUNDS")
    g.add_sensitive_to("Technology", "FFR")
    assert g.edge_count() == 1


def test_edge_count_full_graph() -> None:
    g = _make_tech_graph()
    # 3 BELONGS_TO + 4 COMPETES_WITH (AAPL<->MSFT, AAPL<->GOOGL = 4 directed) + 2 SENSITIVE_TO
    assert g.edge_count() == 9


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def test_get_competitors_known_ticker() -> None:
    g = _make_tech_graph()
    competitors = g.get_competitors("AAPL")
    assert set(competitors) == {"MSFT", "GOOGL"}


def test_get_competitors_unknown_ticker_returns_empty() -> None:
    g = _make_tech_graph()
    assert g.get_competitors("XOM") == []


def test_get_competitors_no_edges() -> None:
    g = FinancialGraph()
    g.add_company("AAPL", "Apple", "Tech", "HW")
    assert g.get_competitors("AAPL") == []


def test_get_sector_peers_returns_others_in_same_sector() -> None:
    g = _make_tech_graph()
    peers = g.get_sector_peers("AAPL")
    assert set(peers) == {"MSFT", "GOOGL"}


def test_get_sector_peers_excludes_self() -> None:
    g = _make_tech_graph()
    peers = g.get_sector_peers("MSFT")
    assert "MSFT" not in peers


def test_get_sector_peers_unknown_ticker() -> None:
    g = _make_tech_graph()
    assert g.get_sector_peers("UNKNOWN") == []


def test_get_macro_factors_via_sector() -> None:
    g = _make_tech_graph()
    factors = g.get_macro_factors("AAPL")
    assert set(factors) == {"FFR", "VIX"}


def test_get_macro_factors_unknown_ticker() -> None:
    g = _make_tech_graph()
    assert g.get_macro_factors("XOM") == []


def test_get_macro_factors_no_sensitive_to() -> None:
    g = FinancialGraph()
    g.add_sector("Technology")
    g.add_company("AAPL", "Apple", "Technology", "HW")
    g.add_belongs_to("AAPL", "Technology")
    assert g.get_macro_factors("AAPL") == []


# ---------------------------------------------------------------------------
# expand_query_tickers
# ---------------------------------------------------------------------------

def test_expand_query_tickers_max_hops_0() -> None:
    g = _make_tech_graph()
    result = g.expand_query_tickers("AAPL", max_hops=0)
    assert result == ["AAPL"]


def test_expand_query_tickers_1hop_includes_competitors() -> None:
    g = _make_tech_graph()
    result = g.expand_query_tickers("AAPL", max_hops=1)
    assert "AAPL" in result
    assert "MSFT" in result
    assert "GOOGL" in result


def test_expand_query_tickers_2hop_includes_sector_peers() -> None:
    g = _make_tech_graph()
    # GOOGL is a competitor of AAPL (1-hop) but also a sector peer (2-hop)
    # MSFT is a competitor of AAPL (1-hop) and sector peer (2-hop)
    result = g.expand_query_tickers("AAPL", max_hops=2)
    assert set(result) == {"AAPL", "MSFT", "GOOGL"}


def test_expand_query_tickers_is_sorted() -> None:
    g = _make_tech_graph()
    result = g.expand_query_tickers("AAPL", max_hops=2)
    assert result == sorted(result)


def test_expand_query_tickers_unknown_ticker_returns_singleton() -> None:
    g = _make_tech_graph()
    result = g.expand_query_tickers("XOM", max_hops=2)
    assert result == ["XOM"]


def test_expand_query_tickers_multi_sector() -> None:
    """Sector peers are sector-specific: Energy peer not included for Tech ticker."""
    g = FinancialGraph()
    g.add_sector("Technology")
    g.add_sector("Energy")
    g.add_company("AAPL", "Apple", "Technology", "HW")
    g.add_company("XOM", "Exxon", "Energy", "Oil")
    g.add_company("CVX", "Chevron", "Energy", "Oil")
    g.add_belongs_to("AAPL", "Technology")
    g.add_belongs_to("XOM", "Energy")
    g.add_belongs_to("CVX", "Energy")
    g.add_competes_with("XOM", "CVX")

    # AAPL has no competitors and no sector peers (only company in Technology)
    aapl_result = g.expand_query_tickers("AAPL", max_hops=2)
    assert aapl_result == ["AAPL"]

    # XOM expands to CVX (competitor + sector peer)
    xom_result = g.expand_query_tickers("XOM", max_hops=2)
    assert set(xom_result) == {"XOM", "CVX"}


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path: Path) -> None:
    g = _make_tech_graph()
    path = tmp_path / "graph.json"
    g.save(path)
    assert path.exists()

    loaded = FinancialGraph.load(path)
    assert loaded.node_count() == g.node_count()
    assert loaded.edge_count() == g.edge_count()


def test_save_load_preserves_competitors(tmp_path: Path) -> None:
    g = _make_tech_graph()
    path = tmp_path / "graph.json"
    g.save(path)
    loaded = FinancialGraph.load(path)
    assert set(loaded.get_competitors("AAPL")) == {"MSFT", "GOOGL"}


def test_save_load_preserves_macro_factors(tmp_path: Path) -> None:
    g = _make_tech_graph()
    path = tmp_path / "graph.json"
    g.save(path)
    loaded = FinancialGraph.load(path)
    assert set(loaded.get_macro_factors("AAPL")) == {"FFR", "VIX"}


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    g = _make_tech_graph()
    path = tmp_path / "subdir" / "nested" / "graph.json"
    g.save(path)
    assert path.exists()


def test_save_load_expand_query_tickers(tmp_path: Path) -> None:
    g = _make_tech_graph()
    path = tmp_path / "graph.json"
    g.save(path)
    loaded = FinancialGraph.load(path)
    result = loaded.expand_query_tickers("AAPL", max_hops=2)
    assert set(result) == {"AAPL", "MSFT", "GOOGL"}
