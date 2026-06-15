"""
Financial knowledge graph store for HiFi Phase 12 GraphRAG (P12-E1-T1, DJ-062, DJ-063).

FinancialGraph wraps a NetworkX DiGraph with typed CRUD operations for three node types
(Company, Sector, MacroFactor) and three edge types (BELONGS_TO, COMPETES_WITH,
SENSITIVE_TO).  Serialisation uses NetworkX JSON format at
data/knowledge_graph/financial_graph.json.

Query expansion (expand_query_tickers) implements the DJ-064 2-hop BFS:
  1-hop: direct competitors (COMPETES_WITH edges)
  2-hop: sector peers (other companies in the same sector via BELONGS_TO)

The expanded ticker set is used as a filter for LanceDB dense search in GraphRetriever.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

# ---------------------------------------------------------------------------
# Node type constants
# ---------------------------------------------------------------------------

_NODE_COMPANY = "company"
_NODE_SECTOR = "sector"
_NODE_MACRO = "macro_factor"

# Edge type constants
_EDGE_BELONGS_TO = "BELONGS_TO"
_EDGE_COMPETES_WITH = "COMPETES_WITH"
_EDGE_SENSITIVE_TO = "SENSITIVE_TO"


class FinancialGraph:
    """
    NetworkX-backed financial entity graph for GraphRAG query expansion.

    Nodes
    -----
    Company     : ticker (str), name, sector, industry
    Sector      : name (str)
    MacroFactor : name (str), series_id (FRED series ID)

    Edges
    -----
    BELONGS_TO    : Company -> Sector      (directed)
    COMPETES_WITH : Company <-> Company    (stored as two directed edges)
    SENSITIVE_TO  : Sector -> MacroFactor  (directed)

    Serialisation
    -------------
    save() / load() use NetworkX JSON (node_link_data / node_link_graph) so the
    graph can be stored and loaded without loss of node/edge attributes.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node writers
    # ------------------------------------------------------------------

    def add_company(
        self,
        ticker: str,
        name: str,
        sector: str,
        industry: str,
    ) -> None:
        """Add or update a Company node."""
        self._g.add_node(
            ticker,
            node_type=_NODE_COMPANY,
            ticker=ticker,
            name=name,
            sector=sector,
            industry=industry,
        )

    def add_sector(self, name: str) -> None:
        """Add or update a Sector node."""
        self._g.add_node(name, node_type=_NODE_SECTOR, name=name)

    def add_macro_factor(self, name: str, series_id: str) -> None:
        """Add or update a MacroFactor node."""
        self._g.add_node(
            name,
            node_type=_NODE_MACRO,
            name=name,
            series_id=series_id,
        )

    # ------------------------------------------------------------------
    # Edge writers
    # ------------------------------------------------------------------

    def add_competes_with(self, ticker_a: str, ticker_b: str) -> None:
        """Add a symmetric COMPETES_WITH edge (stored as two directed edges)."""
        self._g.add_edge(ticker_a, ticker_b, edge_type=_EDGE_COMPETES_WITH)
        self._g.add_edge(ticker_b, ticker_a, edge_type=_EDGE_COMPETES_WITH)

    def add_belongs_to(self, ticker: str, sector: str) -> None:
        """Add a BELONGS_TO edge from a Company node to a Sector node."""
        self._g.add_edge(ticker, sector, edge_type=_EDGE_BELONGS_TO)

    def add_sensitive_to(self, sector: str, macro_factor: str) -> None:
        """Add a SENSITIVE_TO edge from a Sector node to a MacroFactor node."""
        self._g.add_edge(sector, macro_factor, edge_type=_EDGE_SENSITIVE_TO)

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def get_competitors(self, ticker: str) -> list[str]:
        """Return direct competitors of ticker (COMPETES_WITH 1-hop)."""
        if ticker not in self._g:
            return []
        return [
            nbr
            for nbr in self._g.successors(ticker)
            if self._g.edges[ticker, nbr].get("edge_type") == _EDGE_COMPETES_WITH
        ]

    def get_sector_peers(self, ticker: str) -> list[str]:
        """
        Return other Company nodes in the same sector as ticker.

        Sector is determined by the BELONGS_TO edge from ticker.
        """
        sector = self._sector_of(ticker)
        if sector is None:
            return []
        return [
            n
            for n, data in self._g.nodes(data=True)
            if (
                data.get("node_type") == _NODE_COMPANY
                and n != ticker
                and self._g.has_edge(n, sector)
                and self._g.edges[n, sector].get("edge_type") == _EDGE_BELONGS_TO
            )
        ]

    def get_macro_factors(self, ticker: str) -> list[str]:
        """
        Return MacroFactor names sensitive to ticker's sector.

        Follows ticker -> sector (BELONGS_TO) -> macro (SENSITIVE_TO) path.
        """
        sector = self._sector_of(ticker)
        if sector is None:
            return []
        return [
            nbr
            for nbr in self._g.successors(sector)
            if self._g.edges[sector, nbr].get("edge_type") == _EDGE_SENSITIVE_TO
        ]

    def expand_query_tickers(self, ticker: str, max_hops: int = 2) -> list[str]:
        """
        Expand a query ticker to its related Company nodes.

        Returns a sorted list of ticker symbols (Company nodes only):
          - ticker itself (always included)
          - 1-hop (max_hops >= 1): direct competitors via COMPETES_WITH
          - 2-hop (max_hops >= 2): sector peers via BELONGS_TO -> same sector

        Ticker is always included even if not in the graph (graceful fallback).

        Parameters
        ----------
        ticker : str
            The primary ticker to expand.
        max_hops : int
            Expansion depth. 0 = ticker only. 1 = + competitors. 2 = + peers.

        Returns
        -------
        list[str]
            Sorted list of ticker symbols.
        """
        result: set[str] = {ticker}

        if ticker not in self._g:
            return sorted(result)

        if max_hops >= 1:
            result |= set(self.get_competitors(ticker))

        if max_hops >= 2:
            result |= set(self.get_sector_peers(ticker))

        return sorted(result)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise graph to JSON at path. Parent directories are created."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._g)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> FinancialGraph:
        """Load a FinancialGraph from the JSON file at path."""
        data = json.loads(path.read_text())
        fg = cls()
        fg._g = nx.node_link_graph(data, directed=True, multigraph=False)
        return fg

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        """Total number of nodes (all types)."""
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        """Total number of directed edges."""
        return self._g.number_of_edges()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sector_of(self, ticker: str) -> str | None:
        """Return the Sector node connected to ticker via BELONGS_TO, or None."""
        if ticker not in self._g:
            return None
        for nbr in self._g.successors(ticker):
            if self._g.edges[ticker, nbr].get("edge_type") == _EDGE_BELONGS_TO:
                return nbr
        return None
