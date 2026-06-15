"""
Financial knowledge graph construction for HiFi Phase 12 GraphRAG (P12-E1-T2, DJ-063).

build_financial_graph() constructs the financial entity graph deterministically from:
  1. yfinance metadata (sector, industry, name) -- live fetch or ticker_metadata override
  2. Curated competitor seed (DEFAULT_COMPETITORS)
  3. Domain-knowledge macro sensitivity map (DEFAULT_MACRO_SENSITIVITY)

The graph is tight by design (DJ-063): 3 evaluation tickers + sector peers from the
Phase 10 15-ticker universe, 3 macro factors, ~40 edges total.

Default seeds encode Phase 12 scope:
  - Technology: AAPL, MSFT, GOOGL, NVDA, AMZN, META
  - Financial Services: JPM, BAC, GS
  - Energy: XOM, CVX

MacroFactor FRED series IDs:
  - VIX  -> VIXCLS
  - FFR  -> FEDFUNDS
  - CPI  -> CPIAUCSL
"""

from __future__ import annotations

import logging
from typing import Any

from hifi.knowledge.graph_store import FinancialGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default curated seeds (DJ-063)
# ---------------------------------------------------------------------------

DEFAULT_COMPETITORS: dict[str, list[str]] = {
    "AAPL": ["MSFT", "GOOGL"],
    "MSFT": ["AAPL", "GOOGL"],
    "GOOGL": ["AAPL", "MSFT"],
    "NVDA": ["AMZN"],
    "AMZN": ["NVDA"],
    "META": ["GOOGL"],
    "JPM": ["BAC", "GS"],
    "BAC": ["JPM", "GS"],
    "GS": ["JPM", "BAC"],
    "XOM": ["CVX"],
    "CVX": ["XOM"],
}

DEFAULT_MACRO_SENSITIVITY: dict[str, list[str]] = {
    "Technology": ["FFR", "VIX"],
    "Financial Services": ["FFR"],
    "Energy": ["VIX", "CPI"],
}

# FRED series IDs for supported macro factors
_MACRO_FACTOR_SERIES: dict[str, str] = {
    "VIX": "VIXCLS",
    "FFR": "FEDFUNDS",
    "CPI": "CPIAUCSL",
}

# Fallback metadata for known tickers when yfinance is unavailable
_TICKER_FALLBACK: dict[str, dict[str, str]] = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "industry": "Software"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "industry": "Internet Content"},
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology", "industry": "Semiconductors"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Technology", "industry": "Internet Retail"},
    "META": {"name": "Meta Platforms Inc.", "sector": "Technology", "industry": "Internet Content"},
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial Services", "industry": "Banks"},
    "BAC": {"name": "Bank of America Corp.", "sector": "Financial Services", "industry": "Banks"},
    "GS": {
        "name": "The Goldman Sachs Group Inc.",
        "sector": "Financial Services",
        "industry": "Investment Banking",
    },
    "XOM": {"name": "Exxon Mobil Corporation", "sector": "Energy", "industry": "Oil & Gas Integrated"},  # noqa: E501
    "CVX": {"name": "Chevron Corporation", "sector": "Energy", "industry": "Oil & Gas Integrated"},
    # Phase 10 additional tickers
    "CAT": {
        "name": "Caterpillar Inc.",
        "sector": "Industrials",
        "industry": "Farm & Heavy Construction Machinery",
    },
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "industry": "Drug Manufacturers"},
    "NEE": {
        "name": "NextEra Energy Inc.",
        "sector": "Utilities",
        "industry": "Utilities-Regulated Electric",
    },
    "UNH": {"name": "UnitedHealth Group Inc.", "sector": "Healthcare", "industry": "Healthcare Plans"},  # noqa: E501
    "WMT": {"name": "Walmart Inc.", "sector": "Consumer Defensive", "industry": "Discount Stores"},
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "sector": "Financial Services",
        "industry": "Exchange Traded Fund",
    },
}


def _fetch_ticker_metadata(ticker: str) -> dict[str, str]:
    """
    Fetch ticker metadata from yfinance.

    Returns dict with keys: name, sector, industry.
    Falls back to _TICKER_FALLBACK when yfinance is unavailable or returns incomplete data.
    """
    try:
        import yfinance as yf
        info: dict[str, Any] = yf.Ticker(ticker).info
        name = str(info.get("longName") or info.get("shortName") or ticker)
        sector = str(info.get("sector") or "Unknown")
        industry = str(info.get("industry") or "Unknown")
        if sector != "Unknown" and industry != "Unknown":
            return {"name": name, "sector": sector, "industry": industry}
    except Exception as exc:
        logger.debug("yfinance fetch failed for %s: %s", ticker, exc)

    if ticker in _TICKER_FALLBACK:
        logger.debug("Using fallback metadata for %s", ticker)
        return _TICKER_FALLBACK[ticker]

    return {"name": ticker, "sector": "Unknown", "industry": "Unknown"}


def build_financial_graph(
    tickers: list[str],
    competitor_seed: dict[str, list[str]],
    macro_sensitivity: dict[str, list[str]],
    data_dir: str | None = None,
    ticker_metadata: dict[str, dict[str, str]] | None = None,
) -> FinancialGraph:
    """
    Build the financial knowledge graph from yfinance metadata and curated seeds.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to include as Company nodes.
    competitor_seed : dict[str, list[str]]
        Mapping ticker -> list of competitor tickers. Edges added for any pair
        where at least one side appears in `tickers`.
    macro_sensitivity : dict[str, list[str]]
        Mapping sector_name -> list of MacroFactor names. Only sectors that
        appear in the graph (from ticker metadata) receive SENSITIVE_TO edges.
    data_dir : str | None
        Unused in Phase 12 (metadata comes from yfinance or ticker_metadata).
        Reserved for future Parquet-cached metadata reads.
    ticker_metadata : dict[str, dict[str, str]] | None
        Optional override map: ticker -> {name, sector, industry}.
        When provided, yfinance is not called for those tickers.
        Intended for deterministic tests that cannot call the network.

    Returns
    -------
    FinancialGraph
        Populated graph with Company, Sector, MacroFactor nodes and
        BELONGS_TO, COMPETES_WITH, SENSITIVE_TO edges.
    """
    g = FinancialGraph()
    override = ticker_metadata or {}

    # 1. Add MacroFactor nodes for all supported factors
    for macro_name, series_id in _MACRO_FACTOR_SERIES.items():
        g.add_macro_factor(macro_name, series_id)

    # 2. Add Company and Sector nodes + BELONGS_TO edges
    sectors_in_graph: set[str] = set()
    ticker_set = set(tickers)

    for ticker in tickers:
        meta = override[ticker] if ticker in override else _fetch_ticker_metadata(ticker)

        name = meta.get("name", ticker)
        sector = meta.get("sector", "Unknown")
        industry = meta.get("industry", "Unknown")

        g.add_company(ticker, name=name, sector=sector, industry=industry)

        if sector not in sectors_in_graph:
            g.add_sector(sector)
            sectors_in_graph.add(sector)

        g.add_belongs_to(ticker, sector)
        logger.debug("Added Company %s -> Sector %s", ticker, sector)

    # 3. Add COMPETES_WITH edges (bidirectional, curated seed)
    #    Include an edge when EITHER endpoint appears in the requested ticker set
    seen_pairs: set[frozenset[str]] = set()
    for ticker_a, competitors in competitor_seed.items():
        for ticker_b in competitors:
            pair = frozenset((ticker_a, ticker_b))
            if pair in seen_pairs:
                continue
            if ticker_a in ticker_set or ticker_b in ticker_set:
                g.add_competes_with(ticker_a, ticker_b)
                seen_pairs.add(pair)
                logger.debug("Added COMPETES_WITH: %s <-> %s", ticker_a, ticker_b)

    # 4. Add SENSITIVE_TO edges for sectors that exist in the graph
    for sector, macro_factors in macro_sensitivity.items():
        if sector not in sectors_in_graph:
            continue
        for macro_name in macro_factors:
            if macro_name in _MACRO_FACTOR_SERIES:
                g.add_sensitive_to(sector, macro_name)
                logger.debug("Added SENSITIVE_TO: %s -> %s", sector, macro_name)

    logger.info(
        "Financial graph built: %d nodes, %d edges, %d sectors",
        g.node_count(),
        g.edge_count(),
        len(sectors_in_graph),
    )
    return g
