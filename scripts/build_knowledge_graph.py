#!/usr/bin/env python3
"""
Build and save the financial knowledge graph (P12-E1-T4, DJ-063).

Constructs the Phase 12 financial entity graph from curated seeds and
yfinance metadata (with fallback), then saves it to
data/knowledge_graph/financial_graph.json.

Usage:
    uv run python scripts/build_knowledge_graph.py
    HIFI_DATA_DIR=/path/to/data uv run python scripts/build_knowledge_graph.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Support running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hifi.knowledge.graph_construction import (
    DEFAULT_COMPETITORS,
    DEFAULT_MACRO_SENSITIVITY,
    build_financial_graph,
)

_DEFAULT_TICKERS = sorted(set(DEFAULT_COMPETITORS.keys()))


def main() -> None:
    data_dir = Path(os.environ.get("HIFI_DATA_DIR", "data"))
    output_path = data_dir / "knowledge_graph" / "financial_graph.json"

    print(f"Building financial knowledge graph ({len(_DEFAULT_TICKERS)} tickers)...")
    print(f"  Tickers: {', '.join(_DEFAULT_TICKERS)}")

    graph = build_financial_graph(
        tickers=_DEFAULT_TICKERS,
        competitor_seed=DEFAULT_COMPETITORS,
        macro_sensitivity=DEFAULT_MACRO_SENSITIVITY,
        data_dir=str(data_dir),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.save(output_path)

    print(f"\nGraph saved: {output_path}")
    print(f"  Nodes : {graph.node_count()}")
    print(f"  Edges : {graph.edge_count()}")

    # Sector coverage
    sectors_seen: set[str] = set()
    for ticker in _DEFAULT_TICKERS:
        factors = graph.get_macro_factors(ticker)
        if factors:
            sectors_seen.add(ticker)

    print(f"  Tickers with macro sensitivity: {len(sectors_seen)}/{len(_DEFAULT_TICKERS)}")


if __name__ == "__main__":
    main()
