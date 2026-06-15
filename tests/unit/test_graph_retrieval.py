"""
Unit tests for GraphRetriever (P12-E1-T3).

No LanceDB, no embedding model, no live services.
All dependencies are replaced with lightweight mocks.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from hifi.knowledge.graph_retrieval import GraphRetriever
from hifi.knowledge.graph_store import FinancialGraph
from hifi.knowledge.schemas import DocumentChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tech_graph() -> FinancialGraph:
    g = FinancialGraph()
    g.add_sector("Technology")
    g.add_macro_factor("FFR", "FEDFUNDS")
    g.add_company("AAPL", "Apple Inc.", "Technology", "Consumer Electronics")
    g.add_company("MSFT", "Microsoft Corporation", "Technology", "Software")
    g.add_company("GOOGL", "Alphabet Inc.", "Technology", "Internet Content")
    g.add_belongs_to("AAPL", "Technology")
    g.add_belongs_to("MSFT", "Technology")
    g.add_belongs_to("GOOGL", "Technology")
    g.add_competes_with("AAPL", "MSFT")
    g.add_sensitive_to("Technology", "FFR")
    return g


def _make_chunk(ticker: str, text: str = "Sample text.") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{ticker}-001",
        ticker=ticker,
        filing_type="10-K",
        period=date(2023, 3, 31),
        section="MD&A",
        chunk_index=0,
        text=text,
        char_count=len(text),
        approx_tokens=max(1, len(text) // 4),
        chunking_config="A",
    )


def _make_retriever(
    graph: FinancialGraph | None = None,
    chunks: list[DocumentChunk] | None = None,
    embed_result: list[float] | None = None,
) -> tuple[GraphRetriever, MagicMock, MagicMock]:
    """Return (retriever, mock_store, mock_model) configured with given parameters."""
    g = graph or _make_tech_graph()

    store = MagicMock()
    store.search_tickers.return_value = chunks or []

    model = MagicMock()
    model.embed_one.return_value = embed_result or [0.1] * 768

    return GraphRetriever(store=store, embedding_model=model, graph=g), store, model


# ---------------------------------------------------------------------------
# retrieve() — success paths
# ---------------------------------------------------------------------------


def test_retrieve_calls_embed_one():
    retriever, store, model = _make_retriever()
    retriever.retrieve("financial analysis AAPL", ticker="AAPL")
    model.embed_one.assert_called_once_with("financial analysis AAPL")


def test_retrieve_calls_search_tickers():
    retriever, store, _ = _make_retriever()
    retriever.retrieve("query", ticker="AAPL", top_k=3)
    store.search_tickers.assert_called_once()


def test_retrieve_passes_top_k():
    retriever, store, _ = _make_retriever()
    retriever.retrieve("query", ticker="AAPL", top_k=7)
    call_kwargs = store.search_tickers.call_args
    assert call_kwargs.kwargs.get("top_k") == 7 or call_kwargs.args[2] == 7


def test_retrieve_includes_primary_ticker():
    retriever, store, _ = _make_retriever()
    retriever.retrieve("query", ticker="AAPL")
    tickers = (
        store.search_tickers.call_args.kwargs.get("tickers")
        or store.search_tickers.call_args.args[1]
    )
    assert "AAPL" in tickers


def test_retrieve_expands_to_competitor():
    """AAPL competes with MSFT — MSFT must be in expanded set."""
    retriever, store, _ = _make_retriever()
    retriever.retrieve("query", ticker="AAPL")
    tickers = (
        store.search_tickers.call_args.kwargs.get("tickers")
        or store.search_tickers.call_args.args[1]
    )
    assert "MSFT" in tickers


def test_retrieve_expands_to_sector_peers():
    """AAPL and GOOGL are in the same sector — GOOGL should appear via 2-hop."""
    retriever, store, _ = _make_retriever()
    retriever.retrieve("query", ticker="AAPL")
    tickers = (
        store.search_tickers.call_args.kwargs.get("tickers")
        or store.search_tickers.call_args.args[1]
    )
    assert "GOOGL" in tickers


def test_retrieve_returns_chunks_from_store():
    expected = [_make_chunk("AAPL"), _make_chunk("MSFT")]
    retriever, _, _ = _make_retriever(chunks=expected)
    result = retriever.retrieve("query", ticker="AAPL")
    assert result == expected


def test_retrieve_returns_empty_list_when_store_empty():
    retriever, _, _ = _make_retriever(chunks=[])
    result = retriever.retrieve("query", ticker="AAPL")
    assert result == []


# ---------------------------------------------------------------------------
# retrieve() — unknown ticker fallback
# ---------------------------------------------------------------------------


def test_retrieve_unknown_ticker_passes_singleton():
    """Ticker not in graph → expanded set is just [ticker]."""
    retriever, store, _ = _make_retriever()
    retriever.retrieve("query", ticker="XOM")
    tickers = (
        store.search_tickers.call_args.kwargs.get("tickers")
        or store.search_tickers.call_args.args[1]
    )
    assert tickers == ["XOM"]


def test_retrieve_unknown_ticker_returns_list():
    retriever, _, _ = _make_retriever()
    result = retriever.retrieve("query", ticker="UNKNOWN_TICKER")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# retrieve() — embedding failure
# ---------------------------------------------------------------------------


def test_retrieve_embedding_failure_returns_empty():
    graph = _make_tech_graph()
    store = MagicMock()
    model = MagicMock()
    model.embed_one.side_effect = RuntimeError("model unavailable")

    retriever = GraphRetriever(store=store, embedding_model=model, graph=graph)
    result = retriever.retrieve("query", ticker="AAPL")

    assert result == []
    store.search_tickers.assert_not_called()


def test_retrieve_embedding_failure_does_not_raise():
    graph = _make_tech_graph()
    model = MagicMock()
    model.embed_one.side_effect = ConnectionError("no server")
    retriever = GraphRetriever(store=MagicMock(), embedding_model=model, graph=graph)

    result = retriever.retrieve("query", ticker="AAPL")  # must not raise
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# format_context()
# ---------------------------------------------------------------------------


def test_format_context_empty_input_returns_empty_string():
    retriever, _, _ = _make_retriever()
    assert retriever.format_context([]) == ""


def test_format_context_single_chunk_contains_number():
    retriever, _, _ = _make_retriever()
    chunk = _make_chunk("AAPL", "Revenue up 5%.")
    result = retriever.format_context([chunk])
    assert "[1]" in result


def test_format_context_single_chunk_contains_ticker():
    retriever, _, _ = _make_retriever()
    chunk = _make_chunk("AAPL")
    result = retriever.format_context([chunk])
    assert "AAPL" in result


def test_format_context_single_chunk_contains_text():
    retriever, _, _ = _make_retriever()
    chunk = _make_chunk("AAPL", "Operating margin expanded.")
    result = retriever.format_context([chunk])
    assert "Operating margin expanded." in result


def test_format_context_multiple_chunks_numbered():
    retriever, _, _ = _make_retriever()
    chunks = [_make_chunk("AAPL", "Chunk A."), _make_chunk("MSFT", "Chunk B.")]
    result = retriever.format_context(chunks)
    assert "[1]" in result
    assert "[2]" in result


def test_format_context_multiple_chunks_separator():
    retriever, _, _ = _make_retriever()
    chunks = [_make_chunk("AAPL"), _make_chunk("MSFT")]
    result = retriever.format_context(chunks)
    assert "---" in result


def test_format_context_no_trailing_separator():
    """Last chunk should not be followed by a separator."""
    retriever, _, _ = _make_retriever()
    chunks = [_make_chunk("AAPL"), _make_chunk("MSFT")]
    result = retriever.format_context(chunks)
    lines = result.strip().splitlines()
    # The last line should not be a separator
    assert "---" not in lines[-1]


def test_format_context_contains_filing_type():
    retriever, _, _ = _make_retriever()
    chunk = _make_chunk("AAPL")
    result = retriever.format_context([chunk])
    assert "10-K" in result
