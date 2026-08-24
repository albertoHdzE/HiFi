"""Tests for the sentiment agent's EDGAR MD&A fallback (DJ-120).

The sentiment agent retrieved from the knowledge server's ``chunks_a`` table,
which was populated for three tickers. On the other 95 it took its
"Insufficient Data" path and returned Hold at confidence 0.0 — on 97% of passes
across Phase 15 and the Phase 16 live run. A constant member contributes no
information to an ensemble and, worse, silently deflates the disagreement
metrics that are this experiment's dependent variable.
"""

from __future__ import annotations

import pytest

from hifi.knowledge.edgar_retriever import _rank_chunks_by_query

_QUERY = "management outlook guidance forward-looking statements risks revenue growth"


class TestRankChunksByQuery:
    def test_ranks_relevant_chunks_first(self):
        chunks = [
            "Boilerplate about the table of contents and exhibit index.",
            "Management outlook: guidance for revenue growth remains strong.",
            "Numeric tables of property and equipment depreciation schedules.",
        ]
        got = _rank_chunks_by_query(chunks, _QUERY, max_chunks=1)
        assert got == [chunks[1]]

    def test_respects_max_chunks(self):
        chunks = [f"management guidance chunk {i}" for i in range(10)]
        assert len(_rank_chunks_by_query(chunks, _QUERY, max_chunks=4)) == 4

    def test_empty_query_falls_back_to_head(self):
        chunks = ["a", "b", "c"]
        assert _rank_chunks_by_query(chunks, "", max_chunks=2) == ["a", "b"]

    def test_short_terms_ignored_so_query_is_not_all_stopwords(self):
        """Terms of 3 chars or fewer are dropped; a query of only such terms
        must degrade to head selection rather than scoring everything zero
        in an arbitrary order."""
        chunks = ["a", "b", "c"]
        assert _rank_chunks_by_query(chunks, "of to the a", max_chunks=2) == ["a", "b"]

    def test_deterministic_on_ties(self):
        chunks = ["no match one", "no match two", "no match three"]
        first = _rank_chunks_by_query(chunks, _QUERY, max_chunks=2)
        assert first == _rank_chunks_by_query(chunks, _QUERY, max_chunks=2)
        # Ties preserve document order.
        assert first == chunks[:2]

    def test_selection_differs_from_head_slice(self):
        """The point of query scoring: sentiment must not receive the same
        bytes as the fundamental agent, which takes the head of the MD&A.
        Identical evidence would correlate two nominally independent members."""
        chunks = [
            "Item 2 preamble and forward reference boilerplate.",
            "Depreciation tables.",
            "Management outlook and guidance on revenue growth and margin risks.",
        ]
        head = chunks[:1]
        scored = _rank_chunks_by_query(chunks, _QUERY, max_chunks=1)
        assert scored != head


class TestSentimentRetrievalFallback:
    def test_falls_back_to_edgar_when_vector_store_empty(self, monkeypatch):
        """The regression: an empty chunks_a must not end retrieval."""
        import hifi.agents.sentiment_agent as sa

        monkeypatch.setattr(
            sa, "call_tool", lambda *a, **k: {"passages": []}
        )
        monkeypatch.setattr(
            "hifi.knowledge.edgar_retriever.retrieve_mda_context",
            lambda **kw: f"[EDGAR MD&A — {kw['ticker']}]\nmanagement outlook",
        )
        out = sa._retrieve_context("ACN", "2026-08-13", "data")
        assert "EDGAR MD&A — ACN" in out

    def test_falls_back_when_vector_store_raises(self, monkeypatch):
        import hifi.agents.sentiment_agent as sa

        def boom(*a, **k):
            raise RuntimeError("knowledge server down")

        monkeypatch.setattr(sa, "call_tool", boom)
        monkeypatch.setattr(
            "hifi.knowledge.edgar_retriever.retrieve_mda_context",
            lambda **kw: "[EDGAR MD&A — T]\ntext",
        )
        assert "EDGAR MD&A" in sa._retrieve_context("T", "2026-08-13", "data")

    def test_prefers_vector_store_when_populated(self, monkeypatch):
        import hifi.agents.sentiment_agent as sa

        monkeypatch.setattr(sa, "call_tool", lambda *a, **k: {
            "passages": [{
                "rank": 1, "filing_type": "10-K", "section": "mda",
                "period": "2026-05-31", "text": "vector hit",
            }],
        })
        monkeypatch.setattr(
            "hifi.knowledge.edgar_retriever.retrieve_mda_context",
            lambda **kw: "SHOULD NOT BE USED",
        )
        out = sa._retrieve_context("AAPL", "2026-08-13", "data")
        assert "vector hit" in out
        assert "SHOULD NOT BE USED" not in out

    def test_returns_empty_when_both_sources_fail(self, monkeypatch):
        """Fail-open is still the contract — the agent degrades to its default
        signal rather than crashing the nightly pipeline."""
        import hifi.agents.sentiment_agent as sa

        monkeypatch.setattr(sa, "call_tool", lambda *a, **k: {"passages": []})
        monkeypatch.setattr(
            "hifi.knowledge.edgar_retriever.retrieve_mda_context",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("lancedb missing")),
        )
        assert sa._retrieve_context("ACN", "2026-08-13", "data") == ""


@pytest.mark.parametrize("ticker", ["ACN", "T", "NVDA"])
def test_real_corpus_covers_previously_dark_tickers(ticker):
    """Integration check against the real EDGAR table: the tickers that were
    dark for the entire live run must now return substantive context."""
    lancedb = pytest.importorskip("lancedb")
    from pathlib import Path

    from hifi.knowledge.edgar_retriever import retrieve_mda_context

    # Resolve from the repo root and pass db_path explicitly. Relying on the
    # cwd or on HIFI_DATA_DIR makes this test order-dependent: other tests in
    # the suite legitimately point that variable at a tmp_path.
    repo = Path(__file__).resolve().parents[2]
    corpus = repo / "data" / "knowledge.lance"
    if not corpus.exists():
        pytest.skip("EDGAR corpus not present in this checkout")
    db = lancedb.connect(str(corpus))
    tables = db.list_tables()
    names = list(tables.tables) if hasattr(tables, "tables") else list(tables)
    if "hifi-dev-sec-sec-mda" not in names:
        pytest.skip("EDGAR MD&A table not ingested")

    out = retrieve_mda_context(ticker, "2026-08-13", db_path=str(corpus), query=_QUERY)
    assert len(out) > 1000
    assert ticker in out
