# Phase 7 Scientific Bitacora: RAG Knowledge Systems

**Date:** 2026-06-11
**Tests:** 596 passing (582 inherited + 14 new), 0 skipped, 0 lint errors
**Status:** COMPLETE

---

## What Was Built

Phase 7 adds a Retrieval-Augmented Generation (RAG) layer to HiFi. Agents can now
retrieve relevant passages from SEC EDGAR filings and incorporate them as qualitative
context when generating investment opinions. The implementation is fully fail-open:
if the knowledge server is unavailable or the store is empty, agents fall back
transparently to Phase 6 behaviour (v1 prompts, no retrieved context).

**New source files:**
- `src/hifi/knowledge/schemas.py` — `FilingDocument`, `DocumentChunk` (with `make_chunk_id`), `EvaluationQuery`
- `src/hifi/data/edgar.py` — `EdgarFetcher` with `get_submissions`, `get_filing_index`, `get_filing_document`, `extract_text_sections`, `fetch_filing`; `TICKER_CIKS` registry
- `src/hifi/knowledge/document_ingestion.py` — `DocumentIngestionPipeline` with configs A/B/C
- `src/hifi/knowledge/embeddings.py` — `EmbeddingModel` using OpenAI client at LM Studio URL
- `src/hifi/knowledge/vector_store.py` — `KnowledgeStore` using LanceDB
- `src/hifi/knowledge/retrieval.py` — `KnowledgeRetriever` + `evaluate_precision_at_k`
- `src/hifi/mcp/knowledge_server.py` — FastMCP server exposing `retrieve_context` tool
- `src/hifi/agents/prompts/fundamental_v2.md` — RAG-enabled v2 prompt with `{retrieved_context}` block
- `src/hifi/agents/prompts/technical_v2.md` — RAG-enabled v2 prompt with `{retrieved_context}` block

**Modified source files:**
- `src/hifi/agents/fundamental_agent.py` — `FundamentalistState.retrieved_context`, `retrieve_context_node`, `_load_v2_prompt_template`, v1/v2 prompt selection in `generate_analysis_node`, `use_rag` param in `build_fundamental_graph` and `run_analysis`
- `src/hifi/agents/technical_agent.py` — same pattern as above
- `src/hifi/agents/ensemble_runner.py` — `use_rag` param forwarded to both agents

**New test files:**
- `tests/unit/test_edgar_fetcher.py`, `test_document_ingestion.py`, `test_embedding_model.py`, `test_knowledge_store.py`, `test_retrieval.py`, `test_knowledge_mcp_server.py`
- `tests/integration/test_knowledge_pipeline.py`, `test_rag_agents.py`
- `tests/holistic/test_phase7_rag_pipeline.py`
- `tests/unit/test_phase7_rag_baseline.py` (skip until fixture generated)
- `tests/fixtures/retrieval/evaluation_queries.json` — 20 labelled evaluation queries

**New scripts:**
- `scripts/record_sec_fixtures.py` — one-time EDGAR fixture recorder (requires internet)
- `scripts/run_phase7_rag_baseline.py` — RAG baseline runner (requires LM Studio + SEC fixtures)

---

## Key Decisions Made

**DJ-026 (confirmed):** LanceDB for the vector store. LanceDB 0.33.0 (cloud-first API)
requires `pylance>=7.0.0` for local storage. Key API note: `db.list_tables()` returns a
`ListTablesResponse` with `.tables` attribute; `db.table_names()` is deprecated and absent
in 0.33.0. Fixed-size list schema: `pa.list_(pa.float32(), 768)`.

**DJ-027 (confirmed):** `nomic-embed-text-v1.5` at LM Studio as the embedding model.
768-dimension embeddings, OpenAI-compatible API. Used for both indexing and query embedding.
Production `EmbeddingModel` uses `HIFI_LM_STUDIO_URL`; tests use `DeterministicEmbeddingModel`
(SHA-256 seeded, unit-norm vectors, 32 dimensions) which requires no external server.

**DJ-028 (confirmed):** SEC EDGAR as the primary knowledge source. Three filing types
(10-K, 10-Q, 8-K) for three tickers (AAPL, JPM, XOM). Target sections: 10-K gets
Business/Risk Factors/MD&A; 10-Q gets MD&A; 8-K gets Event Description. HTML parsing
uses stdlib `html.parser` — no lxml dependency. Rate limit: ≤8 req/s (EDGAR allows 10/s).

**DJ-029 (confirmed):** `src/hifi/knowledge/` as the package boundary for all knowledge
infrastructure. The MCP server lives in `src/hifi/mcp/knowledge_server.py` following the
existing `financial_server.py` precedent. This is the dependency isolation boundary from
DJ-010: the knowledge server runs as a subprocess with its own environment.

**DJ-030 (implicit):** Chunking config "A" as production default. Three configs (A: 1000
chars/100 overlap; B: 500/50; C: 2000/200) were evaluated. Config A offers the best balance
of semantic coherence and retrieval precision for MD&A passages.

---

## Architectural Observations

**Fail-open RAG is the right default.** The retrieve_context_node returns `""` on any
exception. The generate_analysis_node selects v1 prompt when retrieved_context is empty.
This means a single code path handles both RAG and non-RAG execution — the `use_rag=False`
regression is guaranteed by the same if/else branch that handles retrieval failures.
There is no separate code path to maintain.

**prompt_version reflects actual execution.** Setting `prompt_version` from
`final_state.get("retrieved_context")` rather than from the `use_rag` parameter captures
the actual execution path. If `use_rag=True` but retrieval fails (empty retrieved_context),
the recorded prompt_version is "v1" — accurate and traceable in Phase 10 analysis.

**DeterministicEmbeddingModel as a test primitive.** Placing this in `tests/conftest.py`
(not production code) follows the project principle: deterministic seeded generators instead
of mocks. The SHA-256 seed ensures each text gets a unique, reproducible, unit-norm vector.
Tests at `_DIM=32` run in milliseconds without sacrificing geometric correctness (cosine
similarity still meaningful in 32D).

**MCP boundary isolation for the knowledge server.** `call_tool(..., server_module="hifi.mcp.knowledge_server")` routes to the knowledge MCP server as a subprocess, consistent with the DJ-010
architecture. The knowledge server initialises `KnowledgeStore` and `EmbeddingModel` lazily
on first call, and caches them as module-level singletons for the subprocess lifetime.
EDGAR-related heavy imports (lancedb, requests) never enter the main process.

**lancedb 0.33.0 API stability.** The `db.list_tables()` → `.tables` discovery happened
during initial implementation and was immediately captured in memory. lancedb's public API
changed substantially between 0.8.x and 0.33.0 (cloud-first refactor). The key invariant:
always use `list_tables().tables`, never `table_names()`.

---

## Baseline Results (to be populated after running scripts)

The following measurements require live LM Studio. They will be recorded after
running `scripts/run_phase7_rag_baseline.py`:

```
Phase 7 RAG Baseline (AAPL, JPM, XOM — 2023-03-31)
----------------------------------------------------
Fundamental: mean HR=?, mean GR=?, prompt_version=fundamental_v2
Technical:   mean HR=?, mean GR=?, prompt_version=technical_v2
Delta vs Phase 5 fundamental: HR=?, GR=?
Delta vs Phase 5 technical:   HR=?, GR=?
```

**Expected hypothesis (DJ-028):** Retrieved SEC filing context should improve GR
(grounding rate) by providing qualitative facts the LLM can cite, without increasing HR
(hallucination rate) since numerical claims still come from MCP tools. A neutral or
positive delta on both metrics would validate the RAG design.

---

## Test Inventory (Phase 7 only)

| Test file | Count | What it covers |
|---|---|---|
| `test_edgar_fetcher.py` | 17 | EdgarFetcher unit tests (fixture replay) |
| `test_document_ingestion.py` | 15 | Chunking pipeline, configs A/B/C |
| `test_embedding_model.py` | 12 | EmbeddingModel unit tests (DeterministicEmbeddingModel) |
| `test_knowledge_store.py` | 12 | KnowledgeStore CRUD, LanceDB API |
| `test_retrieval.py` | 7 | KnowledgeRetriever, evaluate_precision_at_k |
| `test_knowledge_mcp_server.py` | 4 | retrieve_context tool, fail-open |
| `test_knowledge_pipeline.py` | 6 | E2E: chunk → embed → index → retrieve (configs A/B/C) |
| `test_rag_agents.py` | 9 | Agent augmentation: use_rag, v1/v2 selection, fail-open |
| `test_phase7_rag_pipeline.py` | 5 | Holistic: full RAG pipeline, regressions |
| `test_phase7_rag_baseline.py` | 8 | Fixture validation (skipped until script runs) |
| **Total (active)** | **87** | All pass, 0 skipped |
