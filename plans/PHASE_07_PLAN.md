# Phase 7: Knowledge Systems — RAG

**Status:** PLANNED

| Epic | Title | Status |
|---|---|---|
| P7-E1 | EDGAR data acquisition + fixture recording | PLANNED |
| P7-E2 | Document ingestion and chunking (3 configs) | PLANNED |
| P7-E3 | Embedding pipeline (nomic-embed-text-v1.5 via LM Studio) | PLANNED |
| P7-E4 | LanceDB vector store | PLANNED |
| P7-E5 | Retrieval pipeline + chunking evaluation (Precision@5) | PLANNED |
| P7-E6 | Knowledge MCP server | PLANNED |
| P7-E7 | Agent augmentation (RAG-enabled v2 agents) | PLANNED |
| P7-E8 | RAG baseline measurement + bitacora | PLANNED |

**David Sections:** §11.2 RAG, §7.3 Knowledge Layer (partial)
**Learning Guide Topics:** 1.1 Chunking Strategies, 1.2 Embedding Models, 1.3 Vector Databases, 1.4 Retrieval Strategies
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 7
**Open Questions Resolved:** OQ-K01 (chunking), OQ-M03 (embedding model — empirically via DJ-030, DJ-031)

---

## Governing Philosophy for This Phase

Phase 7 extends the epistemological boundary of HiFi's agents.

The agents built in Phases 3 and 4 are powerful but epistemically constrained: they know only what the deterministic financial engine can compute from structured data. The Technical Agent knows the RSI. The Fundamental Agent knows the P/E ratio. Neither agent knows what the management team said about forward guidance in the Q1 2023 earnings announcement, what supply chain risks the company disclosed in its annual report, or how the board characterised the competitive landscape. This qualitative information exists in public filings — it is simply not yet accessible to the agents.

This is not a minor limitation. From a complexity science perspective, a financial instrument is embedded in a context that cannot be fully described by price-derived statistics. The stock price is the aggregate of market beliefs about future value. Those beliefs are shaped by narratives: the narrative in the CEO's letter, the risk factor disclosures, the analyst transcript. A multi-agent system that cannot access those narratives is epistemically impoverished — it reasons about the shadow cast by the sun, not the sun.

Retrieval-Augmented Generation (RAG) is the mechanism by which agents are given access to relevant portions of that qualitative context. The insight is architectural: the context window of an LLM is finite, but the relevant information in a corpus of filings is large. RAG selects, from the corpus, the passages most semantically relevant to the current analysis query, and injects them into the context. The agent then reasons over both the deterministic numerical data (from MCP tools) and the retrieved qualitative context (from filings).

**The scientific question Phase 7 answers.** Does qualitative context from SEC filings measurably improve agent analysis quality, as measured by the Phase 5 verification metrics? The hypothesis is:
- Hallucination rate: unchanged or lower (agents are no longer forced to invent narrative context)
- Grounding rate: higher (agents can cite specific passages to ground qualitative claims)
- Disagreement entropy: lower (agents may converge when given consistent factual context)

This hypothesis is testable because Phase 5 established quantitative baselines and Phase 6 built the measurement infrastructure to track changes over time. The LangFuse dashboard will show the HR/GR delta before and after RAG is enabled. This is the scientific output of Phase 7.

**Why RAG before fine-tuning (Phase 11).** The Protocol's sequencing is deliberate. RAG adds external knowledge to the retrieval context at inference time; fine-tuning encodes knowledge into model weights at training time. Fine-tuning is only meaningful when there is enough data about what good outputs look like — which requires the measurement infrastructure from Phases 5 and 6, and enough runs to calibrate. RAG, by contrast, provides immediate access to relevant documents and is trivially reversible (turn off the knowledge server to return to baseline). It is the correct first lever.

**The Arrow continuity principle.** Storage in HiFi follows a single epistemological principle: columnar, Arrow-native, self-describing format. OHLCV data lives in Parquet (Arrow on disk). Embeddings live in LanceDB (Lance format, also Arrow on disk). The same mental model — columnar schema, deterministic reads, file-portable storage — governs every persistent data layer. This is DJ-026.

---

## Background: What the Existing Pipeline Lacks

The Phase 3–6 pipeline produces high-quality numerical analysis grounded by deterministic MCP tools. The verification layer (Phase 5) confirmed this with HR=0.000 for the Fundamental Agent and HR=0.067 for the Technical Agent on the Q1 2023 baseline. These are strong results for a system with no access to qualitative documents.

However, the agents have no access to:
- The narrative context in earnings press releases (8-K filings)
- The Management Discussion and Analysis sections of quarterly reports (10-Q)
- The risk factor disclosures and business strategy sections of annual reports (10-K)

When an agent's prompt asks "what are the key qualitative factors affecting this company's outlook?", the model is forced to draw entirely on its pre-training knowledge (static, potentially stale) rather than the company's most recent public disclosures. This is where hallucination risk is highest — not on numerical claims where the verifier catches errors, but on qualitative narrative claims that the verifier cannot yet check.

Phase 7 changes this. After Phase 7:
- Each agent call can optionally invoke a knowledge MCP tool that retrieves the most relevant passages from the company's recent filings
- Retrieved passages are injected into the generation prompt as an additional context block
- The system prompt instructs the agent to use MCP tool results for all numerical claims and retrieved context for qualitative narrative claims — maintaining the deterministic-first principle for numbers while enriching qualitative reasoning

---

## Pre-Phase Architectural Decisions

All four pre-phase decisions were made and documented in DAVID.md §17 (2026-06-11) before this plan was written. They are restated here for plan self-sufficiency.

**DJ-026 — Vector store: LanceDB.**
LanceDB is Arrow-native columnar storage, directly consistent with the Parquet/pyarrow data layer (DJ-007). Embedded mode requires no server process during development, consistent with the stdio-first principle (DJ-009). The Lance format is to embeddings what Parquet is to OHLCV.

**DJ-027 — Embedding baseline: nomic-embed-text-v1.5.**
Already loaded in LM Studio. 8192-token context handles full 10-K MD&A sections. Matryoshka representation (configurable output dims: 64–768) allows dimensionality tuning. OQ-M03 is resolved empirically in E5 via DJ-031: if Precision@5 >= 0.6, nomic-embed-text-v1.5 is accepted as the production model. If below threshold, BGE-M3 is evaluated before accepting.

**DJ-028 — Document sources: SEC EDGAR (10-K, 10-Q, 8-K).**
Free, authoritative, fixture-recordable. Same 3 tickers (AAPL, JPM, XOM) and same period (Q1 2023) as Phase 5 baseline — RAG improvement is the only variable in the measurement. Earnings call transcripts deferred to Phase 8.

**DJ-029 — Package path: `src/hifi/knowledge/`.**
Consistent with the established `src/hifi/*` package convention. EDGAR acquisition module lives at `src/hifi/data/edgar.py` (data layer), not in the knowledge package.

---

## Key Decisions To Make In This Phase (Empirical)

**DJ-030: Chunk size configuration — to be recorded at P7-E5**

Three chunking configurations are tested experimentally. The one that maximises Precision@5 on the 20-question financial evaluation set is adopted as the production configuration for all subsequent phases.

| Config | Chunk Size | Overlap | Method |
|---|---|---|---|
| A | 512 tokens (~2000 chars) | 10% | Fixed-size with sentence boundary respect |
| B | 1024 tokens (~4000 chars) | 20% | Fixed-size with sentence boundary respect |
| C | Variable | N/A | Semantic (paragraph-based — split on blank lines and section headers) |

Token count approximation: `ceil(len(text) / 4)` — English average ~4 chars/token. This avoids adding `tiktoken` as an explicit dependency (it is available as a transitive dep via `openai` and `langchain-openai` if needed, but character approximation is sufficient for the chunking experiment at this scale).

Decision to record as DJ-030 at P7-E5-T15 (after Precision@5 measured for all 3 configs).

**DJ-031: Embedding model final selection — to be recorded at P7-E5**

Default: nomic-embed-text-v1.5 (DJ-027 baseline). Acceptance criterion: Precision@5 >= 0.6 on the 20-question financial evaluation set using the winning chunking config from DJ-030. If below threshold, evaluate BGE-M3 (download to LM Studio, re-run evaluation). Document which model was tested, measured Precision@5, and why the final choice was made.

Decision to record as DJ-031 at P7-E5-T16 (after evaluation complete).

---

## Interface Design

### `FilingDocument` and `DocumentChunk` schemas

```python
# src/hifi/knowledge/schemas.py

class FilingDocument(BaseModel):
    ticker: str
    cik: str
    filing_type: str        # "10-K", "10-Q", "8-K"
    accession_number: str   # e.g., "0000320193-23-000006"
    period_of_report: date  # Q1 2023 = 2023-03-31
    filed_date: date
    sections: dict[str, str]  # section_name -> extracted_text
    source_url: str
    fetched_at: datetime

class DocumentChunk(BaseModel):
    chunk_id: str           # SHA-256 of ticker+filing_type+period+section+chunk_index
    ticker: str
    filing_type: str
    period: date
    section: str            # "MD&A", "Risk Factors", "Business", "Earnings Release"
    chunk_index: int
    text: str
    char_count: int         # actual character count
    approx_tokens: int      # ceil(char_count / 4)
    chunking_config: str    # "A", "B", or "C"
```

### `EmbeddingModel` interface

```python
# src/hifi/knowledge/embeddings.py

class EmbeddingModel:
    """Calls LM Studio /v1/embeddings endpoint."""

    def __init__(self, model: str = "nomic-embed-text-v1.5",
                 dimensions: int = 768,
                 base_url: str | None = None):
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for single text."""
        ...
```

`base_url` defaults to `HIFI_LM_STUDIO_URL` env var (same as agents). Uses `openai.OpenAI` client's `client.embeddings.create()` — already a project dependency.

### `KnowledgeStore` interface

```python
# src/hifi/knowledge/vector_store.py

class KnowledgeStore:
    """LanceDB-backed embedding store. One Lance table per chunking config."""

    def __init__(self, data_dir: Path, chunking_config: str = "A"):
        ...

    def index_chunks(self, chunks: list[DocumentChunk],
                     embeddings: list[list[float]]) -> int:
        """Index a batch of chunks + embeddings. Returns count indexed."""
        ...

    def search(self, query_embedding: list[float], ticker: str,
               top_k: int = 5) -> list[DocumentChunk]:
        """ANN search filtered by ticker. Returns top_k chunks."""
        ...

    def get_stats(self) -> dict[str, int]:
        """Return total chunks, unique tickers, unique filing types."""
        ...
```

### `KnowledgeRetriever` interface

```python
# src/hifi/knowledge/retrieval.py

class KnowledgeRetriever:
    """End-to-end retrieval: embed query -> search store -> format context."""

    def __init__(self, store: KnowledgeStore, embedding_model: EmbeddingModel):
        ...

    def retrieve(self, query: str, ticker: str,
                 top_k: int = 5) -> list[DocumentChunk]:
        """Return top_k most relevant chunks for a (query, ticker) pair."""
        ...

    def format_context(self, chunks: list[DocumentChunk]) -> str:
        """Format retrieved chunks as a context block for injection into prompts."""
        ...

def evaluate_precision_at_k(retriever: KnowledgeRetriever,
                             queries: list[EvaluationQuery],
                             k: int = 5) -> float:
    """Compute Precision@k over the evaluation query set."""
    ...
```

### `EvaluationQuery` schema

```python
class EvaluationQuery(BaseModel):
    query_id: str
    query: str
    ticker: str
    relevant_section: str      # expected source section
    relevant_filing_type: str  # expected source filing type
    notes: str = ""
```

---

## Epic P7-E1: EDGAR Data Acquisition + Fixtures

**Objective:** Build the SEC EDGAR API client and record HTTP fixtures for AAPL, JPM, and XOM (10-K, 10-Q, 8-K at Q1 2023). This is the data acquisition layer for Phase 7, analogous to Phase 1's `market.py` and `macro.py` fetchers. Production code lives in `src/hifi/data/edgar.py`; fixtures recorded in `tests/fixtures/sec/`.

**SEC EDGAR API.** No API key required. All requests include a `User-Agent` header identifying the project (EDGAR terms of service requirement). Rate limit: 10 requests/second. Base URL: `https://data.sec.gov` for metadata; `https://www.sec.gov/Archives/edgar/data/` for filing documents.

Three API calls per company:
1. `GET https://data.sec.gov/submissions/CIK{cik_padded}.json` — filing history (find accession numbers)
2. `GET https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/index.json` — filing index (find primary document filename)
3. `GET https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{primary_doc}` — actual filing HTML/text

**CIK numbers:** AAPL=0000320193, JPM=0000019617, XOM=0000034088.

**Target filings (most recent at or before 2023-03-31):**
- 10-K: annual report for FY ending ~Sep/Dec 2022
- 10-Q: quarterly report for period ending ~Dec 2022 or Mar 2023
- 8-K: earnings announcement nearest to Q1 2023 reporting

**HTML section extraction.** SEC filings are HTML (iXBRL markup). Clean text extraction uses Python's stdlib `html.parser` (`html.unescape` + tag stripping). Targeted sections are extracted by searching for EDGAR-standard item labels (e.g., "Item 7", "Item 1A", "Results of Operations"). This avoids `lxml` as a dependency.

**`EdgarFetcher` design:**

```python
# src/hifi/data/edgar.py

class EdgarFetcher:
    USER_AGENT = "HiFi Research hifi@localhost"  # required by EDGAR

    def get_submissions(self, cik: str) -> dict:
        """Fetch company submissions JSON from data.sec.gov."""
        ...

    def get_filing_index(self, cik: str, accession_number: str) -> dict:
        """Fetch filing index JSON to identify primary document."""
        ...

    def get_filing_document(self, cik: str, accession_number: str,
                            filename: str) -> str:
        """Download primary filing document. Returns raw HTML text."""
        ...

    def extract_text_sections(self, raw_html: str,
                              filing_type: str) -> dict[str, str]:
        """Strip HTML tags and extract named sections by filing type.
        Returns {section_name: clean_text}."""
        ...

    def fetch_filing(self, ticker: str, cik: str,
                     filing_type: str, as_of_date: date) -> FilingDocument:
        """High-level: find most recent filing of given type, download, extract.
        Returns a FilingDocument."""
        ...
```

**Fixture recording.** `scripts/record_sec_fixtures.py` runs once with internet access. It calls `EdgarFetcher.fetch_filing()` for each of the 9 (ticker × filing_type) combinations and saves:
- Raw HTTP response JSON (submissions) → `tests/fixtures/sec/{ticker}_submissions.json`
- Raw HTML (primary document) → `tests/fixtures/sec/{ticker}_{filing_type}_raw.html`
- Extracted sections → `tests/fixtures/sec/{ticker}_{filing_type}_sections.json`

The extracted sections fixture is the authoritative input for all downstream tests (E2+). Once recorded, no internet access is required for any test.

**Test strategy for E1:** Unit tests use the `responses` library (already in dev deps) to replay the EDGAR HTTP responses from the fixture files. Tests verify: correct URL construction, User-Agent header, CIK zero-padding, accession number formatting, section extraction logic.

| Ticket | Description | Status |
|---|---|---|
| P7-E1-T1 | Implement `EdgarFetcher.get_submissions()` with User-Agent header, CIK zero-padding, rate limit respect | PLANNED |
| P7-E1-T2 | Implement `EdgarFetcher.get_filing_index()` to parse filing index JSON and identify primary document | PLANNED |
| P7-E1-T3 | Implement `EdgarFetcher.get_filing_document()` to download filing HTML | PLANNED |
| P7-E1-T4 | Implement `EdgarFetcher.extract_text_sections()` for 10-K (Item 1, 1A, 7), 10-Q (Part I Item 2), 8-K (full body) using stdlib html.parser | PLANNED |
| P7-E1-T5 | Implement `EdgarFetcher.fetch_filing()` high-level orchestrator: find accession, download, extract | PLANNED |
| P7-E1-T6 | Implement `src/hifi/knowledge/schemas.py` with `FilingDocument`, `DocumentChunk`, `EvaluationQuery` Pydantic models | PLANNED |
| P7-E1-T7 | Implement `src/hifi/knowledge/__init__.py` package stub | PLANNED |
| P7-E1-T8 | Unit test: `get_submissions()` constructs correct URL and zero-pads CIK (responses fixture replay) | PLANNED |
| P7-E1-T9 | Unit test: `extract_text_sections()` for 10-K strips HTML tags and returns non-empty section dicts | PLANNED |
| P7-E1-T10 | Unit test: `extract_text_sections()` for 10-Q returns MD&A section | PLANNED |
| P7-E1-T11 | Unit test: `extract_text_sections()` for 8-K returns full body text | PLANNED |
| P7-E1-T12 | Unit test: `FilingDocument` schema validates period_of_report as date; sections dict is non-empty | PLANNED |
| P7-E1-T13 | Unit test: `DocumentChunk.chunk_id` is a deterministic SHA-256 prefix of its content fields | PLANNED |
| P7-E1-T14 | Write `scripts/record_sec_fixtures.py`: fetch all 9 filings, save fixtures; requires internet | PLANNED |
| P7-E1-T15 | Manual verification: run record_sec_fixtures.py; confirm 9 FilingDocuments with non-empty sections for all 3 tickers | PLANNED |

**Files to create:**
- `src/hifi/data/edgar.py`
- `src/hifi/knowledge/__init__.py`
- `src/hifi/knowledge/schemas.py`
- `scripts/record_sec_fixtures.py`
- `tests/fixtures/sec/` (directory, populated by record script)
- `tests/unit/test_edgar_fetcher.py`

---

## Epic P7-E2: Document Ingestion and Chunking

**Objective:** Transform `FilingDocument` objects (raw extracted section text) into `DocumentChunk` lists using three chunking configurations. This is the preprocessing step before embedding.

**Three chunking strategies.**

*Config A — Fixed-size 512 tokens, 10% overlap.*
Split text into windows of approximately 2000 characters (≈512 tokens at 4 chars/token). Each window overlaps the previous by 200 characters (10%). Respect sentence boundaries within ±50 characters of the target boundary (scan forward/backward for `.`, `!`, `?`, `\n\n`). This prevents splitting a sentence across chunks.

*Config B — Fixed-size 1024 tokens, 20% overlap.*
Same as Config A with window size ~4000 characters and overlap ~800 characters.

*Config C — Semantic paragraph-based.*
Split on double newlines (`\n\n`) and section dividers (lines that are all-caps or begin with "Item", "Part", "Note"). Each paragraph becomes one chunk, regardless of length. Paragraphs longer than 6000 characters are sub-split at sentence boundaries. This preserves semantic units at the cost of variable chunk sizes.

**`DocumentIngestionPipeline` design:**

```python
# src/hifi/knowledge/document_ingestion.py

class DocumentIngestionPipeline:
    """Convert FilingDocuments to DocumentChunk lists."""

    def __init__(self, chunking_config: str):
        """Config must be "A", "B", or "C"."""
        ...

    def chunk_document(self, doc: FilingDocument) -> list[DocumentChunk]:
        """Chunk all sections of a filing. Returns all chunks for that document."""
        ...

    def chunk_section(self, text: str, ticker: str, filing_type: str,
                      period: date, section: str,
                      start_index: int = 0) -> list[DocumentChunk]:
        """Chunk a single section. Returns DocumentChunk list."""
        ...
```

**Chunk IDs.** `chunk_id = hashlib.sha256(f"{ticker}|{filing_type}|{period}|{section}|{chunk_index}|{config}".encode()).hexdigest()[:16]`. Deterministic: same inputs always produce the same ID.

| Ticket | Description | Status |
|---|---|---|
| P7-E2-T1 | Implement `_fixed_size_chunker()` private function: window+overlap splitting with sentence boundary respect | PLANNED |
| P7-E2-T2 | Implement `_semantic_chunker()` private function: paragraph-based splitting on `\n\n` and section headers | PLANNED |
| P7-E2-T3 | Implement `DocumentIngestionPipeline.__init__()` selecting the correct chunker for configs A, B, C | PLANNED |
| P7-E2-T4 | Implement `chunk_section()`: apply chunker, construct `DocumentChunk` Pydantic objects with correct metadata | PLANNED |
| P7-E2-T5 | Implement `chunk_document()`: iterate over sections, call `chunk_section()`, return concatenated list | PLANNED |
| P7-E2-T6 | Unit test: Config A produces chunks with `approx_tokens` between 400 and 600 for typical MD&A text | PLANNED |
| P7-E2-T7 | Unit test: Config A overlap — adjacent chunks share ≥10% of characters from the boundary region | PLANNED |
| P7-E2-T8 | Unit test: Config B produces chunks with `approx_tokens` between 800 and 1200 | PLANNED |
| P7-E2-T9 | Unit test: Config C produces chunks at paragraph boundaries; no paragraph splits mid-sentence | PLANNED |
| P7-E2-T10 | Unit test: `chunk_id` is deterministic — identical inputs produce identical IDs across calls | PLANNED |
| P7-E2-T11 | Unit test: `chunk_document()` with a 3-section FilingDocument produces non-empty chunk list for all 3 configs | PLANNED |
| P7-E2-T12 | Unit test: no empty chunks (text.strip() is non-empty for all returned chunks) | PLANNED |
| P7-E2-T13 | Unit test: Config C does not produce chunks longer than 6000 characters | PLANNED |

**Files to create:**
- `src/hifi/knowledge/document_ingestion.py`
- `tests/unit/test_document_ingestion.py`

---

## Epic P7-E3: Embedding Pipeline

**Objective:** Embed `DocumentChunk` lists using nomic-embed-text-v1.5 via LM Studio's `/v1/embeddings` endpoint (the OpenAI embeddings API, same host as the LLM). Build a deterministic test double for all unit and integration tests that do not require live LM Studio.

**LM Studio embeddings API.** The endpoint is `POST {HIFI_LM_STUDIO_URL}/embeddings`. LM Studio exposes it on the same server as chat completions. The OpenAI Python client handles this natively:

```python
client = OpenAI(base_url=base_url, api_key="lm-studio")
response = client.embeddings.create(model="nomic-embed-text-v1.5", input=texts)
embeddings = [item.embedding for item in response.data]
```

Batch size: embed up to 32 chunks per API call to avoid hitting LM Studio's context limit.

**Matryoshka dimensions.** `nomic-embed-text-v1.5` supports truncated output via the `dimensions` parameter (when using Matryoshka). For Phase 7, use the full 768 dimensions. The parameter is configurable for future experiments.

**`DeterministicEmbeddingModel`.** For all tests that do not require live embeddings, a test double is provided. It derives a deterministic fake embedding from the input text using SHA-256:

```python
# In tests/conftest.py or a test utility module

class DeterministicEmbeddingModel:
    """Produces stable unit-norm fake embeddings from text via hash seeding.
    No external dependencies. Used in all tests that do not require LM Studio."""

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = numpy.random.default_rng(seed)
            vec = rng.standard_normal(self.dimensions)
            vec /= numpy.linalg.norm(vec)  # unit-normalise
            results.append(vec.tolist())
        return results

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
```

This satisfies the "no mocks — deterministic synthetic generators (seeded numpy)" principle. The fake embeddings have the correct shape and are unit-normalised (consistent with cosine similarity search). They are not semantically meaningful, but they allow all storage and retrieval code to be tested deterministically.

| Ticket | Description | Status |
|---|---|---|
| P7-E3-T1 | Implement `EmbeddingModel.__init__()` using `openai.OpenAI` client pointed at `HIFI_LM_STUDIO_URL`; configurable model and dimensions | PLANNED |
| P7-E3-T2 | Implement `EmbeddingModel.embed()` with batch size 32; returns `list[list[float]]` | PLANNED |
| P7-E3-T3 | Implement `EmbeddingModel.embed_one()` as convenience wrapper | PLANNED |
| P7-E3-T4 | Implement `DeterministicEmbeddingModel` in `tests/conftest.py` with SHA-256 seeded numpy unit vectors | PLANNED |
| P7-E3-T5 | Unit test: `DeterministicEmbeddingModel.embed()` returns list of correct length with each vector having `dimensions` floats | PLANNED |
| P7-E3-T6 | Unit test: `DeterministicEmbeddingModel` — same input text always produces the same embedding vector | PLANNED |
| P7-E3-T7 | Unit test: `DeterministicEmbeddingModel` — different input texts produce different embedding vectors | PLANNED |
| P7-E3-T8 | Unit test: `EmbeddingModel` uses `HIFI_LM_STUDIO_URL` env var as base URL (monkeypatched to fail fast) | PLANNED |
| P7-E3-T9 | Unit test: `embed()` splits batches correctly for input lists longer than batch_size=32 | PLANNED |

**Files to create:**
- `src/hifi/knowledge/embeddings.py`
- Addition to `tests/conftest.py` (`DeterministicEmbeddingModel`)
- `tests/unit/test_embedding_model.py`

---

## Epic P7-E4: LanceDB Vector Store

**Objective:** Build `KnowledgeStore`, a LanceDB-backed store for document chunk embeddings. Each chunking config (A, B, C) gets its own Lance table. After DJ-030 selects the winning config, the production store uses that table.

**LanceDB table schema.** Lance tables are defined with PyArrow schemas:

```python
import pyarrow as pa

CHUNK_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("ticker", pa.string()),
    pa.field("filing_type", pa.string()),
    pa.field("period", pa.string()),          # ISO date string: "2023-03-31"
    pa.field("section", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("approx_tokens", pa.int32()),
    pa.field("chunking_config", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 768)),
])
```

This schema is defined with `pyarrow` (already a dependency), consistent with DJ-007.

**`KnowledgeStore` design:**

```python
# src/hifi/knowledge/vector_store.py

class KnowledgeStore:
    def __init__(self, data_dir: Path, chunking_config: str = "A",
                 dimensions: int = 768):
        """Open or create LanceDB database at data_dir/knowledge.lance/.
        Table name: f"chunks_{chunking_config.lower()}"."""
        ...

    def index_chunks(self, chunks: list[DocumentChunk],
                     embeddings: list[list[float]]) -> int:
        """Insert chunks + embeddings into the Lance table. Returns count."""
        ...

    def search(self, query_embedding: list[float], ticker: str,
               top_k: int = 5) -> list[DocumentChunk]:
        """Cosine ANN search filtered by ticker. Returns top_k DocumentChunks."""
        ...

    def get_stats(self) -> dict[str, int]:
        """Return {n_chunks, n_tickers, n_filing_types}."""
        ...

    def clear(self) -> None:
        """Drop and recreate the table. Used in tests and during experiment reset."""
        ...
```

**Test strategy.** Unit tests use `pytest`'s `tmp_path` fixture to create a temporary LanceDB directory. The `DeterministicEmbeddingModel` provides embeddings. No LM Studio, no internet. Cosine search correctness is tested by indexing two chunks with known embeddings and verifying the more similar one is returned first.

| Ticket | Description | Status |
|---|---|---|
| P7-E4-T1 | Add `lancedb>=0.8` to `pyproject.toml` production dependencies | PLANNED |
| P7-E4-T2 | Define `CHUNK_SCHEMA` pyarrow schema in `vector_store.py` | PLANNED |
| P7-E4-T3 | Implement `KnowledgeStore.__init__()`: open LanceDB at `data_dir/knowledge.lance`, create table if absent | PLANNED |
| P7-E4-T4 | Implement `KnowledgeStore.index_chunks()`: build PyArrow table from chunks+embeddings, append to Lance table | PLANNED |
| P7-E4-T5 | Implement `KnowledgeStore.search()`: cosine ANN search with ticker pre-filter, return DocumentChunk list | PLANNED |
| P7-E4-T6 | Implement `KnowledgeStore.get_stats()`: count chunks, unique tickers, unique filing types | PLANNED |
| P7-E4-T7 | Implement `KnowledgeStore.clear()`: drop and recreate table | PLANNED |
| P7-E4-T8 | Unit test: `index_chunks()` with 5 fake chunks → `get_stats()` returns `{n_chunks: 5}` (tmp_path) | PLANNED |
| P7-E4-T9 | Unit test: `search()` returns at most `top_k` results | PLANNED |
| P7-E4-T10 | Unit test: `search()` with ticker filter only returns chunks matching that ticker | PLANNED |
| P7-E4-T11 | Unit test: `search()` cosine order — query vector identical to chunk A's embedding ranks chunk A first | PLANNED |
| P7-E4-T12 | Unit test: `clear()` resets `get_stats()` to zero | PLANNED |
| P7-E4-T13 | Unit test: `KnowledgeStore` for config "A" and config "B" use separate Lance tables (same database) | PLANNED |
| P7-E4-T14 | Unit test: all 3 chunking config tables coexist in the same Lance database directory | PLANNED |

**Files to create:**
- `src/hifi/knowledge/vector_store.py`
- `tests/unit/test_knowledge_store.py`

**Files to modify:**
- `pyproject.toml` (add `lancedb>=0.8`)

---

## Epic P7-E5: Retrieval Pipeline + Chunking Evaluation

**Objective:** Build `KnowledgeRetriever` for end-to-end query → context retrieval. Define the 20-question financial evaluation set. Run all three chunking configurations against it. Measure Precision@5 for each. Record DJ-030 (winning chunk config) and DJ-031 (embedding model acceptance).

**Evaluation query set.** 20 questions are crafted to test retrieval of specific information from the AAPL/JPM/XOM Q1 2023 corpus. Each question has a `relevant_section` and `relevant_filing_type` — the ground-truth source. Precision@5 counts a retrieved chunk as relevant if its `(ticker, section, filing_type)` matches the ground truth.

The 20 evaluation queries are stored in `tests/fixtures/retrieval/evaluation_queries.json`. They are defined as part of this ticket plan and must be written before running the evaluation:

| Query ID | Query | Ticker | Relevant Filing | Relevant Section |
|---|---|---|---|---|
| Q01 | What did Apple management say about iPhone demand and pricing in the most recent quarter? | AAPL | 10-Q | MD&A |
| Q02 | What are Apple's primary risk factors related to international sales and foreign currency? | AAPL | 10-K | Risk Factors |
| Q03 | How did Apple describe its gross margin trend and the factors affecting it? | AAPL | 10-K | MD&A |
| Q04 | What were the key highlights from Apple's most recent earnings announcement? | AAPL | 8-K | Earnings Release |
| Q05 | What is Apple's business strategy with respect to services revenue? | AAPL | 10-K | Business |
| Q06 | How did JPMorgan describe its credit loss provisions in the most recent period? | JPM | 10-Q | MD&A |
| Q07 | What are JPMorgan's principal risk factors related to credit and market risk? | JPM | 10-K | Risk Factors |
| Q08 | What did JPMorgan management say about net interest income and interest rate sensitivity? | JPM | 8-K | Earnings Release |
| Q09 | How did JPMorgan describe its capital position and CET1 ratio? | JPM | 10-Q | MD&A |
| Q10 | What are JPMorgan's business lines and how did each perform? | JPM | 10-K | Business |
| Q11 | What did ExxonMobil say about its capital expenditure plans and investment priorities? | XOM | 8-K | Earnings Release |
| Q12 | What are ExxonMobil's risk factors related to energy transition and regulatory changes? | XOM | 10-K | Risk Factors |
| Q13 | How did ExxonMobil describe its upstream production performance and outlook? | XOM | 10-Q | MD&A |
| Q14 | What is ExxonMobil's strategy for reducing carbon emissions and energy transition? | XOM | 10-K | Business |
| Q15 | How did ExxonMobil explain changes in its refining margin and downstream segment? | XOM | 10-Q | MD&A |
| Q16 | What liquidity and capital resources did Apple disclose? | AAPL | 10-Q | MD&A |
| Q17 | What macroeconomic conditions did JPMorgan's management identify as key risks? | JPM | 10-K | Risk Factors |
| Q18 | What were ExxonMobil's key financial results according to its earnings release? | XOM | 8-K | Earnings Release |
| Q19 | How did Apple describe its research and development spending and priorities? | AAPL | 10-K | Business |
| Q20 | What did JPMorgan say about its investment banking pipeline and deal activity? | JPM | 8-K | Earnings Release |

**Precision@5 calculation.** For each query, retrieve the top 5 chunks. A chunk is relevant if `(chunk.ticker == query.ticker) AND (chunk.section == query.relevant_section) AND (chunk.filing_type == query.relevant_filing_type)`. Precision@5 = (number of relevant chunks in top 5) / 5. Mean Precision@5 = average across all 20 queries.

**Retrieval latency.** Measure wall-clock time for 20 queries. Protocol criterion: < 500ms per query. Record p50 and p99 latency.

**Experiment run.** The evaluation is run against all 3 chunking configs (indexed with `DeterministicEmbeddingModel` embeddings for the unit-level evaluation, then with real embeddings from LM Studio for the one-time baseline). Results determine DJ-030 and DJ-031.

| Ticket | Description | Status |
|---|---|---|
| P7-E5-T1 | Implement `KnowledgeRetriever.__init__()` composing `KnowledgeStore` + `EmbeddingModel` | PLANNED |
| P7-E5-T2 | Implement `KnowledgeRetriever.retrieve()`: embed query → search store → return chunks | PLANNED |
| P7-E5-T3 | Implement `KnowledgeRetriever.format_context()`: format chunks as numbered passage list with source metadata | PLANNED |
| P7-E5-T4 | Implement `evaluate_precision_at_k()`: iterate over EvaluationQuery list, compute P@k per query, return mean | PLANNED |
| P7-E5-T5 | Write `tests/fixtures/retrieval/evaluation_queries.json` with all 20 queries from the table above | PLANNED |
| P7-E5-T6 | Unit test: `retrieve()` returns list of at most `top_k` DocumentChunks (DeterministicEmbeddingModel) | PLANNED |
| P7-E5-T7 | Unit test: `format_context()` includes source metadata (ticker, filing_type, section) in the returned string | PLANNED |
| P7-E5-T8 | Unit test: `evaluate_precision_at_k()` with a perfect mock retriever (always returns relevant chunk first) → P@5 = 1.0 | PLANNED |
| P7-E5-T9 | Unit test: `evaluate_precision_at_k()` with a null retriever (returns no relevant chunks) → P@5 = 0.0 | PLANNED |
| P7-E5-T10 | Unit test: retrieval latency — 20 queries against a store with 200 fake chunks (DeterministicEmbeddingModel) complete in < 500ms total | PLANNED |
| P7-E5-T11 | Integration test: `KnowledgeRetriever` with Config A store (indexed with 3-ticker fixture sections, DeterministicEmbeddingModel) returns at least 1 chunk per query | PLANNED |
| P7-E5-T12 | Integration test: Config B store same as T11 | PLANNED |
| P7-E5-T13 | Integration test: Config C store same as T11 | PLANNED |
| P7-E5-T14 | Integration test: retrieval latency for each config — all 20 queries < 500ms total (DeterministicEmbeddingModel, 3-ticker store) | PLANNED |
| P7-E5-T15 | Record DJ-030: run evaluation with real LM Studio embeddings for all 3 configs; measure P@5; record winning config and P@5 values in DAVID.md §17 | PLANNED |
| P7-E5-T16 | Record DJ-031: if winning config P@5 >= 0.6, accept nomic-embed-text-v1.5; else evaluate BGE-M3 and record comparison | PLANNED |

**Files to create:**
- `src/hifi/knowledge/retrieval.py`
- `tests/fixtures/retrieval/evaluation_queries.json`
- `tests/unit/test_retrieval.py`
- `tests/integration/test_knowledge_pipeline.py`

---

## Epic P7-E6: Knowledge MCP Server

**Objective:** Expose retrieval as an MCP tool via a FastMCP stdio server at `src/hifi/mcp/knowledge_server.py`. Agents call this tool exactly as they call the financial MCP server — no special case in agent code.

**Tool interface:**

```python
# src/hifi/mcp/knowledge_server.py

@mcp.tool()
def retrieve_context(
    query: str,
    ticker: str,
    top_k: int = 5,
) -> dict:
    """Retrieve relevant passages from SEC filings for a given query and ticker.

    Returns:
        {
          "call_id": "<12-char SHA-256>",
          "ticker": "<ticker>",
          "query": "<query>",
          "passages": [
              {"rank": 1, "filing_type": "10-K", "section": "MD&A",
               "period": "2023-03-31", "text": "..."},
              ...
          ],
          "n_retrieved": <int>
        }
    """
```

The response includes `call_id` (Phase 2 pattern) for auditability. The knowledge server is a standalone MCP subprocess, exactly like `financial_server.py`. It is started fresh per call via `call_tool()` in Phase 3's `mcp_client.py`.

**Configuration.** The knowledge server needs to know where the LanceDB database is and which chunking config to use (determined by DJ-030). These are passed via environment variables:
- `HIFI_KNOWLEDGE_DATA_DIR` — path to knowledge database directory (default: `data/knowledge/`)
- `HIFI_KNOWLEDGE_CHUNKING_CONFIG` — winning config from DJ-030 (default: `"A"`)

**Initialisation.** The server initialises `KnowledgeStore` and `EmbeddingModel` at startup. Both components use fail-open defaults: if LM Studio is unavailable, `retrieve_context` returns an empty passages list rather than crashing the agent pipeline.

| Ticket | Description | Status |
|---|---|---|
| P7-E6-T1 | Implement `knowledge_server.py` as FastMCP stdio server with `retrieve_context` tool | PLANNED |
| P7-E6-T2 | Implement server startup: load `KnowledgeStore` and `EmbeddingModel` from env vars; fail-open if store absent | PLANNED |
| P7-E6-T3 | Implement `call_id` generation in tool response (consistent with Phase 2 pattern) | PLANNED |
| P7-E6-T4 | Unit test: `retrieve_context` tool returns correct schema with `call_id`, `passages`, `n_retrieved` | PLANNED |
| P7-E6-T5 | Unit test: `retrieve_context` with empty store returns `{"passages": [], "n_retrieved": 0}` (no crash) | PLANNED |
| P7-E6-T6 | Unit test: `call_id` in knowledge server response is a 12-char hex string | PLANNED |
| P7-E6-T7 | Integration test: `call_tool("retrieve_context", {"query": "...", "ticker": "AAPL"})` via `mcp_client.py` subprocess call returns a valid dict | PLANNED |
| P7-E6-T8 | Integration test: MCP response `call_id` matches the 12-char SHA-256 pattern from Phase 2 | PLANNED |

**Files to create:**
- `src/hifi/mcp/knowledge_server.py`
- `tests/unit/test_knowledge_mcp_server.py`

---

## Epic P7-E7: Agent Augmentation (RAG-Enabled v2 Agents)

**Objective:** Add a `retrieve_context` node to both the Fundamental and Technical LangGraph agents. Create v2 prompt templates that include a `{retrieved_context}` block. The `use_rag=False` default maintains full backward compatibility with all Phase 3–6 tests.

**LangGraph graph change.** The Fundamental Agent graph gains a new node between `call_mcp_tools` and `generate_analysis`:

```
[Before Phase 7]
load_snapshot → call_mcp_tools → generate_analysis → parse_output

[After Phase 7, use_rag=True]
load_snapshot → call_mcp_tools → retrieve_context → generate_analysis → parse_output

[After Phase 7, use_rag=False or no knowledge store]
load_snapshot → call_mcp_tools ─────────────────────→ generate_analysis → parse_output
```

The `retrieve_context` node calls `call_tool("retrieve_context", {...})` against the knowledge MCP server and adds the formatted context to the LangGraph state under a new `retrieved_context: str` key (default `""`).

**State schema change.** A new optional field is added to the graph's `TypedDict` state:

```python
class FundamentalAgentState(TypedDict):
    snapshot: FundamentalsSnapshot
    mcp_results: dict
    retrieved_context: str   # NEW — empty string if RAG disabled
    analysis: FundamentalAnalysis | None
    raw_output: str
```

This is a backward-compatible addition: existing tests that do not set `retrieved_context` see an empty string, which the v1 prompt templates ignore.

**Prompt versioning.** The existing prompts (`fundamental_v1.md`, `technical_v1.md`) are unchanged. New v2 prompts add a section:

```
=== RETRIEVED CONTEXT (SEC FILINGS) ===
{retrieved_context}

IMPORTANT: Use the retrieved context above for qualitative and strategic claims.
Continue to use MCP tool results exclusively for all numerical claims.
If retrieved context is empty, rely on MCP data and pre-training knowledge only.
```

The `generate_analysis` node selects v1 or v2 based on whether `retrieved_context` is non-empty.

**`run_analysis()` signature change.** Backward-compatible:

```python
def run_analysis(
    snapshot: FundamentalsSnapshot,
    tracer: AbstractTracer | None = None,
    use_rag: bool = False,
    knowledge_data_dir: Path | None = None,
) -> FundamentalAnalysis:
```

When `use_rag=True`, the graph is assembled with the `retrieve_context` node. When `use_rag=False` (default), the graph is identical to Phase 6.

| Ticket | Description | Status |
|---|---|---|
| P7-E7-T1 | Add `retrieved_context: str` field to `FundamentalAgentState` TypedDict (default `""`) | PLANNED |
| P7-E7-T2 | Implement `retrieve_context_node` in `fundamental_agent.py`: calls knowledge MCP server, stores result in state | PLANNED |
| P7-E7-T3 | Add `use_rag` parameter to `run_analysis()`; conditionally include `retrieve_context_node` in graph | PLANNED |
| P7-E7-T4 | Create `src/hifi/agents/prompts/fundamental_v2.md` with `{retrieved_context}` block | PLANNED |
| P7-E7-T5 | Update `generate_analysis_node` to select v1 or v2 prompt based on whether `retrieved_context` is non-empty | PLANNED |
| P7-E7-T6 | Add `retrieved_context: str` field to `TechnicalAgentState` TypedDict; implement `retrieve_context_node` in `technical_agent.py` | PLANNED |
| P7-E7-T7 | Add `use_rag` parameter to `run_technical_analysis()`; conditionally include node | PLANNED |
| P7-E7-T8 | Create `src/hifi/agents/prompts/technical_v2.md` with `{retrieved_context}` block | PLANNED |
| P7-E7-T9 | Regression test: `run_analysis(use_rag=False)` produces valid FundamentalAnalysis identical to Phase 6 behaviour | PLANNED |
| P7-E7-T10 | Regression test: `run_technical_analysis(use_rag=False)` produces valid TechnicalAnalysis identical to Phase 6 behaviour | PLANNED |
| P7-E7-T11 | Unit test: `run_analysis(use_rag=True)` with monkeypatched knowledge server returning a known passage — `retrieved_context` in state is non-empty | PLANNED |
| P7-E7-T12 | Unit test: `run_technical_analysis(use_rag=True)` with monkeypatched knowledge server — `retrieved_context` in state is non-empty | PLANNED |
| P7-E7-T13 | Unit test: v2 prompt is selected when `retrieved_context` is non-empty; v1 prompt is selected when empty | PLANNED |
| P7-E7-T14 | Integration test: full `run_analysis(use_rag=True)` with stub LLM and stub knowledge server produces valid `FundamentalAnalysis` | PLANNED |
| P7-E7-T15 | Integration test: full `run_technical_analysis(use_rag=True)` with stub LLM and stub knowledge server produces valid `TechnicalAnalysis` | PLANNED |
| P7-E7-T16 | Regression: all existing Phase 3, 4, 5, 6 tests pass without modification | PLANNED |

**Files to create:**
- `src/hifi/agents/prompts/fundamental_v2.md`
- `src/hifi/agents/prompts/technical_v2.md`
- `tests/integration/test_rag_agents.py`

**Files to modify:**
- `src/hifi/agents/fundamental_agent.py`
- `src/hifi/agents/technical_agent.py`

---

## Epic P7-E8: RAG Baseline Measurement + Bitacora

**Objective:** Run the full RAG-enabled ensemble on AAPL, JPM, XOM at Q1 2023 with real LM Studio embeddings and LLM inference. Compare HR/GR/disagreement metrics against the Phase 5 baseline. Record DJ-030 and DJ-031 with measured values. Write the scientific bitacora.

**`scripts/run_phase7_rag_baseline.py`.** Requires live LM Studio (embeddings + LLM). Requires SEC fixture files (from E1) and a populated knowledge store (from E2–E4). Runs `run_ensemble(use_rag=True)` for each of the 3 baseline tickers. Saves results to `tests/fixtures/baseline/phase7_rag_baseline.json`.

**Measurement report.** The script generates a comparison table:

| Metric | Phase 5 Baseline (no RAG) | Phase 7 (with RAG) | Delta |
|---|---|---|---|
| Fundamental HR | 0.000 | TBD | TBD |
| Fundamental GR | 1.000 | TBD | TBD |
| Technical HR | 0.067 | TBD | TBD |
| Technical GR | 0.667 | TBD | TBD |
| Disagreement entropy | 0.000 | TBD | TBD |
| N contradictions | 0 | TBD | TBD |

This table is the primary scientific result of Phase 7. It is printed to stdout and saved alongside the baseline JSON. It is transcribed into the bitacora.

**Bitacora.** `doc/bitacora/PHASE_07_RAG.md` follows the format of previous phase bitacoras. It records:
- What the hypothesis was
- What was built (technical decisions, surprises, failures)
- The measured results (chunking experiment Precision@5 table, HR/GR delta table)
- DJ-030 and DJ-031 as recorded decisions with numerical evidence
- What changed in the David proximity after Phase 7

| Ticket | Description | Status |
|---|---|---|
| P7-E8-T1 | Write `scripts/run_phase7_rag_baseline.py`: build knowledge store, run ensemble with RAG, save JSON + comparison table | PLANNED |
| P7-E8-T2 | Unit test: `test_phase7_rag_baseline.py` — skip if fixture absent; validates JSON schema when present (baseline HR/GR values within expected range) | PLANNED |
| P7-E8-T3 | Holistic test: `tests/holistic/test_phase7_rag_pipeline.py` — full pipeline from indexed chunks (DeterministicEmbeddingModel) through RAG-enabled ensemble to EnsembleOutput | PLANNED |
| P7-E8-T4 | Holistic test: Phase 6 regression — `run_ensemble(use_rag=False)` still produces identical EnsembleOutput structure | PLANNED |
| P7-E8-T5 | Holistic test: Phase 5 regression — `verify_ensemble()` runs without error on Phase 7 RAG output | PLANNED |
| P7-E8-T6 | Manual: run `make baseline-phase7` (add to Makefile); confirm JSON saved and comparison table printed | PLANNED |
| P7-E8-T7 | Record DJ-030: write P@5 for configs A, B, C into DAVID.md §17 DJ-030 entry | PLANNED |
| P7-E8-T8 | Record DJ-031: write accepted embedding model + measured P@5 into DAVID.md §17 DJ-031 entry | PLANNED |
| P7-E8-T9 | Write `doc/bitacora/PHASE_07_RAG.md` with hypothesis, results, decisions, surprises | PLANNED |
| P7-E8-T10 | Update STATUS.md: Phase 7 status → COMPLETE, key metrics | PLANNED |

**Files to create:**
- `scripts/run_phase7_rag_baseline.py`
- `tests/fixtures/baseline/phase7_rag_baseline.json` (generated by script)
- `tests/unit/test_phase7_rag_baseline.py`
- `tests/holistic/test_phase7_rag_pipeline.py`
- `doc/bitacora/PHASE_07_RAG.md`

**Files to modify:**
- `Makefile` (add `baseline-phase7` target with `check_env lm-studio` guard)
- `scripts/check_env.py` (consider adding `--check knowledge-store` to verify the knowledge database is populated)
- `plans/STATUS.md`

---

## Epic Dependency Graph

```
P7-E1 (EDGAR Acquisition)
  FilingDocument fixtures, EdgarFetcher, schemas
         |
         |
P7-E2 (Document Ingestion)          P7-E3 (Embedding Pipeline)
  DocumentIngestionPipeline           EmbeddingModel
  3 chunking configs                  DeterministicEmbeddingModel
         |                                   |
         +-----------------------------------+
                         |
                 P7-E4 (Vector Store)
                  KnowledgeStore (LanceDB)
                         |
                 P7-E5 (Retrieval + Evaluation)
                  KnowledgeRetriever
                  Precision@5 for 3 configs
                  Record DJ-030, DJ-031
                         |
                 P7-E6 (Knowledge MCP Server)
                  knowledge_server.py
                         |
                 P7-E7 (Agent Augmentation)
                  v2 prompts, retrieve_context nodes
                  use_rag parameter
                         |
                 P7-E8 (Baseline + Bitacora)
                  run_phase7_rag_baseline.py
                  HR/GR delta measurement
```

E1 produces the fixture corpus; E2 and E3 are parallel (chunking and embedding are independent); E4 requires both (indexes chunks with embeddings); E5 requires E4; E6 requires E5 (uses KnowledgeStore internally); E7 requires E6; E8 requires E7.

---

## New Dependencies

**Production:**
- `lancedb>=0.8` — vector store (Arrow-native, embedded mode, no server)

**No additional dependencies needed:**
- Embeddings API: `openai` client (already a production dep) handles `/v1/embeddings`
- HTML parsing: Python stdlib `html.parser` (no `lxml` needed for Phase 7 EDGAR extraction)
- Token approximation: character-based `ceil(len(text) / 4)` — no `tiktoken` needed
- EDGAR HTTP: `requests` (already available as transitive dep via yfinance/fredapi)

**Dev:**
- No new dev dependencies (existing `responses` library handles EDGAR fixture replay)

---

## Phase 7 Quality Gates

| Gate | Criterion | Measured By |
|---|---|---|
| All unit tests pass | pytest tests/unit/, 0 failures | `make test` |
| All integration tests pass | pytest tests/integration/, 0 failures | `make test` |
| Holistic tests pass | pytest tests/holistic/test_phase7_rag_pipeline.py | `make test` |
| Phase 4–6 regressions | All existing holistic tests still pass | `make test` |
| No live LM Studio required | All tests pass without LM Studio (DeterministicEmbeddingModel) | `make test` |
| Lint clean | ruff check src/ tests/ scripts/, 0 errors | `make lint` |
| Retrieval latency | 20 queries < 500ms total on DeterministicEmbeddingModel store | P7-E5-T14 |
| Precision@5 measured | All 3 configs evaluated; DJ-030 recorded in DAVID.md | P7-E5-T15 |
| Embedding model accepted | DJ-031 recorded with P@5 value in DAVID.md | P7-E5-T16 |
| HR/GR delta documented | Comparison table (Phase 5 vs Phase 7 RAG) in bitacora | P7-E8-T9 |
| Backward compatibility | `run_analysis(use_rag=False)` produces identical output to Phase 6 | P7-E7-T9 |

---

## Commit Strategy

| Commit | Epic | Key Files |
|---|---|---|
| Phase 7 / E1: EDGAR acquisition + fixtures | P7-E1 | data/edgar.py, knowledge/schemas.py, scripts/record_sec_fixtures.py |
| Phase 7 / E2: Document ingestion + chunking | P7-E2 | knowledge/document_ingestion.py |
| Phase 7 / E3: Embedding pipeline | P7-E3 | knowledge/embeddings.py, conftest.py (DeterministicEmbeddingModel) |
| Phase 7 / E4: LanceDB vector store | P7-E4 | knowledge/vector_store.py, pyproject.toml (lancedb) |
| Phase 7 / E5: Retrieval + evaluation | P7-E5 | knowledge/retrieval.py, fixtures/retrieval/evaluation_queries.json |
| Phase 7 / E6: Knowledge MCP server | P7-E6 | mcp/knowledge_server.py |
| Phase 7 / E7: Agent augmentation | P7-E7 | fundamental_agent.py, technical_agent.py, prompts/v2, integration tests |
| Phase 7 / E8: RAG baseline + bitacora | P7-E8 | scripts/run_phase7_rag_baseline.py, bitacora, STATUS.md |

---

## Open Questions This Phase Will Answer

**OQ-K01: Optimal chunking strategy for financial documents?**
Answered empirically by Precision@5 on the 20-question evaluation set across configs A, B, C. Result recorded as DJ-030.

**OQ-M03: Which embedding model is best for financial text?**
Answered by measuring Precision@5 under nomic-embed-text-v1.5; evaluated against BGE-M3 if threshold not met. Result recorded as DJ-031.

**OQ-P7-01: Does RAG measurably improve HR/GR metrics?**
Primary scientific question. Answered by the Phase 5 vs Phase 7 comparison table in the bitacora (P7-E8).

**OQ-P7-02: Does RAG affect ensemble disagreement entropy?**
If agents now have consistent factual context, disagreement entropy may decrease. This is observed from the baseline measurement.

**OQ-P7-03: Is the 500ms retrieval latency criterion met with LanceDB?**
Answered by P7-E5-T14 (DeterministicEmbeddingModel, proxy for retrieval speed) and the baseline run.

---

## Connections to Earlier and Later Phases

**Depends on Phase 5:**
- HR/GR/hallucination_rate metrics are the measurement target for RAG improvement
- Phase 5 baseline (0.000/1.000 fundamental, 0.067/0.667 technical) is the reference

**Depends on Phase 6:**
- LangFuse observability allows the HR/GR delta to be tracked as a time series
- `run_ensemble()` tracing is unchanged; RAG adds context but does not modify the tracing architecture
- The LangFuse dashboard will show the pre-RAG vs. post-RAG score delta visually

**Phase 8 (Full Agent Population) depends on Phase 7:**
- The Sentiment Agent and Macro Agent (Phase 8) use the knowledge server for earnings transcript retrieval (deferred from Phase 7 per DJ-028)
- The chunking config (DJ-030) and embedding model (DJ-031) selected in Phase 7 become the defaults for Phase 8

**Phase 10 (Evaluation and Backtesting) depends on Phase 7:**
- RAG-enabled ensemble runs from Phase 7 onward are logged to LangFuse
- Phase 10 queries the LangFuse ClickHouse backend to analyse whether HR trends downward as the knowledge base grows
- This is the empirical test of the RAG hypothesis at scale (Phase 7 tests it on 3 tickers; Phase 10 tests it across the full investable universe)

**Phase 12 (GraphRAG) depends on Phase 7:**
- Phase 12 adds a knowledge graph layer on top of the Phase 7 RAG pipeline
- The evaluation infrastructure from Phase 7 (evaluation_queries.json, Precision@5) is reused to measure GraphRAG vs. RAG improvement
- This resolves OQ-K02 (does GraphRAG improve over plain RAG?)
