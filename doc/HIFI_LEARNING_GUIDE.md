# HiFi Learning Guide & Career Readiness Framework

## Purpose

This document serves three functions:

1. **Learning Roadmap:** Every topic we will encounter building HiFi, organized by domain, with the key concepts and critical questions a practitioner must be able to answer.

2. **Career Readiness Checklist:** Mapped to the competencies expected of a Senior AI/ML Architect with financial domain expertise. Each topic includes the interview-grade questions that separate practitioners from theorists.

3. **HiFi Training Strategy:** Each topic is linked to the specific HiFi component where it is learned through practice. Building HiFi is the curriculum. This document is the syllabus.

## How to Use This Document

Each topic has:

- **Concepts:** What you must understand
- **Critical Questions:** What you must be able to answer (the "five whys" depth)
- **HiFi Component:** Where in HiFi this is practiced
- **Readiness Level:** Self-assessed as you progress

### Readiness Scale

| Level | Symbol | Meaning |
|---|---|---|
| Not Started | `[ ]` | Haven't touched this yet |
| Exploring | `[~]` | Reading, experimenting, forming intuitions |
| Practiced | `[x]` | Built something real, can explain decisions with numbers |
| Can Teach | `[!]` | Can explain to others, defend trade-offs, know failure modes |

**The threshold for career readiness is `[x]` in every core topic and `[!]` in at least 5 topics that define your identity as an architect.**

---

## Domain 1: RAG Engineering

### 1.1 Chunking Strategies `[ ]`

**Concepts:**
- Fixed-size chunking (by token count)
- Semantic chunking (by meaning boundaries)
- Sentence-based and paragraph-based chunking
- Hierarchical chunking (parent-child relationships)
- Chunk overlap: purpose, costs, typical values (10-20%)
- The chunk size spectrum: 128 → 256 → 512 → 1024 → 2048 tokens
- Precision vs. recall trade-off: smaller chunks = more precise retrieval, less context per chunk

**Critical Questions (must answer from experience, not theory):**
- How did you decide your chunk size? What evidence drove the decision?
- What happened when chunks were too large? Too small?
- What overlap did you use and why that specific value?
- How does chunk size interact with embedding model capacity?
- Did you use the same chunking for all document types? Why or why not?
- What is the relationship between chunk size and retrieval precision in your system?

**HiFi Component:** Knowledge Systems (Section 11 of David). Financial documents (10-K filings, earnings calls, news) have different structures requiring different chunking strategies. This is where the theory becomes practice.

**Numbers you should know after building:**
- Chunk sizes tested and selected (with measured retrieval precision for each)
- Number of chunks per document type
- Processing time per document
- Storage requirements

### 1.2 Embedding Models `[ ]`

**Concepts:**
- Dense embeddings: what they represent, dimensionality, distance metrics
- Model families: OpenAI embeddings, BGE, E5, MiniLM, Instructor, Voyage, nomic-embed-text
- Local vs. cloud embedding models
- Domain adaptation: general-purpose vs. domain-specific embeddings
- Embedding dimensionality trade-offs (384 vs. 768 vs. 1024 vs. 1536)
- Matryoshka embeddings (variable dimensionality)
- Cost per embedding (cloud) vs. latency (local)
- Multilingual embedding capabilities

**Critical Questions:**
- Which embedding model did you select and why?
- What metric did you use to evaluate embedding quality? (Cosine similarity? Dot product? Why?)
- Did you evaluate domain-specific performance or rely on general benchmarks?
- What is the embedding dimensionality and why?
- What is your embedding throughput (documents/second)?
- How did you handle documents in different languages?

**HiFi Component:** Knowledge Systems. We must select and evaluate embedding models for financial text specifically — earnings calls, SEC filings, financial news. General benchmarks (MTEB) may not reflect financial domain performance.

### 1.3 Vector Databases `[ ]`

**Concepts:**
- Approximate Nearest Neighbor (ANN) search: the fundamental algorithm
- Index types: HNSW, IVF, flat
- Trade-offs: recall vs. latency vs. memory
- Vector DB options: Pinecone, Weaviate, Qdrant, Chroma, Milvus, pgvector
- Local-first options: Chroma, Qdrant (local mode), pgvector, LanceDB
- Metadata filtering alongside vector search
- Hybrid search (vector + keyword)

**Critical Questions:**
- Explain approximate nearest neighbor search without naming a product
- Why did you choose this vector DB? What were the alternatives?
- What index type did you use? What recall did you achieve?
- How many vectors? What was retrieval latency at that scale?
- How did you handle updates (new documents, changed documents)?
- What is the memory footprint of your index?

**HiFi Component:** Knowledge Systems. Local-first constraint eliminates cloud-only options. We need to evaluate Chroma, Qdrant, LanceDB, or pgvector for our specific scale and query patterns.

### 1.4 Retrieval Strategies `[ ]`

**Concepts:**
- Similarity search (dense retrieval)
- BM25 (sparse/keyword retrieval)
- Hybrid search (combining dense + sparse)
- Reranking: cross-encoders, Cohere Rerank, local rerankers
- Metadata filtering (filter by date, source, type before retrieval)
- Multi-query retrieval (reformulate query for broader recall)
- Contextual compression (summarize retrieved chunks)
- Retrieval evaluation: Precision@K, Recall@K, MRR, NDCG

**Critical Questions:**
- How did you improve retrieval quality beyond naive similarity search?
- What was your retrieval precision? How did you measure it?
- Did you use rerankers? What improvement did they provide?
- How did you handle the case where the answer spans multiple chunks?
- What is your retrieval latency budget and how did you meet it?

**HiFi Component:** Knowledge Systems, Verification Layer. Both RAG for agents and fact-checking for verification depend on retrieval quality.

---

## Domain 2: LLM Engineering

### 2.1 Model Selection & Evaluation `[ ]`

**Concepts:**
- Model families: GPT, Claude, Llama, Qwen, Gemma, Mistral, Phi
- Open vs. closed models: trade-offs (capability, cost, privacy, customizability)
- Model sizing: 7B, 13B, 14B, 32B, 70B — what each tier can and cannot do
- Quantization: GGUF, GPTQ, AWQ, bitsandbytes — what is lost, what is preserved
- Context windows: 4K, 8K, 32K, 128K — practical limits vs. advertised limits
- Benchmark literacy: MMLU, HumanEval, MT-Bench, what they actually measure
- Local inference engines: Ollama, llama.cpp, vLLM, MLX

**Critical Questions:**
- Why this model and not another? What evidence supports the choice?
- What is the actual quality difference between 7B and 70B for your specific task?
- What quantization level did you use? What quality was lost?
- What is your inference latency? Tokens per second?
- How did you evaluate model quality for your specific domain?
- What is the effective context window (vs. advertised) for your use case?

**HiFi Component:** Model Layer (Section 9 of David). We must select, quantize, benchmark, and deploy multiple models locally. Every model choice must be justified with measured performance, not marketing claims.

### 2.2 Prompt Engineering `[ ]`

**Concepts:**
- Zero-shot, few-shot, chain-of-thought prompting
- Structured output generation (JSON mode, constrained generation)
- System prompts: purpose, design, common failures
- Function/tool calling: how it works, when to use it
- Prompt templates and variable injection
- Prompt versioning and management
- Temperature, top-p, top-k: what they control, how to set them
- Prompt injection and security

**Critical Questions:**
- Show me a prompt that failed. How did you fix it?
- How do you enforce structured output from an LLM?
- What is the difference between asking for JSON and using constrained generation?
- How do you version and manage prompts in production?
- What temperature do you use and why?
- How do you prevent prompt injection?

**HiFi Component:** Agent Architecture (Section 10 of David). Each agent has a specialized prompt. Prompt design directly affects agent output quality, structured output compliance, and downstream verification.

### 2.3 Hallucination Detection & Mitigation `[ ]`

**Concepts:**
- Types of hallucination: factual, relational, fabricated references
- Grounding: anchoring LLM output to retrieved evidence
- Claim extraction and verification
- Faithfulness metrics: how well does the output use the provided context?
- Self-consistency checks: does the model agree with itself across runs?
- Deterministic-first principle: removing LLM from verifiable computations entirely
- The irreducibility of hallucination: it is mitigated, never eliminated

**Critical Questions:**
- How do you measure hallucination rate? What is the formula?
- What is your system's hallucination rate? How did you achieve it?
- What is the difference between a hallucination and a wrong answer?
- How do you distinguish factual claims from interpretive claims?
- Can you describe your verification pipeline?
- What happens when a hallucination is detected? (System behavior, not just logging)

**HiFi Component:** Verification Layer (Section 13 of David). This is a central architectural feature. We build a full claim extraction → classification → verification pipeline.

**Interview Note:** If you say "our architecture prevents hallucinations," you fail. If you say "our architecture detects and measures hallucinations, achieving a rate of X% with specific verification mechanisms," you pass.

---

## Domain 3: Multi-Agent Systems

### 3.1 Agent Architecture Fundamentals `[ ]`

**Concepts:**
- What is an agent (perception → reasoning → action cycle)
- Single agent vs. multi-agent: when each is appropriate
- Agent communication patterns: shared memory, message passing, blackboard
- Agent roles: specialist, critic, coordinator, contrarian
- State management across agent interactions
- Agent independence vs. communication trade-offs
- Failure modes: herding, groupthink, deadlocks, cascading errors

**Critical Questions:**
- Why multiple agents instead of one capable agent? What is the measurable benefit?
- How do agents communicate? What state do they share?
- How do you handle agent disagreement?
- What happens when one agent fails or produces garbage?
- How do you prevent agents from simply copying each other?
- What is the coordination cost of your multi-agent system?

**HiFi Component:** Agent Layer (Section 10) + Collective Decision Engine (Section 12). We build 5-7 heterogeneous agents and empirically test whether multi-agent outperforms single-agent.

### 3.2 LangGraph `[ ]`

**Concepts:**
- State graphs: nodes, edges, conditional edges
- State schema: TypedDict or Pydantic state definitions
- Cycles in graphs: why they matter for iterative reasoning
- Conditional routing: branching logic based on state
- Persistence and checkpointing: resumable workflows
- Human-in-the-loop patterns
- Subgraphs: composing complex workflows from simpler ones
- Streaming: real-time output from graph execution

**Critical Questions:**
- What problem does LangGraph solve that LangChain chains cannot?
- Explain the concept of a stateful graph for agent orchestration
- How does checkpointing work and why does it matter for reproducibility?
- How do you handle errors in the middle of a graph execution?
- What is the difference between LangGraph and a simple DAG executor?
- When would you NOT use LangGraph?

**HiFi Component:** Agent orchestration layer. The entire agent coordination — from parallel independent analysis to collective decision — is a LangGraph workflow.

### 3.3 Collective Intelligence & Aggregation `[ ]`

**Concepts:**
- Wisdom of crowds: conditions (diversity, independence, decentralization, aggregation)
- Voting mechanisms: majority, weighted, ranked-choice
- Confidence calibration: do self-reported confidence scores match actual accuracy?
- Structured debate: benefits and risks (group polarization)
- Diversity measurement: how to quantify that agents are truly different
- Herding: detection and prevention
- The contrarian role: devil's advocate vs. authentic dissent (Nemeth et al., 2001)

**Critical Questions:**
- How do you aggregate agent opinions? Why this method?
- How do you know your agents are actually diverse and not producing correlated outputs?
- What is the measured correlation between your agents' decisions?
- Does your ensemble outperform the best individual agent? By how much?
- What happens when the ensemble is confidently wrong?
- How did you decide the number of agents?

**HiFi Component:** Collective Decision Engine (Section 12). We implement and compare multiple aggregation methods: majority vote, confidence-weighted, performance-weighted, structured debate.

---

## Domain 4: Fine-Tuning

### 4.1 Parameter-Efficient Fine-Tuning `[ ]`

**Concepts:**
- Full fine-tuning vs. parameter-efficient methods
- LoRA: Low-Rank Adaptation — what it does mathematically, rank selection
- QLoRA: quantized base model + LoRA adapters
- Adapter placement: which layers to adapt (attention, MLP, all)
- Training data requirements: format, quality, quantity
- Training hyperparameters: learning rate, batch size, epochs, warmup
- Evaluation of fine-tuned models: held-out test set, task-specific metrics
- Catastrophic forgetting: losing general capabilities through fine-tuning
- Merging multiple LoRA adapters

**Critical Questions:**
- Why fine-tune at all? What can't the base model do?
- What LoRA rank did you use and why?
- How much training data did you use? How did you prepare it?
- How did you evaluate whether fine-tuning actually helped?
- Did fine-tuning hurt any general capabilities?
- What was your training time and hardware requirement?

**HiFi Component:** Model Layer (Section 9). We fine-tune multiple models for financial reasoning tasks using LoRA/QLoRA on Apple Silicon via MLX.

### 4.2 Training Data Engineering `[ ]`

**Concepts:**
- Instruction formatting: Alpaca format, ChatML, custom formats
- Data quality > data quantity (Superficial Alignment Hypothesis)
- Synthetic training data generation: using strong models to generate training data for weaker models
- Data contamination: ensuring evaluation data is not in training data
- Domain-specific instruction pairs for financial reasoning
- Preference data for RLHF/DPO (if applicable)

**Critical Questions:**
- How did you create your training data?
- How did you ensure quality?
- Did you use synthetic data? How did you validate it?
- How did you prevent data contamination between train and test sets?
- What format did you use and why?

**HiFi Component:** Dataset Families C and D (Reference Strategies + Explanations) serve as fine-tuning inputs. Data quality engineering is a core skill.

---

## Domain 5: Knowledge Graphs & GraphRAG

### 5.1 Knowledge Graphs `[ ]`

**Concepts:**
- Graph data model: nodes (entities), edges (relationships), properties
- Entity extraction: named entities from financial text
- Relationship extraction: identifying connections between entities
- Graph databases: Neo4j, ArangoDB, or lightweight alternatives
- Graph query languages: Cypher (Neo4j), SPARQL
- Financial knowledge graph: companies, sectors, supply chains, executives, events
- Graph construction: manual, semi-automatic, fully automatic (LLM-extracted)
- Graph maintenance: how to keep the graph current

**Critical Questions:**
- What entities and relationships does your knowledge graph contain?
- How did you construct it? Manual, automatic, or hybrid?
- What is the quality of automatically extracted relationships?
- How do you handle graph evolution (new companies, changed relationships)?
- What queries does the graph enable that a vector store cannot?
- What is the size of your graph (nodes, edges)?

**HiFi Component:** Knowledge Layer (Section 11). Financial knowledge graph connecting companies, sectors, supply chains, macro factors.

### 5.2 GraphRAG `[ ]`

**Concepts:**
- How GraphRAG extends standard RAG with graph traversal
- Community detection in knowledge graphs (Leiden algorithm)
- Graph-based query expansion: using relationships to find related context
- Hierarchical summarization: community summaries at different granularity levels
- Local vs. global search in GraphRAG
- Microsoft's GraphRAG implementation: how it works, limitations
- Cost of GraphRAG: graph construction, indexing, query expansion overhead

**Critical Questions:**
- How does GraphRAG improve over standard RAG for your specific use case?
- What is the measurable improvement in retrieval quality?
- What is the additional cost (latency, computation) of GraphRAG vs. RAG?
- When is GraphRAG NOT worth the complexity?
- How do you construct and maintain the graph that GraphRAG uses?

**HiFi Component:** Knowledge Layer. We implement and compare RAG vs. GraphRAG for financial document retrieval, measuring the actual improvement (if any).

---

## Domain 6: Observability & MLOps

### 6.1 LLM Observability `[ ]`

**Concepts:**
- Tracing: following a request through the entire pipeline
- Spans: individual operations within a trace
- Generations: LLM calls with input/output recording
- Evaluation: automated quality scoring of LLM outputs
- LangFuse: self-hosted LLM observability platform
- Metrics: latency, token usage, cost, quality scores
- Dashboards: real-time monitoring of system health

**Critical Questions:**
- What do you observe in your LLM system? Why those specific things?
- How do you trace a decision back to its inputs?
- What is your latency budget and how do you monitor it?
- How did you detect quality degradation in production?
- What alerting exists and what thresholds trigger alerts?

**HiFi Component:** Observability Layer (Section 14). LangFuse is the primary observability tool, instrumented across all agents and MCP calls.

### 6.2 Drift Detection `[ ]`

**Concepts:**
- Data drift: input distribution changes
- Concept drift: relationship between inputs and outputs changes
- Model drift: model performance degrades over time
- Detection methods: KS test, PSI, CUSUM, windowed comparisons
- Financial regime changes as a source of drift
- Retraining triggers: when and how to respond to detected drift

**Critical Questions:**
- What types of drift did you observe?
- How did you detect drift? What statistical tests?
- What was your retraining strategy?
- How long after drift begins can you detect it?
- What is the cost of not detecting drift (measured in performance degradation)?

**HiFi Component:** Observability Layer, drift detection subsystem. Financial markets are non-stationary — drift detection is not optional, it is essential.

### 6.3 Deployment & Containerization `[ ]`

**Concepts:**
- Docker: images, containers, volumes, networks
- Docker Compose: multi-container orchestration
- Container design: single responsibility, minimal images
- Environment management: dev, research, paper trading
- Health checks and restart policies
- Resource limits: memory, CPU allocation per container
- Local deployment on Apple Silicon: ARM architecture considerations

**Critical Questions:**
- How did you containerize your system?
- How many containers? What does each do?
- What happens when a container crashes?
- How do you manage configuration across environments?
- What is the total resource footprint?
- How does someone who is not you deploy the system?

**HiFi Component:** Deployment Strategy (Section 16). The entire system is containerized for reproducible deployment.

### 6.4 Experiment Tracking `[ ]`

**Concepts:**
- Experiment registries: MLflow, Weights & Biases, custom
- What to track: hyperparameters, data versions, model versions, metrics, artifacts
- Reproducibility: can you re-run an experiment and get the same result?
- Comparison: comparing experiments across versions
- Artifact management: models, datasets, prompts

**Critical Questions:**
- How do you track experiments?
- Can you reproduce a result from 3 months ago? How?
- What metadata do you record per experiment?
- How do you compare two experiments?

**HiFi Component:** Experiment Registry (Section 7.10). Every experiment is uniquely identified and reproducible.

---

## Domain 7: Financial Engineering

### 7.1 Quantitative Analysis `[ ]`

**Concepts:**
- Financial ratios: profitability, liquidity, solvency, efficiency
- Valuation methods: P/E, P/B, EV/EBITDA, DCF
- Technical analysis: trend, momentum, volatility, volume indicators
- Risk metrics: VaR, CVaR, Sharpe, Sortino, Calmar, maximum drawdown
- Portfolio theory: diversification, efficient frontier, risk-return trade-off
- Factor models: Fama-French, momentum factor, quality factor
- Market microstructure: bid-ask spread, market impact, slippage

**Critical Questions:**
- How do you compute and interpret Sharpe ratio? What are its limitations?
- What is the difference between VaR and CVaR? When does it matter?
- How do you handle survivorship bias in backtesting?
- What is look-ahead bias and how do you prevent it?
- How do you account for transaction costs in performance evaluation?
- What is walk-forward validation and why does it matter for financial data?

**HiFi Component:** Deterministic Financial Engine, MCP Financial Calculator Server, Evaluation Framework. All financial computations are deterministic and verifiable.

### 7.2 Backtesting & Evaluation `[ ]`

**Concepts:**
- Walk-forward validation: respecting temporal ordering
- Purged cross-validation (López de Prado): preventing information leakage
- Embargo periods between train and test
- Transaction cost modeling
- Regime-aware evaluation: performance by market condition
- Statistical significance: bootstrap confidence intervals, Diebold-Mariano test
- Common backtesting pitfalls: survivorship bias, look-ahead bias, overfitting

**Critical Questions:**
- How do you prevent look-ahead bias in your evaluation?
- What is purged cross-validation and why is it necessary for financial data?
- How do you test statistical significance of your results?
- What are bootstrap confidence intervals and why do you use them?
- How do you separate skill from luck in backtesting results?

**HiFi Component:** Evaluation Framework (Section 15). Rigorous backtesting with walk-forward validation and statistical testing.

### 7.3 Market Regimes `[ ]`

**Concepts:**
- Regime definition: bull, bear, crisis, recovery, sideways, high/low volatility
- Regime detection methods: HMM, rule-based, threshold-based
- Regime-dependent strategy behavior
- Non-stationarity: markets change their statistical properties over time
- Structural breaks vs. gradual shifts

**Critical Questions:**
- How do you define and detect market regimes?
- How does your system's performance vary across regimes?
- What happens during regime transitions?
- How many regimes do you model and why?

**HiFi Component:** Data Engineering Layer (regime classification), Agent Layer (agents should know the current regime), Evaluation Framework (regime-segmented evaluation).

---

## Domain 8: Complexity Science

### 8.1 Complex Adaptive Systems `[ ]`

**Concepts:**
- Properties: emergence, adaptation, nonlinearity, feedback, self-organization
- Difference between complicated and complex systems
- Financial markets as complex adaptive systems (Arthur, 2021)
- Agent-based modelling: populations of interacting heterogeneous agents
- Phase transitions: qualitative changes in system behavior at critical parameters
- Power laws: heavy-tailed distributions in financial returns (Cont, 2001)
- Feedback loops: positive (amplifying) and negative (stabilizing)

**Critical Questions:**
- What is the difference between emergence and aggregation?
- How do you distinguish genuine emergence from simple averaging?
- What makes a system "complex adaptive" vs. merely "complicated"?
- How would you test for a phase transition in agent collective behavior?
- Why are financial returns heavy-tailed and why does it matter?
- How do feedback loops manifest in your system?

**HiFi Component:** The worldview. Every design decision in HiFi is informed by CAS thinking. This is also the foundation for the future publication.

### 8.2 Collective Intelligence `[ ]`

**Concepts:**
- Wisdom of crowds conditions (Surowiecki, 2005): diversity, independence, decentralization, aggregation
- Diversity-prediction theorem (Page, 2007): Collective Error = Average Individual Error - Diversity
- Group polarization: deliberation pushing groups to extremes (Sunstein, 2006)
- Herding: convergence on majority opinion regardless of private signal
- Minority influence: how dissenters improve group decisions (Nemeth et al., 2001)
- Measurement: disagreement entropy, opinion dispersion, herding coefficient

**Critical Questions:**
- State the Page diversity-prediction theorem. What does it imply for ensemble design?
- How do you measure whether your agents are truly independent?
- Under what conditions does collective intelligence fail?
- What is the role of the contrarian in your system? Does it empirically help?
- How do you measure herding? What herding coefficient do your agents exhibit?

**HiFi Component:** Collective Decision Engine, Complexity Metrics, Ablation Studies. These concepts are operationalized and measured, not just discussed.

### 8.3 Emergence & Measurement `[ ]`

**Concepts:**
- Formal definitions of emergence (weak vs. strong)
- Downward causation: macro-level patterns constraining micro-level behavior
- How to test for emergence empirically (null models, ablation)
- Attractors: stable patterns the system converges to
- Entropy measures: Shannon entropy, transfer entropy
- Network analysis: applied to agent interaction patterns

**Critical Questions:**
- How do you test whether your ensemble exhibits emergence vs. simple averaging?
- What null model do you compare against?
- How do you measure the information content of agent disagreement?
- What would constitute evidence that your system exhibits downward causation?

**HiFi Component:** Long-term research program. These questions are addressed through the Agent Interaction Datasets (Family E) and complexity metrics.

---

## Domain 9: MCP & Tool Design

### 9.1 Model Context Protocol `[ ]`

**Concepts:**
- MCP specification: tools, resources, prompts
- MCP servers: stateful services exposing capabilities to LLMs
- Tool design: making LLM-callable tools with clear schemas
- Transport layer: stdio, SSE, HTTP
- Tool discovery: how agents find available tools
- Security: controlling what tools agents can call
- MCP vs. function calling: standardization and interoperability

**Critical Questions:**
- What is MCP and how does it differ from raw function calling?
- How do you design a good MCP tool interface?
- What happens when a tool call fails?
- How do you test MCP servers independently of the LLM?
- What is the performance overhead of MCP communication?

**HiFi Component:** Intelligence Architecture (Section 6). MCP is the nervous system connecting agents to deterministic engines.

### 9.2 Tool Design Principles `[ ]`

**Concepts:**
- Single responsibility: each tool does one thing well
- Schema design: clear parameter names, types, descriptions
- Error handling: what the tool returns on failure
- Idempotency: calling the same tool twice with the same inputs produces the same result
- Composability: combining simple tools to perform complex operations
- Documentation: tools must be self-describing for LLMs to use them correctly

**Critical Questions:**
- How did you decide what to expose as a tool vs. handle in the LLM prompt?
- How do you handle tool errors gracefully?
- What happens when a tool is slow? How does the agent handle timeouts?
- How do you test tools in isolation?

**HiFi Component:** All MCP servers (Financial Calculator, Market Data, Risk Analytics, etc.).

---

## Domain 10: Software Architecture

### 10.1 Systems Design `[ ]`

**Concepts:**
- Layered architecture: separation of concerns
- Event-driven architecture: loosely coupled components
- Microservices vs. monolith: trade-offs (for a local system)
- API design: REST, gRPC, or internal interfaces
- Data flow: how information moves through the system
- Error handling: cascading failures, circuit breakers, graceful degradation
- Configuration management: separating config from code

**Critical Questions:**
- Draw the architecture of your system. Explain data flow.
- What happens when component X fails?
- How do you handle cascading failures?
- Why did you separate (or not separate) these components?
- What are the interfaces between components? How are they defined?

**HiFi Component:** System Architecture (Section 7). HiFi is a multi-layer system with well-defined interfaces.

### 10.2 Data Engineering `[ ]`

**Concepts:**
- ETL/ELT patterns
- Data versioning: DVC, content hashing, immutable datasets
- Data lineage: tracking transformations from source to feature
- Data quality: validation, completeness checks, anomaly detection
- Storage formats: Parquet, Arrow, DuckDB, SQLite
- Time-series data: specific challenges (temporal ordering, point-in-time accuracy)

**Critical Questions:**
- How do you version datasets?
- Can you trace any feature back to its raw source?
- How do you ensure data quality?
- What happens when a data source fails or provides bad data?
- What storage format did you choose and why?

**HiFi Component:** Data Acquisition + Data Engineering layers.

---

## Domain 11: Classical ML & Deep Learning

### 11.1 Classical Machine Learning `[ ]`

**Concepts:**
- Feature engineering: domain-specific feature creation
- Feature selection: importance, correlation analysis, PCA
- Model families: linear models, trees (Random Forest, XGBoost, LightGBM), SVMs
- Bias-variance trade-off
- Cross-validation: k-fold, time-series split, purged CV
- Evaluation metrics: precision, recall, F1, AUC-ROC, AUC-PR
- Class imbalance: SMOTE, class weights, threshold tuning
- Hyperparameter optimization: grid search, Bayesian optimization

**Critical Questions:**
- What features mattered most? How did you determine this?
- When should you use PCA? When should you not?
- Why XGBoost for this problem? Why not a neural network?
- Which evaluation metric mattered and why?
- How did you handle class imbalance?

**HiFi Component:** Feature engineering for Dataset Family B. Classical ML may serve as baseline models for comparison.

### 11.2 Deep Learning Fundamentals `[ ]`

**Concepts:**
- Architectures: CNN (spatial), LSTM/GRU (sequential), Transformer (attention)
- Attention mechanism: what it does, self-attention, cross-attention
- Training: batch size, learning rate, optimizers (Adam, AdamW), schedulers
- Regularization: dropout, weight decay, early stopping
- Transfer learning: using pretrained models for downstream tasks
- The Transformer architecture: encoder, decoder, encoder-decoder

**Critical Questions:**
- Explain the attention mechanism. What problem does it solve?
- When would you use LSTM vs. Transformer for sequence data?
- What learning rate did you use? How did you find it?
- What batch size? What was the trade-off?
- How did you know when to stop training?

**HiFi Component:** Understanding Transformers is prerequisite for understanding LLMs. Fine-tuning builds directly on these concepts.

---

## Domain 12: Security, Ethics & Compliance

### 12.1 AI Security `[ ]`

**Concepts:**
- Prompt injection: direct and indirect
- Data poisoning: corrupted training/retrieval data
- Model extraction: stealing model capabilities through queries
- PII handling: identifying and protecting personal information
- Embedding security: can sensitive information be recovered from embeddings?
- Access control: who can use which agents and tools
- Audit logging: immutable records of all system actions

**Critical Questions:**
- How do you prevent prompt injection in a multi-agent system?
- How do you handle PII in financial documents?
- What security boundaries exist between components?
- How do you audit who did what in the system?

**HiFi Component:** Relevant throughout, especially in verification and observability layers.

### 12.2 Responsible AI in Finance `[ ]`

**Concepts:**
- Explainability requirements for financial recommendations
- Regulatory context: what can and cannot be automated
- Bias in financial models: sector bias, survivorship bias, data availability bias
- The difference between a recommendation and a decision
- Disclaimer and disclosure requirements

**Critical Questions:**
- Can you explain every recommendation your system makes?
- What biases exist in your training data and how do you address them?
- How do you ensure the system does not give unsuitable recommendations?

**HiFi Component:** Verification Layer, Observability, Structured Rationale generation.

---

## Domain 13: Professional & Leadership Skills

### 13.1 Business Impact & Cost Awareness `[ ]`

**Concepts:**
- ROI of AI projects: how to measure and communicate
- Cost structure: compute, storage, data, engineering time
- Build vs. buy decisions
- KPIs: what the business cares about vs. what the model optimizes
- The "so what?" test: can you explain why this project matters to a non-technical stakeholder?

**Critical Questions:**
- What is the business value of your system?
- What does it cost to run monthly?
- What would happen if the system disappeared tomorrow?
- Why was this project funded?

**HiFi Component:** The "why" behind every HiFi decision. Cost-awareness of local vs. cloud is a central HiFi theme.

### 13.2 Trade-off Thinking `[ ]`

**Concepts:**
- Every design decision involves trade-offs
- Latency vs. quality vs. cost triangle
- Simple vs. sophisticated: when simple wins
- Reversible vs. irreversible decisions
- The "good enough" threshold: when to stop optimizing

**Critical Questions:**
- You have 50ms latency target, GPT-4 quality, 10M documents, limited budget. What do you sacrifice?
- What is the simplest version of your system that still provides value?
- What decisions are you least confident about? What would change your mind?

**HiFi Component:** Every architectural decision in HiFi. The Decision Journal (Section 17 of David) explicitly records trade-offs.

### 13.3 Communication & Teaching `[ ]`

**Concepts:**
- Explaining technical decisions to non-technical stakeholders
- Writing clear technical documentation
- Presenting trade-offs without jargon
- Teaching junior engineers through code review and pair programming
- The "five whys" depth: becoming more precise with each follow-up

**Critical Questions:**
- Explain your system to a business executive in 2 minutes
- Explain your chunking strategy to a junior engineer
- What mistakes do juniors commonly make in your domain?

**HiFi Component:** The educative journal principle. HiFi's documentation is designed to teach, not just specify.

### 13.4 Failure Stories & Production Scars `[ ]`

**Concepts:**
- Real projects fail. Real engineers have scars.
- The value of post-mortems
- Learning from failure vs. celebrating success
- The "everything was successful" red flag

**Critical Questions:**
- Tell me about a project that failed. What did you learn?
- What was the worst production issue you personally solved?
- What technology would you never use again? Why?

**HiFi Component:** The entire HiFi journey. Document failures, not just successes. The educative journal captures what went wrong and why.

---

## Domain 14: Numbers You Must Know

After building HiFi, you should be able to cite approximate numbers for:

### RAG Numbers `[ ]`
- Number of documents ingested: ___
- Number of chunks: ___
- Chunk size selected: ___ tokens
- Embedding dimension: ___
- Retrieval latency (p50): ___ ms
- Retrieval latency (p99): ___ ms
- Retrieval precision@5: ___
- Vector store size on disk: ___ GB

### LLM Numbers `[ ]`
- Models evaluated: ___
- Selected model sizes: ___ B parameters
- Quantization level: ___-bit
- Inference latency per agent: ___ seconds
- Tokens per second (generation): ___
- Context window used (effective): ___ tokens
- Memory per model: ___ GB

### Agent Numbers `[ ]`
- Number of agents: ___
- Hallucination rate: ____%
- Grounding rate: ____%
- Inter-agent correlation (average): ___
- Disagreement entropy (average): ___
- Ensemble improvement over best individual: ___% (directional accuracy)

### Financial Numbers `[ ]`
- Stocks in universe: ___
- Historical data period: ___ to ___
- Evaluation periods tested: ___
- Directional accuracy: ____%
- Sharpe ratio: ___
- Maximum drawdown: ____%
- Paper trading duration: ___ days
- Paper trading return: ____%

### System Numbers `[ ]`
- Total containers: ___
- End-to-end analysis time per stock: ___ seconds
- Total memory footprint: ___ GB
- Disk usage: ___ GB
- Fine-tuning time per model: ___ hours
- Dataset sizes: ___ records

---

## Domain 15: The "Five Whys" Self-Test

For any component of HiFi, you should be able to answer five increasingly specific "why" questions. Here are the chains you should practice:

**Chain 1: Architecture**
1. Why multiple agents? → Diversity reduces correlated errors
2. Why these specific agents? → Each covers an independent information domain
3. Why independent before aggregation? → Independence is a condition for collective intelligence
4. Why confidence-weighted voting? → Not all agents are equally reliable on every stock
5. Why not learned aggregation from the start? → Insufficient training data; simple methods first

**Chain 2: Deterministic-First**
1. Why deterministic engines? → LLMs hallucinate on numerical tasks
2. Why MCP specifically? → Standardized, auditable, tool discovery built-in
3. Why separate servers per capability? → Single responsibility, independent testing, modular replacement
4. Why not embed computations in agent prompts? → Unverifiable, unreproducible, unauditable
5. Why not at least cache LLM computations? → Cache invalidation on live data; deterministic is always fresh

**Chain 3: Fine-Tuning**
1. Why fine-tune? → Base models lack domain-specific financial reasoning
2. Why LoRA? → Full fine-tuning is too expensive for consumer hardware
3. Why rank 16 (or whatever)? → Empirically tested ranks 4/8/16/32; 16 gave best quality/cost
4. Why this training data? → Reference strategy dataset with verified labels
5. Why not just use better prompts? → Tested; fine-tuning improved accuracy by X% on held-out set

---

## David Proximity Tracking

### How to Use

As HiFi is built, update the status column for each David section. This tracks how close the implementation is to the aspirational specification.

### Status Legend

| Symbol | Meaning |
|---|---|
| `---` | Not started |
| `PARTIAL` | Partially implemented (note what's missing) |
| `COMPLETE` | Fully implemented as specified in the David |
| `DEFERRED` | Consciously deferred (with documented reason) |
| `EXCEEDED` | Implementation goes beyond David specification |
| `DIVERGED` | Implementation deliberately differs (with documented reason) |

### Conformance Matrix

| David Section | David Reference | Status | Conformance Notes |
|---|---|---|---|
| Data Acquisition | §7.1 | `---` | |
| Data Engineering | §7.2 | `---` | |
| Knowledge Layer (RAG) | §11.2 | `---` | |
| Knowledge Layer (GraphRAG) | §11.3 | `---` | |
| Knowledge Layer (Knowledge Graph) | §11.1 | `---` | |
| Model Selection | §9.2-9.3 | `---` | |
| Model Fine-Tuning | §9.4 | `---` | |
| Model Inference Stack | §9.5 | `---` | |
| Fundamental Agent | §10.2 | `---` | |
| Valuation Agent | §10.2 | `---` | |
| Technical Agent | §10.2 | `---` | |
| Risk Agent | §10.2 | `---` | |
| Macro Agent | §10.2 | `---` | |
| Sentiment Agent | §10.2 | `---` | |
| Contrarian Agent | §10.2 | `---` | |
| Agent Memory | §10.4 | `---` | |
| Agent Diversity | §10.3 | `---` | |
| Majority Voting | §12.2.1 | `---` | |
| Confidence-Weighted Voting | §12.2.2 | `---` | |
| Performance-Weighted Voting | §12.2.3 | `---` | |
| Structured Debate | §12.2.4 | `---` | |
| Adaptive Aggregation | §12.2.5 | `---` | |
| Claim Extraction | §13.2 | `---` | |
| Numerical Verification | §13.3 | `---` | |
| Consistency Verification | §13.5 | `---` | |
| Hallucination Metrics | §13.4 | `---` | |
| LangFuse Integration | §14.2 | `---` | |
| System Metrics | §14.3 | `---` | |
| Agent Metrics | §14.3 | `---` | |
| Collective Metrics | §14.3 | `---` | |
| Drift Detection | §14.4 | `---` | |
| Dataset Family A (Observations) | §8.2 | `---` | |
| Dataset Family B (Features) | §8.3 | `---` | |
| Dataset Family C (Reference Strategies) | §8.4 | `---` | |
| Dataset Family D (Explanations) | §8.5 | `---` | |
| Dataset Family E (Interactions) | §8.6 | `---` | |
| Dataset Family F (Synthetic) | §8.7 | `---` | |
| Dataset Family G (Evaluation) | §8.8 | `---` | |
| Financial Evaluation | §15.2 | `---` | |
| AI Quality Evaluation | §15.3 | `---` | |
| Complexity Metrics | §15.4 | `---` | |
| Engineering Metrics | §15.5 | `---` | |
| Baseline Comparisons | §15.6 | `---` | |
| Ablation Studies | §15.7 | `---` | |
| Containerization | §16.1 | `---` | |
| Safety Layer | §7.9 | `---` | |
| Experiment Registry | §7.10 | `---` | |
| Paper Trading | §7.9 | `---` | |
| Open Source Release | §8.9 | `---` | |
| MCP Financial Calculator | §6.2 | `---` | |
| MCP Market Data | §6.2 | `---` | |
| MCP Technical Analysis | §6.2 | `---` | |
| MCP Risk Analytics | §6.2 | `---` | |
| MCP Portfolio Analytics | §6.2 | `---` | |
| MCP Knowledge Graph | §6.2 | `---` | |
| MCP Retrieval | §6.2 | `---` | |
| MCP Backtesting | §6.2 | `---` | |
| MCP Observability | §6.2 | `---` | |

### David Proximity Score

At any point, compute:

```
Proximity = (COMPLETE + EXCEEDED) / Total_Sections
Coverage = (COMPLETE + EXCEEDED + PARTIAL) / Total_Sections
Intentional_Gaps = DEFERRED / Total_Sections
```

**Interpretation:**
- Proximity < 30%: Early construction phase
- Proximity 30-60%: Core system taking shape
- Proximity 60-80%: System functional, refinement phase
- Proximity > 80%: Approaching the David
- Coverage > 90% with low Proximity: Lots of partial work, may need focusing

---

## Learning Progress Dashboard

### Summary by Domain

| Domain | Topics | Not Started | Exploring | Practiced | Can Teach |
|---|---|---|---|---|---|
| RAG Engineering | 4 | 4 | 0 | 0 | 0 |
| LLM Engineering | 3 | 3 | 0 | 0 | 0 |
| Multi-Agent Systems | 3 | 3 | 0 | 0 | 0 |
| Fine-Tuning | 2 | 2 | 0 | 0 | 0 |
| Knowledge Graphs & GraphRAG | 2 | 2 | 0 | 0 | 0 |
| Observability & MLOps | 4 | 4 | 0 | 0 | 0 |
| Financial Engineering | 3 | 3 | 0 | 0 | 0 |
| Complexity Science | 3 | 3 | 0 | 0 | 0 |
| MCP & Tool Design | 2 | 2 | 0 | 0 | 0 |
| Software Architecture | 2 | 2 | 0 | 0 | 0 |
| Classical ML & Deep Learning | 2 | 2 | 0 | 0 | 0 |
| Security & Ethics | 2 | 2 | 0 | 0 | 0 |
| Professional & Leadership | 4 | 4 | 0 | 0 | 0 |
| Numbers | 5 | 5 | 0 | 0 | 0 |
| **TOTAL** | **41** | **41** | **0** | **0** | **0** |

### Career Readiness Threshold

**Minimum for Senior AI/ML Architect role:**
- 0 topics at "Not Started"
- Maximum 5 topics at "Exploring"
- At least 30 topics at "Practiced" or higher
- At least 5 topics at "Can Teach"
- All "Numbers" sections filled with real values from HiFi

---

*This document grows with the project. As each HiFi component is built, update the readiness level for the corresponding topics and fill in the numbers. The gap between your current readiness and the target readiness is your learning backlog.*
