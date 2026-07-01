# HiFi

HiFi is a local trading research platform that combines deterministic financial computation with multi-agent language-model analysis. The codebase is organized around a separation of concerns: market, macro, filing, and risk data are processed with reproducible Python pipelines; trading-relevant quantities are computed by deterministic engines and MCP tools; local language models are used for structured interpretation, disagreement, and synthesis; and the resulting decisions are traced, evaluated, and stress-tested with historical workflows.

The project is relevant to systematic trading research because it treats model generation, risk controls, and evaluation as part of one research system rather than as separate scripts. In practice, this means the repository contains not only agent orchestration, but also data provenance, quality checks, ensemble aggregation, drift monitoring, fine-tuning infrastructure, and walk-forward simulation utilities.

## What Is Implemented

### 1. Data layer

HiFi acquires and normalizes several classes of inputs:

- market data from Yahoo Finance via `yfinance`
- macroeconomic series from FRED via `fredapi`
- targeted SEC EDGAR MD&A sections for earnings-related textual context
- stored datasets with provenance metadata and SHA-256 content hashes
- explicit quality checks for completeness, gaps, anomalous price moves, and OHLCV consistency

This part of the system is built to preserve point-in-time structure and make revisions visible rather than implicit.

### 2. Deterministic financial and risk layer

HiFi uses deterministic engines for the parts of the workflow that should not depend on language-model variability. Implemented examples include:

- historical volatility across multiple horizons
- beta estimation
- maximum drawdown
- Sharpe ratio and related tear-sheet metrics
- portfolio VaR checks
- sector exposure checks
- correlation-aware sizing logic

These computations are implemented as ordinary Python functions and MCP-exposed tools, with explicit conventions and typed outputs. The risk-manager MCP server is intentionally free of LLM calls and external network dependence during evaluation.

### 3. Multi-agent local LLM layer

On top of the deterministic layer, HiFi runs a population of specialized agents for:

- fundamental analysis
- technical analysis
- risk analysis
- macro analysis
- sentiment analysis
- contrarian review

The agents are orchestrated through LangGraph and communicate through structured state and context stores. Tool access is restricted by role, so each agent sees only the subset of information intended for that function. This is used to encourage analytical diversity instead of simply having several models restate the same input.

### 4. Collective decision layer

HiFi does not stop at single-agent outputs. It includes collective decision machinery with:

- confidence-weighted voting
- majority voting
- performance-weighted voting
- contrarian-adjusted voting
- disagreement entropy and opinion-dispersion metrics
- structured debate transcripts and revision rounds

This allows the system to compare alternative aggregation rules and inspect disagreement instead of reducing the workflow to one opaque final score.

### 5. Fine-tuning and model diversity

HiFi includes local fine-tuning infrastructure based on `mlx_lm` LoRA training. The codebase contains:

- subprocess-based LoRA training orchestration
- isolated fine-tuning environments
- adapter validation checks
- rank-sweep result handling
- explicit support for multiple local model families through LM Studio

In the current implementation, local LLMs are not treated as a single interchangeable model. The system is designed around model specialization by role and measured evaluation of those choices.

### 6. Observability, evaluation, and monitoring

HiFi includes a strong instrumentation and evaluation layer:

- self-hosted Langfuse traces for LLM generations, tool calls, scores, and final decisions
- deterministic test coverage for core components
- synthetic and historical evaluation helpers
- tear-sheet generation for strategy analysis
- drift monitors for data drift, agent drift, and collective drift
- historical walk-forward simulation scripts and smoke-test orchestration

This is important because the project is not framed as prompt engineering alone; it is framed as a research and validation environment for trading decisions.

## How Determinism Is Handled

HiFi is not "fully deterministic" in the sense that language-model text generation is always perfectly reproducible. Instead, determinism is handled by isolating stochastic and non-stochastic parts of the system.

The deterministic controls in the repository include:

- pure-function financial computations for risk and performance quantities
- explicit `as_of_date` handling for time-aware analysis
- fixed evaluation periods in configuration
- dataset hashing and provenance records
- default reproducibility seed in configuration
- local execution rather than cloud-hosted model services
- temperature `0.0` in the LM Studio client for structured JSON generation
- schema-based parsing and retry logic for agent outputs

The practical design principle is that verifiable quantities such as volatility, drawdown, VaR, correlations, and portfolio constraints are computed deterministically, while LLMs are used for structured interpretation and comparative reasoning on top of that layer.

## GenAI and Classical ML/AI in the Same System

HiFi is intentionally hybrid.

Classical quantitative and ML components include:

- market and macro data engineering
- portfolio and risk analytics
- statistical drift monitoring
- historical simulation and tear-sheet evaluation
- feature-oriented research pipelines and reproducible experiment structure

GenAI-oriented components include:

- local agent orchestration with LangGraph
- MCP tool use from agents
- local model serving through LM Studio
- retrieval over SEC MD&A text with LanceDB-backed workflows
- debate and critique between specialized agents
- LoRA-based local fine-tuning
- Langfuse tracing for agent/tool inspection

This combination makes the repository closer to a research operating system for trading workflows than to a single forecasting model.

## Technology Stack

- Python 3.11
- `pandas`, `numpy`, `scipy`, `pyarrow`
- `yfinance`, `fredapi`
- `quantstats`
- `pydantic`, `pyyaml`
- `langchain`, `langchain-openai`, `langgraph`
- `mcp`
- `langfuse`
- `lancedb`
- `networkx`
- `pytest`, `ruff`, `mypy`
- local model serving through LM Studio
- local LoRA fine-tuning with `mlx_lm`

## Repository Structure

- `src/hifi/data/` - data acquisition, quality control, provenance, regime data, storage
- `src/hifi/engines/` - deterministic financial, technical, macro, and risk computations
- `src/hifi/mcp/` - MCP servers and tool-facing deterministic logic
- `src/hifi/agents/` - specialized agents, local LLM calls, LangGraph orchestration
- `src/hifi/collective/` - voting, debate, drift, memory, and collective decision logic
- `src/hifi/models/` - local fine-tuning utilities
- `scripts/` - acquisition, tracing, labeling, simulation, and operational workflows
- `tests/` - unit, integration, and holistic validation
- `doc/` - protocol, architectural notes, and per-phase learning documents

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -e ".[dev]"
```

## Run Tests

```bash
.venv/bin/pytest tests/ -v
```

Or with the project Makefile:

```bash
make test
```

## Project Documentation

- `doc/HIFI_DAVID.md` - long-form architectural reference
- `doc/HIFI_PROTOCOL_V1.md` - staged execution protocol
- `doc/HIFI_LEARNING_GUIDE.md` - learning and capability map
- `doc/bitacora/` - per-phase scientific logbook
