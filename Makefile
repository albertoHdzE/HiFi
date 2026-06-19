DOCKER_COMPOSE_FILE := docker/langfuse/docker-compose.yml
DOCKER_ENV_FILE     := docker/langfuse/.env

.PHONY: help install test lint lint-fix \
	langfuse-setup langfuse-start langfuse-stop langfuse-restart \
	langfuse-clean langfuse-status langfuse-logs langfuse-seed \
	sec-fixtures acquire-data acquire-data-phase10 \
	baseline-phase3 baseline-phase4 baseline-phase5 baseline-phase6 baseline-phase7 \
	baseline-phase8 bootstrap-phase9 baseline-phase9 \
	bootstrap baseline-phase10 \
	finetune-setup finetune-train finetune-serve finetune-stop \
	generate-reference-strategies label-outcomes baseline-phase11 \
	test-live \
	calibrate-drift verification-baseline-p13 diagnose-sentiment-sgr \
	eval-debate-multiround eval-memory run-scenarios validate-sentiment-corpus \
	acquire-data-phase14 ingest-edgar-mda acquire-macro-phase14 \
	validate-sentiment-corpus-v2 eval-reset eval-ingest-through live-reset \
	walkforward-full walkforward-parallel walkforward-homogeneous walkforward-no-memory \
	walkforward-held-out walkforward-status walkforward-ic

FINETUNE_VENV := venvs/finetune/bin/python

.DEFAULT_GOAL := help

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-26s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

install: ## Install all dependencies including dev extras
	uv sync --extra dev

# ---------------------------------------------------------------------------
# Quality — deterministic tests only (no live deps)
# ---------------------------------------------------------------------------

test: ## Run the full deterministic test suite (no live services required)
	uv run pytest -q --tb=short

lint: ## Check code style with ruff
	uv run ruff check src/ tests/ scripts/

lint-fix: ## Fix auto-fixable style issues with ruff
	uv run ruff check --fix src/ tests/ scripts/

# ---------------------------------------------------------------------------
# LangFuse infrastructure
# ---------------------------------------------------------------------------

langfuse-setup: ## Copy .env.example to .env if absent; print next steps
	@if [ ! -f $(DOCKER_ENV_FILE) ]; then \
		cp docker/langfuse/.env.example $(DOCKER_ENV_FILE); \
		echo "Created $(DOCKER_ENV_FILE). Edit secrets if needed for production."; \
	else \
		echo "$(DOCKER_ENV_FILE) already exists, skipping copy."; \
	fi
	@echo "Next: make langfuse-start, then open http://localhost:3000"

langfuse-start: ## Start LangFuse Docker stack (detached)
	@if [ ! -f $(DOCKER_ENV_FILE) ]; then \
		echo "ERROR: $(DOCKER_ENV_FILE) not found. Run: make langfuse-setup"; \
		exit 1; \
	fi
	docker compose -f $(DOCKER_COMPOSE_FILE) --env-file $(DOCKER_ENV_FILE) up -d

langfuse-stop: ## Stop LangFuse Docker stack
	docker compose -f $(DOCKER_COMPOSE_FILE) down

langfuse-restart: langfuse-stop langfuse-start ## Restart LangFuse Docker stack

langfuse-clean: ## Remove LangFuse stack and all data volumes (full wipe)
	docker compose -f $(DOCKER_COMPOSE_FILE) down -v

langfuse-status: ## Show LangFuse container status
	docker compose -f $(DOCKER_COMPOSE_FILE) ps

langfuse-logs: ## Follow LangFuse web service logs
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f langfuse-web

langfuse-seed: ## Seed LangFuse with Phase 6 baseline traces (requires live instance)
	uv run python scripts/check_env.py --check langfuse
	uv run python scripts/run_phase6_tracing.py

# ---------------------------------------------------------------------------
# SEC fixtures
# ---------------------------------------------------------------------------

sec-fixtures: ## Record SEC EDGAR fixtures for Phase 7 tests (requires internet, idempotent)
	uv run python scripts/record_sec_fixtures.py

acquire-data: ## Acquire Phase 1 market + macro Parquet files (idempotent, skips existing)
	uv run python scripts/acquire_phase1_data.py

acquire-data-phase10: ## Acquire Phase 10 market Parquet files for 12 new tickers (idempotent)
	uv run python scripts/acquire_phase10_data.py

# ---------------------------------------------------------------------------
# Baseline generation + closed-loop validation
#
# Policy: each baseline target runs the complete validation loop:
#   1. Prerequisite check (env, live services, upstream fixtures)
#   2. Baseline script (live run produces fixture)
#   3. Unit fixture tests (validate fixture schema and invariants)
#   4. Holistic pipeline tests (validate full system structure in same env)
#
# This closes the feedback loop between live execution and test validation
# in a single command, ensuring the environment that generated the fixture
# is also the environment that validated it.
# ---------------------------------------------------------------------------

baseline-phase3: ## Phase 3 fundamental agent: generate + validate (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase3_baseline.py
	uv run pytest tests/unit/test_phase3_baseline.py \
	             tests/holistic/test_phase3_agent_pipeline.py \
	             -q --tb=short

baseline-phase4: ## Phase 4 ensemble: generate + validate (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase4_ensemble.py
	uv run pytest tests/unit/test_phase4_baseline.py \
	             tests/holistic/test_phase4_ensemble_pipeline.py \
	             -q --tb=short

baseline-phase5: ## Phase 5 verification: generate + validate (no live deps)
	uv run python scripts/run_phase5_verification.py
	uv run pytest tests/unit/test_phase5_baseline.py \
	             tests/holistic/test_phase5_verification_pipeline.py \
	             -q --tb=short

baseline-phase6: ## Phase 6 tracing: seed LangFuse + validate (requires LangFuse)
	uv run python scripts/check_env.py --check langfuse
	uv run python scripts/run_phase6_tracing.py
	uv run pytest tests/holistic/test_phase6_observability_pipeline.py \
	             -q --tb=short

baseline-phase7: ## Phase 7 RAG: generate + validate (requires internet + LM Studio)
	uv run python scripts/check_env.py --check sec-fixtures || $(MAKE) sec-fixtures
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase7_rag_baseline.py
	uv run pytest tests/unit/test_phase7_rag_baseline.py \
	             tests/holistic/test_phase7_rag_pipeline.py \
	             -q --tb=short

baseline-phase8: ## Phase 8 agent population: generate + validate (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase8_baseline.py
	uv run pytest tests/holistic/test_phase8_agent_population.py \
	             -q --tb=short

bootstrap-phase9: ## Phase 9 performance bootstrap: seed history from 20 quarter-ends (no LM Studio)
	uv run python scripts/check_env.py --check market-data || $(MAKE) acquire-data
	uv run python scripts/run_phase9_bootstrap.py
	uv run python scripts/check_env.py --check phase9-bootstrap

baseline-phase9: ## Phase 9 collective engine: generate + validate (requires LM Studio + bootstrap)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/check_env.py --check phase9-bootstrap || $(MAKE) bootstrap-phase9
	uv run python scripts/run_phase9_baseline.py
	uv run pytest tests/unit/test_phase9_baseline.py \
	             tests/holistic/test_phase9_collective_engine.py \
	             -q --tb=short

bootstrap: ## Phase 10 bootstrap: 15-ticker performance history seed (no LM Studio required)
	uv run python scripts/check_env.py --check market-data || $(MAKE) acquire-data
	uv run python scripts/check_env.py --check phase10-data || $(MAKE) acquire-data-phase10
	uv run python scripts/run_phase10_bootstrap.py
	uv run python scripts/check_env.py --check phase10-bootstrap

baseline-phase10: ## Phase 10 accuracy labeling + tear sheets: generate + validate (no LM Studio)
	uv run python scripts/check_env.py --check phase9-fixture || { \
		echo "Phase 9 fixture required. Run: make baseline-phase9"; exit 1; }
	uv run python scripts/check_env.py --check market-data || $(MAKE) acquire-data
	uv run python scripts/run_phase10_baseline.py
	uv run pytest tests/unit/test_phase10_baseline.py \
	             tests/holistic/test_phase10_evaluation.py \
	             -q --tb=short

# ---------------------------------------------------------------------------
# Full live validation — all baselines in sequence, complete test suite at end
#
# Runs phases 3-5, 7-9 (Phase 6 requires LangFuse, a separate infrastructure
# prerequisite — run `make baseline-phase6` independently when LangFuse is up).
# ---------------------------------------------------------------------------

test-live: ## Full live validation: all baselines + complete test suite (requires LM Studio + internet)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/check_env.py --check market-data || $(MAKE) acquire-data
	$(MAKE) baseline-phase3
	$(MAKE) baseline-phase4
	$(MAKE) baseline-phase5
	$(MAKE) sec-fixtures
	$(MAKE) baseline-phase7
	$(MAKE) baseline-phase8
	$(MAKE) baseline-phase9
	$(MAKE) baseline-phase10
	$(MAKE) label-outcomes
	uv run pytest -q --tb=short

# ---------------------------------------------------------------------------
# Fine-tuning infrastructure (Phase 11, DJ-056, DJ-057, DJ-059, DJ-060)
# ---------------------------------------------------------------------------

finetune-setup: ## Create venvs/finetune/ with pinned mlx+mlx-lm (idempotent)
	bash scripts/setup_finetune_venv.sh
	uv run python scripts/check_env.py --check finetune-venv

finetune-train: ## Run LoRA fine-tuning for both agents (requires finetune-setup + training data)
	uv run python scripts/check_env.py --check finetune-venv
	uv run python scripts/check_env.py --check phase11-data || { \
		echo "Generate training data first: make generate-reference-strategies"; exit 1; }
	uv run python scripts/run_phase11_finetune.py

finetune-serve: ## Start mlx_lm.server for fine-tuned models on ports 1235/1236 (background)
	uv run python scripts/check_env.py --check phase11-adapters || { \
		echo "Train first: make finetune-train"; exit 1; }
	bash scripts/serve_finetune_models.sh

finetune-stop: ## Stop mlx_lm.server instances
	pkill -f "mlx_lm.server" 2>/dev/null || true

generate-reference-strategies: ## Generate Dataset Family C Parquets (no LM Studio required)
	uv run python scripts/check_env.py --check market-data || $(MAKE) acquire-data
	uv run python scripts/check_env.py --check phase10-data || $(MAKE) acquire-data-phase10
	uv run python scripts/generate_reference_strategies.py
	uv run python scripts/check_env.py --check phase11-data

label-outcomes: ## Label unlabeled performance + episodic records where 60d has elapsed (no LM Studio)
	uv run python scripts/run_label_outcomes.py
	uv run python scripts/label_outcomes.py

baseline-phase11: ## Phase 11 fine-tuning eval: generate + validate (requires LM Studio + finetune servers)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/check_env.py --check finetune-venv
	uv run python scripts/check_env.py --check phase11-adapters || $(MAKE) finetune-train
	$(MAKE) finetune-serve
	sleep 15
	uv run python scripts/run_phase11_evaluation.py
	$(MAKE) finetune-stop
	uv run pytest tests/unit/test_phase11_baseline.py \
	             tests/holistic/test_phase11_evaluation.py \
	             -q --tb=short


# ---------------------------------------------------------------------------
# Phase 12: GraphRAG + Structured Debate (DJ-062, DJ-065, DJ-067)
# ---------------------------------------------------------------------------

build-graph: ## Build financial knowledge graph for GraphRAG (Phase 12, no LM Studio required)
	uv run python scripts/build_knowledge_graph.py

graphrag-eval: ## Precision@k: plain RAG vs graph-expanded retrieval (requires LM Studio + built graph)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase12_graphrag_eval.py

eval-phase12: ## Full 2x2 factorial evaluation: 10 dates x 3 tickers x 4 conditions (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	$(MAKE) build-graph
	uv run python scripts/run_phase12_evaluation.py

baseline-phase12: ## Phase 12 baseline: build graph + 1-date pilot run + unit tests (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	$(MAKE) build-graph
	uv run python scripts/run_phase12_baseline.py
	uv run pytest tests/unit/test_graph_store.py \
	             tests/unit/test_graph_construction.py \
	             tests/unit/test_graph_retrieval.py \
	             tests/unit/test_debate_schemas.py \
	             tests/unit/test_debate_nodes.py \
	             tests/unit/test_run_debate.py \
	             -q --tb=short

# ---------------------------------------------------------------------------
# Phase 13: Verification Completeness, Sentiment Intelligence, Resilience
# ---------------------------------------------------------------------------

calibrate-drift: ## E5-T5: Calibrate drift monitors on 2022 rate-shock regime (no LM Studio)
	uv run python scripts/check_env.py --check market-data || $(MAKE) acquire-data
	uv run python scripts/calibrate_drift_monitors.py

verification-baseline-p13: ## E0-T6: Run Phase 13 verification baseline for Risk/Macro/Sentiment (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase13_verification_baseline.py

diagnose-sentiment-sgr: ## DJ-086: Diagnose Gemma 4 E4B / 12B-it SGR failure (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/diagnose_sentiment_sgr.py --all-tickers

validate-sentiment-corpus: ## E1-T1: Validate Phase 7 EDGAR corpus for Sentiment FT gate (requires LanceDB)
	uv run python scripts/validate_sentiment_corpus.py

eval-debate-multiround: ## E2-T4: Multi-round debate eval → OQ-D04 (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase13_debate_eval.py

eval-memory: ## E4-T4: Agent memory influence eval → OQ-M03 (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase13_memory_eval.py

run-scenarios: ## E6-T2: Run F-001/F-002/F-003 synthetic scenarios (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase13_scenarios.py

# ---------------------------------------------------------------------------
# Phase 14: Data acquisition, namespace management, episodic labeling (DJ-090–DJ-093)
# ---------------------------------------------------------------------------

acquire-data-phase14: ## Bulk OHLCV + fundamentals for 100-stock universe 2004-2025 (internet, ~45min)
	uv run python scripts/acquire_phase14_data.py

ingest-edgar-mda: ## EDGAR MD&A Item 7/Item 2 targeted ingestion → LanceDB (internet, 4-8h)
	uv run python scripts/ingest_edgar_mda.py

acquire-macro-phase14: ## Extend FRED macro indicators 2004-2025 (internet, ~5min)
	uv run python scripts/acquire_macro_phase14.py

validate-sentiment-corpus-v2: ## Re-run OQ-S01 corpus gate on expanded EDGAR corpus (E1-T1)
	uv run python scripts/validate_sentiment_corpus.py

eval-reset: ## Drop all hifi-eval-* namespace tables in LanceDB
	uv run python scripts/manage_namespaces.py --action reset --namespace hifi-eval

eval-ingest-through: ## Ingest data through DATE= into hifi-eval namespace (requires DATE=)
	@if [ -z "$(DATE)" ]; then \
		echo "ERROR: DATE= required (e.g. make eval-ingest-through DATE=2020-12-31)"; exit 1; \
	fi
	uv run python scripts/ingest_edgar_mda.py --namespace hifi-eval --through-date $(DATE)
	uv run python scripts/ingest_episodes.py --namespace hifi-eval --through-date $(DATE)

live-reset: ## Drop all hifi-live-* namespace tables in LanceDB
	uv run python scripts/manage_namespaces.py --action reset --namespace hifi-live

# ---------------------------------------------------------------------------
# Phase 15: Walk-Forward Simulation (DJ-097, DJ-096)
# ---------------------------------------------------------------------------

walkforward-full: ## Phase 15 Full: sequential 5-org + episodic RAG (requires LM Studio)
	uv run python scripts/run_phase15_walkforward.py \
		--condition full --period held-out-test

walkforward-parallel: ## Phase 15 Parallel: independent 5-org, no inter-agent sharing (requires LM Studio)
	uv run python scripts/run_phase15_walkforward.py \
		--condition parallel --period held-out-test

walkforward-homogeneous: ## Phase 15 Homogeneous: Phase 13 qwen-dominant config (requires LM Studio)
	uv run python scripts/run_phase15_walkforward.py \
		--condition homogeneous --period held-out-test

walkforward-no-memory: ## Phase 15 No-memory: sequential 5-org, no episodic prefix (requires LM Studio)
	uv run python scripts/run_phase15_walkforward.py \
		--condition no-memory --period held-out-test

walkforward-held-out: ## Phase 15 all four conditions on held-out 2022-2023 (requires LM Studio)
	$(MAKE) walkforward-full
	$(MAKE) walkforward-parallel
	$(MAKE) walkforward-homogeneous
	$(MAKE) walkforward-no-memory

walkforward-status: ## Show checkpoint progress for all conditions (no LM Studio needed)
	@for cond in full parallel homogeneous no-memory; do \
		uv run python scripts/run_phase15_walkforward.py \
			--status --condition $$cond --period held-out-test; \
	done

walkforward-ic: ## Compute IC/IR metrics from completed walkforward JSONs (no LM Studio needed)
	uv run python scripts/compute_phase15_ic.py \
		--period held-out-test --regime-breakdown

