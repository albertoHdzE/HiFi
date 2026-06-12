DOCKER_COMPOSE_FILE := docker/langfuse/docker-compose.yml
DOCKER_ENV_FILE     := docker/langfuse/.env

.PHONY: help install test lint lint-fix \
	langfuse-setup langfuse-start langfuse-stop langfuse-restart \
	langfuse-clean langfuse-status langfuse-logs langfuse-seed \
	sec-fixtures acquire-data \
	baseline-phase3 baseline-phase4 baseline-phase5 baseline-phase6 baseline-phase7 \
	baseline-phase8 bootstrap-phase9 baseline-phase9 \
	test-live

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
	$(MAKE) bootstrap-phase9
	$(MAKE) baseline-phase9
	uv run pytest -q --tb=short
