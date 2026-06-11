DOCKER_COMPOSE_FILE := docker/langfuse/docker-compose.yml
DOCKER_ENV_FILE     := docker/langfuse/.env

.PHONY: help install test lint lint-fix \
	langfuse-setup langfuse-start langfuse-stop langfuse-restart \
	langfuse-clean langfuse-status langfuse-logs langfuse-seed \
	baseline-phase3 baseline-phase4 baseline-phase5 baseline-phase6 baseline-phase7

.DEFAULT_GOAL := help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

install: ## Install all dependencies including dev extras
	uv sync --extra dev

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test: ## Run the full test suite
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

langfuse-seed: ## Seed LangFuse with Phase 5 baseline traces (requires live instance)
	uv run python scripts/check_env.py --check langfuse
	uv run python scripts/run_phase6_tracing.py

# ---------------------------------------------------------------------------
# Baseline generation
# ---------------------------------------------------------------------------

baseline-phase3: ## Generate Phase 3 fundamental agent baseline (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase3_baseline.py

baseline-phase4: ## Generate Phase 4 ensemble baseline (requires LM Studio)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase4_ensemble.py

baseline-phase5: ## Generate Phase 5 verification baseline (no live deps)
	uv run python scripts/run_phase5_verification.py

baseline-phase6: ## Seed LangFuse with Phase 6 tracing baseline (requires live instance)
	uv run python scripts/check_env.py --check langfuse
	uv run python scripts/run_phase6_tracing.py

baseline-phase7: ## Generate Phase 7 RAG baseline (requires LM Studio + knowledge store)
	uv run python scripts/check_env.py --check lm-studio
	uv run python scripts/run_phase7_rag_baseline.py
