# HiFi Dev Workflow

Single reference for daily development operations. All commands run from the project root.

## First-time setup

```bash
make install          # uv sync --extra dev
make langfuse-setup   # creates docker/langfuse/.env from .env.example
make langfuse-start   # starts LangFuse Docker stack (Postgres, ClickHouse, Redis)
make test             # verify all tests pass
```

After `langfuse-start`, open `http://localhost:3000`, create an org + project, and export
the API keys (see `doc/setup/LANGFUSE_SETUP.md` for the full step-by-step).

## Daily commands

| Command | What it does |
|---|---|
| `make test` | Run the full test suite (no live deps needed) |
| `make lint` | Check code style with ruff |
| `make lint-fix` | Fix auto-fixable style issues |
| `make install` | Sync dependencies after `pyproject.toml` changes |

## LangFuse operations

| Command | What it does |
|---|---|
| `make langfuse-setup` | Copy `.env.example` to `.env` if absent |
| `make langfuse-start` | Start all services (detached) |
| `make langfuse-stop` | Stop all services |
| `make langfuse-restart` | Stop then start |
| `make langfuse-status` | Show container health |
| `make langfuse-logs` | Follow web service logs |
| `make langfuse-clean` | **Full wipe** — removes all trace data (irreversible) |
| `make langfuse-seed` | Send Phase 5 baseline traces to the dashboard |

## Baseline generation

Baselines are cumulative. Generate in order; each phase reads the previous fixture.

```bash
make baseline-phase3   # fundamental agent  — requires LM Studio
make baseline-phase4   # ensemble           — requires LM Studio
make baseline-phase5   # verification       — no live deps
make baseline-phase6   # LangFuse tracing   — requires live LangFuse instance
```

Phase 3 and 4 require LM Studio running with a model loaded at
`HIFI_LM_STUDIO_URL` (default: `http://localhost:1234/v1`).

Fixtures are written to `tests/fixtures/baseline/`. They are git-tracked and unlock
the `@pytest.mark.skip`-guarded baseline tests automatically once present.

## Prerequisite checker

`scripts/check_env.py` is called automatically by Makefile targets before any script
that needs a live service. You can also invoke it directly:

```bash
uv run python scripts/check_env.py --check lm-studio
uv run python scripts/check_env.py --check langfuse
uv run python scripts/check_env.py --check phase4-fixture
uv run python scripts/check_env.py --check phase5-fixture
```

Exit code 0 = OK, 1 = missing prerequisite with a clear remediation message.

## Key environment variables

| Variable | Default | Used by |
|---|---|---|
| `HIFI_LM_STUDIO_URL` | `http://localhost:1234/v1` | Phase 3, 4 baselines |
| `LANGFUSE_HOST` | `http://localhost:3000` | Phase 6 tracing |
| `LANGFUSE_PUBLIC_KEY` | (required) | Phase 6 tracing |
| `LANGFUSE_SECRET_KEY` | (required) | Phase 6 tracing |
| `LANGFUSE_ENABLED` | `true` | All tracing; auto-set to `false` in tests |

## Test markers

```bash
uv run pytest -m unit          # fast unit tests only
uv run pytest -m integration   # integration tests
uv run pytest -m holistic      # full pipeline tests
uv run pytest -q               # all tests, quiet output
```

All tests set `LANGFUSE_ENABLED=false` via a session-scoped autouse fixture — no live
LangFuse instance is required to run the suite.
