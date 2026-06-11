# LangFuse Setup Guide

LangFuse is HiFi's observability backend. It stores every `run_ensemble()` call as a
structured trace: LLM generations, MCP tool spans, verification scores, and the collective
decision. This creates the historical time series required for Phase 10 analysis.

This guide covers the full setup from zero to a visible trace in the dashboard.

## Why self-hosted

HiFi processes financial data (ticker names, numerical claims, model rationales) that must
not be transmitted to external services. LangFuse Cloud is therefore excluded regardless of
its capabilities. The self-hosted Docker Compose stack runs entirely on the local machine.

## Prerequisites

- Docker Desktop (macOS) or Docker Engine + Compose plugin (Linux)
- Ports 3000 (LangFuse UI), 5432 (Postgres), 8123/9000 (ClickHouse), 6379 (Redis)
  must be available

## Step 1: Copy the environment file

```bash
cp docker/langfuse/.env.example docker/langfuse/.env
```

The `.env` file contains local secrets and is git-ignored. You only need to change the
default values for production deployments. For local development the defaults work.

Optionally generate stronger secrets:

```bash
# NEXTAUTH_SECRET
openssl rand -base64 32

# ENCRYPTION_KEY (must be exactly 64 hex chars)
openssl rand -hex 32
```

## Step 2: Start the stack

```bash
docker compose -f docker/langfuse/docker-compose.yml --env-file docker/langfuse/.env up -d
```

First-run note: ClickHouse initialises schema migrations and takes 15-30 seconds.
Check readiness with:

```bash
docker compose -f docker/langfuse/docker-compose.yml ps
```

All five services (`langfuse-web`, `langfuse-worker`, `db`, `clickhouse`, `redis`) should
show status `running (healthy)`.

## Step 3: Create a project in the UI

1. Open `http://localhost:3000` in a browser.
2. Create an account (the first user becomes the admin).
3. Create an organisation (e.g., "HiFi").
4. Create a project (e.g., "hifi-local").
5. In the project settings, go to **API Keys** and generate a new key pair.
6. Copy the **Public Key** (starts with `pk-lf-...`) and **Secret Key** (starts with
   `sk-lf-...`).

## Step 4: Configure the Python SDK

Export the credentials in your shell session (or add to `~/.zshrc` / `~/.bashrc`):

```bash
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_ENABLED=true
```

Alternatively, update `docker/langfuse/.env` with these values and source it:

```bash
source docker/langfuse/.env
```

## Step 5: Verify connectivity

Run the Phase 6 baseline script to send three traces (AAPL, JPM, XOM) to LangFuse:

```bash
uv run python scripts/run_phase6_tracing.py
```

Open `http://localhost:3000`, navigate to your project, and confirm three traces appear
in the Traces view. Each trace should have six scores:
`fundamental_hr`, `fundamental_gr`, `technical_hr`, `technical_gr`,
`disagreement_entropy`, `n_contradictions`.

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `LANGFUSE_HOST` | `http://localhost:3000` | LangFuse server URL |
| `LANGFUSE_PUBLIC_KEY` | (required) | Project public key from UI |
| `LANGFUSE_SECRET_KEY` | (required) | Project secret key from UI |
| `LANGFUSE_ENABLED` | `true` | Set to `false` to disable all tracing |

## Running tests

All HiFi tests set `LANGFUSE_ENABLED=false` automatically via a session-scoped pytest
fixture (DJ-025). No live LangFuse instance is required to run the test suite:

```bash
uv run pytest -q
```

## Stopping the stack

```bash
docker compose -f docker/langfuse/docker-compose.yml down
```

To also remove data volumes (full reset):

```bash
docker compose -f docker/langfuse/docker-compose.yml down -v
```

## Architecture note (DJ-022)

LangFuse v3 is chosen over v2 because it includes a ClickHouse analytics backend that
makes historical queries over thousands of traces fast. The ClickHouse backend is required
for Phase 10's backtesting analysis. Migrating from v2 to v3 after data accumulates is
more disruptive than starting with v3, so v3 is adopted from Phase 6 onward.
