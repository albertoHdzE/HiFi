# Bitacora: Phase 0 -- Project Infrastructure

## Objective

Set up the development environment, repository structure, tooling, and conventions that every subsequent phase depends on.

## David Sections Addressed

- Section 4.5: Reproducibility
- Section 4.6: Modularity
- Section 7.10: Experiment Registry (foundation)

## Decisions Made

### DJ-006: Python environment manager

- **Decision:** uv
- **Rationale:** uv is the fastest Python package manager available (10-100x faster than pip). It supports lockfiles for reproducibility, handles virtual environments transparently, and has native Apple Silicon support. It is actively maintained by the Astral team (same as ruff).
- **Alternatives considered:** poetry (slower resolution, heavier), conda (overkill for this project, mixing package managers is risky)
- **Status:** Accepted

### DJ-007: Storage format (Phase 1 preparation)

- **Decision:** Parquet (initial)
- **Rationale:** Parquet is columnar, compressed, and fast for analytical queries. It is the standard for financial time-series storage in Python. DuckDB can read Parquet natively if we need SQL later. SQLite is row-oriented and less efficient for time-series workloads.
- **Status:** Accepted (may revisit if query patterns change)

## Insights

- Python 3.13 is available via pyenv, but uv selected Python 3.12.13 for the venv. This is fine -- 3.12 has broader library compatibility. The pyproject.toml requires >=3.11.
- M3 Ultra confirmed as development hardware -- aligns with the David's "Ideal" tier (128-192GB RAM, parallel inference possible).
- uv handles the virtual environment and dependency resolution in a single tool, reducing tooling complexity. Install of all dependencies completed in under 2 seconds.
- ruff auto-fixed import sorting and modernized type annotations (Optional[X] to X | None). This confirms ruff is correctly configured and enforcing modern Python style.
- Pydantic validation catches configuration errors at load time, not at use time. This is the "hard constraint" layer -- if the config is invalid, the system refuses to start.
- Synthetic OHLCV generation using geometric Brownian motion produces realistic price dynamics. The fixture guarantees: high >= max(open, close) and low <= min(open, close) for every bar. This invariant is tested.

## Results

- **17 tests written, 17 passing** in 0.02 seconds
- **0 lint errors** after ruff auto-fix
- **Epic plan:** plans/PHASE_00_PLAN.md (4 epics, 20 tickets, all DONE)
- **Context handoff:** plans/STATUS.md created for session continuity

## Test Strategy

- pytest with markers: `unit`, `integration`, `holistic`
- Synthetic data fixtures with deterministic seeds (numpy.random.Generator with seed=42)
- No mocks. Recorded API fixtures for external data sources (to be created in Phase 1).
- OHLCV synthetic data generated using geometric Brownian motion for realistic price dynamics.
- Financial statement data generated with constrained random values preserving accounting identities.

## Structure Created

```
HiFi/
  src/hifi/           -- 12 subpackages (data, engines, mcp, knowledge, models,
                         agents, collective, verification, observability,
                         execution, evaluation, config)
  tests/              -- unit, integration, holistic directories
  configs/            -- YAML configuration files
  data/               -- gitignored data directories
  doc/bitacora/       -- phase-level scientific logbook
  docker/             -- containerization (future)
  notebooks/          -- exploration (future)
  scripts/            -- utilities (future)
```

## Open Items Carried Forward

- OQ-A04: Sequential vs. parallel agent inference on M-series -- deferred to Phase 3+
- OQ-K01: Chunking strategy -- deferred to Phase 7
