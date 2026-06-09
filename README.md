# HiFi: High-Fidelity Financial Intelligence

A fully local, open-source, multi-agent financial intelligence platform with verifiable decision making and deterministic financial reasoning.

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

## Project Documentation

- `doc/HIFI_DAVID.md` -- Aspirational reference specification (the "David")
- `doc/HIFI_PROTOCOL_V1.md` -- Execution protocol (18 phases)
- `doc/HIFI_LEARNING_GUIDE.md` -- Learning roadmap and David proximity tracker
- `doc/bitacora/` -- Per-phase scientific logbook
