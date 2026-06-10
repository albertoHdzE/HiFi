#!/usr/bin/env bash
# scripts/setup_ta_venv.sh
#
# Create and populate the venvs/ta/ virtual environment for the Phase 8+
# Technical Indicators MCP server (DJ-010).
#
# This venv pins pandas==1.5.3 (pandas-ta requires pandas < 2.0) and runs
# the indicators server as an isolated subprocess. The main HiFi environment
# never imports from this venv; communication is via stdio MCP JSON-RPC.
#
# Prerequisites:
#   - uv installed (https://docs.astral.sh/uv/)
#   - Python 3.11 available (uv will fetch it if not present)
#
# Usage:
#   bash scripts/setup_ta_venv.sh
#
# After setup, verify:
#   venvs/ta/bin/python -c "import pandas_ta; print(pandas_ta.__version__)"
#
# Phase 15 note: at containerization, this venv becomes a separate Docker
# service image. The setup script is superseded by a Dockerfile in that phase.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/venvs/ta"
REQUIREMENTS="${VENV_DIR}/requirements.txt"

echo "Creating venvs/ta with Python 3.11..."
uv venv "${VENV_DIR}" --python 3.11

echo "Installing pinned dependencies from ${REQUIREMENTS}..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install -r "${REQUIREMENTS}" --quiet

echo "Verifying installation..."
"${VENV_DIR}/bin/python" -c "
import pandas as pd
import pandas_ta as ta
print(f'pandas version: {pd.__version__}')
print(f'pandas-ta version: {ta.version}')
print('venvs/ta setup complete')
"
