#!/usr/bin/env bash
# scripts/setup_finetune_venv.sh
#
# Create and populate the venvs/finetune/ virtual environment for Phase 11
# LoRA fine-tuning (DJ-056).
#
# Isolates mlx and mlx-lm from the main HiFi project environment. All
# fine-tuning scripts are invoked via venvs/finetune/bin/python.
#
# Prerequisites:
#   - uv installed (https://docs.astral.sh/uv/)
#   - Python 3.13 available (uv will fetch it if not present)
#
# Usage:
#   bash scripts/setup_finetune_venv.sh
#
# After setup, verify:
#   venvs/finetune/bin/python -c "import mlx_lm; print(mlx_lm.__version__)"
#
# Phase 15 note: at containerization, this venv becomes a separate training
# Docker service image. The setup script is superseded by a Dockerfile.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/venvs/finetune"

echo "Creating venvs/finetune/ with Python 3.13..."
uv venv "${VENV_DIR}" --python 3.13 --seed --clear

echo "Installing mlx and mlx-lm (pinned to tested versions)..."
"${VENV_DIR}/bin/pip" install --quiet \
    "mlx==0.31.1" \
    "mlx-lm==0.31.1"

echo "Verifying installation..."
"${VENV_DIR}/bin/python" -c "import mlx_lm; print(f'mlx_lm {mlx_lm.__version__} OK')"
echo "venvs/finetune/ ready."
