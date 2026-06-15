#!/usr/bin/env bash
# scripts/serve_finetune_models.sh
#
# Start mlx_lm.server for fine-tuned Technical and Fundamental agents (DJ-057).
#
# Technical Agent: port 1235 (base + technical_v2 adapter, DJ-082)
# Fundamental Agent: port 1236 (base + fundamental_v1 adapter)
# Both run alongside LM Studio (port 1234).
#
# Requires:
#   - venvs/finetune/ created: make finetune-setup
#   - Adapters trained: make finetune-train
#   - Base model downloaded (fetched from HuggingFace on first run)
#
# Usage:
#   bash scripts/serve_finetune_models.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/venvs/finetune"
# Use local LM Studio copy (avoids re-downloading 32GB; DJ-056)
MODEL_BASE="${HOME}/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-8bit"
ADAPTERS_DIR="${REPO_ROOT}/data/adapters"

if [ ! -f "${VENV}/bin/python" ]; then
    echo "ERROR: venvs/finetune/ not found. Run: make finetune-setup"
    exit 1
fi

if [ ! -d "${ADAPTERS_DIR}/technical_v2" ] || [ ! -d "${ADAPTERS_DIR}/fundamental_v1" ]; then
    echo "ERROR: Adapters not found in ${ADAPTERS_DIR}/. Run: make finetune-train"
    exit 1
fi

echo "Starting mlx_lm.server for Technical Agent (port 1235)..."
"${VENV}/bin/python" -m mlx_lm server \
    --model "${MODEL_BASE}" \
    --adapter-path "${ADAPTERS_DIR}/technical_v2" \
    --port 1235 \
    --log-level WARNING &
echo "Technical fine-tuned server PID: $!"

echo "Starting mlx_lm.server for Fundamental Agent (port 1236)..."
"${VENV}/bin/python" -m mlx_lm server \
    --model "${MODEL_BASE}" \
    --adapter-path "${ADAPTERS_DIR}/fundamental_v1" \
    --port 1236 \
    --log-level WARNING &
echo "Fundamental fine-tuned server PID: $!"

echo "Servers starting. Health check in 10s..."
sleep 10
curl -s http://localhost:1235/health >/dev/null && echo "Port 1235 OK" || echo "Port 1235 NOT READY"
curl -s http://localhost:1236/health >/dev/null && echo "Port 1236 OK" || echo "Port 1236 NOT READY"
