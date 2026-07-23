#!/usr/bin/env bash
# Nightly Phase 16 live paper trading cycle (DJ-111).
# Launched by launchd (com.hifi.live-execute) every day at 19:00 local.
#
# Pre-flight: verifies LM Studio, fine-tune servers, and Docker/LangFuse.
# Skips weekends (no market close data worth trading on).

set -uo pipefail

REPO="/Users/alberto/Documents/projects/HiFi"
UV="/Users/alberto/.local/bin/uv"
LOG_DIR="${REPO}/data/live/logs"
LOG="${LOG_DIR}/nightly_$(date +%Y%m%d).log"
mkdir -p "${LOG_DIR}"

exec >> "${LOG}" 2>&1
echo "=== nightly_live_execute $(date '+%Y-%m-%d %H:%M:%S') ==="

# Weekend guard: 6=Saturday 7=Sunday
dow=$(date +%u)
if [ "${dow}" -ge 6 ]; then
    echo "Weekend (dow=${dow}) — skipping."
    exit 0
fi

cd "${REPO}"

# Pre-flight: fine-tune servers (launchd keeps them alive; wait up to 3 min)
for i in $(seq 1 18); do
    ok1=$(curl -s -m 3 http://localhost:1235/health | grep -c ok || true)
    ok2=$(curl -s -m 3 http://localhost:1236/health | grep -c ok || true)
    [ "${ok1}" = "1" ] && [ "${ok2}" = "1" ] && break
    echo "waiting for fine-tune servers (${i}/18)..."
    sleep 10
done

# Pre-flight: LangFuse. START it (not just check) so AI telemetry is captured;
# on nights it was down we silently lost prompts/tokens/latency. Fail-open:
# if it never comes up, the run still proceeds (sidecars are the durable record).
if ! curl -s -m 5 http://localhost:3000/api/public/health >/dev/null; then
    echo "LangFuse down — starting stack..."
    docker info >/dev/null 2>&1 || open -a Docker
    for i in $(seq 1 24); do docker info >/dev/null 2>&1 && break; sleep 5; done
    docker compose -f docker/langfuse/docker-compose.yml \
        --env-file docker/langfuse/.env up -d >/dev/null 2>&1 || true
    for i in $(seq 1 24); do
        curl -s -m 3 http://localhost:3000/api/public/health >/dev/null 2>&1 && break
        echo "waiting for LangFuse (${i}/24)..."; sleep 5
    done
fi
curl -s -m 5 http://localhost:3000/api/public/health >/dev/null \
    && echo "LangFuse up — tracing enabled" \
    || echo "WARNING: LangFuse still down; run proceeds, AI telemetry to sidecars only"

"${UV}" run python scripts/run_phase16_live.py --account all --execute
rc=$?
echo "=== finished rc=${rc} $(date '+%Y-%m-%d %H:%M:%S') ==="
exit ${rc}
