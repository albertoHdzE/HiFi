#!/usr/bin/env bash
# Nightly Phase 16 live paper trading cycle (DJ-111).
# Launched by launchd (com.hifi.live-execute) every day at 19:00 local.
#
# Pre-flight: verifies LM Studio, fine-tune servers, and Docker/LangFuse.
# Skips weekends and refuses to start during the US cash session (see below).
#
# Usage:
#   nightly_live_execute.sh                     normal run
#   nightly_live_execute.sh --check-window      print window verdict, exit 0/1, run nothing
#   nightly_live_execute.sh --allow-market-hours  run anyway (also: ALLOW_MARKET_HOURS=1)

set -uo pipefail

REPO="/Users/alberto/Documents/projects/HiFi"
UV="/Users/alberto/.local/bin/uv"
LOG_DIR="${REPO}/data/live/logs"
LOG="${LOG_DIR}/nightly_$(date +%Y%m%d).log"
mkdir -p "${LOG_DIR}"

ALLOW_MARKET_HOURS="${ALLOW_MARKET_HOURS:-0}"
CHECK_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --allow-market-hours) ALLOW_MARKET_HOURS=1 ;;
        --check-window)       CHECK_ONLY=1 ;;
        *) echo "unknown argument: ${arg}" >&2; exit 64 ;;
    esac
done

# Timing guard (DJ-118). The experiment's protocol is: decide on completed
# closes in the evening, orders fill at the NEXT open. A cycle takes ~5-6.5 h,
# so *when* it is launched determines whether that holds. Two failure modes:
#
#   1. Launched inside the cash session (09:30-16:00 ET) -> the last OHLCV bar
#      is a live partial and orders fill intraday, not at an open. Blocked.
#   2. Launched pre-market -> starts clean but finishes mid-session, so the
#      DAY orders fill intraday. Warned, not blocked (the decision inputs are
#      still complete closes, which is the part that matters scientifically).
#
# All arithmetic is in ET because that is what Alpaca and the bar timestamps
# use; local time is CST, which crosses midnight relative to ET. Note this is
# a clock guard only — it does not know US market holidays.
RUN_HOURS=6
market_window_check() {
    local et_dow et_hm now_min open_min close_min finish_min
    et_dow=$(TZ=America/New_York date +%u)
    et_hm=$(TZ=America/New_York date +%H%M)
    now_min=$(( 10#${et_hm:0:2} * 60 + 10#${et_hm:2:2} ))
    open_min=$(( 9 * 60 + 30 ))
    close_min=$(( 16 * 60 ))

    if [ "${et_dow}" -ge 6 ]; then
        echo "Weekend in ET (dow=${et_dow}) — no session to trade into."
        return 2
    fi
    if [ "${now_min}" -ge "${open_min}" ] && [ "${now_min}" -lt "${close_min}" ]; then
        echo "REFUSING: market is OPEN (${et_hm} ET). Decisions would read a partial"
        echo "bar and orders would fill intraday instead of at the next open."
        echo "Run tonight after 16:00 ET, or override: ALLOW_MARKET_HOURS=1 make live-nightly"
        return 1
    fi
    # Pre-market launch that would still be running when the bell rings.
    if [ "${now_min}" -lt "${open_min}" ]; then
        finish_min=$(( now_min + RUN_HOURS * 60 ))
        if [ "${finish_min}" -ge "${open_min}" ]; then
            echo "WARNING: launched ${et_hm} ET; a ~${RUN_HOURS} h cycle finishes after the 09:30 open,"
            echo "so orders will fill intraday rather than at the open. Proceeding."
        fi
    fi
    return 0
}

if [ "${CHECK_ONLY}" -eq 1 ]; then
    market_window_check
    exit $?
fi

exec >> "${LOG}" 2>&1
echo "=== nightly_live_execute $(date '+%Y-%m-%d %H:%M:%S') ==="

market_window_check
window_rc=$?
if [ "${window_rc}" -eq 2 ]; then
    echo "Weekend — skipping."
    exit 0
fi
if [ "${window_rc}" -eq 1 ]; then
    if [ "${ALLOW_MARKET_HOURS}" = "1" ]; then
        echo "ALLOW_MARKET_HOURS=1 — proceeding anyway; annotate this date as off-protocol."
    else
        exit 75
    fi
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
