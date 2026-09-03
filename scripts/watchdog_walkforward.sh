#!/usr/bin/env bash
# HiFi Walk-Forward Watchdog
# Monitors the orchestrator, restarts on crash, advances through all 4 conditions.
#
# Invoked by hand or from a scheduler; it is idempotent and safe to re-run every
# few minutes. It is NOT currently scheduled — `crontab -l` is empty — so a sweep
# advances only while something is calling this.
#
# Conditions run in sequence (agent-first, checkpoint-resume):
#   full → no-memory → parallel → homogeneous
# After all 4: computes IC/IR/herding metrics.

set -uo pipefail

# Derived, not hard-coded: a second checkout would otherwise drive the sweep in
# one tree while writing sidecars into the other.
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$PROJECT/data"
LOGFILE="/tmp/hifi_watchdog.log"
PERIOD="held-out-test"
CONDITIONS=("full" "no-memory" "parallel" "homogeneous")
EXPECTED_PORTFOLIOS=24   # 24 month-end dates in held-out-test

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] $*" | tee -a "$LOGFILE"; }

# ── service checks ────────────────────────────────────────────────────────────

lm_studio_alive() {
    curl -s --max-time 3 "http://localhost:1234/v1/models" >/dev/null 2>&1
}

# There is deliberately no start_finetune_server here (DJ-139).
#
# This script used to launch mlx_lm on port 1235 with
# `--adapter-path data/adapters/technical_v2` before every orchestrator run.
# That adapter was retired at DJ-124: the project's own measurement had it
# emitting Buy @ 0.70 on 98/98 tickers and collapsing ensemble entropy from
# 0.367 to 0.000. The sweep this watchdog drives goes through
# hifi.live.models._AGENT_CONFIG, which has had no route to 1235 or 1236 since
# DJ-135 — so the server was started, listened, and served nothing, while
# standing ready for anything that still probed the port.

# ── progress ──────────────────────────────────────────────────────────────────

portfolios_done() {
    local cond="$1"
    find "$DATA/walkforward/$cond" -name "portfolio.json" 2>/dev/null | wc -l | tr -d ' \t\n'
}

sidecars_for() {
    local cond="$1" agent="$2"
    find "$DATA/runs" -name "*_${agent}.json" -path "*/${cond}-*" 2>/dev/null | wc -l | tr -d ' \t\n'
}

progress_summary() {
    local cond="$1"
    local f t r m s c p
    f=$(sidecars_for "$cond" fundamental)
    t=$(sidecars_for "$cond" technical)
    r=$(sidecars_for "$cond" risk)
    m=$(sidecars_for "$cond" macro)
    s=$(sidecars_for "$cond" sentiment)
    c=$(sidecars_for "$cond" contrarian)
    p=$(portfolios_done "$cond")
    echo "fund=${f} tech=${t} risk=${r} macro=${m} sent=${s} cont=${c} portfolios=${p}/${EXPECTED_PORTFOLIOS}"
}

# ── condition logic ───────────────────────────────────────────────────────────

condition_complete() {
    local cond="$1"
    local n
    n=$(portfolios_done "$cond")
    [ "$n" -ge "$EXPECTED_PORTFOLIOS" ]
}

next_condition() {
    for c in "${CONDITIONS[@]}"; do
        condition_complete "$c" || { echo "$c"; return 0; }
    done
    echo "ALL_DONE"
}

# ── orchestrator management ───────────────────────────────────────────────────

orchestrator_running() {
    pgrep -f "hifi_walkforward" >/dev/null 2>&1
}

start_orchestrator() {
    local cond="$1"
    local logf="/tmp/walkforward_${cond}.log"
    lm_studio_alive || { log "ERROR: LM Studio (port 1234) not responding"; return 1; }
    cd "$PROJECT"
    nohup uv run python scripts/hifi_walkforward.py \
        --agent all --aggregate --pipeline \
        --condition "$cond" --period "$PERIOD" --quiet \
        >> "$logf" 2>&1 &
    local pid=$!
    log "Started orchestrator PID=${pid} condition=${cond} log=${logf}"
}

# ── IC/IR computation ─────────────────────────────────────────────────────────

run_ic_computation() {
    log "All 4 conditions complete. Running IC/IR/herding computation..."
    cd "$PROJECT"
    uv run python scripts/compute_phase15_ic.py \
        >> /tmp/walkforward_ic.log 2>&1 \
        && log "IC computation DONE. Full scientific run COMPLETE." \
        || log "WARNING: IC computation failed — check /tmp/walkforward_ic.log"
}

# ── main ──────────────────────────────────────────────────────────────────────

log "=== watchdog check ==="

NEXT=$(next_condition)

if [ "$NEXT" = "ALL_DONE" ]; then
    run_ic_computation
    exit 0
fi

PROG=$(progress_summary "$NEXT")
log "condition=${NEXT} | ${PROG}"

if orchestrator_running; then
    PIDS=$(pgrep -f "hifi_walkforward" | tr '\n' ',' | sed 's/,$//')
    log "Orchestrator RUNNING (PIDs=${PIDS})"
else
    log "Orchestrator NOT running — starting for condition=${NEXT}"
    start_orchestrator "$NEXT"
fi
