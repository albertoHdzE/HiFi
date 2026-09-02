#!/usr/bin/env bash
# HiFi Walk-Forward Watchdog
# Monitors the orchestrator, restarts on crash, advances through all 4 conditions.
# Called by Claude's cron every 10 minutes.
#
# Conditions run in sequence (agent-first, checkpoint-resume):
#   full → no-memory → parallel → homogeneous
# After all 4: computes IC/IR/herding metrics.

set -uo pipefail

PROJECT="/Users/alberto/Documents/projects/HiFi"
DATA="$PROJECT/data"
LOGFILE="/tmp/hifi_watchdog.log"
PERIOD="held-out-test"
CONDITIONS=("full" "no-memory" "parallel" "homogeneous")
EXPECTED_PORTFOLIOS=24   # 24 month-end dates in held-out-test

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] $*" | tee -a "$LOGFILE"; }

# ── service checks ────────────────────────────────────────────────────────────

port_alive() {
    curl -s --max-time 3 "http://localhost:$1/health" >/dev/null 2>&1
}

lm_studio_alive() {
    curl -s --max-time 3 "http://localhost:1234/v1/models" >/dev/null 2>&1
}

start_finetune_server() {
    port_alive 1235 && return 0
    log "Starting mlx_lm finetune server (port 1235)..."
    cd "$PROJECT"
    nohup venvs/finetune/bin/python -m mlx_lm server \
        --model "$HOME/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-8bit" \
        --adapter-path data/adapters/technical_v2 \
        --port 1235 --log-level WARNING \
        > /tmp/mlx_tech_1235.log 2>&1 &
    local wait=0
    while [ $wait -lt 30 ]; do
        sleep 3; wait=$((wait + 3))
        port_alive 1235 && { log "Finetune server UP (${wait}s)"; return 0; }
    done
    log "WARNING: finetune server did not respond after 30s"
}

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
    start_finetune_server
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
