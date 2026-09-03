#!/usr/bin/env bash
# Genesis reset — retire one generation of the live experiment and open the next.
#
#   scripts/genesis_reset.sh --archive --generation 2
#   scripts/genesis_reset.sh --clear   --generation 2 --genesis-date 2026-09-03
#
# A "generation" is one continuous run of all four arms from the same capital on
# the same date. Between generations the record must be moved somewhere it
# cannot be appended to, and the per-arm state that is *tied to the old capital*
# must be removed — otherwise the next generation inherits a high-water mark, a
# position book and a decision history that describe an account that no longer
# exists.
#
# Order, and why each step is guarded:
#
#   1. --archive          (this script)   evidence is copied before anything moves
#   2. reset the accounts (Alpaca, by hand — all four together, or they are not
#                          comparable, and comparability is the whole design)
#   3. --clear            (this script)   state removed, genesis marker advanced
#   4. make live-nightly                  the new generation's first cycle
#
# --archive refuses to overwrite an existing archive: an archive is evidence.
# --clear refuses unless the archive is complete: the record must exist
# somewhere before it stops existing here.
#
# Supersedes scripts/archive/genesis2_reset.sh, which hard-coded generation 2,
# hard-coded one shakedown log, and predated three state files that now exist —
# book_state.json (DJ-130), dry_runs.jsonl (DJ-136) and the genesis marker
# itself, which nothing in the codebase writes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIVE="$ROOT/data/live"
LOGS="$LIVE/logs"
MARKER="$LIVE/genesis_date.txt"
ACCOUNTS=(A B C D)

#: State tied to the retired generation's capital. Every one of these describes
#: an account balance, a position book or a decision taken against them, so
#: carrying any of it across a reset makes the new generation's first night a
#: continuation of the old one wearing new numbers.
#:
#: hwm.json in particular: the drawdown guard ratchets against it (DJ-129b), so
#: a high-water mark left over from a $101k book would read the fresh $100k
#: account as already down and could halt the arm on night one.
CAPITAL_STATE=(
    hwm.json
    decisions.jsonl
    decisions.jsonl.bak
    equity.jsonl
    portfolio_history.json
    circuit_breakers.jsonl
    book_state.json
    dry_runs.jsonl
)

#: Deliberately NOT cleared. These are agent-signal evidence: what the ensemble
#: said about a security on a date, and how a shadow personality would have
#: voted. Neither is a function of the account balance, so neither is invalidated
#: by a reset, and both are more useful continuous than restarted.
#:      walkforward/   shadow_personality.jsonl

usage() {
    cat >&2 <<'EOF'
usage:
  genesis_reset.sh --archive --generation N
  genesis_reset.sh --clear   --generation N --genesis-date YYYY-MM-DD

  --generation N     the generation being retired; archive lands in
                     data/live/_genesisN_archive
  --genesis-date     first decision date of the NEW generation, written to
                     data/live/genesis_date.txt
EOF
    exit 64
}

die() { echo "REFUSING: $*" >&2; exit 65; }

MODE=""
GENERATION=""
NEW_GENESIS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --archive|--clear) [ -z "$MODE" ] || usage; MODE="${1#--}"; shift ;;
        --generation)      GENERATION="${2:-}"; shift 2 ;;
        --genesis-date)    NEW_GENESIS="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done

[ -n "$MODE" ] || usage
case "$GENERATION" in
    ''|*[!0-9]*) echo "ERROR: --generation N required (a positive integer)" >&2; exit 64 ;;
esac

ARCHIVE="$LIVE/_genesis${GENERATION}_archive"

# Portable ISO-8601 date validation: no GNU `date -d`, no BSD `date -j` guessing.
is_iso_date() {
    python3 - "$1" <<'PY'
import datetime, sys
try:
    datetime.date.fromisoformat(sys.argv[1])
except ValueError:
    sys.exit(1)
PY
}

case "$MODE" in

archive)
    # `[ ... ] && die` would be wrong here: under `set -e` a false test is a
    # failing command and would abort the script instead of proceeding.
    if [ -d "$ARCHIVE" ]; then
        die "$ARCHIVE already exists — archives are never overwritten."
    fi

    retiring="$(cat "$MARKER" 2>/dev/null || true)"
    if [ -z "$retiring" ]; then
        echo "WARNING: no genesis marker at $MARKER; archiving every log." >&2
    else
        echo "Retiring generation ${GENERATION}, opened ${retiring}."
    fi

    mkdir -p "$ARCHIVE"

    for a in "${ACCOUNTS[@]}"; do
        [ -d "$LIVE/$a" ] || die "$LIVE/$a missing — refusing a partial archive."
        cp -R "$LIVE/$a" "$ARCHIVE/$a"
        echo "archived: arm $a ($(find "$ARCHIVE/$a" -type f | wc -l | tr -d ' ') files)"
    done

    # The genesis marker travels with the record it dates. Without it an archive
    # is a pile of rows whose "days since genesis" cannot be recomputed.
    if [ -f "$MARKER" ]; then
        cp "$MARKER" "$ARCHIVE/genesis_date.txt"
    fi

    # Repairs made to the record are part of the record (DJ-136).
    if [ -d "$LIVE/_dj136_backup" ]; then
        cp -R "$LIVE/_dj136_backup" "$ARCHIVE/_dj136_backup"
        echo "archived: _dj136_backup (pre-repair decisions and equity)"
    fi

    # Every night's log from this generation. The cutoff is the genesis marker,
    # so the archive holds the logs for the rows it holds and no others.
    mkdir -p "$ARCHIVE/logs"
    n_logs=0
    cutoff="${retiring//-/}"
    for f in "$LOGS"/nightly_*.log "$LOGS"/verify_*.log; do
        [ -f "$f" ] || continue
        stamp="$(basename "$f")"; stamp="${stamp##*_}"; stamp="${stamp%.log}"
        if [ -z "$cutoff" ] || [ "$stamp" -ge "$cutoff" ] 2>/dev/null; then
            cp "$f" "$ARCHIVE/logs/"
            n_logs=$((n_logs + 1))
        fi
    done
    echo "archived: ${n_logs} nightly/verify logs${cutoff:+ from ${retiring} onward}"

    echo "Archive complete: $ARCHIVE"
    echo "Next: reset the four Alpaca accounts to \$100,000 TOGETHER, then --clear."
    ;;

clear)
    [ -n "$NEW_GENESIS" ] || { echo "ERROR: --genesis-date YYYY-MM-DD required" >&2; exit 64; }
    is_iso_date "$NEW_GENESIS" || die "'$NEW_GENESIS' is not an ISO-8601 date."

    for a in "${ACCOUNTS[@]}"; do
        [ -d "$ARCHIVE/$a" ] || die "$ARCHIVE/$a missing — run --archive first. The record must exist somewhere before it stops existing here."
    done

    # A marker that moves backwards would make "days since genesis" negative and
    # silently reclassify the deployment phase the agents are told they are in.
    previous="$(cat "$MARKER" 2>/dev/null || true)"
    if [ -n "$previous" ] && ! [[ "$NEW_GENESIS" > "$previous" ]]; then
        die "--genesis-date $NEW_GENESIS is not after the current marker $previous."
    fi

    for a in "${ACCOUNTS[@]}"; do
        d="$LIVE/$a"
        mkdir -p "$d"
        for f in "${CAPITAL_STATE[@]}"; do rm -f "$d/$f"; done
        touch "$d/decisions.jsonl" "$d/equity.jsonl"
        echo "cleared: arm $a (hwm.json deleted — the fresh baseline re-seeds it)"
    done

    # Nothing in the codebase writes this marker; hifi.agents.context only reads
    # it, to tell each agent how many sessions old the deployment is and whether
    # it is in DEPLOYMENT or STEADY phase. Left stale it does not error — it
    # tells the agents they are managing an established book on night one.
    printf '%s\n' "$NEW_GENESIS" > "$MARKER"
    echo "genesis marker: ${previous:-<unset>} -> $NEW_GENESIS"

    echo "State cleared for ${ACCOUNTS[*]}. Ready for the first cycle of generation $((GENERATION + 1))."
    ;;

*) usage ;;
esac
