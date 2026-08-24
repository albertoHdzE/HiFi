#!/usr/bin/env bash
# Genesis II reset — Steps 1 and 3 of doc/GENESIS_CHECKLIST.md (2026-08-24).
#
#   scripts/genesis2_reset.sh --archive   # Step 1: copy live records to evidence archive
#   scripts/genesis2_reset.sh --clear     # Step 3: wipe state files (REQUIRES archive)
#
# Order matters: archive BEFORE clear, and --clear only AFTER the Alpaca
# dashboard account resets (Step 2). Both modes are guarded: --archive refuses
# to overwrite an existing archive (it is evidence); --clear refuses to run
# unless the archive is complete (the record must exist somewhere before it
# ceases to exist here).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIVE="$ROOT/data/live"
ARCHIVE="$LIVE/_genesis1_archive"
ACCOUNTS=(A B C D)
SHAKEDOWN_LOG="$ROOT/data/live/logs/nightly_20260823.log"

usage() { echo "usage: $0 --archive | --clear" >&2; exit 64; }

[ $# -eq 1 ] || usage

case "$1" in
  --archive)
    if [ -d "$ARCHIVE" ]; then
      echo "REFUSING: $ARCHIVE already exists — archives are never overwritten." >&2
      exit 65
    fi
    mkdir -p "$ARCHIVE"
    for a in "${ACCOUNTS[@]}"; do
      cp -R "$LIVE/$a" "$ARCHIVE/$a"
      echo "archived: $a ($(ls "$ARCHIVE/$a" | wc -l | tr -d ' ') files)"
    done
    if [ -f "$SHAKEDOWN_LOG" ]; then
      cp "$SHAKEDOWN_LOG" "$ARCHIVE/"
      echo "archived: nightly_20260823.log (Gate 1 shakedown evidence)"
    else
      echo "WARNING: shakedown log not found at $SHAKEDOWN_LOG" >&2
    fi
    echo "Archive complete: $ARCHIVE"
    ;;

  --clear)
    for a in "${ACCOUNTS[@]}"; do
      [ -d "$ARCHIVE/$a" ] || {
        echo "REFUSING: $ARCHIVE/$a missing — run --archive first. The record must exist somewhere before it stops existing here." >&2
        exit 66
      }
    done
    for a in "${ACCOUNTS[@]}"; do
      d="$LIVE/$a"
      rm -f "$d/hwm.json" \
            "$d/decisions.jsonl" \
            "$d/equity.jsonl" \
            "$d/portfolio_history.json" \
            "$d/circuit_breakers.jsonl"
      mkdir -p "$d"
      touch "$d/decisions.jsonl" "$d/equity.jsonl"
      echo "cleared: $a (hwm.json deleted — fresh 100k-dollar baseline re-seeds it)"
    done
    echo "State cleared for A, B, C, D. Ready for tonight's genesis run."
    ;;

  *) usage ;;
esac
