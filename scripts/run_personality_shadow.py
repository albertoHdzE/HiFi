"""Personality shadow replay (Phase 20, DJ-130 companion).

Replays each night's STORED agent votes through the four personality postures
and records what each would have decided — without trading, without LLM calls,
without touching the live arms.

Source: data/live/<acct>/walkforward/<date>/<condition>/<Y>/<M>/<TICKER>.json
        (EnsembleDecision sidecars written by run_aggregate_mode)
Output: data/live/<acct>/shadow_personality.jsonl (append; one row per
        (date, ticker): baseline + aggressive + conservative + careful)

The baseline row is cross-checked against the stored collective decision;
mismatches are counted and logged (they indicate schema drift, not trades).

LLM arms only (A/B). Arm D runs a single deterministic strategy — it has no
vote vector to replay, and imposing a posture on it would misrepresent its
design. C is a control by construction.

Usage:
    uv run python scripts/run_personality_shadow.py                # latest date, A+B
    uv run python scripts/run_personality_shadow.py --account A \
        --date 2026-08-24                                          # explicit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_ISO_DATE = __import__("re").compile(r"^\d{4}-\d{2}-\d{2}$")


def _latest_walkforward_date(account_dir: Path) -> str | None:
    """Newest dated dir. Only ISO dates qualify — condition-name dirs
    ('parallel', 'full') share this tree and sort after any date."""
    wf = account_dir / "walkforward"
    if not wf.exists():
        return None
    dates = sorted(p.name for p in wf.iterdir()
                   if p.is_dir() and _ISO_DATE.match(p.name))
    return dates[-1] if dates else None


def _iter_ensembles(account_dir: Path, date: str):
    """Yield (ticker, ensemble_decision_dict) for every stored ensemble JSON."""
    for cond_dir in sorted((account_dir / "walkforward" / date).iterdir()):
        if not cond_dir.is_dir():
            continue
        for jf in sorted(cond_dir.rglob("*.json")):
            try:
                ens = json.loads(jf.read_text()).get("ensemble_decision") or {}
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("unreadable ensemble %s (%s)", jf, exc)
                continue
            ticker = jf.stem
            yield ticker, ens


def replay(account: str, date: str, data_dir: str = str(_ROOT / "data")) -> dict:
    from hifi.collective.personality import PERSONALITIES, posture_vote

    acct_dir = Path(data_dir) / "live" / account
    votes_path = acct_dir / "walkforward" / date
    if not votes_path.exists():
        raise SystemExit(f"No walkforward ensembles for {account} on {date}")

    out_path = acct_dir / "shadow_personality.jsonl"
    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            existing = {json.loads(row).get("date")
                          for row in f if row.strip()}
    if date in existing:
        logger.info("[%s] %s already replayed — skipping (delete the row to redo)",
                    account, date)
        return {"rows": 0, "mismatches": 0}

    rows, mismatches = 0, 0
    with open(out_path, "a") as f:
        for ticker, ed in _iter_ensembles(acct_dir, date):
            decisions = ed.get("agent_decisions") or []
            confidences = ed.get("agent_confidences") or []
            votes = list(zip(decisions, confidences, strict=False))
            if not votes:
                continue

            stored_decision = ed.get("collective_decision")
            base = posture_vote(votes, PERSONALITIES["baseline"])
            if stored_decision and base["decision"] != stored_decision:
                mismatches += 1
                logger.warning("[%s] %s %s baseline %s != stored %s",
                               account, date, ticker, base["decision"],
                               stored_decision)

            row = {
                "replayed_at": __import__("datetime").datetime.now().isoformat(),
                "date": date,
                "ticker": ticker,
                "n_agents": len(votes),
                "stored_collective": stored_decision,
                "personalities": {
                    name: posture_vote(votes, prof)
                    for name, prof in PERSONALITIES.items()
                },
            }
            f.write(json.dumps(row) + "\n")
            rows += 1

    logger.info("[%s] %s shadow replay: %d tickers, %d baseline mismatches",
                account, date, rows, mismatches)
    return {"rows": rows, "mismatches": mismatches}


def main() -> None:
    ap = argparse.ArgumentParser(description="Personality shadow replay (DJ-130)")
    ap.add_argument("--account", default="A,B", help="comma-separated LLM arms")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest)")
    args = ap.parse_args()

    for account in [a.strip() for a in args.account.split(",")]:
        acct_dir = _ROOT / "data" / "live" / account
        date = args.date or _latest_walkforward_date(acct_dir)
        if not date:
            logger.warning("[%s] no walkforward dates found — skipping", account)
            continue
        replay(account, date)


if __name__ == "__main__":
    main()
