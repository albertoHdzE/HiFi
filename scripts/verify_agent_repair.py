#!/usr/bin/env python
"""Did the DJ-133 repairs change what the agents actually decide?

The fixes are verified at the input boundary: ratio coverage went 0/97 -> 97/97
(DJ-133a) and the sentiment context's day-to-day similarity went 0.999879 ->
~0.87 (DJ-133b). Neither of those is evidence that the *agents* behave
differently. An LLM handed better inputs can still emit the same constant.

This script reads a run's per-agent sidecars and reports the properties whose
collapse WAS the defect, against the contaminated 2026-08-27 baseline measured
before the repairs. Thresholds are declared here, ahead of the run, so the
verdict cannot be fitted to whatever comes back.

Usage
-----
    uv run python scripts/verify_agent_repair.py --date 2026-08-31
    uv run python scripts/verify_agent_repair.py --date 2026-08-31 --baseline 2026-08-27
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
from pathlib import Path

from hifi.agents.roster import VOTING_AGENTS  # noqa: E402

#: Sidecars are keyed by CONDITION, not by arm letter. Arm A runs the
#: "parallel" ensemble and arm B the "full" sequential one; C and D bypass
#: the agents entirely. Globbing "full-*" alone would silently inspect only
#: arm B and report it as the whole system.
LLM_CONDITIONS = ["parallel", "full"]

#: Measured on 2026-08-27, before the repairs. Two of five agents were pure
#: constants and the ensemble reached unanimous Hold on 57/97 names.
BASELINE_2026_08_27 = {
    "fundamental": {"modal_share": 1.00, "n_unique_conf": 2},
    "sentiment": {"modal_share": 1.00, "n_unique_conf": 2},
    "technical": {"modal_share": 0.75, "n_unique_conf": 5},
    "risk": {"modal_share": 0.80, "n_unique_conf": 6},
    "macro": {"modal_share": 0.92, "n_unique_conf": 6},
}

#: Pass criteria, declared before the run.
#:
#: An agent is "degenerate" if it puts essentially all of its votes on one
#: option. 0.95 is deliberately permissive: a genuinely bearish day can produce
#: a lopsided book, and we are testing for a stuck constant, not for balance.
MAX_MODAL_SHARE = 0.95
#: Below this, "confidence" carries no ordering information worth allocating on.
MIN_UNIQUE_CONFIDENCES = 3
#: The fundamental agent must actually receive ratios. This is the DJ-133a fix
#: itself, checked end to end rather than in a unit test.
MIN_RATIO_COVERAGE = 0.90


def _run_dirs(date: str) -> list[str]:
    out: list[str] = []
    for cond in LLM_CONDITIONS:
        out.extend(sorted(glob.glob(f"data/runs/{cond}-{date}-*")))
    return out


def _load_agent_signals(date: str, agent: str) -> list[dict]:
    out = []
    for d in _run_dirs(date):
        for f in Path(d).glob(f"*_{agent}.json"):
            try:
                sig = json.loads(f.read_text()).get("signal") or {}
            except Exception:
                continue
            if sig:
                out.append(sig)
    return out


def _profile(signals: list[dict]) -> dict:
    if not signals:
        return {"n": 0}
    decisions = collections.Counter(s.get("decision") for s in signals)
    confs = [s.get("confidence") for s in signals if s.get("confidence") is not None]
    modal, modal_n = decisions.most_common(1)[0]
    gaps = collections.Counter()
    for s in signals:
        for g in s.get("data_gaps") or []:
            gaps[g] += 1
    return {
        "n": len(signals),
        "decisions": dict(decisions),
        "modal": modal,
        "modal_share": modal_n / len(signals),
        "n_unique_conf": len(set(confs)),
        "conf_median": statistics.median(confs) if confs else None,
        "top_gaps": gaps.most_common(3),
    }


def _ratio_coverage(date: str) -> tuple[int, int]:
    """Fraction of fundamental passes that actually saw valuation ratios."""
    sigs = _load_agent_signals(date, "fundamental")
    if not sigs:
        return 0, 0
    blind = 0
    for s in sigs:
        gaps = set(s.get("data_gaps") or [])
        if {"pe", "pb", "ps"} & gaps:
            blind += 1
    return len(sigs) - blind, len(sigs)


def _ensemble_profile(date: str) -> dict:
    """Unanimity mass, which is what drove the Buy count to zero."""
    per_ticker: dict[str, list[str]] = collections.defaultdict(list)
    for agent in VOTING_AGENTS:
        for d in _run_dirs(date):
            ticker = Path(d).name.rsplit("-", 1)[-1]
            for f in Path(d).glob(f"*_{agent}.json"):
                try:
                    sig = json.loads(f.read_text()).get("signal") or {}
                except Exception:
                    continue
                if sig.get("decision"):
                    per_ticker[ticker].append(sig["decision"])
    unanimous = sum(1 for v in per_ticker.values() if len(v) >= 2 and len(set(v)) == 1)
    return {
        "tickers": len(per_ticker),
        "unanimous": unanimous,
        "unanimous_share": unanimous / len(per_ticker) if per_ticker else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="Decision date to verify, YYYY-MM-DD")
    ap.add_argument("--baseline", default="2026-08-27", help="Contaminated run to compare")
    args = ap.parse_args()

    print(f"=== Agent repair verification: {args.date} (baseline {args.baseline}) ===\n")

    failures: list[str] = []

    covered, total = _ratio_coverage(args.date)
    if total == 0:
        print("NO FUNDAMENTAL SIGNALS FOUND — did the run complete?")
        return 2
    share = covered / total
    verdict = "PASS" if share >= MIN_RATIO_COVERAGE else "FAIL"
    if verdict == "FAIL":
        failures.append(f"ratio coverage {share:.0%} < {MIN_RATIO_COVERAGE:.0%}")
    print(f"[{verdict}] DJ-133a ratio coverage: {covered}/{total} ({share:.0%}) "
          f"— was 0/97 (0%) on {args.baseline}\n")

    print(f"{'agent':12} {'n':>4} {'modal':>6} {'share':>7} {'was':>7} "
          f"{'uniq_conf':>10} {'was':>5}  verdict")
    for agent in VOTING_AGENTS:
        p = _profile(_load_agent_signals(args.date, agent))
        if not p["n"]:
            failures.append(f"{agent}: no signals")
            print(f"{agent:12} {'--':>4}  NO SIGNALS")
            continue
        base = BASELINE_2026_08_27.get(agent, {})
        bad = []
        if p["modal_share"] > MAX_MODAL_SHARE:
            bad.append("constant")
        if p["n_unique_conf"] < MIN_UNIQUE_CONFIDENCES:
            bad.append("flat-confidence")
        if bad:
            failures.append(f"{agent}: {', '.join(bad)}")
        print(f"{agent:12} {p['n']:>4} {p['modal']:>6} {p['modal_share']:>6.0%} "
              f"{base.get('modal_share', float('nan')):>6.0%} "
              f"{p['n_unique_conf']:>10} {base.get('n_unique_conf', 0):>5}  "
              f"{'FAIL: ' + ','.join(bad) if bad else 'PASS'}")
        if p["top_gaps"]:
            print(f"{'':12} gaps: {p['top_gaps']}")

    ens = _ensemble_profile(args.date)
    print(f"\nensemble unanimity: {ens['unanimous']}/{ens['tickers']} "
          f"({ens['unanimous_share']:.0%}) — was 57/97 (59%) on {args.baseline}")
    print("  (unanimity is not itself a defect; it is the mechanism by which two "
          "stuck agents drove the Buy count to zero)")

    print("\n" + "=" * 60)
    if failures:
        print("VERDICT: FAIL — do not start Genesis III")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERDICT: PASS — agents respond to the repaired inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
