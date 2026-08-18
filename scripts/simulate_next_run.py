"""Read-only preview of what the next live cycle would trade (DJ-121).

Answers "what will tonight's run actually do?" without touching anything: it
reads live account state and the most recent ensemble signals on disk, runs the
real compose -> risk -> allocate pipeline, and reports the orders that would be
placed plus the portfolio they would produce.

**Nothing is written and no order is sent.** The broker connection is used only
for GET (positions, equity, cash). No file under data/ is created or modified.

Scope, stated plainly
---------------------
This exercises the stage that was broken (DJ-121: allocation) using signals
from the most recent completed decision date. Tonight's run will generate
*fresh* signals for the current session, so the ticker list will differ. What
this validates is the mechanism — that convictions become correctly sized,
cash-feasible, constraint-respecting orders — not tomorrow's exact basket.

The agent stage is not re-run here; it takes hours for 98 tickers x 6 agents,
and its health is already independently checked by the section 0 provenance
gate (tool-failure rate 0.000 as of 2026-08-14).

Usage
-----
    uv run python scripts/simulate_next_run.py
    uv run python scripts/simulate_next_run.py --accounts A,B --date 2026-08-14
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

logger = logging.getLogger("simulate")

_CONDITION = {"A": "parallel", "B": "full", "C": "control", "D": "riskbudget"}


def _load_env() -> None:
    env = _ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        m = re.match(r"([A-Z][A-Z0-9_]*)=(.*)$", line.strip())
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip("\"'"))


def _latest_signal_date(account: str, data_dir: str) -> str | None:
    base = Path(data_dir) / "live" / account / "walkforward"
    if not base.exists():
        return None
    dates = sorted(d.name for d in base.iterdir() if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name))
    return dates[-1] if dates else None


def simulate(account: str, date: str | None, data_dir: str) -> dict | None:
    """Return a preview dict for one account, or None when not simulable."""
    from run_phase16_live import (
        _get_sectors,
        _get_tickers,
        get_executor,
        load_ensemble_signals,
        run_mcp_pipeline,
    )

    condition = _CONDITION[account]
    tickers = _get_tickers(smoke=False)

    if condition == "control":
        # C bypasses the pipeline entirely: buy-once-hold, already fully invested.
        return {"account": account, "condition": condition, "skipped":
                "no signal layer by design (buy-once-hold null model)"}

    executor = get_executor(account)
    if executor is None:
        return {"account": account, "condition": condition, "skipped": "no broker credentials"}

    try:
        equity = executor.get_portfolio_value()
        cash = executor.get_account_cash()
        positions = executor.get_positions()
        invested = sum(p.market_value for p in positions.values())

        if condition == "riskbudget":
            from hifi.execution.riskbudget_strategy import get_riskbudget_signals
            sig_date = date or _latest_signal_date("A", data_dir) or ""
            signals = get_riskbudget_signals(
                tickers, sig_date, data_dir, sectors=_get_sectors()
            ).get("signals", [])
        else:
            sig_date = date or _latest_signal_date(account, data_dir)
            if not sig_date:
                return {"account": account, "condition": condition,
                        "skipped": "no ensemble signals on disk"}
            signals = load_ensemble_signals(tickers, sig_date, condition, account)

        if not signals:
            return {"account": account, "condition": condition,
                    "skipped": f"no signals for {sig_date}"}

        snapshot = run_mcp_pipeline(signals, tickers, executor)
        orders = snapshot.orders

        buys = [o for o in orders if o["side"] == "BUY"]
        sells = [o for o in orders if o["side"] == "SELL"]
        buy_notional = sum(o["estimated_value"] for o in buys)
        sell_notional = sum(o["estimated_value"] for o in sells)

        # Projected book after the orders fill.
        projected: dict[str, float] = {s: p.market_value for s, p in positions.items()}
        for o in orders:
            delta = o["estimated_value"] * (1 if o["side"] == "BUY" else -1)
            projected[o["ticker"]] = projected.get(o["ticker"], 0.0) + delta
        projected = {k: v for k, v in projected.items() if v > 1.0}

        sectors = _get_sectors()
        by_sector: dict[str, float] = collections.defaultdict(float)
        for sym, val in projected.items():
            by_sector[sectors.get(sym, "Unknown")] += val / equity if equity else 0.0

        return {
            "account": account,
            "condition": condition,
            "signal_date": sig_date,
            "signal_mix": dict(collections.Counter(s.get("decision") for s in signals)),
            "equity": equity,
            "cash": cash,
            "invested": invested,
            "exposure_before": invested / equity if equity else 0.0,
            "n_positions_before": len(positions),
            "orders": orders,
            "n_buy": len(buys),
            "n_sell": len(sells),
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "cash_after": cash - buy_notional + sell_notional,
            "n_positions_after": len(projected),
            "exposure_after": sum(projected.values()) / equity if equity else 0.0,
            "max_position_pct": ((max(projected.values()) / equity * 100)
                             if projected and equity else 0.0),
            "max_sector_pct": (max(by_sector.values()) * 100) if by_sector else 0.0,
        }
    finally:
        executor.disconnect()


def _checks(r: dict) -> list[tuple[bool, str]]:
    """Constraint checks the live run must satisfy."""
    return [
        (r["cash_after"] >= 0,
         f"cash stays non-negative (${r['cash_after']:,.0f} after)"),
        (r["buy_notional"] <= r["cash"],
         f"buy demand ${r['buy_notional']:,.0f} <= cash ${r['cash']:,.0f}"),
        (r["max_position_pct"] <= 5.5,
         f"largest position {r['max_position_pct']:.2f}% <= 5% (+tolerance)"),
        (r["max_sector_pct"] <= 21.0,
         f"largest sector {r['max_sector_pct']:.2f}% <= 20% (+tolerance)"),
        (all(o["quantity"] > 0 for o in r["orders"]),
         "all order quantities positive"),
        (r["n_positions_after"] >= r["n_positions_before"],
         f"book widens {r['n_positions_before']} -> {r['n_positions_after']} positions"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accounts", default="A,B,C,D")
    ap.add_argument("--date", default=None, help="signal date (default: most recent on disk)")
    ap.add_argument("--data-dir", default=os.environ.get("HIFI_DATA_DIR", "data"))
    ap.add_argument("--show-orders", type=int, default=6, help="orders to list per arm")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    _load_env()

    print("=" * 78)
    print("SANDBOX PREVIEW — read-only. No orders placed, no files written.")
    print("Signals are from the last completed decision date; tonight's will differ.")
    print("=" * 78)

    all_ok = True
    for account in [a.strip().upper() for a in args.accounts.split(",") if a.strip()]:
        r = simulate(account, args.date, args.data_dir)
        print(f"\n{'-' * 78}\nArm {account} ({_CONDITION.get(account, '?')})\n{'-' * 78}")
        if r is None or "skipped" in r:
            print(f"  skipped: {(r or {}).get('skipped', 'unavailable')}")
            continue

        print(f"  signals {r['signal_date']}: {r['signal_mix']}")
        print(f"  before : equity ${r['equity']:,.0f}  cash ${r['cash']:,.0f}  "
              f"{r['n_positions_before']} positions  exposure {r['exposure_before']*100:.1f}%")
        print(f"  orders : {r['n_buy']} buy (${r['buy_notional']:,.0f}) / "
              f"{r['n_sell']} sell (${r['sell_notional']:,.0f})")
        for o in r["orders"][:args.show_orders]:
            print(f"             {o['side']:<4} {o['ticker']:<6} {o['quantity']:>10.3f} "
                  f"  ${o['estimated_value']:>10,.0f}")
        if len(r["orders"]) > args.show_orders:
            print(f"             ... and {len(r['orders']) - args.show_orders} more")
        print(f"  after  : {r['n_positions_after']} positions  "
              f"exposure {r['exposure_after']*100:.1f}%  cash ${r['cash_after']:,.0f}")

        print("  checks :")
        for ok, msg in _checks(r):
            print(f"             {'PASS' if ok else 'FAIL'}  {msg}")
            all_ok &= ok

    print(f"\n{'=' * 78}")
    print("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED — inspect before running live")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
