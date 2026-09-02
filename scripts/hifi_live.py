#!/usr/bin/env python
"""CLI for the nightly live cycle. All logic lives in ``hifi.live``.

Four-account live ablation on Alpaca paper trading (DJ-111):
  A  parallel ensemble          (Phase 15 champion)
  B  full sequential ensemble   (herding contrast)
  C  equal-weight buy-and-hold  (null model, no LLM)
  D  riskbudget calm_exposure   (deterministic quant, external provider)

All arms trade the same 98-ticker PHASE14_UNIVERSE, one decision cycle per day:
the evening after the close, with orders filling at the next open.

Usage:
  uv run python scripts/hifi_live.py --status                   # all accounts
  uv run python scripts/hifi_live.py --update-data              # refresh OHLCV
  uv run python scripts/hifi_live.py --account all --dry-run    # plan only
  uv run python scripts/hifi_live.py --account all              # agents, no orders
  uv run python scripts/hifi_live.py --account all --execute    # full nightly batch

Prefer `make live-nightly` over calling this directly: the nightly wrapper adds
the pre-flight (LM Studio, LangFuse) and the market-hours guard. A run started
without them on 2026-08-31 completed with no telemetry at all.

Env (.env):
  A: ALPACA_API_KEY_FIRST / ALPACA_SECRET_FIRST   (also _A, or unsuffixed)
  B: ALPACA_API_KEY_SECOND / ALPACA_SECRET_SECOND (also _B)
  C: ALPACA_API_KEY_THIRD / ALPACA_SECRET_THIRD   (also _C)
  D: ALPACA_API_KEY_FOURTH / ALPACA_SECRET_FOURTH (also _D)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from hifi.live.accounts import _ACCOUNTS, get_executor, show_status  # noqa: E402
from hifi.live.cycle import run_batch  # noqa: E402
from hifi.live.guards import _start_thread_watchdog  # noqa: E402
from hifi.live.market import update_data  # noqa: E402
from hifi.live.paths import _DATA_DIR, _get_tickers, _today_str  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live paper trading: 4-account ablation (DJ-111)"
    )
    parser.add_argument("--account", type=str, default="A",
                        choices=["A", "B", "C", "D", "all"], help="Which account(s) to run")
    parser.add_argument("--execute", action="store_true", help="Place real paper orders")
    parser.add_argument("--dry-run", action="store_true", help="Print orders without executing")
    parser.add_argument("--status", action="store_true",
                        help="Show portfolio status for all provisioned accounts")
    parser.add_argument("--update-data", action="store_true", help="Only refresh OHLCV data")
    parser.add_argument("--snapshot", action="store_true",
                        help="Capture equity/positions/history for all accounts (no trading)")
    parser.add_argument("--smoke", action="store_true",
                        help="Use 22-ticker smoke universe instead of 98")
    parser.add_argument("--date", type=str, default=None,
                        help="Override decision date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true",
                        help="Run even if this account already decided for the date "
                             "(protocol deviation — annotate the date)")
    args = parser.parse_args()

    # Only guard long inference runs; --status/--update-data are short.
    if args.execute or args.dry_run:
        _start_thread_watchdog()

    tickers = _get_tickers(smoke=args.smoke)
    date = args.date or _today_str()

    if args.status:
        for account in _ACCOUNTS:
            executor = get_executor(account)
            if executor is not None:
                show_status(account, executor)
                executor.disconnect()
        return

    if args.snapshot:
        from hifi.execution.portfolio_recorder import record_account
        for account in _ACCOUNTS:
            executor = get_executor(account)
            if executor is not None:
                record_account(executor, account, _DATA_DIR, decision_date=date)
                executor.disconnect()
        return

    if args.update_data:
        update_data(tickers)
        return

    accounts = list(_ACCOUNTS) if args.account == "all" else [args.account]
    run_batch(tickers, date, accounts, dry_run=args.dry_run, execute=args.execute,
              force=args.force, resolve_session=args.date is None)


if __name__ == "__main__":
    main()
