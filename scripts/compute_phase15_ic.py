"""
Phase 15 IC/IR metrics computation (DJ-096).

Reads walkforward output JSONs produced by run_phase15_walkforward.py,
computes 60-day forward returns from the pre-downloaded OHLCV parquets,
and outputs an IC/IR/herding table per condition and regime.

No LLM calls are made — this is pure offline computation.

Usage
-----
    # IC across all four conditions (held-out-test period)
    uv run python scripts/compute_phase15_ic.py \\
        --period held-out-test --output-dir data/walkforward

    # IC for a single condition
    uv run python scripts/compute_phase15_ic.py \\
        --condition full --period held-out-test

    # Quiet: print only the summary table
    uv run python scripts/compute_phase15_ic.py \\
        --period held-out-test --quiet

Output columns
--------------
  condition : full | parallel | homogeneous | no-memory
  regime    : all | bull_low_vol | bear_high_vol | rate_shock | recovery | neutral
  n_pairs   : number of (ticker, date) pairs with both signal and forward return
  ic        : Spearman rank IC
  p_value   : two-sided p-value for IC
  ir        : Information Ratio (monthly IC / IC_std) — "n/a" when < 2 dates
  herding   : fraction of ensemble runs with all-agent agreement
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CONDITIONS = ["full", "parallel", "homogeneous", "no-memory"]
_DEFAULT_HORIZON = 60  # trading days


# ---------------------------------------------------------------------------
# OHLCV cache
# ---------------------------------------------------------------------------


def _load_ohlcv(ticker: str, data_dir: str) -> object | None:
    """Load OHLCV parquet for a ticker, returning None if not found."""
    import pandas as pd

    path = Path(data_dir) / "market" / ticker / "ohlcv.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:
        logger.warning("Failed to load OHLCV for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Scan and load walkforward JSONs
# ---------------------------------------------------------------------------


def scan_walkforward_dir(
    output_dir: str,
    condition: str,
    dates: list[str] | None = None,
) -> list[dict]:
    """
    Scan the walkforward directory for one condition and return loaded records.

    Parameters
    ----------
    output_dir : str
        Root walkforward directory.
    condition : str
        One of: full, parallel, homogeneous, no-memory.
    dates : list[str] | None
        If provided, only include records whose date is in this list.

    Returns
    -------
    list[dict]
        List of dicts with keys: condition, date, ticker, output (EnsembleOutput dict).
    """
    cond_dir = Path(output_dir) / condition
    if not cond_dir.exists():
        return []

    date_set = set(dates) if dates else None
    records = []

    for year_dir in sorted(cond_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for json_file in sorted(month_dir.glob("*.json")):
                ticker = json_file.stem
                # Reconstruct date from path: year/month/ticker.json
                year = year_dir.name
                month = month_dir.name
                import calendar
                last_day = calendar.monthrange(int(year), int(month))[1]
                date = f"{year}-{month}-{last_day:02d}"

                if date_set is not None and date not in date_set:
                    continue

                try:
                    output = json.loads(json_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Could not load %s: %s", json_file, exc)
                    continue

                records.append({
                    "condition": condition,
                    "date": date,
                    "ticker": ticker,
                    "output": output,
                })

    return records


# ---------------------------------------------------------------------------
# IC computation per condition × regime
# ---------------------------------------------------------------------------


def compute_condition_ic(
    records: list[dict],
    data_dir: str,
    horizon_trading_days: int = _DEFAULT_HORIZON,
) -> dict:
    """
    Compute IC, IR, and herding for one condition.

    Parameters
    ----------
    records : list[dict]
        Records from scan_walkforward_dir() for one condition.
    data_dir : str
        Data directory for OHLCV parquets.
    horizon_trading_days : int
        Forward return horizon in trading days.

    Returns
    -------
    dict with keys:
      n_pairs_total, n_pairs_with_return, ic_all, ir_all, herding_all,
      by_regime (dict: regime → {n_pairs, ic, ir, herding})
    """
    from hifi.simulation.metrics import (
        buy_strength,
        compute_herding_coefficient,
        compute_ic,
        compute_ir,
        forward_return_from_ohlcv,
    )

    # Cache OHLCV data per ticker
    ohlcv_cache: dict[str, object] = {}

    pairs_all: list[tuple[float, float]] = []      # (signal, return)
    ic_by_date: dict[str, list[tuple[float, float]]] = {}  # date → pairs
    by_regime: dict[str, list[tuple[float, float]]] = {}

    outputs_all: list[dict] = []

    for record in records:
        ticker = record["ticker"]
        date = record["date"]
        output = record["output"]

        if ticker not in ohlcv_cache:
            ohlcv_cache[ticker] = _load_ohlcv(ticker, data_dir)

        ohlcv = ohlcv_cache[ticker]
        if ohlcv is None:
            continue

        fwd_return = forward_return_from_ohlcv(
            ohlcv, date, horizon_trading_days=horizon_trading_days
        )
        if fwd_return is None:
            continue

        signal = buy_strength(output)
        if signal is None:
            continue

        pairs_all.append((signal, fwd_return))
        outputs_all.append(output)

        ic_by_date.setdefault(date, []).append((signal, fwd_return))

        # Regime from stored output (classify from ensemble_decision or fallback)
        regime = _extract_regime(output, ticker, date, data_dir)
        by_regime.setdefault(regime, []).append((signal, fwd_return))

    # Compute overall IC
    n_pairs = len(pairs_all)
    ic_all_result = None
    if n_pairs >= 2:
        signals_all, returns_all = zip(*pairs_all, strict=False)
        ic_all_result = compute_ic(list(signals_all), list(returns_all))

    # Compute monthly IC series → IR
    monthly_ics = []
    for date_pairs in ic_by_date.values():
        if len(date_pairs) >= 2:
            s, r = zip(*date_pairs, strict=False)
            monthly_ics.append(compute_ic(list(s), list(r)).ic)

    ir_all = compute_ir(monthly_ics) if len(monthly_ics) >= 2 else None

    # Compute per-regime IC
    regime_results: dict[str, dict] = {}
    for regime, regime_pairs in by_regime.items():
        n = len(regime_pairs)
        if n < 2:
            regime_results[regime] = {"n_pairs": n, "ic": None, "p_value": None}
            continue
        s, r = zip(*regime_pairs, strict=False)
        r_ic = compute_ic(list(s), list(r))
        regime_herding = compute_herding_coefficient([
            rec["output"] for rec in records
            if _extract_regime(rec["output"], rec["ticker"], rec["date"], data_dir) == regime
        ])
        regime_results[regime] = {
            "n_pairs": n,
            "ic": r_ic.ic,
            "p_value": r_ic.p_value,
            "herding": regime_herding,
        }

    herding_all = compute_herding_coefficient(outputs_all)

    return {
        "n_pairs_total": len(records),
        "n_pairs_with_return": n_pairs,
        "ic": ic_all_result.ic if ic_all_result else None,
        "p_value": ic_all_result.p_value if ic_all_result else None,
        "ir": ir_all,
        "herding": herding_all,
        "monthly_ic_windows": len(monthly_ics),
        "by_regime": regime_results,
    }


def _extract_regime(output: dict, ticker: str, date: str, data_dir: str) -> str:
    """Extract regime from output dict or recompute from data."""
    # Try signals list first (fundamental agent may carry regime in analysis)
    # Fall back to SPY/macro classification
    try:
        import pandas as pd

        from hifi.data.regime import classify_regime

        spy_path = Path(data_dir) / "market" / "SPY" / "ohlcv.parquet"
        macro_path = Path(data_dir) / "macro" / "macro.parquet"
        if spy_path.exists() and macro_path.exists():
            spy = pd.read_parquet(spy_path)
            macro = pd.read_parquet(macro_path)
            spy.index = pd.to_datetime(spy.index)
            macro.index = pd.to_datetime(macro.index)
            return classify_regime(date, spy, macro)
    except Exception:
        pass
    return "neutral"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt_ic(v: float | None) -> str:
    return f"{v:+.4f}" if v is not None else "  n/a "


def _fmt_p(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "  n/a "


def _fmt_ir(v: float | None) -> str:
    return f"{v:+.3f}" if v is not None else " n/a"


def print_results_table(results: dict[str, dict]) -> None:
    """Print a summary table of IC/IR/herding per condition."""
    header = (
        f"{'Condition':<14}  {'N_pairs':>8}  {'IC':>8}  {'p-value':>8}"
        f"  {'IR':>7}  {'Herding':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for cond, r in sorted(results.items()):
        print(
            f"{cond:<14}  "
            f"{r['n_pairs_with_return']:>8}  "
            f"{_fmt_ic(r['ic']):>8}  "
            f"{_fmt_p(r['p_value']):>8}  "
            f"{_fmt_ir(r['ir']):>7}  "
            f"{r['herding']:>8.4f}"
        )
    print("=" * len(header))


def print_regime_table(condition: str, regime_data: dict[str, dict]) -> None:
    """Print regime-conditional IC table for one condition."""
    print(f"\nRegime breakdown — condition: {condition}")
    header = f"  {'Regime':<16}  {'N_pairs':>8}  {'IC':>8}  {'p-value':>8}  {'Herding':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for regime, r in sorted(regime_data.items()):
        print(
            f"  {regime:<16}  "
            f"{r['n_pairs']:>8}  "
            f"{_fmt_ic(r.get('ic')):>8}  "
            f"{_fmt_p(r.get('p_value')):>8}  "
            f"{r.get('herding', 0.0):>8.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Phase 15 IC/IR metrics")
    p.add_argument(
        "--condition",
        nargs="+",
        default=None,
        help=f"Conditions to evaluate (default: all). Choices: {_CONDITIONS}",
    )
    p.add_argument(
        "--period",
        default="held-out-test",
        help="Evaluation period (default: held-out-test)",
    )
    p.add_argument(
        "--output-dir",
        default="data/walkforward",
        help="Walkforward output directory",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Data directory for OHLCV parquets",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=_DEFAULT_HORIZON,
        help=f"Forward return horizon in trading days (default: {_DEFAULT_HORIZON})",
    )
    p.add_argument(
        "--regime-breakdown",
        action="store_true",
        help="Print per-regime IC breakdown for each condition",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the summary table",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Write full results to this JSON file",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir or os.environ.get("HIFI_DATA_DIR", "data")
    conditions = args.condition or _CONDITIONS

    from hifi.simulation.schedule import get_period_dates
    dates = get_period_dates(args.period)

    if not args.quiet:
        print(
            f"Computing IC: period={args.period} dates={len(dates)} "
            f"conditions={conditions}"
        )

    all_results: dict[str, dict] = {}

    for condition in conditions:
        if not args.quiet:
            print(f"  Scanning condition={condition} ...", end=" ", flush=True)

        records = scan_walkforward_dir(args.output_dir, condition, dates=dates)

        if not records:
            if not args.quiet:
                print("0 records found.")
            all_results[condition] = {
                "n_pairs_total": 0,
                "n_pairs_with_return": 0,
                "ic": None,
                "p_value": None,
                "ir": None,
                "herding": 0.0,
                "monthly_ic_windows": 0,
                "by_regime": {},
            }
            continue

        result = compute_condition_ic(records, data_dir, horizon_trading_days=args.horizon)
        all_results[condition] = result

        if not args.quiet:
            ic_str = f"{result['ic']:+.4f}" if result['ic'] is not None else "n/a"
            print(f"{len(records)} records, IC={ic_str}")

    # Print summary table
    print_results_table(all_results)

    # Per-regime breakdown
    if args.regime_breakdown:
        for condition, result in all_results.items():
            if result.get("by_regime"):
                print_regime_table(condition, result["by_regime"])

    # Write JSON output if requested
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(all_results, indent=2),
            encoding="utf-8",
        )
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
