"""Offline sweep modes: the MCP pipeline and checkpoint status over many dates.

The live cycle runs one date against a broker; these run many dates against a
simulated book, sharing ``hifi.simulation.pipeline.run_pipeline`` with the live
path so that a portfolio composed offline and one composed tonight are composed
by the same code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hifi.agents.roster import CANONICAL_ORDER
from hifi.live import paths

logger = logging.getLogger(__name__)

_CAPITAL = 500_000.0


def _load_ohlcv(
    data_dir: str, tickers: list[str], as_of_date: str
) -> dict[str, list[dict]]:
    try:
        import pandas as pd  # noqa: PLC0415

        result: dict[str, list[dict]] = {}
        for ticker in tickers:
            path = Path(data_dir) / "market" / ticker / "ohlcv.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df.columns = df.columns.str.lower()   # normalize 'Close' → 'close'
            df.index = pd.to_datetime(df.index)
            df = df[df.index <= as_of_date].tail(90)
            if df.empty:
                continue
            result[ticker] = [
                {"date": str(idx.date()), "close": float(row["close"])}
                for idx, row in df.iterrows()
            ]
        return result
    except Exception as exc:
        logger.warning("OHLCV load error: %s", exc)
        return {}


def run_pipeline_mode(
    condition: str,
    dates: list[str],
    tickers: list[str],
    data_dir: str,
    output_dir: str,
    dry_run: bool,
    quiet: bool,
) -> dict[str, int]:
    """
    For each date: collect signals from ensemble JSONs → run MCP pipeline → write portfolio.json.
    Checkpoint-resume: skips if portfolio.json already exists.
    """
    from hifi.data.universe import get_sector  # noqa: PLC0415
    from hifi.simulation.pipeline import run_pipeline  # noqa: PLC0415

    n_done = n_skip = n_fail = 0

    for date in dates:
        port_path = paths._portfolio_path(output_dir, condition, date)
        if port_path.exists():
            n_skip += 1
            continue

        if dry_run:
            n_done += 1
            continue

        # Collect signals from ensemble JSONs for this date
        signals: list[dict] = []
        for ticker in tickers:
            ens_path = paths._ensemble_path(output_dir, condition, date, ticker)
            if not ens_path.exists():
                continue
            try:
                data = json.loads(ens_path.read_text(encoding="utf-8"))
                ed = data.get("ensemble_decision", {})
                decision = ed.get("collective_decision") or "Hold"
                confidence = float(ed.get("collective_confidence") or 0.5)
                signals.append({
                    "ticker": ticker,
                    "decision": decision,
                    "confidence": confidence,
                    "sector": get_sector(ticker) or "Unknown",
                })
            except Exception as exc:
                logger.warning("Could not read ensemble JSON %s %s: %s", ticker, date, exc)

        if not signals:
            logger.warning("No signals for %s %s; skipping pipeline", condition, date)
            n_fail += 1
            continue

        try:
            ohlcv = _load_ohlcv(data_dir, tickers, date)
            prices = {t: rows[-1]["close"] for t, rows in ohlcv.items() if rows}

            portfolio_state = {
                "portfolio": {},
                "portfolio_value": _CAPITAL,
                "hwm_value": _CAPITAL,
                "holdings": {},
                "prices": prices,
            }
            constraints = {
                "max_single_stock": 0.05,
                "max_sector": 0.20,
                "min_position": 0.005,
                "capital": _CAPITAL,
                "current_capital": 0.0,
            }

            snapshot = run_pipeline(signals, ohlcv, portfolio_state, constraints)

            port_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = port_path.with_suffix(".tmp")
            tmp.write_text(snapshot.to_json(), encoding="utf-8")
            tmp.rename(port_path)
            n_done += 1
            if not quiet:
                logger.info(
                    "PIPELINE %s %s: buy=%d orders=%d notional=%.0f → %s",
                    condition, date,
                    snapshot.n_buy, len(snapshot.orders),
                    snapshot.total_estimated_value, port_path,
                )
        except Exception as exc:
            n_fail += 1
            logger.error("PIPELINE FAIL %s %s: %s", condition, date, exc)

    if dry_run:
        print(
            f"[dry-run] --pipeline --condition {condition}\n"
            f"  dates={len(dates)}  would_run={n_done}  skip={n_skip}"
        )
    else:
        print(f"Pipeline: done={n_done} skip={n_skip} fail={n_fail}")
    return {"done": n_done, "skip": n_skip, "fail": n_fail}


def run_status_mode(
    conditions: list[str],
    dates: list[str],
    tickers: list[str],
    data_dir: str,
    output_dir: str,
) -> None:
    total_passes = len(dates) * len(tickers)
    total_ensembles = len(dates) * len(tickers)
    total_portfolios = len(dates)

    print(f"\nStatus  |  dates={len(dates)}  tickers={len(tickers)}")
    print(f"{'Condition':<16}  {'Sidecars':>10}  {'Ensembles':>10}  {'Portfolios':>11}")
    print("  " + "-" * 58)

    for condition in conditions:
        # Count sidecars: 6 agents × N_dates × N_tickers
        n_sidecars = sum(
            1
            for at in CANONICAL_ORDER
            for d in dates
            for t in tickers
            if paths._sidecar_path(data_dir, condition, d, t, at).exists()
        )
        total_sidecars = len(CANONICAL_ORDER) * total_passes

        n_ensembles = sum(
            1 for d in dates for t in tickers
            if paths._ensemble_path(output_dir, condition, d, t).exists()
        )

        n_portfolios = sum(
            1 for d in dates
            if paths._portfolio_path(output_dir, condition, d).exists()
        )

        print(
            f"  {condition:<16}"
            f"  {n_sidecars:>5}/{total_sidecars:<5}"
            f"  {n_ensembles:>5}/{total_ensembles:<5}"
            f"  {n_portfolios:>5}/{total_portfolios:<5}"
        )
