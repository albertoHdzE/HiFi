"""Filesystem layout and universe lookups for the live and walk-forward runs.

Every path the nightly cycle reads or writes is derived here, so a question like
"where do arm B's decisions land" has one answer. ``_ROOT`` is resolved from this
file's location rather than from the working directory: the cycle is launched by
launchd, by make, and by hand, and only the package location is invariant across
the three.

``_DATA_DIR`` deliberately ignores ``HIFI_DATA_DIR``. The MCP servers honour that
variable so tests can point them at a fixture tree, but the live orchestrator
must not be redirectable by an environment variable that a stray export could
set: a night that silently wrote its decision records into a temp directory
would look like a night that never ran.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

#: Repository root: src/hifi/live/paths.py -> up four -> the checkout.
_ROOT = Path(__file__).resolve().parents[3]


_DATA_DIR = str(_ROOT / "data")
_OUTPUT_DIR = str(_ROOT / "data" / "live")
_DB_PATH = str(_ROOT / "data" / "knowledge.lance")

_EDGAR_NAMESPACE = "hifi-dev-sec"
_CONTEXT_NAMESPACE = "hifi-live-context"


def _get_tickers(smoke: bool = False) -> list[str]:
    if smoke:
        from hifi.data.smoke_universe import SMOKE_UNIVERSE
        return [row["ticker"] for row in SMOKE_UNIVERSE]
    from hifi.data.universe import PHASE14_UNIVERSE
    return [row["ticker"] for row in PHASE14_UNIVERSE]


def _get_sectors() -> dict[str, str]:
    from hifi.data.universe import PHASE14_UNIVERSE
    return {row["ticker"]: row["sector"] for row in PHASE14_UNIVERSE}


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _account_dir(account: str) -> Path:
    return Path(_OUTPUT_DIR) / account


def _decisions_log(account: str) -> Path:
    return _account_dir(account) / "decisions.jsonl"


def _breaker_log(account: str) -> Path:
    return _account_dir(account) / "circuit_breakers.jsonl"


def _hwm_path(account: str) -> Path:
    return _account_dir(account) / "hwm.json"


def _run_id(condition: str, date: str, ticker: str) -> str:
    return f"{condition}-{date}-{ticker}"


def _sidecar_path(data_dir: str, condition: str, date: str, ticker: str, agent_type: str) -> Path:
    rid = _run_id(condition, date, ticker)
    return Path(data_dir) / "runs" / rid / f"{ticker}_{agent_type}.json"


def _ensemble_path(output_dir: str, condition: str, date: str, ticker: str) -> Path:
    year, month, _ = date.split("-")
    return Path(output_dir) / condition / year / month / f"{ticker}.json"


def _portfolio_path(output_dir: str, condition: str, date: str) -> Path:
    year, month, _ = date.split("-")
    return Path(output_dir) / condition / year / month / "portfolio.json"


def _resolve_tickers(tickers_arg: list[str] | None) -> list[str]:
    """An explicit ticker list, or the full universe when none is given.

    The walk-forward CLI's spelling of ``_get_tickers()``; kept as a separate
    name because it takes the parsed argument rather than a smoke flag, but it
    delegates so the two cannot disagree about what "the universe" means.
    """
    return list(tickers_arg) if tickers_arg else _get_tickers()


def _resolve_dates(period: str, start_date: str | None, end_date: str | None) -> list[str]:
    from hifi.simulation.schedule import (  # noqa: PLC0415
        WalkForwardPeriod,
        generate_month_ends,
        get_multi_period_dates,
        get_period_dates,
    )

    if start_date and end_date:
        return generate_month_ends(start_date, end_date)
    if period == "all":
        return get_multi_period_dates([
            WalkForwardPeriod.VALIDATION,
            WalkForwardPeriod.HELD_OUT_TEST,
            WalkForwardPeriod.WALK_FORWARD,
        ])
    return get_period_dates(period)
