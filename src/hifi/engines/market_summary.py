"""Engine-computed market summaries for agent context (Phase 20, branch
feature/context-engine-summaries).

Three bounded, deterministic one-liners that situate tonight's decision in
market state — computed by engines from stored data, never by the LLM,
never from prompt-resident price series (DJ-002 discipline):

1. **Regime** (DJ-089b classifier, live-wired): delegates to
   :func:`hifi.data.regime.classify_regime`, reading the *canonical* nested
   SPY store and per-series FRED parquets. The previous live path
   (``ensemble_runner._get_regime_label``) reads ``market/SPY.parquet`` and
   ``macro/macro.parquet`` — both nonexistent since the DJ-120 store migration
   — so it has been silently returning "neutral". This module restores real
   inputs and stamps every input's last date into the output so staleness is
   visible, not hidden.

2. **Relative strength vs sector**: ticker's n-session simple return minus the
   median n-session return of same-sector universe peers. Median (not mean)
   resists single-peer distortion; >=3 qualifying peers required.

3. **Book VaR**: historical simulation of DAILY portfolio returns over a
   trailing window. Date-intersection alignment across holdings — never
   positional tail-splicing — and weights renormalized over covered names
   (the correct pattern, per audit finding C15; ``risk_manager`` carries the
   flawed variant on its own fix queue). Output is a ONE-DAY VaR estimated
   from ``window`` sessions and is labelled as such everywhere it appears —
   the horizon lives in the wording, never in a suffix (C18 lesson).

Point-in-time discipline: every series is filtered to ``index <= as_of``
before any computation. Nothing forward-looking can enter, including at edges.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REL_STRENGTH_WINDOW = 20       # sessions
REL_STRENGTH_MIN_PEERS = 3
VAR_WINDOW = 60                # estimation sessions
VAR_MIN_ALIGNED = 30           # aligned daily observations required
_VAR_LEVEL = 5                 # percentile -> 95% VaR


# ---------------------------------------------------------------------------
# Loaders (canonical stores only; cached per process)
# ---------------------------------------------------------------------------


def _nested_path(data_dir: str, ticker: str):
    from pathlib import Path

    return Path(data_dir) / "market" / ticker / "ohlcv.parquet"


@lru_cache(maxsize=512)
def _load_close(data_dir: str, ticker: str) -> pd.Series | None:
    """Close-price series (DatetimeIndex, ascending) from the canonical store."""
    path = _nested_path(data_dir, ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("market_summary: unreadable %s (%s)", path, exc)
        return None
    if df.columns.dtype == object or "close" in [c.lower() for c in df.columns]:
        df.columns = [str(c).lower() for c in df.columns]
    close_col = next((c for c in df.columns if c.startswith("adj") or c == "close"), None)
    if close_col is None:
        return None
    s = df[close_col]
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except (TypeError, ValueError):
            return None
    return s.sort_index().rename(ticker)


def _load_fred_series(data_dir: str, stem: str) -> pd.Series | None:
    """Per-series FRED parquet (columns 'date'/'value') as a dated Series."""
    from pathlib import Path

    path = Path(data_dir) / "macro" / f"{stem}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("market_summary: unreadable %s (%s)", path, exc)
        return None
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "value" not in cols:
        return None
    s = pd.Series(df[cols["value"]].values,
                  index=pd.to_datetime(df[cols["date"]]))
    return s.dropna().sort_index()


def _last_dates(spy_last, fed_last, vix_last) -> dict[str, str | None]:
    fmt = lambda x: x.strftime("%Y-%m-%d") if x is not None else None  # noqa: E731
    return {"spy": fmt(spy_last), "fedfunds": fmt(fed_last), "vix": fmt(vix_last)}


# ---------------------------------------------------------------------------
# 1. Regime
# ---------------------------------------------------------------------------


def regime_snapshot(as_of_date: str, data_dir: str) -> dict | None:
    """DJ-089b classification from LIVE data paths, with input freshness."""
    try:
        from hifi.data.regime import classify_regime
    except ImportError:  # pragma: no cover
        return None

    spy = _load_close(data_dir, "SPY")
    fed = _load_fred_series(data_dir, "FEDFUNDS")
    vix = _load_fred_series(data_dir, "VIXCLS")
    spy_stamps = _last_dates(
        spy.index.max() if spy is not None else None,
        fed.index.max() if fed is not None else None,
        vix.index.max() if vix is not None else None,
    )
    if spy is None or fed is None:
        return {
            "label": None,
            "reason": "benchmark data unavailable",
            "inputs": spy_stamps,
        }

    as_of = pd.Timestamp(as_of_date)
    spy_bounded = spy[spy.index <= as_of]
    wide = pd.DataFrame({
        "fed_funds_rate": fed,
        "vix": vix,
    }).dropna(how="all")

    label = classify_regime(as_of_date, spy_bounded.to_frame(name="Close"), wide)
    return {"label": label, "reason": None, "inputs": spy_stamps}


# ---------------------------------------------------------------------------
# 2. Relative strength vs sector peers
# ---------------------------------------------------------------------------


def _n_session_return(close: pd.Series, as_of: pd.Timestamp, n: int) -> float | None:
    bounded = close[close.index <= as_of].dropna()
    if len(bounded) < n + 1:
        return None
    base = float(bounded.iloc[-(n + 1)])
    if base <= 0:
        return None
    return float(bounded.iloc[-1]) / base - 1.0


def relative_strength(ticker: str, sector: str | None, as_of_date: str,
                      data_dir: str, n: int = REL_STRENGTH_WINDOW) -> dict | None:
    """Ticker n-session simple return vs median same-sector peer return."""
    from hifi.data.universe import PHASE14_UNIVERSE

    as_of = pd.Timestamp(as_of_date)
    mine = _load_close(data_dir, ticker)
    ret_self = _n_session_return(mine, as_of, n) if mine is not None else None
    if ret_self is None:
        return None

    if sector is None:
        rows = [r for r in PHASE14_UNIVERSE if r["ticker"] == ticker]
        sector = rows[0]["sector"] if rows else None
    if sector is None:
        return None

    peer_rets: list[float] = []
    for row in PHASE14_UNIVERSE:
        peer = row["ticker"]
        if peer == ticker or row["sector"] != sector:
            continue
        closes = _load_close(data_dir, peer)
        if closes is None:
            continue
        r = _n_session_return(closes, as_of, n)
        if r is not None:
            peer_rets.append(r)

    if len(peer_rets) < REL_STRENGTH_MIN_PEERS:
        return None
    median_peer = float(np.median(peer_rets))
    return {
        "ticker_return": round(ret_self, 4),
        "peer_median": round(median_peer, 4),
        "delta_pp": round((ret_self - median_peer) * 100.0, 2),
        "n_peers": len(peer_rets),
        "window_sessions": n,
    }


# ---------------------------------------------------------------------------
# 3. Book VaR (historical simulation, date-aligned)
# ---------------------------------------------------------------------------


def book_var_95(weights: dict[str, float], as_of_date: str, data_dir: str,
                window: int = VAR_WINDOW) -> dict | None:
    """One-day historical-simulation VaR(95%) of the current book.

    Alignment rule (C15 corrected): portfolio daily returns exist only on
    dates present for EVERY covered holding — intersection, never positional
    splice. Weights are renormalized over covered names; a name without data
    can never silently halve the estimate. Requires >= VAR_MIN_ALIGNED
    aligned sessions, else None (reported, not guessed).
    """
    as_of = pd.Timestamp(as_of_date)
    series: dict[str, pd.Series] = {}
    covered: dict[str, float] = {}
    for tkr, w in weights.items():
        if w <= 0:
            continue
        close = _load_close(data_dir, tkr)
        if close is None:
            continue
        bounded = close[close.index <= as_of].dropna()
        rets = bounded.pct_change().dropna()
        rets = rets.tail(window + 1)
        if len(rets) < 2:
            continue
        series[tkr] = rets
        covered[tkr] = float(w)

    if not series:
        return None

    total_w = sum(covered.values())
    norm = {t: w / total_w for t, w in covered.items()}

    # Intersection of dates across all covered holdings.
    common: pd.Index | None = None
    for rets in series.values():
        common = rets.index if common is None else common.intersection(rets.index)
    if common is None or len(common) < VAR_MIN_ALIGNED:
        n_aligned = 0 if common is None else len(common)
        return {
            "var_95_1d": None,
            "reason": f"insufficient aligned history ({n_aligned} sessions)",
            "covered_names": sorted(norm),
            "window_sessions": window,
        }

    port = pd.Series(0.0, index=common)
    for tkr, rets in series.items():
        port = port.add(rets.loc[common] * norm[tkr], fill_value=0.0)

    var = float(-np.percentile(port.to_numpy(), _VAR_LEVEL))
    return {
        "var_95_1d": round(max(var, 0.0), 4),
        "reason": None,
        "covered_names": sorted(norm),
        "aligned_sessions": int(len(common)),
        "window_sessions": window,
        "renormalized_from": sorted(weights),
    }
