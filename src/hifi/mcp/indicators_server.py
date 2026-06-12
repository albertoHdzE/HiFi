"""
HiFi Extended Technical Indicators MCP Server (P8-E0).

Exposes additional TA indicators via FastMCP stdio transport using pandas-ta.
Must be run with venvs/ta/bin/python to access pandas-ta (pandas 1.5.3 stack).

The main process calls this server via call_tool() with python_executable set to
venvs/ta/bin/python (DJ-035). PYTHONPATH must include the project src/ directory
so that the hifi package is importable inside venvs/ta/.

Transport (DJ-009): stdio over SSE/HTTP, matching the financial_server pattern.

Tools exposed:
    get_extended_indicators  —  ADX, Stochastic (%K/%D), OBV, Williams %R, CCI

Pandas-ta (and pandas 1.5.3) are imported lazily inside tool functions so that
the module can be imported in the main environment (for structure inspection)
without triggering ImportError. At runtime in venvs/ta/, the imports succeed.

Data directory: controlled by HIFI_DATA_DIR environment variable (default: "data").

Error contract: same as financial_server.py — tools return a dict with an "error"
key on failure, never raise exceptions to the caller.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("hifi-indicators")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _data_dir() -> Path:
    """Return the data root directory from environment."""
    return Path(os.environ.get("HIFI_DATA_DIR", "data"))


def _call_id(**kwargs: Any) -> str:
    """Compute a short deterministic call identifier (first 12 hex of SHA-256)."""
    payload = json.dumps(kwargs, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Data loading helpers (pandas 1.5.3 compatible, no hifi.data imports)
# ---------------------------------------------------------------------------


def _load_ohlcv_df(ticker: str):
    """
    Load OHLCV data for a ticker as a pandas DataFrame.

    Uses pandas read_parquet directly (no hifi.data.storage imports) to avoid
    dependency on pandas >= 3.0 features used in the main environment.

    Returns a DataFrame with columns including Date, Open, High, Low, Close, Volume.
    Raises FileNotFoundError when no parquet file exists for the ticker.
    """
    import pandas as pd

    pattern = str(_data_dir() / "market" / f"{ticker}_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No OHLCV data found for ticker '{ticker}'")

    df = pd.read_parquet(files[-1])
    df = df.reset_index()

    # Normalise column names: lowercase, strip spaces
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "index" in df.columns:
        df["date"] = pd.to_datetime(df["index"])

    df = df.sort_values("date").reset_index(drop=True)
    return df


def _filter_to_date(df, as_of_date_str: str):
    """Return only rows with date <= as_of_date."""
    import pandas as pd

    as_of = pd.Timestamp(as_of_date_str)
    return df[df["date"] <= as_of].copy()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_extended_indicators(
    ticker: str,
    date: str,
    window: int = 14,
) -> dict:
    """
    Compute extended technical indicators using pandas-ta.

    Supplements the Phase 2 basic indicators (SMA, EMA, RSI, MACD, Bollinger,
    ATR) with five additional indicators that require pandas-ta:

    - adx            Average Directional Index [0, 100]; > 25 = trending
    - stoch_k        Stochastic %K [0, 100]; > 80 overbought, < 20 oversold
    - stoch_d        Stochastic %D [0, 100]; signal line of %K
    - obv            On-Balance Volume (cumulative volume flow)
    - willr          Williams %R [-100, 0]; > -20 overbought, < -80 oversold
    - cci            Commodity Channel Index; > 100 overbought, < -100 oversold

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. "AAPL").
    date : str
        ISO 8601 analysis date. Only bars on or before this date are used.
    window : int, default 14
        Lookback window for ADX, Stochastic, Williams %R, and CCI.

    Returns
    -------
    dict
        Extended indicator values (floats or null) plus a call_id field.
        On error: {"error": "<CODE>", "detail": "<message>", "call_id": "..."}.
    """
    cid = _call_id(tool="get_extended_indicators", ticker=ticker, date=date, window=window)

    # Load data first (no pandas_ta needed). Returns TICKER_NOT_FOUND before
    # attempting the pandas-ta import so the unit tests work in the main env.
    try:
        df = _load_ohlcv_df(ticker)
    except FileNotFoundError:
        return {"error": "TICKER_NOT_FOUND", "detail": f"No data for {ticker}", "call_id": cid}

    try:
        df = _filter_to_date(df, date)
        if df.empty:
            return {
                "error": "DATE_OUT_OF_RANGE",
                "detail": f"No bars for {ticker} on or before {date}",
                "call_id": cid,
            }

        # pandas-ta expects OHLCV columns with these exact names.
        # Import here (lazy) so the module is importable in the main env without pandas-ta.
        import pandas_ta as ta  # noqa: PLC0415

        required = {"open", "high", "low", "close", "volume"}
        available = set(df.columns)
        missing = required - available
        if missing:
            return {
                "error": "COMPUTATION_ERROR",
                "detail": f"Missing columns: {missing}",
                "call_id": cid,
            }

        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # --- ADX ---
        adx_result = ta.adx(high, low, close, length=window)
        adx_val = None
        if adx_result is not None and not adx_result.empty:
            col = [c for c in adx_result.columns if c.startswith("ADX_")]
            if col:
                v = adx_result[col[0]].iloc[-1]
                adx_val = float(v) if v == v else None  # NaN check

        # --- Stochastic ---
        stoch_result = ta.stoch(high, low, close, k=window, d=3)
        stoch_k_val = None
        stoch_d_val = None
        if stoch_result is not None and not stoch_result.empty:
            k_cols = [c for c in stoch_result.columns if c.startswith("STOCHk_")]
            d_cols = [c for c in stoch_result.columns if c.startswith("STOCHd_")]
            if k_cols:
                v = stoch_result[k_cols[0]].iloc[-1]
                stoch_k_val = float(v) if v == v else None
            if d_cols:
                v = stoch_result[d_cols[0]].iloc[-1]
                stoch_d_val = float(v) if v == v else None

        # --- OBV ---
        obv_result = ta.obv(close, volume)
        obv_val = None
        if obv_result is not None and len(obv_result) > 0:
            v = obv_result.iloc[-1]
            obv_val = float(v) if v == v else None

        # --- Williams %R ---
        willr_result = ta.willr(high, low, close, length=window)
        willr_val = None
        if willr_result is not None and len(willr_result) > 0:
            v = willr_result.iloc[-1]
            willr_val = float(v) if v == v else None

        # --- CCI ---
        cci_result = ta.cci(high, low, close, length=window)
        cci_val = None
        if cci_result is not None and len(cci_result) > 0:
            v = cci_result.iloc[-1]
            cci_val = float(v) if v == v else None

        return {
            "ticker": ticker,
            "as_of_date": date,
            "adx": adx_val,
            "stoch_k": stoch_k_val,
            "stoch_d": stoch_d_val,
            "obv": obv_val,
            "willr": willr_val,
            "cci": cci_val,
            "window": window,
            "call_id": cid,
        }

    except Exception as exc:
        logger.exception("get_extended_indicators failed for %s", ticker)
        return {"error": "COMPUTATION_ERROR", "detail": str(exc), "call_id": cid}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
