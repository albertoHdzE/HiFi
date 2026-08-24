"""
HiFi Financial Calculator MCP Server (P2-E5).

Exposes six tools via stdio MCP transport (DJ-009) wrapping all Phase 2
deterministic financial engines. Agents connect to this server as a
subprocess and call tools via JSON-RPC over stdin/stdout.

Transport decision (DJ-009): stdio over SSE/HTTP for Phase 2-14.
Single-machine local deployment. No sockets, no authentication.
SSE/HTTP will be evaluated at Phase 15 for multi-container deployments.

Data access pattern per tool:
  1. Validate input parameters (ISO 8601 date, positive window, etc.)
  2. Load the relevant dataset from HIFI_DATA_DIR/market/ or macro/
  3. Call the appropriate engine function
  4. Return the result serialised to a JSON-safe dict (None fields as null)

Error contract: tools return a dict with an "error" key on failure, never
raise exceptions to the caller. Error keys:
  TICKER_NOT_FOUND   - no parquet file found for the requested ticker
  DATE_OUT_OF_RANGE  - no bars available on or before the requested date
  SNAPSHOT_INVALID   - snapshot_json could not be parsed
  COMPUTATION_ERROR  - unexpected failure inside an engine function

Data directory: controlled by the HIFI_DATA_DIR environment variable
(default: "data" relative to the process working directory). At test time,
the calling process sets this variable to the fixtures directory.

Fundamental data (FundamentalsSnapshot) is not yet stored as Parquet files
in Phase 2 (no write_fundamentals() in storage.py). The tools that need a
snapshot accept a `snapshot_json` parameter containing the snapshot serialised
as a JSON string. This is a Phase 2 simplification; Phase 7+ will implement
proper storage and load-from-registry semantics for fundamentals.

Phase 5 note: Each tool response includes a unique `call_id` field derived
from the input parameters. This is the audit trail hook for Phase 5 (Claim
Verification) — the verifier matches agent-cited numbers to `call_id`s.

Usage
-----
As a subprocess (the normal agent-side usage)::

    import subprocess
    proc = subprocess.Popen(
        ["python", "-m", "hifi.mcp.financial_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )

From the command line::

    HIFI_DATA_DIR=/path/to/data python -m hifi.mcp.financial_server
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("hifi-financial-calculator")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _data_dir() -> Path:
    """Return the data root directory from environment."""
    return Path(os.environ.get("HIFI_DATA_DIR", "data"))


def _call_id(**kwargs: Any) -> str:
    """
    Compute a short deterministic identifier for a tool call.

    Used by Phase 5 (Verification) to match agent claims to tool outputs.
    The ID is the first 12 hex digits of SHA-256(JSON(sorted params)).
    """
    payload = json.dumps(kwargs, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_ohlcv(ticker: str):
    """
    Load OHLCV bars for a ticker from the canonical market store.

    Path resolution is delegated to hifi.data.market_store, which prefers the
    nested data/market/{ticker}/ohlcv.parquet layout and falls back to the
    legacy flat fixtures. Tries read_ohlcv (HiFi metadata format) first; falls
    back to the raw frame reader for both the nested store and test fixtures,
    neither of which carries the HiFi metadata block.

    Raises
    ------
    FileNotFoundError
        When no layout has data for the ticker.
    """
    from hifi.data.market_store import resolve_ohlcv_path
    from hifi.data.storage import read_ohlcv

    path = resolve_ohlcv_path(ticker, _data_dir())
    try:
        return read_ohlcv(path)
    except ValueError:
        # No HiFi metadata block: nested store parquet or raw yfinance fixture
        return _load_raw_ohlcv(path, ticker)


def _load_raw_ohlcv(path: Path, ticker: str):
    """Load an OHLCV parquet that carries no HiFi metadata block.

    Handles both shapes in the store: the nested canonical layout (DatetimeIndex
    named Date, no "Adj Close") and the legacy flat yfinance fixtures (Date
    column, "Adj Close" present).
    """
    import math
    from datetime import datetime as _dt

    import pyarrow.parquet as pq

    from hifi.data.schemas import OHLCVBar, OHLCVDataset, ProvenanceRecord

    table = pq.read_table(path)
    df = table.to_pandas()
    # Nested store keeps the date in the index; flat fixtures keep it in a column.
    if "Date" not in df.columns:
        df = df.reset_index()
        for cand in ("Date", "date", "index"):
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break
    # Label provenance honestly: the nested store is real market data, the flat
    # files are test fixtures. Downstream provenance panels rely on this.
    src = "market_store" if path.name == "ohlcv.parquet" else "fixture"
    prov = ProvenanceRecord(source=src, fetched_at=_dt(2023, 4, 1))
    bars = []
    for _, row in df.iterrows():
        dt = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
        try:
            o = float(row["Open"])
            h = float(row["High"])
            lv = float(row["Low"])
            c = float(row["Close"])
            adj = float(row["Adj Close"]) if "Adj Close" in row else None
            vol = float(row.get("Volume", 0) or 0)
        except (TypeError, ValueError):
            continue
        if any(math.isnan(x) for x in [o, h, lv, c]):
            continue
        if adj is not None and math.isnan(adj):
            adj = None
        try:
            bars.append(OHLCVBar(ticker=ticker, date=dt, open=o, high=h, low=lv,
                                 close=c, volume=vol, adjusted_close=adj, source=src))
        except Exception:
            continue
    dates = [b.date for b in bars]
    from datetime import date as _date
    return OHLCVDataset(ticker=ticker, bars=bars, source=src,
                        fetched_at=_dt(2023, 4, 1),
                        date_from=min(dates) if dates else _date(2023, 1, 3),
                        date_to=max(dates) if dates else _date(2023, 4, 1),
                        provenance=prov)


def _load_all_macro() -> dict:
    """
    Load all macro parquet files from HIFI_DATA_DIR/macro/.

    Returns a dict mapping series_id -> MacroDataset.
    """
    from hifi.data.storage import read_macro

    pattern = str(_data_dir() / "macro" / "*.parquet")
    files = glob.glob(pattern)
    datasets = {}
    for f in files:
        try:
            ds = read_macro(Path(f))
            datasets[ds.series_id] = ds
        except Exception:
            logger.warning("Failed to load macro file: %s", f)
    return datasets


def _parse_date(date_str: str) -> date:
    """Parse an ISO 8601 date string."""
    return date.fromisoformat(date_str)


def _parse_snapshot(snapshot_json: str):
    """Deserialise a FundamentalsSnapshot from a JSON string."""
    from hifi.data.schemas import FundamentalsSnapshot

    try:
        data = json.loads(snapshot_json)
        return FundamentalsSnapshot.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Invalid snapshot_json: {exc}") from exc


def _price_at(dataset, as_of: date) -> float | None:
    """Return the closing price on or before as_of_date."""
    bars = sorted(
        [b for b in dataset.bars if b.date <= as_of], key=lambda b: b.date
    )
    return bars[-1].close if bars else None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_financial_ratios(
    ticker: str,
    date: str,
    snapshot_json: str,
) -> dict:
    """
    Compute fundamental valuation and profitability ratios.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. "AAPL").
    date : str
        ISO 8601 date for which to look up the closing price.
    snapshot_json : str
        JSON-serialised FundamentalsSnapshot from Phase 1 data acquisition.

    Returns
    -------
    dict
        FinancialRatioResult fields (pe, pb, ps, roe, roa, debt_equity, ...)
        or an error dict with "error" and "detail" keys.
    """
    from hifi.engines.fundamental import compute_financial_ratios

    cid = _call_id(tool="get_financial_ratios", ticker=ticker, date=date)
    try:
        as_of = _parse_date(date)
        snapshot = _parse_snapshot(snapshot_json)
        dataset = _load_ohlcv(ticker)
        current_price = _price_at(dataset, as_of)
        if current_price is None:
            return {
                "error": "DATE_OUT_OF_RANGE",
                "detail": f"No bars for {ticker} on or before {date}",
                "call_id": cid,
            }
        result = compute_financial_ratios(snapshot, current_price)
        return {**result.model_dump(), "call_id": cid}
    except FileNotFoundError:
        return {"error": "TICKER_NOT_FOUND", "detail": f"No data for {ticker}", "call_id": cid}
    except ValueError as exc:
        return {"error": "SNAPSHOT_INVALID", "detail": str(exc), "call_id": cid}
    except Exception as exc:
        logger.exception("get_financial_ratios failed for %s", ticker)
        return {"error": "COMPUTATION_ERROR", "detail": str(exc), "call_id": cid}


@mcp.tool()
def get_growth_metrics(snapshot_json: str) -> dict:
    """
    Compute growth rates and profitability margins.

    Parameters
    ----------
    snapshot_json : str
        JSON-serialised FundamentalsSnapshot.

    Returns
    -------
    dict
        GrowthMetricsResult fields (net_margin; YoY fields None in Phase 2).
    """
    from hifi.engines.fundamental import compute_growth_metrics

    cid = _call_id(tool="get_growth_metrics", snapshot=snapshot_json[:32])
    try:
        snapshot = _parse_snapshot(snapshot_json)
        result = compute_growth_metrics(snapshot)
        return {**result.model_dump(), "call_id": cid}
    except ValueError as exc:
        return {"error": "SNAPSHOT_INVALID", "detail": str(exc), "call_id": cid}
    except Exception as exc:
        logger.exception("get_growth_metrics failed")
        return {"error": "COMPUTATION_ERROR", "detail": str(exc), "call_id": cid}


@mcp.tool()
def get_technical_indicators(
    ticker: str,
    date: str,
    window: int = 20,
) -> dict:
    """
    Compute technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR.

    Parameters
    ----------
    ticker : str
    date : str
        ISO 8601 date. Only bars on or before this date are used.
    window : int, default 20
        Window in bars for SMA, EMA, and Bollinger Bands.

    Returns
    -------
    dict
        TechnicalIndicatorsResult fields or an error dict.
    """
    from hifi.engines.technical import compute_technical_indicators

    cid = _call_id(tool="get_technical_indicators", ticker=ticker, date=date, window=window)
    try:
        as_of = _parse_date(date)
        dataset = _load_ohlcv(ticker)
        result = compute_technical_indicators(dataset.bars, as_of, window)
        return {**result.model_dump(), "call_id": cid}
    except FileNotFoundError:
        return {"error": "TICKER_NOT_FOUND", "detail": f"No data for {ticker}", "call_id": cid}
    except Exception as exc:
        logger.exception("get_technical_indicators failed for %s", ticker)
        return {"error": "COMPUTATION_ERROR", "detail": str(exc), "call_id": cid}


@mcp.tool()
def get_risk_metrics(
    ticker: str,
    date: str,
    benchmark_ticker: str | None = None,
    risk_free_rate: float = 0.0,
    window: int = 252,
) -> dict:
    """
    Compute portfolio risk metrics: volatility (20d/60d/252d), Sharpe, max
    drawdown, VaR, and beta.

    Parameters
    ----------
    ticker : str
    date : str
        ISO 8601 date. Trailing window ends here.
    benchmark_ticker : str | None
        Benchmark ticker for beta computation (typically "SPY"). None → beta None.
    risk_free_rate : float, default 0.0
        Annualised risk-free rate for Sharpe (e.g., 0.04 for 4%).
    window : int, default 252
        Trailing window for Sharpe and max drawdown.

    Returns
    -------
    dict
        RiskMetricsResult fields or an error dict.
    """
    from hifi.engines.risk import compute_risk_metrics

    cid = _call_id(
        tool="get_risk_metrics", ticker=ticker, date=date,
        benchmark=benchmark_ticker, rf=risk_free_rate, window=window,
    )
    try:
        as_of = _parse_date(date)
        dataset = _load_ohlcv(ticker)
        benchmark = None
        if benchmark_ticker:
            try:
                benchmark = _load_ohlcv(benchmark_ticker)
            except FileNotFoundError:
                logger.warning("Benchmark %s not found; beta will be None", benchmark_ticker)
        result = compute_risk_metrics(dataset, as_of, benchmark, risk_free_rate, window)
        return {**result.model_dump(), "call_id": cid}
    except FileNotFoundError:
        return {"error": "TICKER_NOT_FOUND", "detail": f"No data for {ticker}", "call_id": cid}
    except Exception as exc:
        logger.exception("get_risk_metrics failed for %s", ticker)
        return {"error": "COMPUTATION_ERROR", "detail": str(exc), "call_id": cid}


@mcp.tool()
def get_valuation_context(
    ticker: str,
    date: str,
    snapshot_json: str,
) -> dict:
    """
    Compute valuation context: trailing P/E percentile and 52-week price ratios.

    Parameters
    ----------
    ticker : str
    date : str
        ISO 8601 date (as_of_date for the valuation snapshot).
    snapshot_json : str
        JSON-serialised FundamentalsSnapshot.

    Returns
    -------
    dict
        ValuationResult fields or an error dict.
    """
    from hifi.engines.fundamental import compute_valuation_context

    cid = _call_id(tool="get_valuation_context", ticker=ticker, date=date)
    try:
        as_of = _parse_date(date)
        snapshot = _parse_snapshot(snapshot_json)
        dataset = _load_ohlcv(ticker)
        result = compute_valuation_context(snapshot, dataset, as_of)
        return {**result.model_dump(), "call_id": cid}
    except FileNotFoundError:
        return {"error": "TICKER_NOT_FOUND", "detail": f"No data for {ticker}", "call_id": cid}
    except ValueError as exc:
        return {"error": "SNAPSHOT_INVALID", "detail": str(exc), "call_id": cid}
    except Exception as exc:
        logger.exception("get_valuation_context failed for %s", ticker)
        return {"error": "COMPUTATION_ERROR", "detail": str(exc), "call_id": cid}


@mcp.tool()
def get_macro_snapshot(date: str) -> dict:
    """
    Return a cross-section of macroeconomic indicators as of a date.

    Loads all macro parquet files from HIFI_DATA_DIR/macro/ and extracts
    point-in-time values via forward-filling.

    Parameters
    ----------
    date : str
        ISO 8601 date.

    Returns
    -------
    dict
        MacroSnapshotResult fields (fed_funds_rate, cpi_yoy, unemployment_rate,
        yield_10y, yield_2y, yield_curve_slope, vix, gdp_growth) or error dict.
    """
    from hifi.engines.macro import compute_macro_snapshot

    cid = _call_id(tool="get_macro_snapshot", date=date)
    try:
        as_of = _parse_date(date)
        datasets = _load_all_macro()
        if not datasets:
            return {
                "error": "NO_MACRO_DATA",
                "detail": f"No macro parquet files in {_data_dir() / 'macro'}",
                "call_id": cid,
            }
        result = compute_macro_snapshot(datasets, as_of)
        return {**result.model_dump(), "call_id": cid}
    except Exception as exc:
        logger.exception("get_macro_snapshot failed")
        return {"error": "COMPUTATION_ERROR", "detail": str(exc), "call_id": cid}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
