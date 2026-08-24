"""
Training data generation for Phase 11 LoRA fine-tuning (P11-E1, DJ-054, DJ-059).

Responsibilities
----------------
1. generate_max_return_labels()    — forward-return labels for Technical Agent
2. generate_risk_adjusted_labels() — Sharpe-based labels for Fundamental Agent
3. format_as_jsonl()               — format labels into mlx_lm chat-format JSONL
4. FineTuneEvaluationResult        — Pydantic schema for three-tier evaluation results

Labeling rules (DJ-054):
  Technical Agent (max-return, 60-day horizon):
    label = "Buy"  if forward_return > +0.02
    label = "Sell" if forward_return < -0.02
    label = "Hold" otherwise

  Fundamental Agent (risk-adjusted, 60-day horizon):
    label = "Buy"  if Sharpe_60d > 0.8
    label = "Sell" if Sharpe_60d < 0.3
    label = "Hold" otherwise

Look-ahead bias is acknowledged and intentional (David §8.4, DJ-054):
  Reference strategies are hindsight-based training labels. The model learns
  from perfect-information examples during training; at inference time it only
  sees the indicators available as of the analysis date.

Format convention for format_as_jsonl (mlx_lm chat format):
  Each example: {"messages": [
    {"role": "system",    "content": "<agent system prompt>"},
    {"role": "user",      "content": "<analysis request with indicator values>"},
    {"role": "assistant", "content": "<JSON AgentSignal with decision=label>"}
  ]}
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_BUY_THRESHOLD = 0.02   # DJ-054
_SELL_THRESHOLD = -0.02  # DJ-054
_SHARPE_BUY = 0.8        # DJ-054
_SHARPE_SELL = 0.3       # DJ-054
_ANNUALISATION = math.sqrt(252)


# ---------------------------------------------------------------------------
# Internal OHLCV loader
# ---------------------------------------------------------------------------


def _load_close_series(ticker: str, data_dir: str) -> pd.Series | None:
    """
    Load close price series for ticker from the most-recent OHLCV Parquet.

    Returns a pd.Series with DatetimeIndex (ascending), or None if no file found.
    Prefers Adj Close / adjusted_close over Close / close when available.
    """
    from hifi.data.market_store import resolve_ohlcv_path

    try:
        path = resolve_ohlcv_path(ticker, data_dir)
    except FileNotFoundError:
        logger.warning("No OHLCV Parquet for %s in %s", ticker, data_dir)
        return None

    df = pd.read_parquet(path)

    # Normalise index to DatetimeIndex
    for col in ("date", "Date"):
        if col in df.columns:
            df = df.set_index(col)
            break
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    col = next(
        (c for c in ("adjusted_close", "Adj Close", "close", "Close") if c in df.columns),
        None,
    )
    if col is None:
        logger.warning("No close column for %s in %s", ticker, path)
        return None

    return df[col].dropna().astype(float)


def _load_ohlcv_df(ticker: str, data_dir: str) -> pd.DataFrame | None:
    """
    Load full OHLCV DataFrame for ticker with normalised column names.

    Returns a DataFrame with DatetimeIndex and columns:
      open, high, low, close, volume, adjusted_close (when available)
    Returns None when no Parquet file exists.
    """
    from hifi.data.market_store import resolve_ohlcv_path

    try:
        path = resolve_ohlcv_path(ticker, data_dir)
    except FileNotFoundError:
        logger.warning("No OHLCV Parquet for %s in %s", ticker, data_dir)
        return None

    df = pd.read_parquet(path)

    for col in ("date", "Date"):
        if col in df.columns:
            df = df.set_index(col)
            break
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Normalise column names to lowercase with underscores
    rename = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
        "Adj Close": "adjusted_close",
    }
    df = df.rename(columns=rename)
    return df


# ---------------------------------------------------------------------------
# Label generators (DJ-054)
# ---------------------------------------------------------------------------


def generate_max_return_labels(
    ticker: str,
    data_dir: str,
    horizon_days: int = 60,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Generate max-return reference strategy labels for a ticker.

    For each trading day t with at least horizon_days remaining:
      forward_return = (close[t+horizon_days] - close[t]) / close[t]
      label = "Buy"  if forward_return > threshold
      label = "Sell" if forward_return < -threshold
      label = "Hold" otherwise

    Returns DataFrame with columns: date, ticker, label, forward_return, horizon_days.
    Excludes any period where insufficient data remains (unlabeled).
    """
    _empty = pd.DataFrame(columns=["date", "ticker", "label", "forward_return", "horizon_days"])
    series = _load_close_series(ticker, data_dir)
    if series is None or len(series) < horizon_days + 1:
        return _empty

    prices = series.values
    dates = series.index

    rows = []
    for i in range(len(prices) - horizon_days):
        p0 = prices[i]
        if p0 == 0:
            continue
        p1 = prices[i + horizon_days]
        fwd = (p1 - p0) / p0
        if fwd > threshold:
            label = "Buy"
        elif fwd < -threshold:
            label = "Sell"
        else:
            label = "Hold"
        rows.append({
            "date": dates[i].date(),
            "ticker": ticker,
            "label": label,
            "forward_return": float(fwd),
            "horizon_days": horizon_days,
        })

    return pd.DataFrame(rows)


def generate_risk_adjusted_labels(
    ticker: str,
    data_dir: str,
    horizon_days: int = 60,
    sharpe_buy: float = 0.8,
    sharpe_sell: float = 0.3,
) -> pd.DataFrame:
    """
    Generate risk-adjusted reference strategy labels using rolling forward Sharpe.

    For each trading day t, compute Sharpe over the forward horizon_days window:
      forward_returns = daily_pct_change(close[t:t+horizon_days])
      sharpe = mean(forward_returns) / std(forward_returns) * sqrt(252)
      label = "Buy"  if sharpe > sharpe_buy
      label = "Sell" if sharpe < sharpe_sell
      label = "Hold" otherwise

    Look-ahead bias is acknowledged and intentional (David §8.4): these are
    reference strategy labels, not live predictions.

    Returns same DataFrame schema as generate_max_return_labels, plus sharpe_60d column.
    """
    _empty = pd.DataFrame(
        columns=["date", "ticker", "label", "sharpe_60d", "forward_return", "horizon_days"]
    )
    series = _load_close_series(ticker, data_dir)
    if series is None or len(series) < horizon_days + 2:
        return _empty

    prices = series.values
    dates = series.index

    rows = []
    for i in range(len(prices) - horizon_days):
        window = prices[i:i + horizon_days + 1]
        rets = np.diff(window) / window[:-1]

        std = np.std(rets, ddof=1)
        if std == 0 or np.isnan(std):
            continue

        sharpe = float(np.mean(rets) / std * _ANNUALISATION)
        p0, p1 = prices[i], prices[i + horizon_days]
        fwd = float((p1 - p0) / p0) if p0 != 0 else 0.0

        if sharpe > sharpe_buy:
            label = "Buy"
        elif sharpe < sharpe_sell:
            label = "Sell"
        else:
            label = "Hold"

        rows.append({
            "date": dates[i].date(),
            "ticker": ticker,
            "label": label,
            "sharpe_60d": sharpe,
            "forward_return": fwd,
            "horizon_days": horizon_days,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# JSONL formatter
# ---------------------------------------------------------------------------


def _parse_prompt_template(template_path: str) -> tuple[str, str]:
    """
    Parse a HiFi prompt template Markdown file.

    Returns (system_content, user_template):
      system_content  — everything between '## System\\n' and '\\n## User\\n'
      user_template   — everything after '## User\\n'
    """
    text = Path(template_path).read_text()
    if "## System" not in text or "## User" not in text:
        raise ValueError(f"Template missing ## System or ## User sections: {template_path}")

    after_system = text.split("## System", 1)[1]
    system_raw, user_raw = after_system.split("## User", 1)
    return system_raw.strip(), user_raw.strip()


def _technical_rationale(decision: str, ti: dict, rm: dict) -> tuple[str, str]:
    """
    Generate a template-based rationale and key_concern for the technical agent.

    References actual computed indicator values to teach citation patterns.
    Returns (rationale, key_concern).
    """
    parts = []
    rsi = ti.get("rsi")
    if rsi is not None:
        interp = "oversold" if rsi < 40 else ("overbought" if rsi > 60 else "neutral")
        parts.append(f"RSI of {rsi:.1f} indicates {interp} conditions")

    sma = ti.get("sma")
    ema = ti.get("ema")
    if sma is not None and ema is not None:
        trend = "bullish" if ema > sma else "bearish"
        parts.append(f"EMA {ema:.2f} vs SMA {sma:.2f} shows {trend} short-term momentum")

    macd_hist = ti.get("macd_hist")
    if macd_hist is not None:
        mom = "positive" if macd_hist > 0 else "negative"
        parts.append(f"MACD histogram of {macd_hist:.3f} confirms {mom} momentum")

    sharpe = rm.get("sharpe_252d")
    if sharpe is not None:
        parts.append(f"Sharpe ratio of {sharpe:.2f} over 252 days")

    rationale = ". ".join(parts) + "." if parts else "Analysis based on available technical data."

    vol = rm.get("hist_vol_252d")
    drawdown = rm.get("max_drawdown_252d")
    if vol is not None:
        key_concern = f"Historical volatility of {vol:.2f} annualised"
        if drawdown is not None:
            key_concern += f" with maximum drawdown of {drawdown:.2f}"
        key_concern += "."
    elif drawdown is not None:
        key_concern = f"Maximum drawdown of {drawdown:.2f} over 252 days."
    else:
        key_concern = "Insufficient risk data to quantify primary concern."

    return rationale, key_concern


def _fundamental_rationale(decision: str, fundamentals: dict) -> tuple[str, str]:
    """
    Generate a template-based rationale for the fundamental agent.

    For synthetic fundamentals, produces generic but schema-compliant text.
    """
    pe = fundamentals.get("pe_ratio")
    roe = fundamentals.get("roe")
    revenue_growth = fundamentals.get("revenue_growth_yoy")

    parts = []
    if pe is not None:
        valuation = "elevated" if pe > 25 else ("reasonable" if pe > 15 else "low")
        parts.append(f"P/E ratio of {pe:.1f} appears {valuation}")
    if roe is not None:
        parts.append(f"Return on equity of {roe:.2f} indicates capital efficiency")
    if revenue_growth is not None:
        direction = "growing" if revenue_growth > 0 else "contracting"
        parts.append(f"Revenue {direction} at {revenue_growth:.1%} year-over-year")

    rationale = ". ".join(parts) + "." if parts else "Analysis based on available fundamental data."
    key_concern = "Macro environment uncertainty and valuation risk at current levels."
    return rationale, key_concern


def format_as_jsonl(
    labels_df: pd.DataFrame,
    ticker: str,
    agent_type: str,
    data_dir: str,
    prompt_template_path: str,
    mcp_tool_outputs: dict | None = None,
) -> list[dict]:
    """
    Format labeled periods as mlx_lm-compatible JSONL chat messages.

    Each example: {"messages": [system, user, assistant]}
    - system: agent system prompt from prompt_template_path
    - user: analysis request with real or synthetic indicator values
    - assistant: JSON matching AgentSignal schema with decision from label

    When mcp_tool_outputs is None, computes indicator values from OHLCV
    data at data_dir using the deterministic engines (no MCP subprocess).

    Parameters
    ----------
    labels_df : pd.DataFrame
        From generate_max_return_labels() or generate_risk_adjusted_labels().
        Must have columns: date, ticker, label.
    ticker : str
        Ticker symbol.
    agent_type : str
        "technical" or "fundamental".
    data_dir : str
        Root data directory for loading OHLCV.
    prompt_template_path : str
        Path to the agent's Markdown prompt template (e.g. prompts/technical_v1.md).
    mcp_tool_outputs : dict | None
        Pre-computed MCP tool outputs indexed by date string. When None,
        values are computed from OHLCV.

    Returns
    -------
    list[dict]
        Ready for json.dumps per line.
    """
    if labels_df.empty:
        return []

    system_content, user_template = _parse_prompt_template(prompt_template_path)

    examples = []

    if agent_type == "technical":
        examples = _format_technical(
            labels_df, ticker, system_content, user_template, data_dir, mcp_tool_outputs
        )
    elif agent_type == "fundamental":
        examples = _format_fundamental(
            labels_df, ticker, system_content, user_template, data_dir, mcp_tool_outputs
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}. Expected 'technical' or 'fundamental'.")  # noqa: E501

    return examples


def _format_technical(
    labels_df: pd.DataFrame,
    ticker: str,
    system_content: str,
    user_template: str,
    data_dir: str,
    mcp_tool_outputs: dict | None,
) -> list[dict]:
    """Format examples for the Technical Agent."""
    from hifi.engines.risk import compute_risk_metrics
    from hifi.engines.technical import compute_technical_indicators

    ohlcv_df = _load_ohlcv_df(ticker, data_dir)

    examples = []
    for _, row in labels_df.iterrows():
        analysis_date = row["date"]
        label = row["label"]
        as_of = analysis_date if isinstance(analysis_date, date) else analysis_date.date()
        as_of_str = str(as_of)

        if mcp_tool_outputs is not None:
            ti_dict = mcp_tool_outputs.get(as_of_str, {}).get("technical_indicators", {})
            rm_dict = mcp_tool_outputs.get(as_of_str, {}).get("risk_metrics", {})
        elif ohlcv_df is not None:
            bars = _df_to_bars(ohlcv_df, ticker)
            ti = compute_technical_indicators(bars, as_of)
            rm = compute_risk_metrics(_make_dataset(ticker, bars), as_of)
            ti_dict = {k: v for k, v in ti.model_dump().items() if v is not None}
            rm_dict = {k: v for k, v in rm.model_dump().items() if v is not None}
        else:
            ti_dict, rm_dict = {}, {}

        rationale, key_concern = _technical_rationale(label, ti_dict, rm_dict)

        data_gaps = [k for k in ("rsi", "macd", "atr", "sharpe_252d") if k not in {**ti_dict, **rm_dict}]  # noqa: E501

        user_msg = user_template.replace("{ticker}", ticker).replace("{as_of_date}", as_of_str)
        user_msg = user_msg.replace("{technical_indicators}", json.dumps(ti_dict, indent=2))
        user_msg = user_msg.replace("{risk_metrics}", json.dumps(rm_dict, indent=2))
        user_msg = user_msg.replace("{data_gaps_list}", ", ".join(data_gaps) if data_gaps else "None")  # noqa: E501

        assistant_payload = {
            "decision": label,
            "confidence": 0.7,
            "rationale": rationale,
            "key_concern": key_concern,
            "time_horizon": "medium-term",
        }
        examples.append({
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": json.dumps(assistant_payload)},
            ]
        })

    return examples


def _format_fundamental(
    labels_df: pd.DataFrame,
    ticker: str,
    system_content: str,
    user_template: str,
    data_dir: str,
    mcp_tool_outputs: dict | None,
) -> list[dict]:
    """Format examples for the Fundamental Agent using synthetic fundamentals."""
    import random

    rng = random.Random(42)

    examples = []
    for _, row in labels_df.iterrows():
        analysis_date = row["date"]
        label = row["label"]
        as_of_str = str(analysis_date if isinstance(analysis_date, date) else analysis_date.date())

        if mcp_tool_outputs is not None:
            fin_ratios = mcp_tool_outputs.get(as_of_str, {}).get("financial_ratios", {})
            growth = mcp_tool_outputs.get(as_of_str, {}).get("growth_metrics", {})
            valuation = mcp_tool_outputs.get(as_of_str, {}).get("valuation_context", {})
            macro_snap = mcp_tool_outputs.get(as_of_str, {}).get("macro_snapshot", {})
        else:
            # Synthetic but realistic fundamental values for training data
            pe = round(rng.uniform(12, 35), 1)
            roe = round(rng.uniform(0.08, 0.35), 3)
            debt_equity = round(rng.uniform(0.1, 2.5), 2)
            rev_growth = round(rng.uniform(-0.05, 0.20), 3)
            earnings_growth = round(rng.uniform(-0.10, 0.30), 3)

            fin_ratios = {"pe_ratio": pe, "roe": roe, "debt_to_equity": debt_equity}
            growth = {"revenue_growth_yoy": rev_growth, "earnings_growth_yoy": earnings_growth}
            valuation = {"pe_1y_percentile": round(rng.uniform(0.2, 0.9), 2)}
            macro_snap = {
                "fed_funds_rate": round(rng.uniform(0.5, 5.5), 2),
                "cpi_yoy": round(rng.uniform(1.5, 8.0), 2),
            }

        rationale, key_concern = _fundamental_rationale(label, fin_ratios)

        data_gaps = [k for k in ("revenue_growth_yoy", "roe") if k not in fin_ratios and k not in growth]  # noqa: E501

        user_msg = user_template.replace("{ticker}", ticker).replace("{as_of_date}", as_of_str)
        user_msg = user_msg.replace("{financial_ratios}", json.dumps(fin_ratios, indent=2))
        user_msg = user_msg.replace("{growth_metrics}", json.dumps(growth, indent=2))
        user_msg = user_msg.replace("{valuation_context}", json.dumps(valuation, indent=2))
        user_msg = user_msg.replace("{macro_snapshot}", json.dumps(macro_snap, indent=2))
        user_msg = user_msg.replace("{data_gaps_list}", ", ".join(data_gaps) if data_gaps else "None")  # noqa: E501

        assistant_payload = {
            "decision": label,
            "confidence": 0.7,
            "rationale": rationale,
            "key_concern": key_concern,
        }
        examples.append({
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": json.dumps(assistant_payload)},
            ]
        })

    return examples


# ---------------------------------------------------------------------------
# OHLCVBar / OHLCVDataset conversion helpers
# ---------------------------------------------------------------------------


def _df_to_bars(df: pd.DataFrame, ticker: str) -> list:
    """Convert a normalised OHLCV DataFrame to list[OHLCVBar]."""
    from hifi.data.schemas import OHLCVBar

    bars = []
    for idx, row in df.iterrows():
        ts = idx.date() if hasattr(idx, "date") else idx
        try:
            bar = OHLCVBar(
                ticker=ticker,
                date=ts,
                open=float(row.get("open", row.get("close", 1.0))),
                high=float(row.get("high", row.get("close", 1.0))),
                low=float(row.get("low", row.get("close", 1.0))),
                close=float(row.get("close", 1.0)),
                volume=float(row.get("volume", 0.0)),
                adjusted_close=float(row["adjusted_close"])
                if "adjusted_close" in row and pd.notna(row["adjusted_close"])
                else None,
            )
            bars.append(bar)
        except Exception:
            continue
    return bars


def _make_dataset(ticker: str, bars: list) -> object:
    """Wrap list[OHLCVBar] in OHLCVDataset for risk engine."""
    from hifi.data.schemas import OHLCVBar, OHLCVDataset, ProvenanceRecord

    if not bars:
        from datetime import date as date_cls

        today = date_cls.today()
        return OHLCVDataset(
            ticker=ticker,
            bars=[],
            source="synthetic",
            fetched_at=datetime.now(UTC),
            date_from=today,
            date_to=today,
            provenance=ProvenanceRecord(source="synthetic", fetched_at=datetime.now(UTC)),
        )

    typed_bars: list[OHLCVBar] = bars
    dates = [b.date for b in typed_bars]
    from hifi.data.schemas import ProvenanceRecord

    return OHLCVDataset(
        ticker=ticker,
        bars=typed_bars,
        source="synthetic",
        fetched_at=datetime.now(UTC),
        date_from=min(dates),
        date_to=max(dates),
        provenance=ProvenanceRecord(source="synthetic", fetched_at=datetime.now(UTC)),
    )


# ---------------------------------------------------------------------------
# Evaluation result schema (DJ-058, P11-E4-T2)
# ---------------------------------------------------------------------------


class FineTuneEvaluationResult(BaseModel):
    """
    Three-tier evaluation result comparing base vs fine-tuned models (DJ-058).

    Captures individual quality (HR/GR), collective accuracy, and diversity
    impact for one (ticker, analysis_date) pair.

    diversity_preserved: True if finetuned diversity >= 0.9 * base diversity.
    gr_improved_technical: True if GR improvement >= 0.05.
    gr_improved_fundamental: True if GR improvement >= 0.05.
    """

    ticker: str
    analysis_date: str
    base_technical_gr: float = Field(ge=0.0, le=1.0)
    base_fundamental_gr: float = Field(ge=0.0, le=1.0)
    finetuned_technical_gr: float = Field(ge=0.0, le=1.0)
    finetuned_fundamental_gr: float = Field(ge=0.0, le=1.0)
    base_pairwise_diversity: float = Field(ge=0.0, le=1.0)
    finetuned_pairwise_diversity: float = Field(ge=0.0, le=1.0)
    base_disagreement_entropy: float = Field(ge=0.0)
    finetuned_disagreement_entropy: float = Field(ge=0.0)
    diversity_preserved: bool = Field(default=False)
    gr_improved_technical: bool = Field(default=False)
    gr_improved_fundamental: bool = Field(default=False)
    generated_at: str

    @model_validator(mode="after")
    def _compute_derived_fields(self) -> FineTuneEvaluationResult:
        """Auto-compute diversity_preserved and gr_improved from the raw metrics."""
        self.diversity_preserved = (
            self.finetuned_pairwise_diversity >= 0.9 * self.base_pairwise_diversity
        )
        self.gr_improved_technical = (
            self.finetuned_technical_gr - self.base_technical_gr >= 0.05
        )
        self.gr_improved_fundamental = (
            self.finetuned_fundamental_gr - self.base_fundamental_gr >= 0.05
        )
        return self
