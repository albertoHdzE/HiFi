"""
E5-T5: Drift monitor calibration on historical regime data (P13-E5, DJ-077).

Split: baseline period / recent (post-rate-shock) period, all splits at 2022-01-01.

  KS + chi-sq baseline: 2020-01-02 to 2021-12-31
  CUSUM baseline      : 2021-01-04 to 2021-12-31  (stable bull year; COVID
                         excluded to avoid extreme 2020-Q1 events inflating mean)
  Recent (all three)  : 2022-01-03 to 2023-06-30

Purpose: verify that all three DriftMonitor methods correctly identify
the 2022 Fed rate-shock regime change using only deterministic market data
(no LLMs required).

Expected results
----------------
  KS data drift   : ALERT  — realized_vol and RSI distributions shift
                    (rate shock increased vol and pushed RSI into oversold territory)
  Chi-sq agent    : ALERT  — momentum-based decision proportions shift
                    (bear market shifts Sell fraction; bull market shifts Buy fraction)
  CUSUM collective: ALERT  — herding proxy persistently elevated in 2022 sell-off
                    (fraction of tickers below 50d MA: ~28% in 2021, ~50% in 2022)

Herding proxy (CUSUM)
---------------------
Fraction of panel tickers currently trading below their 50-day moving average.
High value → most stocks in downtrend → agents are likely to agree on Sell
(collective "risk-off" herding). This maps to the Phase 12 herding_coefficient
concept: when all agents face similarly bearish data, they tend to agree.
The 2022 bear market (rate-shock driven) produced a sustained period where
most tickers (ex-energy) were below their 50d MA, lifting this fraction from
a 2021 baseline of ~28% to a 2022 mean of ~50%.

Output: tests/fixtures/baseline/phase13_drift_calibration.json

Tickers
-------
  KS + chi-sq : AAPL, JPM, XOM  (Phase 13 evaluation tickers)
  CUSUM panel : all 16 available tickers
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hifi.collective.drift import DriftMonitor  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKET_DIR = REPO_ROOT / "data" / "market"
OUTPUT_PATH = REPO_ROOT / "tests" / "fixtures" / "baseline" / "phase13_drift_calibration.json"

# KS and chi-sq: two full years of pre-shock data (includes mild COVID recovery)
KS_CHISQ_BASELINE_START = date(2020, 1, 2)
KS_CHISQ_BASELINE_END = date(2021, 12, 31)

# CUSUM: 2021 only — stable bull-market year immediately before the shock.
# Excluding 2020 avoids the COVID-crash extreme (vol > 80%) dominating the
# baseline statistics and inflating the threshold above 2022 levels.
CUSUM_BASELINE_START = date(2021, 1, 4)
CUSUM_BASELINE_END = date(2021, 12, 31)

# All monitors share the same "recent" window (post-shock regime)
RECENT_START = date(2022, 1, 3)
RECENT_END = date(2023, 6, 30)

PRIMARY_TICKERS = ["AAPL", "JPM", "XOM"]
ALL_TICKERS = [
    "AAPL", "AMZN", "BAC", "CAT", "CVX", "GOOGL", "GS", "JNJ",
    "JPM", "MSFT", "NEE", "NVDA", "SPY", "UNH", "WMT", "XOM",
]

# Momentum thresholds for proxy agent decisions
BUY_THRESHOLD = 0.05    # 20-day trailing return > +5% → Buy
SELL_THRESHOLD = -0.05  # 20-day trailing return < −5% → Sell
MOMENTUM_WINDOW = 20    # trading days

VOL_WINDOW = 20         # days for rolling realized vol
RSI_PERIOD = 14


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------


def _load_price(ticker: str) -> pd.Series:
    """Return adjusted_close (or close fallback) for one ticker, DatetimeIndex."""
    files = list(MARKET_DIR.glob(f"{ticker}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet found for {ticker} in {MARKET_DIR}")
    df = pd.read_parquet(files[0])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["adjusted_close"].where(df["adjusted_close"].notna(), df["close"])


def _realized_vol(price: pd.Series) -> pd.Series:
    """20-day rolling annualized realized volatility from log returns."""
    log_ret = np.log(price / price.shift(1))
    return log_ret.rolling(window=VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(252)


def _rsi(price: pd.Series, n: int = RSI_PERIOD) -> pd.Series:
    """
    RSI using Wilder's EWM (mirrors src/hifi/engines/technical.py).

    alpha = 1/n, min_periods=n so early bars return NaN.
    """
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs_vals = np.where(
            avg_loss.values == 0.0,
            np.where(avg_gain.values == 0.0, np.nan, np.inf),
            avg_gain.values / avg_loss.values,
        )
    rs = pd.Series(rs_vals, index=price.index)
    return 100.0 - (100.0 / (1.0 + rs))


def _momentum_decisions(price: pd.Series) -> pd.Series:
    """
    Proxy agent decision from 20-day trailing return.

    > +BUY_THRESHOLD  → "Buy"
    < SELL_THRESHOLD  → "Sell"
    Otherwise         → "Hold"
    """
    ret = price.pct_change(periods=MOMENTUM_WINDOW)
    d = pd.Series("Hold", index=price.index)
    d[ret > BUY_THRESHOLD] = "Buy"
    d[ret < SELL_THRESHOLD] = "Sell"
    return d


def build_feature_df(ticker: str) -> pd.DataFrame:
    """
    Return a DataFrame with columns [realized_vol, rsi_14, momentum_decision]
    for one ticker. Rows with NaN in vol or RSI are dropped.
    """
    price = _load_price(ticker)
    vol = _realized_vol(price).rename("realized_vol")
    rsi = _rsi(price).rename("rsi_14")
    mom = _momentum_decisions(price).rename("momentum_decision")
    df = pd.concat([vol, rsi, mom], axis=1).dropna(subset=["realized_vol", "rsi_14"])
    return df


# ---------------------------------------------------------------------------
# CUSUM herding proxy: fraction of panel tickers below 50-day MA
# ---------------------------------------------------------------------------


def compute_frac_below_ma(tickers: list[str], ma_window: int = 50) -> pd.Series:
    """
    For each calendar day, compute the fraction of panel tickers where
    the adjusted close is below its ma_window-day moving average.

    Interpretation: high fraction → most tickers in downtrend → agents
    receiving bearish signals → collective Sell herding.
    """
    below_series: list[pd.Series] = []
    for ticker in tickers:
        try:
            price = _load_price(ticker)
        except FileNotFoundError:
            continue
        ma = price.rolling(window=ma_window, min_periods=ma_window).mean()
        below = (price < ma).astype(float)
        below.name = ticker
        below_series.append(below)

    panel = pd.concat(below_series, axis=1).dropna()
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Utility: slice by date range
# ---------------------------------------------------------------------------


def _slice(s: pd.Series | pd.DataFrame, start: date, end: date):
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    return s.loc[(s.index >= ts) & (s.index <= te)]


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------


def run_calibration() -> dict:
    monitor = DriftMonitor()

    # -----------------------------------------------------------------------
    # 1. Build features for primary tickers
    # -----------------------------------------------------------------------
    print("Computing features for primary tickers...")
    per_ticker: dict[str, pd.DataFrame] = {}
    numeric_frames: list[pd.DataFrame] = []
    for ticker in PRIMARY_TICKERS:
        feat = build_feature_df(ticker)
        per_ticker[ticker] = feat
        numeric_frames.append(feat[["realized_vol", "rsi_14"]])

    combined = pd.concat(numeric_frames, axis=0)
    baseline_df = _slice(combined, KS_CHISQ_BASELINE_START, KS_CHISQ_BASELINE_END)
    recent_df = _slice(combined, RECENT_START, RECENT_END)
    print(f"  KS/chi-sq baseline obs: {len(baseline_df)}")
    print(f"  Recent obs:             {len(recent_df)}")

    # -----------------------------------------------------------------------
    # 2. KS data drift check (realized_vol + rsi_14)
    # -----------------------------------------------------------------------
    print("\nRunning KS data drift check...")
    ks_result = monitor.check_data_drift(recent_df, baseline_df)
    print(f"  Alert: {ks_result.alert}")
    print(f"  {ks_result.description}")

    # -----------------------------------------------------------------------
    # 3. Chi-sq agent drift check (momentum proxy decisions)
    # -----------------------------------------------------------------------
    print("\nRunning chi-sq agent drift check...")
    baseline_decisions: list[str] = []
    recent_decisions: list[str] = []
    for _ticker, feat in per_ticker.items():
        baseline_decisions.extend(
            _slice(feat, KS_CHISQ_BASELINE_START, KS_CHISQ_BASELINE_END)[
                "momentum_decision"
            ].dropna().tolist()
        )
        recent_decisions.extend(
            _slice(feat, RECENT_START, RECENT_END)["momentum_decision"].dropna().tolist()
        )

    n_base = len(baseline_decisions)
    baseline_dist = {
        "Buy": baseline_decisions.count("Buy") / n_base,
        "Hold": baseline_decisions.count("Hold") / n_base,
        "Sell": baseline_decisions.count("Sell") / n_base,
    }
    n_recent = len(recent_decisions)
    recent_dist = {
        "Buy": recent_decisions.count("Buy") / n_recent,
        "Hold": recent_decisions.count("Hold") / n_recent,
        "Sell": recent_decisions.count("Sell") / n_recent,
    }
    print(f"  Baseline distribution (2020-2021): {baseline_dist}")
    print(f"  Recent  distribution (2022-2023):  {recent_dist}")

    chi_result = monitor.check_agent_drift(recent_decisions, baseline_dist)
    print(f"  Alert: {chi_result.alert}")
    print(f"  {chi_result.description}")

    # -----------------------------------------------------------------------
    # 4. CUSUM collective drift (fraction below 50d MA)
    # -----------------------------------------------------------------------
    print("\nComputing herding proxy (fraction of tickers below 50d MA)...")
    frac_below = compute_frac_below_ma(ALL_TICKERS, ma_window=50)

    cusum_baseline_series = _slice(frac_below, CUSUM_BASELINE_START, CUSUM_BASELINE_END)
    cusum_recent_series = _slice(frac_below, RECENT_START, RECENT_END)

    baseline_mean = float(cusum_baseline_series.mean())
    baseline_std = float(cusum_baseline_series.std())
    recent_mean = float(cusum_recent_series.mean())

    print(f"  2021 baseline: mean={baseline_mean:.4f}, std={baseline_std:.4f}")
    print(f"  2022 recent:   mean={recent_mean:.4f}")
    slack = baseline_mean + 0.5 * baseline_std
    print(f"  CUSUM slack={slack:.4f}, threshold={3 * baseline_std:.4f}")

    cusum_result = monitor.check_collective_drift(
        herding_series=cusum_recent_series.tolist(),
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
    )
    print(f"  Alert: {cusum_result.alert}")
    print(f"  {cusum_result.description}")

    # -----------------------------------------------------------------------
    # 5. Regime detection summary
    # -----------------------------------------------------------------------
    print("\nRegime detection summary:")
    detections = {
        "data_drift_ks": ks_result.alert,
        "agent_drift_chisq": chi_result.alert,
        "collective_drift_cusum": cusum_result.alert,
    }
    for check, detected in detections.items():
        print(f"  {check}: {'DETECTED' if detected else 'NOT DETECTED'}")

    all_detected = all(detections.values())
    any_detected = any(detections.values())

    if all_detected:
        print("\nREGIME CHANGE CONFIRMED: All three monitors alert on 2022 rate shock.")
    elif any_detected:
        print("\nPARTIAL DETECTION: Check individual results.")
    else:
        print("\nWARNING: No monitor detected the 2022 regime change.")

    # -----------------------------------------------------------------------
    # 6. Per-ticker feature statistics
    # -----------------------------------------------------------------------
    feature_stats: dict[str, dict] = {}
    for ticker, feat in per_ticker.items():
        b = _slice(feat, KS_CHISQ_BASELINE_START, KS_CHISQ_BASELINE_END)
        r = _slice(feat, RECENT_START, RECENT_END)
        feature_stats[ticker] = {
            "baseline_vol_mean": float(b["realized_vol"].mean()),
            "recent_vol_mean": float(r["realized_vol"].mean()),
            "vol_ratio": float(r["realized_vol"].mean() / b["realized_vol"].mean()),
            "baseline_rsi_mean": float(b["rsi_14"].mean()),
            "recent_rsi_mean": float(r["rsi_14"].mean()),
        }

    # -----------------------------------------------------------------------
    # 7. Output payload
    # -----------------------------------------------------------------------
    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "phase": 13,
            "ticket": "E5-T5",
            "description": "Drift monitor calibration on 2022 rate-shock regime change",
            "ks_chisq_baseline_period": (
                f"{KS_CHISQ_BASELINE_START} to {KS_CHISQ_BASELINE_END}"
            ),
            "cusum_baseline_period": (
                f"{CUSUM_BASELINE_START} to {CUSUM_BASELINE_END} "
                "(2021 only; COVID-2020 excluded to avoid extreme vol inflating mean)"
            ),
            "recent_period": f"{RECENT_START} to {RECENT_END}",
            "primary_tickers": PRIMARY_TICKERS,
            "cusum_panel_tickers": ALL_TICKERS,
            "ks_chisq_baseline_n": len(baseline_df),
            "recent_n": len(recent_df),
        },
        "data_drift_ks": {
            "alert": ks_result.alert,
            "min_p_value": ks_result.statistic,
            "threshold": ks_result.threshold,
            "feature_alerts": ks_result.feature_alerts,
            "description": ks_result.description,
        },
        "agent_drift_chisq": {
            "alert": chi_result.alert,
            "chi_squared": chi_result.statistic,
            "p_value": chi_result.p_value,
            "threshold": chi_result.threshold,
            "baseline_distribution": baseline_dist,
            "recent_distribution": recent_dist,
            "description": chi_result.description,
        },
        "collective_drift_cusum": {
            "alert": cusum_result.alert,
            "cusum_ck": cusum_result.statistic,
            "threshold_3sigma": cusum_result.threshold,
            "cusum_baseline_herding_mean": baseline_mean,
            "cusum_baseline_herding_std": baseline_std,
            "recent_herding_mean": recent_mean,
            "herding_proxy": (
                "fraction of panel tickers below their 50-day moving average"
            ),
            "description": cusum_result.description,
        },
        "per_ticker_feature_stats": feature_stats,
        "regime_detected": {
            "all_three": all_detected,
            "any": any_detected,
            "by_monitor": detections,
        },
    }

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Phase 13 E5-T5: Drift Monitor Calibration")
    print(f"KS/chi-sq baseline : {KS_CHISQ_BASELINE_START} → {KS_CHISQ_BASELINE_END}")
    print(f"CUSUM baseline     : {CUSUM_BASELINE_START} → {CUSUM_BASELINE_END}")
    print(f"Recent (all three) : {RECENT_START} → {RECENT_END}")
    print("=" * 60)

    output = run_calibration()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to: {OUTPUT_PATH}")

    if not output["regime_detected"]["any"]:
        print("\nERROR: Calibration FAILED — no monitor detected the regime change.")
        sys.exit(1)
    print("Calibration PASSED.")


if __name__ == "__main__":
    main()
