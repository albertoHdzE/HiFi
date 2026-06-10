"""Shared test fixtures for HiFi.

All fixtures use deterministic seeds for reproducibility.
No mocks are used -- synthetic data is generated with controlled randomness.
"""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from hifi.config.loader import HiFiConfig, load_config

# --- Raw fixture loader ---


def read_raw_ohlcv_fixture(path: Path, ticker: str):
    """
    Load an OHLCVDataset from a raw yfinance-format parquet fixture.

    The Phase 1 test fixtures are plain pandas DataFrames written directly
    from yfinance (columns: Date, Open, High, Low, Close, Adj Close, Volume).
    They do NOT have the HiFi dataset metadata block required by read_ohlcv().
    This helper constructs an OHLCVDataset directly from the raw columns.
    """
    import math

    import pyarrow.parquet as pq

    from hifi.data.schemas import OHLCVBar, OHLCVDataset, ProvenanceRecord

    table = pq.read_table(path)
    df = table.to_pandas()

    prov = ProvenanceRecord(
        source="fixture",
        fetched_at=datetime(2023, 4, 1, 0, 0, 0),
    )

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
            bars.append(OHLCVBar(
                ticker=ticker, date=dt,
                open=o, high=h, low=lv, close=c,
                volume=vol, adjusted_close=adj, source="fixture",
            ))
        except Exception:
            continue

    dates = [b.date for b in bars]
    return OHLCVDataset(
        ticker=ticker, bars=bars, source="fixture",
        fetched_at=datetime(2023, 4, 1, 0, 0, 0),
        date_from=min(dates) if dates else date(2023, 1, 3),
        date_to=max(dates) if dates else date(2023, 4, 1),
        provenance=prov,
    )

# --- Paths ---


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    """Return the configs directory."""
    return project_root / "configs"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


# --- Configuration ---


@pytest.fixture
def default_config() -> HiFiConfig:
    """Load the default configuration."""
    return load_config()


# --- Reproducibility ---


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a deterministic random number generator.

    Every test that needs randomness should use this fixture
    instead of calling np.random directly. This ensures
    reproducibility across test runs.
    """
    return np.random.default_rng(seed=42)


# --- Synthetic Financial Data ---


@pytest.fixture
def synthetic_ohlcv(rng: np.random.Generator) -> dict:
    """Generate deterministic synthetic OHLCV data for testing.

    Produces 252 trading days (one year) of realistic-looking
    price data using geometric Brownian motion with controlled seed.

    Returns:
        Dictionary with keys: dates, open, high, low, close, volume
    """
    n_days = 252
    initial_price = 100.0
    daily_return_mean = 0.0005
    daily_return_std = 0.02

    # Generate log returns
    log_returns = rng.normal(daily_return_mean, daily_return_std, n_days)

    # Build close prices from cumulative returns
    close = initial_price * np.exp(np.cumsum(log_returns))

    # Derive OHLV from close with deterministic noise
    daily_range = rng.uniform(0.005, 0.03, n_days)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_prices = low + rng.uniform(0, 1, n_days) * (high - low)
    volume = rng.integers(100_000, 10_000_000, n_days)

    return {
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "n_days": n_days,
        "initial_price": initial_price,
    }


@pytest.fixture
def synthetic_financials(rng: np.random.Generator) -> dict:
    """Generate deterministic synthetic financial statement data.

    Produces 4 quarters of financial data with realistic ratios.

    Returns:
        Dictionary with quarterly financial metrics.
    """
    revenue_base = 50_000_000_000  # 50B base revenue
    quarters = 4

    revenue = revenue_base * (1 + rng.uniform(0.02, 0.15, quarters))
    net_income = revenue * rng.uniform(0.15, 0.30, quarters)
    total_assets = revenue * rng.uniform(2.5, 4.0, quarters)
    total_equity = total_assets * rng.uniform(0.3, 0.6, quarters)
    total_debt = total_assets - total_equity
    current_assets = total_assets * rng.uniform(0.2, 0.4, quarters)
    current_liabilities = current_assets * rng.uniform(0.5, 1.2, quarters)
    shares_outstanding = rng.integers(1_000_000_000, 5_000_000_000, quarters)
    stock_price = rng.uniform(100, 300, quarters)

    return {
        "revenue": revenue,
        "net_income": net_income,
        "total_assets": total_assets,
        "total_equity": total_equity,
        "total_debt": total_debt,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "shares_outstanding": shares_outstanding,
        "stock_price": stock_price,
        "quarters": quarters,
    }
