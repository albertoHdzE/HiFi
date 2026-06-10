"""
Canonical data schemas for the HiFi data acquisition layer.

These models define the contract between data ingestion (Phase 1) and all
downstream layers: financial engines, agents, evaluation. Every data point
entering the system is validated against one of these schemas at ingestion
time. No raw DataFrames or dicts cross module boundaries.

Design decisions:
- Pydantic v2 for runtime validation. Errors surface immediately at the data
  boundary, not silently inside downstream computation.
- OHLCV bars enforce H >= max(O,C) and L <= min(O,C). Violations indicate
  unadjusted or corrupted data and are rejected before reaching any engine.
- ProvenanceRecord is attached to every dataset, carrying source, timestamp,
  request parameters, and content hash. This satisfies David sections 4.3
  (verifiability) and 4.5 (reproducibility).
- content_hash in ProvenanceRecord is optional at construction because the
  Parquet file does not yet exist. The versioning layer populates it after
  writing to disk (P1-E5).
- Financial statement fields (revenue, net_income, etc.) have no positivity
  constraint: losses, negative equity, and negative rates are all valid data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ProvenanceRecord(BaseModel):
    """
    Metadata attached to every dataset describing its origin.

    Every dataset produced by HiFi carries one ProvenanceRecord so that any
    result can be traced back to its source, the exact request that produced
    it, and the file that stores it.

    Attributes:
        source: Data provider name (e.g. "yfinance", "FRED").
        fetched_at: UTC datetime when the download was initiated.
        parameters: Request parameters used to fetch the data (ticker,
            series_id, start/end dates, frequency, etc.).
        content_hash: SHA-256 hash of the Parquet file. None until written
            to disk by the versioning layer.
    """

    source: str
    fetched_at: datetime
    parameters: dict[str, Any] = Field(default_factory=dict)
    content_hash: str | None = None

    def compute_signature(self) -> str:
        """
        Compute a deterministic hex digest over source and parameters.

        This signature identifies a logical download request independently of
        when it was made. Two records with identical source and parameters
        produce the same signature, enabling deduplication and cache lookups.

        This is NOT a file integrity check; that role belongs to content_hash.
        """
        payload = json.dumps(
            {"source": self.source, "parameters": self.parameters},
            sort_keys=True,
            default=str,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class OHLCVBar(BaseModel):
    """
    A single OHLCV price bar for one ticker on one trading day.

    Invariants enforced at construction:
    - open, high, low, close are strictly positive (negative or zero price is
      a data error, not a valid market condition for equities)
    - volume is non-negative (zero volume on a trading day is unusual but valid,
      e.g., ETF creation/redemption days)
    - high >= max(open, close)
    - low <= min(open, close)
    - adjusted_close, when present, is strictly positive
    """

    ticker: str
    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    adjusted_close: float | None = Field(default=None)
    source: str = "unknown"

    @field_validator("adjusted_close", mode="before")
    @classmethod
    def _adjusted_close_positive(cls, v: Any) -> Any:
        if v is not None and v <= 0:
            raise ValueError(f"adjusted_close must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _ohlcv_relationships(self) -> OHLCVBar:
        errors: list[str] = []
        if self.high < self.low:
            errors.append(f"high ({self.high}) < low ({self.low})")
        if self.high < self.open:
            errors.append(f"high ({self.high}) < open ({self.open})")
        if self.high < self.close:
            errors.append(f"high ({self.high}) < close ({self.close})")
        if self.low > self.open:
            errors.append(f"low ({self.low}) > open ({self.open})")
        if self.low > self.close:
            errors.append(f"low ({self.low}) > close ({self.close})")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class OHLCVDataset(BaseModel):
    """
    A collection of OHLCV bars for a single ticker with full provenance.

    The bars list may be empty (no trading data in the requested window is
    a valid outcome, not an error). All bars that are present must belong to
    the same ticker as the dataset.
    """

    ticker: str
    bars: list[OHLCVBar]
    source: str
    fetched_at: datetime
    date_from: date
    date_to: date
    provenance: ProvenanceRecord

    @field_validator("bars")
    @classmethod
    def _bars_match_ticker(
        cls, bars: list[OHLCVBar], info: Any
    ) -> list[OHLCVBar]:
        ticker = info.data.get("ticker")
        if ticker is not None:
            for bar in bars:
                if bar.ticker != ticker:
                    raise ValueError(
                        f"Bar ticker '{bar.ticker}' does not match "
                        f"dataset ticker '{ticker}'"
                    )
        return bars


class FundamentalsSnapshot(BaseModel):
    """
    A point-in-time view of a company's financial statements for one reporting period.

    Fields are None when the data source does not provide them or when the
    company does not report a metric. None is not zero: callers must handle
    missing data explicitly rather than assuming zero.

    All monetary values are in the currency reported by the data source
    (USD for US equities via yfinance). No currency normalisation is applied
    in Phase 1.
    """

    ticker: str
    period_end: date
    revenue: float | None = None
    net_income: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    total_equity: float | None = None
    eps: float | None = None
    pe_ratio: float | None = None
    market_cap: float | None = None
    source: str
    fetched_at: datetime
    provenance: ProvenanceRecord


class MacroIndicator(BaseModel):
    """
    A single observation from a macroeconomic time series.

    Values can be negative: negative GDP growth, sub-zero real interest rates,
    and negative yield spreads are all valid economic states.
    """

    series_id: str
    date: date
    value: float


class MacroDataset(BaseModel):
    """
    A named macroeconomic time series with full provenance.

    The frequency field reflects the native FRED publication frequency
    (e.g., "monthly" for CPI, "quarterly" for GDP). Forward-filling to daily
    frequency for alignment with OHLCV data is performed by the fetcher,
    not stored here. This keeps the raw data pure and the transformation
    documented and testable separately.
    """

    series_id: str
    name: str
    frequency: str
    unit: str
    observations: list[MacroIndicator]
    source: str = "FRED"
    fetched_at: datetime
    date_from: date
    date_to: date
    provenance: ProvenanceRecord

    @field_validator("observations")
    @classmethod
    def _observations_match_series(
        cls, observations: list[MacroIndicator], info: Any
    ) -> list[MacroIndicator]:
        series_id = info.data.get("series_id")
        if series_id is not None:
            for obs in observations:
                if obs.series_id != series_id:
                    raise ValueError(
                        f"Observation series_id '{obs.series_id}' does not match "
                        f"dataset series_id '{series_id}'"
                    )
        return observations
