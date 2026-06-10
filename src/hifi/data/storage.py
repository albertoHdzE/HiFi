"""
Parquet read/write utilities for HiFi datasets.

Every dataset written by HiFi is stored as a single Parquet file. The dataset
metadata (ticker, source, dates, provenance) is embedded in the Parquet schema
metadata under the key "hifi_dataset_metadata" as a UTF-8 JSON string. This
keeps each dataset self-describing: everything needed to reconstruct the full
typed object is in one file.

Design decisions:
- One Parquet file per dataset (no sidecar files to lose track of).
- Schema metadata for dataset-level fields; columns for observation-level data.
- All datetimes stored in UTC ISO 8601 format within the JSON metadata.
- Dates (date-only) stored as ISO 8601 strings in metadata and as date32 in
  the Parquet column schema.
- Parquet stores float64 for all price/value columns; int64 volume is cast to
  float64 on write for schema uniformity (fractional shares make float correct).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hifi.data.schemas import (
    MacroDataset,
    MacroIndicator,
    OHLCVBar,
    OHLCVDataset,
    ProvenanceRecord,
)

_METADATA_KEY = b"hifi_dataset_metadata"


# ---------------------------------------------------------------------------
# Internal serialisation helpers
# ---------------------------------------------------------------------------


def _provenance_to_dict(prov: ProvenanceRecord) -> dict:
    return {
        "source": prov.source,
        "fetched_at": prov.fetched_at.isoformat(),
        "parameters": prov.parameters,
        "content_hash": prov.content_hash,
    }


def _provenance_from_dict(d: dict) -> ProvenanceRecord:
    return ProvenanceRecord(
        source=d["source"],
        fetched_at=datetime.fromisoformat(d["fetched_at"]),
        parameters=d.get("parameters", {}),
        content_hash=d.get("content_hash"),
    )


# ---------------------------------------------------------------------------
# OHLCVDataset
# ---------------------------------------------------------------------------


def write_ohlcv(dataset: OHLCVDataset, path: Path) -> Path:
    """
    Write an OHLCVDataset to a Parquet file.

    The columns written are: date, open, high, low, close, volume, adjusted_close.
    Dataset-level metadata (ticker, source, dates, provenance) is embedded in the
    Parquet schema metadata.

    Returns the written path (same as input) so callers can chain operations.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": float(bar.volume),
            "adjusted_close": bar.adjusted_close,
        }
        for bar in dataset.bars
    ]

    schema = pa.schema(
        [
            pa.field("date", pa.date32()),
            pa.field("open", pa.float64()),
            pa.field("high", pa.float64()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
            pa.field("volume", pa.float64()),
            pa.field("adjusted_close", pa.float64()),
        ]
    )

    if rows:
        arrays = {col: [r[col] for r in rows] for col in schema.names}
        table = pa.table(arrays, schema=schema)
    else:
        table = schema.empty_table()

    metadata = {
        "ticker": dataset.ticker,
        "source": dataset.source,
        "fetched_at": dataset.fetched_at.isoformat(),
        "date_from": dataset.date_from.isoformat(),
        "date_to": dataset.date_to.isoformat(),
        "provenance": _provenance_to_dict(dataset.provenance),
    }
    merged = {**(table.schema.metadata or {}), _METADATA_KEY: json.dumps(metadata).encode()}
    table = table.replace_schema_metadata(merged)

    pq.write_table(table, path)
    return path


def read_ohlcv(path: Path, ticker_override: str | None = None) -> OHLCVDataset:
    """
    Read an OHLCVDataset from a Parquet file written by write_ohlcv.

    The ticker must be present in the Parquet schema metadata. If ticker_override
    is provided it replaces the stored ticker (useful for testing with generic fixtures).
    """
    path = Path(path)
    table = pq.read_table(path)
    raw_meta = table.schema.metadata or {}

    if _METADATA_KEY not in raw_meta:
        raise ValueError(f"No HiFi dataset metadata found in {path}")

    meta = json.loads(raw_meta[_METADATA_KEY])
    ticker = ticker_override or meta["ticker"]
    provenance = _provenance_from_dict(meta["provenance"])
    source = meta["source"]
    fetched_at = datetime.fromisoformat(meta["fetched_at"])
    date_from = date.fromisoformat(meta["date_from"])
    date_to = date.fromisoformat(meta["date_to"])

    df = table.to_pandas()
    bars: list[OHLCVBar] = []
    for _, row in df.iterrows():
        bars.append(
            OHLCVBar(
                ticker=ticker,
                date=row["date"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                adjusted_close=_read_optional_float(row["adjusted_close"]),
                source=source,
            )
        )

    return OHLCVDataset(
        ticker=ticker,
        bars=bars,
        source=source,
        fetched_at=fetched_at,
        date_from=date_from,
        date_to=date_to,
        provenance=provenance,
    )


def _read_optional_float(v: object) -> float | None:
    """Return float(v) or None when v is NaN/None/unconvertible."""
    import math

    try:
        f = float(v)  # type: ignore[arg-type]
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# MacroDataset
# ---------------------------------------------------------------------------


def write_macro(dataset: MacroDataset, path: Path) -> Path:
    """
    Write a MacroDataset to a Parquet file.

    Columns written: date, value. Dataset-level metadata (series_id, name,
    frequency, unit, source, dates, provenance) is embedded in the Parquet schema
    metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {"date": obs.date, "value": obs.value}
        for obs in dataset.observations
    ]

    schema = pa.schema(
        [
            pa.field("date", pa.date32()),
            pa.field("value", pa.float64()),
        ]
    )

    if rows:
        arrays = {col: [r[col] for r in rows] for col in schema.names}
        table = pa.table(arrays, schema=schema)
    else:
        table = schema.empty_table()

    metadata = {
        "series_id": dataset.series_id,
        "name": dataset.name,
        "frequency": dataset.frequency,
        "unit": dataset.unit,
        "source": dataset.source,
        "fetched_at": dataset.fetched_at.isoformat(),
        "date_from": dataset.date_from.isoformat(),
        "date_to": dataset.date_to.isoformat(),
        "provenance": _provenance_to_dict(dataset.provenance),
    }
    merged = {**(table.schema.metadata or {}), _METADATA_KEY: json.dumps(metadata).encode()}
    table = table.replace_schema_metadata(merged)

    pq.write_table(table, path)
    return path


def read_macro(path: Path) -> MacroDataset:
    """
    Read a MacroDataset from a Parquet file written by write_macro.
    """
    path = Path(path)
    table = pq.read_table(path)
    raw_meta = table.schema.metadata or {}

    if _METADATA_KEY not in raw_meta:
        raise ValueError(f"No HiFi dataset metadata found in {path}")

    meta = json.loads(raw_meta[_METADATA_KEY])
    series_id = meta["series_id"]
    provenance = _provenance_from_dict(meta["provenance"])

    df = table.to_pandas()
    observations = [
        MacroIndicator(
            series_id=series_id,
            date=row["date"],
            value=float(row["value"]),
        )
        for _, row in df.iterrows()
    ]

    return MacroDataset(
        series_id=series_id,
        name=meta["name"],
        frequency=meta["frequency"],
        unit=meta["unit"],
        observations=observations,
        source=meta["source"],
        fetched_at=datetime.fromisoformat(meta["fetched_at"]),
        date_from=date.fromisoformat(meta["date_from"]),
        date_to=date.fromisoformat(meta["date_to"]),
        provenance=provenance,
    )
