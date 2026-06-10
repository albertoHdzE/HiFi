"""
Dataset versioning and provenance registry for HiFi.

Provides:
- content_hash(path): compute the SHA-256 hash of any file, used to
  fingerprint Parquet datasets and detect modifications.
- DatasetRegistry: a lightweight JSON-backed catalog of every dataset
  written to disk. Each entry records the ticker/series_id, source,
  date range, download timestamp, file path, and content hash.

Design decisions:
- Content-based hashing (SHA-256) rather than timestamps: a timestamp
  tells you when a file was written, not what is in it. A hash tells you
  both. If we re-download data and the provider has revised old values
  (common with FRED), the new hash differs and the registry records this.
- JSON registry: simple, human-readable, diff-able in git. An SQLite
  database would be more robust for large catalogs but adds a new
  dependency. JSON is sufficient for Phase 1 (10 tickers + 7 macro series).
  The registry is migrated to SQLite in a later phase if it becomes a
  bottleneck.
- Registry path: defaults to data/registry.json (relative to project root).
  Tests always pass an explicit path to avoid touching the project registry.
- Integrity verification: the registry stores the hash at write time. At
  any later point, verify_integrity(entry) re-hashes the file and compares.
  A difference means the file was modified after registration (e.g., by a
  failed write that left a partial file, or an accidental manual edit).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_BLOCK_SIZE = 65536  # 64 KiB read blocks for streaming hash
_DEFAULT_REGISTRY_PATH = Path("data/registry.json")


def content_hash(path: Path) -> str:
    """
    Compute the SHA-256 hash of a file by reading it in 64 KiB blocks.

    Streaming avoids loading large Parquet files fully into memory.
    The returned hex digest is stable for the same file content regardless
    of when or where it was written.

    Parameters
    ----------
    path : Path
        Path to the file to hash. Must exist and be readable.

    Returns
    -------
    str
        Lowercase 64-character hex digest.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_BLOCK_SIZE):
            h.update(block)
    return h.hexdigest()


@dataclass
class RegistryEntry:
    """
    A single record in the DatasetRegistry.

    Fields:
    - dataset_id: unique identifier of the form "{ticker}_{source}" or
      "{series_id}_{source}" (e.g. "AAPL_yfinance", "FEDFUNDS_FRED").
    - source: data provider name.
    - date_from: start of the data window (ISO 8601 string).
    - date_to: end of the data window (ISO 8601 string).
    - file_path: absolute or relative path to the Parquet file.
    - content_hash: SHA-256 hex digest at registration time.
    - registered_at: ISO 8601 UTC timestamp of when this entry was created.
    """

    dataset_id: str
    source: str
    date_from: str
    date_to: str
    file_path: str
    content_hash: str
    registered_at: str


class DatasetRegistry:
    """
    JSON-backed catalog of HiFi datasets.

    Each registry instance corresponds to one JSON file. The registry is
    loaded from disk on construction (if the file exists) and written back
    after every mutation. This is safe for Phase 1 (sequential single-process
    usage). Concurrent writes are NOT supported.

    Parameters
    ----------
    path : Path
        Path to the JSON registry file. Created on first write if it
        does not exist.
    """

    def __init__(self, path: Path = _DEFAULT_REGISTRY_PATH) -> None:
        self._path = Path(path)
        self._entries: dict[str, RegistryEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register(
        self,
        dataset_id: str,
        source: str,
        date_from: str,
        date_to: str,
        file_path: Path,
    ) -> RegistryEntry:
        """
        Hash the file at file_path, create a RegistryEntry, and persist it.

        If a prior entry exists for the same dataset_id, it is overwritten.
        This happens when data is re-downloaded (revised values, extended range).

        Parameters
        ----------
        dataset_id : str
            Unique key (e.g. "AAPL_yfinance"). Convention: "{id}_{source}".
        source : str
            Data provider name.
        date_from : str
            ISO 8601 date string for the start of the data window.
        date_to : str
            ISO 8601 date string for the end of the data window.
        file_path : Path
            Path to the Parquet file to register.

        Returns
        -------
        RegistryEntry
            The newly created entry.
        """
        file_path = Path(file_path)
        file_hash = content_hash(file_path)

        entry = RegistryEntry(
            dataset_id=dataset_id,
            source=source,
            date_from=date_from,
            date_to=date_to,
            file_path=str(file_path),
            content_hash=file_hash,
            registered_at=datetime.now(UTC).isoformat(),
        )
        self._entries[dataset_id] = entry
        self._save()
        return entry

    def lookup(self, dataset_id: str) -> RegistryEntry | None:
        """
        Return the RegistryEntry for dataset_id, or None if not found.

        Parameters
        ----------
        dataset_id : str
            Key used during registration.
        """
        return self._entries.get(dataset_id)

    def verify_integrity(self, entry: RegistryEntry) -> bool:
        """
        Re-hash the file and compare against the stored hash.

        Returns True if the file content matches the registered hash,
        False if the file has been modified or cannot be read.

        Parameters
        ----------
        entry : RegistryEntry
            Entry to verify (obtained from lookup or register).
        """
        try:
            current = content_hash(Path(entry.file_path))
            return current == entry.content_hash
        except (FileNotFoundError, OSError):
            return False

    def all_entries(self) -> list[RegistryEntry]:
        """Return all registered entries sorted by dataset_id."""
        return sorted(self._entries.values(), key=lambda e: e.dataset_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            self._entries = {}
            return
        with open(self._path) as f:
            raw: list[dict[str, Any]] = json.load(f)
        self._entries = {
            item["dataset_id"]: RegistryEntry(**item) for item in raw
        }

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(
                [asdict(e) for e in self.all_entries()],
                f,
                indent=2,
            )
