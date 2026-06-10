"""
Unit tests for content_hash and DatasetRegistry (P1-E5).

No file system interactions with the real project data directory.
All registry tests use tmp_path to avoid state leakage between test runs.

Tickets covered:
- P1-E5-T6: Hash is stable for identical content
- P1-E5-T7: Hash changes when content changes
- P1-E5-T8: Registry stores and retrieves entries correctly
- P1-E5-T9: Integrity check detects file tampering
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hifi.data.versioning import DatasetRegistry, content_hash

# ---------------------------------------------------------------------------
# P1-E5-T6: Hash stability for identical content
# ---------------------------------------------------------------------------


class TestContentHash:
    """T6, T7: SHA-256 is stable for same content and changes for different content."""

    def test_hash_stable_same_file(self, tmp_path: Path) -> None:
        """T6: calling content_hash twice on the same file returns the same digest."""
        f = tmp_path / "dataset.parquet"
        f.write_bytes(b"simulated parquet content")
        assert content_hash(f) == content_hash(f)

    def test_hash_stable_across_instances(self, tmp_path: Path) -> None:
        """T6: two files with identical content have the same hash."""
        f1 = tmp_path / "a.parquet"
        f2 = tmp_path / "b.parquet"
        content = b"identical content bytes 12345"
        f1.write_bytes(content)
        f2.write_bytes(content)
        assert content_hash(f1) == content_hash(f2)

    def test_hash_is_hex_string(self, tmp_path: Path) -> None:
        """T6: returned hash is a 64-character lowercase hex string (SHA-256)."""
        f = tmp_path / "test.parquet"
        f.write_bytes(b"test data")
        h = content_hash(f)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_changes_when_content_changes(self, tmp_path: Path) -> None:
        """T7: modifying one byte changes the hash."""
        f = tmp_path / "data.parquet"
        f.write_bytes(b"version one content")
        h1 = content_hash(f)
        f.write_bytes(b"version two content")
        h2 = content_hash(f)
        assert h1 != h2

    def test_hash_empty_file(self, tmp_path: Path) -> None:
        """T6: empty file has a stable, known SHA-256 hash."""
        f = tmp_path / "empty.parquet"
        f.write_bytes(b"")
        # SHA-256 of empty string is always this value
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert content_hash(f) == expected

    def test_hash_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """T7: hashing a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            content_hash(tmp_path / "no_such_file.parquet")

    def test_hash_large_file(self, tmp_path: Path) -> None:
        """T6: hash works correctly on files larger than one 64 KiB block."""
        f = tmp_path / "large.parquet"
        # Write ~200 KiB -- larger than the 64 KiB streaming block
        f.write_bytes(b"x" * 200_000)
        h = content_hash(f)
        assert len(h) == 64


# ---------------------------------------------------------------------------
# P1-E5-T8: Registry stores and retrieves correctly
# ---------------------------------------------------------------------------


class TestDatasetRegistry:
    """T8, T9: DatasetRegistry CRUD and integrity verification."""

    def _make_parquet(self, tmp_path: Path, name: str = "aapl.parquet") -> Path:
        """Create a minimal Parquet-like file for testing."""
        p = tmp_path / name
        p.write_bytes(b"fake parquet data for " + name.encode())
        return p

    def test_register_and_lookup(self, tmp_path: Path) -> None:
        """T8: a registered entry can be looked up by dataset_id."""
        reg = DatasetRegistry(tmp_path / "registry.json")
        pq = self._make_parquet(tmp_path)

        reg.register(
            dataset_id="AAPL_yfinance",
            source="yfinance",
            date_from="2023-01-03",
            date_to="2023-04-01",
            file_path=pq,
        )

        retrieved = reg.lookup("AAPL_yfinance")
        assert retrieved is not None
        assert retrieved.dataset_id == "AAPL_yfinance"
        assert retrieved.source == "yfinance"

    def test_lookup_missing_returns_none(self, tmp_path: Path) -> None:
        """T8: looking up an unregistered dataset_id returns None."""
        reg = DatasetRegistry(tmp_path / "registry.json")
        assert reg.lookup("NONEXISTENT_yfinance") is None

    def test_register_stores_content_hash(self, tmp_path: Path) -> None:
        """T8: the registered entry contains the correct content hash."""
        reg = DatasetRegistry(tmp_path / "registry.json")
        pq = self._make_parquet(tmp_path)

        entry = reg.register(
            dataset_id="AAPL_yfinance",
            source="yfinance",
            date_from="2023-01-03",
            date_to="2023-04-01",
            file_path=pq,
        )

        expected_hash = content_hash(pq)
        assert entry.content_hash == expected_hash

    def test_register_stores_all_fields(self, tmp_path: Path) -> None:
        """T8: all metadata fields are persisted in the entry."""
        reg = DatasetRegistry(tmp_path / "registry.json")
        pq = self._make_parquet(tmp_path)

        entry = reg.register(
            dataset_id="FEDFUNDS_FRED",
            source="FRED",
            date_from="2022-01-01",
            date_to="2022-12-31",
            file_path=pq,
        )

        assert entry.source == "FRED"
        assert entry.date_from == "2022-01-01"
        assert entry.date_to == "2022-12-31"
        assert entry.file_path == str(pq)
        assert entry.registered_at  # non-empty

    def test_registry_persists_to_json(self, tmp_path: Path) -> None:
        """T8: entries survive constructing a second registry instance from the same path."""
        reg_path = tmp_path / "registry.json"
        pq = self._make_parquet(tmp_path)

        reg1 = DatasetRegistry(reg_path)
        reg1.register(
            dataset_id="AAPL_yfinance",
            source="yfinance",
            date_from="2023-01-03",
            date_to="2023-04-01",
            file_path=pq,
        )

        # Create a fresh registry from the same JSON file
        reg2 = DatasetRegistry(reg_path)
        retrieved = reg2.lookup("AAPL_yfinance")
        assert retrieved is not None
        assert retrieved.source == "yfinance"

    def test_register_overwrites_same_id(self, tmp_path: Path) -> None:
        """T8: registering the same dataset_id twice replaces the entry."""
        reg = DatasetRegistry(tmp_path / "registry.json")
        pq = self._make_parquet(tmp_path)

        reg.register(
            dataset_id="AAPL_yfinance",
            source="yfinance",
            date_from="2023-01-03",
            date_to="2023-04-01",
            file_path=pq,
        )
        # Overwrite with extended range
        reg.register(
            dataset_id="AAPL_yfinance",
            source="yfinance",
            date_from="2022-01-03",
            date_to="2023-04-01",
            file_path=pq,
        )

        entry = reg.lookup("AAPL_yfinance")
        assert entry.date_from == "2022-01-03"
        assert len(reg.all_entries()) == 1  # still one entry

    def test_multiple_entries_stored(self, tmp_path: Path) -> None:
        """T8: registry holds multiple independent entries."""
        reg = DatasetRegistry(tmp_path / "registry.json")
        pq_aapl = self._make_parquet(tmp_path, "aapl.parquet")
        pq_jpm = self._make_parquet(tmp_path, "jpm.parquet")

        reg.register("AAPL_yfinance", "yfinance", "2023-01-03", "2023-04-01", pq_aapl)
        reg.register("JPM_yfinance", "yfinance", "2023-01-03", "2023-04-01", pq_jpm)

        assert len(reg.all_entries()) == 2
        assert reg.lookup("AAPL_yfinance") is not None
        assert reg.lookup("JPM_yfinance") is not None


# ---------------------------------------------------------------------------
# P1-E5-T9: Integrity check detects tampering
# ---------------------------------------------------------------------------


class TestIntegrityVerification:
    """T9: verify_integrity returns True for unmodified files and False after tampering."""

    def test_fresh_file_passes_integrity(self, tmp_path: Path) -> None:
        """T9: a file that has not been touched since registration passes."""
        reg = DatasetRegistry(tmp_path / "registry.json")
        pq = tmp_path / "data.parquet"
        pq.write_bytes(b"clean dataset content")

        entry = reg.register("AAPL_yfinance", "yfinance", "2023-01-03", "2023-04-01", pq)
        assert reg.verify_integrity(entry)

    def test_modified_file_fails_integrity(self, tmp_path: Path) -> None:
        """T9: modifying the file after registration causes verify_integrity to return False."""
        reg = DatasetRegistry(tmp_path / "registry.json")
        pq = tmp_path / "data.parquet"
        pq.write_bytes(b"original dataset content")

        entry = reg.register("AAPL_yfinance", "yfinance", "2023-01-03", "2023-04-01", pq)
        # Tamper with the file
        pq.write_bytes(b"TAMPERED dataset content")

        assert not reg.verify_integrity(entry)

    def test_deleted_file_fails_integrity(self, tmp_path: Path) -> None:
        """T9: deleting the file after registration causes verify_integrity to return False."""
        reg = DatasetRegistry(tmp_path / "registry.json")
        pq = tmp_path / "data.parquet"
        pq.write_bytes(b"dataset content")

        entry = reg.register("AAPL_yfinance", "yfinance", "2023-01-03", "2023-04-01", pq)
        pq.unlink()  # Delete the file

        assert not reg.verify_integrity(entry)
