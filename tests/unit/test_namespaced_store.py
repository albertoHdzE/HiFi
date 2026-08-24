"""
Unit tests for NamespacedLanceDB (E6-T1, DJ-093).

All tests use tmp_path for isolated LanceDB databases. No network, no LLMs.
Tests verify: table name construction, namespace isolation, and round-trip
write/read across namespace switches.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from hifi.knowledge.namespaced_store import NamespacedLanceDB

# Minimal two-field schema for testing (no embeddings needed here)
_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("value", pa.string()),
    ]
)


def _make_data(rows: list[dict[str, str]]) -> pa.Table:
    return pa.table(
        {
            "id": [r["id"] for r in rows],
            "value": [r["value"] for r in rows],
        },
        schema=_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Table name construction
# ---------------------------------------------------------------------------


def test_prefixed_with_namespace(tmp_path: pytest.TempPathFactory) -> None:
    ns = NamespacedLanceDB(str(tmp_path / "test.lance"), namespace="hifi-dev")
    assert ns._prefixed("episodes") == "hifi-dev-episodes"
    assert ns._prefixed("chunks_a") == "hifi-dev-chunks_a"
    assert ns._prefixed("context") == "hifi-dev-context"


def test_prefixed_no_namespace(tmp_path: pytest.TempPathFactory) -> None:
    """Empty namespace: no prefix added (backward compatible)."""
    ns = NamespacedLanceDB(str(tmp_path / "test.lance"), namespace="")
    assert ns._prefixed("episodes") == "episodes"
    assert ns._prefixed("chunks_a") == "chunks_a"


def test_namespace_property(tmp_path: pytest.TempPathFactory) -> None:
    ns = NamespacedLanceDB(str(tmp_path / "test.lance"), namespace="hifi-eval")
    assert ns.namespace == "hifi-eval"


def test_default_namespace_is_empty_string(tmp_path: pytest.TempPathFactory) -> None:
    ns = NamespacedLanceDB(str(tmp_path / "test.lance"))
    assert ns.namespace == ""


# ---------------------------------------------------------------------------
# create + list_tables
# ---------------------------------------------------------------------------


def test_create_and_list_tables(tmp_path: pytest.TempPathFactory) -> None:
    db_path = str(tmp_path / "test.lance")
    ns = NamespacedLanceDB(db_path, namespace="hifi-dev")
    ns.create_table("items", data=_make_data([{"id": "1", "value": "a"}]))
    assert "items" in ns.list_tables()


def test_list_tables_no_namespace_sees_physical_names(tmp_path: pytest.TempPathFactory) -> None:
    """No-namespace instance sees the physical table name (with prefix)."""
    db_path = str(tmp_path / "test.lance")
    ns = NamespacedLanceDB(db_path, namespace="hifi-dev")
    ns.create_table("items", data=_make_data([{"id": "1", "value": "a"}]))

    ns_raw = NamespacedLanceDB(db_path, namespace="")
    assert "hifi-dev-items" in ns_raw.list_tables()


def test_list_tables_empty_initially(tmp_path: pytest.TempPathFactory) -> None:
    db_path = str(tmp_path / "test.lance")
    ns = NamespacedLanceDB(db_path, namespace="hifi-dev")
    assert ns.list_tables() == []


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------


def test_namespace_isolation(tmp_path: pytest.TempPathFactory) -> None:
    """Tables created in hifi-dev are not visible from hifi-eval."""
    db_path = str(tmp_path / "test.lance")
    dev = NamespacedLanceDB(db_path, namespace="hifi-dev")
    eval_ns = NamespacedLanceDB(db_path, namespace="hifi-eval")

    dev.create_table("items", data=_make_data([{"id": "1", "value": "dev"}]))

    assert "items" in dev.list_tables()
    assert "items" not in eval_ns.list_tables()


def test_separate_namespaces_coexist(tmp_path: pytest.TempPathFactory) -> None:
    """dev and eval tables coexist in the same database without contamination."""
    db_path = str(tmp_path / "test.lance")
    dev = NamespacedLanceDB(db_path, namespace="hifi-dev")
    eval_ns = NamespacedLanceDB(db_path, namespace="hifi-eval")

    dev.create_table("items", data=_make_data([{"id": "1", "value": "dev"}]))
    eval_ns.create_table("items", data=_make_data([{"id": "2", "value": "eval"}]))

    assert "items" in dev.list_tables()
    assert "items" in eval_ns.list_tables()

    dev_rows = dev.open_table("items").to_pandas()
    eval_rows = eval_ns.open_table("items").to_pandas()
    assert dev_rows["value"].iloc[0] == "dev"
    assert eval_rows["value"].iloc[0] == "eval"


def test_three_namespaces_independent(tmp_path: pytest.TempPathFactory) -> None:
    """dev, eval, live namespaces each see only their own tables."""
    db_path = str(tmp_path / "test.lance")
    dev = NamespacedLanceDB(db_path, namespace="hifi-dev")
    eval_ns = NamespacedLanceDB(db_path, namespace="hifi-eval")
    live = NamespacedLanceDB(db_path, namespace="hifi-live")

    dev.create_table("records", data=_make_data([{"id": "d", "value": "dev"}]))
    eval_ns.create_table("records", data=_make_data([{"id": "e", "value": "eval"}]))
    live.create_table("records", data=_make_data([{"id": "l", "value": "live"}]))

    assert dev.list_tables() == ["records"]
    assert eval_ns.list_tables() == ["records"]
    assert live.list_tables() == ["records"]

    assert dev.open_table("records").to_pandas()["value"].iloc[0] == "dev"
    assert eval_ns.open_table("records").to_pandas()["value"].iloc[0] == "eval"
    assert live.open_table("records").to_pandas()["value"].iloc[0] == "live"


# ---------------------------------------------------------------------------
# table_exists
# ---------------------------------------------------------------------------


def test_table_exists_true(tmp_path: pytest.TempPathFactory) -> None:
    db_path = str(tmp_path / "test.lance")
    ns = NamespacedLanceDB(db_path, namespace="hifi-dev")
    ns.create_table("items", data=_make_data([{"id": "1", "value": "a"}]))
    assert ns.table_exists("items") is True


def test_table_exists_false(tmp_path: pytest.TempPathFactory) -> None:
    db_path = str(tmp_path / "test.lance")
    ns = NamespacedLanceDB(db_path, namespace="hifi-dev")
    assert ns.table_exists("nonexistent") is False


def test_table_exists_cross_namespace_returns_false(tmp_path: pytest.TempPathFactory) -> None:
    """A table that exists in dev does not exist in eval."""
    db_path = str(tmp_path / "test.lance")
    dev = NamespacedLanceDB(db_path, namespace="hifi-dev")
    eval_ns = NamespacedLanceDB(db_path, namespace="hifi-eval")
    dev.create_table("items", data=_make_data([{"id": "1", "value": "a"}]))
    assert dev.table_exists("items") is True
    assert eval_ns.table_exists("items") is False


# ---------------------------------------------------------------------------
# drop_table
# ---------------------------------------------------------------------------


def test_drop_table(tmp_path: pytest.TempPathFactory) -> None:
    db_path = str(tmp_path / "test.lance")
    ns = NamespacedLanceDB(db_path, namespace="hifi-dev")
    ns.create_table("items", data=_make_data([{"id": "1", "value": "a"}]))
    assert ns.table_exists("items")
    ns.drop_table("items")
    assert not ns.table_exists("items")


def test_drop_table_does_not_affect_other_namespace(tmp_path: pytest.TempPathFactory) -> None:
    """Dropping a dev table leaves the eval namespace untouched."""
    db_path = str(tmp_path / "test.lance")
    dev = NamespacedLanceDB(db_path, namespace="hifi-dev")
    eval_ns = NamespacedLanceDB(db_path, namespace="hifi-eval")

    dev.create_table("items", data=_make_data([{"id": "1", "value": "a"}]))
    eval_ns.create_table("items", data=_make_data([{"id": "2", "value": "b"}]))

    dev.drop_table("items")

    assert not dev.table_exists("items")
    assert eval_ns.table_exists("items")  # eval unaffected


# ---------------------------------------------------------------------------
# open_or_create_table
# ---------------------------------------------------------------------------


def test_open_or_create_creates_when_missing(tmp_path: pytest.TempPathFactory) -> None:
    db_path = str(tmp_path / "test.lance")
    ns = NamespacedLanceDB(db_path, namespace="hifi-dev")
    table = ns.open_or_create_table("items", _SCHEMA)
    assert table is not None
    assert ns.table_exists("items")


def test_open_or_create_opens_existing_without_wiping_data(
    tmp_path: pytest.TempPathFactory,
) -> None:
    db_path = str(tmp_path / "test.lance")
    ns = NamespacedLanceDB(db_path, namespace="hifi-dev")
    # Create with data
    ns.create_table("items", data=_make_data([{"id": "1", "value": "existing"}]))
    # open_or_create should open existing (not wipe data)
    table = ns.open_or_create_table("items", _SCHEMA)
    df = table.to_pandas()
    assert len(df) == 1
    assert df["value"].iloc[0] == "existing"


def test_open_or_create_new_table_is_empty(tmp_path: pytest.TempPathFactory) -> None:
    db_path = str(tmp_path / "test.lance")
    ns = NamespacedLanceDB(db_path, namespace="hifi-dev")
    table = ns.open_or_create_table("fresh", _SCHEMA)
    df = table.to_pandas()
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Round-trip write/read across namespace switch
# ---------------------------------------------------------------------------


def test_round_trip_write_read_same_namespace(tmp_path: pytest.TempPathFactory) -> None:
    """Write with one instance, re-open same namespace, read back data."""
    db_path = str(tmp_path / "test.lance")

    ns1 = NamespacedLanceDB(db_path, namespace="hifi-dev")
    ns1.create_table(
        "records",
        data=_make_data([{"id": "r1", "value": "hello"}, {"id": "r2", "value": "world"}]),
    )

    # Re-open using a new instance pointing at the same db
    ns2 = NamespacedLanceDB(db_path, namespace="hifi-dev")
    table = ns2.open_table("records")
    df = table.to_pandas()

    assert len(df) == 2
    assert set(df["id"]) == {"r1", "r2"}


def test_namespace_switch_reads_correct_data(tmp_path: pytest.TempPathFactory) -> None:
    """Switching namespace reads the correct (different) table content."""
    db_path = str(tmp_path / "test.lance")

    dev = NamespacedLanceDB(db_path, namespace="hifi-dev")
    eval_ns = NamespacedLanceDB(db_path, namespace="hifi-eval")

    dev.create_table("data", data=_make_data([{"id": "d1", "value": "dev-data"}]))
    eval_ns.create_table("data", data=_make_data([{"id": "e1", "value": "eval-data"}]))

    dev_df = dev.open_table("data").to_pandas()
    eval_df = eval_ns.open_table("data").to_pandas()

    assert dev_df["value"].iloc[0] == "dev-data"
    assert eval_df["value"].iloc[0] == "eval-data"


# ---------------------------------------------------------------------------
# KnowledgeStore backward compatibility
# ---------------------------------------------------------------------------


def test_knowledge_store_namespace_default_backward_compatible(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """KnowledgeStore with default namespace='' behaves identically to Phase 13."""
    from hifi.knowledge.vector_store import KnowledgeStore

    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=32)
    stats = store.get_stats()
    assert stats["n_chunks"] == 0
    # Physical table name should be "chunks_a" (no prefix)
    from hifi.knowledge.namespaced_store import NamespacedLanceDB

    ns = NamespacedLanceDB(str(tmp_path / "knowledge.lance"), namespace="")
    assert "chunks_a" in ns.list_tables()


def test_knowledge_store_with_namespace(tmp_path: pytest.TempPathFactory) -> None:
    """KnowledgeStore with namespace='hifi-dev' uses prefixed table name."""
    from hifi.knowledge.vector_store import KnowledgeStore

    store = KnowledgeStore(
        data_dir=tmp_path, chunking_config="A", dimensions=32, namespace="hifi-dev"
    )
    stats = store.get_stats()
    assert stats["n_chunks"] == 0
    # Physical table should be "hifi-dev-chunks_a"
    from hifi.knowledge.namespaced_store import NamespacedLanceDB

    ns = NamespacedLanceDB(str(tmp_path / "knowledge.lance"), namespace="")
    assert "hifi-dev-chunks_a" in ns.list_tables()


def test_knowledge_store_different_namespaces_isolated(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """KnowledgeStore in hifi-dev and hifi-eval share no tables."""
    from hifi.knowledge.vector_store import KnowledgeStore

    store_dev = KnowledgeStore(
        data_dir=tmp_path, chunking_config="A", dimensions=32, namespace="hifi-dev"
    )
    store_eval = KnowledgeStore(
        data_dir=tmp_path, chunking_config="A", dimensions=32, namespace="hifi-eval"
    )
    # Neither store sees the other's tables
    dev_ns = store_dev._ns
    eval_ns = store_eval._ns
    assert "chunks_a" in dev_ns.list_tables()
    assert "chunks_a" in eval_ns.list_tables()
    # Underlying physical names are different
    ns_raw = NamespacedLanceDB(str(tmp_path / "knowledge.lance"), namespace="")
    all_tables = ns_raw.list_tables()
    assert "hifi-dev-chunks_a" in all_tables
    assert "hifi-eval-chunks_a" in all_tables
