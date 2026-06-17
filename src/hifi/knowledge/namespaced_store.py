"""
NamespacedLanceDB: namespace-prefixed LanceDB table access (E6-T1, DJ-093).

Enables dev/eval/live separation within a single LanceDB database directory.
All table operations are routed through a namespace prefix:
  - namespace="hifi-dev"  -> tables: "hifi-dev-episodes", "hifi-dev-chunks_a", ...
  - namespace="hifi-eval" -> tables: "hifi-eval-episodes", ...
  - namespace=""           -> no prefix (backward compatible with Phase 7-13 tables)

All LanceDB wrappers that need namespace support accept an optional
``namespace: str = ""`` parameter and route table operations here.

Phase 15 walk-forward evaluation uses ``namespace="hifi-eval"`` with temporal
filtering so that no future data leaks into earlier evaluation windows (DJ-093).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NamespacedLanceDB:
    """
    Wraps a LanceDB client with a namespace prefix on all table names.

    Enables dev/eval/live separation without separate database directories.
    Multiple NamespacedLanceDB instances may point at the same ``db_path``
    with different namespaces; they share the same underlying database files
    but operate on non-overlapping sets of tables.

    Parameters
    ----------
    db_path : str
        Path to the LanceDB database directory (created if absent).
    namespace : str
        Prefix for all table names.  Use ``""`` for no prefix (backward
        compatible with existing Phase 7-13 tables).
        Recommended values: ``"hifi-dev"``, ``"hifi-eval"``, ``"hifi-live"``.
    """

    def __init__(self, db_path: str, namespace: str = "") -> None:
        self._namespace = namespace
        self._db_path = db_path
        import lancedb

        self._db = lancedb.connect(db_path)

    @property
    def namespace(self) -> str:
        """The namespace prefix used by this instance."""
        return self._namespace

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prefixed(self, table_name: str) -> str:
        """Return the physical table name with namespace prefix applied."""
        if self._namespace:
            return f"{self._namespace}-{table_name}"
        return table_name

    def _list_all_tables(self) -> list[str]:
        """Return all physical table names from the underlying LanceDB.

        Handles both the older lancedb API (TableNameList with a ``.tables``
        attribute) and the newer API that returns a plain list.
        """
        result = self._db.list_tables()
        if hasattr(result, "tables"):
            return list(result.tables)
        return list(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_table(self, table_name: str) -> Any:
        """Open a table, applying the namespace prefix to its name."""
        return self._db.open_table(self._prefixed(table_name))

    def create_table(self, table_name: str, data: Any = None, schema: Any = None) -> Any:
        """Create a table, applying the namespace prefix to its name.

        Parameters
        ----------
        table_name : str
            Logical table name (prefix applied automatically).
        data : PyArrow Table, optional
            Initial data for the table.
        schema : PyArrow Schema, optional
            Schema to use when creating an empty table.
        """
        kwargs: dict[str, Any] = {}
        if data is not None:
            kwargs["data"] = data
        if schema is not None:
            kwargs["schema"] = schema
        return self._db.create_table(self._prefixed(table_name), **kwargs)

    def drop_table(self, table_name: str) -> None:
        """Drop a table, applying the namespace prefix to its name."""
        self._db.drop_table(self._prefixed(table_name))

    def list_tables(self) -> list[str]:
        """Return logical table names visible from this namespace.

        With ``namespace=""``: returns all physical table names unchanged.
        With ``namespace="hifi-dev"``: returns only tables whose physical
        name starts with ``"hifi-dev-"``, with the prefix stripped.
        """
        all_tables = self._list_all_tables()
        if not self._namespace:
            return all_tables
        prefix = f"{self._namespace}-"
        return [t[len(prefix) :] for t in all_tables if t.startswith(prefix)]

    def table_exists(self, table_name: str) -> bool:
        """Return True if the (prefixed) table exists in the database."""
        return table_name in self.list_tables()

    def open_or_create_table(self, table_name: str, schema: Any) -> Any:
        """Open an existing table or create it empty with the given PyArrow schema.

        Convenience method used by LanceDB wrappers to avoid boilerplate.

        Parameters
        ----------
        table_name : str
            Logical table name (prefix applied automatically).
        schema : pyarrow.Schema
            Schema for the new table if it does not yet exist.

        Returns
        -------
        lancedb.table.Table
            The opened or newly created table.
        """
        import pyarrow as pa

        if self.table_exists(table_name):
            return self.open_table(table_name)
        empty = pa.table(
            {field.name: pa.array([], type=field.type) for field in schema},
            schema=schema,
        )
        return self.create_table(table_name, data=empty)
