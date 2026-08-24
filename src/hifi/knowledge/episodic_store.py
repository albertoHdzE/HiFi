"""
EpisodicStore: LanceDB-backed episodic memory for agent decisions (E5-T2, DJ-092).

Stores one EpisodeRecord per agent call and per ensemble run.  Records are
embedded on reasoning_summary via an injected EmbeddingModel so that
search() can do semantic cosine ANN retrieval.

Sentinel values for nullable fields (LanceDB does not support native NULLs
in fixed-type PyArrow schemas without optional fields):
  - collective_decision, labeled_at : "" means NULL
  - forward_return                  : float("nan") means NULL
  - outcome_correct                 : -1 means NULL, 0 = False, 1 = True

search(ticker, regime, sector, outcome_correct=True, n=5):
  1. Embed a structured query string from parameters.
  2. Cosine ANN search for 5*n candidates.
  3. Post-filter: outcome_correct==1 (when requested), regime_label, sector.
  4. Return up to n EpisodeRecord objects.

get_unlabeled_past_horizon(horizon_days=60):
  Returns all records where labeled_at == "" AND decision_date + horizon_days
  is today or earlier.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
from pydantic import BaseModel

from hifi.knowledge.namespaced_store import NamespacedLanceDB

logger = logging.getLogger(__name__)

# Sentinel constants
_STR_NULL = ""
_INT8_NULL: int = -1


def _make_schema(embedding_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("episode_id", pa.string()),
            pa.field("ticker", pa.string()),
            pa.field("decision_date", pa.string()),
            pa.field("regime_label", pa.string()),
            pa.field("sector", pa.string()),
            pa.field("agent_type", pa.string()),
            pa.field("decision", pa.string()),
            pa.field("confidence", pa.float64()),
            pa.field("collective_decision", pa.string()),   # ""=None
            pa.field("forward_return", pa.float64()),       # NaN=None
            pa.field("outcome_correct", pa.int8()),         # -1=None, 0=False, 1=True
            pa.field("reasoning_summary", pa.string()),
            pa.field("labeled_at", pa.string()),            # ""=None
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
        ]
    )


class EpisodeRecord(BaseModel):
    """Single agent decision episode, with optional outcome fields."""

    episode_id: str
    ticker: str
    decision_date: str           # ISO 8601 "YYYY-MM-DD"
    regime_label: str            # RegimeLabel value
    sector: str
    agent_type: str              # "fundamental" | … | "ensemble"
    decision: str                # "Buy" | "Hold" | "Sell"
    confidence: float
    collective_decision: str | None = None
    forward_return: float | None = None
    outcome_correct: bool | None = None
    reasoning_summary: str = ""
    labeled_at: str | None = None


class EpisodicStore:
    """
    LanceDB-backed store for agent decision episodes with embedding-based search.

    Parameters
    ----------
    embedding_model : object
        Any object with:
          .dimensions -> int
          .embed(texts: list[str]) -> list[list[float]]
          .embed_one(text: str) -> list[float]
    namespace : str
        LanceDB namespace prefix (default "hifi-episodes").
    db_path : str | None
        LanceDB directory path. Defaults to
        ``{HIFI_DATA_DIR}/knowledge.lance`` (env var, else ``data/knowledge.lance``).
    """

    TABLE = "episodes"

    def __init__(
        self,
        embedding_model: Any,
        namespace: str = "hifi-episodes",
        db_path: str | None = None,
    ) -> None:
        self._model = embedding_model
        dim = embedding_model.dimensions
        self._schema = _make_schema(dim)

        _path = db_path or str(
            Path(os.environ.get("HIFI_DATA_DIR", "data")) / "knowledge.lance"
        )
        Path(_path).mkdir(parents=True, exist_ok=True)
        self._ns = NamespacedLanceDB(_path, namespace=namespace)
        self._table = self._ns.open_or_create_table(self.TABLE, self._schema)

    # ------------------------------------------------------------------
    # Sentinel helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_row(record: EpisodeRecord, embedding: list[float]) -> dict[str, Any]:
        """Convert EpisodeRecord to a storage dict (sentinel encoding)."""
        oc: int = _INT8_NULL
        if record.outcome_correct is True:
            oc = 1
        elif record.outcome_correct is False:
            oc = 0

        fr = float("nan") if record.forward_return is None else float(record.forward_return)

        return {
            "episode_id": record.episode_id,
            "ticker": record.ticker,
            "decision_date": record.decision_date,
            "regime_label": record.regime_label,
            "sector": record.sector,
            "agent_type": record.agent_type,
            "decision": record.decision,
            "confidence": float(record.confidence),
            "collective_decision": record.collective_decision or _STR_NULL,
            "forward_return": fr,
            "outcome_correct": oc,
            "reasoning_summary": record.reasoning_summary,
            "labeled_at": record.labeled_at or _STR_NULL,
            "embedding": pa.array(embedding, type=pa.float32()),
        }

    @staticmethod
    def _from_row(row: dict[str, Any]) -> EpisodeRecord:
        """Convert a storage dict back to EpisodeRecord (sentinel decoding)."""
        oc_raw = int(row.get("outcome_correct", _INT8_NULL))
        outcome_correct = None if oc_raw == _INT8_NULL else oc_raw == 1

        fr_raw = float(row.get("forward_return", float("nan")))
        forward_return = None if (fr_raw != fr_raw) else fr_raw  # NaN check

        def _opt_str(v: Any) -> str | None:
            s = str(v) if v is not None else _STR_NULL
            return None if s == _STR_NULL else s

        return EpisodeRecord(
            episode_id=str(row["episode_id"]),
            ticker=str(row["ticker"]),
            decision_date=str(row["decision_date"]),
            regime_label=str(row["regime_label"]),
            sector=str(row["sector"]),
            agent_type=str(row["agent_type"]),
            decision=str(row["decision"]),
            confidence=float(row["confidence"]),
            collective_decision=_opt_str(row.get("collective_decision", "")),
            forward_return=forward_return,
            outcome_correct=outcome_correct,
            reasoning_summary=str(row.get("reasoning_summary", "")),
            labeled_at=_opt_str(row.get("labeled_at", "")),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, episode: EpisodeRecord) -> None:
        """Embed reasoning_summary and append episode to the table."""
        embedding = self._model.embed_one(episode.reasoning_summary or episode.agent_type)
        row_dict = self._to_row(episode, embedding)

        batch = pa.table(
            {k: [v] for k, v in row_dict.items()},
            schema=self._schema,
        )
        self._table.add(batch)
        logger.debug(
            "Stored episode %s ticker=%s agent=%s decision=%s",
            episode.episode_id, episode.ticker, episode.agent_type, episode.decision,
        )

    def update(self, episode: EpisodeRecord) -> None:
        """Replace an existing episode (delete + re-add).  Idempotent."""
        self._table.delete(f"episode_id = '{episode.episode_id}'")
        self.add(episode)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        ticker: str,
        regime: str,
        sector: str,
        outcome_correct: bool | None = True,
        n: int = 5,
    ) -> list[EpisodeRecord]:
        """
        Cosine ANN search for past episodes + post-filter.

        Parameters
        ----------
        ticker : str
            Target ticker (used in query construction).
        regime : str
            Regime label to filter on (e.g. "bull_low_vol").
        sector : str
            GICS sector to filter on.
        outcome_correct : bool | None
            If True: only return correctly-called episodes.
            If None: return all regardless of outcome.
        n : int
            Maximum number of results.

        Returns
        -------
        list[EpisodeRecord]
        """
        try:
            df = self._table.to_pandas()
        except Exception:
            return []

        if df.empty:
            return []

        # Post-filter: regime, sector, outcome_correct
        mask = (
            (df["regime_label"] == regime)
            & (df["sector"] == sector)
        )
        if outcome_correct is True:
            mask &= df["outcome_correct"] == 1
        elif outcome_correct is False:
            mask &= df["outcome_correct"] == 0

        filtered = df[mask].copy()
        if filtered.empty:
            return []

        # ANN search to rank candidates by semantic similarity
        query_text = f"{ticker} {regime} {sector} financial analysis"
        try:
            query_vec = self._model.embed_one(query_text)
            candidates = (
                self._table.search(query_vec, vector_column_name="embedding")
                .metric("cosine")
                .limit(n * 5)
                .to_list()
            )
            ranked_ids = [str(row["episode_id"]) for row in candidates]
            id_order = {eid: i for i, eid in enumerate(ranked_ids)}
            filtered = filtered.copy()
            filtered["_rank"] = filtered["episode_id"].map(id_order).fillna(9999)
            filtered = filtered.sort_values("_rank")
        except Exception as exc:
            logger.debug("ANN search failed, using unranked results: %s", exc)

        rows = filtered.head(n).to_dict("records")
        return [self._from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Unlabeled queries
    # ------------------------------------------------------------------

    def get_unlabeled_past_horizon(
        self,
        horizon_days: int = 60,
        today: date | None = None,
    ) -> list[EpisodeRecord]:
        """
        Return episodes past the labeling horizon that have not yet been labeled.

        Parameters
        ----------
        horizon_days : int
            Number of calendar days after decision_date before labeling is possible.
        today : date | None
            Override for current date (used in tests).

        Returns
        -------
        list[EpisodeRecord]
            Records where labeled_at == "" AND decision_date + horizon_days <= today.
        """
        _today = today or date.today()
        cutoff = _today - timedelta(days=horizon_days)

        try:
            df = self._table.to_pandas()
        except Exception:
            return []

        if df.empty:
            return []

        # labeled_at == "" means unlabeled (sentinel for None)
        mask = (
            (df["labeled_at"] == _STR_NULL)
            & (df["decision_date"] <= cutoff.isoformat())
        )
        rows = df[mask].to_dict("records")
        return [self._from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of episodes in the table."""
        try:
            return len(self._table.to_pandas())
        except Exception:
            return 0
