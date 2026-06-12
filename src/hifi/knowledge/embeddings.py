"""
Embedding pipeline for HiFi Phase 7 RAG (P7-E3).

Wraps the OpenAI-compatible /v1/embeddings endpoint exposed by LM Studio.
Uses the same base_url as the agent LLM (HIFI_LM_STUDIO_URL env var).

Default model: nomic-embed-text-v1.5 (DJ-027)
Default dimensions: 768 (full Matryoshka; configurable for experiments)
Batch size: 32 texts per API call (stays within LM Studio context limits)
"""

from __future__ import annotations

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "nomic-embed-text-v1.5"
_DEFAULT_DIMENSIONS = 768
_BATCH_SIZE = 32


class EmbeddingModel:
    """
    Embed texts using LM Studio's /v1/embeddings endpoint.

    Uses the OpenAI Python client (already a project dependency) pointed at
    the LM Studio server URL. The API is identical to the OpenAI embeddings API.

    Parameters
    ----------
    model : str
        Embedding model identifier in LM Studio (default: nomic-embed-text-v1.5).
    dimensions : int
        Output vector dimensionality (default: 768). Supports Matryoshka
        truncation for nomic-embed-text-v1.5 (valid: 64-768).
    base_url : str | None
        LM Studio server base URL. Defaults to HIFI_LM_STUDIO_URL env var
        (default: http://localhost:1234/v1).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        dimensions: int = _DEFAULT_DIMENSIONS,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        resolved_url = base_url or os.environ.get(
            "HIFI_LM_STUDIO_URL", "http://localhost:1234/v1"
        )
        self._client = OpenAI(base_url=resolved_url, api_key="lm-studio")

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Splits into batches of at most 32 texts to stay within LM Studio limits.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text. Each vector has ``dimensions``
            floats.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[batch_start : batch_start + _BATCH_SIZE]
            logger.debug(
                "Embedding batch %d/%d (%d texts)",
                batch_start // _BATCH_SIZE + 1,
                (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE,
                len(batch),
            )
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_one(self, text: str) -> list[float]:
        """
        Embed a single text. Convenience wrapper around embed().
        """
        return self.embed([text])[0]
