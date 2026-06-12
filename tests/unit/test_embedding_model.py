"""
Unit tests for EmbeddingModel and DeterministicEmbeddingModel (P7-E3).

Live LM Studio is not required. EmbeddingModel URL/env var tests use
monkeypatching. DeterministicEmbeddingModel tests are fully deterministic.
"""

from __future__ import annotations

import numpy as np

from hifi.knowledge.embeddings import EmbeddingModel

# Import DeterministicEmbeddingModel from conftest where it lives
from tests.conftest import DeterministicEmbeddingModel

# ---------------------------------------------------------------------------
# P7-E3-T5: DeterministicEmbeddingModel returns correct shape
# ---------------------------------------------------------------------------


def test_deterministic_embed_returns_correct_length():
    model = DeterministicEmbeddingModel(dimensions=768)
    texts = ["Hello world", "Apple revenue declined"]
    result = model.embed(texts)
    assert len(result) == 2
    assert all(len(v) == 768 for v in result)


def test_deterministic_embed_default_dimensions():
    model = DeterministicEmbeddingModel()
    assert model.dimensions == 768


def test_deterministic_embed_custom_dimensions():
    model = DeterministicEmbeddingModel(dimensions=128)
    result = model.embed(["test text"])
    assert len(result[0]) == 128


# ---------------------------------------------------------------------------
# P7-E3-T6: Same input always produces the same embedding
# ---------------------------------------------------------------------------


def test_deterministic_same_input_same_vector():
    model = DeterministicEmbeddingModel(dimensions=64)
    text = "Apple Inc. reported revenue of $117 billion."
    v1 = model.embed_one(text)
    v2 = model.embed_one(text)
    assert v1 == v2


def test_deterministic_batch_same_as_individual():
    model = DeterministicEmbeddingModel(dimensions=64)
    texts = ["text A", "text B", "text C"]
    batch_result = model.embed(texts)
    for i, t in enumerate(texts):
        individual = model.embed_one(t)
        assert batch_result[i] == individual


# ---------------------------------------------------------------------------
# P7-E3-T7: Different inputs produce different embeddings
# ---------------------------------------------------------------------------


def test_deterministic_different_inputs_different_vectors():
    model = DeterministicEmbeddingModel(dimensions=128)
    v1 = model.embed_one("Apple iPhone revenue")
    v2 = model.embed_one("JPMorgan credit loss provisions")
    assert v1 != v2


# ---------------------------------------------------------------------------
# Unit-norm property
# ---------------------------------------------------------------------------


def test_deterministic_embedding_is_unit_normalised():
    model = DeterministicEmbeddingModel(dimensions=128)
    v = model.embed_one("Some financial text about earnings.")
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-6, f"Expected unit norm, got {norm}"


def test_deterministic_embed_empty_list():
    model = DeterministicEmbeddingModel()
    result = model.embed([])
    assert result == []


# ---------------------------------------------------------------------------
# P7-E3-T8: EmbeddingModel uses HIFI_LM_STUDIO_URL env var
# ---------------------------------------------------------------------------


def test_embedding_model_uses_lm_studio_url_env_var(monkeypatch):
    """EmbeddingModel should pick up HIFI_LM_STUDIO_URL from env."""
    monkeypatch.setenv("HIFI_LM_STUDIO_URL", "http://custom-host:5678/v1")
    model = EmbeddingModel()
    # The client's base_url should reflect the env var
    assert "custom-host" in str(model._client.base_url)


def test_embedding_model_accepts_explicit_base_url():
    model = EmbeddingModel(base_url="http://test-server:1234/v1")
    assert "test-server" in str(model._client.base_url)


def test_embedding_model_default_attributes():
    model = EmbeddingModel()
    assert model.dimensions == 768
    assert model.model == "nomic-embed-text-v1.5"


# ---------------------------------------------------------------------------
# P7-E3-T9: batch splitting for large input lists
# ---------------------------------------------------------------------------


def test_deterministic_model_handles_large_batch():
    """Batches of > 32 texts should work correctly."""
    model = DeterministicEmbeddingModel(dimensions=32)
    texts = [f"text number {i}" for i in range(100)]
    result = model.embed(texts)
    assert len(result) == 100
    # Verify determinism for a sample
    assert result[42] == model.embed_one("text number 42")
