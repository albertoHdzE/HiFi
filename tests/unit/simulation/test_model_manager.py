"""
Unit tests for hifi.simulation.model_manager (E0-T3, DJ-106).

Tests the public API with monkeypatched subprocess and HTTP calls.
No real LM Studio instance required.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_lms_run(monkeypatch, returncode: int, output: str = ""):
    """Patch _lms_run to return a fixed (returncode, output) pair."""
    import hifi.simulation.model_manager as mm

    monkeypatch.setattr(mm, "_lms_run", lambda *args, **kwargs: (returncode, output))


def _patch_lm_api(monkeypatch, response: dict):
    """Patch _lm_api to return a fixed response dict."""
    import hifi.simulation.model_manager as mm

    monkeypatch.setattr(mm, "_lm_api", lambda *args, **kwargs: response)


# ---------------------------------------------------------------------------
# get_loaded_ids
# ---------------------------------------------------------------------------


def test_get_loaded_ids_returns_empty_on_connection_error(monkeypatch):
    import hifi.simulation.model_manager as mm

    def _raise(*args, **kwargs):
        raise ConnectionRefusedError("LM Studio not running")

    monkeypatch.setattr(mm, "_lm_api", _raise)
    result = mm.get_loaded_ids()
    assert result == set()


def test_get_loaded_ids_returns_loaded_models(monkeypatch):
    _patch_lm_api(monkeypatch, {
        "data": [
            {"id": "model-a", "state": "loaded"},
            {"id": "model-b", "state": "loading"},
            {"id": "model-c", "state": "loaded"},
        ]
    })
    from hifi.simulation.model_manager import get_loaded_ids

    result = get_loaded_ids()
    assert result == {"model-a", "model-c"}


# ---------------------------------------------------------------------------
# model_is_loaded
# ---------------------------------------------------------------------------


def test_model_is_loaded_false_when_empty(monkeypatch):
    _patch_lm_api(monkeypatch, {"data": []})
    from hifi.simulation.model_manager import model_is_loaded

    assert model_is_loaded("my-model") is False


def test_model_is_loaded_true_exact_match(monkeypatch):
    _patch_lm_api(monkeypatch, {"data": [{"id": "my-model", "state": "loaded"}]})
    from hifi.simulation.model_manager import model_is_loaded

    assert model_is_loaded("my-model") is True


def test_model_is_loaded_true_substring_match(monkeypatch):
    _patch_lm_api(
        monkeypatch,
        {"data": [{"id": "mlx-community/Llama-3.3-70B-Instruct-4bit", "state": "loaded"}]},
    )
    from hifi.simulation.model_manager import model_is_loaded

    # Substring: short form matches full form
    assert model_is_loaded("Llama-3.3-70B-Instruct-4bit") is True


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


def test_load_model_already_loaded_returns_true_without_lms(monkeypatch):
    _patch_lm_api(monkeypatch, {"data": [{"id": "my-model", "state": "loaded"}]})
    lms_called = []

    import hifi.simulation.model_manager as mm
    monkeypatch.setattr(mm, "_lms_run", lambda *a, **kw: (lms_called.append(a), (0, ""))[1])

    result = mm.load_model("my-model")
    assert result is True
    assert lms_called == []  # lms CLI not invoked


def test_load_model_success(monkeypatch):
    # Not loaded → lms load succeeds
    call_count = {"n": 0}

    import hifi.simulation.model_manager as mm

    def _api(*args, **kwargs):
        call_count["n"] += 1
        return {"data": []}  # always empty (not loaded)

    monkeypatch.setattr(mm, "_lm_api", _api)
    _patch_lms_run(monkeypatch, 0, "")

    result = mm.load_model("new-model")
    assert result is True


def test_load_model_failure(monkeypatch):
    import hifi.simulation.model_manager as mm

    monkeypatch.setattr(mm, "_lm_api", lambda *a, **kw: {"data": []})
    _patch_lms_run(monkeypatch, 1, "Error: model not found")

    result = mm.load_model("bad-model")
    assert result is False


# ---------------------------------------------------------------------------
# unload_model
# ---------------------------------------------------------------------------


def test_unload_model_success(monkeypatch):
    _patch_lms_run(monkeypatch, 0)
    from hifi.simulation.model_manager import unload_model

    # Should not raise
    unload_model("my-model")


def test_unload_model_failure_does_not_raise(monkeypatch):
    _patch_lms_run(monkeypatch, 1, "model not loaded")
    from hifi.simulation.model_manager import unload_model

    # Should log warning but not raise
    unload_model("bad-model")
