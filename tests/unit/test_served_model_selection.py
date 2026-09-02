"""Model routing on the live path: probe, and the absence of any substitution.

Two defects converge here.

DJ-120 (2026-08-04): an unrelated `transformers` job downloaded
google/gemma-2-2b-it into the shared HF cache, the orchestrator picked the served
model by ["data"][0]["id"], every technical request was routed to a 2B model that
rejects the system role, and the night died as 98x "404 System role not
supported" -> 98x AGGREGATE FAIL -> no ensemble for arms A and B. The name-based
selector that fixed it belonged to the mlx_lm fine-tune servers, which are now
retired; what survives, and what these tests pin, is the *usability probe* — the
part that catches a model that is served and healthy yet cannot answer. It now
guards the LM Studio path, which is the one that actually runs every night.

DJ-135: ``_setup_agent_model`` used to fall back to a fine-tune server when the
configured model would not load, overwriting HIFI_FUNDAMENTAL_MODEL and returning
True so the sweep proceeded. That is DJ-124's shape — a model nobody chose,
recorded in sidecars that look valid. TestNoModelSubstitution pins that no such
path exists any more.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hifi.live import models as orch

LM = "http://localhost:1234/v1"
MODEL = "qwen2.5-coder-32b-instruct-mlx"


class TestProbeChatModel:
    """Tri-state: abort only on positive evidence the model is wrong.

    A probe that merely fails to answer in time is NOT evidence of a wrong
    model — a cold 32B took >120 s to first token in live testing. Treating
    that as failure would abort healthy nights, which is strictly worse than
    the outage this guard exists to prevent.
    """

    def _resp(self, status=200):
        class _R:
            def __init__(self, s): self.status = s
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _R(status)

    def _http_error(self, code, body=b""):
        import io
        import urllib.error
        return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body))

    def test_healthy_model_probes_ok(self):
        with patch("urllib.request.urlopen", return_value=self._resp(200)):
            status, detail = orch._probe_chat_model(LM, MODEL)
        assert status == "ok" and detail == "ok"

    def test_system_role_rejection_is_unusable(self):
        # The verbatim 2026-08-04 signature: the model is served and listed, it
        # just cannot accept our prompt shape. Nothing short of a probe sees it.
        err = self._http_error(404, b'{"error": "System role not supported"}')
        with patch("urllib.request.urlopen", side_effect=err):
            status, detail = orch._probe_chat_model(LM, "google/gemma-2-2b-it")
        assert status == "unusable"
        assert "System role not supported" in detail

    def test_cold_model_timeout_is_inconclusive_not_failure(self):
        # REGRESSION: the first live probe of the correct model timed out at
        # 120 s because the 32B weights were cold. Must not abort the run.
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            status, detail = orch._probe_chat_model(LM, MODEL)
        assert status == "inconclusive" and "timed out" in detail

    def test_connection_failure_is_inconclusive(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            status, _ = orch._probe_chat_model(LM, MODEL)
        assert status == "inconclusive"

    def test_server_5xx_is_inconclusive(self):
        # 5xx is a verdict on the server, not on our model choice.
        with patch("urllib.request.urlopen", side_effect=self._http_error(503)):
            status, _ = orch._probe_chat_model(LM, MODEL)
        assert status == "inconclusive"

    @pytest.mark.parametrize("code", [400, 404, 422])
    def test_any_4xx_is_unusable(self, code):
        with patch("urllib.request.urlopen", side_effect=self._http_error(code)):
            status, _ = orch._probe_chat_model(LM, MODEL)
        assert status == "unusable"

    def test_non_200_body_response_is_unusable(self):
        with patch("urllib.request.urlopen", return_value=self._resp(404)):
            status, _ = orch._probe_chat_model(LM, MODEL)
        assert status == "unusable"


class TestProbeGuardsTheLivePath:
    """The probe now runs on every LM Studio load, not only the retired servers."""

    def test_unusable_model_aborts_the_pass(self, monkeypatch):
        monkeypatch.delenv("HIFI_TECHNICAL_MODEL", raising=False)
        with patch.object(orch, "_probe_chat_model",
                          return_value=("unusable", "HTTP 404: System role")), \
             patch("hifi.simulation.model_manager.model_is_loaded", return_value=True):
            ok = orch._setup_agent_model("technical", MODEL, "HIFI_TECHNICAL_MODEL", 300)
        assert ok is False, (
            "a served-but-unusable model must abort the pass, not run 98 tickers "
            "into 98 identical 404s"
        )

    def test_inconclusive_probe_proceeds(self, monkeypatch):
        monkeypatch.delenv("HIFI_TECHNICAL_MODEL", raising=False)
        with patch.object(orch, "_probe_chat_model",
                          return_value=("inconclusive", "timed out")), \
             patch("hifi.simulation.model_manager.model_is_loaded", return_value=True):
            ok = orch._setup_agent_model("technical", MODEL, "HIFI_TECHNICAL_MODEL", 300)
        assert ok is True, "cold weights are not evidence of a wrong model"


class TestNoModelSubstitution:
    """DJ-135. A pass that cannot run its configured model must fail, not swap."""

    def test_load_failure_returns_false_and_sets_nothing(self, monkeypatch):
        monkeypatch.delenv("HIFI_FUNDAMENTAL_MODEL", raising=False)
        monkeypatch.delenv("HIFI_FUNDAMENTAL_FINETUNE_URL", raising=False)
        with patch("hifi.simulation.model_manager.model_is_loaded", return_value=False), \
             patch("hifi.simulation.model_manager.unload_all"), \
             patch("hifi.simulation.model_manager.load_model", return_value=False):
            ok = orch._setup_agent_model(
                "fundamental", "llama-3.3-70b-instruct", "HIFI_FUNDAMENTAL_MODEL", 600)

        assert ok is False
        import os
        assert "HIFI_FUNDAMENTAL_MODEL" not in os.environ, (
            "a failed load must leave no model configured; the old code set this "
            "to the fine-tune model and returned True (DJ-124's shape)"
        )
        assert "HIFI_FUNDAMENTAL_FINETUNE_URL" not in os.environ

    def test_no_source_line_can_set_a_finetune_url(self):
        src = Path(orch.__file__).read_text()
        for var in ("HIFI_FUNDAMENTAL_FINETUNE_URL", "HIFI_TECHNICAL_FINETUNE_URL",
                    "HIFI_FUNDAMENTAL_FINETUNE_MODEL"):
            assert f'os.environ["{var}"] =' not in src, (
                f"{var} is assigned somewhere in hifi.live.models; the only "
                "permitted operation on a fine-tune env var is os.environ.pop"
            )

    def test_stale_finetune_url_in_the_environment_is_scrubbed(self, monkeypatch):
        # The value could arrive from a shell export or a stale .env; the agents
        # read it unconditionally, so the orchestrator must clear it.
        monkeypatch.setenv("HIFI_TECHNICAL_FINETUNE_URL", "http://localhost:1235/v1")
        with patch.object(orch, "_probe_chat_model", return_value=("ok", "ok")), \
             patch("hifi.simulation.model_manager.model_is_loaded", return_value=True):
            orch._setup_agent_model("technical", MODEL, "HIFI_TECHNICAL_MODEL", 300)
        import os
        assert "HIFI_TECHNICAL_FINETUNE_URL" not in os.environ

    @pytest.mark.parametrize("config_name",
                             ["_AGENT_CONFIG", "_HOMOGENEOUS_AGENT_CONFIG"])
    def test_every_agent_names_its_model(self, config_name):
        # A None model id used to mean "route to the fine-tune server". No entry
        # may carry one, or the routing question reopens.
        for agent_type, model_id, env_var, timeout, _ctx in getattr(orch, config_name):
            assert model_id, f"{agent_type} in {config_name} names no model"
            assert env_var, f"{agent_type} in {config_name} names no env var"
            assert timeout > 0

    def test_retired_finetune_constants_are_gone(self):
        for name in ("_TECHNICAL_FINETUNE_URL", "_FUNDAMENTAL_FINETUNE_URL",
                     "_FINETUNE_HEALTH_1235", "_FINETUNE_HEALTH_1236",
                     "_FINETUNE_MODEL", "_select_served_model", "_port_is_listening"):
            assert not hasattr(orch, name), (
                f"{name} survives; it belongs to the retired fine-tune routing"
            )
