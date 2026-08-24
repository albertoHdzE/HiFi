"""Technical fine-tune server model selection and usability probe (DJ-120).

Regression suite for the 2026-08-04 outage: an unrelated `transformers` job
downloaded google/gemma-2-2b-it into the shared ~/.cache/huggingface/hub, the
orchestrator picked the served model with ["data"][0]["id"], every technical
request was routed to a 2B model that rejects the system role, and the night
died as 98x "404 System role not supported" -> 98x AGGREGATE FAIL -> no
ensemble for arms A and B.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import run_phase15_orchestrator as orch  # noqa: E402

# The exact string the server registers for our fine-tuned technical model.
SERVED = "/Users/alberto/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-8bit"
WANTED = "qwen2.5-coder-32b"


class TestSelectServedModel:
    def test_picks_local_path_over_cache_noise(self):
        # The literal 2026-08-04 listing, in the order that broke production.
        ids = ["google/gemma-2-2b-it", SERVED]
        assert orch._select_served_model(ids, WANTED) == SERVED

    def test_order_does_not_matter(self):
        # The defect was positional; selection must be order-invariant.
        ids = [SERVED, "google/gemma-2-2b-it"]
        assert orch._select_served_model(ids, WANTED) == SERVED

    @pytest.mark.parametrize("noise", [
        "EleutherAI/pythia-410m",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",      # same family, different size
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-2b-it",
    ])
    def test_ignores_every_advances_exam_model(self, noise):
        # Every model the sibling project prefetches must be inert.
        assert orch._select_served_model([noise, SERVED], WANTED) == SERVED

    def test_single_hf_id_accepted_when_no_local_path(self):
        # Server started with an HF id rather than a path: still unambiguous.
        ids = ["google/gemma-2-2b-it", "Qwen/Qwen2.5-Coder-32B-Instruct"]
        assert orch._select_served_model(ids, WANTED) == "Qwen/Qwen2.5-Coder-32B-Instruct"

    def test_hf_original_alongside_local_prefers_local(self):
        # THE SILENT-FAILURE CASE. The HF original of the same model carries no
        # technical_v2 adapter; picking it would yield plausible, wrong output
        # with no error anywhere. The locally-pathed (loaded) model must win.
        ids = ["Qwen/Qwen2.5-Coder-32B-Instruct", SERVED]
        assert orch._select_served_model(ids, WANTED) == SERVED

    def test_no_match_raises_with_diagnostic(self):
        ids = ["google/gemma-2-2b-it", "EleutherAI/pythia-410m"]
        with pytest.raises(LookupError, match="no served model matches"):
            orch._select_served_model(ids, WANTED)

    def test_empty_listing_raises(self):
        with pytest.raises(LookupError):
            orch._select_served_model([], WANTED)

    def test_two_local_paths_is_ambiguous_not_a_guess(self):
        ids = [SERVED, "/opt/models/Qwen2.5-Coder-32B-Instruct-8bit"]
        with pytest.raises(LookupError, match="ambiguous"):
            orch._select_served_model(ids, WANTED)

    def test_two_hf_ids_no_local_is_ambiguous(self):
        ids = ["Qwen/Qwen2.5-Coder-32B-Instruct", "someone/Qwen2.5-Coder-32B-fork"]
        with pytest.raises(LookupError, match="ambiguous"):
            orch._select_served_model(ids, WANTED)

    def test_matching_is_case_insensitive(self):
        assert orch._select_served_model(["/m/QWEN2.5-CODER-32B-x"], WANTED) \
            == "/m/QWEN2.5-CODER-32B-x"


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
            status, detail = orch._probe_chat_model("http://localhost:1235/v1", SERVED)
        assert status == "ok" and detail == "ok"

    def test_system_role_rejection_is_unusable(self):
        # The verbatim 2026-08-04 signature. Selection alone cannot detect this:
        # the model is served and listed, it just cannot accept our prompt shape.
        err = self._http_error(404, b'{"error": "System role not supported"}')
        with patch("urllib.request.urlopen", side_effect=err):
            status, detail = orch._probe_chat_model(
                "http://localhost:1235/v1", "google/gemma-2-2b-it")
        assert status == "unusable"
        assert "System role not supported" in detail

    def test_cold_model_timeout_is_inconclusive_not_failure(self):
        # REGRESSION: the first live probe of the correct model timed out at
        # 120 s because the 32B weights were cold. Must not abort the run.
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            status, detail = orch._probe_chat_model("http://localhost:1235/v1", SERVED)
        assert status == "inconclusive" and "timed out" in detail

    def test_connection_failure_is_inconclusive(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            status, _ = orch._probe_chat_model("http://localhost:1235/v1", SERVED)
        assert status == "inconclusive"

    def test_server_5xx_is_inconclusive(self):
        # 5xx is a verdict on the server, not on our model choice.
        with patch("urllib.request.urlopen", side_effect=self._http_error(503)):
            status, _ = orch._probe_chat_model("http://localhost:1235/v1", SERVED)
        assert status == "inconclusive"

    @pytest.mark.parametrize("code", [400, 404, 422])
    def test_any_4xx_is_unusable(self, code):
        with patch("urllib.request.urlopen", side_effect=self._http_error(code)):
            status, _ = orch._probe_chat_model("http://localhost:1235/v1", SERVED)
        assert status == "unusable"

    def test_non_200_body_response_is_unusable(self):
        with patch("urllib.request.urlopen", return_value=self._resp(404)):
            status, _ = orch._probe_chat_model("http://localhost:1235/v1", SERVED)
        assert status == "unusable"
