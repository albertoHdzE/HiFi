"""DJ-135: the values that moved out of scripts/ must not have changed.

``scripts/run_phase16_live.py`` and ``scripts/run_phase15_orchestrator.py`` held
2,340 lines of the running experiment. Moving them into ``hifi.live`` was meant
to be a pure relocation, and "meant to be" is not a property a repository has —
so the constants and pure functions that crossed the boundary are pinned here
against the values captured immediately before the move.

Anything that changes a number in this file changes what the arms trade, and is
a decision to be made deliberately rather than a refactor.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from hifi.live import accounts, guards, models, paths

_LIVE = pathlib.Path(paths.__file__).parent


class TestModelConfigUnchanged:
    """Model heterogeneity is the treatment. These strings are the experiment."""

    def test_agent_models_and_timeouts(self):
        assert models._AGENT_CONFIG == [
            ("fundamental", "llama-3.3-70b-instruct", "HIFI_FUNDAMENTAL_MODEL", 600, None),
            ("technical", "qwen2.5-coder-32b-instruct-mlx", "HIFI_TECHNICAL_MODEL", 300, None),
            ("risk", "mistral-small-3.2-24b-instruct-2506-mlx", "HIFI_RISK_MODEL", 300, None),
            ("macro", "deepseek-r1-distill-qwen-32b", "HIFI_MACRO_MODEL", 600, None),
            ("sentiment", "gemma-3-12b-it", "HIFI_SENTIMENT_MODEL", 300, 8192),
            ("contrarian", "mlx-qwen3.5-35b-a3b", "HIFI_CONTRARIAN_MODEL", 300, None),
        ]

    def test_sentiment_keeps_its_enlarged_context_window(self):
        # Gemma 12B's default (~4096) truncates tickers with long EDGAR passages;
        # AAPL alone needs ~4,357 tokens of prompt + output.
        ctx = {row[0]: row[4] for row in models._AGENT_CONFIG}
        assert ctx["sentiment"] == 8192
        assert all(v is None for k, v in ctx.items() if k != "sentiment")

    def test_homogeneous_condition_still_available(self):
        # The Phase 15 re-run needs this condition; it is not dead config.
        assert models._agent_config_for_condition("homogeneous") is \
            models._HOMOGENEOUS_AGENT_CONFIG
        for cond in ("full", "parallel", "no-memory"):
            assert models._agent_config_for_condition(cond) is models._AGENT_CONFIG


class TestArmRegistryUnchanged:
    def test_arm_to_condition_mapping(self):
        assert {a: c["condition"] for a, c in accounts._ACCOUNTS.items()} == {
            "A": "parallel", "B": "full", "C": "control", "D": "riskbudget",
        }

    def test_credential_suffix_order(self):
        # Order matters: arm A falls back to unsuffixed keys, so a wrong order
        # would point two arms at the same brokerage account.
        assert accounts._ACCOUNTS["A"]["suffixes"] == ["_FIRST", "_A", ""]
        assert accounts._ACCOUNTS["B"]["suffixes"] == ["_SECOND", "_B"]
        assert accounts._ACCOUNTS["C"]["suffixes"] == ["_THIRD", "_C"]
        assert accounts._ACCOUNTS["D"]["suffixes"] == ["_FOURTH", "_D"]

    def test_only_arm_a_can_use_unsuffixed_credentials(self):
        for arm, cfg in accounts._ACCOUNTS.items():
            if arm != "A":
                assert "" not in cfg["suffixes"]


class TestRiskLimitsUnchanged:
    def test_thresholds(self):
        assert guards._DAILY_LOSS_LIMIT == 0.02
        assert guards._POSITION_LOSS_LIMIT == 0.10
        assert guards._POSITION_IMPACT_LIMIT == 0.02
        assert guards._VANISH_LOOKBACK_SNAPSHOTS == 5

    @pytest.mark.parametrize("n,exposure,expected", [
        (1, 1.0, 0.10),    # fully concentrated: collapses to the position limit
        (2, 1.0, 0.10),
        (10, 1.0, 0.20),
        (30, 1.0, 0.60),   # wide book: relaxes, so breadth is not taxed (DJ-119)
        (98, 1.0, 1.96),
        # Half-deployed: each name is a SMALLER share of equity, so it takes a
        # larger adverse move to cost 2% of the book. The threshold relaxes.
        (30, 0.5, 1.20),
    ])
    def test_effective_halt_threshold(self, n, exposure, expected):
        assert guards.effective_halt_threshold(n, exposure) == pytest.approx(expected)

    @pytest.mark.parametrize("n,exposure", [(0, 1.0), (-1, 1.0), (10, 0.0)])
    def test_degenerate_books_never_halt(self, n, exposure):
        assert guards.effective_halt_threshold(n, exposure) == float("inf")


class TestPathsUnchanged:
    def test_client_order_id_format(self):
        # The broker's idempotency key. A change here makes every prior night's
        # ids unrecognisable and re-enables double-fills on a resumed run.
        assert accounts._client_order_id("A", "2026-08-31", "AAPL", "Buy") == \
            "hifiA-2026-08-31-buy-AAPL"
        assert len(accounts._client_order_id("D", "2026-08-31", "GOOGL", "sell")) <= 48

    def test_decision_and_state_paths(self):
        assert paths._account_dir("B").as_posix().endswith("data/live/B")
        assert paths._decisions_log("B").name == "decisions.jsonl"
        assert paths._breaker_log("B").name == "circuit_breakers.jsonl"
        assert paths._hwm_path("B").name == "hwm.json"

    def test_walkforward_paths(self):
        assert paths._run_id("full", "2026-08-31", "AAPL") == "full-2026-08-31-AAPL"
        assert paths._sidecar_path("d", "full", "2026-08-31", "AAPL", "risk").as_posix() == \
            "d/runs/full-2026-08-31-AAPL/AAPL_risk.json"
        assert paths._ensemble_path("o", "full", "2026-08-31", "AAPL").as_posix() == \
            "o/full/2026/08/AAPL.json"
        assert paths._portfolio_path("o", "full", "2026-08-31").as_posix() == \
            "o/full/2026/08/portfolio.json"

    def test_universe_size(self):
        assert len(paths._get_tickers()) == len(paths._get_sectors())
        assert paths._resolve_tickers(None) == paths._get_tickers()


class TestNoModuleShadowing:
    """A local named for a sibling module silently breaks every call through it.

    This is not hypothetical: ``log_arm_invariance(accounts: list[str])`` took a
    parameter named for the ``accounts`` module, so ``accounts.get_executor``
    resolved to a list and the arm-invariance probe died on the first real run
    after the extraction. Unit tests did not catch it because they patch the
    function rather than call it.
    """

    _MODULES = {p.stem for p in _LIVE.glob("*.py")} - {"__init__"}

    def _bound_names(self, fn: ast.AST) -> set[str]:
        names = {a.arg for a in
                 fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            if isinstance(node, ast.For | ast.comprehension):
                names |= {t.id for t in ast.walk(node.target)
                          if isinstance(t, ast.Name)}
        return names

    @pytest.mark.parametrize("module", sorted(_MODULES))
    def test_no_local_shadows_a_sibling_module(self, module):
        path = _LIVE / f"{module}.py"
        tree = ast.parse(path.read_text())
        imported = {a.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module == "hifi.live"
                    for a in node.names}
        offenders = []
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            clash = self._bound_names(fn) & imported
            if clash:
                offenders.append(f"{module}.{fn.name}():{fn.lineno} shadows {sorted(clash)}")
        assert not offenders, (
            "a local binding hides a sibling module, so every attribute access "
            f"through it fails at runtime: {offenders}"
        )


class TestNoScriptImportsRemain:
    """The sys.path insert that started all of this must not come back."""

    #: Modules that only a CLI in scripts/ may import.
    _FORBIDDEN = {"run_phase16_live", "run_phase15_orchestrator"}

    @pytest.mark.parametrize("module", sorted({p.stem for p in _LIVE.glob("*.py")}))
    def test_live_package_never_imports_from_scripts(self, module):
        # Parsed, not grepped: the docstrings in this package name the old
        # scripts on purpose, to record where the code came from. Only a real
        # import or a real sys.path mutation is a violation.
        tree = ast.parse((_LIVE / f"{module}.py").read_text())
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [a.name for a in node.names if a.name in self._FORBIDDEN]
            elif isinstance(node, ast.ImportFrom):
                if node.module in self._FORBIDDEN:
                    offenders.append(node.module)
            elif isinstance(node, ast.Attribute) and node.attr in ("insert", "append"):
                tgt = node.value
                if (isinstance(tgt, ast.Attribute) and tgt.attr == "path"
                        and isinstance(tgt.value, ast.Name) and tgt.value.id == "sys"):
                    offenders.append("sys.path mutation")
        assert not offenders, (
            f"hifi.live.{module} reaches back into scripts/ ({offenders}); the "
            "package must not depend on the CLI that calls it"
        )

    def test_the_guard_detects_a_planted_violation(self, tmp_path):
        planted = tmp_path / "p.py"
        planted.write_text("import sys\nsys.path.insert(0, 'scripts')\n"
                           "import run_phase16_live\n")
        tree = ast.parse(planted.read_text())
        found = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Import)
                 and any(a.name in self._FORBIDDEN for a in n.names)]
        assert found
