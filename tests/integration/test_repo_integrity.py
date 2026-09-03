"""Every module imports, every declared entry point exists, nothing dangles.

The DJ-135 cleanup moved 2,340 lines between packages, archived 30 scripts and
repointed every Makefile target. All of that is the kind of change whose failure
mode is an ImportError at 19:00 on a night nobody is watching — the tests passed
because the tests never imported the thing that moved.

These are cheap and run in the normal suite.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _src_modules() -> list[str]:
    return sorted(
        "hifi." + str(p.relative_to(_REPO / "src" / "hifi").with_suffix("")
                      ).replace("/", ".")
        for p in (_REPO / "src" / "hifi").rglob("*.py")
        if p.name != "__init__.py"
    )


class TestEverySrcModuleImports:
    @pytest.mark.parametrize("module", _src_modules())
    def test_import(self, module):
        importlib.import_module(module)


class TestEveryScriptParsesAndResolvesTheRepoRoot:
    """The archived scripts moved one directory deeper (DJ-135).

    Each computed the repo root as ``parent.parent``; from ``scripts/archive/``
    that now points at ``scripts/``, so every one of them would read and write
    into the wrong tree while appearing to work.
    """

    @pytest.mark.parametrize("path", sorted(
        p.relative_to(_REPO).as_posix()
        for p in (_REPO / "scripts").rglob("*.py")))
    def test_parses(self, path):
        ast.parse((_REPO / path).read_text())

    @pytest.mark.parametrize("path", sorted(
        p.relative_to(_REPO).as_posix()
        for p in (_REPO / "scripts" / "archive").glob("*.py")))
    def test_archived_script_resolves_the_repo_root(self, path):
        src = (_REPO / path).read_text()
        if "_ROOT" not in src and "PROJECT_ROOT" not in src:
            pytest.skip("script does not resolve a repo root")
        assert "parent.parent" not in src, (
            "an archived script still computes the repo root as parent.parent, "
            "which from scripts/archive/ resolves to scripts/"
        )
        # Depth check, independently of how the script spells it.
        assert (_REPO / path).resolve().parents[2] == _REPO

    @pytest.mark.parametrize("path", sorted(
        p.relative_to(_REPO).as_posix()
        for p in (_REPO / "scripts").rglob("*.sh")))
    def test_shell_script_parses(self, path):
        # Same class of failure as an unparseable .py, and shell is where the
        # operational entry points live: the nightly wrapper and the reset.
        r = subprocess.run(["bash", "-n", str(_REPO / path)],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr

    @pytest.mark.parametrize("path", sorted(
        p.relative_to(_REPO).as_posix()
        for p in (_REPO / "scripts").rglob("*.sh")))
    def test_shell_script_resolves_the_repo_root(self, path):
        """`dirname $0/..` has exactly the .py hazard, and was not guarded.

        genesis2_reset.sh moved into scripts/archive/ carrying a one-level
        `dirname/..`, which from there resolves to scripts/ — it would have
        archived into scripts/data/live/ and reported success.
        """
        src = (_REPO / path).read_text()
        # Both spellings in the tree: "$0" and "${BASH_SOURCE[0]}".
        m = re.search(
            r'ROOT="\$\(cd "\$\(dirname "(?:\$0|\$\{BASH_SOURCE\[0\]\})"\)'
            r'((?:/\.\.)+)"',
            src)
        if m is None:
            pytest.skip("script does not resolve a repo root this way")
        # Depth is counted, not string-matched, so reformatting cannot fool it.
        levels = m.group(1).count("..")
        depth = len((_REPO / path).relative_to(_REPO).parts) - 1
        assert levels == depth, (
            f"{path} climbs {levels} level(s) to find the repo root but sits "
            f"{depth} deep; it would read and write into "
            f"{(_REPO / path).parents[levels - 1]}"
        )


class TestMakefileTargetsResolve:
    def _makefile(self) -> str:
        return (_REPO / "Makefile").read_text()

    def test_every_referenced_script_exists(self):
        missing = [s for s in sorted(set(
            re.findall(r"scripts/[A-Za-z0-9_/]+\.(?:py|sh)", self._makefile())))
            if not (_REPO / s).exists()]
        assert not missing, f"Makefile targets point at missing scripts: {missing}"

    def test_every_referenced_pytest_path_exists(self):
        missing = [p for p in sorted(set(
            re.findall(r"tests/[A-Za-z0-9_/]+\.py", self._makefile())))
            if not (_REPO / p).exists()]
        assert not missing, f"Makefile runs missing test files: {missing}"

    def test_every_phony_target_is_defined(self):
        mk = self._makefile()
        phony = re.search(r"\.PHONY:((?:[^\n\\]|\\\n)*)", mk)
        declared = set(phony.group(1).replace("\\\n", " ").split())
        defined = set(re.findall(r"^([a-zA-Z0-9_-]+):", mk, re.M))
        assert not (declared - defined), (
            f"declared .PHONY but never defined: {sorted(declared - defined)}"
        )

    def test_every_target_has_a_help_string(self):
        # `make help` greps for `## `; a target without one is invisible.
        mk = self._makefile()
        undocumented = [t for t in re.findall(r"^([a-zA-Z0-9_-]+):(.*)$", mk, re.M)
                        if "##" not in t[1] and t[0] not in {"live-reset"}]
        assert not undocumented, (
            f"targets missing from `make help`: {[t[0] for t in undocumented]}"
        )

    def test_help_runs_and_lists_the_operational_targets(self):
        r = subprocess.run(["make", "help"], cwd=_REPO, capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 0
        for target in ("test", "lint", "coverage", "typecheck", "live-nightly",
                       "live-plan", "live-verify", "refresh-data", "archive-help"):
            assert target in r.stdout, f"`make help` does not mention {target}"

    def test_the_retired_live_targets_are_gone(self):
        # One entry point for a real cycle (DJ-135). live-dry-run and
        # live-execute were two ways to say `live-nightly`.
        mk = self._makefile()
        assert "\nlive-dry-run:" not in mk
        assert "\nlive-execute:" not in mk


class TestArchiveIsInert:
    """Nothing on a running path may import from scripts/archive/."""

    def test_no_src_module_references_the_archive(self):
        offenders = []
        for p in (_REPO / "src").rglob("*.py"):
            src = p.read_text()
            if "scripts.archive" in src or "scripts/archive" in src:
                offenders.append(p.relative_to(_REPO).as_posix())
        assert not offenders, f"src/ reaches into the archive: {offenders}"

    def test_the_archive_readme_indexes_every_archived_script(self):
        readme = (_REPO / "scripts" / "archive" / "README.md").read_text()
        undocumented = [p.name for p in (_REPO / "scripts" / "archive").glob("*.py")
                        if p.stem not in readme]
        assert not undocumented, (
            f"archived without an index entry, so nobody can tell what produced "
            f"what: {undocumented}"
        )

    def test_operational_scripts_stayed_out_of_the_archive(self):
        for name in ("hifi_live.py", "hifi_walkforward.py", "refresh_data.py",
                     "verify_agent_repair.py", "nightly_live_execute.sh"):
            assert (_REPO / "scripts" / name).exists(), f"{name} is not operational"
            assert not (_REPO / "scripts" / "archive" / name).exists()


class TestEntryPointsAreThin:
    """The CLIs parse arguments; hifi.live does the work (DJ-135)."""

    @pytest.mark.parametrize("script", ["hifi_live.py", "hifi_walkforward.py"])
    def test_the_cli_defines_almost_no_logic(self, script):
        tree = ast.parse((_REPO / "scripts" / script).read_text())
        funcs = [n.name for n in tree.body
                 if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
        assert set(funcs) <= {"main", "_parse_args", "_tickers"}, (
            f"{script} has grown logic of its own: {funcs}. It should parse "
            "arguments and dispatch into hifi.live."
        )

    @pytest.mark.parametrize("script", ["hifi_live.py", "hifi_walkforward.py"])
    def test_the_cli_help_runs(self, script):
        r = subprocess.run([sys.executable, str(_REPO / "scripts" / script), "--help"],
                           cwd=_REPO, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-800:]
        assert "usage:" in r.stdout

    def test_the_live_cli_exposes_the_documented_flags(self):
        r = subprocess.run([sys.executable, str(_REPO / "scripts" / "hifi_live.py"),
                            "--help"], cwd=_REPO, capture_output=True, text=True,
                           timeout=120)
        for flag in ("--account", "--execute", "--dry-run", "--status",
                     "--update-data", "--snapshot", "--smoke", "--date", "--force"):
            assert flag in r.stdout, f"{flag} disappeared from the live CLI"


class TestRunbookMatchesReality:
    """A runbook that names a command that does not exist is worse than none."""

    def _runbook(self) -> str:
        return (_REPO / "RUNBOOK.md").read_text()

    def test_every_make_target_it_names_exists(self):
        mk = (_REPO / "Makefile").read_text()
        defined = set(re.findall(r"^([a-zA-Z0-9_-]+):", mk, re.M))
        named = set(re.findall(r"make ([a-z][a-z0-9-]+)", self._runbook()))
        assert not (named - defined), (
            f"RUNBOOK names targets that do not exist: {sorted(named - defined)}"
        )

    def test_every_path_it_names_under_src_exists(self):
        missing = [p for p in sorted(set(
            re.findall(r"src/hifi/[a-z_/]+\.py", self._runbook())))
            if not (_REPO / p).exists()]
        assert not missing, f"RUNBOOK points at missing modules: {missing}"

    def test_it_documents_the_dry_run_separation(self):
        # DJ-136: an operator must know a dry run does not enter the record.
        assert "DRY=1" in self._runbook()


class TestEveryArmsDependenciesAreDeclared:
    """An arm may not depend on something no manifest mentions (DJ-136).

    Arm D's signal provider, ``riskbudget``, lived in the venv and in no
    manifest: it had been installed by hand from a sibling repo. A routine
    `uv sync` removed it, and the next cycle ran arms A, B and C normally while
    D failed with ModuleNotFoundError — three arms of a four-arm ablation, which
    is not the experiment.

    The failure was loud in the log and silent in the result: nothing compares
    the arms that ran against the arms that were supposed to.
    """

    def test_riskbudget_is_importable(self):
        importlib.import_module("riskbudget.mcp_server")

    def test_riskbudget_is_declared_in_pyproject(self):
        pp = (_REPO / "pyproject.toml").read_text()
        assert "riskbudget" in pp, (
            "arm D's provider is not declared, so `uv sync` will remove it and "
            "the arm will disappear from the next cycle"
        )

    def test_every_arms_signal_source_resolves(self):
        """One check per arm, at the level of 'can this arm produce signals'."""
        from hifi.live.accounts import _ACCOUNTS

        for arm, cfg in _ACCOUNTS.items():
            condition = cfg["condition"]
            if condition in ("parallel", "full"):
                importlib.import_module("hifi.live.ensemble")
            elif condition == "control":
                importlib.import_module("hifi.live.strategies")
            elif condition == "riskbudget":
                importlib.import_module("hifi.execution.riskbudget_strategy")
                importlib.import_module("riskbudget.mcp_server")
            else:
                pytest.fail(f"arm {arm} has an unrecognised condition {condition!r}")


class TestRetiredAdaptersStayRetired:
    """DJ-124's failure mode was not the adapter. It was the wiring outliving it.

    ``technical_v2`` was measured emitting Buy @ 0.70 on 98/98 tickers and
    collapsing ensemble entropy 0.367 -> 0.000. It was rejected in a bitacora
    while remaining the artifact the technical agent actually loaded.

    DJ-135 removed the routes from ``hifi.live``. DJ-139 found the rest still
    standing: ``run_phase15_smoke.py`` probed port 1235 and would have found it
    READY, and ``watchdog_walkforward.sh`` started an mlx_lm server against the
    adapter before every sweep. Both are ports of entry to the *Phase 15 re-run*,
    the run whose entire purpose is to produce an uncontaminated result.

    The exemption is deliberate and narrow: ``scripts/archive/`` may name them,
    because reproducing the negative result the paper reports requires serving
    the adapter that produced it.
    """

    #: Adapters the project's own evaluation rejected (DJ-058, DJ-124).
    _RETIRED_ADAPTERS = ("technical_v1", "technical_v2", "fundamental_v1")

    #: The ports those adapters were served on.
    _RETIRED_PORTS = ("1235", "1236")

    def _running_scripts(self) -> list[Path]:
        """Everything under scripts/ that is not archived."""
        return [p for p in (_REPO / "scripts").rglob("*")
                if p.suffix in {".py", ".sh"} and "archive" not in p.parts]

    @staticmethod
    def _code_lines(path: Path) -> list[str]:
        """Lines that execute. A comment explaining a retirement is not a load."""
        return [line for line in path.read_text().splitlines()
                if not line.lstrip().startswith(("#", "*"))]

    def test_no_running_script_serves_a_retired_adapter(self):
        offenders = []
        for p in self._running_scripts():
            for line in self._code_lines(p):
                for adapter in self._RETIRED_ADAPTERS:
                    if re.search(rf"--adapter-path\s+\S*{adapter}", line):
                        offenders.append(f"{p.relative_to(_REPO)} -> {adapter}")
        assert not offenders, (
            f"a running script loads a retired adapter: {offenders}. DJ-124 is "
            "how a rejected artifact keeps voting."
        )

    def test_no_running_script_routes_an_agent_to_the_retired_ports(self):
        offenders = []
        for p in self._running_scripts():
            for line in self._code_lines(p):
                for port in self._RETIRED_PORTS:
                    if f"localhost:{port}" in line or f"--port {port}" in line:
                        offenders.append(
                            f"{p.relative_to(_REPO)}: {line.strip()[:70]}")
        assert not offenders, (
            f"a running script still talks to the retired fine-tune ports: "
            f"{offenders}"
        )

    def test_the_live_agent_config_names_no_retired_adapter(self):
        from hifi.live.models import _AGENT_CONFIG, _HOMOGENEOUS_AGENT_CONFIG

        for config in (_AGENT_CONFIG, _HOMOGENEOUS_AGENT_CONFIG):
            for entry in config:
                model_id = entry[1]
                for adapter in self._RETIRED_ADAPTERS:
                    assert adapter not in model_id, (
                        f"agent {entry[0]} is configured on {model_id}, which "
                        f"names the retired {adapter}"
                    )

    def test_the_makefile_flags_the_reproduction_target_as_retired(self):
        """`make finetune-serve` still exists, for Phase 11 reproduction only.
        An operator reading `make help` must be told what it serves."""
        mk = (_REPO / "Makefile").read_text()
        line = next(x for x in mk.splitlines() if x.startswith("finetune-serve:"))
        assert "RETIRED" in line, (
            "finetune-serve advertises itself without saying the adapters were "
            "rejected; that is how one gets started by mistake"
        )


class TestTheSuiteDoesNotWriteIntoTheRealDataTree:
    """A test that writes into data/ corrupts the thing it is verifying.

    Found during DJ-136: hifi.data.refresh._register constructs
    DatasetRegistry() with its default path, which resolves to the repository's
    own data/registry.json. The refresh tests therefore wrote provenance rows
    whose file_path pointed into /tmp/pytest-of-... — entries describing files
    that no longer exist, in the artefact that backs the reproducibility claim.

    Redirecting it needed the patch at the USE site: DatasetRegistry binds its
    default path as a default argument at class-definition time, so patching
    versioning._DEFAULT_REGISTRY_PATH has no effect. That is the kind of thing
    a guard notices and a reader does not.
    """

    #: Files under data/ that a test run must never create or modify.
    _PROTECTED = ("registry.json",)

    def test_no_stray_registry_after_the_suite(self):
        for name in self._PROTECTED:
            path = _REPO / "data" / name
            if not path.exists():
                continue
            import json
            entries = json.loads(path.read_text())
            rows = entries if isinstance(entries, list) else entries.values()
            tmp_rows = [r for r in rows
                        if "pytest-of-" in str(r.get("file_path", ""))]
            assert not tmp_rows, (
                f"data/{name} contains {len(tmp_rows)} entries pointing at "
                "pytest temp directories; a test wrote into the real data tree"
            )
