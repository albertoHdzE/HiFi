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
