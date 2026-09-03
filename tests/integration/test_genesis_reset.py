"""The generation reset, at shell level.

``scripts/genesis_reset.sh`` is run once per generation, by hand, at the moment
the live record is at its most fragile: the arms have been reset at the broker,
the old rows are still on disk, and nothing downstream can tell a row written
before the reset from one written after. Its two guards — never overwrite an
archive, never clear without one — are the only thing standing between a
generation boundary and a silently continuous record.

A script that runs once per generation is a script that is never exercised, so
it is tested against a synthetic ``data/live`` tree rather than the real one. The
script resolves its root as ``dirname($0)/..``, which is what makes that
possible: copy it into ``tmp/scripts/`` and it operates on ``tmp/data/live/``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "genesis_reset.sh"

_ARMS = ("A", "B", "C", "D")


@pytest.fixture
def tree(tmp_path):
    """A miniature repo: the script, and a data/live tree it can act on."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(_SCRIPT, tmp_path / "scripts" / "genesis_reset.sh")

    live = tmp_path / "data" / "live"
    (live / "logs").mkdir(parents=True)
    (live / "genesis_date.txt").write_text("2026-08-24\n")

    for arm in _ARMS:
        d = live / arm
        d.mkdir()
        (d / "decisions.jsonl").write_text(
            json.dumps({"decision_date": "2026-08-25", "account": arm}) + "\n")
        (d / "equity.jsonl").write_text(
            json.dumps({"decision_date": "2026-08-25", "equity": 100_000.0}) + "\n")
        (d / "hwm.json").write_text(json.dumps({"hwm": 101_293.53}))
        (d / "book_state.json").write_text(json.dumps({"equity": 101_293.53}))
        (d / "dry_runs.jsonl").write_text(
            json.dumps({"decision_date": "2026-09-02", "dry_run": True}) + "\n")
        (d / "portfolio_history.json").write_text("{}")
        (d / "circuit_breakers.jsonl").write_text("{}\n")
        # Agent-signal evidence, which a reset must NOT touch.
        (d / "shadow_personality.jsonl").write_text(
            json.dumps({"date": "2026-09-02"}) + "\n")
        (d / "walkforward" / "2026-09-02").mkdir(parents=True)
        (d / "walkforward" / "2026-09-02" / "AAPL.json").write_text("{}")

    # One log from before the retiring generation, two from inside it.
    (live / "logs" / "nightly_20260820.log").write_text("previous generation\n")
    (live / "logs" / "nightly_20260825.log").write_text("this generation\n")
    (live / "logs" / "verify_20260902.log").write_text("this generation\n")

    return tmp_path


def run(tree, *args):
    return subprocess.run(
        ["bash", str(tree / "scripts" / "genesis_reset.sh"), *args],
        capture_output=True, text=True, cwd=tree, timeout=60,
    )


def _live(tree) -> Path:
    return tree / "data" / "live"


def _archive(tree, generation: int = 2) -> Path:
    return _live(tree) / f"_genesis{generation}_archive"


class TestArgumentHandling:
    def test_no_arguments_is_a_usage_error(self, tree):
        r = run(tree)
        assert r.returncode == 64
        assert "usage:" in r.stderr

    def test_a_mode_without_a_generation_is_refused(self, tree):
        r = run(tree, "--archive")
        assert r.returncode == 64
        assert "--generation" in r.stderr

    def test_a_non_numeric_generation_is_refused(self, tree):
        # `_genesis${GENERATION}_archive` would otherwise happily build a path
        # out of whatever was typed.
        r = run(tree, "--archive", "--generation", "two")
        assert r.returncode == 64

    def test_the_two_modes_are_mutually_exclusive(self, tree):
        r = run(tree, "--archive", "--clear", "--generation", "2")
        assert r.returncode == 64


class TestArchive:
    def test_it_copies_every_arm(self, tree):
        r = run(tree, "--archive", "--generation", "2")
        assert r.returncode == 0, r.stderr
        for arm in _ARMS:
            assert (_archive(tree) / arm / "decisions.jsonl").exists()
            assert (_archive(tree) / arm / "hwm.json").exists()

    def test_it_leaves_the_live_tree_untouched(self, tree):
        """Archive is a copy, not a move. Between --archive and --clear the
        accounts are reset at the broker; if the archive had moved the rows,
        that window would have no record at all."""
        run(tree, "--archive", "--generation", "2")
        for arm in _ARMS:
            assert (_live(tree) / arm / "decisions.jsonl").exists()

    def test_the_genesis_marker_travels_with_the_record_it_dates(self, tree):
        run(tree, "--archive", "--generation", "2")
        assert (_archive(tree) / "genesis_date.txt").read_text().strip() \
            == "2026-08-24"

    def test_it_archives_only_this_generations_logs(self, tree):
        """The cutoff is the genesis marker, so an archive holds the logs for
        the rows it holds — not the previous generation's as well."""
        run(tree, "--archive", "--generation", "2")
        names = {p.name for p in (_archive(tree) / "logs").iterdir()}
        assert names == {"nightly_20260825.log", "verify_20260902.log"}, names

    def test_it_archives_the_dj136_repair_backup_when_present(self, tree):
        backup = _live(tree) / "_dj136_backup"
        backup.mkdir()
        (backup / "A_decisions.jsonl").write_text("{}\n")
        run(tree, "--archive", "--generation", "2")
        assert (_archive(tree) / "_dj136_backup" / "A_decisions.jsonl").exists()

    def test_it_succeeds_when_there_is_no_repair_backup(self, tree):
        assert not (_live(tree) / "_dj136_backup").exists()
        assert run(tree, "--archive", "--generation", "2").returncode == 0

    def test_it_succeeds_when_there_is_no_genesis_marker(self, tree):
        (_live(tree) / "genesis_date.txt").unlink()
        r = run(tree, "--archive", "--generation", "2")
        assert r.returncode == 0, r.stderr
        assert "WARNING" in r.stderr

    def test_it_refuses_to_overwrite_an_existing_archive(self, tree):
        assert run(tree, "--archive", "--generation", "2").returncode == 0
        r = run(tree, "--archive", "--generation", "2")
        assert r.returncode == 65
        assert "never overwritten" in r.stderr

    def test_it_refuses_a_partial_archive(self, tree):
        """Better no archive than one missing an arm: --clear checks for the
        arm directories, and three of four would pass that check."""
        shutil.rmtree(_live(tree) / "D")
        r = run(tree, "--archive", "--generation", "2")
        assert r.returncode == 65
        assert "partial archive" in r.stderr


class TestClear:
    def _archived(self, tree):
        assert run(tree, "--archive", "--generation", "2").returncode == 0

    def test_it_refuses_without_an_archive(self, tree):
        r = run(tree, "--clear", "--generation", "2",
                "--genesis-date", "2026-09-03")
        assert r.returncode == 65
        assert "run --archive first" in r.stderr
        assert (_live(tree) / "A" / "decisions.jsonl").read_text().strip()

    def test_it_refuses_when_the_archive_is_missing_one_arm(self, tree):
        self._archived(tree)
        shutil.rmtree(_archive(tree) / "C")
        r = run(tree, "--clear", "--generation", "2",
                "--genesis-date", "2026-09-03")
        assert r.returncode == 65

    def test_it_requires_a_genesis_date(self, tree):
        self._archived(tree)
        r = run(tree, "--clear", "--generation", "2")
        assert r.returncode == 64
        assert "--genesis-date" in r.stderr

    @pytest.mark.parametrize("bad", ["2026-13-01", "03-09-2026", "tomorrow"])
    def test_it_refuses_a_malformed_genesis_date(self, tree, bad):
        self._archived(tree)
        r = run(tree, "--clear", "--generation", "2", "--genesis-date", bad)
        assert r.returncode == 65

    def test_it_refuses_a_marker_that_moves_backwards(self, tree):
        """A marker before the previous one makes 'days since genesis' negative
        and reclassifies the phase the agents are told they are in."""
        self._archived(tree)
        r = run(tree, "--clear", "--generation", "2",
                "--genesis-date", "2026-08-01")
        assert r.returncode == 65
        assert "not after" in r.stderr

    def test_it_removes_every_capital_linked_file(self, tree):
        self._archived(tree)
        assert run(tree, "--clear", "--generation", "2",
                   "--genesis-date", "2026-09-03").returncode == 0
        for arm in _ARMS:
            d = _live(tree) / arm
            assert not (d / "hwm.json").exists(), (
                "a high-water mark from the old capital would read the fresh "
                "account as already in drawdown (DJ-129b)")
            for name in ("book_state.json", "dry_runs.jsonl",
                         "portfolio_history.json", "circuit_breakers.jsonl"):
                assert not (d / name).exists(), f"{name} survived the reset"

    def test_it_leaves_decisions_and_equity_present_but_empty(self, tree):
        """Absent and empty are not the same to the readers: an empty file is a
        generation with no rows yet, a missing one is an arm that never ran."""
        self._archived(tree)
        run(tree, "--clear", "--generation", "2", "--genesis-date", "2026-09-03")
        for arm in _ARMS:
            for name in ("decisions.jsonl", "equity.jsonl"):
                p = _live(tree) / arm / name
                assert p.exists() and p.read_text() == ""

    def test_it_preserves_agent_signal_evidence(self, tree):
        """walkforward/ and shadow_personality.jsonl are what the ensemble said
        about a security on a date. Nothing about that is a function of the
        account balance, so a capital reset does not invalidate it."""
        self._archived(tree)
        run(tree, "--clear", "--generation", "2", "--genesis-date", "2026-09-03")
        for arm in _ARMS:
            d = _live(tree) / arm
            assert (d / "shadow_personality.jsonl").read_text().strip()
            assert (d / "walkforward" / "2026-09-02" / "AAPL.json").exists()

    def test_it_advances_the_genesis_marker(self, tree):
        """Nothing in the codebase writes this file; hifi.agents.context only
        reads it. Left at the old date it does not error — it tells the agents
        they are managing an established book on night one."""
        self._archived(tree)
        run(tree, "--clear", "--generation", "2", "--genesis-date", "2026-09-03")
        assert (_live(tree) / "genesis_date.txt").read_text().strip() == "2026-09-03"

    def test_the_advanced_marker_parses_as_the_code_reads_it(self, tree):
        from hifi.agents import context

        self._archived(tree)
        run(tree, "--clear", "--generation", "2", "--genesis-date", "2026-09-03")
        assert context.genesis_date(str(tree / "data")) == "2026-09-03"

    def test_the_archive_still_holds_the_record_afterwards(self, tree):
        self._archived(tree)
        run(tree, "--clear", "--generation", "2", "--genesis-date", "2026-09-03")
        for arm in _ARMS:
            assert (_archive(tree) / arm / "decisions.jsonl").read_text().strip()
            assert (_archive(tree) / arm / "hwm.json").exists()


class TestItCoversEveryFileTheCycleWrites:
    """The old genesis2_reset.sh cleared five files. Three more have appeared
    since — book_state.json (DJ-130), dry_runs.jsonl (DJ-136) and the genesis
    marker — and nothing connected the reset to the code that writes them, so
    each was added to the cycle without being added here.

    This test fails when the next one appears.
    """

    def test_every_per_arm_path_in_hifi_live_paths_is_handled(self):
        from hifi.live import paths

        script = _SCRIPT.read_text()
        handled_or_kept = set()
        for fn in (paths._decisions_log, paths._breaker_log,
                   paths._dry_run_log, paths._hwm_path):
            handled_or_kept.add(fn("A").name)

        missing = [n for n in handled_or_kept if n not in script]
        assert not missing, (
            f"hifi.live.paths writes {missing} per arm and genesis_reset.sh "
            "neither clears nor deliberately keeps them; the next generation "
            "would inherit them"
        )

    def test_the_files_the_live_tree_actually_holds_are_all_accounted_for(self):
        """Against the real data/live, not a fixture: a file that exists on disk
        and appears nowhere in the script is one nobody has decided about."""
        live = _REPO / "data" / "live"
        if not (live / "A").exists():
            pytest.skip("no live tree in this checkout")
        script = _SCRIPT.read_text()
        unaccounted = sorted({
            p.name
            for arm in _ARMS if (live / arm).is_dir()
            for p in (live / arm).iterdir()
            if p.name not in script
        })
        assert not unaccounted, (
            f"data/live/<ARM>/ holds {unaccounted}, which genesis_reset.sh does "
            "not mention in either its cleared or its kept list"
        )
