"""The nightly wrapper's market-hours guard, at shell level.

``scripts/nightly_live_execute.sh`` enforces the experiment's timing protocol —
decide on completed closes, fill at the next open — and nothing tested it. The
guard is the difference between a decision made on a settled bar and one made on
a live partial, which is a scientific property, not an operational nicety.

``--check-window`` is pure: it prints a verdict and exits 0 or 1 without running
anything. That makes it testable by shimming ``date`` onto PATH, which is what
these tests do. The shim answers both forms the script uses:
``TZ=America/New_York date +%u`` and ``date +%H%M``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "nightly_live_execute.sh"


@pytest.fixture
def fake_clock(tmp_path):
    """Return a factory that pins the script's view of ET day-of-week and time."""
    def _at(dow: int, hhmm: str, env: dict | None = None):
        shim = tmp_path / "bin"
        shim.mkdir(exist_ok=True)
        (shim / "date").write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            f'  +%u) echo {dow} ;;\n'
            f'  +%H%M) echo {hhmm} ;;\n'
            '  *) /bin/date "$@" ;;\n'
            "esac\n"
        )
        (shim / "date").chmod(0o755)
        environ = {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}"}
        environ.update(env or {})
        return subprocess.run(
            ["bash", str(_SCRIPT), "--check-window"],
            capture_output=True, text=True, env=environ, cwd=_REPO, timeout=60,
        )
    return _at


class TestTheCashSessionIsRefused:
    """Inside 09:30-16:00 ET the last OHLCV bar is a live partial."""

    @pytest.mark.parametrize("hhmm", ["0930", "1200", "1559"])
    def test_refuses_during_the_session(self, fake_clock, hhmm):
        r = fake_clock(dow=3, hhmm=hhmm)
        assert r.returncode == 1, f"a run at {hhmm} ET was allowed to start"
        assert "REFUSING" in r.stdout
        assert "partial" in r.stdout

    def test_the_refusal_says_how_to_proceed(self, fake_clock):
        r = fake_clock(dow=3, hhmm="1200")
        assert "16:00 ET" in r.stdout
        assert "ALLOW_MARKET_HOURS=1" in r.stdout, (
            "a refusal with no stated override is an operator dead end"
        )

    def test_check_window_reports_the_verdict_regardless_of_the_override(
            self, fake_clock):
        """--check-window is a predicate, not a policy.

        ALLOW_MARKET_HOURS does not change the verdict here; it changes what the
        *caller* does with it. Keeping the two separate means the raw fact — the
        market is open — is always available to a log or a human, even on a run
        that overrides it.
        """
        r = fake_clock(dow=3, hhmm="1200", env={"ALLOW_MARKET_HOURS": "1"})
        assert r.returncode == 1
        assert "REFUSING" in r.stdout

    def test_the_full_run_honours_the_override(self, fake_clock):
        # Same clock, without --check-window, would proceed. Asserted on the
        # source because running it would place orders.
        src = _SCRIPT.read_text()
        assert 'if [ "${ALLOW_MARKET_HOURS}" = "1" ]; then' in src
        assert "annotate this date as off-protocol" in src, (
            "an overridden run must mark itself as a protocol deviation"
        )

    def test_the_makefile_applies_the_override_at_the_call_site(self):
        mk = (_REPO / "Makefile").read_text()
        target = mk[mk.index("live-nightly:"):]
        assert "--check-window" in target
        assert 'ALLOW_MARKET_HOURS' in target, (
            "make live-nightly reads the verdict but offers no override, so an "
            "operator has no supported way to run off-protocol deliberately"
        )

    @pytest.mark.parametrize("hhmm", ["0929", "1600", "1601"])
    def test_the_boundaries_are_exclusive_at_the_close(self, fake_clock, hhmm):
        # 09:30 is in-session, 16:00 is not: the close bar is settled at 16:00.
        assert fake_clock(dow=3, hhmm=hhmm).returncode == 0


class TestEveningRunsAreTheProtocol:
    @pytest.mark.parametrize("hhmm", ["1700", "1900", "2330"])
    def test_evenings_proceed_without_comment(self, fake_clock, hhmm):
        r = fake_clock(dow=3, hhmm=hhmm)
        assert r.returncode == 0
        assert "WARNING" not in r.stdout and "REFUSING" not in r.stdout


class TestPreMarketWarnsButProceeds:
    """The inputs are still complete closes, which is the part that matters."""

    @pytest.mark.parametrize("hhmm", ["0400", "0800"])
    def test_a_run_that_would_finish_after_the_open_warns(self, fake_clock, hhmm):
        r = fake_clock(dow=3, hhmm=hhmm)
        assert r.returncode == 0, "a pre-market start was blocked; it should warn"
        assert "WARNING" in r.stdout
        assert "fill intraday" in r.stdout

    def test_a_very_early_run_that_finishes_before_the_open_is_silent(self, fake_clock):
        # 00:30 + 6 h = 06:30, comfortably before 09:30.
        r = fake_clock(dow=3, hhmm="0030")
        assert r.returncode == 0
        assert "WARNING" not in r.stdout


class TestWeekendsAreAllowed:
    """DJ-121. The risk was never the day of the week — it was deciding twice on
    the same information, which the session-date resolution now handles."""

    @pytest.mark.parametrize("dow", [6, 7])
    def test_weekend_runs_proceed(self, fake_clock, dow):
        r = fake_clock(dow=dow, hhmm="1200")
        assert r.returncode == 0, "a weekend run was refused (pre-DJ-121 behaviour)"
        assert "Weekend" in r.stdout

    def test_the_weekend_message_explains_the_dating(self, fake_clock):
        r = fake_clock(dow=6, hhmm="1200")
        assert "last completed" in r.stdout
        assert "next open" in r.stdout

    def test_a_weekend_midday_is_not_treated_as_a_cash_session(self, fake_clock):
        # The weekend branch must be checked BEFORE the session-hours branch.
        r = fake_clock(dow=7, hhmm="1030")
        assert r.returncode == 0
        assert "REFUSING" not in r.stdout


class TestCheckWindowRunsNothing:
    def test_it_exits_without_touching_lm_studio_or_the_broker(self, fake_clock):
        r = fake_clock(dow=3, hhmm="1900")
        assert "waiting for LM Studio" not in r.stdout
        assert "nightly_live_execute" not in r.stdout, (
            "--check-window fell through into the real run"
        )

    def test_it_writes_no_log_file(self, fake_clock):
        before = set((_REPO / "data" / "live" / "logs").glob("*")) \
            if (_REPO / "data" / "live" / "logs").exists() else set()
        fake_clock(dow=3, hhmm="1900")
        after = set((_REPO / "data" / "live" / "logs").glob("*")) \
            if (_REPO / "data" / "live" / "logs").exists() else set()
        assert after == before


class TestArgumentHandling:
    def _run(self, *args, env=None):
        return subprocess.run(["bash", str(_SCRIPT), *args], capture_output=True,
                              text=True, cwd=_REPO, timeout=60,
                              env={**os.environ, **(env or {})})

    def test_an_unknown_flag_is_rejected(self):
        r = self._run("--check-window", "--nonsense")
        assert r.returncode == 64, "an unknown flag was silently ignored"
        assert "unknown argument" in r.stderr


class TestDryAndRealPathsDifferOnlyByTheFlag:
    """`make live-nightly DRY=1` is only trustworthy if this holds.

    The whole point of DRY=1 is that a verification run is the production path
    minus one flag. If the two branches drifted, a passing dry run would stop
    being evidence about the real one.
    """

    def _invocations(self) -> list[str]:
        src = _SCRIPT.read_text()
        return [ln.strip() for ln in src.splitlines()
                if "hifi_live.py" in ln and not ln.strip().startswith("#")]

    def test_there_are_exactly_two_invocations(self):
        assert len(self._invocations()) == 2, (
            f"expected one dry and one real invocation, found: {self._invocations()}"
        )

    def test_they_differ_only_by_execute(self):
        dry, real = sorted(self._invocations(), key=len)
        assert real.replace(" --execute", "") == dry, (
            f"the dry and real paths have diverged:\n  dry:  {dry}\n  real: {real}"
        )

    def test_both_run_every_account(self):
        for line in self._invocations():
            assert "--account all" in line

    def test_the_dry_path_writes_to_a_separate_log(self):
        src = _SCRIPT.read_text()
        assert "VERIFY_LOG" in src, (
            "a dry run appending to the nightly log makes the two "
            "indistinguishable after the fact"
        )


class TestPreflightIsPresent:
    """Running hifi_live.py directly skips these; on 2026-08-31 that produced a
    full cycle with no telemetry at all."""

    def test_it_waits_for_lm_studio(self):
        assert "localhost:1234/v1/models" in _SCRIPT.read_text()

    def test_it_starts_langfuse_rather_than_only_checking_it(self):
        src = _SCRIPT.read_text()
        assert "docker compose" in src and "up -d" in src, (
            "the wrapper checks LangFuse but does not start it; nights when it "
            "was down silently lost prompts, tokens and latency"
        )

    def test_langfuse_failure_is_fail_open(self):
        src = _SCRIPT.read_text()
        assert "run proceeds" in src, (
            "telemetry is not a precondition for trading; sidecars are the "
            "durable record"
        )
