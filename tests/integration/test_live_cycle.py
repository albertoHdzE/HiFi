"""A whole nightly cycle, end to end, against a mock broker.

Nothing tested ``run_account_cycle`` or ``run_batch`` before this — the 379-line
spine of the experiment, and the only place where the guards, the pipeline, the
strategies and the record actually meet. The unit tests around them patch the
very functions whose interaction is the risk: DJ-136 sat in plain sight because
every test of ``log_episode`` called it directly, and none ran the cycle that
decides whether to call it at all.

No LLM and no broker. Ensemble signals come from sidecar fixtures written to
``tmp_path``, which is what the live path reads anyway
(``ensemble.load_ensemble_signals``), so the pipeline, the allocator and the
order loop run for real.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hifi.execution.broker import Position
from hifi.live import accounts, cycle, guards, market, paths

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture
def live_root(tmp_path, monkeypatch):
    """Redirect the whole live tree into tmp_path.

    One patch point suffices because every module reaches these through the
    ``paths`` module rather than importing them by value (DJ-135). If that ever
    regresses, these tests write into the real data/live and the assertions on
    absence start passing for the wrong reason — which is why
    test_no_test_writes_into_the_real_data_dir exists below.
    """
    monkeypatch.setattr(paths, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(paths, "_OUTPUT_DIR", str(tmp_path / "live"))
    return tmp_path


def _executor(equity=100_000.0, cash=100_000.0, positions=None, last_equity=None):
    ex = MagicMock()
    ex.get_portfolio_value.return_value = equity
    ex.get_account_cash.return_value = cash
    ex.get_positions.return_value = positions or {}
    ex.get_client_order_ids.return_value = set()
    ex.is_fractionable.return_value = True
    ex.place_market_order.return_value = MagicMock(
        status="accepted", order_id="oid", filled_avg_price=None)
    acct = MagicMock()
    acct.equity = str(equity)
    acct.cash = str(cash)
    acct.last_equity = str(last_equity if last_equity is not None else equity)
    ex.client.get_account.return_value = acct
    ex.client.get_asset.return_value = MagicMock(tradable=True)
    return ex


def _write_sidecar_ensemble(root: Path, account: str, date: str, condition: str,
                            decisions: dict[str, str]) -> None:
    """Write the ensemble JSONs the live path loads signals from."""
    year, month, _ = date.split("-")
    out = root / "live" / account / "walkforward" / date / condition / year / month
    out.mkdir(parents=True, exist_ok=True)
    for ticker, decision in decisions.items():
        (out / f"{ticker}.json").write_text(json.dumps({
            "ticker": ticker,
            "ensemble_decision": {
                "collective_decision": decision,
                "collective_confidence": 0.8,
            },
        }))


def _decisions(root: Path, account: str) -> list[dict]:
    p = root / "live" / account / "decisions.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _dry_runs(root: Path, account: str) -> list[dict]:
    p = root / "live" / account / "dry_runs.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


DATE = "2026-08-31"
TICKERS = ["AAPL", "MSFT", "JPM"]


@pytest.fixture
def prices(monkeypatch):
    monkeypatch.setattr(market, "_latest_prices",
                        lambda t: {"AAPL": 200.0, "MSFT": 400.0, "JPM": 150.0})


# ---------------------------------------------------------------------------
# DJ-136: a dry run must not touch the experimental record
# ---------------------------------------------------------------------------


class TestDryRunsDoNotContaminateTheRecord:
    """The defect: verification runs wrote rows indistinguishable from real ones.

    Arm C accumulated three rows for 2026-09-01 — one per `make live-plan` — and
    ``already_decided`` then reported that the arm had already traded, so the
    next real cycle would have skipped the null model that every other arm is
    measured against.
    """

    @pytest.mark.parametrize("account,condition", [
        ("A", "parallel"), ("B", "full"), ("C", "control"), ("D", "riskbudget"),
    ])
    def test_no_execute_writes_no_decision_row(self, live_root, prices,
                                               account, condition, monkeypatch):
        _write_sidecar_ensemble(live_root, account, DATE, condition,
                                {"AAPL": "Buy", "MSFT": "Hold", "JPM": "Sell"})
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor())
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        # --no-execute genuinely runs the agents (that is the point of DRY=1);
        # stub the sweep so this test stays offline, and read the sidecars the
        # fixture already wrote.
        # record_account is patched so that THIS test fails on its own
        # assertion. Unpatched, the pre-fix code reaches it and dies serialising
        # a MagicMock — a real failure, but one that reports the wrong cause.
        # test_dry_run_does_not_snapshot_equity covers that call directly.
        with patch("hifi.live.ensemble.run_ensemble"), \
             patch("hifi.execution.portfolio_recorder.record_account"), \
             patch("hifi.execution.riskbudget_strategy.get_riskbudget_signals",
                   return_value={"signals": [
                       {"ticker": "AAPL", "decision": "Buy", "confidence": 0.8}],
                       "strategy": "calm", "strategy_version": "1",
                       "call_id": "x", "skipped": []}):
            cycle.run_account_cycle(account, TICKERS, DATE,
                                    dry_run=False, execute=False)

        assert _decisions(live_root, account) == [], (
            f"arm {account} wrote to decisions.jsonl during a --no-execute run; "
            "already_decided would then skip the next real cycle for this date"
        )

    def test_the_dry_row_is_kept_somewhere(self, live_root, prices, monkeypatch):
        # Not silence: knowing a verification run happened is worth keeping.
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor())
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        with patch("hifi.execution.portfolio_recorder.record_account"):
            cycle.run_account_cycle("C", TICKERS, DATE, dry_run=False, execute=False)

        rows = _dry_runs(live_root, "C")
        assert len(rows) == 1
        assert rows[0]["dry_run"] is True
        assert rows[0]["decision_date"] == DATE

    def test_repeated_dry_runs_do_not_block_a_real_cycle(self, live_root, prices,
                                                         monkeypatch):
        # The literal 2026-09-01 arm-C failure: three `make live-plan` runs.
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor())
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        with patch("hifi.execution.portfolio_recorder.record_account"):
            for _ in range(3):
                cycle.run_account_cycle("C", TICKERS, DATE, dry_run=True, execute=False)

        assert _decisions(live_root, "C") == []
        assert accounts.already_decided("C", DATE) is False, (
            "three dry runs marked the date decided; the real cycle would skip "
            "the control arm"
        )

    def test_dry_run_does_not_snapshot_equity(self, live_root, prices, monkeypatch):
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor())
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        with patch("hifi.execution.portfolio_recorder.record_account") as rec:
            cycle.run_account_cycle("C", TICKERS, DATE, dry_run=False, execute=False)
        rec.assert_not_called()

    def test_a_real_cycle_still_writes_exactly_one_row(self, live_root, prices,
                                                      monkeypatch):
        # The guard must not have turned into "never record".
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor())
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        with patch("hifi.execution.portfolio_recorder.record_account") as rec:
            cycle.run_account_cycle("C", TICKERS, DATE, dry_run=False, execute=True)

        rows = _decisions(live_root, "C")
        assert len(rows) == 1
        assert "dry_run" not in rows[0]
        rec.assert_called_once()

    def test_a_real_cycle_is_not_repeated(self, live_root, prices, monkeypatch):
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor())
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        with patch("hifi.execution.portfolio_recorder.record_account"):
            for _ in range(2):
                cycle.run_account_cycle("C", TICKERS, DATE,
                                        dry_run=False, execute=True)
        assert len(_decisions(live_root, "C")) == 1, "one decision per arm per day"


# ---------------------------------------------------------------------------
# The cycle itself, per condition
# ---------------------------------------------------------------------------


class TestControlArm:
    def test_buys_the_universe_once_then_holds(self, live_root, prices, monkeypatch):
        ex = _executor()
        monkeypatch.setattr(accounts, "get_executor", lambda a: ex)
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        with patch("hifi.execution.portfolio_recorder.record_account"):
            cycle.run_account_cycle("C", TICKERS, DATE, dry_run=False, execute=True)

        rows = _decisions(live_root, "C")
        assert rows[0]["n_orders"] == len(TICKERS)
        assert rows[0]["signals"] == [], "the null model has no signal layer"

        # Second day, already holding: no further buying.
        held = {t: Position(ticker=t, qty=10, market_value=1000.0,
                            avg_entry_price=100.0, unrealized_pnl=0.0,
                            side="long")
                for t in TICKERS}
        ex2 = _executor(positions=held)
        monkeypatch.setattr(accounts, "get_executor", lambda a: ex2)
        with patch("hifi.execution.portfolio_recorder.record_account"):
            cycle.run_account_cycle("C", TICKERS, "2026-09-01",
                                    dry_run=False, execute=True)
        assert _decisions(live_root, "C")[1]["n_orders"] == 0


class TestEnsembleArms:
    @pytest.mark.parametrize("account,condition", [("A", "parallel"), ("B", "full")])
    def test_sidecar_signals_reach_orders(self, live_root, prices, account,
                                          condition, monkeypatch):
        _write_sidecar_ensemble(live_root, account, DATE, condition,
                                {"AAPL": "Buy", "MSFT": "Buy", "JPM": "Hold"})
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor())
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        with patch("hifi.live.ensemble.run_ensemble"), \
             patch("hifi.execution.portfolio_recorder.record_account"):
            cycle.run_account_cycle(account, TICKERS, DATE,
                                    dry_run=False, execute=True)

        row = _decisions(live_root, account)[0]
        assert row["condition"] == condition
        assert len(row["signals"]) == 3
        buys = {s["ticker"] for s in row["signals"] if s["decision"] == "Buy"}
        assert buys == {"AAPL", "MSFT"}
        assert row["n_orders"] > 0, "two Buys at 0.8 confidence produced no order"

    def test_missing_sidecars_skip_the_pipeline_without_a_record(
            self, live_root, prices, monkeypatch):
        # No ensemble was produced: that is a failed night, not a decision.
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor())
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        with patch("hifi.live.ensemble.run_ensemble"):
            cycle.run_account_cycle("A", TICKERS, DATE, dry_run=False, execute=True)
        assert _decisions(live_root, "A") == []


class TestRiskbudgetArm:
    def test_provider_signals_carry_attribution(self, live_root, prices, monkeypatch):
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor())
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        payload = {"signals": [{"ticker": "AAPL", "decision": "Buy", "confidence": 0.9}],
                   "strategy": "calm_exposure", "strategy_version": "2.1",
                   "call_id": "abc123", "skipped": ["XYZ"]}
        with patch("hifi.execution.riskbudget_strategy.get_riskbudget_signals",
                   return_value=payload), \
             patch("hifi.execution.portfolio_recorder.record_account"):
            cycle.run_account_cycle("D", TICKERS, DATE, dry_run=False, execute=True)

        meta = _decisions(live_root, "D")[0]["strategy_meta"]
        assert meta["provider"] == "riskbudget"
        assert meta["strategy_version"] == "2.1", (
            "which version produced which orders is required for traceability "
            "(DJ-113)"
        )
        assert meta["skipped"] == ["XYZ"]


# ---------------------------------------------------------------------------
# Interactions that only appear in the whole cycle
# ---------------------------------------------------------------------------


class TestGuardsInTheCycle:
    def test_tripped_breaker_suppresses_orders_but_still_records_equity(
            self, live_root, prices, monkeypatch):
        """DJ-119: a halt suppresses ORDERS, not OBSERVATION.

        The return used to precede record_account, so a halted arm silently
        stopped capturing its equity curve — arm C froze at 2026-07-17 while the
        others ran to 07-27. A gap in the benchmark's curve is worse than the
        halt that caused it.
        """
        ex = _executor()
        monkeypatch.setattr(accounts, "get_executor", lambda a: ex)
        with patch.object(guards, "check_circuit_breakers", return_value=True), \
             patch("hifi.execution.portfolio_recorder.record_account") as rec:
            cycle.run_account_cycle("C", TICKERS, DATE, dry_run=False, execute=True)

        ex.place_market_order.assert_not_called()
        rec.assert_called_once()
        assert _decisions(live_root, "C") == []

    def test_halt_between_signals_and_submission_stops_the_arm(
            self, live_root, prices, monkeypatch):
        """DJ-129c: hours of inference separate the first check from the first order."""
        ex = _executor()
        monkeypatch.setattr(accounts, "get_executor", lambda a: ex)
        with patch.object(guards, "check_circuit_breakers", side_effect=[False, True]), \
             patch("hifi.execution.portfolio_recorder.record_account"):
            cycle.run_account_cycle("C", TICKERS, DATE, dry_run=False, execute=True)

        ex.place_market_order.assert_not_called()
        assert _decisions(live_root, "C") == []

    def test_hwm_ratchets_before_trading_and_never_falls(self, live_root, prices,
                                                         monkeypatch):
        """DJ-129b: the drawdown breaker needs a real baseline, not today's equity."""
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor(equity=110_000.0))
        with patch("hifi.execution.portfolio_recorder.record_account"):
            cycle.run_account_cycle("C", TICKERS, DATE, dry_run=False, execute=True)
        peak = json.loads((live_root / "live" / "C" / "hwm.json").read_text())["hwm"]
        assert peak == pytest.approx(110_000.0)

        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor(equity=90_000.0))
        with patch("hifi.execution.portfolio_recorder.record_account"):
            cycle.run_account_cycle("C", TICKERS, "2026-09-01",
                                    dry_run=False, execute=True)
        after = json.loads((live_root / "live" / "C" / "hwm.json").read_text())
        assert after["hwm"] == pytest.approx(110_000.0), "a falling market lowered the mark"
        assert after["equity_now"] == pytest.approx(90_000.0)

    def test_a_rejected_order_is_recorded_not_raised(self, live_root, prices,
                                                     monkeypatch):
        """DJ-123: one delisted symbol must not take down an arm mid-execution."""
        ex = _executor()
        ex.place_market_order.side_effect = [
            MagicMock(status="accepted", order_id="1", filled_avg_price=None),
            RuntimeError("asset not found: MSFT"),
            MagicMock(status="accepted", order_id="3", filled_avg_price=None),
        ]
        monkeypatch.setattr(accounts, "get_executor", lambda a: ex)
        monkeypatch.setattr(accounts, "show_status", lambda a, e: None)
        with patch("hifi.execution.portfolio_recorder.record_account"):
            cycle.run_account_cycle("C", TICKERS, DATE, dry_run=False, execute=True)

        statuses = [o["status"] for o in _decisions(live_root, "C")[0]["orders"]]
        assert "rejected" in statuses
        assert statuses.count("accepted") == 2, "a rejection stopped the other orders"


class TestBatchIsolation:
    def test_one_failing_arm_does_not_abort_the_others(self, live_root, prices,
                                                       monkeypatch):
        """DJ-117: a network failure on one arm must not cost the whole night."""
        def _boom(account, *a, **k):
            if account == "B":
                raise RuntimeError("broker unreachable")

        monkeypatch.setattr(cycle, "run_account_cycle", _boom)
        monkeypatch.setattr(market, "update_data", lambda t: {})
        monkeypatch.setattr(market, "_last_completed_session", lambda t: DATE)
        monkeypatch.setattr(guards, "check_data_coverage", lambda t: True)
        monkeypatch.setattr(guards, "check_tradability", lambda t, a: [])
        monkeypatch.setattr(guards, "log_arm_invariance", lambda a: None)

        failed = cycle.run_batch(TICKERS, DATE, ["A", "B", "C", "D"],
                                 dry_run=False, execute=True)
        assert failed == ["B"]

    def test_a_starved_universe_aborts_before_writing_anything(
            self, live_root, monkeypatch):
        """DJ-120: agents render missing data as conviction, so the gate blocks."""
        monkeypatch.setattr(market, "update_data", lambda t: {})
        monkeypatch.setattr(market, "_last_completed_session", lambda t: DATE)
        monkeypatch.setattr(guards, "check_data_coverage", lambda t: False)
        with pytest.raises(SystemExit) as exc:
            cycle.run_batch(TICKERS, DATE, ["A"], dry_run=False, execute=True)
        assert exc.value.code == 2
        assert _decisions(live_root, "A") == []


class TestHarnessIsolation:
    def test_no_test_writes_into_the_real_data_dir(self, live_root):
        """If paths redirection regresses, the assertions above pass vacuously.

        Every assertion in this file is of the form "no row was written". Those
        are satisfied trivially if the writes are landing in the real
        data/live/ instead of tmp_path — so the redirection itself is asserted.
        """
        assert str(live_root) == paths._DATA_DIR
        assert paths._account_dir("C").is_relative_to(live_root)
        assert paths._decisions_log("C").is_relative_to(live_root)
        assert paths._dry_run_log("C").is_relative_to(live_root)
