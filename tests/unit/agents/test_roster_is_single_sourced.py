"""DJ-135: the ensemble roster must have exactly one definition.

It had six. Four modules listed six agents and two listed five — the difference
was correct (the two were counting voters, and the contrarian does not vote) but
indistinguishable from a copy that had drifted. The failure mode is silent:
an analytics module iterating five agent names over six sidecars reports on five
and says nothing about the sixth.

These tests fail if anyone reintroduces a literal.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from hifi.agents.roster import CANONICAL_ORDER, NON_VOTING, VOTING_AGENTS

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "scripts"))


class TestRosterContent:
    def test_canonical_order_is_the_causal_execution_order(self):
        assert CANONICAL_ORDER == [
            "fundamental", "technical", "risk", "macro", "sentiment", "contrarian",
        ], "each agent sees the summaries of those before it; the order is causal"

    def test_contrarian_is_last_and_does_not_vote(self):
        assert CANONICAL_ORDER[-1] == NON_VOTING == "contrarian"
        assert NON_VOTING not in VOTING_AGENTS, (
            "the contrarian reviews the others; counting its pass as a vote would "
            "bias herding, disagreement entropy and unanimity mass"
        )

    def test_voting_agents_is_canonical_order_minus_the_reviewer(self):
        assert VOTING_AGENTS == [a for a in CANONICAL_ORDER if a != NON_VOTING]
        assert len(VOTING_AGENTS) == 5

    def test_roster_has_no_duplicates(self):
        assert len(set(CANONICAL_ORDER)) == len(CANONICAL_ORDER)


class TestEveryConsumerAgrees:
    """Each site that used to hold its own literal now reports the roster."""

    def test_agent_context_reexports_canonical_order(self):
        from hifi.knowledge import agent_context

        assert agent_context.CANONICAL_ORDER is CANONICAL_ORDER

    def test_ensemble_runner_uses_all_six(self):
        from hifi.agents import ensemble_runner

        assert ensemble_runner._ALL_AGENTS is CANONICAL_ORDER

    def test_decision_audit_reports_only_voters(self):
        from hifi.analytics import decision_audit

        assert decision_audit.AGENTS is VOTING_AGENTS

    def test_orchestrator_runs_all_six_passes(self):
        import run_phase15_orchestrator as orch

        assert orch.CANONICAL_ORDER is CANONICAL_ORDER

    def test_walkforward_scores_only_voters(self):
        import run_phase15_walkforward as wf

        assert wf._AGENT_TYPES is VOTING_AGENTS

    def test_verifier_gates_only_voters(self):
        import verify_agent_repair as ver

        assert ver.VOTING_AGENTS is VOTING_AGENTS

    def test_orchestrator_config_covers_every_agent_in_the_roster(self):
        import run_phase15_orchestrator as orch

        for cfg_name in ("_AGENT_CONFIG", "_HOMOGENEOUS_AGENT_CONFIG"):
            configured = [row[0] for row in getattr(orch, cfg_name)]
            assert configured == CANONICAL_ORDER, (
                f"{cfg_name} does not match the roster; an agent in the roster "
                "with no model config would be skipped without comment"
            )


class TestNoLiteralsSurvive:
    """An AST-level guard: the roster must not be spelled out in code again.

    Parsing rather than grepping matters here. Prose naming the agents is fine
    and common ("agent_type: one of fundamental, technical, ..."), so a textual
    search flags docstrings and trains the reader to ignore it. Only an actual
    list or tuple *display* of string literals is a copy of the roster.
    """

    _NAMES = frozenset(CANONICAL_ORDER)

    def _roster_literals(self, path: Path) -> list[int]:
        tree = ast.parse(path.read_text())
        hits = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.List | ast.Tuple):
                continue
            elts = [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            # Three or more agent names in one display is a roster copy, not a
            # coincidence; two allows legitimate pairs such as ("risk", "macro").
            if len(self._NAMES.intersection(elts)) >= 3:
                hits.append(node.lineno)
        return hits

    @pytest.mark.parametrize("subdir", ["src", "scripts"])
    def test_no_module_spells_out_the_roster(self, subdir):
        offenders = []
        for path in sorted((_ROOT / subdir).rglob("*.py")):
            if path.name == "roster.py" or "archive" in path.parts:
                continue
            offenders += [f"{path.relative_to(_ROOT)}:{ln}"
                          for ln in self._roster_literals(path)]
        assert not offenders, (
            "the roster is spelled out as a literal here; import CANONICAL_ORDER "
            f"or VOTING_AGENTS from hifi.agents.roster instead: {offenders}"
        )

    def test_the_guard_itself_detects_a_planted_copy(self, tmp_path):
        # A guard that cannot fail is not a guard.
        planted = tmp_path / "planted.py"
        planted.write_text('AGENTS = ["fundamental", "technical", "risk"]\n')
        assert self._roster_literals(planted) == [1]

    def test_the_guard_ignores_prose(self, tmp_path):
        doc = tmp_path / "doc.py"
        doc.write_text('"""One of: fundamental, technical, risk, macro."""\n')
        assert self._roster_literals(doc) == []
