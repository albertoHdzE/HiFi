"""The one JSON extractor every agent uses (DJ-140).

Seven copies of this function existed, functionally identical, differing only in
their docstrings — the shape DJ-135 removed from the ensemble roster. A parser
that is copied drifts, and two agents disagreeing about what counts as a
parseable response is a confound in a study whose dependent variable is agent
disagreement.

The behavioural change made while consolidating: valid JSON that is not an
object now returns None rather than being handed to a caller that will
immediately call ``.get`` on it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hifi.agents.json_parsing import extract_json

_REPO = Path(__file__).resolve().parents[3]

#: Every module that used to carry its own copy.
_FORMER_OWNERS = [
    "src/hifi/agents/fundamental_agent.py",
    "src/hifi/agents/technical_agent.py",
    "src/hifi/agents/risk_agent.py",
    "src/hifi/agents/macro_agent.py",
    "src/hifi/agents/sentiment_agent.py",
    "src/hifi/agents/contrarian_agent.py",
    "src/hifi/collective/debate_nodes.py",
]


class TestItParsesWhatModelsActuallyEmit:
    def test_a_bare_object(self):
        assert extract_json('{"decision": "Buy"}') == {"decision": "Buy"}

    def test_leading_and_trailing_whitespace(self):
        assert extract_json('\n\n  {"decision": "Hold"}  \n') == {"decision": "Hold"}

    @pytest.mark.parametrize("fence", ["```json", "```", "```JSON"])
    def test_markdown_code_fences_are_stripped(self, fence):
        text = f'{fence}\n{{"decision": "Sell", "confidence": 0.8}}\n```'
        assert extract_json(text) == {"decision": "Sell", "confidence": 0.8}

    def test_prose_around_the_object(self):
        text = 'Here is my analysis:\n{"decision": "Buy"}\nHope that helps.'
        assert extract_json(text) == {"decision": "Buy"}

    def test_the_widest_span_is_taken_so_nested_objects_survive(self):
        text = 'note {"decision": "Buy", "evidence": {"pe": 12}} end'
        assert extract_json(text) == {"decision": "Buy", "evidence": {"pe": 12}}


class TestItReturnsNoneRatherThanSomethingUnusable:
    """Every caller does ``parsed.get("decision")`` immediately."""

    @pytest.mark.parametrize("text", ["", "   ", "I cannot answer that.",
                                      "{not json at all}", "{"])
    def test_unparseable_text(self, text):
        assert extract_json(text) is None

    @pytest.mark.parametrize("text", ['["Buy", "Sell"]', "42", '"Buy"',
                                      "true", "null"])
    def test_valid_json_that_is_not_an_object(self, text):
        """A list, a number and a bare string are all valid JSON and none of
        them is an answer. Before consolidation these were returned as-is and
        the caller raised AttributeError inside its own try/except, recording a
        parse failure that named neither the cause nor the text."""
        assert extract_json(text) is None

    def test_a_fenced_list_is_also_none(self):
        assert extract_json('```json\n["Buy"]\n```') is None

    def test_an_empty_object_is_an_object(self):
        # Distinct from None: the model answered, the answer was empty. The
        # caller's .get() returns None per field, which is the right outcome.
        assert extract_json("{}") == {}


class TestThereIsExactlyOneDefinition:
    @pytest.mark.parametrize("path", _FORMER_OWNERS)
    def test_no_module_redefines_the_extractor(self, path):
        tree = ast.parse((_REPO / path).read_text())
        defs = [n.name for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
                and "extract_json" in n.name]
        assert not defs, (
            f"{path} defines {defs} again; import it from "
            "hifi.agents.json_parsing instead"
        )

    @pytest.mark.parametrize("path", _FORMER_OWNERS)
    def test_each_former_owner_imports_the_shared_one(self, path):
        assert "from hifi.agents.json_parsing import extract_json" in \
            (_REPO / path).read_text()

    def test_the_alias_the_call_sites_use_is_the_shared_function(self):
        from hifi.agents import risk_agent
        from hifi.collective import debate_nodes

        assert risk_agent._extract_json is extract_json
        assert debate_nodes._extract_json is extract_json
