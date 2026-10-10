"""The agent search request: its JSON Schema is the typed request's export, and a malformed request is
refused with the field that breaks it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_navigator.directives.agent_search_request import (
    InvalidAgentSearchRequestError,
    parse_agent_search_request,
    schema_text,
)

SCHEMA = Path(__file__).parents[1] / "src/jev_navigator/schemas/agent-search-v1.json"
LIMIT = "code that refuses an order over the item limit"
REFUND = "code that refunds an order"


def request(**changes) -> dict:
    base = {
        "hypotheses": [
            {
                "id": "h1",
                "mechanism": "one helper owns the item limit",
                "evidence": [{"id": "e1", "point": LIMIT}],
                "refuted_by": [{"id": "r1", "point": REFUND}],
            }
        ],
        "terms": [],
        "anchors": [{"file": "shop/orders.py", "line": 5}],
        "files": ["shop/orders.py"],
        "scope": {"include": [], "exclude": [], "with_tests": False},
        "follow": ["callers", "callees"],
    }
    return {**base, **changes}


def test_a_valid_request_parses_from_json_text_and_from_a_mapping_and_fills_the_budget() -> None:
    from_text = parse_agent_search_request(json.dumps(request()))
    assert from_text == parse_agent_search_request(request())
    assert from_text.budget_requests == 8
    assert from_text.hypotheses[0].refuted_by[0].point == REFUND


def test_the_schema_file_is_the_export_of_the_typed_request() -> None:
    assert SCHEMA.read_text() == schema_text()


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"goal": "x"}, "unknown field `goal`"),
        ({"scope": {"include": []}}, "missing required field `exclude` - at `$.scope`"),
        ({"hypotheses": []}, "length >= 1 - at `$.hypotheses`"),
        ({"follow": ["imports"]}, "at `$.follow[0]`"),
        ({"follow": ["callers", "callers"]}, "follow names a relation twice"),
        ({"anchors": [{"file": "a.py", "line": 9, "end": 3}]}, "end 3 is before line 9"),
        ({"budget_requests": 0}, "at `$.budget_requests`"),
    ],
)
def test_a_request_is_refused_with_the_field_that_breaks_it(changes: dict, named: str) -> None:
    with pytest.raises(InvalidAgentSearchRequestError, match="invalid agent search request") as refused:
        parse_agent_search_request(request(**changes))
    assert named in str(refused.value)


@pytest.mark.parametrize(
    ("hypothesis", "named"),
    [
        ({"evidence": [{"id": "e1", "point": LIMIT, "at_least": 2}]}, "unknown field `at_least`"),
        ({"evidence": [{"id": "e_1", "point": LIMIT}]}, "at `$.hypotheses[0].evidence[0].id`"),
        ({"evidence": [{"id": f"e{n}", "point": LIMIT} for n in range(5)]}, "length <= 4"),
        ({"refuted_by": [{"id": f"r{n}", "point": REFUND} for n in range(3)]}, "length <= 2"),
        ({"refuted_by": [{"id": "e1", "point": REFUND}]}, "point ids must be unique; repeated ['e1']"),
        ({"mechanism": ""}, "at `$.hypotheses[0].mechanism`"),
    ],
)
def test_a_hypothesis_is_refused_with_the_field_that_breaks_it(hypothesis: dict, named: str) -> None:
    [base] = request()["hypotheses"]
    with pytest.raises(InvalidAgentSearchRequestError) as refused:
        parse_agent_search_request(request(hypotheses=[{**base, **hypothesis}]))
    assert named in str(refused.value)
