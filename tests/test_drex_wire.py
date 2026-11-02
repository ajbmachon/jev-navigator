"""Drex's wire dialect: criteria as text and no base64 data URL, everything else as the judge built it."""

from __future__ import annotations

import json

from jev_navigator.adapters.drex_wire import ZERO_WIDTH_SPACE, drex_wire


def test_a_structured_criterion_becomes_its_compact_json_text_and_text_or_null_pass_unchanged():
    # Arrange
    described = {"what": "The code itself.", "not_for": "Calls to it.", "examples": ["prüft `x > 0`"]}
    questions = {
        "check": {"type": "noul", "instructions": "y?", "criteria": {"true": described, "false": "Other."}},
        "pick": {"type": "choice", "instructions": "Which?", "criteria": {"0": "first", "none": None}},
        "rate": {"type": "score", "instructions": "How well?", "criteria": ["poorly", {"what": "well"}]},
        "plain": {"type": "noul", "instructions": "z?"},
    }

    # Act
    _, wire = drex_wire({}, questions)

    # Assert
    assert wire["check"]["criteria"] == {"true": json.dumps(described, ensure_ascii=False), "false": "Other."}
    assert "prüft" in wire["check"]["criteria"]["true"]
    assert wire["pick"]["criteria"] == {"0": "first", "none": None}
    assert wire["rate"]["criteria"] == ["poorly", '{"what": "well"}']
    assert wire["plain"] == {"type": "noul", "instructions": "z?"}


def test_only_base64_data_urls_are_broken_and_keys_never_change():
    # Arrange
    state = {
        "data:image/png;base64,key": "src='data:image/png;base64,AAAA'",
        "items": ["Data:Text/Plain;Base64,QQ==", "data: rows", "metadata:x;base64 without comma", 3, None],
    }

    # Act
    wire, _ = drex_wire(state, {})

    # Assert
    assert list(wire) == list(state)
    assert wire["data:image/png;base64,key"] == f"src='data{ZERO_WIDTH_SPACE}:image/png;base64,AAAA'"
    assert wire["items"] == [
        f"Data{ZERO_WIDTH_SPACE}:Text/Plain;Base64,QQ==",
        "data: rows",
        "metadata:x;base64 without comma",
        3,
        None,
    ]


def test_the_callers_request_is_left_untouched():
    criteria = {"true": {"what": "yes"}, "false": {"what": "no"}}
    state = {"code": "data:image/gif;base64,R0lG"}
    questions = {"check": {"type": "noul", "instructions": "y?", "criteria": criteria}}
    before = json.dumps({"state": state, "questions": questions})

    drex_wire(state, questions)

    assert json.dumps({"state": state, "questions": questions}) == before
