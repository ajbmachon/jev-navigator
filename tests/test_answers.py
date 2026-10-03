from __future__ import annotations

import pytest

from jev_navigator.judgments.answers import (
    JevResponse,
    NoulAnswer,
    reported_input_tokens,
    response_from_raw,
    response_to_raw,
)

ANSWERS = {"q": {"type": "noul", "noul": 0.9}}


def test_a_response_without_usage_has_no_input_tokens() -> None:
    assert response_from_raw({"model": "m", "answers": ANSWERS}).input_tokens is None
    assert response_from_raw({"model": "m", "usage": None, "answers": ANSWERS}).input_tokens is None
    assert response_from_raw({"model": "m", "usage": {}, "answers": ANSWERS}).input_tokens is None


def test_a_reported_zero_stays_zero() -> None:
    assert (
        response_from_raw({"model": "m", "usage": {"input_tokens": 0}, "answers": ANSWERS}).input_tokens == 0
    )


@pytest.mark.parametrize("value", [None, "12", -1, True, 1.5])
def test_only_a_non_negative_integer_counts_as_reported(value: object) -> None:
    assert reported_input_tokens({"usage": {"input_tokens": value}}) is None


def test_the_raw_form_keeps_unreported_unreported_on_a_round_trip() -> None:
    unreported = JevResponse({"q": NoulAnswer(0.9)}, "m")
    reported = JevResponse({"q": NoulAnswer(0.9)}, "m", 12)

    assert response_from_raw(response_to_raw(unreported)).input_tokens is None
    assert response_from_raw(response_to_raw(reported)).input_tokens == 12
