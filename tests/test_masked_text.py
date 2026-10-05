"""masked_lines, the owner of the text a request may show, with the real SecretMasker."""

from __future__ import annotations

import time

import pytest
from secret_shapes import COPY_VALUE, SECRET_SHAPES, pieces

from jev_navigator.judgments.masked_text import masked_lines
from jev_navigator.judgments.secrets import (
    MASK,
    Copies,
    SecretMasker,
)


@pytest.mark.parametrize("value", SECRET_SHAPES.values(), ids=SECRET_SHAPES.keys())
def test_a_keyed_value_of_any_shape_leaves_no_piece_in_the_masked_lines(value: str) -> None:
    # Arrange
    lines = ["{", f'  "token": "{value}",', '  "output": "web/gen.js"', "}"]

    # Act
    masked = masked_lines(lines, "app/build.json", SecretMasker())

    # Assert
    assert [piece for piece in pieces(value) if piece in "\n".join(masked)] == []
    assert masked[2] == lines[2]


def test_a_value_found_anywhere_is_masked_on_every_line_that_copies_it() -> None:
    # Arrange
    lines = [f'API_TOKEN = "{COPY_VALUE}"', "", f'post("{COPY_VALUE}", order)']

    # Act
    masked = masked_lines(lines, "app/client.py", SecretMasker())

    # Assert
    assert masked == ('API_TOKEN = "[MASKED]"', "", 'post("[MASKED]", order)')


def test_a_short_value_is_masked_only_as_a_whole_word() -> None:
    # Arrange
    lines = ['password = "hunter2"', 'print("hunter2", "hunter2x")']

    # Act
    masked = masked_lines(lines, "app/a.py", SecretMasker())

    # Assert
    assert masked[1] == f'print("{MASK}", "hunter2x")'


def test_a_value_spanning_lines_keeps_the_line_count() -> None:
    # Arrange
    body = [f"{COPY_VALUE}{number:08d}" for number in range(3)]
    lines = ["before", "-----BEGIN RSA PRIVATE KEY-----", *body, "-----END RSA PRIVATE KEY----- after", "end"]

    # Act
    masked = masked_lines(lines, "deploy/key.txt", SecretMasker())

    # Assert
    assert masked == ("before", MASK, "", "", "", " after", "end")


def test_finding_copies_of_ten_thousand_remembered_values_stays_in_milliseconds() -> None:
    # Arrange: one regex per value took 785 ms here; a request masks against every remembered value
    remembered = [f"{number:05d}Kq8mLx2PzR7vWn4TsB9c" for number in range(10_000)]
    text = "x = 1\n" * 3_000 + f'send("{remembered[1234]}")\n'

    # Act
    started = time.process_time()
    masked = Copies(remembered).sub(text)
    spent = time.process_time() - started

    # Assert: in this process's own CPU time, which other jobs' load leaves alone, a bound eight
    # times what it measures and eight times below one regex per value
    assert remembered[1234] not in masked and f'send("{MASK}")' in masked
    assert spent < 0.1


def test_a_short_value_glued_after_a_long_values_copy_is_masked_in_the_masked_lines() -> None:
    # Arrange
    lines = [
        'password = "hunter2"',
        'api_key = "sk_abcdefghijklmnop"',
        "joined = sk_abcdefghijklmnophunter2 end",
    ]

    # Act
    masked = masked_lines(lines, "app/a.py", SecretMasker())

    # Assert
    assert masked[2] == f"joined = {MASK}{MASK} end"
