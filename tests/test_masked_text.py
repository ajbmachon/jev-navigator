"""masked_lines and trimmed_cut, the owner of the text a request may show, with the real SecretMasker."""

from __future__ import annotations

import pytest
from secret_shapes import COPY_VALUE, SECRET_SHAPES, pieces

from jev_navigator.judgments.masked_text import masked_lines, trimmed_cut
from jev_navigator.judgments.secrets import MASK, SecretMasker


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


def test_a_cut_through_a_word_drops_the_word_up_to_whitespace_or_a_quote() -> None:
    # Arrange
    text = 'alpha bravo "charlie delta" echo'

    # Act
    cut = trimmed_cut(text, text.index("ravo"), text.index("elta") + 2)

    # Assert
    assert cut == ' "charlie '


def test_a_cut_never_drops_the_kept_span_even_when_no_whitespace_reaches_it() -> None:
    # Arrange
    text = "a=" + "b" * 50 + "=web/gen.js;c=" + "d" * 50
    keep = (text.index("web/gen.js"), text.index("web/gen.js") + len("web/gen.js"))

    # Act
    cut = trimmed_cut(text, 10, len(text) - 10, keep)

    # Assert
    assert cut == "web/gen.js"


def test_a_cut_on_whitespace_keeps_both_edges() -> None:
    # Arrange
    text = "alpha bravo charlie"

    # Act
    cut = trimmed_cut(text, 5, 11)

    # Assert
    assert cut == " bravo"
