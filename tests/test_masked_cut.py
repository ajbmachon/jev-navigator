"""masked_cut: the one way a region of a source is cut for a request. Every probe runs the real
SecretMasker over the whole source; the values are made up here and were never credentials."""

from __future__ import annotations

import pytest

from jev_navigator.judgments.masked_cut import masked_cut
from jev_navigator.judgments.secrets import MASK, SecretMasker

PATH = "web/gen.js"
VALUES = {
    "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    + "Qm9yZWFsUGluZVRlc3RWYWx1ZU9ubHlOb3RBQ3JlZGVudGlhbEp1c3RBUHJvYmUx"
    + ".Zk8qLm3NpR7sTv2WxY5zAb9CdE1fGh4JkL6mNo",
    "password-with-symbols": "Kq8mLx2PzR7v!Wn4TsB9cHd3@FgJ6aE1yUo5#Rt0Xb3Nc8Vm2",
    "dotted-key": "Lp4Kx9Qm2Rz7Vn3Ts8Wb5Yc1.Hd6Fg0Jk4Ma9Nb2Pc7Qd3R.e8Sf1Tg5Uh0Vi6Wj2Xk9Yl4Z",
    "hex": "9f3a1c7e5b2d8046af1e3c9b7d5a2f80" + "c4e6a8b0d2f41357968ace0bdf135792",
    "alphanumeric": "Vq7Lm2Xz9Rk4Tn8Wb3Yc6Hd1Fg5Jp0Ns2Qa7Ue",
}


def _pieces(value: str, size: int) -> set[str]:
    return {value[at : at + size] for at in range(len(value) - size + 1)}


def _leaks(text: str, value: str) -> set[str]:
    return {piece for piece in _pieces(value, 4) if piece in text}


def _keep(source: str) -> tuple[int, int]:
    at = source.index(PATH)
    return at, at + len(PATH)


@pytest.mark.parametrize("value", VALUES.values(), ids=VALUES.keys())
def test_a_cut_starting_anywhere_before_the_path_sends_none_of_a_keyed_value(value: str) -> None:
    # Arrange: every start from the key's first character to the value's closing quote
    source = '{"token": "' + value + '", "pad": "' + "x" * 40 + f'", "output": "{PATH}"}}'
    value_end = source.index(value) + len(value) + 1

    # Act
    cuts = [
        masked_cut(source, start, len(source), SecretMasker(), file="app/build.json", keep=_keep(source))
        for start in range(value_end + 1)
    ]

    # Assert
    assert [cut for cut in cuts if _leaks(cut, value) or PATH not in cut] == []


@pytest.mark.parametrize("value", VALUES.values(), ids=VALUES.keys())
def test_a_cut_ending_anywhere_after_the_path_sends_none_of_a_keyed_value(value: str) -> None:
    # Arrange: every end from the key's first character to the line's end
    source = f'{{"output": "{PATH}", "token": "' + value + '"}'
    key_start = source.index('"token"')

    # Act
    cuts = [
        masked_cut(source, 0, end, SecretMasker(), file="app/build.json", keep=_keep(source))
        for end in range(key_start, len(source) + 1)
    ]

    # Assert
    assert [cut for cut in cuts if _leaks(cut, value) or PATH not in cut] == []


def test_a_value_found_anywhere_in_the_source_is_masked_where_the_cut_shows_it_without_its_key() -> None:
    # Arrange
    value = VALUES["password-with-symbols"]
    source = f'password = "{value}"\n' + "filler line\n" * 50 + f'deploy("{value}", "{PATH}")\n'
    start = source.index("deploy(")

    # Act
    cut = masked_cut(
        source, start, len(source) - 1, SecretMasker(), file="scripts/deploy.py", keep=_keep(source)
    )

    # Assert
    assert cut == f'deploy("{MASK}", "{PATH}")'


def test_a_cut_through_a_word_drops_the_word_up_to_whitespace_or_a_quote() -> None:
    # Arrange
    source = 'alpha bravo "charlie delta" echo'

    # Act
    cut = masked_cut(source, source.index("ravo"), source.index("elta") + 2, SecretMasker())

    # Assert
    assert cut == ' "charlie '


def test_a_cut_never_drops_the_kept_span_even_when_no_whitespace_reaches_it() -> None:
    # Arrange
    source = "a=" + "b" * 50 + f"={PATH};c=" + "d" * 50

    # Act
    cut = masked_cut(source, 10, len(source) - 10, SecretMasker(), keep=_keep(source))

    # Assert
    assert cut == PATH


def test_a_cut_without_a_masker_only_drops_the_split_pieces() -> None:
    # Arrange
    value = VALUES["alphanumeric"]
    source = f'token = "{value}" and more'

    # Act
    cut = masked_cut(source, 0, source.index(value) + 10, None)

    # Assert
    assert cut == 'token = "'
