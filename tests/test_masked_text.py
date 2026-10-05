"""masked_lines, the owner of the text a request may show, with the real SecretMasker."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest
from git_repos import write_files
from secret_shapes import COPY_VALUE, SECRET_SHAPES, pieces

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.judgments.masked_text import masked_lines
from jev_navigator.judgments.secrets import (
    MASK,
    Copies,
    SecretMasker,
    hidden_scope,
    mask_request,
    masked_values,
)


@pytest.mark.parametrize("value", SECRET_SHAPES.values(), ids=SECRET_SHAPES.keys())
def test_a_keyed_value_of_any_shape_leaves_no_piece_in_the_masked_lines(value: str) -> None:
    # Arrange
    lines = ["{", f'  "token": "{value}",', '  "output": "web/gen.js"', "}"]

    # Act
    masked = masked_lines(lines, "app/build.json", SecretMasker(), hidden_scope())

    # Assert
    assert [piece for piece in pieces(value) if piece in "\n".join(masked)] == []
    assert masked[2] == lines[2]


def test_a_value_found_anywhere_is_masked_on_every_line_that_copies_it() -> None:
    # Arrange
    lines = [f'API_TOKEN = "{COPY_VALUE}"', "", f'post("{COPY_VALUE}", order)']

    # Act
    masked = masked_lines(lines, "app/client.py", SecretMasker(), hidden_scope())

    # Assert
    assert masked == ('API_TOKEN = "[MASKED]"', "", 'post("[MASKED]", order)')


def test_a_short_value_is_masked_only_as_a_whole_word() -> None:
    # Arrange
    lines = ['password = "hunter2"', 'print("hunter2", "hunter2x")']

    # Act
    masked = masked_lines(lines, "app/a.py", SecretMasker(), hidden_scope())

    # Assert
    assert masked[1] == f'print("{MASK}", "hunter2x")'


def test_a_value_spanning_lines_keeps_the_line_count() -> None:
    # Arrange
    body = [f"{COPY_VALUE}{number:08d}" for number in range(3)]
    lines = ["before", "-----BEGIN RSA PRIVATE KEY-----", *body, "-----END RSA PRIVATE KEY----- after", "end"]

    # Act
    masked = masked_lines(lines, "deploy/key.txt", SecretMasker(), hidden_scope())

    # Assert
    assert masked == ("before", MASK, "", "", "", " after", "end")


def test_finding_copies_of_ten_thousand_remembered_values_stays_in_milliseconds() -> None:
    # Arrange: one regex per value took 785 ms here; a request masks against every remembered value
    remembered = [f"{number:05d}Kq8mLx2PzR7vWn4TsB9c" for number in range(10_000)]
    text = "x = 1\n" * 3_000 + f'send("{remembered[1234]}")\n'

    # Act
    started = time.perf_counter()
    masked = Copies(remembered).sub(text)
    elapsed = time.perf_counter() - started

    # Assert: a bound eight times what it measures, and eight times below one regex per value
    assert remembered[1234] not in masked and f'send("{MASK}")' in masked
    assert elapsed < 0.1


def test_a_request_hides_the_copy_of_a_value_a_live_scope_remembers() -> None:
    # Arrange
    scope = hidden_scope()
    scope.add([COPY_VALUE])

    # Act
    masked, _, _ = mask_request({"code": f'post("{COPY_VALUE}")'}, {}, SecretMasker())

    # Assert
    assert masked == {"code": f'post("{MASK}")'}


def test_a_files_hidden_values_go_away_with_its_index(tmp_path: Path) -> None:
    # Arrange
    write_files(tmp_path, {"app/settings.py": f'API_TOKEN = "{COPY_VALUE}"\n'})
    index = CodeIndex(tmp_path, ["app/settings.py"], fact_cache_dir=tmp_path / "cache")
    index.read_slice(Span("app/settings.py", 1, 1))
    remembered_while_open = COPY_VALUE in masked_values({}, SecretMasker())

    # Act
    del index
    gc.collect()

    # Assert
    assert remembered_while_open and COPY_VALUE not in masked_values({}, SecretMasker())
