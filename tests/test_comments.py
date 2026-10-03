"""Comments and facts read from source: kinds, the noise filter, declaration comments and cheap facts."""

from __future__ import annotations

from pathlib import Path

import pytest
from git_repos import git

from jev_navigator.comments import (
    CodeAboveReason,
    code_above_comment,
    comment_facts,
    comments_in_diff,
    find_comments,
    noise_reason,
)
from jev_navigator.facts import DATE, TICKET_REFERENCE, find_facts, outside_names
from jev_navigator.index.code_index import CodeIndex

ORDERS = '''\
# Copyright 2026 Example Ltd. Licensed under the MIT License.
"""Order helpers."""

import os  # noqa: F401

# ----------------------------------------


# Loads the order limit.
# TODO: make this configurable
def load_limit():
    """Returns the maximum items per order."""
    return int(os.environ.get("LIMIT", "5"))


def save(order):
    try:
        order.save()
    except Exception:
        # ignore failures, the next sync retries (see PAY-123)
        pass
    # old_value = compute(order)
    return order
'''


@pytest.fixture
def comment_index(tmp_path: Path) -> CodeIndex:
    (tmp_path / "orders.py").write_text(ORDERS)
    return CodeIndex(tmp_path, ["orders.py"])


def summary(blocks) -> list[tuple[int, int, str]]:
    return [(block.span.start, block.span.end, block.kind.value) for block in blocks]


def test_find_comments_merges_line_comments_and_names_each_kind(comment_index: CodeIndex) -> None:
    # Act
    blocks = find_comments(comment_index).kept

    # Assert
    assert summary(blocks) == [
        (1, 1, "header"),
        (2, 2, "docstring"),
        (4, 4, "tool_directive"),
        (6, 6, "block"),
        (9, 10, "declaration"),
        (12, 12, "docstring"),
        (20, 20, "block"),
        (22, 22, "block"),
    ]


def test_the_ready_made_noise_filter_drops_dividers_licences_and_directives(comment_index: CodeIndex) -> None:
    # Act
    blocks = find_comments(comment_index, drop=noise_reason).kept

    # Assert
    assert [block.span.start for block in blocks] == [2, 9, 12, 20, 22]


def test_dropped_comments_are_returned_with_their_reason_so_the_total_stays_known(
    comment_index: CodeIndex,
) -> None:
    # Act
    found = find_comments(comment_index, drop=noise_reason)

    # Assert
    assert [(entry.block.span.start, entry.reason) for entry in found.dropped] == [
        (1, "licence_header"),
        (4, "tool_directive"),
        (6, "divider"),
    ]
    assert len(found.kept) + len(found.dropped) == len(find_comments(comment_index).kept)


def test_declaration_comments_and_docstrings_carry_their_symbol(comment_index: CodeIndex) -> None:
    # Act
    by_start = {block.span.start: block for block in find_comments(comment_index, drop=noise_reason).kept}

    # Assert
    assert by_start[9].attached.name == "load_limit"
    assert by_start[9].text.splitlines() == ["# Loads the order limit.", "# TODO: make this configurable"]
    assert by_start[12].attached.name == "load_limit"
    assert by_start[2].attached is None


def test_cheap_facts_are_marked(comment_index: CodeIndex) -> None:
    # Act
    by_start = {block.span.start: block for block in find_comments(comment_index, drop=noise_reason).kept}

    # Assert
    assert "todo_without_owner" in by_start[9].fact_names()
    assert "ticket_reference" in by_start[20].fact_names()
    assert "commented_out_code" in by_start[22].fact_names()
    assert comment_facts("# TODO(maintainer): owner named") == ()
    assert [fact.name for fact in comment_facts("# fixed on 2026-09-28 in #1302")] == [
        "date",
        "ticket_reference",
    ]


def test_facts_carry_their_span_and_matched_text() -> None:
    # Act
    facts = comment_facts("# first line\n# fixed on 2026-09-28")

    # Assert
    assert [(fact.name, fact.line, fact.text) for fact in facts] == [("date", 2, "2026-09-28")]
    assert "# first line\n# fixed on 2026-09-28"[facts[0].start : facts[0].end] == "2026-09-28"


def test_the_optional_name_filter_skips_dates_inside_paths_file_names_and_identifiers() -> None:
    # Arrange
    rules = [DATE.with_filter(outside_names), TICKET_REFERENCE]
    text = (
        "# see docs/adr/ADR-7-2026-09-27.md and report-2026-09-28.json and run_2026-09-29_done,"
        " decided on 2026-09-30."
    )

    # Act
    dates = [fact.text for fact in find_facts(text, rules) if fact.name == "date"]
    unfiltered = [fact.text for fact in find_facts(text, [DATE]) if fact.name == "date"]

    # Assert
    assert dates == ["2026-09-30"]
    assert unfiltered == ["2026-09-27", "2026-09-28", "2026-09-30"]


def test_code_above_comment_stops_at_the_enclosing_block_opener(comment_index: CodeIndex) -> None:
    # Act
    above = code_above_comment(comment_index, "orders.py", 20)

    # Assert
    assert above.reason == CodeAboveReason.FOUND
    assert (above.code.span.start, above.code.span.end) == (19, 19)
    assert above.code.text.strip() == "except Exception:"


def test_a_comment_below_a_blank_line_has_no_code_above_and_says_why(comment_index: CodeIndex) -> None:
    # Act
    above = code_above_comment(comment_index, "orders.py", 9)

    # Assert
    assert above.code is None
    assert above.reason == CodeAboveReason.BLANK_LINE


def test_a_comment_on_the_first_line_has_no_code_above(comment_index: CodeIndex) -> None:
    # Act
    above = code_above_comment(comment_index, "orders.py", 1)

    # Assert
    assert above.code is None
    assert above.reason == CodeAboveReason.START_OF_FILE


def test_comments_in_diff_includes_untouched_comments_above_changed_code(sample_repo: Path) -> None:
    # Arrange
    path = sample_repo / "app/comments.py"
    path.write_text(path.read_text().replace("    return amount\n", "    return amount * 2\n"))
    git(sample_repo, "commit", "-qam", "change charge only")
    index = CodeIndex.from_git(sample_repo)

    # Act
    touched = comments_in_diff(index, "HEAD~1", "HEAD").kept

    # Assert
    assert [(block.span.file, block.span.start) for block in touched] == [("app/comments.py", 3)]


def test_comments_in_diff_finds_changes_in_files_with_non_ascii_names(sample_repo: Path) -> None:
    # Arrange
    path = sample_repo / "app/größe.py"
    path.write_text("# Returns the size.\ndef groesse():\n    return 1\n")
    git(sample_repo, "add", ".")
    git(sample_repo, "commit", "-qm", "size")
    path.write_text("# Returns the size.\ndef groesse():\n    return 2\n")
    git(sample_repo, "commit", "-qam", "change size")
    index = CodeIndex.from_git(sample_repo)

    # Act
    touched = comments_in_diff(index, "HEAD~1", "HEAD").kept

    # Assert
    assert [(block.span.file, block.span.start) for block in touched] == [("app/größe.py", 1)]


def test_blocks_of_different_syntax_are_never_merged(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "auth.ts").write_text(
        "/**\n * Checks the token.\n */\n"
        "// SECURITY: never log the token\n// even in debug\n"
        "export function check() {}\n"
    )
    index = CodeIndex(tmp_path, ["auth.ts"])

    # Act
    blocks = find_comments(index).kept

    # Assert
    assert [(block.span.start, block.span.end, block.kind.value) for block in blocks] == [
        (1, 3, "jsdoc"),
        (4, 5, "declaration"),
    ]
