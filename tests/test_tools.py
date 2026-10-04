from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest
from git_repos import git

from jev_navigator.index import tools

MISSING_OBJECT = "0" * 40


def stored_blob(repository: Path, text: str) -> str:
    """Writes ``text`` into the repository's object store only and returns its object id."""
    git(repository, "init", "-q")
    return git(repository, "hash-object", "-w", "--stdin", stdin=text).strip()


def test_export_writes_each_blob_at_its_path(tmp_path: Path) -> None:
    # Arrange
    repository, destination = tmp_path / "repo", tmp_path / "export"
    repository.mkdir()
    object_id = stored_blob(repository, "def kept():\n    return 1\n")

    # Act
    tools.export_blobs(repository, {"app/kept.py": object_id, "copy.py": object_id}, destination)

    # Assert
    assert (destination / "app/kept.py").read_text() == "def kept():\n    return 1\n"
    assert (destination / "copy.py").read_text() == "def kept():\n    return 1\n"


def test_export_refuses_a_path_that_leaves_the_destination(tmp_path: Path) -> None:
    # Arrange
    repository, destination = tmp_path / "repo", tmp_path / "export"
    repository.mkdir()
    object_id = stored_blob(repository, "escaped = True\n")

    # Act
    with pytest.raises(tools.ToolFailedError, match="outside"):
        tools.export_blobs(repository, {"../escaped.py": object_id}, destination)

    # Assert
    assert not (tmp_path / "escaped.py").exists()


def test_export_fails_loudly_when_git_lacks_an_object(tmp_path: Path) -> None:
    # Arrange
    repository = tmp_path / "repo"
    repository.mkdir()
    stored_blob(repository, "present = True\n")

    # Act and assert
    with pytest.raises(tools.ToolFailedError, match=MISSING_OBJECT):
        tools.export_blobs(repository, {"gone.py": MISSING_OBJECT}, tmp_path / "export")


INVALID_RULE = "id: broken\nlanguage: python\nrule:\n  kind: not_a_real_kind\n"
VALID_RULE = "id: function\nlanguage: python\nrule:\n  kind: function_definition\n"


def _write_python_files(root: Path, *names: str) -> None:
    for name in names:
        (root / name).write_text("def f():\n    return 1\n")


def stand_in_ast_grep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    """Puts a script in ast-grep's place. ``body`` is Python that sees the scanned ``files`` and may
    print to stdout and stderr and exit."""
    script = tmp_path / "stand-in-ast-grep"
    script.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        "files = [argument for argument in sys.argv[1:] if argument.endswith('.py')]\n" + body
    )
    script.chmod(0o755)
    monkeypatch.setattr(tools, "AST_GREP", str(script))


def test_a_rule_ast_grep_rejects_fails_with_its_exit_code_and_message(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "a.py").write_text("def a():\n    return 1\n")

    # Act / Assert
    with pytest.raises(tools.ToolFailedError, match=r"exited 8: (?s:.*)invalid kind"):
        list(tools.ast_grep_rules(INVALID_RULE, ["a.py"], tmp_path, refused={}))


def test_a_chunk_that_fails_fails_the_scan_after_the_matches_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: one file per command; the command for b.py fails
    _write_python_files(tmp_path, "a.py", "b.py", "c.py")
    stand_in_ast_grep(
        tmp_path,
        monkeypatch,
        "if 'b.py' in files:\n    print('ast-grep: panicked on b.py', file=sys.stderr)\n    sys.exit(3)\n"
        "for file in files:\n    print(json.dumps({'file': file}))\n",
    )
    monkeypatch.setattr(tools, "MAX_FILES_PER_COMMAND", 1)
    matches: list[dict] = []

    # Act
    with pytest.raises(tools.ToolFailedError) as failure:
        matches.extend(tools.ast_grep_rules(VALID_RULE, ["a.py", "b.py", "c.py"], tmp_path, refused={}))

    # Assert
    assert matches == [{"file": "a.py"}]
    assert str(failure.value) == f"{tools.AST_GREP} exited 3: ast-grep: panicked on b.py"


def test_a_process_killed_partway_through_a_line_reports_why_it_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the process prints one whole match, half of the next, then dies
    _write_python_files(tmp_path, "a.py", "b.py")
    stand_in_ast_grep(
        tmp_path,
        monkeypatch,
        'print(json.dumps({\'file\': \'a.py\'}))\nsys.stdout.write(\'{"file": "b.py", "text": "unterm\')\n'
        "sys.stdout.flush()\nprint('ast-grep: out of memory', file=sys.stderr)\nsys.exit(137)\n",
    )

    # Act / Assert
    with pytest.raises(tools.ToolFailedError) as failure:
        list(tools.ast_grep_rules(VALID_RULE, ["a.py", "b.py"], tmp_path, refused={}))
    assert str(failure.value) == f"{tools.AST_GREP} exited 137: ast-grep: out of memory"
    assert isinstance(failure.value.__cause__, ValueError)


def test_a_file_whose_name_starts_with_a_dash_is_scanned_as_a_file(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "-x.py").write_text("def x():\n    return 1\n")

    # Act
    matches = list(tools.ast_grep_rules(VALID_RULE, ["-x.py"], tmp_path, refused={}))

    # Assert
    assert [match["file"] for match in matches] == ["-x.py"]


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX preprocessor script")
def test_ripgrep_ignores_a_configured_preprocessor(tmp_path: Path, monkeypatch) -> None:
    # A ripgrep config in the environment (RIPGREP_CONFIG_PATH) can name `--pre=<program>`, which
    # ripgrep runs for each searched file. Over an untrusted repository that is code execution, so
    # jvn's searches must ignore the config entirely.
    marker = tmp_path / "preprocessor-ran"
    preprocessor = tmp_path / "pre.sh"
    preprocessor.write_text(f'#!/bin/sh\n: > "{marker}"\ncat "$1"\n')
    preprocessor.chmod(preprocessor.stat().st_mode | stat.S_IXUSR)
    config = tmp_path / "rg.conf"
    config.write_text(f"--pre={preprocessor}\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))

    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "a.py").write_text("needle = 1\n")

    found = tuple(hit.file for hit in tools.ripgrep_fixed("needle", ("a.py",), repository, 5))

    assert found == ("a.py",)  # the search still works
    assert not marker.exists()  # but the configured preprocessor never ran


def test_listing_outside_git_ignores_a_configured_ripgrep_filter(tmp_path: Path, monkeypatch) -> None:
    # Outside a Git worktree the file inventory comes from `rg --files`; a ripgrep config must not
    # change which files the index sees there either.
    config = tmp_path / "rg.conf"
    config.write_text("--glob=!a.py\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))
    directory = tmp_path / "plain"
    directory.mkdir()
    (directory / "a.py").write_text("needle = 1\n")

    assert tools.listed_files(directory) == ("a.py",)
