"""Which files an index covers, and every file it leaves out, named with the reason."""

from __future__ import annotations

from pathlib import Path

import pytest
from git_repos import commit_all, commit_files, git, write_files

from jev_navigator.index import listing, tools
from jev_navigator.index.code_index import CodeIndex

APP = "def handle():\n    return 1\n"


def _nested_project(tmp_path: Path) -> Path:
    """A folder without a .git of its own, inside a repository that ignores some of its files."""
    parent = tmp_path / "evidence"
    write_files(parent, {".gitignore": "copies/project/vendor/\n*.gen.ts\n", "notes.md": "notes\n"})
    commit_all(parent)
    child = parent / "copies" / "project"
    write_files(
        child,
        {
            "app.py": APP,
            "vendor/lib.py": "def helper():\n    return 2\n",
            "schema.gen.ts": "export function generated() { return 3; }\n",
        },
    )
    return child


def test_a_folder_inside_another_repository_indexes_its_files_and_names_the_ignored_ones(
    tmp_path: Path,
) -> None:
    # Arrange
    child = _nested_project(tmp_path)

    # Act
    index = CodeIndex.from_directory(child)

    # Assert
    assert index.files == ("app.py",)
    assert [span.file for span in index.find_definition("handle")] == ["app.py"]
    assert index.not_indexed_files == {"schema.gen.ts": "ignored", "vendor/": "ignored"}


def test_a_folder_inside_another_repository_labels_edited_and_new_files_as_worktree_reads(
    tmp_path: Path,
) -> None:
    # Arrange
    parent = tmp_path / "parent"
    commit_files(parent, {"copies/project/kept.py": "def kept():\n    return 0\n"})
    child = parent / "copies" / "project"
    write_files(child, {"edited.py": "def edited():\n    return 1\n"})
    git(parent, "add", ".")
    git(parent, "commit", "-qm", "edited")
    write_files(child, {"edited.py": "def edited():\n    return 2\n", "new.py": "def new():\n    return 3\n"})
    commit = git(parent, "rev-parse", "HEAD").strip()

    # Act
    index = CodeIndex.from_directory(child)
    revisions = {
        name: index.read_slice(index.find_definition(name)[0]).commit for name in ("kept", "edited", "new")
    }

    # Assert
    assert revisions == {"kept": commit, "edited": f"{commit}+worktree", "new": f"{commit}+worktree"}


def test_a_folder_outside_git_names_what_its_ignore_files_leave_out(tmp_path: Path) -> None:
    # Arrange
    write_files(
        tmp_path,
        {
            ".ignore": "build/\nsrc/generated.py\n",
            "src/main.py": "def main():\n    return 0\n",
            "src/generated.py": "def generated():\n    return 1\n",
            "build/deep/out.py": "def out():\n    return 2\n",
        },
    )

    # Act
    index = CodeIndex.from_directory(tmp_path)

    # Assert
    assert index.files == (".ignore", "src/main.py")
    assert index.not_indexed_files == {"build/": "ignored", "src/generated.py": "ignored"}


def test_a_folder_outside_git_names_its_symbolic_links_without_following_them(tmp_path: Path) -> None:
    # Arrange: one link to a file, one to a folder outside the root, one inside an ignored folder
    outside = tmp_path / "outside"
    write_files(outside, {"secret.py": "def secret():\n    return 0\n"})
    root = tmp_path / "plain"
    write_files(root, {".ignore": "build/\n", "app.py": APP, "build/out.py": APP})
    (root / "link.py").symlink_to(root / "app.py")
    (root / "shared").symlink_to(outside)
    (root / "build" / "again.py").symlink_to(root / "app.py")

    # Act
    index = CodeIndex.from_directory(root)

    # Assert
    assert index.files == (".ignore", "app.py")
    assert index.not_indexed_files == {
        "build/": "ignored",
        "link.py": "a symbolic link",
        "shared": "a symbolic link",
    }


def test_a_nested_repository_and_a_symbolic_link_are_named_not_indexed(tmp_path: Path) -> None:
    # Arrange
    commit_files(tmp_path, {"app.py": APP})
    commit_files(tmp_path / "inner", {"inner.py": "def inner():\n    return 0\n"})
    (tmp_path / "link.py").symlink_to(tmp_path / "app.py")

    # Act
    index = CodeIndex.from_directory(tmp_path)

    # Assert
    assert index.files == ("app.py",)
    assert index.not_indexed_files == {
        "inner/": "a separate git repository",
        "link.py": "a symbolic link",
    }


def test_from_git_names_untracked_and_ignored_files_under_a_folder_inside_another_repository(
    tmp_path: Path,
) -> None:
    # Arrange
    child = _nested_project(tmp_path)

    # Act
    index = CodeIndex.from_git(child)

    # Assert
    assert index.files == ()
    assert index.not_indexed_files == {
        "app.py": "not tracked by git",
        "schema.gen.ts": "ignored",
        "vendor/": "ignored",
    }


def test_from_git_on_a_folder_inside_another_repository_labels_an_edited_file_as_a_worktree_read(
    tmp_path: Path,
) -> None:
    # Arrange
    parent = tmp_path / "parent"
    commit_files(
        parent,
        {"copies/project/kept.py": "def kept():\n    return 0\n", "copies/project/edited.py": APP},
    )
    child = parent / "copies" / "project"
    write_files(child, {"edited.py": "def handle():\n    return 2\n"})
    commit = git(parent, "rev-parse", "HEAD").strip()

    # Act
    index = CodeIndex.from_git(child)
    revisions = {name: index.read_slice(index.find_definition(name)[0]).commit for name in ("kept", "handle")}

    # Assert
    assert index.files == ("edited.py", "kept.py")
    assert revisions == {"kept": commit, "handle": f"{commit}+worktree"}


def test_from_git_names_a_requested_path_with_no_file(tmp_path: Path) -> None:
    # Arrange
    commit_files(tmp_path, {"app.py": APP})

    # Act
    index = CodeIndex.from_git(tmp_path, ["app.py", "gone.py"], max_files=2)

    # Assert
    assert index.files == ("app.py",)
    assert index.not_indexed_files == {"gone.py": "no file at this path"}


def test_a_repository_git_refuses_is_reported_instead_of_listed_as_a_plain_directory(
    tmp_path: Path, monkeypatch
) -> None:
    # Arrange: git refuses a repository it believes another user owns
    repository = tmp_path / "repository"
    commit_files(repository, {"a.py": "needle = 1\n"})
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")

    # Act and assert
    with pytest.raises(tools.ToolFailedError, match="dubious ownership"):
        listing.working_files(repository)


def test_a_plain_directory_is_listed_whatever_language_git_speaks(tmp_path: Path, monkeypatch) -> None:
    # Arrange: a translated "not a git repository" must still mean a plain directory
    directory = tmp_path / "plain"
    directory.mkdir()
    (directory / "a.py").write_text("needle = 1\n")
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    monkeypatch.setenv("LANGUAGE", "de")
    monkeypatch.delenv("LC_ALL", raising=False)

    # Act and assert
    assert listing.working_files(directory).files == ("a.py",)


def test_listing_outside_git_ignores_a_configured_ripgrep_filter(tmp_path: Path, monkeypatch) -> None:
    # Outside a Git worktree the file inventory comes from `rg --files`; a ripgrep config must not
    # change which files the index sees there either.
    config = tmp_path / "rg.conf"
    config.write_text("--glob=!a.py\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))
    directory = tmp_path / "plain"
    directory.mkdir()
    (directory / "a.py").write_text("needle = 1\n")

    assert listing.working_files(directory).files == ("a.py",)


def test_a_symbolic_link_to_a_folder_is_named_a_link_not_a_separate_repository(tmp_path: Path) -> None:
    # Arrange
    commit_files(tmp_path, {"app.py": APP, "lib/util.py": "def util():\n    return 0\n"})
    (tmp_path / "shared").symlink_to(tmp_path / "lib", target_is_directory=True)

    # Act
    index = CodeIndex.from_directory(tmp_path)

    # Assert
    assert index.files == ("app.py", "lib/util.py")
    assert index.not_indexed_files == {"shared": "a symbolic link"}


def test_a_change_outside_the_folder_never_labels_the_same_path_inside_it_a_worktree_read(
    tmp_path: Path,
) -> None:
    # Arrange: git status names the edited app/a.py from the top of the repository, the same path
    # the folder's own unchanged app/a.py has relative to the folder
    parent = tmp_path / "parent"
    commit_files(
        parent,
        {
            "app/a.py": "def outside():\n    return 0\n",
            "copies/project/app/a.py": "def inside():\n    return 0\n",
        },
    )
    write_files(parent, {"app/a.py": "def outside():\n    return 1\n"})
    child = parent / "copies" / "project"
    commit = git(parent, "rev-parse", "HEAD").strip()

    # Act
    index = CodeIndex.from_directory(child)

    # Assert
    assert index.read_slice(index.find_definition("inside")[0]).commit == commit
