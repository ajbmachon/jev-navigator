from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.comments import find_comments
from jev_navigator.index import tools
from jev_navigator.index.bindings import BindingStatus
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.fact_cache import FactCache

BUNDLE = "orchestrator/contracts/launch-contract.mjs"
STATEMENT = "export function launch(){return 1};"


def _one_line(characters: int) -> str:
    return (STATEMENT * (characters // len(STATEMENT) + 1))[:characters]


def _repository(tmp_path: Path, bundle_characters: int) -> Path:
    files = {
        BUNDLE: _one_line(bundle_characters),
        "src/small.py": "def small():\n    return 1\n",
        "src/importer.js": f"import {{ launch }} from '../{BUNDLE}';\nlaunch();\n",
    }
    commit_files(tmp_path / "repo", files)
    return tmp_path / "repo"


class _AstGrepRecorder:
    """Stands in for the ast-grep process only: it records every command and answers 'no matches',
    so a missing guard shows as a recorded command, never as a real multi-gigabyte parse."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.commands: list[list[str]] = []
        original = tools._json_lines

        def run(arguments, cwd):
            if arguments[0] == tools.AST_GREP and "scan" in arguments:
                self.commands.append(list(arguments))
                return iter(())
            return original(arguments, cwd)

        monkeypatch.setattr(tools, "_json_lines", run)

    def received(self, file: str) -> bool:
        return any(file in command for command in self.commands)


def test_a_file_over_the_memory_bound_is_never_parsed_and_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ast_grep = _AstGrepRecorder(monkeypatch)
    index = CodeIndex.from_git(_repository(tmp_path, 668_777), fact_cache_dir=tmp_path / "facts")

    index.functions_in_files(index.files)

    assert not ast_grep.received(BUNDLE)
    assert index.unavailable_files[BUNDLE] == (
        "too large to parse: estimated parse peak 22 GB, longest line 668,777 characters"
    )
    assert index.parser_scans_pending == ()


def test_a_file_under_the_bound_is_parsed_even_when_a_trigger_flags_it(tmp_path: Path) -> None:
    repository = _repository(tmp_path, 20_000)
    index = CodeIndex.from_git(repository, fact_cache_dir=tmp_path / "facts")

    functions = index.functions_in(BUNDLE)

    assert any(span.name == "launch" for span in functions)
    assert BUNDLE not in index.unavailable_files
    assert FactCache(tmp_path / "facts").load(BUNDLE, (repository / BUNDLE).read_bytes()) is not None


def test_a_refused_file_leaves_no_fact_cache_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _AstGrepRecorder(monkeypatch)
    repository = _repository(tmp_path, 668_777)
    index = CodeIndex.from_git(repository, fact_cache_dir=tmp_path / "facts")

    index.functions_in(BUNDLE)
    index.functions_in("src/small.py")

    cache = FactCache(tmp_path / "facts")
    assert cache.load(BUNDLE, (repository / BUNDLE).read_bytes()) is None
    assert cache.load("src/small.py", (repository / "src/small.py").read_bytes()) is not None


def test_an_importer_of_a_guarded_file_keeps_the_import_and_its_name_binds_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _AstGrepRecorder(monkeypatch)
    index = CodeIndex.from_git(_repository(tmp_path, 668_777), fact_cache_dir=tmp_path / "facts")

    index.functions_in(BUNDLE)
    binding = index.binding_of("src/importer.js", 2, "launch", None)

    assert BUNDLE in index.imports("src/importer.js")
    assert binding.status == BindingStatus.UNKNOWN
    assert BUNDLE in binding.reason


def test_comment_scanning_never_parses_a_guarded_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ast_grep = _AstGrepRecorder(monkeypatch)
    index = CodeIndex.from_git(_repository(tmp_path, 668_777), fact_cache_dir=tmp_path / "facts")

    found = find_comments(index)

    assert not ast_grep.received(BUNDLE)
    assert found.refused_files[BUNDLE].startswith("too large to parse: estimated parse peak 22 GB")


def test_the_door_yields_the_matches_of_the_files_it_parsed_and_names_the_ones_it_refused(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path, 668_777)
    refused: dict[str, str] = {}

    matches = list(
        tools.ast_grep_rules(
            "id: function\nlanguage: python\nrule:\n  kind: function_definition",
            ["src/small.py"],
            repository,
            refused=refused,
        )
    )
    guarded = list(
        tools.ast_grep_rules(
            "id: function\nlanguage: javascript\nrule:\n  kind: function_declaration",
            [BUNDLE],
            repository,
            refused=refused,
        )
    )

    assert [match["file"] for match in matches] == ["src/small.py"]
    assert guarded == []
    assert set(refused) == {BUNDLE}


SCAN_IN_A_FRESH_PROCESS = """
import json, resource, sys
from pathlib import Path
from jev_navigator.index.code_index import CodeIndex
index = CodeIndex.from_git(Path(sys.argv[1]), fact_cache_dir=Path(sys.argv[2]))
index.functions_in_files(index.files)
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(json.dumps({"peak_bytes": peak, "unavailable": index.unavailable_files}))
"""


def test_scanning_a_scope_with_a_guarded_file_keeps_peak_memory_bounded(tmp_path: Path) -> None:
    repository = _repository(tmp_path, 668_777)

    completed = subprocess.run(
        [sys.executable, "-c", SCAN_IN_A_FRESH_PROCESS, str(repository), str(tmp_path / "facts")],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(completed.stdout)
    peak_megabytes = report["peak_bytes"] / (1 if sys.platform == "darwin" else 1024) / 1_000_000
    assert peak_megabytes < 250
    assert BUNDLE in report["unavailable"]


FUNCTION_RULE = "id: function\nlanguage: typescript\nrule:\n  kind: function_declaration"


def _scan_typescript(repository: Path) -> list[dict]:
    return list(tools.ast_grep_rules(FUNCTION_RULE, ["a.ts"], repository, refused={}))


def test_a_repositorys_own_sgconfig_cannot_change_what_is_found(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    commit_files(
        repository,
        {
            "a.ts": "export function a() { return 1 }\n",
            "sgconfig.yml": "languageGlobs:\n  javascript: ['*.ts']\n",
        },
    )

    matches = _scan_typescript(repository)

    assert [match["file"] for match in matches] == ["a.ts"]


def test_a_repositorys_custom_language_library_is_never_loaded(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    commit_files(
        repository,
        {
            "a.ts": "export function a() { return 1 }\n",
            "sgconfig.yml": (
                "customLanguages:\n  mylang:\n    libraryPath: ./missing.so\n    extensions: [my]\n"
            ),
        },
    )

    matches = _scan_typescript(repository)

    assert [match["file"] for match in matches] == ["a.ts"]


def test_a_config_passed_by_the_caller_replaces_the_repositorys_config_and_is_not_merged_with_it(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    commit_files(
        repository,
        {
            "a.ts": "export function a() { return 1 }\n",
            "sgconfig.yml": "languageGlobs:\n  javascript: ['*.ts']\n",
        },
    )
    unrelated_remapping = "languageGlobs:\n  json: ['*.nothing']\n"

    matches = list(
        tools.ast_grep_rules(FUNCTION_RULE, ["a.ts"], repository, config=unrelated_remapping, refused={})
    )

    assert [match["file"] for match in matches] == ["a.ts"]
