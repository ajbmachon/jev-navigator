from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.cli_statistics import create_statistics_pack
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

        def run(arguments, cwd, decode):
            if arguments[0] == tools.AST_GREP and "scan" in arguments:
                self.commands.append(list(arguments))
                return iter(())
            return original(arguments, cwd, decode)

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
        "too large to parse: estimated parse peak 22 GB, longest line 668,777 bytes"
    )
    assert index.parser_scans_pending == ()


def test_a_refused_file_is_never_counted_as_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _AstGrepRecorder(monkeypatch)
    index = CodeIndex.from_git(_repository(tmp_path, 668_777), fact_cache_dir=tmp_path / "facts")

    index.functions_in_files(index.files)

    assert BUNDLE in index.unavailable_files
    assert BUNDLE not in index.parsed_files
    assert "src/small.py" in index.parsed_files


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


def test_comment_scanning_never_reads_the_repositorys_own_sgconfig(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    commit_files(
        repository,
        {
            "a.ts": "// explains why admit exists\nexport function admit() { return 1 }\n",
            "sgconfig.yml": (
                "customLanguages:\n  mylang:\n    libraryPath: ./missing.so\n    extensions: [my]\n"
            ),
        },
    )
    index = CodeIndex.from_git(repository, fact_cache_dir=tmp_path / "facts")

    found = find_comments(index)

    assert found.refused_files == {}
    assert [block.span.file for block in found.kept] == ["a.ts"]


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


REFUSED_CHARACTERS = 200_000


def _bundle_defining(name: str, extra: str = "") -> str:
    statement = f"export function {name}(){{return 1}};"
    return (extra + statement * (REFUSED_CHARACTERS // len(statement) + 1))[:REFUSED_CHARACTERS]


def test_stats_name_a_refused_file_as_never_scanned_with_its_reason(tmp_path: Path) -> None:
    # Arrange
    repository = _repository(tmp_path, REFUSED_CHARACTERS)
    index = CodeIndex.from_git(repository, fact_cache_dir=tmp_path / "facts")

    # Act
    pack = create_statistics_pack(repository, (), tmp_path / "pack", index=index)

    # Assert
    report = (tmp_path / "pack" / "statistics.md").read_text()
    assert BUNDLE not in pack["scope"]["measured"]
    assert BUNDLE in pack["scope"]["unmeasured"]
    assert pack["coverage"]["complete"] is False
    assert BUNDLE not in pack["counts"]["per_file"]
    assert f"`{BUNDLE}` ({pack['coverage']['unavailable'][BUNDLE]})" in report
    assert "Fully covered" not in report


def test_a_name_re_exported_through_a_refused_file_binds_unknown_and_never_follows_a_decoy(
    tmp_path: Path,
) -> None:
    # Arrange: the barrel re-exports everything of a refused bundle; an unrelated file defines the name too
    files = {
        "lib/bundle.js": _bundle_defining("charge"),
        "lib/index.js": "export * from './bundle.js';\n",
        "tools/legacy.js": "function charge() {\n  return 2;\n}\n",
        "app/pay.js": "import { charge } from '../lib/index.js';\ncharge();\n",
    }
    commit_files(tmp_path / "repo", files)
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")

    # Act
    binding = index.binding_of("app/pay.js", 2, "charge", None)

    # Assert
    assert binding.status == BindingStatus.UNKNOWN
    assert "lib/bundle.js" in binding.reason
    assert binding.target is None


def test_text_search_finds_a_refused_file_whether_or_not_a_scan_ran_first(tmp_path: Path) -> None:
    # Arrange
    files = {
        "dist/bundle.mjs": _bundle_defining("consume", extra="const queue='orders.payment_queue';"),
        "src/reader.py": "def read():\n    return 'orders.payment_queue'\n",
    }
    commit_files(tmp_path / "repo", files)
    before_scan = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")
    after_scan = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")
    after_scan.functions_in_files(after_scan.files)

    # Act
    found_before = before_scan.search_text("orders.payment_queue")
    found_after = after_scan.search_text("orders.payment_queue")

    # Assert
    assert [(hit.file, hit.line) for hit in found_after] == [("dist/bundle.mjs", 1), ("src/reader.py", 2)]
    assert found_after == found_before


def test_a_refused_file_hides_only_the_names_its_bytes_mention(tmp_path: Path) -> None:
    # Arrange
    files = {
        "dist/bundle.mjs": _bundle_defining("consume"),
        "src/pay.py": "def pay():\n    refund()\n    consume()\n",
    }
    commit_files(tmp_path / "repo", files)
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")
    index.functions_in_files(index.files)

    # Act
    unmentioned = index.binding_of("src/pay.py", 2, "refund", None)
    mentioned = index.binding_of("src/pay.py", 3, "consume", None)

    # Assert
    assert unmentioned.status == BindingStatus.UNRESOLVED
    assert mentioned.status == BindingStatus.UNKNOWN
    assert "dist/bundle.mjs" in mentioned.reason


def test_a_refused_file_is_reported_refused_and_never_as_parsed_or_unreadable(tmp_path: Path) -> None:
    # Arrange
    index = CodeIndex.from_git(_repository(tmp_path, REFUSED_CHARACTERS), fact_cache_dir=tmp_path / "facts")

    # Act
    index.functions_in_files(index.files)

    # Assert
    assert set(index.refused_files) == {BUNDLE}
    assert index.unavailable_files[BUNDLE] == index.refused_files[BUNDLE]
    assert BUNDLE in index.available_files
    assert index.parser_scans_pending == ()


def test_a_large_file_that_cannot_be_measured_is_refused_with_the_error(tmp_path: Path) -> None:
    # Arrange
    repository = _repository(tmp_path, REFUSED_CHARACTERS)
    (repository / BUNDLE).chmod(0)
    refused: dict[str, str] = {}

    # Act
    try:
        matches = list(tools.ast_grep_rules(FUNCTION_RULE, [BUNDLE], repository, refused=refused))
    finally:
        (repository / BUNDLE).chmod(0o644)

    # Assert
    assert matches == []
    assert refused[BUNDLE].startswith("could not be measured: PermissionError")


def test_a_refused_file_is_measured_once_for_the_life_of_the_index(tmp_path: Path) -> None:
    # Arrange
    scans: list[tuple[str, str, int]] = []
    index = CodeIndex.from_git(
        _repository(tmp_path, REFUSED_CHARACTERS),
        fact_cache_dir=tmp_path / "facts",
        scan_observer=lambda *event: scans.append(event),
    )
    index.functions_in(BUNDLE)

    # Act
    index.functions_in(BUNDLE)
    index.functions_in_files([BUNDLE])

    # Assert
    assert [event for event in scans if event[1] == "started"] == [("facts", "started", 1)]
