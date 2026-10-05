from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from git_repos import commit_all, commit_files, write_files

from jev_navigator.cli_statistics import create_statistics_pack
from jev_navigator.comments import find_comments
from jev_navigator.index import file_shape, tools
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

        def run(arguments, cwd, **callbacks):
            if arguments[0] == tools.AST_GREP and "scan" in arguments:
                self.commands.append(list(arguments))
                return iter(())
            return original(arguments, cwd, **callbacks)

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
    reason = index.unavailable_files[BUNDLE]
    assert reason.startswith("too large to parse: estimated parse peak ")
    assert reason.endswith(", 1 line, longest line 668,777 bytes")
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


GUARDED_EXPORTS = "vendor/bundle.js"


@pytest.mark.parametrize(
    ("importer", "name"),
    [
        ("import build from '../vendor/bundle.js';\nbuild();\n", "build"),
        ("const { start } = require('../vendor/bundle.js');\nstart();\n", "start"),
        ("import { start } from './barrel.js';\nstart();\n", "start"),
    ],
    ids=["default import", "named require", "through a barrel"],
)
def test_a_name_imported_from_a_guarded_file_binds_unknown_where_its_bytes_never_say_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, importer: str, name: str
) -> None:
    # Arrange: the guarded file's one line exports a default and never says `default` or `start`
    _AstGrepRecorder(monkeypatch)
    statement = "module.exports=build;function build(){return 1};"
    files = {
        GUARDED_EXPORTS: (statement * (668_777 // len(statement) + 1))[:668_777],
        "src/barrel.js": "export * from '../vendor/bundle.js';\n",
        "src/importer.js": importer,
    }
    commit_files(tmp_path / "repo", files)
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")

    # Act
    binding = index.binding_of("src/importer.js", 2, name, None)

    # Assert
    assert GUARDED_EXPORTS in index.refused_files
    assert binding.status == BindingStatus.UNKNOWN
    assert GUARDED_EXPORTS in binding.reason


def test_comment_scanning_never_parses_a_guarded_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ast_grep = _AstGrepRecorder(monkeypatch)
    index = CodeIndex.from_git(_repository(tmp_path, 668_777), fact_cache_dir=tmp_path / "facts")

    found = find_comments(index)

    assert not ast_grep.received(BUNDLE)
    assert found.refused_files[BUNDLE].startswith("too large to parse: estimated parse peak ")


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


SKIPPED_BY_AST_GREP = "src/skipped.ts"
"""ast-grep 0.45.1 exits 0 and prints nothing for a file of more than 3,000,000 bytes and more than
200,000 lines, even for a rule on ``kind: program``. This one has 1,000,001 lines of 3 bytes."""


def _repository_with_a_file_ast_grep_skips(tmp_path: Path) -> Path:
    files = {
        SKIPPED_BY_AST_GREP: "x;\n" * 1_000_001,
        "src/comment_only.ts": "// nothing to find here\n",
        "src/empty.ts": "",
        "src/small.ts": "export function small() { return 1; }\n",
    }
    commit_files(tmp_path / "repo", files)
    return tmp_path / "repo"


FUNCTION_RULE = "id: function\nlanguage: typescript\nrule:\n  kind: function_declaration"


def _without_the_memory_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests reach ast-grep's own size skip (over 3,000,000 bytes and 200,000 lines) with a file the
    side-by-side share would refuse, so they lift that share: what they prove is the detection of any
    file ast-grep skips in a side-by-side run."""
    monkeypatch.setattr(file_shape, "MAX_PARSE_PEAK_MB", float("inf"))


def test_the_door_names_a_file_ast_grep_skipped_without_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_the_memory_bound(monkeypatch)
    repository = _repository_with_a_file_ast_grep_skips(tmp_path)
    refused: dict[str, str] = {}

    matches = list(
        tools.ast_grep_rules(
            FUNCTION_RULE,
            [SKIPPED_BY_AST_GREP, "src/comment_only.ts", "src/empty.ts", "src/small.ts"],
            repository,
            refused=refused,
        )
    )

    assert [match["file"] for match in matches] == ["src/small.ts"]
    assert set(refused) == {SKIPPED_BY_AST_GREP}, (
        "a parsed file with no match, or an empty one, is not a skipped file"
    )
    assert refused[SKIPPED_BY_AST_GREP].startswith("not parsed: ast-grep skipped the file")


def test_a_file_ast_grep_skipped_is_reported_unavailable_and_never_cached_as_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_the_memory_bound(monkeypatch)
    repository = _repository_with_a_file_ast_grep_skips(tmp_path)
    index = CodeIndex.from_git(repository, fact_cache_dir=tmp_path / "facts")

    index.functions_in_files(index.files)

    assert index.unavailable_files[SKIPPED_BY_AST_GREP].startswith("not parsed: ast-grep skipped the file")
    assert set(index.unavailable_files) == {SKIPPED_BY_AST_GREP}
    cache = FactCache(tmp_path / "facts")
    assert cache.load(SKIPPED_BY_AST_GREP, (repository / SKIPPED_BY_AST_GREP).read_bytes()) is None
    assert cache.load("src/comment_only.ts", (repository / "src/comment_only.ts").read_bytes()) is not None


def test_a_source_file_that_is_not_valid_utf8_is_named_as_such_not_taken_for_empty(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    write_files(repository, {"src/small.ts": "export function small() { return 1; }\n"})
    (repository / "src/latin.ts").write_bytes("export function caf\u00e9() { return 1 }\n".encode("latin-1"))
    commit_all(repository)
    refused: dict[str, str] = {}

    matches = list(
        tools.ast_grep_rules(FUNCTION_RULE, ["src/latin.ts", "src/small.ts"], repository, refused=refused)
    )

    assert [match["file"] for match in matches] == ["src/small.ts"]
    assert refused == {"src/latin.ts": "not parsed: not valid UTF-8"}


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


def test_a_component_holding_one_huge_image_string_is_parsed_not_refused(tmp_path: Path) -> None:
    image = "A" * 2_500_000
    commit_files(
        tmp_path / "repo",
        {"background.tsx": f"export function Background() {{\n  return <img src='{image}' />;\n}}\n"},
    )
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")

    functions = index.functions_in("background.tsx")

    assert [span.name for span in functions] == ["Background"]
    assert index.unavailable_files == {}


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


SINGLE_PARSE_LIMIT_MB = 754
"""The single-file limit at JVN's default memory settings: the allowance less Python's share."""
REAL_CODE = Path(tools.__file__).with_name("code_index.py").read_text()


def _real_code_file(size_bytes: int) -> str:
    """JVN's own code_index.py repeated to ``size_bytes``: real code of ordinary short lines."""
    copies = size_bytes // len(REAL_CODE) + 1
    return REAL_CODE * copies


def _with_single_parse_limit(monkeypatch: pytest.MonkeyPatch, megabytes: float) -> None:
    monkeypatch.setattr(tools, "single_parse_limit_mb", lambda: megabytes)


class _AstGrepCommands:
    """Records the file list of every ast-grep scan and lets the real ast-grep run it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.scans: list[list[str]] = []
        real = tools._json_lines

        def recorded(arguments, *rest, **options):
            if arguments[0] == tools.AST_GREP and "scan" in arguments:
                self.scans.append(list(arguments))
            return real(arguments, *rest, **options)

        monkeypatch.setattr(tools, "_json_lines", recorded)

    def scan_of(self, file: str) -> list[str]:
        (scan,) = [scan for scan in self.scans if file in scan]
        return scan


def test_a_file_over_the_side_by_side_share_is_parsed_alone_and_its_facts_are_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: 4.7 MB of real code is over the 250 MB side-by-side share and under the single-file limit
    _with_single_parse_limit(monkeypatch, SINGLE_PARSE_LIMIT_MB)
    ast_grep = _AstGrepCommands(monkeypatch)
    big = _real_code_file(4_700_000)
    commit_files(
        tmp_path / "repo",
        {"src/big.py": big, "src/one.py": REAL_CODE, "src/small.py": "def small():\n    return 1\n"},
    )
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")

    # Act
    index.functions_in_files(index.files)

    # Assert
    alone = ast_grep.scan_of("src/big.py")
    assert alone[alone.index("--") + 1 :] == ["src/big.py"]
    assert alone[alone.index("--threads") + 1] == "1"
    assert ast_grep.scan_of("src/small.py") == ast_grep.scan_of("src/one.py")
    copies = big.count(REAL_CODE)
    assert len(index.functions_in("src/big.py")) == copies * len(index.functions_in("src/one.py"))
    assert index.refused_files == {}


def test_a_file_over_the_single_file_limit_is_refused_and_names_that_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: 55,000 dense lines of 45 operands, 14.7 MB, really peaked at 909 MB alone
    _with_single_parse_limit(monkeypatch, SINGLE_PARSE_LIMIT_MB)
    ast_grep = _AstGrepCommands(monkeypatch)
    dense_line = "v = " + " + ".join(f"a{operand}" for operand in range(45)) + "\n"
    commit_files(
        tmp_path / "repo",
        {"gen/dense_ops.py": dense_line * 55_000, "src/small.py": "def small():\n    return 1\n"},
    )
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")

    # Act
    index.functions_in_files(index.files)

    # Assert
    assert not any("gen/dense_ops.py" in scan for scan in ast_grep.scans)
    reason = index.refused_files["gen/dense_ops.py"]
    assert reason.startswith(
        "too large to parse: estimated parse peak 1 GB, over the 754 MB one file may take"
    )


def test_a_long_string_of_data_is_still_parsed_side_by_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: documenso's background.tsx holds one 2.5 MB SVG path on one line and peaks at 58 MB
    _with_single_parse_limit(monkeypatch, SINGLE_PARSE_LIMIT_MB)
    ast_grep = _AstGrepCommands(monkeypatch)
    path = "M708 195.8c.4-1.5.8-3.5 2-4.7 " * 84_000
    background = f'export function Background() {{\n  return <path d="{path}" />;\n}}\n'
    commit_files(
        tmp_path / "repo",
        {"src/background.tsx": background, "src/small.tsx": "export function small() { return 1; }\n"},
    )
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")

    # Act
    index.functions_in_files(index.files)

    # Assert
    assert ast_grep.scan_of("src/background.tsx") == ast_grep.scan_of("src/small.tsx")
    assert [span.name for span in index.functions_in("src/background.tsx")] == ["Background"]


ALONE_IN_A_FRESH_PROCESS = """
import json, resource, sys
from pathlib import Path
from jev_navigator.index import tools
from jev_navigator.index.code_index import CodeIndex
tools.single_parse_limit_mb = lambda: float(sys.argv[3])
index = CodeIndex.from_git(Path(sys.argv[1]), fact_cache_dir=Path(sys.argv[2]))
functions = len(index.functions_in("src/big.py"))
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(json.dumps({"peak_bytes": peak, "functions": functions, "refused": index.refused_files}))
"""


def test_a_file_parsed_alone_peaks_inside_the_single_file_limit(tmp_path: Path) -> None:
    # Arrange
    commit_files(tmp_path / "repo", {"src/big.py": _real_code_file(4_700_000)})

    # Act
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            ALONE_IN_A_FRESH_PROCESS,
            str(tmp_path / "repo"),
            str(tmp_path / "facts"),
            str(SINGLE_PARSE_LIMIT_MB),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # Assert
    report = json.loads(completed.stdout)
    peak_megabytes = report["peak_bytes"] / (1 if sys.platform == "darwin" else 1024) / 1_000_000
    assert report["refused"] == {} and report["functions"] > 0
    assert peak_megabytes < SINGLE_PARSE_LIMIT_MB


def test_without_a_single_file_limit_a_file_over_the_side_by_side_share_is_refused(tmp_path: Path) -> None:
    # Arrange: no memory settings name a single-file limit, so nothing may be parsed alone
    commit_files(tmp_path / "repo", {"src/big.py": _real_code_file(4_700_000)})
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")

    # Act
    index.functions_in_files(index.files)

    # Assert
    assert ", over the 250 MB one file may take," in index.refused_files["src/big.py"]


def test_a_file_parsed_alone_that_ast_grep_skips_is_named_not_taken_for_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: 110,000 small functions, 3.4 MB on 220,000 lines, over ast-grep's own size skip but within
    # the single-file limit, so the file is parsed alone and ast-grep prints nothing for it
    _with_single_parse_limit(monkeypatch, SINGLE_PARSE_LIMIT_MB)
    many = "".join(f"def f{n}():\n    return {n}\n" for n in range(110_000))
    commit_files(tmp_path / "repo", {"src/many.py": many, "src/one.py": "def one():\n    return 1\n"})
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")

    # Act
    index.functions_in_files(index.files)

    # Assert
    assert index.refused_files == {"src/many.py": tools.NOT_PARSED_REASON}
    assert [span.name for span in index.functions_in("src/one.py")] == ["one"]
