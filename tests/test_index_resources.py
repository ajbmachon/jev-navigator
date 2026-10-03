"""What the index costs to run: parser memory, processes started, and what stays cached."""

from __future__ import annotations

import subprocess
import sys
import tracemalloc
from collections import Counter
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.index import code_index, tools
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.scope_scan import Unparsed, scan_facts

MIXED_SCOPE = {
    "app/orders.py": "from app.rules import check\n\n\ndef place(order):\n    return check(order)\n",
    "app/rules.py": "LIMIT = 3\n\n\ndef check(order):\n    return len(order) <= LIMIT\n",
    "web/routes.ts": "import { handle } from './handle';\nexport function routes(app) {\n"
    "  app.post('/x', (req) => handle(req));\n}\n",
    "web/handle.ts": "export const handle = (req) => new Reply(req.body);\nclass Reply {}\n",
    "web/typed.js": "// @flow\nexport function typed(value: string): string {\n  return value.trim();\n}\n",
    "web/plain.js": "export function plain(value) {\n  return value;\n}\n",
}


def nested_callbacks(depth: int, filler_lines: int) -> str:
    """Each level wraps everything inside it in a call, so ast-grep repeats the inner text once per
    level, as callback-heavy TypeScript does. Filler comments keep facts sparse per byte."""
    filler = "".join(f"  // filler line {n} keeps the facts sparse per byte\n" for n in range(filler_lines))
    source = "export function inner() { return 1; }\n"
    for level in range(depth):
        source = f"run{level}(() => {{\n{filler}{source}}});\n"
    return source


def callback_tree(root: Path, file_count: int) -> tuple[list[str], int]:
    root.mkdir()
    text = nested_callbacks(depth=12, filler_lines=40)
    files = [f"m{n}.ts" for n in range(file_count)]
    for name in files:
        (root / name).write_text(text)
    return files, file_count * len(text.encode())


def peak_bytes_while_scanning(root: Path, files: list[str]) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    scan_facts(files, root, Unparsed())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def scanned(root: Path) -> tuple[dict, frozenset[str]]:
    unparsed = Unparsed()
    facts = scan_facts(sorted(MIXED_SCOPE), root, unparsed)
    return facts, unparsed.files


def test_fact_scan_memory_grows_by_less_than_the_added_source(tmp_path: Path) -> None:
    # Arrange: two scopes of identical files, the second eight times larger.
    small_files, small_bytes = callback_tree(tmp_path / "small", file_count=4)
    large_files, large_bytes = callback_tree(tmp_path / "large", file_count=32)

    # Act
    small_peak = peak_bytes_while_scanning(tmp_path / "small", small_files)
    large_peak = peak_bytes_while_scanning(tmp_path / "large", large_files)

    # Assert: the extra memory is less than the extra source, so it cannot scale with the JSON.
    assert large_peak - small_peak < large_bytes - small_bytes


def test_scanning_one_file_per_parser_process_gives_the_same_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spawned: Counter[str]
) -> None:
    # Arrange
    commit_files(tmp_path, MIXED_SCOPE)
    together, together_unparsed = scanned(tmp_path)
    spawned.clear()
    monkeypatch.setattr(tools, "MAX_FILES_PER_COMMAND", 1)

    # Act
    one_by_one, one_by_one_unparsed = scanned(tmp_path)

    # Assert
    assert spawned[tools.AST_GREP] == len(MIXED_SCOPE)
    assert one_by_one == together
    assert one_by_one_unparsed == together_unparsed
    assert together["web/typed.js"].structure.functions[0].name == "typed"


def test_a_file_list_longer_than_the_argument_limit_is_split_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spawned: Counter[str]
) -> None:
    # Arrange
    commit_files(tmp_path, MIXED_SCOPE)
    files = sorted(MIXED_SCOPE)
    together = (scanned(tmp_path), set(tools.ripgrep_files("return", files, tmp_path)))
    spawned.clear()
    monkeypatch.setattr(tools, "MAX_ARGUMENT_BYTES", 30)

    # Act
    split = (scanned(tmp_path), set(tools.ripgrep_files("return", files, tmp_path)))

    # Assert
    assert spawned[tools.AST_GREP] > 2
    assert spawned[tools.RIPGREP] > 1
    assert split == together


def test_the_line_cache_holds_at_most_its_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(code_index, "LINE_CACHE_FILES", 2)
    commit_files(tmp_path, MIXED_SCOPE)
    index = CodeIndex.from_git(tmp_path, fact_cache_dir=tmp_path.parent / "facts")

    # Act
    line_counts = {file: len(index.lines(file)) for file in index.files}
    again = index.lines("app/orders.py")

    # Assert
    assert index._lines_of.cache_info().currsize == 2
    assert line_counts["app/orders.py"] == len(again) == 5


def test_a_repeated_text_search_starts_no_second_process(sample_index: CodeIndex, spawned) -> None:
    # Act
    first = sample_index.search_text("orders.max_items")
    second = sample_index.search_text("orders.max_items")

    # Assert
    assert second == first
    assert spawned[tools.RIPGREP] == 1


def test_co_changed_files_reads_history_once_per_file(sample_index: CodeIndex, spawned) -> None:
    # Act
    first = sample_index.co_changed_files("app/orders.py")
    narrower = sample_index.co_changed_files("app/orders.py", limit=1)

    # Assert
    assert narrower == first[:1]
    assert spawned["git log"] == 1


def test_prefetched_names_share_one_search_and_find_what_lone_lookups_find(
    sample_repo: Path, spawned: Counter[str]
) -> None:
    # Arrange
    names = ("validate_order", "check_limits", "handleOrder", "parseOrder", "absent_name")
    alone = CodeIndex.from_git(sample_repo, fact_cache_dir=sample_repo.parent / "alone")
    expected = {name: (alone.find_definition(name), alone.call_site_count(name)) for name in names}
    index = CodeIndex.from_git(sample_repo, fact_cache_dir=sample_repo.parent / "prefetched")
    spawned.clear()

    # Act
    index.prefetch_names(names)
    searches_for_prefetch = spawned[tools.RIPGREP]
    found = {name: (index.find_definition(name), index.call_site_count(name)) for name in names}

    # Assert
    assert searches_for_prefetch == 1
    assert spawned[tools.RIPGREP] == 1
    assert found == expected


def test_a_file_removed_after_inventory_is_reported_when_a_search_meets_it(sample_repo: Path) -> None:
    # Arrange
    index = CodeIndex.from_git(sample_repo, fact_cache_dir=sample_repo.parent / "facts")
    (sample_repo / "app/settings.py").unlink()

    # Act
    hits = index.search_text("orders.max_items")
    definitions = index.find_definition("check_limits")

    # Assert
    assert [hit.file for hit in hits] == ["app/validation.py", "app/validation.py"]
    assert [span.file for span in definitions] == ["app/validation.py"]
    assert "app/settings.py" in index.unavailable_files


def test_facts_on_one_line_are_ordered_by_position_then_name(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "chain.ts").write_text(
        "export const o = { b() { return 1; }, a() { return 2; } };\n"
        "outer(middle(inner(1)), new Box(2)).then(done);\n"
    )

    # Act: a fresh parse each time, since ast-grep may print matches in any order.
    runs = [scan_facts(["chain.ts"], tmp_path, Unparsed())["chain.ts"] for _ in range(5)]

    # Assert
    assert all(run == runs[0] for run in runs)
    assert [span.name for span in runs[0].structure.functions] == ["a", "b"]
    assert [call.name for call in runs[0].calls if call.line == 2] == [
        "outer",
        "then",
        "middle",
        "inner",
        "Box",
    ]


def test_listing_callees_searches_once_for_every_name_called(tmp_path: Path, spawned: Counter[str]) -> None:
    # Arrange
    commit_files(
        tmp_path,
        {
            "app/main.py": "def main():\n    first()\n    second()\n    third()\n",
            "app/steps.py": "def first():\n    return 1\n\n\ndef second():\n    return 2\n",
            "app/other.py": "def third():\n    return 3\n",
        },
    )
    index = CodeIndex.from_git(tmp_path, fact_cache_dir=tmp_path.parent / "facts")
    main = index.functions_in("app/main.py")[0]
    spawned.clear()

    # Act
    edges = index.callee_edges(main)
    definitions = [index.find_definition(edge.name)[0].file for edge in edges]

    # Assert
    assert [edge.name for edge in edges] == ["first", "second", "third"]
    assert definitions == ["app/steps.py", "app/steps.py", "app/other.py"]
    assert spawned[tools.RIPGREP] == 1


def test_an_index_built_in_a_test_never_writes_the_user_fact_cache(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "module.py").write_text("def run():\n    return 1\n")
    user_cache = Path.home() / ".cache"

    # Act
    index = CodeIndex(tmp_path, ["module.py"])
    index.functions_in("module.py")

    # Assert
    assert not index._fact_cache.root.is_relative_to(user_cache)
    assert list(index._fact_cache.root.rglob("*.json"))


def test_a_file_changed_after_its_lines_were_evicted_is_reported_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(code_index, "LINE_CACHE_FILES", 1)
    (tmp_path / "a.py").write_text("def a():\n    return 1\n")
    (tmp_path / "b.py").write_text("def b():\n    return 2\n")
    index = CodeIndex(tmp_path, ["a.py", "b.py"])
    first = index.read_slice(index.functions_in("a.py")[0])
    index.lines("b.py")
    (tmp_path / "a.py").write_text("def a():\n    return 'changed'\n")

    # Act
    again = index.read_slice(first.span)

    # Assert
    assert "changed" not in again.text
    assert again.file_sha256 == first.file_sha256
    assert "changed" in index.unavailable_files["a.py"]


def test_lines_split_where_the_parser_counts_them_even_after_a_lone_carriage_return(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "module.py").write_bytes(b"first = 1\rsecond = 2\ndef run():\n    return 1\n")
    index = CodeIndex(tmp_path, ["module.py"])

    # Act
    run = index.functions_in("module.py")[0]

    # Assert
    assert run.start == 2
    assert index.read_slice(run).text.splitlines()[0] == "def run():"


def test_prefetching_more_names_than_a_command_line_holds_still_searches_once(
    sample_repo: Path, spawned: Counter[str]
) -> None:
    # Arrange: about 1.3 MB of names, over the 1 MB macOS argument limit.
    names = [f"name_that_appears_nowhere_{n:08d}" for n in range(36_000)] + ["check_limits"]
    index = CodeIndex.from_git(sample_repo, fact_cache_dir=sample_repo.parent / "facts")
    spawned.clear()

    # Act
    index.prefetch_names(names)
    definitions = index.find_definition("check_limits")

    # Assert
    assert spawned[tools.RIPGREP] == 1
    assert [span.file for span in definitions] == ["app/validation.py"]


def test_a_jvn_process_started_by_a_test_uses_the_test_fact_cache_too() -> None:
    # Act
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from jev_navigator.index.fact_cache import FactCache; print(FactCache().root)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # Assert
    assert not Path(child.stdout.strip()).is_relative_to(Path.home() / ".cache")


def test_the_scan_and_the_fact_cache_agree_that_a_file_is_flow(tmp_path: Path) -> None:
    # Arrange: the pragma sits below a long comment header.
    header = [f"// header line {n}" for n in range(40)]
    body = ["// @flow", "export function typed(value: string): string { return value; }"]
    (tmp_path / "typed.js").write_text("\n".join([*header, *body]) + "\n")
    cache_root = tmp_path / "facts"
    index = CodeIndex(tmp_path, ["typed.js"], fact_cache_dir=cache_root)

    # Act
    names = [span.name for span in index.functions_in("typed.js")]

    # Assert: the tsx grammar read the type annotations, and the facts are cached as flow.
    assert names == ["typed"]
    assert "typed.js" not in index.observed_unparsed_files
    assert [path.parent.name for path in cache_root.rglob("*.json")] == ["flow"]


def test_a_file_saved_while_the_parser_runs_is_reported_and_its_facts_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the editor saves the file after the index read it and before ast-grep parses it.
    policy = "def admit(item):\n    return item\n\n\ndef deny(item):\n    return None\n"
    repository, cache_root = tmp_path / "repository", tmp_path / "facts"
    commit_files(repository, {"app/policy.py": policy})
    index = CodeIndex.from_git(repository, fact_cache_dir=cache_root)
    real_rules = tools.ast_grep_rules

    def rules_while_an_editor_saves(*arguments, **options):
        matches = real_rules(*arguments, **options)
        (repository / "app/policy.py").write_text("# saved by an editor\n" + policy)
        yield from matches

    monkeypatch.setattr(tools, "ast_grep_rules", rules_while_an_editor_saves)

    # Act
    definitions = index.find_definition("deny")

    # Assert
    assert definitions == ()
    assert "changed" in index.unavailable_files["app/policy.py"]
    assert not list(cache_root.rglob("*.json"))


def test_a_declaration_on_a_first_line_after_a_byte_order_mark_keeps_its_name(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "settings.py").write_bytes("\ufeffLIMIT = 3\nOTHER = 4\n".encode())
    (tmp_path / "flags.ts").write_bytes("\ufeffexport const enabled = true;\n".encode())

    # Act
    facts = scan_facts(["settings.py", "flags.ts"], tmp_path, Unparsed())

    # Assert
    assert [span.name for span in facts["settings.py"].structure.declarations] == ["LIMIT", "OTHER"]
    assert [span.name for span in facts["flags.ts"].structure.declarations] == ["enabled"]
