"""The persistent name table: a new index answers name lookups from it, without searching or parsing."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import sqlite3
import threading
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

import pytest
from git_repos import commit_files, git

from jev_navigator.confirmation import day_of, today
from jev_navigator.index import fact_cache, languages, name_table, scope_scan, tools
from jev_navigator.index.code_index import CodeIndex

REPOSITORY = {
    "app/rules.py": "LIMIT = 3\n\n\ndef check(order):\n    return len(order) <= LIMIT\n",
    "app/orders.py": "from app.rules import LIMIT, check\n\n\ndef place(order):\n"
    "    return check(order) and run(order, LIMIT)\n\n\ndef run(order, limit):\n    return order\n",
    "web/handle.ts": 'export const handle = (req) => fetch("https://example.test/hidden-path").then(check);\n'
    "export function check(x) {\n  return x;\n}\n",
    "web/store.ts": "export class Store {\n  save(row) {\n    return this.check(row);\n  }\n"
    "  check(row) {\n    return row;\n  }\n}\n",
}
NAMES = ("check", "LIMIT", "place", "run", "handle", "then", "save", "absent")


def every_lookup(index: CodeIndex) -> dict[str, tuple]:
    """Each name's definitions, callers, call count and references, and each file's imports."""
    names = {
        name: (
            index.find_definition(name),
            index.find_callers(name),
            index.call_site_count(name),
            index.find_references(name),
        )
        for name in NAMES
    }
    return {**names, "imports": tuple(index.imports(file) for file in index.available_files)}


def parsed_files(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every file ast-grep is handed from now on; ast-grep still runs."""
    parsed: list[str] = []
    real_rules = tools.ast_grep_rules

    def recorded_rules(rules, files, *arguments, **options):
        parsed.extend(files)
        return real_rules(rules, files, *arguments, **options)

    monkeypatch.setattr(tools, "ast_grep_rules", recorded_rules)
    return parsed


def table_rows(cache: Path, table: str = "names") -> list[tuple]:
    rows = []
    for path in sorted((cache / "names").glob("*.sqlite")):
        with sqlite3.connect(path) as database:
            rows += database.execute(f"select * from {table}").fetchall()
    return rows


def answered_from_scratch(root: Path, monkeypatch: pytest.MonkeyPatch, cache: Path) -> dict[str, tuple]:
    """The lookups of an index with an empty fact cache and an empty name table."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    return every_lookup(CodeIndex.from_git(root))


def test_a_warm_name_lookup_starts_no_text_search_and_parses_no_file(
    tmp_path: Path, spawned: Counter[str]
) -> None:
    # Arrange
    commit_files(tmp_path, REPOSITORY)
    cold = every_lookup(CodeIndex.from_git(tmp_path))
    index = CodeIndex.from_git(tmp_path)
    spawned.clear()

    # Act
    warm = every_lookup(index)

    # Assert
    assert warm == cold
    assert spawned[tools.RIPGREP] == 0
    assert spawned[tools.AST_GREP] == 0


def test_a_changed_file_updates_only_its_own_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_cache_root: Path
) -> None:
    # Arrange
    commit_files(tmp_path, REPOSITORY)
    every_lookup(CodeIndex.from_git(tmp_path))
    files_before = set(table_rows(private_cache_root, "files"))
    (tmp_path / "app/rules.py").write_text("LIMIT = 3\n\n\ndef verify(order):\n    return order\n")
    parsed = parsed_files(monkeypatch)

    # Act
    index = CodeIndex.from_git(tmp_path)
    changed = every_lookup(index)
    verify = index.find_definition("verify")

    # Assert
    assert parsed == ["app/rules.py"]
    assert len(set(table_rows(private_cache_root, "files")) - files_before) == 1
    assert [span.file for span in verify] == ["app/rules.py"]
    assert [span.file for span in changed["check"][0]] == ["web/handle.ts", "web/store.ts"]
    assert changed == answered_from_scratch(tmp_path, monkeypatch, tmp_path.parent / "fresh")


def test_a_rule_change_rebuilds_every_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, private_cache_root: Path
) -> None:
    # Arrange
    commit_files(tmp_path, REPOSITORY)
    every_lookup(CodeIndex.from_git(tmp_path))
    tables_before = set((private_cache_root / "names").glob("*.sqlite"))
    request.addfinalizer(name_table.table_identity.cache_clear)
    request.addfinalizer(fact_cache._rules_identity.cache_clear)
    monkeypatch.setitem(languages.FUNCTION_KINDS, "python", ("function_definition", "lambda"))
    fact_cache._rules_identity.cache_clear()
    name_table.table_identity.cache_clear()

    # Act
    every_lookup(CodeIndex.from_git(tmp_path))

    # Assert
    [rebuilt] = set((private_cache_root / "names").glob("*.sqlite")) - tables_before
    with sqlite3.connect(rebuilt) as database:
        assert database.execute("select count(*) from files").fetchone() == (len(REPOSITORY),)


def test_a_deleted_file_answers_no_lookup_and_is_reported(tmp_path: Path, spawned: Counter[str]) -> None:
    # Arrange
    commit_files(tmp_path, REPOSITORY)
    every_lookup(CodeIndex.from_git(tmp_path))
    index = CodeIndex.from_git(tmp_path)
    (tmp_path / "web/handle.ts").unlink()
    spawned.clear()

    # Act
    definitions = index.find_definition("check")

    # Assert
    assert [span.file for span in definitions] == ["app/rules.py", "web/store.ts"]
    assert index.unavailable_files == {"web/handle.ts": "disappeared after inventory"}
    assert spawned[tools.AST_GREP] == 0


def test_a_file_deleted_after_the_scope_was_covered_is_reported_and_proves_nothing(tmp_path: Path) -> None:
    # Arrange: the table describes the files as listed; handle.ts is deleted after that
    commit_files(tmp_path, REPOSITORY)
    every_lookup(CodeIndex.from_git(tmp_path))
    index = CodeIndex.from_git(tmp_path)
    index.find_definition("check")
    (tmp_path / "web/handle.ts").unlink()

    # Act
    callers = index.find_callers("then")

    # Assert
    assert [(site.file, site.binding.status) for site in callers] == [("web/handle.ts", "unknown")]
    assert index.unavailable_files == {"web/handle.ts": "disappeared after inventory"}


def test_a_warm_run_needs_no_write_lock_on_the_table(tmp_path: Path, private_cache_root: Path) -> None:
    # Arrange: another process holds the table's write lock for the whole warm run
    commit_files(tmp_path, REPOSITORY)
    every_lookup(CodeIndex.from_git(tmp_path))
    [path] = (private_cache_root / "names").glob("*.sqlite")
    writer = sqlite3.connect(path, isolation_level=None)
    writer.execute("begin immediate")
    outcome: list[object] = []

    def warm_run() -> None:
        try:
            index = CodeIndex.from_git(tmp_path)
            outcome.append(every_lookup(index))
            index.functions_in("web/store.ts")
        except Exception as error:  # noqa: BLE001 - the assertion below reports it
            outcome.append(error)

    # Act
    run = threading.Thread(target=warm_run, daemon=True)
    run.start()
    run.join(timeout=10)
    writer.execute("rollback")

    # Assert
    assert not run.is_alive(), "the warm run waited for the table's write lock"
    assert len(outcome) == 1 and isinstance(outcome[0], dict)


def test_a_checkout_that_converts_line_endings_answers_warm_lookups(
    tmp_path: Path, spawned: Counter[str]
) -> None:
    # Arrange: Git stores LF and checks out CRLF, so every working file differs from its blob
    commit_files(tmp_path, {".gitattributes": "* text eol=crlf\n", **REPOSITORY})
    git(tmp_path, "rm", "-q", "--cached", "-r", ".")
    git(tmp_path, "reset", "-q", "--hard")
    assert b"\r\n" in (tmp_path / "app/rules.py").read_bytes()
    assert git(tmp_path, "status", "--porcelain") == ""
    cold = every_lookup(CodeIndex.from_git(tmp_path))
    index = CodeIndex.from_git(tmp_path)
    spawned.clear()

    # Act
    warm = every_lookup(index)

    # Assert
    assert [span.file for span in warm["check"][0]] == ["app/rules.py", "web/handle.ts", "web/store.ts"]
    assert warm == cold
    assert index.unavailable_files == {}
    assert spawned[tools.AST_GREP] == 0


def test_a_checkout_whose_filter_changes_lines_never_lends_its_rows_to_a_plain_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: one commit, checked out plainly and through a filter that adds a first line, so both
    # report the same blob for app/rules.py while their definitions sit on different lines
    plain, filtered = tmp_path / "plain", tmp_path / "filtered"
    commit_files(plain, REPOSITORY)
    git(tmp_path, "clone", "-q", str(plain), str(filtered))
    git(filtered, "config", "filter.header.smudge", "sh -c \"printf '# header\\n'; cat\"")
    git(filtered, "config", "filter.header.clean", "sed 1d")
    (filtered / ".git/info/attributes").write_text("*.py filter=header\n")
    git(filtered, "rm", "-q", "--cached", "-r", ".")
    git(filtered, "reset", "-q", "--hard")
    assert (filtered / "app/rules.py").read_text().startswith("# header\n")
    assert git(filtered, "status", "--porcelain") == ""
    expected = answered_from_scratch(plain, monkeypatch, tmp_path / "fresh-cache")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "shared-cache"))
    every_lookup(CodeIndex.from_git(filtered))

    # Act
    answered = every_lookup(CodeIndex.from_git(plain))

    # Assert
    assert answered == expected


@pytest.mark.parametrize("change", ["rewritten", "deleted"])
def test_a_place_answered_from_the_table_reads_as_its_listed_blob_after_the_file_changes(
    tmp_path: Path, change: str
) -> None:
    # Arrange: alpha's place comes from rows of a.py's listed blob, then a.py changes before any read
    original = "def alpha():\n    return 1\n"
    commit_files(tmp_path, {"a.py": original, "b.py": "def beta():\n    return alpha()\n"})
    CodeIndex.from_git(tmp_path).find_definition("alpha")
    index = CodeIndex.from_git(tmp_path)
    (definition,) = index.find_definition("alpha")
    if change == "rewritten":
        (tmp_path / "a.py").write_text("# a new first line\n# and another\ndef omega():\n    return 2\n")
    else:
        (tmp_path / "a.py").unlink()

    # Act
    code = index.read_slice(definition)

    # Assert
    assert code.text == original.rstrip("\n")
    assert code.file_sha256 == hashlib.sha256(original.encode()).hexdigest()
    assert "a.py" in index.unavailable_files


@pytest.mark.parametrize("lookup", ["find_callers", "find_references"])
def test_a_warm_table_with_an_empty_fact_cache_loads_a_names_facts_in_one_scan(
    tmp_path: Path, spawned: Counter[str], lookup: str
) -> None:
    # Arrange: check is defined twice, and six files call it and pass it on; the table is warm, the
    # facts are not
    uses = {
        f"app/use_{n}.py": f"from app.rules import check\n\n\ndef use_{n}(order):\n"
        "    run(check)\n    return check(order)\n"
        for n in range(6)
    }
    repository = tmp_path / "repository"
    commit_files(repository, {**REPOSITORY, **uses})
    expected = getattr(CodeIndex.from_git(repository, fact_cache_dir=tmp_path / "warm-facts"), lookup)(
        "check"
    )
    index = CodeIndex.from_git(repository, fact_cache_dir=tmp_path / "empty-facts")
    spawned.clear()

    # Act
    found = getattr(index, lookup)("check")

    # Assert
    assert found == expected
    assert spawned[tools.AST_GREP] == 1


def test_a_name_nothing_uses_loads_no_facts_on_a_warm_table(tmp_path: Path, spawned: Counter[str]) -> None:
    # Arrange: place is defined but never called or passed on; the table is warm, the facts are not
    repository = tmp_path / "repository"
    commit_files(repository, REPOSITORY)
    every_lookup(CodeIndex.from_git(repository, fact_cache_dir=tmp_path / "warm-facts"))
    index = CodeIndex.from_git(repository, fact_cache_dir=tmp_path / "empty-facts")
    spawned.clear()

    # Act
    found = (index.find_callers("place"), index.find_references("place"))

    # Assert
    assert found == ((), ())
    assert spawned[tools.AST_GREP] == 0


def test_a_files_definitions_come_from_the_table_with_their_lines(
    tmp_path: Path, spawned: Counter[str]
) -> None:
    # Arrange
    commit_files(tmp_path, REPOSITORY)
    parsed = CodeIndex.from_git(tmp_path)
    expected = {
        file: tuple(
            span
            for span in dict.fromkeys((*parsed.symbols_in(file), *parsed.declarations_in(file)))
            if span.name != "<anonymous>"
        )
        for file in REPOSITORY
    }
    every_lookup(parsed)
    index = CodeIndex.from_git(tmp_path)
    spawned.clear()

    # Act
    found = {file: index.definitions_in(file) for file in REPOSITORY}

    # Assert
    assert found == expected
    assert [span.name for span in found["web/store.ts"]] == ["Store", "save", "check"]
    assert spawned[tools.AST_GREP] == 0 and spawned[tools.RIPGREP] == 0


@pytest.mark.parametrize("build", [CodeIndex.from_git, CodeIndex.from_directory])
@pytest.mark.parametrize("staged_first", [False, True])
def test_rows_follow_the_working_tree_not_the_staged_blob(tmp_path: Path, build, staged_first: bool) -> None:
    # Arrange: rules.py is edited in the working tree, after an earlier staged edit in one case
    commit_files(tmp_path, REPOSITORY)
    every_lookup(build(tmp_path))
    if staged_first:
        (tmp_path / "app/rules.py").write_text("def staged_only(order):\n    return order\n")
        git(tmp_path, "add", "app/rules.py")
        every_lookup(build(tmp_path))
    (tmp_path / "app/rules.py").write_text("def in_working_tree(order):\n    return order\n")

    # Act
    index = build(tmp_path)
    found = {name: index.find_definition(name) for name in ("in_working_tree", "staged_only", "check")}

    # Assert
    assert [span.file for span in found["in_working_tree"]] == ["app/rules.py"]
    assert found["staged_only"] == ()
    assert "app/rules.py" not in {span.file for span in found["check"]}
    assert index.unavailable_files == {}


def test_no_table_row_holds_a_string_literal(tmp_path: Path, private_cache_root: Path) -> None:
    # Arrange
    commit_files(tmp_path, REPOSITORY)

    # Act
    every_lookup(CodeIndex.from_git(tmp_path))

    # Assert
    stored = " ".join(str(value) for row in table_rows(private_cache_root) for value in row)
    assert table_rows(private_cache_root)
    assert "hidden-path" not in stored
    assert "example.test" not in stored
    assert '"' not in stored and "'" not in stored


def _look_up_after(barrier, root: str, answer: str) -> None:
    Path(answer).parent.mkdir(parents=True, exist_ok=True)
    barrier.wait()
    Path(answer).write_text(repr(every_lookup(CodeIndex.from_git(Path(root)))))


def build_at_once(repository: Path, folder: Path, writers: int) -> list[tuple[int | None, str]]:
    """Each writer's exit code and answers, all of them indexing ``repository`` from one barrier."""
    context = multiprocessing.get_context("spawn")
    barrier, answers = context.Barrier(writers), [folder / f"answer-{n}.txt" for n in range(writers)]
    processes = [
        context.Process(target=_look_up_after, args=(barrier, str(repository), str(answer)))
        for answer in answers
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
    return [
        (process.exitcode, answer.read_text() if answer.exists() else "")
        for process, answer in zip(processes, answers, strict=True)
    ]


def test_processes_building_the_table_at_once_leave_it_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: five rounds, each racing four writers on an empty cache, so a race shows in one of them
    repository = tmp_path / "repository"
    commit_files(repository, REPOSITORY)
    expected = repr(every_lookup(CodeIndex.from_git(repository)))
    rounds = []

    # Act
    for round_number in range(5):
        cache = tmp_path / f"cache-{round_number}"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        rounds.append((cache, build_at_once(repository, tmp_path / f"answers-{round_number}", writers=4)))

    # Assert
    for cache, outcomes in rounds:
        assert outcomes == [(0, expected)] * 4
        rows = table_rows(cache / "jev-navigator")
        assert rows and len(rows) == len(set(rows))
        for path in (cache / "jev-navigator" / "names").glob("*.sqlite"):
            with sqlite3.connect(path) as database:
                assert database.execute("pragma integrity_check").fetchone() == ("ok",)


def test_a_cold_scope_writes_its_rows_in_a_few_transactions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: 450 file contents, each waiting for a disk sync when its transaction commits
    files = {f"app/m{n}.py": f"def f{n}():\n    return {n}\n" for n in range(450)}
    commit_files(tmp_path, files)
    commits: list[str] = []
    real_open = name_table.open_shared_database

    def traced_open(*arguments):
        database = real_open(*arguments)
        database.set_trace_callback(
            lambda statement: commits.append(statement) if statement == "COMMIT" else None
        )
        return database

    monkeypatch.setattr(name_table, "open_shared_database", traced_open)

    # Act
    definitions = [CodeIndex.from_git(tmp_path).find_definition(f"f{n}") for n in (0, 449)]

    # Assert
    assert [[span.file for span in found] for found in definitions] == [["app/m0.py"], ["app/m449.py"]]
    assert len(commits) == 1


def contents_with_calls(prefix: str, count: int, rows: int) -> dict[str, scope_scan.FileFacts]:
    """``count`` file contents of ``rows`` calls each, keyed by made-up blob ids."""
    return {
        hashlib.sha1(f"{prefix}{n}".encode()).hexdigest(): scope_scan.FileFacts(
            scope_scan.FileStructure((), (), ()),
            tuple(
                scope_scan.CallMatch(f"f{n}.ts", line + 1, f"name{line % 4000}", None) for line in range(rows)
            ),
            (),
        )
        for n in range(count)
    }


def _begin_a_write_while_another_writes(path: str, started, outcome) -> None:
    started.wait()
    time.sleep(0.2)
    with closing(sqlite3.connect(path, timeout=1)) as database:
        try:
            database.execute("begin immediate")
            database.commit()
            outcome.put("written")
        except sqlite3.OperationalError as error:
            outcome.put(str(error))


def test_a_second_process_writes_while_a_large_scope_is_added(private_cache_root: Path) -> None:
    # Arrange: 300,000 rows to add; another process may wait one second for the write lock (JVN
    # waits 30), so it fails if one transaction held the lock for the whole add
    table = name_table.NameTable()
    large = contents_with_calls("large", 60, 5_000)
    context = multiprocessing.get_context("spawn")
    started, outcome = context.Event(), context.Queue()
    other = context.Process(
        target=_begin_a_write_while_another_writes, args=(str(table.path), started, outcome)
    )
    other.start()

    # Act
    started.set()
    table.add(large)
    other.join(timeout=60)

    # Assert
    assert outcome.get(timeout=5) == "written"
    assert len(table.rows("name0")) == 60 * 2


def test_the_table_lives_in_the_cache_root(private_cache_root: Path) -> None:
    # Act
    path = name_table.NameTable().path

    # Assert
    assert path.parent == private_cache_root / "names"


OTHER_REPOSITORY = {
    "lib/other.py": "def elsewhere(x):\n    return x\n",
    "lib/more.py": "def more():\n    return 2\n",
}


def set_confirmed(cache: Path, days_ago: int) -> None:
    """Every file content in the table was last confirmed ``days_ago`` days ago."""
    [path] = (cache / "names").glob("*.sqlite")
    with sqlite3.connect(path) as database:
        database.execute("update files set confirmed = ?", (today() - days_ago,))


def confirmed_days(cache: Path) -> list[int]:
    [path] = (cache / "names").glob("*.sqlite")
    with sqlite3.connect(path) as database:
        return [day for (day,) in database.execute("select confirmed from files")]


def test_covering_a_scope_confirms_its_files_today(tmp_path: Path, private_cache_root: Path) -> None:
    # Arrange
    commit_files(tmp_path, REPOSITORY)
    every_lookup(CodeIndex.from_git(tmp_path))
    set_confirmed(private_cache_root, 40)

    # Act
    every_lookup(CodeIndex.from_git(tmp_path))

    # Assert
    assert confirmed_days(private_cache_root) == [today()] * len(REPOSITORY)


def test_forgetting_unconfirmed_files_keeps_every_answer_for_the_scope_a_run_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_cache_root: Path, spawned: Counter[str]
) -> None:
    # Arrange: two repositories share the table; only the first is covered again
    kept, gone = tmp_path / "kept", tmp_path / "gone"
    commit_files(kept, REPOSITORY)
    commit_files(gone, OTHER_REPOSITORY)
    every_lookup(CodeIndex.from_git(kept))
    every_lookup(CodeIndex.from_git(gone))
    set_confirmed(private_cache_root, 40)
    every_lookup(CodeIndex.from_git(kept))

    # Act
    forgotten = name_table.NameTable().forget_unconfirmed(before=today() - 30, limit=2_000)
    spawned.clear()
    warm = every_lookup(CodeIndex.from_git(kept))

    # Assert
    assert forgotten == len(OTHER_REPOSITORY)
    assert len(confirmed_days(private_cache_root)) == len(REPOSITORY)
    assert spawned[tools.AST_GREP] == 0
    assert warm == answered_from_scratch(kept, monkeypatch, tmp_path / "fresh")


def test_forgetting_takes_the_least_recently_confirmed_first_and_stops_at_the_limit(
    tmp_path: Path, private_cache_root: Path
) -> None:
    # Arrange
    commit_files(tmp_path, REPOSITORY)
    every_lookup(CodeIndex.from_git(tmp_path))
    [path] = (private_cache_root / "names").glob("*.sqlite")
    with sqlite3.connect(path) as database:
        blobs = [blob for (blob,) in database.execute("select blob from files order by blob")]
        for age, blob in enumerate(blobs):
            database.execute("update files set confirmed = ? where blob = ?", (today() - 40 - age, blob))

    # Act
    forgotten = name_table.NameTable().forget_unconfirmed(before=today(), limit=2)

    # Assert
    with sqlite3.connect(path) as database:
        left = {blob for (blob,) in database.execute("select blob from files")}
        orphans = database.execute("select count(*) from names where blob not in (select blob from files)")
        assert orphans.fetchone() == (0,)
    assert forgotten == 2
    assert left == set(blobs[:2])


def test_forgetting_returns_the_freed_space_to_the_disk(tmp_path: Path, private_cache_root: Path) -> None:
    # Arrange
    commit_files(
        tmp_path, {f"app/module_{n}.py": f"def f{n}(x):\n    return g{n}(x)\n" * 40 for n in range(30)}
    )
    every_lookup(CodeIndex.from_git(tmp_path))
    table = name_table.NameTable()
    with sqlite3.connect(table.path) as database:
        database.execute("pragma wal_checkpoint(truncate)")
    before = table.path.stat().st_size

    # Act
    table.forget_unconfirmed(before=today() + 1, limit=2_000)

    # Assert
    assert table.path.stat().st_size < before / 2


def test_opening_the_table_marks_its_file_used(tmp_path: Path, private_cache_root: Path) -> None:
    # Arrange
    commit_files(tmp_path, REPOSITORY)
    every_lookup(CodeIndex.from_git(tmp_path))
    path = name_table.NameTable().path
    stamp = time.time() - 10 * 86_400
    os.utime(path, (stamp, stamp))

    # Act
    name_table.NameTable()

    # Assert
    assert day_of(path.stat().st_mtime) == today()


def test_a_lookup_whose_stamp_cannot_be_written_keeps_its_entries_and_says_so(
    tmp_path: Path, private_cache_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange: another process holds the write lock past the table's (shortened) busy wait
    commit_files(tmp_path, REPOSITORY)
    every_lookup(CodeIndex.from_git(tmp_path))
    set_confirmed(private_cache_root, 40)
    table = name_table.NameTable()
    with sqlite3.connect(table.path) as database:
        blobs = [blob for (blob,) in database.execute("select blob from files")]
    table._db.execute("pragma busy_timeout = 50")
    writer = sqlite3.connect(table.path, isolation_level=None)
    writer.execute("begin immediate")

    # Act
    try:
        entries = table.entries(blobs)
    finally:
        writer.execute("rollback")

    # Assert
    assert set(entries) == set(blobs)
    assert "not stamped" in caplog.text
    assert confirmed_days(private_cache_root) == [today() - 40] * len(blobs)
