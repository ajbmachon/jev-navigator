import json
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from git_repos import commit_files, read_files

from jev_navigator.confirmation import day_of, today
from jev_navigator.index import fact_cache, imports, languages, scope_scan, spans, tools
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.fact_cache import FactCache
from jev_navigator.index.scope_scan import FileFacts, FileStructure, Unparsed, scan_facts
from jev_navigator.index.spans import Span


def test_new_index_reuses_facts_and_changed_content_is_reparsed(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    source = repository / "module.py"
    source.write_text("def original():\n    return service()\n")
    cache = tmp_path / "cache"
    scans = []
    actual_scan = tools.ast_grep_rules

    def observe_scan(rules, files, *arguments, **options):
        scans.append(tuple(files))
        return actual_scan(rules, files, *arguments, **options)

    monkeypatch.setattr(tools, "ast_grep_rules", observe_scan)
    first = CodeIndex.from_directory(repository, fact_cache_dir=cache)
    expected = first.functions_in("module.py")
    assert any(span.name == "original" for span in expected)
    assert scans
    scans.clear()
    warm = CodeIndex.from_directory(repository, fact_cache_dir=cache)
    assert warm.functions_in("module.py") == expected
    assert scans == [], "a new index must reuse persisted facts without parsing"

    source.write_text("def replacement():\n    return another_service()\n")
    changed = CodeIndex.from_directory(repository, fact_cache_dir=cache)
    result = changed.functions_in("module.py")
    assert any(span.name == "replacement" for span in result)
    assert not any(span.name == "original" for span in result)
    assert scans, "changed source must not reuse the old syntax facts"


def test_warm_index_preserves_incomplete_parser_coverage(tmp_path):

    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "broken.py").write_text("def broken(\n")
    (repository / "use.py").write_text("from broken import broken\n\nbroken()\n")
    cache = tmp_path / "cache"
    cold = CodeIndex.from_directory(repository, fact_cache_dir=cache)
    cold.functions_in("broken.py")
    assert "broken.py" in cold.observed_unparsed_files
    warm = CodeIndex.from_directory(repository, fact_cache_dir=cache)
    warm.functions_in("broken.py")
    assert "broken.py" in warm.observed_unparsed_files
    assert warm.find_callers("broken")[0].binding.status == "unknown", "the unread lines must persist"


@pytest.fixture
def example(tmp_path):
    content = b"def handler():\n    return service()\n"
    source = tmp_path / "module.py"
    source.write_bytes(content)
    unparsed = Unparsed()
    facts = scan_facts(read_files(tmp_path, ["module.py"]), tmp_path, unparsed)
    assert not unparsed.files
    assert facts["module.py"].calls
    return content, facts["module.py"]


def test_roundtrip_rebinds_paths_without_retaining_source(tmp_path, example):
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    cache.save("module.py", content, facts)
    restored = cache.load("copied.py", content)
    assert restored is not None
    assert [(c.name, c.line) for c in restored.calls] == [(c.name, c.line) for c in facts.calls]
    assert all(c.file == "copied.py" for c in restored.calls)
    assert all(s.file == "copied.py" for s in restored.structure.functions)
    assert all(content.decode() not in p.read_text() for p in cache.root.rglob("*.json"))


@pytest.fixture
def rule_identity_reset(request):
    """Rules patched in a test change the identity the fact cache computes once per process."""
    fact_cache._rules_identity.cache_clear()
    request.addfinalizer(fact_cache._rules_identity.cache_clear)


def test_content_language_parser_and_rules_invalidate(tmp_path, example, monkeypatch, rule_identity_reset):
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    cache.save("module.py", content, facts)
    assert cache.load("module.py", content) is not None
    assert cache.load("module.py", content + b"\n") is None
    assert cache.load("module.ts", content) is None
    previous_parser = cache.parser
    cache.parser = previous_parser + "-different"
    assert cache.load("module.py", content) is None
    cache.parser = previous_parser
    monkeypatch.setitem(languages.FUNCTION_KINDS, "python", ("function_definition", "lambda"))
    fact_cache._rules_identity.cache_clear()
    assert cache.load("module.py", content) is None


def test_a_change_to_the_flow_rules_is_a_cache_miss_for_javascript(
    tmp_path, monkeypatch, rule_identity_reset
):
    # Arrange: JavaScript the JavaScript grammar only partly reads takes its facts from the flow rules
    content = b"export function typed(value: string): string {\n  return value;\n}\n"
    cache = FactCache(tmp_path / "cache")
    cache.save("typed.js", content, FileFacts(FileStructure((), (), ()), (), ()))

    # Act
    monkeypatch.setitem(languages.FUNCTION_KINDS, languages.FLOW_LANGUAGE, ("function_declaration",))
    fact_cache._rules_identity.cache_clear()
    reused = cache.load("typed.js", content)

    # Assert
    assert reused is None


@pytest.mark.parametrize("broken", ["{", "null", "[]", '{"structure":{}}'])
def test_corrupt_entries_are_cache_misses(tmp_path, example, broken):
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    cache.save("module.py", content, facts)
    [entry] = cache.root.rglob("*.json")
    entry.write_text(broken)
    assert cache.load("module.py", content) is None


def test_entry_missing_completeness_is_a_cache_miss(tmp_path, example):
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    cache.save("module.py", content, facts)
    [entry] = cache.root.rglob("*.json")
    raw = json.loads(entry.read_text())
    del raw["incomplete"]
    entry.write_text(json.dumps(raw))

    assert cache.load("module.py", content) is None


def test_concurrent_writers_publish_one_complete_entry(tmp_path, example, monkeypatch):
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    ready = Barrier(2)
    replace = Path.replace

    def publish(path, target):
        ready.wait()
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", publish)
    with ThreadPoolExecutor(max_workers=2) as workers:
        jobs = [workers.submit(cache.save, "module.py", content, facts) for _ in range(2)]
        for job in jobs:
            job.result()
    assert cache.load("module.py", content) == facts


def test_facts_cached_under_other_rules_are_parsed_again(tmp_path, monkeypatch, rule_identity_reset):

    repo = tmp_path / "repo"
    repo.mkdir()
    content = b"class Box { v() { return 1; } }\n"
    (repo / "box.ts").write_bytes(content)
    cache_root = tmp_path / "cache"
    # Persisted v8 artifact: the method was named from its enclosing physical line, so both
    # declarations collapsed into Box. A parser correction must invalidate this old result.
    collapsed = Span("box.ts", 1, 1, "Box")
    old = FileFacts(FileStructure((collapsed,), (collapsed,), ()), (), (), False)
    with monkeypatch.context() as previous:
        previous.setitem(languages.FUNCTION_KINDS, "typescript", ("function_declaration",))
        FactCache(cache_root).save("box.ts", content, old)
    fact_cache._rules_identity.cache_clear()
    scans = []
    index = CodeIndex.from_directory(
        repo,
        fact_cache_dir=cache_root,
        scan_observer=lambda *event: scans.append(event),
    )
    assert [span.name for span in index.functions_in("box.ts")] == ["v"]
    assert {span.name for span in index.symbols_in("box.ts")} == {"Box", "v"}
    assert any(event[1] == "started" for event in scans)


@pytest.mark.parametrize(
    "module",
    [scope_scan, languages, imports, spans, tools],
    ids=["scope_scan", "languages", "imports", "spans", "tools"],
)
def test_a_change_to_the_code_that_runs_the_parser_or_reads_its_matches_is_a_cache_miss(
    module, tmp_path, example, monkeypatch, request
):
    # Arrange
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    cache.save("module.py", content, facts)
    edited = tmp_path / Path(module.__file__).name
    edited.write_text(Path(module.__file__).read_text() + "\n# an edit to how matches become facts\n")
    request.addfinalizer(fact_cache._match_reader_source.cache_clear)
    request.addfinalizer(fact_cache._rules_identity.cache_clear)

    # Act
    monkeypatch.setattr(module, "__file__", str(edited))
    fact_cache._match_reader_source.cache_clear()
    fact_cache._rules_identity.cache_clear()
    reused = cache.load("module.py", content)

    # Assert
    assert reused is None


def test_the_fact_cache_never_writes_a_literal_quoted_by_a_receiver(tmp_path, private_cache_root):
    # Arrange
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/client.ts": 'export function load(client, cfg) {\n  client("api-key-literal").fetch(1);\n'
            '  return cfg["token-literal"].get(2);\n}\n'
        },
    )

    # Act
    CodeIndex.from_git(repository).find_callers("fetch")

    # Assert
    written = " ".join(path.read_text() for path in (private_cache_root / "facts").rglob("*.json"))
    assert '"fetch"' in written
    assert "literal" not in written


def _days_ago(path: Path, days: int) -> None:
    stamp = time.time() - days * 86_400
    os.utime(path, (stamp, stamp))


def test_a_rule_change_puts_entries_in_another_identity_folder_and_retires_the_old_one(
    tmp_path, example, monkeypatch, rule_identity_reset
):
    # Arrange
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    cache.save("module.py", content, facts)
    [before] = cache.current_folders()

    # Act
    monkeypatch.setitem(languages.FUNCTION_KINDS, "python", ("function_definition", "lambda"))
    fact_cache._rules_identity.cache_clear()
    cache.save("module.py", content, facts)

    # Assert
    [after] = cache.current_folders()
    assert after != before and after.parent == before.parent == cache.root / "python"
    assert cache.retired() == [before]
    assert [path.parent for path in cache.root.rglob("*.json")] in ([before, after], [after, before])


def test_an_entry_in_the_layout_before_identity_folders_is_retired(tmp_path, example):
    # Arrange
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    cache.save("module.py", content, facts)
    flat = cache.root / "python" / f"{'0' * 64}.json"
    flat.write_text("{}")

    # Act
    retired = cache.retired()

    # Assert
    assert retired == [flat]


def test_loading_an_entry_confirms_it_today(tmp_path, example):
    # Arrange
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    cache.save("module.py", content, facts)
    [entry] = cache.root.rglob("*.json")
    _days_ago(entry, 40)

    # Act
    loaded = FactCache(tmp_path / "cache").load("module.py", content)

    # Assert
    assert loaded == facts
    assert day_of(entry.stat().st_mtime) == today()


def test_a_corrupt_entry_is_never_confirmed(tmp_path, example):
    # Arrange
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    cache.save("module.py", content, facts)
    [entry] = cache.root.rglob("*.json")
    entry.write_text("{")
    _days_ago(entry, 40)

    # Act
    loaded = cache.load("module.py", content)

    # Assert
    assert loaded is None
    assert day_of(entry.stat().st_mtime) == today() - 40


def test_a_load_marks_its_identity_folder_used_even_when_it_misses(tmp_path, example):
    # Arrange
    content, facts = example
    FactCache(tmp_path / "cache").save("module.py", content, facts)
    [folder] = FactCache(tmp_path / "cache").current_folders()
    _days_ago(folder, 10)

    # Act
    FactCache(tmp_path / "cache").load("other.py", b"x = 1\n")

    # Assert
    assert day_of(folder.stat().st_mtime) == today()


@pytest.mark.skipif(not hasattr(os, "chflags"), reason="needs BSD file flags to refuse a stamp to its owner")
def test_a_cache_that_refuses_stamps_still_serves_its_facts(tmp_path, example, request):
    # Arrange: the entry and its identity folder are immutable, so neither can be stamped
    content, facts = example
    cache = FactCache(tmp_path / "cache")
    cache.save("module.py", content, facts)
    [entry] = cache.root.rglob("*.json")
    _days_ago(entry, 40)
    for path in (entry, entry.parent):
        os.chflags(path, stat.UF_IMMUTABLE)
        request.addfinalizer(lambda path=path: os.chflags(path, 0))

    # Act
    loaded = FactCache(tmp_path / "cache").load("module.py", content)

    # Assert
    assert loaded == facts
    assert day_of(entry.stat().st_mtime) == today() - 40
