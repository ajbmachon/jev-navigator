import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from jev_navigator.index import fact_cache
from jev_navigator.index.fact_cache import FactCache
from jev_navigator.index.scope_scan import Unparsed, scan_facts


def test_new_index_reuses_facts_and_changed_content_is_reparsed(tmp_path, monkeypatch):
    from jev_navigator.index import tools
    from jev_navigator.index.code_index import CodeIndex

    repository = tmp_path / "repo"
    repository.mkdir()
    source = repository / "module.py"
    source.write_text("def original():\n    return service()\n")
    cache = tmp_path / "cache"
    scans = []
    actual_scan = tools.ast_grep_rules

    def observe_scan(rules, files, root):
        scans.append(tuple(files))
        return actual_scan(rules, files, root)

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
    from jev_navigator.index.code_index import CodeIndex

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
    facts = scan_facts(
        ["module.py"], tmp_path, lambda file: (tmp_path / file).read_text().splitlines(), unparsed
    )
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


def test_content_language_parser_and_rules_invalidate(tmp_path, example, monkeypatch):
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
    monkeypatch.setattr(fact_cache, "FACT_RULE_VERSION", fact_cache.FACT_RULE_VERSION + "-different")
    assert cache.load("module.py", content) is None


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


def test_node_name_fix_reparses_facts_cached_by_the_previous_rule_version(tmp_path, monkeypatch):
    from jev_navigator.index.code_index import CodeIndex
    from jev_navigator.index.scope_scan import FileFacts, FileStructure
    from jev_navigator.index.spans import Span

    repo = tmp_path / "repo"
    repo.mkdir()
    content = b"class Box { v() { return 1; } }\n"
    (repo / "box.ts").write_bytes(content)
    cache_root = tmp_path / "cache"
    # Persisted v8 artifact: the method was named from its enclosing physical line, so both
    # declarations collapsed into Box. A parser correction must invalidate this old result.
    collapsed = Span("box.ts", 1, 1, "Box")
    old = FileFacts(FileStructure((collapsed,), (collapsed,), (), (collapsed,)), (), (), False)
    with monkeypatch.context() as previous:
        previous.setattr(fact_cache, "FACT_RULE_VERSION", "combined-facts-v8-export-surface")
        FactCache(cache_root).save("box.ts", content, old)
    scans = []
    index = CodeIndex.from_directory(
        repo,
        fact_cache_dir=cache_root,
        scan_observer=lambda *event: scans.append(event),
    )
    assert [span.name for span in index.functions_in("box.ts")] == ["v"]
    assert {span.name for span in index.symbols_in("box.ts")} == {"Box", "v"}
    assert any(event[1] == "started" for event in scans)
