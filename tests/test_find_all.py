import threading
from pathlib import Path

from jev_navigator.directives.find_all import find_all
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.testing import ScriptedJevClient


def repository(tmp_path: Path) -> CodeIndex:
    files = {
        "rules.py": "def accept(order):\n    return len(order.items) <= 4\n",
        "api.py": "from rules import accept\n\ndef submit(order):\n    return accept(order)\n",
        "other.py": (
            "def fits(cart):\n    return not len(cart.items) > 4\n\n"
            "def maybe(config):\n    return config.validate()\n"
        ),
        "unrelated.py": "def accept(text):\n    return text.lower()\n",
        "web.ts": "export function capacity(order: {items: unknown[]}) { return order.items.length <= 4; }\n",
    }
    for name, source in files.items():
        (tmp_path / name).write_text(source)
    return CodeIndex(tmp_path, files)


def client() -> ScriptedJevClient:
    # A provider fixture labels these concrete bodies; the real parser, graph, batching and
    # composition are exercised. This proves execution, not Jev's semantic accuracy.
    def answer(question_id, question, state):
        item = state["items"][int(question_id.rsplit("#", 1)[1])]
        labels = {
            ("rules.py", "accept"): 0.95,
            ("other.py", "fits"): 0.94,
            ("web.ts", "capacity"): 0.93,
            ("other.py", "maybe"): 0.5,
        }
        return labels.get((item["file"], item["name"]), 0.05)

    return ScriptedJevClient(nouls=answer)


def test_connected_first_then_disconnected_python_and_typescript_without_duplicate_judgments(tmp_path):
    index = repository(tmp_path)
    seed = next(span for span in index.find_definition("accept") if span.file == "rules.py")
    provider = client()

    result = find_all(index, Judge(provider), "the order item limit check", [seed, seed])

    assert {(r.item["file"], r.item["name"]) for r in result.matched} == {
        ("rules.py", "accept"),
        ("other.py", "fits"),
        ("web.ts", "capacity"),
    }
    assert [r.item["name"] for r in result.uncertain] == ["maybe"]
    assert {(r.item["file"], r.item["name"]) for r in result.negative} == {
        ("api.py", "submit"),
        ("unrelated.py", "accept"),
    }
    assert len(result.judged) == 6
    assert len({r.item["span_key"] for r in result.judged}) == 6
    assert result.coverage == "functions_examined"
    first_items = provider.requests[0][0]["items"]
    assert {item["file"] for item in first_items} == {"rules.py", "api.py"}
    assert len(provider.requests) == 2
    assert all(len(questions) > 1 for _, questions in provider.requests)
    assert all(r.item["file_sha256"] for r in result.judged)


def test_component_only_stays_partial_and_repeat_reuses_answers(tmp_path):
    index = repository(tmp_path)
    seed = index.find_definition("fits")[0]
    provider = client()
    judge = Judge(provider, store=JsonlAnswerStore(tmp_path / "answers.jsonl"), served_model=provider.model)

    partial = find_all(index, judge, "the order item limit check", [seed], include_disconnected=False)
    complete = find_all(index, judge, "the order item limit check", [seed])
    replay = find_all(index, judge, "the order item limit check", [seed])

    assert partial.coverage == "partial"
    assert partial.remaining_files
    assert {r.item["name"] for r in partial.matched} == {"fits"}
    assert complete.coverage == "functions_examined"
    assert next(r for r in complete.judged if r.item["name"] == "fits").from_store
    assert replay.calls == 0
    assert len(replay.matched) == 3


def test_cancelled_before_search_does_not_trigger_final_parser_scans(tmp_path):
    index = repository(tmp_path)
    provider = client()

    result = find_all(index, Judge(provider), "order limits", [], cancelled=lambda: True)

    assert result.stopped_by == "cancelled"
    assert result.coverage == "partial"
    assert set(result.remaining_files) == set(index.files)
    assert index.parser_scans_pending == ("facts",)
    assert result.calls == 0
    assert provider.requests == []


def test_missing_grammar_is_retained_as_a_coverage_gap(tmp_path):
    (tmp_path / "policy.unknown").write_text("reject when items exceed four")
    index = CodeIndex(tmp_path, ["policy.unknown"])

    result = find_all(index, Judge(client()), "order limits", [])

    assert result.unsupported_files == ("policy.unknown",)
    assert result.coverage == "scope_incomplete"
    assert result.judged == ()


def test_fallback_batches_real_parser_work_and_keeps_every_function(tmp_path):
    for number in range(12):
        (tmp_path / f"part_{number}.py").write_text(f"def check_{number}(x):\n    return len(x) <= 4\n")
    scans = []
    index = CodeIndex(
        tmp_path,
        [f"part_{number}.py" for number in range(12)],
        fact_cache_dir=tmp_path / "facts",
        scan_observer=lambda name, event, count: scans.append((event, count)),
    )

    result = find_all(index, Judge(ScriptedJevClient(default_noul=0.95)), "item limit check", [])

    assert {r.item["name"] for r in result.matched} == {f"check_{number}" for number in range(12)}
    assert [count for event, count in scans if event == "started"] == [12]
    assert result.calls == 1


def test_budget_stop_retains_completed_batches_and_cache_only_replay(tmp_path):
    for name in ("alpha", "beta", "gamma"):
        (tmp_path / f"{name}.py").write_text(
            f"def {name}(item):\n    note = {('x' * 40000)!r}\n    return len(item) <= 3\n"
        )
    index = CodeIndex(tmp_path, ["alpha.py", "beta.py", "gamma.py"])
    provider = ScriptedJevClient(default_noul=0.96)
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    stopped = find_all(index, Judge(provider, max_calls=1, store=store), "item limit", [])
    assert stopped.stopped_by == "budget"
    assert stopped.coverage == "partial"
    assert stopped.calls == 1
    assert len(stopped.judged) == 1
    assert set(stopped.remaining_files) == {"beta.py", "gamma.py"}
    cached = find_all(
        index, Judge(provider, max_calls=0, store=store, served_model=provider.model), "item limit", []
    )
    assert cached.calls == 0
    assert len(cached.judged) == 1
    assert cached.judged[0].from_store
    assert cached.stopped_by == "budget"
    completed = find_all(
        index, Judge(provider, max_calls=2, store=store, served_model=provider.model), "item limit", []
    )
    assert completed.coverage == "functions_examined"
    assert len(completed.judged) == 3
    assert completed.calls == 2


class ReleasesInOrder:
    """A provider that holds a wave's requests until all of them arrived, then answers them one at a
    time, by the first item's line, ascending or descending, so the judge sees them complete in a
    chosen order."""

    def __init__(self, expected: int, *, descending: bool) -> None:
        self.script = ScriptedJevClient(default_noul=0.05)
        self.expected = expected
        self.descending = descending
        self.arrived: list[int] = []
        self.answered: list[int] = []
        self.turns = threading.Condition()

    @property
    def model(self) -> str:
        return self.script.model

    def send(self, state, questions):
        line = state["items"][0]["lines"][0]
        with self.turns:
            self.arrived.append(line)
            self.turns.notify_all()
            self.turns.wait_for(lambda: len(self.arrived) >= self.expected, timeout=10)
            order = sorted(self.arrived, reverse=self.descending)
            self.turns.wait_for(lambda: order[len(self.answered)] == line, timeout=10)
            self.answered.append(line)
            self.turns.notify_all()
        return self.script.send(state, questions)

    def parse(self, raw):
        return self.script.parse(raw)


def test_find_all_lists_its_judged_functions_in_file_and_line_order_however_answers_arrive(tmp_path):
    # Arrange: forty functions in ten batches, answered in line order in one run and reversed in the other
    source = "".join(
        f"def check_{index}(items):\n    return len(items) <= {index}\n\n" for index in range(40)
    )
    (tmp_path / "checks.py").write_text(source)
    index = CodeIndex(tmp_path, {"checks.py": source})
    runs = []

    # Act
    for descending in (False, True):
        provider = ReleasesInOrder(10, descending=descending)
        result = find_all(index, Judge(provider, items_per_request=4), "the item limit check", [])
        runs.append(([r.item["span_key"] for r in result.judged], provider.answered))

    # Assert: the providers answered in opposite orders, and both results list the same order
    (forward, forward_answered), (backward, backward_answered) = runs
    assert forward_answered == sorted(forward_answered) and backward_answered == sorted(
        backward_answered, reverse=True
    )
    assert forward == backward
    assert forward == sorted(forward, key=lambda key: int(key.split(":")[1].split("-")[0]))
