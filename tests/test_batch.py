"""Public batch behavior with real files, parser bindings, history and the existing Judge."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from jev_navigator.batch import Cursor, Operation, run_batch, run_batch_async
from jev_navigator.cli import main
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.judgments.answers import JevResponse, NoulAnswer
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.store import SHARED_STORE_VARIABLE


def repository(tmp_path: Path) -> Path:
    files = {
        "app/core.py": "def check(value):\n    return value > 0\n",
        "app/use.py": "from app.core import check\n\ndef run(value):\n    return check(value)\n",
        "app/other.py": "def check(value):\n    return False\n\ndef other(value):\n    return check(value)\n",
        "tests/test_use.py": "from app.use import run\n\ndef test_run():\n    assert run(1)\n",
        "web/core.ts": "export function check(x: number) { return x > 0; }\n",
        "web/index.ts": "export { check as validate } from './core';\n",
        "web/use.ts": (
            "import { validate } from './index';\nexport function run(x: number) { return validate(x); }\n"
        ),
        "README.md": "See app/core.py for validation.\n",
    }
    for file, text in files.items():
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_batch_resolves_owners_barrels_and_test_candidates(tmp_path):
    root = repository(tmp_path)
    with CodeIndex.from_directory(root) as index:
        result = run_batch(
            index,
            [
                Operation("outline", file="app/use.py"),
                Operation("def", name="check", file="app/core.py"),
                Operation("callers", name="check", file="app/core.py"),
                Operation("callees", file="web/use.ts", line=2),
                Operation("tests_of", file="app/use.py", name="run"),
                Operation("named_files", query="See app/core.py"),
                Operation("names", name="check"),
                Operation("refs", name="check", file="app/core.py"),
                Operation("callers", name="check", file="web/core.ts"),
                Operation("named_files"),
            ],
        )
    assert not any(page.error for page in result.pages)
    assert [page.number for page in result.pages] == list(range(1, 11))
    assert result.pages[0].items[0]["imports"] == ("app/core.py",)
    assert result.pages[1].items[0]["file"] == "app/core.py"
    assert [row["file"] for row in result.pages[2].items] == ["app/use.py"]
    edge = result.pages[3].items[0]
    assert edge["name"] == "validate"
    assert edge["binding"]["status"] == "resolved"
    assert edge["binding"]["target"]["file"] == "web/core.ts"
    assert result.pages[4].items[0]["file"] == "tests/test_use.py"
    assert result.pages[5].items[0]["file"] == "app/core.py"
    assert result.pages[6].items[0]["name"] == "check"
    assert any(row["file"] == "app/use.py" for row in result.pages[7].items)
    assert [row["file"] for row in result.pages[8].items] == ["web/use.ts"]
    assert result.pages[9].total == 8
    assert "README.md" in {row["file"] for row in result.pages[9].items}
    assert result.calls == 0


def test_pages_reconstruct_source_including_long_unicode_line(tmp_path):
    expected = ["first", '"\\🙂' * 2000, "last"]
    (tmp_path / "long.txt").write_text("\n".join(expected))
    rows = []
    fragments = ""
    op = Operation("show", file="long.txt", end=3, limit=1)
    with CodeIndex.from_directory(tmp_path) as index:
        while True:
            response = run_batch(index, [op], max_chars=1600)
            assert len(response.render()) <= 1600
            page = response.pages[0]
            assert not page.error
            assert page.total == 1
            for row in page.items:
                if "row_json" in row:
                    assert row["row_json_offset"] == len(fragments)
                    fragments += row["row_json"]
                    if len(fragments) == row["row_json_chars"]:
                        rows.append(json.loads(fragments))
                        fragments = ""
                else:
                    rows.append(row)
            if page.next is None:
                break
            op = replace(op, cursor=page.next)
    assert [(row["line"], row["end"]) for row in rows] == [(1, 3)]
    assert [line for row in rows for line in row["text"].split("\n")] == expected


class StandIn:
    model = "stand-in"

    def __init__(self):
        self.batches = []

    def ask(self, state, questions):
        self.batches.append([item["id"] for item in state["items"]])
        return JevResponse(
            {key: NoulAnswer(0.9 if "return 19" in str(state) else 0.2) for key in questions}, self.model
        )

    def close(self):
        pass


def test_rank_uses_supplied_candidate_order_groups_of_16_and_refuses_implicit_calls(tmp_path):
    code = "\n".join(f"def function_{n}(): return {n}" for n in range(19))
    (tmp_path / "code.py").write_text(code)
    spans = tuple(Span("code.py", n, n) for n in range(19, 0, -1))
    client = StandIn()
    judge = Judge(client, items_per_request=4, max_calls=2)
    op = Operation("rank", query="return values", candidates=spans)
    with CodeIndex.from_directory(tmp_path) as index:
        refused = run_batch(index, [op])
        assert refused.calls == 0
        assert refused.pages[0].error == "ValueError"
        response = run_batch(index, [op], judge=judge)
    assert not response.pages[0].error
    assert client.batches == [[span.key for span in spans[:16]], [span.key for span in spans[16:]]]
    assert response.calls == judge.calls == 2
    assert response.pages[0].total == 19
    assert len(response.pages[0].items) == 19
    assert response.pages[0].items[0]["answers"]["match_query"]["request_sha256"]
    with CodeIndex.from_directory(tmp_path) as index:
        partial = run_batch(index, [op], judge=Judge(StandIn(), max_calls=1))
    assert partial.pages[0].total == 19
    assert sum(row["status"] == "judged" for row in partial.pages[0].items) == 16
    pending = [row for row in partial.pages[0].items if row["status"] == "not_judged"]
    assert len(pending) == 3
    assert all(row["score"] is None and row["reason"] for row in pending)


def test_cli_fact_batch_needs_no_client_and_keeps_success_beside_failure(tmp_path, capsys):
    repository(tmp_path)
    request = {
        "command": "batch",
        "repo": str(tmp_path),
        "operations": [
            {"op": "refs", "query": "check", "scopes": ["app"], "regex": False, "window": 1},
            {"op": "show", "file": "missing.py"},
        ],
    }
    assert main(["--json", json.dumps(request)]) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["calls"] == 0
    assert any("    return check(value)" in row["text"].split("\n") for row in response["pages"][0]["items"])
    assert response["pages"][1]["error"] == "ValueError"
    assert (
        main(
            [
                "batch",
                json.dumps({"operations": [{"op": "show", "file": "README.md", "window": 2}]}),
                "--repo",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["pages"][0]["items"][0]["text"] == (
        "See app/core.py for validation."
    )


def test_cli_rank_reuses_exact_answer_store_without_new_transport(tmp_path, capsys, monkeypatch):
    from jev_navigator import cli_batch

    repository(tmp_path)
    client = StandIn()
    monkeypatch.setattr(cli_batch, "system_one_client", lambda _: client)
    monkeypatch.setattr(cli_batch, "load_typesafe_environment", lambda _: None)
    monkeypatch.setenv(SHARED_STORE_VARIABLE, str(tmp_path / "answers.sqlite"))
    request = {
        "operations": [
            {
                "op": "rank",
                "query": "validation",
                "candidates": [{"file": "app/core.py", "start": 1, "end": 2}],
            }
        ]
    }
    argv = ["batch", json.dumps(request), "--repo", str(tmp_path), "--max-calls", "1"]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["calls"] == 1
    assert main(["batch", json.dumps(request), "--repo", str(tmp_path), "--replay"]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["calls"] == 0
    assert replay["replayed_answers"] == 1
    assert len(client.batches) == 1


def test_cochange_counts_scope_and_displays_the_remaining_tail(tmp_path):
    repository(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "together",
        ],
        check=True,
    )
    with CodeIndex.from_directory(tmp_path) as index:
        result = run_batch(index, [Operation("cochange", file="app/core.py", limit=1)])
    page = result.pages[0]
    assert page.total == 7
    assert len(page.items) == 1
    assert page.items[0]["commits"] == 1
    assert page.next is not None


def test_async_runtime_preserves_order_and_cap_with_its_own_client(tmp_path):
    class AsyncStandIn(StandIn):
        async def ask(self, state, questions):
            await asyncio.sleep(0)
            return super().ask(state, questions)

    (tmp_path / "code.py").write_text("\n".join(f"def fn_{n}(): return {n}" for n in range(19)))
    candidates = tuple(Span("code.py", n, n) for n in range(19, 0, -1))
    client = AsyncStandIn()
    judge = Judge(client, max_calls=1)
    with CodeIndex.from_directory(tmp_path) as index:
        result = asyncio.run(
            run_batch_async(
                index,
                [
                    Operation("rank", query="values", candidates=candidates),
                    Operation("show", file="code.py", line=1, end=2),
                ],
                judge=judge,
            )
        )
    assert client.batches == [[span.key for span in candidates[:16]]]
    assert [page.number for page in result.pages] == [1, 2]
    assert not any(page.error for page in result.pages)
    assert result.calls == 1
    assert sum(row["status"] == "not_judged" for row in result.pages[0].items) == 3
    assert result.pages[1].items[0]["end"] == 2


def test_rank_paging_preserves_scores_without_another_live_request(tmp_path):
    from jev_navigator.judgments.store import SqliteAnswerStore

    (tmp_path / "code.py").write_text("\n".join(f"def f{n}(): return {n}" for n in range(19)))
    client = StandIn()
    judge = Judge(client, max_calls=2, store=SqliteAnswerStore(tmp_path / "answers.sqlite"))
    op = Operation(
        "rank", query="return values", candidates=tuple(Span("code.py", n, n) for n in range(1, 20))
    )
    rows = []
    with CodeIndex.from_directory(tmp_path) as index:
        while True:
            response = run_batch(index, [op], judge=judge, max_chars=1600)
            page = response.pages[0]
            assert not page.error
            assert len(response.render()) <= 1600
            rows.extend(page.items)
            if page.next is None:
                break
            op = replace(op, cursor=page.next)
    assert len(client.batches) == judge.calls == 2
    assert len(rows) == 19
    assert all(row["status"] == "judged" and row["answers"] for row in rows)
    assert len({row["candidate"] for row in rows}) == 19

    uncached = Judge(StandIn(), max_calls=2)
    initial = replace(op, cursor=Cursor())
    with CodeIndex.from_directory(tmp_path) as index:
        first = run_batch(index, [initial], judge=uncached, max_chars=1600)
        continuation = replace(initial, cursor=first.pages[0].next)
        refused = run_batch(index, [continuation], judge=uncached, max_chars=1600)
    assert refused.pages[0].error == "ValueError"
    assert "store" in refused.pages[0].items[0]["error"]
    assert uncached.calls == 2
