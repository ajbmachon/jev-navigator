from __future__ import annotations

from pathlib import Path

from git_repos import commit_files, git

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.rebuild import rebuild_request
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.testing import ScriptedJevClient

LIMITS = Check(
    "limits",
    "Does `{item}.code` limit how many items `claim.subject` may hold?",
    Criterion("A line compares the item count with a limit."),
    Criterion("No such comparison."),
)
CLAIM = {"claim": {"subject": "an order"}}


def judge_check_limits(index: CodeIndex, store_path: Path) -> None:
    span = index.find_definition("check_limits")[0]
    code = index.read_slice(span)
    item = {"file": span.file, "lines": [span.start, span.end], "commit": code.commit, "code": code.text}
    Judge(ScriptedJevClient(), store=JsonlAnswerStore(store_path)).check_each(LIMITS, [item], CLAIM)


def stored_record(store_path: Path):
    return JsonlAnswerStore(store_path).records()[0]


def test_a_stored_request_is_rebuilt_exactly_from_the_repository_at_its_commit(
    sample_index: CodeIndex, sample_repo: Path, tmp_path: Path
) -> None:
    # Arrange
    judge_check_limits(sample_index, tmp_path / "answers.jsonl")
    record = stored_record(tmp_path / "answers.jsonl")
    (sample_repo / "app/validation.py").write_text("# the checkout moved on\n")

    # Act
    rebuilt = rebuild_request(record, CodeIndex.at_commit(sample_repo, sample_index.commit), CLAIM)

    # Assert
    assert rebuilt.matches and rebuilt.request_sha256 == record.request_sha256
    assert "def check_limits" not in (tmp_path / "answers.jsonl").read_text()


def test_a_mismatch_names_the_part_that_changed(
    sample_index: CodeIndex, sample_repo: Path, tmp_path: Path
) -> None:
    # Arrange
    judge_check_limits(sample_index, tmp_path / "answers.jsonl")
    record = stored_record(tmp_path / "answers.jsonl")
    path = sample_repo / "app/validation.py"
    path.write_text(path.read_text().replace("<= limit", "< limit"))
    git(sample_repo, "commit", "-qam", "tighten")

    # Act
    rebuilt = rebuild_request(record, CodeIndex.from_git(sample_repo), {"claim": {"subject": "a basket"}})

    # Assert
    assert not rebuilt.matches
    assert rebuilt.differences == ("shared state", "code of item 0 (app/validation.py lines 10-12)")


def test_a_batch_whose_items_share_a_masked_value_is_rebuilt_exactly(tmp_path: Path) -> None:
    # Arrange: the second item copies the value its file keys outside it; a key in another file
    # only is not hidden (cross-file copies are on the later list)
    repo = tmp_path / "repo"
    repo.mkdir()
    commit_files(
        repo,
        {
            "settings.py": 'WEBHOOK_TOKEN = "order-hook-4f7a1c"\n',
            "hooks.py": (
                'HOOK_TOKEN = "order-hook-4f7a1c"\n\n\n'
                'def send(order):\n    return post("order-hook-4f7a1c", order)\n'
            ),
        },
    )
    index = CodeIndex.from_git(repo)
    items = [
        {"file": file, "lines": [first, last], "commit": index.commit, "code": index.read_slice(span).text}
        for file, first, last, span in (
            ("settings.py", 1, 1, Span("settings.py", 1, 1)),
            ("hooks.py", 4, 5, Span("hooks.py", 4, 5)),
        )
    ]
    store_path = tmp_path / "answers.jsonl"
    client = ScriptedJevClient()
    Judge(client, store=JsonlAnswerStore(store_path)).check_each(LIMITS, items, CLAIM)

    # Act
    rebuilt = rebuild_request(stored_record(store_path), index, CLAIM)

    # Assert
    assert "order-hook-4f7a1c" not in str(client.requests[0][0])
    assert rebuilt.matches
