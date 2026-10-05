"""A provider failure ends a CLI search with the same resumable state a cancel writes."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest
from git_repos import commit_files
from test_cli_run_logs import TARGET, limit_client, marked_repository

from jev_navigator import cli
from jev_navigator.cli import create_evidence_pack
from jev_navigator.directives.find_code import SearchBudget
from jev_navigator.judgments.journal import RawAttempt, RawResponse
from jev_navigator.judgments.questions import request_sha256
from jev_navigator.testing import ScriptedJevClient

START = "app/entry.py:5"


def target_request(position: int, state: Mapping) -> bool:
    """The request about the target's own body. It shares its round with a sibling, so picking it by
    content, not by arrival, keeps the round from finding the target in spite of the failure."""
    del position
    return "len(item) <= 3" in state.get("slice", {}).get("code", "")


def import_line(position: int, state: Mapping) -> bool:
    del position
    return state["slice"]["code"].startswith("from .policy import admit")


class ProviderError(RuntimeError):
    """A failure the provider reports, such as a 503."""


class FailsOnRequest:
    """Answers like ``script``, except that the requests ``fails`` picks, by position and state,
    raise ``error``; a sibling sent in the same round is still answered. It records every request it
    receives, answered or not."""

    def __init__(
        self, script: ScriptedJevClient, fails: Callable[[int, Mapping], bool], error: BaseException
    ) -> None:
        self.script = script
        self.fails = fails
        self.error = error
        self.received: list[tuple[Mapping, Mapping]] = []
        self.model = script.model
        self._counting = threading.Lock()

    def ask(self, state: Mapping, questions: Mapping):
        with self._counting:
            self.received.append((state, questions))
            position = len(self.received)
        if self.fails(position, state):
            raise self.error
        return self.script.ask(state, questions)

    def close(self) -> None:
        pass


def provider_error() -> ProviderError:
    error = ProviderError("Jev answered 503")
    error.__cause__ = ConnectionResetError("connection reset by peer")
    return error


def hashes(requests: list[tuple[Mapping, Mapping]]) -> list[str]:
    return [request_sha256(state, questions) for state, questions in requests]


def closable(client: ScriptedJevClient) -> ScriptedJevClient:
    client.close = lambda: None
    return client


def use_clients(monkeypatch: pytest.MonkeyPatch, clients: Iterator) -> None:
    monkeypatch.setattr(cli, "load_typesafe_environment", lambda environment: None)
    monkeypatch.setattr(cli, "system_one_client", lambda environment: next(clients))


def find_command(repository: Path, output: Path, store: Path, *options: str) -> list[str]:
    command = ["find", TARGET, "--repo", str(repository), "--prefix", "app/", "--start", START]
    return [*command, "--max-calls", "5", "--out", str(output), "--answer-store", str(store), *options]


def unstarted_command(workflow: str, repository: Path, output: Path, store: Path, *options: str) -> list[str]:
    """A search that chooses its own entry point: no ``--start``."""
    command = [workflow, TARGET, "--repo", str(repository), "--prefix", "app/", "--max-calls", "none"]
    return [*command, "--out", str(output), "--answer-store", str(store), *options]


def once(fails: Callable[[int, Mapping], bool]) -> Callable[[int, Mapping], bool]:
    fired = threading.Event()

    def first_match(position: int, state: Mapping) -> bool:
        if fired.is_set() or not fails(position, state):
            return False
        fired.set()
        return True

    return first_match


def first_request(position: int, state: Mapping) -> bool:
    del state
    return position == 1


def enumerating(name: str) -> Callable[[int, Mapping], bool]:
    """A Find All enumeration batch whose items include the function ``name``."""

    def holds(position: int, state: Mapping) -> bool:
        del position
        return any(item.get("code", "").startswith(f"def {name}(") for item in state.get("items", []))

    return holds


def many_functions_repository(root: Path) -> Path:
    """The marked policy code plus 40 small functions, so Find All enumerates in three batches."""
    marked_repository(root)
    helpers = "".join(f"def helper_{number}(value):\n    return value + {number}\n\n" for number in range(40))
    commit_files(root, {"app/helpers.py": helpers})
    return root


def uninterrupted(workflow: str, repository: Path, tmp_path: Path) -> tuple[dict, ScriptedJevClient]:
    client = limit_client()
    manifest = create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        (),
        tmp_path / f"whole-{workflow}",
        SearchBudget(max_calls=None),
        client,
        answer_store=tmp_path / f"whole-{workflow}.sqlite",
        workflow=workflow,
    )
    return manifest, client


def verdicts(manifest: dict) -> list[tuple[str, str, str]]:
    """Every function Find All judged, with its verdict; whether an answer came from the store is
    left out, because a resumed run replays what the stopped run already paid for."""
    search = manifest["search"]
    judged = [*search["found"], *search["unsure"], *search["searched"]]
    return sorted((entry["source"]["file"], entry["name"], entry["verdict"]) for entry in judged)


def items_asked(requests: list[tuple[Mapping, Mapping]]) -> list[str]:
    """Each enumerated item once per request that asked it, by its content: batches regroup on Resume,
    so request hashes differ while the items asked must not."""
    items = (item for state, _ in requests for item in state.get("items", []))
    return sorted(json.dumps(item, sort_keys=True) for item in items)


def uninterrupted_requests(repository: Path, tmp_path: Path) -> tuple[dict, list[str]]:
    client = limit_client()
    manifest = create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        (START,),
        tmp_path / "whole",
        SearchBudget(max_calls=5),
        client,
        answer_store=tmp_path / "whole-answers.sqlite",
    )
    return manifest, hashes(client.requests)


def manifest_of(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text())


def journal_failures(folder: Path) -> list[dict]:
    records = [json.loads(line) for line in (folder / "journal.jsonl").read_text().splitlines()]
    return [record for record in records if record["kind"] == "failure"]


def test_a_failed_find_exits_1_with_the_error_and_its_resume_reaches_the_uninterrupted_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    whole, expected = uninterrupted_requests(repository, tmp_path)
    store = tmp_path / "answers.sqlite"
    failing = FailsOnRequest(limit_client(), target_request, provider_error())
    resuming = closable(limit_client())
    use_clients(monkeypatch, iter([failing, resuming]))
    first, second = tmp_path / "first", tmp_path / "second"

    # Act
    failed_status = cli.main(find_command(repository, first, store))
    failed_stderr = capsys.readouterr().err
    resumed_status = cli.main(find_command(repository, second, store, "--resume", str(first)))

    # Assert
    failed = manifest_of(first)
    assert failed_status == 1
    assert "Jev answered 503" in failed_stderr
    assert f"--resume {first.resolve()}" in failed_stderr
    assert failed["search"]["outcome"] == "failed"
    assert failed["search"]["failure"]["type"] == "ProviderError"
    assert failed["search"]["failure"]["message"] == "Jev answered 503"
    assert failed["search"]["failure"]["causes"] == [
        {"type": "ConnectionResetError", "message": "connection reset by peer"}
    ]
    assert failed["search"]["failure"]["request_id"] in [row["request_id"] for row in journal_failures(first)]
    assert failed["search"]["failure"]["route"] is None
    assert (first / "resume.json").is_file()
    resumed = manifest_of(second)
    assert resumed_status == 0
    assert resumed["search"]["outcome"] == whole["search"]["outcome"] == "found"
    assert resumed["search"]["found"] == whole["search"]["found"]
    sent = hashes(failing.received)
    failed_request = sent.pop(
        next(i for i, (state, _) in enumerate(failing.received) if target_request(0, state))
    )
    assert sorted(sent + hashes(resuming.requests)) == sorted(expected)
    assert failed_request in hashes(resuming.requests)


def test_a_new_find_after_a_failed_one_runs_as_usual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    store = tmp_path / "answers.sqlite"
    use_clients(
        monkeypatch,
        iter([FailsOnRequest(limit_client(), target_request, provider_error()), closable(limit_client())]),
    )
    cli.main(find_command(repository, tmp_path / "failed", store))

    # Act
    status = cli.main(find_command(repository, tmp_path / "fresh", store))

    # Assert
    assert status == 0
    assert manifest_of(tmp_path / "fresh")["search"]["outcome"] == "found"


class FailsOnARoute:
    """A routed provider: the second request's attempt on route ``backup`` fails; the others are
    answered like ``script``."""

    def __init__(self, script: ScriptedJevClient) -> None:
        self.script = script
        self.model = script.model
        self.sent = 0

    def send_with_attempts(self, state: Mapping, questions: Mapping, on_attempt=None) -> RawResponse:
        self.sent += 1
        if self.sent != 2:
            return self.script.send(state, questions)
        if on_attempt is not None:
            on_attempt(RawAttempt(1, 1.0, b"", error_type="ProviderError", error="503", route="backup"))
        raise provider_error()

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        return self.send_with_attempts(state, questions)

    def parse(self, raw: RawResponse):
        return self.script.parse(raw)

    def close(self) -> None:
        pass


def test_a_failed_request_on_a_route_names_that_route_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    use_clients(monkeypatch, iter([FailsOnARoute(limit_client())]))

    # Act
    status = cli.main(
        find_command(repository, tmp_path / "routed", tmp_path / "answers.sqlite", "--beam-width", "1")
    )

    # Assert
    assert status == 1
    assert manifest_of(tmp_path / "routed")["search"]["failure"]["route"] == "backup"


def test_a_find_that_finds_its_target_while_a_sibling_fails_reports_both_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    use_clients(monkeypatch, iter([FailsOnRequest(limit_client(), import_line, provider_error())]))
    output = tmp_path / "found"

    # Act
    status = cli.main(find_command(repository, output, tmp_path / "answers.sqlite"))

    # Assert
    manifest = manifest_of(output)
    assert status == 0
    assert manifest["search"]["outcome"] == "found"
    assert manifest["search"]["found"][0]["source"]["file"] == "app/policy.py"
    assert manifest["search"]["failure"]["message"] == "Jev answered 503"
    assert "- Failure: ProviderError: Jev answered 503" in (output / "report.md").read_text()


def test_a_failure_raised_while_handling_another_error_lists_that_error_as_its_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    error = ProviderError("Jev answered 503")
    error.__context__ = TimeoutError("read timed out")
    use_clients(monkeypatch, iter([FailsOnRequest(limit_client(), target_request, error)]))

    # Act
    cli.main(find_command(repository, tmp_path / "failed", tmp_path / "answers.sqlite"))

    # Assert
    assert manifest_of(tmp_path / "failed")["search"]["failure"]["causes"] == [
        {"type": "TimeoutError", "message": "read timed out"}
    ]


def asks(question_id: str) -> Callable[[tuple[Mapping, Mapping]], bool]:
    return lambda request: any(asked.startswith(question_id) for asked in request[1])


def test_a_find_all_whose_seed_search_fails_never_starts_its_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    failing = FailsOnRequest(limit_client(), target_request, provider_error())
    use_clients(monkeypatch, iter([failing]))
    command = find_command(repository, tmp_path / "findall", tmp_path / "answers.sqlite")
    command[0] = "findall"

    # Act
    status = cli.main(command)

    # Assert
    assert status == 1
    assert manifest_of(tmp_path / "findall")["search"]["outcome"] == "failed"
    assert not any(map(asks(cli.FIND_ALL_QUESTION.question_id), failing.received))
    assert (tmp_path / "findall" / "resume.json").is_file()


@pytest.mark.parametrize(
    ("interruption", "status"), [(provider_error(), 1), (KeyboardInterrupt(), 130)], ids=["failure", "ctrl_c"]
)
def test_an_entry_selection_stopped_by_a_failure_or_ctrl_c_resumes_to_the_uninterrupted_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: BaseException, status: int
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    whole, whole_client = uninterrupted("find", repository, tmp_path)
    store = tmp_path / "answers.sqlite"
    stopping = FailsOnRequest(limit_client(), first_request, interruption)
    resuming = closable(limit_client())
    use_clients(monkeypatch, iter([stopping, resuming]))
    first, second = tmp_path / "first", tmp_path / "second"

    # Act
    stopped_status = cli.main(unstarted_command("find", repository, first, store))
    resumed_status = cli.main(unstarted_command("find", repository, second, store, "--resume", str(first)))

    # Assert
    assert stopped_status == status
    assert manifest_of(first)["search"]["entry_selection_pending"] is True
    stop = "failed" if status == 1 else "cancelled"
    report = (first / "report.md").read_text()
    assert f"- Entry selection stopped ({stop}); Resume chooses it again." in report
    assert (first / "resume.json").is_file()
    assert resumed_status == 0
    assert manifest_of(second)["search"]["found"] == whole["search"]["found"]
    asked = hashes(stopping.received)
    stopped_request = asked.pop(0)
    assert sorted(asked + hashes(resuming.requests)) == sorted(hashes(whole_client.requests))
    assert stopped_request in hashes(resuming.requests)


@pytest.mark.parametrize(
    ("interruption", "status"), [(provider_error(), 1), (KeyboardInterrupt(), 130)], ids=["failure", "ctrl_c"]
)
def test_a_find_all_enumeration_stopped_by_a_failure_or_ctrl_c_resumes_to_the_uninterrupted_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: BaseException, status: int
) -> None:
    # Arrange
    repository = many_functions_repository(tmp_path / "repository")
    whole, whole_client = uninterrupted("findall", repository, tmp_path)
    store = tmp_path / "answers.sqlite"
    stopping = FailsOnRequest(limit_client(), once(enumerating("helper_20")), interruption)
    resuming = closable(limit_client())
    use_clients(monkeypatch, iter([stopping, resuming]))
    first, second = tmp_path / "first", tmp_path / "second"

    # Act
    stopped_status = cli.main(unstarted_command("findall", repository, first, store))
    resumed_status = cli.main(unstarted_command("findall", repository, second, store, "--resume", str(first)))

    # Assert
    stopped, resumed = manifest_of(first), manifest_of(second)
    assert stopped_status == status
    assert stopped["search"]["outcome"] == ("failed" if status == 1 else "cancelled")
    assert (first / "resume.json").is_file()
    assert resumed_status == 0
    assert resumed["search"]["outcome"] == whole["search"]["outcome"] == "scope_examined"
    assert verdicts(resumed) == verdicts(whole)
    stopped_batch = next(request for request in stopping.received if enumerating("helper_20")(0, request[0]))
    answered = [request for request in stopping.received if request is not stopped_batch]
    assert items_asked(answered + resuming.requests) == items_asked(whole_client.requests)
