"""The trace evidence pack: one callable, the real parser and index, scripted judgments only.

These tests prove pack delivery and evidence retention — gaps, uncertain links, the persisted JSON
and Markdown — over a real committed repository parsed by ``CodeIndex``. The scripted client makes
no claim about model answer quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.cli_trace import SCHEMA_VERSION, create_trace_evidence_pack
from jev_navigator.judgments.store import SHARED_STORE_VARIABLE
from jev_navigator.testing import ScriptedJevClient

WORKFLOW_FILES = {
    "workflow.py": (
        "from pipeline import normalize\n"
        "from delivery import reject, respond\n"
        "\n"
        "def audit(value):\n"
        "    return value\n"
        "\n"
        "def handle_order(request):\n"
        "    payload = request.body\n"
        "    normalized = normalize(payload)\n"
        "    audit(normalized)\n"
        "    if normalized['accepted']:\n"
        "        return respond(normalized)\n"
        "    return reject(normalized)\n"
        "\n"
        "HANDLERS = {'POST /orders': handle_order}\n"
    ),
    "pipeline.py": "def normalize(payload):\n    return {'accepted': payload != '', 'payload': payload}\n",
    "delivery.py": (
        "def respond(order):\n    return {'status': 201, 'body': order}\n\n"
        "def reject(order):\n    return {'status': 422, 'body': order}\n"
    ),
}

QUESTION = "How does an order request become an HTTP result?"


@pytest.mark.parametrize("json_mode", [False, True])
def test_trace_command_writes_a_real_pack_with_default_output(
    tmp_path: Path, monkeypatch, capsys, json_mode: bool
) -> None:
    from jev_navigator import cli

    repository = _workflow_repository(tmp_path)
    client = _evidence_client()
    client.close = lambda: None
    monkeypatch.setattr(cli, "TypeSafeJevClient", lambda: client)
    monkeypatch.setattr(cli, "_load_typesafe_environment", lambda environment: None)
    monkeypatch.chdir(tmp_path)
    request = {
        "command": "trace",
        "target": QUESTION,
        "repo": str(repository),
        "start": ["workflow.py:8"],
        "max_calls": 1,
    }
    argv = (
        ["--json", json.dumps(request)]
        if json_mode
        else ["trace", QUESTION, "--repo", str(repository), "--start", "workflow.py:8", "--max-calls", "1"]
    )
    assert cli.main(argv) == 0
    output = capsys.readouterr()
    packs = list((tmp_path / "jvn-results").glob("*/manifest.json"))
    assert len(packs) == 1
    manifest = json.loads(packs[0].read_text())
    assert manifest["trace"]["outcome"] == "completed"
    assert manifest["trace"]["links"]
    assert len(client.requests) == 1
    if json_mode:
        response = json.loads(output.out)
        assert Path(response["manifest"]) == packs[0]
        assert response["trace"] == manifest["trace"]
    else:
        assert "completed (1 live calls)" in output.out


def test_trace_schema_and_help_describe_the_real_start_requirement(capsys) -> None:
    from jev_navigator.cli import main

    assert main(["schema", "trace"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert set(schema["required"]) == {"target", "start"}
    assert schema["properties"]["start"]["minItems"] == 1
    assert schema["properties"]["start"]["type"] == "array"
    assert schema["examples"][0]["command"] == "trace"
    assert "max_steps" not in schema["properties"]
    assert main(["help", "trace"]) == 0
    assert "--start" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["trace", QUESTION])
    assert error.value.code == 2


def _workflow_repository(root: Path, *, registration: bool = True) -> Path:
    files = dict(WORKFLOW_FILES)
    if not registration:
        files["workflow.py"] = files["workflow.py"].replace(
            "\nHANDLERS = {'POST /orders': handle_order}\n", ""
        )
    repository = root / "repository"
    commit_files(repository, files)
    return repository


def _evidence_client() -> ScriptedJevClient:
    signals = {
        "trace_input_origin": "request.body",
        "trace_transformation": "normalize(payload)",
        "trace_handoff": "HANDLERS =",
        "trace_observable_outcome": "respond(normalized)",
        "trace_relevant_branch": "if normalized['accepted']",
    }

    def answer(question_id, _question, state):
        name = question_id.split("@", 1)[0]
        slot = int(question_id.rsplit("#", 1)[1])
        item = state["trace"][slot]
        return 0.95 if signals[name] in json.dumps(item) else 0.05

    return ScriptedJevClient(nouls=answer)


def _pack(repository: Path, output: Path, client: ScriptedJevClient, **kwargs) -> dict:
    return create_trace_evidence_pack(
        repository,
        QUESTION,
        ("workflow.py:8",),
        output,
        client,
        fact_cache_dir=repository.parent / "fact-cache",
        **kwargs,
    )


def _bulk_workflow_repository(root: Path) -> Path:
    """The workflow repository with bodies long enough to force several batched requests."""
    bulk = "x" * 30_000
    files = {
        name: source.replace("):\n", f"):\n    bulk = '{bulk}'\n") for name, source in WORKFLOW_FILES.items()
    }
    repository = root / "repository"
    commit_files(repository, files)
    return repository


def _statuses(manifest: dict) -> dict[str, str]:
    return {o["name"]: o["status"] for o in manifest["trace"]["obligations"]}


def test_pack_persists_reviewable_json_and_markdown_evidence(tmp_path: Path) -> None:
    repository = _workflow_repository(tmp_path)
    output = tmp_path / "pack"
    client = _evidence_client()

    # Act
    manifest = _pack(repository, output, client)

    # Assert: the callable returns exactly what it persisted, from the real index and graph.
    written = json.loads((output / "manifest.json").read_text())
    assert manifest == written
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["question"] == QUESTION
    assert manifest["source"]["revision"]  # committed repository revision recorded
    assert manifest["trace"]["graph_stop"] == "fixed_point"
    links = manifest["trace"]["links"]
    assert any(
        link["source"]
        and link["target"]
        and link["source"]["name"] == "handle_order"
        and link["target"]["name"] == "normalize"
        and link["binding"]["proven"]
        for link in links
    )
    assert {span["name"] for span in manifest["trace"]["included"]} >= {"handle_order", "normalize"}
    # Assert: the five typed obligations are evidence-backed, each with source-identified evidence.
    assert set(_statuses(manifest).values()) == {"evidence_backed"}
    for obligation in manifest["trace"]["obligations"]:
        assert obligation["evidence"]
        for evidence in obligation["evidence"]:
            assert evidence["source"]["file"]
            assert evidence["source"]["lines"]
            assert evidence["source"]["commit"]
            assert evidence["source"]["file_sha256"]
            assert evidence["request_sha256"]
    # Assert: one batched live request through the shared journal and answer store owners.
    assert manifest["provider"]["served_model"] == "jev-scripted"
    assert manifest["provider"]["calls"] == 1
    assert len(client.requests) == 1
    answers = (output / "answers.jsonl").read_text().strip().splitlines()
    assert answers
    journal = (output / "journal.jsonl").read_text()
    assert '"kind": "request"' in journal and '"kind": "terminal"' in journal
    report = (output / "report.md").read_text()
    assert "# Workflow trace evidence pack" in report
    assert QUESTION in report
    for obligation in manifest["trace"]["obligations"]:
        assert f"### {obligation['name']} — evidence_backed" in report
    assert "not proof of a correct handoff" in report


def test_pack_counts_responses_that_reported_no_usage_instead_of_adding_zero_tokens(tmp_path: Path) -> None:
    repository = _workflow_repository(tmp_path)
    client = _evidence_client()
    client.input_tokens_per_call = None

    manifest = _pack(repository, tmp_path / "pack", client)

    assert manifest["provider"]["calls"] > 0
    assert manifest["provider"]["responses_without_usage"] == manifest["provider"]["calls"]
    assert manifest["provider"]["input_tokens"] == 0


def test_pack_sums_the_input_tokens_the_provider_reported(tmp_path: Path) -> None:
    repository = _workflow_repository(tmp_path)
    client = _evidence_client()
    client.input_tokens_per_call = 37

    manifest = _pack(repository, tmp_path / "pack", client)

    assert manifest["provider"]["responses_without_usage"] == 0
    assert manifest["provider"]["input_tokens"] == 37 * manifest["provider"]["calls"]


def test_damaged_workflow_persists_the_specific_gap(tmp_path: Path) -> None:
    repository = _workflow_repository(tmp_path, registration=False)
    output = tmp_path / "pack"
    client = _evidence_client()

    # Act
    manifest = _pack(repository, output, client)

    # Assert: the gap survives into the persisted pack; connected-but-unevidenced stays a gap.
    statuses = _statuses(json.loads((output / "manifest.json").read_text()))
    assert statuses["trace_handoff"] == "gap_to_investigate"
    assert statuses["trace_input_origin"] == "evidence_backed"
    assert statuses["trace_transformation"] == "evidence_backed"
    assert statuses["trace_observable_outcome"] == "evidence_backed"
    handoff = next(o for o in manifest["trace"]["obligations"] if o["name"] == "trace_handoff")
    assert handoff["evidence"] == []
    report = (output / "report.md").read_text()
    assert "### trace_handoff — gap_to_investigate" in report
    assert "No supplied source crossed the yes threshold" in report


def test_candidate_and_missing_links_are_retained_in_the_persisted_pack(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "workflow.py": "def start(value):\n    notify(value)\n    missing_sink(value)\n",
            "first.py": "def notify(value):\n    return value\n",
            "second.py": "def notify(value):\n    return value\n",
        },
    )
    output = tmp_path / "pack"

    # Act
    manifest = create_trace_evidence_pack(
        repository,
        "Where is a value notified?",
        ("workflow.py:1",),
        output,
        ScriptedJevClient(default_noul=0.05),
    )

    # Assert: ambiguous name matches stay candidates; missing targets stay unresolved; both persist.
    persisted = json.loads((output / "manifest.json").read_text())["trace"]["unresolved_links"]
    notify = [link for link in persisted if link["name"] == "notify"]
    assert len(notify) == 2
    assert all(link["binding"]["status"] == "candidate" for link in notify)
    assert all(not link["binding"]["proven"] for link in notify)
    missing = next(link for link in persisted if link["name"] == "missing_sink")
    assert missing["target"] is None
    assert missing["binding"]["status"] == "unresolved"
    assert set(_statuses(manifest).values()) == {"gap_to_investigate"}


def test_cancelled_walk_writes_the_pack_without_a_single_request(tmp_path: Path) -> None:
    repository = _workflow_repository(tmp_path)
    output = tmp_path / "pack"
    client = ScriptedJevClient(default_noul=0.95)

    # Act
    manifest = _pack(repository, output, client, cancelled=lambda: True)

    # Assert: no model call at all, every obligation unresolved, and the pack is still reviewable.
    assert manifest["trace"]["outcome"] == "cancelled"
    assert not client.requests
    assert set(_statuses(manifest).values()) == {"unresolved"}
    assert all(not o["evidence"] and not o["checked"] for o in manifest["trace"]["obligations"])
    assert json.loads((output / "manifest.json").read_text())["trace"]["outcome"] == "cancelled"
    assert "Outcome: **cancelled**" in (output / "report.md").read_text()
    assert "call budget stopped" not in (output / "report.md").read_text()


def test_cancel_after_static_depth_boundary_is_not_completed(tmp_path: Path) -> None:
    repository = _workflow_repository(tmp_path)
    client = _evidence_client()
    manifest = _pack(repository, tmp_path / "pack", client, depth=0, cancelled=lambda: True)
    assert manifest["trace"]["graph_stop"] == "depth"
    assert manifest["trace"]["outcome"] == "cancelled"
    assert not client.requests


def test_cancel_between_batches_keeps_answers_and_sends_no_next_request(tmp_path: Path) -> None:
    repository = _bulk_workflow_repository(tmp_path)
    client = _evidence_client()
    manifest = _pack(repository, tmp_path / "pack", client, cancelled=lambda: bool(client.requests))
    assert manifest["trace"]["outcome"] == "cancelled"
    assert manifest["provider"]["calls"] == len(client.requests) == 1
    assert any(o["checked"] for o in manifest["trace"]["obligations"])
    assert all(not o["examined"] for o in manifest["trace"]["obligations"])


def test_explicit_depth_stop_is_preserved_as_partial_traversal(tmp_path: Path) -> None:
    repository = _workflow_repository(tmp_path)
    manifest = _pack(repository, tmp_path / "pack", _evidence_client(), depth=0)
    assert manifest["trace"]["outcome"] == "depth"


def test_budget_stop_writes_the_pack_with_honest_partial_coverage(tmp_path: Path, monkeypatch) -> None:
    repository = _bulk_workflow_repository(tmp_path)
    output = tmp_path / "pack"
    client = _evidence_client()

    # Act
    manifest = _pack(repository, output, client, max_calls=2)

    # Assert: the cap stops a later batch, but the pack is still written with what was answered.
    assert manifest["trace"]["outcome"] == "budget"
    assert manifest["provider"]["calls"] == 2
    assert len(client.requests) == 2
    obligations = manifest["trace"]["obligations"]
    assert all(not obligation["examined"] for obligation in obligations)
    assert any(obligation["evidence"] for obligation in obligations)
    for obligation in obligations:
        if obligation["evidence"]:
            assert obligation["status"] == "evidence_backed"
        else:
            # Unexamined spans stay unresolved; the pack never calls them a gap.
            assert obligation["status"] == "unresolved"
    # Uncertain static links are kept exactly as a full run reports them.
    complete = _pack(repository, tmp_path / "complete", _evidence_client())
    assert complete["trace"]["outcome"] == "completed"
    assert manifest["trace"]["unresolved_links"] == complete["trace"]["unresolved_links"]
    report = (output / "report.md").read_text()
    assert "Outcome: **budget**" in report

    # A replay over the persisted pack alone, with an empty shared store, preserves the same answers
    # with no live call at all.
    monkeypatch.setenv(SHARED_STORE_VARIABLE, str(tmp_path / "empty-shared.sqlite"))
    replayed = _pack(
        repository,
        tmp_path / "replay",
        _evidence_client(),
        max_calls=0,
        served_model="jev-scripted",
        answers_from=output / "answers.jsonl",
    )
    assert replayed["provider"]["calls"] == 0
    assert replayed["provider"]["replayed_answers"] > 0
    assert manifest["provider"]["replayed_answers"] == 0
    assert replayed["trace"]["outcome"] == "budget"
    for obligation, original in zip(replayed["trace"]["obligations"], obligations, strict=True):
        assert obligation["status"] == original["status"]
        for evidence, original_evidence in zip(obligation["evidence"], original["evidence"], strict=True):
            assert evidence["from_store"]
            assert evidence["request_sha256"] == original_evidence["request_sha256"]


def test_second_pack_replays_the_persisted_answers_from_the_store(tmp_path: Path) -> None:
    repository = _workflow_repository(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    _pack(repository, first, _evidence_client())

    # Act: a fresh output seeded with the first pack's answer store, and a fresh client.
    client = _evidence_client()
    manifest = _pack(
        repository, second, client, served_model="jev-scripted", answers_from=first / "answers.jsonl"
    )

    # Assert: identical questions about identical code replay; no new request was made.
    assert not client.requests
    assert manifest["provider"]["calls"] == 0
    assert manifest["provider"]["served_model"] == "jev-scripted"
    assert set(_statuses(manifest).values()) == {"evidence_backed"}
    assert all(
        evidence["from_store"]
        for obligation in manifest["trace"]["obligations"]
        for evidence in obligation["evidence"]
    )
