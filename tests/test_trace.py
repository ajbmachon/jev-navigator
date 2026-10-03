from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import BudgetedClient

from jev_navigator.directives.trace import (
    TRACE_EVIDENCE_CHECKS,
    EvidenceStatus,
    trace_workflow,
)
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import CodeSlice, Span
from jev_navigator.judgments.client import JEV_INPUT_BOX_CHARS, MAX_REQUEST_CHARS
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.testing import ScriptedJevClient


def test_trace_batches_inventory_before_resolving_each_name(tmp_path, monkeypatch):
    from jev_navigator import operations
    from jev_navigator.index import tools

    index = _workflow_index(tmp_path)
    root = index.enclosing_symbol("workflow.py", 7)
    searches = []
    run = tools.run_command

    def observe(arguments, *args, **kwargs):
        if arguments[0] == tools.RIPGREP:
            searches.append(arguments)
        return run(arguments, *args, **kwargs)

    monkeypatch.setattr(tools, "run_command", observe)
    graph = operations.trace_graph(index, (root,))
    assert {span.name for span in graph.functions} == {
        "handle_order",
        "audit",
        "normalize",
        "respond",
        "reject",
    }
    assert graph.stop == "fixed_point"
    assert searches == [], "A complete static walk needs one fact inventory, not a search per name"


def _workflow_source(*, transformation: bool, registration: bool, consumer: bool) -> str:
    imports = ["from pipeline import normalize"] if transformation else []
    if consumer:
        imports.append("from delivery import reject, respond")
    lines = [*imports, "", "def audit(value):", "    return value", "", "def handle_order(request):"]
    lines += ["    payload = request.body"]
    lines += ["    normalized = normalize(payload)" if transformation else "    normalized = payload"]
    lines += ["    audit(normalized)", "    if normalized['accepted']:"]
    if consumer:
        lines += ["        return respond(normalized)", "    return reject(normalized)"]
    else:
        lines += ["        return normalized", "    return normalized"]
    if registration:
        lines += ["", "HANDLERS = {'POST /orders': handle_order}"]
    return "\n".join(lines) + "\n"


def _workflow_index(
    root: Path, *, transformation: bool = True, registration: bool = True, consumer: bool = True
) -> CodeIndex:
    (root / "workflow.py").write_text(
        _workflow_source(
            transformation=transformation,
            registration=registration,
            consumer=consumer,
        )
    )
    (root / "pipeline.py").write_text(
        "def normalize(payload):\n    return {'accepted': payload != '', 'payload': payload}\n"
    )
    (root / "delivery.py").write_text(
        "def respond(order):\n"
        "    return {'status': 201, 'body': order}\n\n"
        "def reject(order):\n"
        "    return {'status': 422, 'body': order}\n"
    )
    return CodeIndex.from_directory(root)


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


def _trace(index: CodeIndex):
    client = _evidence_client()
    root = index.find_definition("handle_order")[0]
    result = trace_workflow(index, Judge(client), "How does an order request become an HTTP result?", [root])
    return result, client


def _statuses(result) -> dict[str, EvidenceStatus]:
    return {obligation.name: obligation.status for obligation in result.obligations}


def test_complete_workflow_has_source_backed_evidence_for_each_atomic_obligation(
    tmp_path: Path,
) -> None:
    index = _workflow_index(tmp_path)

    # Act
    result, client = _trace(index)

    # Assert: the real parser and binding graph establish connectivity independently of Jev.
    assert result.graph.stop == "fixed_point"
    assert {span.name for span in result.graph.functions} == {
        "audit",
        "handle_order",
        "normalize",
        "reject",
        "respond",
    }
    assert {
        (link.source.name, link.target.name)
        for link in result.graph.links
        if link.source is not None
        and link.target is not None
        and link.binding is not None
        and link.binding.proven
        and link.relation == "call"
    } >= {
        ("handle_order", "normalize"),
        ("handle_order", "respond"),
        ("handle_order", "reject"),
    }
    registration = next(link for link in result.graph.links if link.relation == "collection")
    assert registration.target is not None and registration.target.name == "handle_order"
    assert registration.binding is not None and registration.binding.proven

    assert _statuses(result) == {
        check.name: EvidenceStatus.EVIDENCE_BACKED for check in TRACE_EVIDENCE_CHECKS
    }
    # Assert: request transport only. The five typed obligations reach the external Jev client port
    # together in one batched request, which asks every question family about every supplied item.
    assert len(client.requests) == 1, "the five evidence duties travel together, not one round trip each"
    state, questions = client.requests[0]
    assert state["workflow"] == {"question": "How does an order request become an HTTP result?"}
    assert state["trace"], "the batched request carries the supplied items"
    for slot in range(len(state["trace"])):
        asked = {f"{check.question_id}#{slot}" for check in TRACE_EVIDENCE_CHECKS}
        assert asked <= set(questions), f"item {slot} is not asked about by every question family"
    for obligation in result.obligations:
        assert obligation.evidence
        for evidence in obligation.evidence:
            assert evidence.item["file"]
            assert evidence.item["lines"]
            assert "commit" in evidence.item
            assert evidence.item["file_sha256"]
            assert evidence.request_sha256
    assert {span.name for span in result.excluded} == {"audit", "reject"}


@pytest.mark.parametrize(
    ("missing", "expected_gap"),
    [
        ("transformation", "trace_transformation"),
        ("registration", "trace_handoff"),
        ("consumer", "trace_observable_outcome"),
    ],
)
def test_damaged_real_workflow_reports_the_specific_missing_evidence(
    tmp_path: Path, missing: str, expected_gap: str
) -> None:
    index = _workflow_index(
        tmp_path,
        transformation=missing != "transformation",
        registration=missing != "registration",
        consumer=missing != "consumer",
    )

    # Act
    result, _ = _trace(index)

    # Assert
    statuses = _statuses(result)
    assert statuses[expected_gap] == EvidenceStatus.GAP_TO_INVESTIGATE
    assert statuses["trace_input_origin"] == EvidenceStatus.EVIDENCE_BACKED
    assert statuses["trace_relevant_branch"] == EvidenceStatus.EVIDENCE_BACKED
    if missing != "transformation":
        assert statuses["trace_transformation"] == EvidenceStatus.EVIDENCE_BACKED
    if missing != "registration":
        assert statuses["trace_handoff"] == EvidenceStatus.EVIDENCE_BACKED
    if missing != "consumer":
        assert statuses["trace_observable_outcome"] == EvidenceStatus.EVIDENCE_BACKED


def test_unsure_judgment_stays_unresolved_instead_of_becoming_a_gap(tmp_path: Path) -> None:
    index = _workflow_index(tmp_path)
    client = ScriptedJevClient(default_noul=0.5)
    root = index.find_definition("handle_order")[0]

    # Act
    result = trace_workflow(index, Judge(client), "How are orders handled?", [root])

    # Assert
    assert {obligation.status for obligation in result.obligations} == {EvidenceStatus.UNRESOLVED}
    assert all(obligation.unresolved for obligation in result.obligations)
    assert result.included == (root,)
    assert {span.name for span in result.excluded} == {"audit", "normalize", "reject", "respond"}


def test_name_only_and_missing_targets_remain_unresolved_static_links(tmp_path: Path) -> None:
    (tmp_path / "workflow.py").write_text("def start(value):\n    notify(value)\n    missing_sink(value)\n")
    (tmp_path / "first.py").write_text("def notify(value):\n    return value\n")
    (tmp_path / "second.py").write_text("def notify(value):\n    return value\n")
    index = CodeIndex.from_directory(tmp_path)
    root = index.find_definition("start")[0]

    # Act
    result = trace_workflow(
        index,
        Judge(ScriptedJevClient(default_noul=0.05)),
        "Where is a value notified?",
        [root],
    )

    # Assert
    notify = [link for link in result.unresolved_links if link.name == "notify"]
    assert len(notify) == 2
    assert all(link.binding is not None and link.binding.status == "candidate" for link in notify)
    missing = next(link for link in result.unresolved_links if link.name == "missing_sink")
    assert missing.target is None
    assert missing.binding is not None and missing.binding.status == "unresolved"


def test_trace_requires_a_concrete_start_instead_of_inventing_one(tmp_path: Path) -> None:
    index = _workflow_index(tmp_path)

    with pytest.raises(ValueError, match="concrete start"):
        trace_workflow(index, Judge(ScriptedJevClient()), "How are orders handled?", [])


def _bulky_workflow_index(root: Path) -> CodeIndex:
    """The same real workflow with bodies too long for two to share a Judge batch."""
    _workflow_index(root)
    bulk = "x" * (JEV_INPUT_BOX_CHARS * 3 // 5)
    for path in sorted(root.glob("*.py")):
        path.write_text(path.read_text().replace("):\n", f"):\n    bulk = '{bulk}'\n"))
    return CodeIndex.from_directory(root)


def test_budget_stop_keeps_answered_batches_and_never_marks_the_rest_a_gap(tmp_path: Path) -> None:
    index = _bulky_workflow_index(tmp_path)
    root = index.find_definition("handle_order")[0]
    provider = _evidence_client()
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    judge = Judge(provider, max_calls=2, store=store)

    # Act
    result = trace_workflow(index, judge, "How does an order request become an HTTP result?", [root])

    # Assert: the cap stops a later batch, yet the answered batches survive with their evidence.
    assert result.budget_stopped
    assert judge.calls == 2
    assert len(provider.requests) == 2
    assert {len(obligation.checked) for obligation in result.obligations} == {2}
    for obligation in result.obligations:
        assert not obligation.examined
        if obligation.evidence:
            assert obligation.status == EvidenceStatus.EVIDENCE_BACKED
        else:
            # Unexamined spans stay unresolved; their silence is never reported as a gap.
            assert obligation.status == EvidenceStatus.UNRESOLVED
        assert obligation.status != EvidenceStatus.GAP_TO_INVESTIGATE
    # The static graph and its links are untouched by the budget stop.
    complete = trace_workflow(
        index, Judge(_evidence_client()), "How does an order request become an HTTP result?", [root]
    )
    assert result.unresolved_links == complete.unresolved_links

    # Cached answers replay the preserved evidence without a single live call.
    replay_judge = Judge(
        ScriptedJevClient(default_noul=0.05),
        max_calls=0,
        store=store,
        served_model=provider.model,
    )
    replayed = trace_workflow(index, replay_judge, "How does an order request become an HTTP result?", [root])
    assert replay_judge.calls == 0
    assert replayed.budget_stopped
    for obligation, original in zip(replayed.obligations, result.obligations, strict=True):
        assert [answer.item["span_key"] for answer in obligation.checked] == [
            answer.item["span_key"] for answer in original.checked
        ]
        assert all(answer.from_store for answer in obligation.checked)
        assert obligation.status == original.status


def test_cancelled_walk_builds_no_trace_items_or_slices_before_its_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _workflow_index(tmp_path)
    client = ScriptedJevClient(default_noul=0.95)
    root = index.find_definition("handle_order")[0]
    slices: list[Span] = []
    original = index.read_slice

    def counting(span: Span, origin: str = "") -> CodeSlice:
        slices.append(span)
        return original(span, origin=origin)

    monkeypatch.setattr(index, "read_slice", counting)

    # Act
    result = trace_workflow(index, Judge(client), "How are orders handled?", [root], cancelled=lambda: True)

    # Assert: the stop is checked before any parser slice or item is built.
    assert result.graph.stop == "cancelled"
    assert not slices
    assert not client.requests
    assert {obligation.status for obligation in result.obligations} == {EvidenceStatus.UNRESOLVED}


def test_cancelled_static_walk_does_not_start_external_judgments(tmp_path: Path) -> None:
    index = _workflow_index(tmp_path)
    client = ScriptedJevClient(default_noul=0.95)
    root = index.find_definition("handle_order")[0]

    # Act
    result = trace_workflow(
        index,
        Judge(client),
        "How are orders handled?",
        [root],
        cancelled=lambda: True,
    )

    # Assert
    assert result.graph.stop == "cancelled"
    assert not client.requests
    assert {obligation.status for obligation in result.obligations} == {EvidenceStatus.UNRESOLVED}


def _hub_index(root: Path, callers: int, line_chars: int) -> CodeIndex:
    """A hub span referenced from `callers` long call sites: the sanitized failing shape."""
    (root / "hub.py").write_text("def hub(request):\n    return request.body\n")
    lines = []
    for index in range(callers):
        pad = "x" * line_chars
        lines.append(f"def caller_{index}(request):\n    hub('{pad}')\n")
    (root / "callers.py").write_text("\n".join(lines))
    return CodeIndex.from_directory(root)


def test_a_hub_item_keeps_every_link_fact_without_the_repeated_identity_boilerplate(tmp_path: Path) -> None:
    """The saved trace run's request 5 was one 227-char span whose 165 links carried 138,002
    characters of per-link identity boilerplate; no input budget could carry it."""
    index = _hub_index(tmp_path, callers=150, line_chars=200)
    root = index.find_definition("hub")[0]
    client = BudgetedClient(MAX_REQUEST_CHARS)

    result = trace_workflow(index, Judge(client), "How does a request become a result?", [root])

    assert result.budget_stopped is False
    assert client.refusals == 0, "no request over the measured input budget is ever sent"
    hub_item = next(
        item
        for state, _ in client.requests
        for item in state["trace"]
        if item.get("span_key") == "hub.py:1-2"
    )
    assert len(hub_item["links"]) == 150
    assert all(isinstance(link, str) for link in hub_item["links"])
    dense = "\n".join(hub_item["links"])
    for fact in ("self", "self#hub", "caller_7", "call", "at callers.py:", "binding"):
        assert fact in dense
    for obligation in result.obligations:
        assert obligation.status is EvidenceStatus.EVIDENCE_BACKED


def test_class_trace_assigns_method_evidence_to_its_lexical_owner(tmp_path: Path) -> None:
    """A class with many methods must not repeat every method reference in one giant class item."""
    names = [f"target_{number}" for number in range(12)]
    (tmp_path / "targets.py").write_text(
        "\n".join(f"def {name}(): return {number}" for number, name in enumerate(names))
    )
    methods = [
        f"    def method_{number}(self):\n        return ({', '.join(names)})  # {'x' * 180}\n"
        for number in range(50)
    ]
    (tmp_path / "hub.py").write_text("class Hub:\n" + "\n".join(methods))
    index = CodeIndex.from_directory(tmp_path)
    hub = index.find_definition("Hub")[0]
    client = BudgetedClient(MAX_REQUEST_CHARS)

    result = trace_workflow(index, Judge(client), "Which methods use the targets?", [hub])

    assert client.refusals == 0
    assert all(obligation.examined for obligation in result.obligations)
    assert {link.target.name for link in result.graph.links if link.source == hub} >= {
        "method_0",
        "method_49",
    }
    assert not any(link.source == hub and link.name in names for link in result.graph.links)
    assert sum(link.name in names for link in result.graph.links) >= 50 * len(names)
