"""Frozen round registration: bind a benchmark before any call, verify it before every read.

A "round" is any measurement someone may later quote: a directive's accuracy on a fixed case
list, a comparison between two samplers, a published benchmark of a question set. The failure
mode the pattern prevents is silent drift — a question reworded after a bad score, a case
added or dropped, a rule "fixed" post hoc — after which the number still looks measured but no
longer measures the claim.

A round registers, in code, the facts that define it: the case ids, the question set (hashed,
never trusted by name), the decision rule and the metadata of the run (library commit, Python
version). ``freeze`` writes those facts as a manifest whose hash a sidecar binds. ``verify``
re-derives every fact at read time and raises on any difference; a manifest edited after the
fact fails its own hash. Nothing here calls Jev or reads an answer store — the round binds
*what was asked and how it is scored*; ``judgments.store`` keeps the answers, and
``registered_request_sha256`` ties the two together through the request hash an
``AnswerRecord`` already carries.

The rule and questions are passed as JSON-serializable values, so a frozen manifest records
them byte-stably across interpreter versions. A rule that is code rather than data freezes as
its source text: pass ``rule_source=inspect.getsource(the_function)`` and ``verify`` refuses a
live registration whose source no longer matches.

usage:
    registration = RoundRegistration(case_ids=(...), questions={...}, rule={...},
                                     rule_source=inspect.getsource(rule_fn))
    freeze(round_dir, registration, verifier_report=Path("report.md"))
    verify(round_dir)                     # raises FrozenRoundError on any difference
    verify(round_dir, registration)       # also re-checks the code's own registration
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..index import tools
from .secrets import DEFAULT_MASKER, Masker, mask_request

MANIFEST_LINE = "manifest.json sha256: "
REGISTRATION_FILE = "registration.json"
MANIFEST_FILE = "manifest.json"
SIDECAR_FILE = "FROZEN.txt"
VERIFIER_REPORT_FILE = "verifier-report.txt"


class FrozenRoundError(RuntimeError):
    """The round's files or runtime differ from its registration."""


@dataclass(frozen=True)
class RoundRegistration:
    """What defines a round: the cases, the questions, the rule, and how the run was made.

    ``questions`` and ``rule`` are JSON-serializable values; ``rule_source`` freezes the
    decision rule's code text when the rule is code. ``case_ids`` is the exact, ordered
    population of the round; scoring must account for every one of them and no other.
    Supply ``library_commit`` explicitly when known; an empty value means unknown, never
    the commit of the caller's unrelated working directory.
    """

    case_ids: tuple[str, ...]
    questions: dict
    rule: dict
    rule_source: str = ""
    library_commit: str = ""
    python: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.case_ids:
            raise FrozenRoundError("a round needs case ids")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise FrozenRoundError("case ids must be unique")
        if not self.questions or not self.rule:
            raise FrozenRoundError("a round needs questions and a rule")
        if not self.python:
            object.__setattr__(self, "python", ".".join(platform.python_version_tuple()[:2]))

    def to_json(self) -> dict:
        return {
            "case_ids": list(self.case_ids),
            "questions": self.questions,
            "rule": self.rule,
            "rule_source": self.rule_source,
            "library_commit": self.library_commit,
            "python": self.python,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: dict) -> RoundRegistration:
        return cls(
            case_ids=tuple(data["case_ids"]),
            questions=data["questions"],
            rule=data["rule"],
            rule_source=data.get("rule_source", ""),
            library_commit=data.get("library_commit", ""),
            python=data.get("python", ""),
            notes=data.get("notes", ""),
        )


def freeze(round_dir: Path, registration: RoundRegistration, verifier_report: Path | None = None) -> dict:
    """Write ``registration.json``, bind it in ``manifest.json``, bind the manifest in ``FROZEN.txt``.

    The checkout state at freeze time is recorded in the manifest (commit, and whether it had
    uncommitted changes), so a freeze can always be traced to code even when made mid-work.
    The verifier report (whatever human review approved the round) is hashed into the
    manifest, so the freeze answers to it. An already-frozen round is never re-frozen; copy it
    aside and register a new round instead.
    """
    round_dir = Path(round_dir)
    if (round_dir / SIDECAR_FILE).exists():
        raise FrozenRoundError(f"{round_dir}: already frozen; a frozen round is never re-frozen")
    report_bytes = None
    if verifier_report is not None:
        try:
            report_bytes = Path(verifier_report).read_bytes()
        except OSError as exc:
            raise FrozenRoundError(f"cannot read verifier report: {exc}") from exc
    round_dir.mkdir(parents=True, exist_ok=True)
    if report_bytes is not None:
        (round_dir / VERIFIER_REPORT_FILE).write_bytes(report_bytes)
    fields = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "library_commit": registration.library_commit,
        "python": registration.python,
        "checkout_commit": _git_commit(Path.cwd()),
        "uncommitted_changes": _checkout_dirty(Path.cwd()),
        "registration_sha256": _write_json(round_dir / REGISTRATION_FILE, registration.to_json()),
        "questions_sha256": _hash_value(registration.questions),
        "verifier_report_sha256": hashlib.sha256(report_bytes).hexdigest()
        if report_bytes is not None
        else None,
    }
    manifest_sha = _write_json(round_dir / MANIFEST_FILE, fields)
    (round_dir / SIDECAR_FILE).write_text(MANIFEST_LINE + manifest_sha + "\n")
    return verify(round_dir)


def verify(round_dir: Path, registration: RoundRegistration | None = None) -> dict:
    """The manifest, after every frozen fact was re-derived and checked; raises on any difference.

    With ``registration`` given, the code's own registration must agree with the frozen one —
    this is how a checkout keeps a round registered: an edited registration fails the frozen
    manifest, and a rule's frozen source must still match. The hash chain (FROZEN.txt binds
    the manifest, the manifest binds the registration and retained report) detects changes
    beneath an unchanged sidecar. Retain that sidecar in a trusted versioned record: this
    local hash chain is not a signature and cannot detect replacement of the whole chain.
    """
    round_dir = Path(round_dir)
    sidecar, manifest_path = round_dir / SIDECAR_FILE, round_dir / MANIFEST_FILE
    if not sidecar.exists() or not manifest_path.exists():
        raise FrozenRoundError(f"{round_dir}: no frozen manifest to verify")
    bound = [
        line.removeprefix(MANIFEST_LINE).strip()
        for line in sidecar.read_text().splitlines()
        if line.startswith(MANIFEST_LINE)
    ]
    if not bound:
        raise FrozenRoundError(f"{round_dir}: {SIDECAR_FILE} binds no manifest")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != bound[-1]:
        raise FrozenRoundError("the manifest was edited after it was frozen")
    fields = json.loads(manifest_path.read_text())

    registration_path = round_dir / REGISTRATION_FILE
    if hashlib.sha256(registration_path.read_bytes()).hexdigest() != fields["registration_sha256"]:
        raise FrozenRoundError("the registration was edited after it was frozen")
    frozen = RoundRegistration.from_json(json.loads(registration_path.read_text()))

    _same("questions", _hash_value(frozen.questions), fields["questions_sha256"])
    if registration is not None:
        _same("registration", registration.to_json(), frozen.to_json())
    _same("library commit", frozen.library_commit, fields["library_commit"])
    _same("python", frozen.python, fields["python"])
    if fields.get("verifier_report_sha256") is not None:
        try:
            report_hash = hashlib.sha256((round_dir / VERIFIER_REPORT_FILE).read_bytes()).hexdigest()
        except OSError as exc:
            raise FrozenRoundError(f"cannot read frozen verifier report: {exc}") from exc
        _same("verifier report", report_hash, fields["verifier_report_sha256"])
    return fields


def registered_request_sha256(
    registration: RoundRegistration, state: dict, *, masker: Masker | None = DEFAULT_MASKER
) -> str:
    """A request is part of the round only when asked with the registered questions.

    Compare against the ``request_sha256`` an ``AnswerRecord`` (or a Journal line) already
    carries: a stored answer whose hash differs was asked under other questions and must not
    count towards the round. Use the same ``masker`` configuration as the Judge; the default
    applies its built-in secret masking before hashing.
    """
    from .questions import request_sha256

    questions = registration.questions
    if masker:
        state, questions, _ = mask_request(state, questions, masker)
    return request_sha256(state, questions)


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_value(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _checkout_dirty(repository: Path) -> bool | None:
    if _git_commit(repository) is None:
        return None  # checkout state is unknown outside a repository or before its first commit
    return bool(tools.git(["status", "--porcelain"], repository))


def _git_commit(repository: Path) -> str | None:
    """HEAD's commit; None outside a Git worktree or before its first commit. A repository git
    refuses raises, so a frozen round never records a refused checkout as having no commit."""
    if not tools.inside_git_worktree(repository):
        return None
    return tools.head_commit(repository) or None


def _same(what: str, found, registered) -> None:
    if found != registered:
        raise FrozenRoundError(f"{what} differs from the registration")
