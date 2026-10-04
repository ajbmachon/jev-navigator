"""Choose concrete initial places for a semantic search from the tracked repository tree.

Code owns the hierarchy and every option. Jev only chooses among the actual directories, files,
and source spans code offers. Unchosen options remain in the receipt as an uninspected entry
frontier; a Choice probability orders alternatives but is never treated as a Noul probability that
the code contains the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from ..errors import JvnRefusal, UsageError
from ..index.code_index import CodeIndex
from ..index.languages import language_of
from ..index.spans import Span
from ..judgments.judge import Judge, PickResult
from ..judgments.questions import Pick
from .places import Place, function_place, range_place

MAX_OPTIONS = 200
DIRECTORY_EXAMPLES = 3
PREVIEW_LINES = 3

CHOOSE_PATH = Pick(
    name="automatic_entry_path",
    instructions=(
        "Which supplied directory or file is the best place to continue looking for the "
        "code described by `target.description`? Choose from the supplied repository entries only."
    ),
)
CHOOSE_SPAN = Pick(
    name="automatic_entry_span",
    instructions=(
        "Which supplied source span is most likely to contain the code described by "
        "`target.description`? Choose from the supplied spans only."
    ),
)


class NoSafeEntryError(JvnRefusal, RuntimeError):
    """Jev chose no path or span confidently enough to start the search there."""


@dataclass(frozen=True)
class EntryDecision:
    level: str
    parent: str
    chosen: str
    confidence: float | None
    probabilities: dict[str, float]
    request_sha256: str | None
    options: tuple[dict, ...]

    def to_json(self) -> dict:
        return {
            "level": self.level,
            "parent": self.parent,
            "chosen": self.chosen,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "request_sha256": self.request_sha256,
            "options": list(self.options),
        }


@dataclass(frozen=True)
class EntryCandidate:
    place: Place
    selection_probability: float | None


@dataclass(frozen=True)
class EntrySelection:
    selected_file: str
    candidates: tuple[EntryCandidate, ...]
    decisions: tuple[EntryDecision, ...]

    def to_json(self) -> dict:
        return {
            "selected_file": self.selected_file,
            "decisions": [decision.to_json() for decision in self.decisions],
            "candidates": [
                {
                    "place": candidate.place.key,
                    "signature": candidate.place.signature,
                    "selection_probability": candidate.selection_probability,
                    "selected": position == 0,
                }
                for position, candidate in enumerate(self.candidates)
            ],
        }


@dataclass(frozen=True)
class _PathEntry:
    kind: str
    path: str
    files: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class _SpanEntry:
    span: Span
    description: str


def choose_initial_candidates(index: CodeIndex, judge: Judge, target: str) -> EntrySelection:
    """Select one file and one span, retaining every closed-choice receipt and span alternative."""
    files = tuple(file for file in index.available_files if language_of(file))
    if not files:
        raise UsageError("the repository scope contains no supported code files")
    decisions: list[EntryDecision] = []
    parent = ""
    while True:
        entries = _path_entries(index, files, parent)
        chosen, decision = _choose_path(judge, target, parent, entries)
        decisions += decision
        if chosen.kind == "file":
            selected_file = chosen.path
            break
        parent = chosen.path

    spans = _source_spans(index, selected_file)
    if not spans:
        end = min(40, len(index.lines(selected_file)))
        candidate = EntryCandidate(
            range_place(index, selected_file, 1, end, "automatic entry selection"), None
        )
        return EntrySelection(selected_file, (candidate,), tuple(decisions))
    selected, span_decisions, probabilities = _choose_span(judge, target, selected_file, spans)
    decisions += span_decisions
    ordered = [selected, *(entry for entry in spans if entry != selected)]
    ordered[1:] = sorted(
        ordered[1:], key=lambda entry: (-probabilities.get(entry.span.key, 0.0), entry.span.key)
    )
    candidates = tuple(
        EntryCandidate(
            function_place(index, entry.span, "automatic entry selection"),
            probabilities.get(entry.span.key),
        )
        for entry in ordered
    )
    return EntrySelection(selected_file, candidates, tuple(decisions))


def _path_entries(index: CodeIndex, files: tuple[str, ...], parent: str) -> tuple[_PathEntry, ...]:
    prefix = PurePosixPath(parent).parts
    grouped: dict[tuple[str, str], list[str]] = {}
    for file in files:
        parts = PurePosixPath(file).parts
        if parts[: len(prefix)] != prefix:
            continue
        remainder = parts[len(prefix) :]
        if len(remainder) == 1:
            grouped.setdefault(("file", file), []).append(file)
        elif remainder:
            path = "/".join((*prefix, remainder[0]))
            grouped.setdefault(("directory", path), []).append(file)
    return tuple(
        _PathEntry(kind, path, tuple(paths), _path_description(index, kind, path, paths))
        for (kind, path), paths in sorted(grouped.items())
    )


def _path_description(index: CodeIndex, kind: str, path: str, files: list[str]) -> str:
    if kind == "file":
        return f"file {path}: {_preview(index, path, 1)}"
    examples = "; ".join(f"{file}: {_preview(index, file, 1)}" for file in files[:DIRECTORY_EXAMPLES])
    return f"directory {path}/ ({len(files)} code files); examples: {examples}"


def _source_spans(index: CodeIndex, file: str) -> tuple[_SpanEntry, ...]:
    unique = {
        span.key: _SpanEntry(span, _span_description(index, span))
        for span in (*index.symbols_in(file), *index.declarations_in(file))
    }
    return tuple(unique[key] for key in sorted(unique, key=lambda key: (unique[key].span.start, key)))


def _span_description(index: CodeIndex, span: Span) -> str:
    return f"{span.key}: {_preview(index, span.file, span.start)}"


def _preview(index: CodeIndex, file: str, start: int) -> str:
    lines = index.lines(file)[start - 1 : start - 1 + PREVIEW_LINES]
    return " ".join(line.strip() for line in lines if line.strip())[:360]


def _choose_path(
    judge: Judge,
    target: str,
    parent: str,
    entries: tuple[_PathEntry, ...],
) -> tuple[_PathEntry, list[EntryDecision]]:
    selected, decisions, _ = _choose(
        judge,
        CHOOSE_PATH,
        target,
        "path",
        parent or "/",
        entries,
        lambda entry: entry.description,
        lambda entry: entry.path,
    )
    return selected, decisions


def _choose_span(
    judge: Judge,
    target: str,
    file: str,
    entries: tuple[_SpanEntry, ...],
) -> tuple[_SpanEntry, list[EntryDecision], dict[str, float]]:
    return _choose(
        judge,
        CHOOSE_SPAN,
        target,
        "span",
        file,
        entries,
        lambda entry: entry.description,
        lambda entry: entry.span.key,
    )


def _choose(judge, question, target, level, parent, entries, describe, identify):
    remaining = list(entries)
    decisions: list[EntryDecision] = []
    while len(remaining) > MAX_OPTIONS:
        groups = [
            remaining[offset : offset + MAX_OPTIONS] for offset in range(0, len(remaining), MAX_OPTIONS)
        ]
        descriptions = [
            f"{level} options {identify(group[0])} through {identify(group[-1])} "
            f"({len(group)} actual entries)"
            for group in groups
        ]
        chosen_group, decision, _ = _pick(
            judge,
            question,
            target,
            f"{level}_group",
            parent,
            groups,
            descriptions,
            lambda group: f"{identify(group[0])}…{identify(group[-1])}",
        )
        decisions.append(decision)
        remaining = chosen_group
    if len(remaining) == 1:
        only = remaining[0]
        decision = EntryDecision(
            level,
            parent,
            identify(only),
            None,
            {},
            None,
            ({"id": "0", "entry": identify(only), "description": describe(only)},),
        )
        return only, [*decisions, decision], {}
    chosen, decision, probabilities = _pick(
        judge,
        question,
        target,
        level,
        parent,
        remaining,
        [describe(entry) for entry in remaining],
        identify,
    )
    by_entry = {
        identify(entry): probabilities.get(str(position), 0.0) for position, entry in enumerate(remaining)
    }
    return chosen, [*decisions, decision], by_entry


def _pick(judge, question, target, level, parent, entries, descriptions, identify):
    options = {str(position): description for position, description in enumerate(descriptions)}
    result: PickResult | None = judge.pick(
        question,
        options,
        {"target": {"description": target}, "current": parent},
    )
    if result is None:
        raise NoSafeEntryError(f"automatic entry selection had no safe {level} options")
    position = int(result.choice)
    option_rows = tuple(
        {"id": str(index), "entry": identify(entry), "description": descriptions[index]}
        for index, entry in enumerate(entries)
    )
    decision = EntryDecision(
        level,
        parent,
        identify(entries[position]),
        result.confidence,
        dict(result.probabilities),
        result.request_sha256,
        option_rows,
    )
    return entries[position], decision, dict(result.probabilities)
