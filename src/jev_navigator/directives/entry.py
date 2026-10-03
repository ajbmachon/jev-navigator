"""Choose concrete initial places for a semantic search from the tracked repository tree.

Code owns the hierarchy and every option. Jev only chooses among the actual directories, files,
and source spans code offers. Unchosen options remain in the receipt as an uninspected entry
frontier; a Choice probability orders alternatives but is never treated as a Noul probability that
the code contains the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from ..index.code_index import CodeIndex
from ..index.languages import language_of
from ..index.spans import Span
from ..judgments.client import JEV_INPUT_BOX_CHARS
from ..judgments.judge import Judge, PickResult
from ..judgments.questions import Pick
from .places import Place, function_place, range_place

MAX_OPTIONS = 200
SPAN_PREVIEW_LINES = 3
LISTED_PATHS = 12
MAIN_FILES = 6
DIRECT_MAIN_FILES = 2
SYMBOLS_PER_FILE = 8
FILE_READ_CAP = 30
DOC_LINE_CHARS = 120
DOC_SCAN_LINES = 40
DESCRIPTION_CHARS = 2_000
REQUEST_RESERVE_CHARS = 4_000
_DIRECTIVE_MARKERS = ("eslint", "noqa", "@ts-", "prettier-ignore", "pylint:", "type: ignore", "ruff:")
_DOC_MARKERS = ('"""', "'''", "/**", "//", "#", "*")

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
        raise ValueError("the repository scope contains no supported code files")
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
    ordered = sorted(grouped.items())
    symbols = _symbols_of_main_files(index, [(kind, path, paths) for (kind, path), paths in ordered])
    limit = _description_limit(len(ordered))
    return tuple(
        _PathEntry(kind, path, tuple(paths), _path_description(index, kind, path, paths, symbols, limit))
        for (kind, path), paths in ordered
    )


def _description_limit(option_count: int) -> int:
    """Each option's share of the request box, so the options together always fit it."""
    share = (JEV_INPUT_BOX_CHARS - REQUEST_RESERVE_CHARS) // max(option_count, 1)
    return min(DESCRIPTION_CHARS, share)


def _main_files(kind: str, path: str, files: list[str]) -> list[str]:
    """A file's own entry; for a directory, a few direct files and one file per subfolder, chosen by
    name alone: shallowest first, package markers left out."""
    if kind == "file":
        return [path]
    root = _display_root(path, files)
    candidates = [file for file in files if PurePosixPath(file).name != "__init__.py"]
    direct = sorted(file for file in candidates if _below(root, file) == 1)
    deeper = sorted(file for file in candidates if _below(root, file) > 1)
    representatives = {}
    for file in sorted(deeper, key=lambda file: (_below(root, file), file)):
        representatives.setdefault(PurePosixPath(file).relative_to(root).parts[0], file)
    ordered = [*direct[:DIRECT_MAIN_FILES], *(representatives[name] for name in sorted(representatives))]
    return [*ordered, *direct[DIRECT_MAIN_FILES:]][:MAIN_FILES]


def _below(root: str, file: str) -> int:
    return len(PurePosixPath(file).relative_to(root).parts)


def _display_root(path: str, files: list[str]) -> str:
    """The directory a listing is relative to: a chain of folders holding nothing but one subfolder is
    skipped, so ``src/main/java`` shows what is inside it and not three levels of nothing."""
    root = path
    while True:
        if any(_below(root, file) == 1 for file in files):
            return root
        children = {PurePosixPath(file).relative_to(root).parts[0] for file in files}
        if len(children) != 1:
            return root
        root = f"{root}/{children.pop()}"


def _symbols_of_main_files(
    index: CodeIndex, entries: list[tuple[str, str, list[str]]]
) -> dict[str, tuple[str, ...]]:
    """Top-level symbol names per file, reading at most ``FILE_READ_CAP`` files for the whole request,
    one main file of every option per round so no option takes the budget from the others."""
    queues = [_main_files(*entry) for entry in entries]
    symbols: dict[str, tuple[str, ...]] = {}
    for round_number in range(max((len(queue) for queue in queues), default=0)):
        for queue in queues:
            if len(symbols) >= FILE_READ_CAP:
                return symbols
            if round_number < len(queue):
                symbols[queue[round_number]] = _top_level_names(index, queue[round_number])
    return symbols


def _top_level_names(index: CodeIndex, file: str) -> tuple[str, ...]:
    names: list[str] = []
    outer_end = 0
    for span in sorted(index.symbols_in(file), key=lambda span: (span.start, -span.end)):
        if span.start <= outer_end or not _is_named(span.name) or span.name in names:
            continue
        names.append(span.name)
        outer_end = span.end
    return tuple(names[:SYMBOLS_PER_FILE])


def _is_named(name: str) -> bool:
    return bool(name) and not name.startswith("<")


def _path_description(
    index: CodeIndex,
    kind: str,
    path: str,
    files: list[str],
    symbols: dict[str, tuple[str, ...]],
    limit: int,
) -> str:
    if kind == "file":
        return _bounded(_file_description(index, path, symbols), limit)
    return _bounded(_directory_description(path, files, symbols), limit)


def _file_description(index: CodeIndex, path: str, symbols: dict[str, tuple[str, ...]]) -> str:
    doc = _first_doc_line(index.lines(path)[:DOC_SCAN_LINES])
    names = symbols.get(path)
    parts = [f"file {path}:"]
    if doc:
        parts.append(doc)
    if names:
        parts.append(f"Symbols: {', '.join(names)}")
    return " ".join(parts)


def _directory_description(path: str, files: list[str], symbols: dict[str, tuple[str, ...]]) -> str:
    root = _display_root(path, files)
    relative = sorted((str(PurePosixPath(file).relative_to(root)) for file in files), key=_by_depth_then_name)
    parts = [f"directory {path}/ ({len(files)} code files)."]
    if root != path:
        parts.append(f"All of it is under {root}/.")
    folders = _subfolder_counts(relative)
    if folders:
        parts.append(f"Subfolders: {_listed([f'{name}/ ({count})' for name, count in folders])}.")
    parts.append(f"Files: {_listed(relative)}.")
    main = [
        f"{PurePosixPath(file).relative_to(root)}: {', '.join(symbols[file])}"
        for file in _main_files("directory", path, files)
        if symbols.get(file)
    ]
    if main:
        parts.append(f"Main files: {'; '.join(main)}")
    return " ".join(parts)


def _by_depth_then_name(relative: str) -> tuple[int, str]:
    return len(PurePosixPath(relative).parts), relative


def _subfolder_counts(relative: list[str]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for file in relative:
        parts = PurePosixPath(file).parts
        if len(parts) > 1:
            counts[parts[0]] = counts.get(parts[0], 0) + 1
    return sorted(counts.items())


def _listed(items: list[str]) -> str:
    shown = ", ".join(items[:LISTED_PATHS])
    return shown if len(items) <= LISTED_PATHS else f"{shown} (+{len(items) - LISTED_PATHS} more)"


def _first_doc_line(lines: tuple[str, ...]) -> str:
    """The first line of the first docstring or comment among the file's opening lines, without its
    markers. Imports may come first; tool directives such as ``eslint-disable`` or ``noqa`` are not docs."""
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(_DOC_MARKERS) or stripped.startswith("#!") or _is_directive(stripped):
            continue
        text = stripped.strip("\"'/#* ").strip()
        if text and not _is_directive(text):
            return text[:DOC_LINE_CHARS]
    return ""


def _is_directive(comment: str) -> bool:
    return any(marker in comment for marker in _DIRECTIVE_MARKERS)


def _bounded(description: str, limit: int) -> str:
    return description if len(description) <= limit else description[: limit - 1] + "…"


def _source_spans(index: CodeIndex, file: str) -> tuple[_SpanEntry, ...]:
    unique = {
        span.key: _SpanEntry(span, _span_description(index, span))
        for span in (*index.symbols_in(file), *index.declarations_in(file))
    }
    return tuple(unique[key] for key in sorted(unique, key=lambda key: (unique[key].span.start, key)))


def _span_description(index: CodeIndex, span: Span) -> str:
    return f"{span.key}: {_preview(index, span.file, span.start)}"


def _preview(index: CodeIndex, file: str, start: int) -> str:
    lines = index.lines(file)[start - 1 : start - 1 + SPAN_PREVIEW_LINES]
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
        raise RuntimeError(f"automatic entry selection had no safe {level} options")
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
