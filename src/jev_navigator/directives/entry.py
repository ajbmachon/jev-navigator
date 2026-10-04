"""Choose concrete initial places for a semantic search from the tracked repository tree.

Code owns the hierarchy and every option. Jev only chooses among the actual directories, files,
and source spans code offers. Unchosen options remain in the receipt as an uninspected entry
frontier; a Choice probability orders alternatives but is never treated as a Noul probability that
the code contains the target.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..index.code_index import CodeIndex
from ..index.languages import language_of
from ..index.spans import Span
from ..judgments.client import JEV_INPUT_BOX_CHARS
from ..judgments.judge import Judge, PickResult
from ..judgments.questions import Pick, serialized_chars
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
OPTION_OVERHEAD_CHARS = serialized_chars({"199": ""})
"""What one option adds to a question beyond its description: its key, quotes, colon and comma."""
CUT_MARK = "…"
_DIRECTIVE_MARKERS = ("eslint", "noqa", "@ts-", "prettier-ignore", "pylint:", "type: ignore", "ruff:")
_BLOCK_CLOSERS = {'"""': '"""', "'''": "'''", "/**": "*/", "/*": "*/"}
_LINE_OPENERS = ("//", "#")
Mask = Callable[[str], str]

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
    mask = _mask_of(judge)
    while True:
        entries = _path_entries(files, parent)
        chosen, decision = _choose_path(
            judge, target, parent, entries, lambda shown: _path_descriptions(index, shown, mask)
        )
        decisions += decision
        if chosen.kind == "file":
            selected_file = chosen.path
            break
        parent = chosen.path

    spans = _source_spans(index, selected_file, mask)
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


def _mask_of(judge: Judge) -> Mask:
    """The judge's own masker, so what the receipt shows is what the request offers."""
    return judge.masker.mask if judge.masker is not None else str


def _path_entries(files: tuple[str, ...], parent: str) -> tuple[_PathEntry, ...]:
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
    return tuple(_PathEntry(kind, path, tuple(paths)) for (kind, path), paths in sorted(grouped.items()))


def _path_descriptions(index: CodeIndex, entries: Sequence[_PathEntry], mask: Mask) -> list[str]:
    """The descriptions of the options one request shows. Only their main files are read, so a level
    with more entries than one request reads nothing for the groups it is not asked about."""
    symbols = _symbols_of_main_files(
        index, [(entry.kind, entry.path, list(entry.files)) for entry in entries]
    )
    return [
        _path_description(index, entry.kind, entry.path, list(entry.files), symbols, mask)
        for entry in entries
    ]


def _description_limit(question: Pick, state: dict, option_count: int) -> int:
    """Each option's share of the request box once the state and the question are paid for, so the
    options together always fit it."""
    fixed = serialized_chars(state) + serialized_chars(question.to_question({}))
    share = (JEV_INPUT_BOX_CHARS - fixed) // max(option_count, 1) - OPTION_OVERHEAD_CHARS
    return max(1, min(DESCRIPTION_CHARS, share))


def _offered(judge: Judge, question: Pick, state: dict, descriptions: Sequence[str]) -> list[str]:
    """The descriptions as the request carries them and the receipt records them: masked first, then
    cut, so a cut can never split a secret the masker would have recognised."""
    mask = _mask_of(judge)
    limit = _description_limit(question, state, len(descriptions))
    return [_bounded(mask(description), limit) for description in descriptions]


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
    """Top-level symbol names per file. Every option gets its first main file read; the rest of the
    ``FILE_READ_CAP`` budget goes round-robin, one main file of every option per round, so no option
    takes the budget from the others. The files are parsed in one batch."""
    queues = [_main_files(*entry) for entry in entries]
    files = _round_robin(queues)[: max(FILE_READ_CAP, len(entries))]
    index.functions_in_files(files)
    return {file: _top_level_names(index, file) for file in files}


def _round_robin(queues: list[list[str]]) -> list[str]:
    ordered = []
    for round_number in range(max((len(queue) for queue in queues), default=0)):
        ordered += [queue[round_number] for queue in queues if round_number < len(queue)]
    return list(dict.fromkeys(ordered))


def _top_level_names(index: CodeIndex, file: str) -> tuple[str, ...]:
    names = dict.fromkeys(span.name for span in index.top_level_symbols(file) if _is_named(span.name))
    return tuple(list(names)[:SYMBOLS_PER_FILE])


def _is_named(name: str) -> bool:
    return bool(name) and not name.startswith("<")


def _path_description(
    index: CodeIndex,
    kind: str,
    path: str,
    files: list[str],
    symbols: dict[str, tuple[str, ...]],
    mask: Mask,
) -> str:
    if kind == "file":
        return _file_description(index, path, symbols, mask)
    return _directory_description(path, files, symbols)


def _file_description(index: CodeIndex, path: str, symbols: dict[str, tuple[str, ...]], mask: Mask) -> str:
    doc = _first_doc_line(_masked_lines(index.lines(path)[:DOC_SCAN_LINES], mask))
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


def _masked_lines(lines: Sequence[str], mask: Mask) -> list[str]:
    """The lines masked as one text, before any of them is cut: a secret can span lines or be
    longer than what a later cut keeps."""
    return mask("\n".join(lines)).split("\n")


def _first_doc_line(lines: Sequence[str]) -> str:
    """The first line of the first docstring or comment among the file's opening lines, without its
    markers. Imports may come first; tool directives such as ``eslint-disable`` or ``noqa`` are not docs."""
    closer: str | None = None
    for line in lines:
        text, closer = _comment_text(line.strip(), closer)
        if text and not _is_directive(text):
            return text[:DOC_LINE_CHARS]
    return ""


def _comment_text(line: str, closer: str | None) -> tuple[str | None, str | None]:
    """The text of one line of a comment and the closer of the block it leaves open. Markers are
    removed only where they belong: an opener at the start, a matching closer at the end."""
    if closer is not None:
        return _block_line(line, closer)
    if line.startswith("#!"):
        return None, None
    for opener in _BLOCK_CLOSERS:
        if line.startswith(opener):
            return _block_line(line[len(opener) :], _BLOCK_CLOSERS[opener])
    for opener in _LINE_OPENERS:
        if line.startswith(opener):
            return line.lstrip(opener[0]).strip(), None
    return None, None


def _block_line(text: str, closer: str) -> tuple[str, str | None]:
    text = text.strip()
    if closer == "*/":
        text = text.lstrip("*").strip()
    if closer in text:
        return text.split(closer, 1)[0].strip(), None
    return text, closer


def _is_directive(comment: str) -> bool:
    return any(marker in comment for marker in _DIRECTIVE_MARKERS)


def _bounded(description: str, limit: int) -> str:
    """``description`` cut so that its size in the request, ASCII-escaped as ``serialized_chars``
    measures it, is at most ``limit``: a non-ASCII character costs its whole escape."""
    if _escaped_chars(description) <= limit:
        return description
    room = limit - _escaped_chars(CUT_MARK)
    if room < 0:
        return ""
    kept: list[str] = []
    for character in description:
        room -= _escaped_chars(character)
        if room < 0:
            break
        kept.append(character)
    return "".join(kept) + CUT_MARK


def _escaped_chars(text: str) -> int:
    return serialized_chars(text) - serialized_chars("")


def _source_spans(index: CodeIndex, file: str, mask: Mask) -> tuple[_SpanEntry, ...]:
    unique = {
        span.key: _SpanEntry(span, _span_description(index, span, mask))
        for span in (*index.symbols_in(file), *index.declarations_in(file))
    }
    return tuple(unique[key] for key in sorted(unique, key=lambda key: (unique[key].span.start, key)))


def _span_description(index: CodeIndex, span: Span, mask: Mask) -> str:
    return f"{span.key}: {_preview(index, span.file, span.start, mask)}"


def _preview(index: CodeIndex, file: str, start: int, mask: Mask) -> str:
    lines = _masked_lines(index.lines(file)[start - 1 : start - 1 + SPAN_PREVIEW_LINES], mask)
    return " ".join(line.strip() for line in lines if line.strip())[:360]


def _choose_path(
    judge: Judge,
    target: str,
    parent: str,
    entries: tuple[_PathEntry, ...],
    describe: Callable[[Sequence[_PathEntry]], list[str]],
) -> tuple[_PathEntry, list[EntryDecision]]:
    selected, decisions, _ = _choose(
        judge,
        CHOOSE_PATH,
        target,
        "path",
        parent or "/",
        entries,
        describe,
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
        lambda shown: [entry.description for entry in shown],
        lambda entry: entry.span.key,
    )


def _choose(judge, question, target, level, parent, entries, describe, identify):
    """Narrow ``entries`` to one: groups of ``MAX_OPTIONS`` are chosen by their range first, and
    ``describe`` is asked only for the options of the request that shows them."""
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
        (shown,) = _offered(judge, question, _state(target, parent), describe([only]))
        decision = EntryDecision(
            level,
            parent,
            identify(only),
            None,
            {},
            None,
            ({"id": "0", "entry": identify(only), "description": shown},),
        )
        return only, [*decisions, decision], {}
    chosen, decision, probabilities = _pick(
        judge,
        question,
        target,
        level,
        parent,
        remaining,
        describe(remaining),
        identify,
    )
    by_entry = {
        identify(entry): probabilities.get(str(position), 0.0) for position, entry in enumerate(remaining)
    }
    return chosen, [*decisions, decision], by_entry


def _state(target: str, parent: str) -> dict:
    return {"target": {"description": target}, "current": parent}


def _pick(judge, question, target, level, parent, entries, descriptions, identify):
    state = _state(target, parent)
    descriptions = _offered(judge, question, state, descriptions)
    options = {str(position): description for position, description in enumerate(descriptions)}
    result: PickResult | None = judge.pick(question, options, state)
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
