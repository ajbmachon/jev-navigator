"""Visible file summaries for a bounded navigation Choice, never a claim about unseen code."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from ..index.code_index import CodeIndex
from ..index.imports import import_lines
from ..index.languages import language_of
from ..judgments.answers import ChoiceAnswer
from .scent import words


@dataclass(frozen=True)
class OutlineLimits:
    definitions: int = 32
    imports: int = 16
    matches: int = 12
    line_chars: int = 240

    def __post_init__(self):
        if min(self.definitions, self.imports, self.matches, self.line_chars) < 1:
            raise ValueError("Outline limits must be positive")


DEFAULT_OUTLINE_LIMITS = OutlineLimits()


@dataclass(frozen=True)
class FileOutline:
    path: str
    definitions: tuple[tuple[str, int, int], ...]
    imports: tuple[tuple[int, str], ...]
    matches: tuple[tuple[int, str], ...]
    omitted: dict[str, int]
    clipped_lines: tuple[int, ...]

    def render(self) -> str:
        rows = [self.path]
        rows.extend(f"def {name}:{start}-{end}" for name, start, end in self.definitions)
        rows.extend(f"import {line}: {text}" for line, text in self.imports)
        rows.extend(f"match {line}: {text}" for line, text in self.matches)
        if any(self.omitted.values()):
            rows.append(
                "omitted " + ", ".join(f"{count} {kind}" for kind, count in self.omitted.items() if count)
            )
        if self.clipped_lines:
            rows.append("clipped lines: " + ", ".join(map(str, self.clipped_lines)))
        return "\n".join(rows)


def outline(
    index: CodeIndex,
    path: str,
    terms: Sequence[str] = (),
    *,
    limits: OutlineLimits = DEFAULT_OUTLINE_LIMITS,
) -> FileOutline:
    """Read and parse only this file. Every clipping or omission is explicit."""
    lines = index.lines(path)
    definitions = ()
    if language_of(path):
        spans = sorted(set((*index.symbols_in(path), *index.declarations_in(path))))
        definitions = tuple((span.name, span.start, span.end) for span in spans if span.is_named)
    source = "\n".join(lines)
    import_numbers = import_lines(source, path) if language_of(path) else frozenset()
    imported = [(number, lines[number - 1].strip()) for number in sorted(import_numbers)]
    query = set(words(" ".join(terms)))
    matches = [
        (number, text.strip()) for number, text in enumerate(lines, 1) if query.intersection(words(text))
    ]
    visible = [*imported[: limits.imports], *matches[: limits.matches]]
    clipped = tuple(dict.fromkeys(number for number, text in visible if len(text) > limits.line_chars))
    return FileOutline(
        path,
        definitions[: limits.definitions],
        tuple((number, text[: limits.line_chars]) for number, text in imported[: limits.imports]),
        tuple((number, text[: limits.line_chars]) for number, text in matches[: limits.matches]),
        {
            "definitions": max(0, len(definitions) - limits.definitions),
            "imports": max(0, len(imported) - limits.imports),
            "matches": max(0, len(matches) - limits.matches),
        },
        clipped,
    )


def outline_choice(query: str, files: Sequence[FileOutline]) -> dict:
    """Prepare one Choice among 1..16 visible outlines and none; no provider call."""
    if not 1 <= len(files) <= 16 or len({file.path for file in files}) != len(files):
        raise ValueError("Offer between one and sixteen distinct file outlines")
    state = {"query": query, "files": {str(i): asdict(file) for i, file in enumerate(files, 1)}}
    question = {
        "type": "choice",
        "instructions": (
            "Which numbered file in `files` has the strongest visible evidence for where to read the "
            "implementation described by `query`? Compare its definitions, imports and matching lines. "
            "Choose a place to read, not whether the described behavior is correct. Use none when no "
            "outline shows a connection. Omitted or clipped content supplies no evidence."
        ),
        "criteria": {
            **{str(i): f"Read file {i}, whose outline is files.{i}." for i in range(1, len(files) + 1)},
            "none": "None of these outlines shows a connection to the requested implementation.",
        },
    }
    return {"state": state, "questions": {"read_file": question}}


def selected_file(answer: ChoiceAnswer, files: Sequence[FileOutline], *, min_confidence: float) -> str | None:
    """A caller's measured confidence policy admits a read. None keeps navigation unresolved."""
    options = {str(i): file.path for i, file in enumerate(files, 1)}
    if answer.choice not in {*options, "none"}:
        raise ValueError("The selected option was not offered")
    if answer.confidence < min_confidence or answer.choice == "none":
        return None
    return options[answer.choice]
