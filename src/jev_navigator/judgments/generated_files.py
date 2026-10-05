"""Jev's generated-file judgment (G1): one Noul per file the scope could not decide in code.

``scope.resolve_scope`` decides a file whose shape flags it wherever code can (a linguist attribute, a
generated header, a vendored or output folder) and hands the rest on in
``ResolvedScope.awaiting_generated_judgment``. Each reaches Jev as one entry: its path, its measured
facts, up to ``MAX_IMPORTERS`` files that import it with their true count, up to ``MAX_NAMED_BY``
files that name its path with the line that names it and their true count, and two excerpts. The
line the answer draws is André's (04.10.2026, 12:55): generated means no person edits the file
as source. A file the secret scan refuses is never sent; it is named as not judged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..index import listing, tools
from ..index.code_index import CodeIndex
from ..index.file_shape import FileShape
from ..index.scope import is_test_file
from ..index.spans import TextHit
from .judge import CheckResult, Judge
from .masked_cut import masked_cut
from .questions import Check, Criterion
from .secrets import Masker, SecretInRequestError, mask_request, refuse_if_secret

FILES = "files"
EXCERPT_CHARS = 2_000
MAX_IMPORTERS = 10
MAX_NAMED_BY = 5
NAMING_LINE_CHARS = 200
NOT_JUDGED_SECRET = "not judged: the secret scan refused its entry"
_PACKAGE_ENTRY_STEMS = frozenset({"index", "__init__"})
_PATH_CHARACTERS = r"A-Za-z0-9_\-"

GENERATED_FILE = Check(
    name="generated_file",
    instructions="Look only at `{item}`. Is this file generated, meaning no person edits it as source?",
    yes=Criterion(
        "A tool writes or re-creates the file: a bundle, minified code, a code generator's output, or a"
        " recording, extract or dump that people re-create from data.",
        not_for="A long data table, inline SVG or image string that a person keeps up to date by hand.",
    ),
    no=Criterion(
        "A person writes or edits the file as source, even when it is long or holds data.",
        not_for="A recording, extract or dump that people re-create from data instead of editing it.",
    ),
)


@dataclass(frozen=True)
class GeneratedJudgments:
    """Jev's answer for each judged file, and the reason each other file was not judged."""

    judged: Mapping[str, CheckResult]
    not_judged: Mapping[str, str]


def judge_generated_files(
    judge: Judge, index: CodeIndex, awaiting: Mapping[str, FileShape]
) -> GeneratedJudgments:
    naming = files_naming(index.root, sorted(awaiting))
    entries = {
        path: generated_file_entry(index, path, awaiting[path], naming[path], judge.masker)
        for path in sorted(awaiting)
    }
    refused = {path for path, entry in entries.items() if _refused_by_secret_scan(judge, entry)}
    sendable = [path for path in entries if path not in refused]
    results = judge.check_each(GENERATED_FILE, [entries[path] for path in sendable], list_name=FILES)
    return GeneratedJudgments(
        {result.item["file"]: result for result in results}, dict.fromkeys(sorted(refused), NOT_JUDGED_SECRET)
    )


def generated_file_entry(
    index: CodeIndex, path: str, shape: FileShape, naming: Sequence[TextHit], masker: Masker | None
) -> dict:
    """One file as Jev sees it: path, measured facts, importers, the files naming it (``naming``, from
    ``files_naming``) and the two excerpts. Every field is a measurement or real text, never a verdict:
    no trigger names and no reasons. ``file_shape`` measures lines in bytes, so the line fields say so.
    Each naming line and excerpt is a ``masked_cut`` of its whole file with ``masker``."""
    importers = importers_of(index, path)
    text = "\n".join(index.lines(path))
    return {
        "file": path,
        "size_bytes": shape.size_bytes,
        "line_count": shape.line_count,
        "longest_line_bytes": shape.longest_line,
        "average_line_bytes": round(shape.chars_per_line, 1),
        "importers": list(importers[:MAX_IMPORTERS]),
        "importer_count": len(importers),
        "named_by": [_naming_entry(index.root, hit, path, masker) for hit in naming[:MAX_NAMED_BY]],
        "named_by_count": len(naming),
        **_excerpts(text, path, masker),
    }


def files_naming(root: Path, paths: Sequence[str]) -> dict[str, tuple[TextHit, ...]]:
    """For each of ``paths``, the first line of every other file in the directory listing that names
    it as a whole path, non-test files first, then by file. One ripgrep pass, which prints only paths,
    finds the files naming any of them; each path's lines are then searched in those files alone."""
    if not paths:
        return {}
    candidates = tools.ripgrep_files(paths, listing.working_files(root).files, root)
    return {path: _first_lines_naming(root, path, candidates) for path in paths}


def _first_lines_naming(root: Path, path: str, candidates: Sequence[str]) -> tuple[TextHit, ...]:
    """The first line of each candidate other than ``path`` that names it, as its file and line number
    with the name alone as text, so a one-line bundle never reaches Python whole while searching."""
    others = [file for file in candidates if file != path]
    hits = tools.ripgrep_windows(
        _whole_path(tools.literal_pattern(path)), others, root, max_hits=1, context_bytes=0
    )
    return tuple(sorted(hits, key=_naming_order))


def importers_of(index: CodeIndex, path: str) -> tuple[str, ...]:
    """The scope files whose imports resolve to ``path``: one ripgrep for its import stem narrows the
    candidates, and the index's own text import reader decides, so nothing is parsed."""
    others = [file for file in index.files if file != path]
    candidates = tools.ripgrep_files(_import_stem(path), others, index.root)
    return tuple(sorted(file for file in candidates if path in index.imports(file)))


def _import_stem(path: str) -> str:
    """The name an import of ``path`` spells: the file's stem, or its folder's name for a package entry."""
    pure = PurePosixPath(path)
    return pure.parent.name if pure.stem in _PACKAGE_ENTRY_STEMS else pure.stem


def _whole_path(literal: str) -> str:
    """The path pattern ``literal`` as a whole path, in a syntax both ripgrep and Python read, so without
    lookaround, with the path itself in the group ``path``: an optional ``./`` or ``/`` after a character
    no path holds, and no path character or file extension after it. ``lib/web/a.js``, ``web/a.json``
    and a URL ending in ``/web/a.js`` do not name ``web/a.js``; neither do ``../web/a.js`` and
    ``$root/web/a.js``, whose folder is relative or variable."""
    before = rf"(?:^|[^{_PATH_CHARACTERS}./\n])(?:\./|/)?"
    after = rf"(?:$|[^{_PATH_CHARACTERS}/.\n]|\.(?:$|[^A-Za-z0-9\n]))"
    return f"{before}(?P<path>{literal}){after}"


def _naming_order(hit: TextHit) -> tuple[bool, str]:
    return is_test_file(hit.file), hit.file


def _naming_entry(root: Path, hit: TextHit, path: str, masker: Masker | None) -> dict:
    """The naming line as Jev sees it: the whole line when it fits ``NAMING_LINE_CHARS``, otherwise
    that many characters around the name, cut from the whole file by ``masked_cut``."""
    source = (root / hit.file).read_bytes().decode(errors="replace")
    line_start, line_end = _line_span(source, hit.line)
    named = re.compile(_whole_path(re.escape(path)), re.MULTILINE).search(source, line_start, line_end)
    if named is None:
        raise ValueError(f"{hit.file}:{hit.line} no longer names {path}")
    keep = named.span("path")
    start, end = _around(keep, line_start, line_end)
    text = masked_cut(source, start, end, masker, file=hit.file, keep=keep).strip()
    return {"file": hit.file, "line": hit.line, "text": text}


def _line_span(source: str, number: int) -> tuple[int, int]:
    start = 0
    for _ in range(number - 1):
        start = source.index("\n", start) + 1
    end = source.find("\n", start)
    return start, len(source) if end < 0 else end


def _around(keep: tuple[int, int], line_start: int, line_end: int) -> tuple[int, int]:
    if line_end - line_start <= NAMING_LINE_CHARS:
        return line_start, line_end
    centre = (keep[0] + keep[1]) // 2
    start = min(max(line_start, centre - NAMING_LINE_CHARS // 2), line_end - NAMING_LINE_CHARS)
    return start, start + NAMING_LINE_CHARS


def _excerpts(text: str, path: str, masker: Masker | None) -> dict[str, str]:
    if len(text) <= 2 * EXCERPT_CHARS:
        return {"opening": masked_cut(text, 0, len(text), masker, file=path)}
    middle = (len(text) - EXCERPT_CHARS) // 2
    return {
        "opening": masked_cut(text, 0, EXCERPT_CHARS, masker, file=path),
        "middle": masked_cut(text, middle, middle + EXCERPT_CHARS, masker, file=path),
    }


def _refused_by_secret_scan(judge: Judge, entry: Mapping) -> bool:
    """Whether the judge's own mask-then-scan would refuse a request holding only this entry."""
    state: Mapping = {FILES: [entry]}
    masked: frozenset[str] = frozenset()
    if judge.masker is not None:
        state, _, masked = mask_request(state, {}, judge.masker)
    try:
        refuse_if_secret(state, {}, judge.scanner, masked)
    except SecretInRequestError:
        return True
    return False
