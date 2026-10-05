"""Jev's generated-file judgment (G1): one Noul per file the scope could not decide in code.

``scope.resolve_scope`` decides a file whose shape flags it wherever code can (a linguist attribute, a
generated header, a vendored or output folder) and hands the rest on in
``ResolvedScope.awaiting_generated_judgment``. Each reaches Jev as one entry: its path, its measured
facts, up to ``MAX_IMPORTERS`` files that import it with their true count, up to ``MAX_NAMED_BY``
files that name its path with the line that names it and their count when searched (a file edited
since then is left out of the lines), and two excerpts. The
line the answer draws is André's (04.10.2026, 12:55): generated means no person edits the file
as source. A file the secret scan refuses is never sent; it is named as not judged.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path, PurePosixPath

from ..index import listing, tools
from ..index.code_index import CodeIndex
from ..index.file_shape import FileShape
from ..index.scope import is_test_file
from ..index.spans import TextHit
from .judge import CheckResult, Judge
from .masked_text import masked_lines
from .questions import Check, Criterion
from .secrets import SecretInRequestError, mask_request, refuse_if_secret

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
    namer_lines = cache(partial(_masked_namer_lines, index))
    entries = {
        path: generated_file_entry(index, path, awaiting[path], naming[path], namer_lines)
        for path in sorted(awaiting)
    }
    refused = {path for path, entry in entries.items() if _refused_by_secret_scan(judge, entry)}
    sendable = [path for path in entries if path not in refused]
    results = judge.check_each(GENERATED_FILE, [entries[path] for path in sendable], list_name=FILES)
    return GeneratedJudgments(
        {result.item["file"]: result for result in results}, dict.fromkeys(sorted(refused), NOT_JUDGED_SECRET)
    )


def generated_file_entry(
    index: CodeIndex,
    path: str,
    shape: FileShape,
    naming: Sequence[TextHit],
    namer_lines: Callable[[str], tuple[str, ...]] | None = None,
) -> dict:
    """One file as Jev sees it: path, measured facts, importers, the files naming it (``naming``, from
    ``files_naming``) and the two excerpts. Every field is a measurement or real text, never a verdict:
    no trigger names and no reasons. ``file_shape`` measures lines in bytes, so the line fields say so.
    Each naming line and excerpt is cut from its whole file masked by the index's masker;
    ``namer_lines`` gives a naming file's masked lines, so a caller can mask each file once."""
    lines_of = namer_lines or partial(_masked_namer_lines, index)
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
        "named_by": [
            entry for hit in naming[:MAX_NAMED_BY] if (entry := _naming_entry(lines_of(hit.file), hit, path))
        ],
        "named_by_count": len(naming),
        **_excerpts(text),
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


def _masked_namer_lines(index: CodeIndex, file: str) -> tuple[str, ...]:
    """A naming file's lines, masked as one text by the index's masker; it may lie outside the scope."""
    source = (index.root / file).read_bytes().decode(errors="replace")
    return masked_lines(source.split("\n"), file, index.masker, index.hidden)


def _naming_entry(lines: Sequence[str], hit: TextHit, path: str) -> dict | None:
    """The hit's line from the naming file's masked ``lines`` as Jev sees it: whole when it fits
    ``NAMING_LINE_CHARS``, otherwise that many characters around the name. None when the file no
    longer names ``path`` on that line, because it changed after the search."""
    line = lines[hit.line - 1] if hit.line <= len(lines) else ""
    named = re.compile(_whole_path(re.escape(path))).search(line)
    if named is None:
        return None
    keep = named.span("path")
    start, end = _around(keep, len(line))
    return {"file": hit.file, "line": hit.line, "text": line[start:end].strip()}


def _around(keep: tuple[int, int], line_length: int) -> tuple[int, int]:
    if line_length <= NAMING_LINE_CHARS:
        return 0, line_length
    centre = (keep[0] + keep[1]) // 2
    start = min(max(0, centre - NAMING_LINE_CHARS // 2), line_length - NAMING_LINE_CHARS)
    return start, start + NAMING_LINE_CHARS


def _excerpts(text: str) -> dict[str, str]:
    if len(text) <= 2 * EXCERPT_CHARS:
        return {"opening": text}
    middle = (len(text) - EXCERPT_CHARS) // 2
    return {
        "opening": text[:EXCERPT_CHARS],
        "middle": text[middle : middle + EXCERPT_CHARS],
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
