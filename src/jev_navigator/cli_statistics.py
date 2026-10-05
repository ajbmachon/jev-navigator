"""Write a structural statistics pack (JSON + Markdown) from the index the caller already has.

One call measures a repository's functions and classes through ``directives.statistics`` — the owner
of counting and ranking, whose docstring this module defers to — and persists what it measured in
``statistics.json`` and ``statistics.md``. Nothing here parses source a second time, greps source
text, counts symbols by its own arithmetic or calls a model: the scan, the counts and the ranking
belong to ``directives.statistics`` and ``index.code_index.CodeIndex``, which are asked for the
scope's facts once per pack rather than once per file or once per symbol.

What is persisted stays honest about its own coverage. A file with no grammar to read (a document, a
manifest) is ``skipped``; a file the index never inventoried, or that vanished from disk before the
scan, is ``unmeasured`` and what it holds is unknown rather than absent; a file the grammar only
partly recovered is ``unparsed``, and the symbols recovered there are a floor, never a complete
count. A ``limit`` shortens the ranking that is *shown*, never what was *measured*, and both numbers
are written down beside each other so a reader can tell them apart — including when they are equal,
since a ranking that stops short of the end of a file could always hide a narrower symbol in it.

Where the index cannot separate a symbol from what holds it — a method on a one-line class, a
function and its inner arrow sharing a line — the pack records what the parser reported and names
that limit instead of filling the gap with a guessed count. The options are explicit and mechanical:
which kinds to measure (``kinds``), whether nested functions join the ranking (``held``),
how many of the widest to show (``limit``), which files to measure
(``scope``, defaulting to everything the index inventoried) and whether to quote the measured source
(``quote_source``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .cli_trace import not_indexed_lines
from .directives import statistics
from .index.code_index import CodeIndex

SCHEMA_VERSION = "jev-navigator.statistics-pack/v1"

STATISTICS_OPERATIONS = ("count", "largest", "range")
STATISTICS_KINDS = ("function", "class")

# Naming a limit is not measuring around it. These sentences travel with the pack so that a reader
# knows what the numbers above them are, and stay text so that nothing is invented to fill a gap.
_LIMITS = (
    "Symbols come from the index's own parser, so a symbol nested inside another is found by line "
    "containment: a symbol on the same single line as its holder — a method on a one-line class, a "
    "function and its inner arrow sharing a line — has no holder to report, and a class is counted as "
    "a class only where the index lists a class there, which the parser reports apart from any "
    "method written on the same line. What is counted here is what the parser reported, never a "
    "guess at what the file holds.",
)

_HOLDING_LIMIT = (
    "With methods and functions nested inside another function kept out of the ranking, the ranking "
    "contains only top-level symbols. Counts still include every parsed symbol of the requested kinds."
)


def create_statistics_pack(
    repository: Path,
    prefixes: Sequence[str] = (),
    output: Path | None = None,
    operations: Sequence[str] = STATISTICS_OPERATIONS,
    *,
    kinds: Sequence[str] = STATISTICS_KINDS,
    held: bool = False,
    limit: int | None = None,
    min_lines: int | None = None,
    max_lines: int | None = None,
    scope: Sequence[str] | None = None,
    quote_source: bool = False,
    index: CodeIndex | None = None,
    fact_cache_dir: Path | None = None,
) -> dict:
    """Measure ``repository`` and write ``statistics.json`` + ``statistics.md`` into ``output``.

    ``prefixes`` narrows the file inventory the index walks; ``scope`` names the exact files to
    measure and, when given, replaces that inventoried default. ``operations`` chooses the sections
    written: ``count`` (how many functions and classes each file and the whole scope holds),
    ``largest`` (the widest symbols, ties kept) and ``range`` (every measured symbol with its
    inclusive min/max line range). ``kinds`` chooses what counts as a symbol, ``held`` decides
    whether methods and nested functions are measured beside the symbols holding them — with them
    kept out, the ranking lists only top-level symbols — and
    ``limit`` caps only how many of the widest are shown. ``quote_source`` quotes the measured source
    of each symbol listed. Returns the measured manifest.
    """
    # Check the options before creating anything, so a call refused for a bad option leaves nothing
    # behind: a half-written output directory is as misleading as an empty pack.
    _validate(operations, kinds, limit)
    if any(value is not None and value < 1 for value in (min_lines, max_lines)):
        raise ValueError("line sizes must be positive")
    if min_lines is not None and max_lines is not None and min_lines > max_lines:
        raise ValueError("min_lines must not exceed max_lines")
    output = _output_directory(output)
    repository = Path(repository).resolve()
    prefixes = tuple(prefixes)
    kinds = tuple(kinds)
    scope = None if scope is None else tuple(dict.fromkeys(scope))
    if index is None:
        index = CodeIndex.from_directory(
            repository, prefixes=prefixes, exclude_paths=(output,), fact_cache_dir=fact_cache_dir
        )
    counts = (
        None
        if "count" not in operations
        else statistics.count_symbols(index, scope, kinds=kinds, largest_of=limit, held=held)
    )
    # One measurement per pack, from the owner of counting and ranking: the counts already carry the
    # ranking they were measured with, and a pack that only ranks or lists asks for that once.
    ranking = (
        counts.largest
        if counts is not None
        else statistics.largest_functions(index, scope, kinds=kinds, limit=limit, held=held)
    )
    coverage = ranking.coverage
    pack = {
        "schema_version": SCHEMA_VERSION,
        "repository": str(repository),
        "revision": index.commit,
        "operations": list(operations),
        "options": {
            "kinds": list(kinds),
            "held": held,
            "limit": limit,
            "scope": None if scope is None else list(scope),
            "quote_source": quote_source,
            "min_lines": min_lines,
            "max_lines": max_lines,
        },
        "scope": {
            "prefixes": list(prefixes),
            "measured": list(coverage.measured),
            "skipped": list(coverage.skipped),
            "unmeasured": list(coverage.unmeasured),
            "unparsed": list(coverage.unparsed),
        },
        "coverage": {
            "complete": coverage.complete,
            "statement": coverage.statement,
            "unavailable": dict(index.unavailable_files),
            "not_indexed": index.not_indexed_files,
        },
        "limits": [*_LIMITS, *([_HOLDING_LIMIT] if not held else [])],
    }
    if counts is not None:
        pack["counts"] = _counts_section(counts)
    if "largest" in operations:
        pack["largest"] = _largest_section(index, ranking, limit, quote_source)
    if "range" in operations:
        pack["ranges"] = _ranges_section(index, ranking, quote_source, min_lines, max_lines)
    _write_json(output / "statistics.json", pack)
    (output / "statistics.md").write_text(_report(pack))
    return pack


def _output_directory(output: Path | None) -> Path:
    """Refuse a file, and a directory that already holds something, so packs are never mixed."""
    if output is None:
        raise ValueError("a statistics pack needs somewhere to write statistics.json and statistics.md")
    output = Path(output).expanduser()
    if output.is_file():
        raise ValueError(f"statistics output would have to be a directory and is a file: {output}")
    if output.is_dir() and any(output.iterdir()):
        raise ValueError(
            f"statistics output directory is not empty, so an old pack would be mixed "
            f"into the new one: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output.resolve()


def _validate(operations: Sequence[str], kinds: Sequence[str], limit: int | None) -> None:
    if not operations:
        raise ValueError("name at least one statistics operation: " + ", ".join(STATISTICS_OPERATIONS))
    unknown = [name for name in operations if name not in STATISTICS_OPERATIONS]
    if unknown:
        raise ValueError(
            f"statistics knows the operations {', '.join(STATISTICS_OPERATIONS)}; nothing called "
            f"{', '.join(unknown)}"
        )
    if not kinds:
        raise ValueError("a statistics pack measures symbol kinds, so name the kinds to measure")
    unknown_kinds = [kind for kind in kinds if kind not in STATISTICS_KINDS]
    if unknown_kinds:
        raise ValueError(
            f"a symbol is a function or a class here, so there is nothing to measure called "
            f"{', '.join(unknown_kinds)}"
        )
    if limit is not None and limit < 1:
        raise ValueError("limit says how many of the widest to show, so it needs a positive number")


def _symbol(index: CodeIndex, symbol: statistics.Symbol, *, quoted: bool) -> dict:
    """One measured symbol as evidence: where it sits, how wide it is, and what holds it."""
    span = symbol.span
    record = {
        "file": span.file,
        "name": span.name,
        "kind": symbol.kind,
        "lines": [span.start, span.end],
        "size": symbol.size,
        "held": symbol.held,
        "holder": (
            None
            if symbol.holder is None
            else f"{symbol.holder.file}:{symbol.holder.start}-{symbol.holder.end} {symbol.holder.name}"
        ),
    }
    if quoted:
        record["source"] = statistics.read_source(index, symbol).text
    return record


def _counts_section(counts: statistics.SymbolCount) -> dict:
    """What each measured file holds, with the ranking the same measurement drew from it."""
    return {
        "totals": counts.total,
        "symbol_lines": counts.symbol_lines,
        "caveat": counts.caveat,
        "per_file": {
            file: {
                "counts": measured.counts,
                "symbol_lines": measured.symbol_lines,
                "parsed": measured.parsed,
                "held": [str(symbol) for symbol in measured.held],
            }
            for file, measured in counts.per_file.items()
        },
    }


def _largest_section(index: CodeIndex, ranking: statistics.Largest, limit: int | None, quoted: bool) -> dict:
    """The widest symbols, every tie kept, beside the count they were drawn from."""
    return {
        "limit": limit,
        "measured": len(ranking.measured),
        "shown": len(ranking.largest),
        "truncated": ranking.truncated,
        "complete": ranking.complete,
        "caveat": ranking.caveat,
        "biggest": [_symbol(index, symbol, quoted=quoted) for symbol in ranking.biggest],
        "ranking": [_symbol(index, symbol, quoted=quoted) for symbol in ranking.largest],
    }


def _ranges_section(
    index: CodeIndex,
    ranking: statistics.Largest,
    quoted: bool,
    min_lines: int | None,
    max_lines: int | None,
) -> dict:
    """Every symbol this pack measured, with the lines it fills: a listing, never a top-N cut."""
    per_file: dict[str, list] = {}
    for symbol in ranking.measured:
        if min_lines is not None and symbol.size < min_lines:
            continue
        if max_lines is not None and symbol.size > max_lines:
            continue
        per_file.setdefault(symbol.span.file, []).append(_symbol(index, symbol, quoted=quoted))
    return {
        "measured": len(ranking.measured),
        "listed": sum(len(symbols) for symbols in per_file.values()),
        "per_file": per_file,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _where(symbol: dict) -> str:
    return f"{symbol['file']}:{symbol['lines'][0]}-{symbol['lines'][1]}"


def _header(pack: dict) -> list[str]:
    scope, options = pack["scope"], pack["options"]
    scope_shown = ", ".join(f"`{prefix}`" for prefix in scope["prefixes"]) or "whole directory"
    if scope["measured"]:
        scope_shown += ", measured: " + ", ".join(f"`{path}`" for path in scope["measured"])
    holders = (
        "beside the symbols holding them, so a class and a method inside it are measured apart"
        if options["held"]
        else "out of the ranking, leaving only top-level symbols"
    )
    ranking = "all measured symbols" if options["limit"] is None else f"only the {options['limit']} widest"
    return [
        "# Structural statistics",
        "",
        f"- Schema: `{pack['schema_version']}`",
        f"- Repository: `{pack['repository']}` at revision `{pack['revision'] or 'no git revision'}`",
        f"- Scope: {scope_shown}",
        f"- Operations: {', '.join(pack['operations'])}",
        f"- Kinds measured: {', '.join(options['kinds'])}",
        f"- Methods and functions nested in another function are kept: {holders}",
        f"- Widest symbols shown here: {ranking} (a limit shortens the ranking shown, never the measurement)",
        f"- Measured source quoted in this pack: {'yes' if options['quote_source'] else 'no'}",
        "",
    ]


def _counts_report(pack: dict) -> list[str]:
    """One row per measured file, and the scope's totals under it."""
    counts = pack["counts"]
    kinds = pack["options"]["kinds"]
    lines = [
        "## Counts",
        "",
        "| File | " + " | ".join(kinds) + " | Symbol lines | Partly parsed |",
        "| --- |" + " ---: |" * (len(kinds) + 1) + " --- |",
    ]
    for file, measured in counts["per_file"].items():
        parsed = "no" if measured["parsed"] else "yes, so the count is a floor"
        cells = [f"`{file}`"] + [str(measured["counts"].get(kind, 0)) for kind in kinds]
        cells += [str(measured["symbol_lines"]), parsed]
        lines.append("| " + " | ".join(cells) + " |")
    totals = ", ".join(f"{kind}: {counts['totals'].get(kind, 0)}" for kind in kinds)
    lines += [
        "",
        f"The scope held {totals} over {counts['symbol_lines']} symbol lines. A nested symbol's lines "
        "are counted once, so symbol lines show coverage and are never a sum of symbol sizes.",
        "",
        f"Coverage: {counts['caveat'] or 'every file in scope was scanned and parsed in full.'}",
        "",
    ]
    return lines


def _largest_report(pack: dict) -> list[str]:
    """The ranking the pack shows, said in the same numbers the JSON carries."""
    largest = pack["largest"]
    shown = "all of them" if largest["limit"] is None else f"only {largest['shown']}"
    lines = [
        "## Largest symbols",
        "",
        f"Measured {largest['measured']} symbols and shows {shown}"
        + (", so what is left out could hold a narrower symbol of its own" if largest["truncated"] else "")
        + ".",
        "",
        "| Lines | Size | Symbol | Kind | Held by |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for symbol in largest["ranking"]:
        holder = "-" if symbol["holder"] is None else f"`{symbol['holder']}`"
        lines.append(
            f"| `{_where(symbol)}` | {symbol['size']} | `{symbol['name']}` | {symbol['kind']} | {holder} |"
        )
    tied = ", ".join(
        f"`{_where(symbol)}` {symbol['name']} ({symbol['size']} lines)" for symbol in largest["biggest"]
    )
    lines += ["", f"Widest, and every one of them kept: {tied}.", ""]
    return lines


def _ranges_report(pack: dict) -> list[str]:
    """The full listing, so nothing measured is hidden by a ranking that stopped short."""
    ranges = pack["ranges"]
    lines = ["## Line ranges", "", "| File | Symbol | Kind | Lines |", "| --- | --- | --- | --- |"]
    for file, symbols in ranges["per_file"].items():
        for symbol in symbols:
            lines.append(
                f"| `{file}` | `{symbol['name']}` | {symbol['kind']} | "
                f"{symbol['lines'][0]}-{symbol['lines'][1]} |"
            )
    lines += [
        "",
        f"Listed {ranges['listed']} of {ranges['measured']} measured symbols; "
        f"inclusive size filter: {pack['options']['min_lines']} to {pack['options']['max_lines']} lines "
        "(null means no bound). Counts and measurement coverage are unchanged.",
        "",
    ]
    return lines


def _coverage_report(pack: dict) -> list[str]:
    """What the measurement could not see, and the two limits that shaped what it could."""
    scope = pack["scope"]
    gaps = []
    if scope["unmeasured"]:
        gaps.append(f"{len(scope['unmeasured'])} file(s) were never scanned")
    if scope["unparsed"]:
        gaps.append(f"{len(scope['unparsed'])} file(s) parsed only partially")
    lines = ["## Coverage and limits", ""]
    if gaps or not scope["measured"]:
        lines.append(f"Not fully covered: {'; '.join(gaps) or 'nothing in scope was measured'}.")
    else:
        lines.append("Fully covered within the measured scope: all measured files parsed successfully.")
    if scope["skipped"]:
        skipped = ", ".join(f"`{path}`" for path in scope["skipped"])
        lines.append(f"Skipped outside parser coverage: {skipped}.")
    if scope["unmeasured"] or scope["unparsed"]:
        reasons = pack["coverage"]["unavailable"]
        never = ", ".join(_named_with_reason(path, reasons) for path in scope["unmeasured"])
        partly = ", ".join(f"`{path}`" for path in scope["unparsed"])
        named = "; ".join(
            label
            for label in (
                f"never scanned: {never}" if never else "",
                f"partly parsed: {partly}" if partly else "",
            )
            if label
        )
        lines += [
            "",
            f"Named here, and named in `statistics.json` under `scope.unmeasured` and "
            f"`scope.unparsed` ({named}): what those files hold is unknown, not absent, so every "
            "count taken above them is a floor.",
        ]
    not_indexed = pack["coverage"]["not_indexed"]
    if not_indexed:
        listed_in = "`coverage.not_indexed` in `statistics.json`"
        lines += [
            "",
            "Not indexed, so outside every count above:",
            "",
            *not_indexed_lines(not_indexed, listed_in),
        ]
    lines += [""] + [f"{number}. {limit}" for number, limit in enumerate(pack["limits"], start=1)]
    return lines


def _named_with_reason(path: str, reasons: dict[str, str]) -> str:
    return f"`{path}` ({reasons[path]})" if path in reasons else f"`{path}`"


def _report(pack: dict) -> str:
    """Render the manifest as Markdown, quoting the same numbers the JSON carries."""
    lines = _header(pack)
    if "count" in pack["operations"]:
        lines += _counts_report(pack)
    if "largest" in pack["operations"]:
        lines += _largest_report(pack)
    if "range" in pack["operations"]:
        lines += _ranges_report(pack)
    lines += _coverage_report(pack)
    return "\n".join(lines) + "\n"
