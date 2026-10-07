"""Execute independent typed searches, without a model or a second frontier.

Plans are caller data. Results retain every approach's provenance and can feed any
existing search through ``PlanSource``. Rank orders the union, never thread completion.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Annotated, Literal

import msgspec

from . import sources
from .index.bindings import names_exactly
from .index.code_index import CodeIndex
from .index.languages import language_read
from .index.scope import _matches, is_test_file
from .index.units import LineAnchor, Reading, Unit, UnitReader
from .sources import Reach, Seeds

Nonempty = Annotated[str, msgspec.Meta(min_length=1)]


class FindText(
    msgspec.Struct, tag="find_text", tag_field="operation", frozen=True, forbid_unknown_fields=True
):
    pattern: Nonempty
    scopes: tuple[str, ...] = ()
    pattern_kind: Literal["literal", "regex"] = "literal"
    ignore_case: bool = False


class MatchingFiles(
    msgspec.Struct, tag="matching_files", tag_field="operation", frozen=True, forbid_unknown_fields=True
):
    globs: tuple[str, ...]
    scopes: tuple[str, ...] = ()


class FileUnits(
    msgspec.Struct, tag="file_units", tag_field="operation", frozen=True, forbid_unknown_fields=True
):
    path: Nonempty


class Definitions(
    msgspec.Struct, tag="definitions", tag_field="operation", frozen=True, forbid_unknown_fields=True
):
    name: Nonempty
    owner: Nonempty


class Callers(msgspec.Struct, tag="callers", tag_field="operation", frozen=True, forbid_unknown_fields=True):
    name: Nonempty
    owner: Nonempty


class Callees(msgspec.Struct, tag="callees", tag_field="operation", frozen=True, forbid_unknown_fields=True):
    path: Nonempty
    line: Annotated[int, msgspec.Meta(ge=1)]


class References(
    msgspec.Struct, tag="references", tag_field="operation", frozen=True, forbid_unknown_fields=True
):
    name: Nonempty
    scopes: tuple[str, ...] = ()
    kind: Literal["identifier", "literal"] = "identifier"


class NamedImports(
    msgspec.Struct, tag="named_imports", tag_field="operation", frozen=True, forbid_unknown_fields=True
):
    path: Nonempty


class ConfigKey(
    msgspec.Struct, tag="config_key", tag_field="operation", frozen=True, forbid_unknown_fields=True
):
    key: Nonempty
    scopes: tuple[str, ...] = ()


class TestsOf(msgspec.Struct, tag="tests_of", tag_field="operation", frozen=True, forbid_unknown_fields=True):
    path: Nonempty
    name: Nonempty


Operation = (
    FindText
    | MatchingFiles
    | FileUnits
    | Definitions
    | Callers
    | Callees
    | References
    | NamedImports
    | ConfigKey
    | TestsOf
)


class TermProvenance(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    term: Nonempty
    source: Nonempty
    transformation: str = "copied"


class Approach(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    rank: Annotated[int, msgspec.Meta(ge=1, le=10)]
    call: Operation
    provenance: tuple[TermProvenance, ...]
    reason: str = ""
    gap: str = ""


class SearchPlan(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    approaches: Annotated[tuple[Approach, ...], msgspec.Meta(max_length=10)]


def decode_plan(data: str | bytes) -> SearchPlan:
    plan = msgspec.json.decode(data, type=SearchPlan)
    _validate(plan)
    return plan


def plan_schema() -> dict:
    return msgspec.json.schema(SearchPlan)


@dataclass(frozen=True)
class ApproachOutcome:
    approach: Approach
    reaches: tuple[Reach, ...]
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanCandidate:
    unit: Unit
    approaches: tuple[Approach, ...]
    reaches: tuple[Reach, ...]


@dataclass(frozen=True)
class PlanResult:
    candidates: tuple[PlanCandidate, ...]
    outcomes: tuple[ApproachOutcome, ...]


def execute_plan(index: CodeIndex, plan: SearchPlan, *, box_chars: int, workers: int = 10) -> PlanResult:
    """Start independent lookups together; merge canonical units in approach rank order.

    Invalid targets, empty results and source exclusions remain in outcomes. No failed
    argument is substituted. Parser/provider failures propagate to the owning host.
    """
    _validate(plan)
    if box_chars <= 0 or workers <= 0:
        raise ValueError("box_chars and workers must be positive")
    ordered = sorted(plan.approaches, key=lambda approach: approach.rank)
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(ordered)))) as pool:
        futures = {}
        for approach in ordered:
            key = msgspec.json.encode(approach.call)
            if key not in futures:
                futures[key] = pool.submit(_gather, index, approach, box_chars)
        outcomes = []
        for approach in ordered:
            outcome = futures[msgspec.json.encode(approach.call)].result()
            outcomes.append(ApproachOutcome(approach, outcome.reaches, outcome.problems))
    reader = UnitReader(index, box_chars, listed_only=False, reading=Reading.MIXED)
    units: dict[str, Unit] = {}
    provenance: dict[str, list[Approach]] = {}
    routes: dict[str, list[Reach]] = {}
    resolved = []
    for outcome in outcomes:
        problems = list(outcome.problems)
        for reach in outcome.reaches:
            if isinstance(reach.at, str):
                # Parse and release one file at a time, never a repository's parser output.
                listing = reader.list_files((reach.at,))
                found = listing.units
                problems.extend(f"{path}: {reason}" for path, reason in listing.unlisted.items())
            else:
                found, problem = reader.resolve(reach.at)
                if problem:
                    problems.append(problem)
            for unit in found:
                units.setdefault(unit.id, unit)
                if outcome.approach not in provenance.setdefault(unit.id, []):
                    provenance[unit.id].append(outcome.approach)
                if reach not in routes.setdefault(unit.id, []):
                    routes[unit.id].append(reach)
        resolved.append(ApproachOutcome(outcome.approach, outcome.reaches, tuple(dict.fromkeys(problems))))
    return PlanResult(
        tuple(PlanCandidate(unit, tuple(provenance[key]), tuple(routes[key])) for key, unit in units.items()),
        tuple(resolved),
    )


@dataclass(frozen=True)
class PlanSource:
    """An executed plan's places join ``sources=`` of an existing mini-workflow.

    ``result`` is kept by the caller for invalid-argument and exclusion accounting.
    The search owns unit resolution, request grouping, judging and stopping.
    """

    result: PlanResult
    name: str = "plan"
    label: str = "planned approaches"

    def reach(self, index: CodeIndex, seeds: Seeds) -> Iterable[Reach]:
        for outcome in self.result.outcomes:
            for reach in outcome.reaches:
                yield Reach(
                    reach.at,
                    f"plan:{outcome.approach.rank}:{reach.source}",
                    reach.seed,
                    reach.distance,
                    reach.names,
                )


def _validate(plan: SearchPlan) -> None:
    # Struct constructors are also public: apply the same schema as JSON callers.
    msgspec.convert(msgspec.to_builtins(plan), type=SearchPlan)
    ranks = [approach.rank for approach in plan.approaches]
    if len(ranks) != len(set(ranks)):
        raise ValueError("approach ranks must be unique")


def _path(index: CodeIndex, path: str) -> str:
    if path not in index.files:
        raise ValueError(f"{path!r} is not a file in the index")
    return path


def _scoped(index: CodeIndex, scopes: tuple[str, ...]) -> tuple[str, ...]:
    for scope in scopes:
        if scope not in ("", ".", "./") and not any(_matches(scope, file) for file in index.files):
            raise ValueError(f"scope {scope!r} matches no file in the index")
    return tuple(
        file
        for file in index.files
        if not scopes or any(scope in ("", ".", "./") or _matches(scope, file) for scope in scopes)
    )


def _gather(index: CodeIndex, approach: Approach, box_chars: int) -> ApproachOutcome:
    try:
        reaches = tuple(_places(index, approach.call, box_chars))
    except (ValueError, re.error) as error:
        return ApproachOutcome(approach, (), (str(error),))
    problems = [] if reaches else ["no matching places"]
    if isinstance(approach.call, FindText | ConfigKey):
        files = _scoped(index, approach.call.scopes)
        excluded = index.text_files_left_out(tuple(file for file in files if not language_read(file)))
        problems.extend(f"{file}: {reason}" for file, reason in excluded.items())
    return ApproachOutcome(approach, reaches, tuple(problems))


def _text(index: CodeIndex, call: FindText) -> Iterable[Reach]:
    files = _scoped(index, call.scopes)
    if call.pattern_kind == "literal" and not call.ignore_case:
        allowed = frozenset(files)
        for hit in index.search_text(call.pattern):
            if hit.file in allowed:
                yield Reach(LineAnchor(hit.file, hit.line), "find_text", call.pattern, 1)
        return
    expression = call.pattern if call.pattern_kind == "regex" else re.escape(call.pattern)
    pattern = re.compile(expression, re.IGNORECASE if call.ignore_case else 0)
    # CodeIndex has an exact-text primitive. Regex adds line matching over that same
    # admissible source reader; source exclusions still apply during unit resolution.
    excluded = index.text_files_left_out(tuple(file for file in files if not language_read(file)))
    for file in files:
        if file in excluded:
            continue
        for line, text in enumerate(index.lines(file), 1):
            if pattern.search(text):
                yield Reach(LineAnchor(file, line), "find_text", call.pattern, 1)


def _places(index: CodeIndex, call: Operation, box_chars: int) -> Iterable[Reach]:
    if isinstance(call, FindText):
        yield from _text(index, call)
    elif isinstance(call, MatchingFiles):
        positive = tuple(glob for glob in call.globs if not glob.startswith("!"))
        negative = tuple(glob[1:] for glob in call.globs if glob.startswith("!"))
        for file in _scoped(index, call.scopes):
            if (not positive or any(_matches(glob, file) for glob in positive)) and not any(
                _matches(glob, file) for glob in negative
            ):
                yield Reach(file, "matching_files", ", ".join(call.globs), 1)
    elif isinstance(call, FileUnits):
        yield Reach(_path(index, call.path), "file_units", call.path, 1)
    elif isinstance(call, Definitions | Callers):
        owner = _path(index, call.owner)
        definitions = tuple(span for span in index.find_definition(call.name) if span.file == owner)
        if not definitions:
            raise ValueError(f"{call.name!r} has no definition in {owner!r}")
        if isinstance(call, Definitions):
            for span in definitions:
                yield Reach(LineAnchor(span.file, span.start), "definition", call.name, 1)
        else:
            reader = UnitReader(index, box_chars, listed_only=False, reading=Reading.MIXED)
            units = tuple(
                unit for span in definitions for unit in reader.resolve(LineAnchor(owner, span.start))[0]
            )
            yield from sources.CALLERS.reach(index, Seeds(units=units))
            for reference in index.find_references(call.name):
                if any(names_exactly(reference.binding, span) for span in definitions):
                    yield Reach(LineAnchor(reference.file, reference.line), "reference", call.name, 1)
    elif isinstance(call, Callees):
        reader = UnitReader(index, box_chars, listed_only=False, reading=Reading.MIXED)
        units, problem = reader.resolve(LineAnchor(_path(index, call.path), call.line))
        if problem:
            raise ValueError(problem)
        seeds = Seeds(units=units)
        yield from sources.CALLEES.reach(index, seeds)
        yield from sources.CLIENT_CALLS.reach(index, seeds)
    elif isinstance(call, References):
        allowed = frozenset(_scoped(index, call.scopes))
        source = sources.REFERENCES if call.kind == "identifier" else sources.LITERALS
        for reach in source.reach(index, Seeds(names=(call.name,), literals=(call.name,))):
            path = reach.at if isinstance(reach.at, str) else reach.at.file
            if path in allowed:
                yield reach
    elif isinstance(call, NamedImports):
        seeds = Seeds(files=(_path(index, call.path),))
        for source in (sources.IMPORTS, sources.NAMED_FILES, sources.TEXT_NAMED_FILES):
            yield from source.reach(index, seeds)
    elif isinstance(call, ConfigKey):
        yield from _text(index, FindText(call.key, call.scopes))
    elif isinstance(call, TestsOf):
        path = PurePosixPath(_path(index, call.path))
        if not any(span.file == call.path for span in index.find_definition(call.name)):
            raise ValueError(f"{call.name!r} has no definition in {call.path!r}")
        stems = {f"test_{path.stem}", f"{path.stem}_test", f"{path.stem}.test", f"{path.stem}.spec"}
        found = set()
        for file in index.files:
            if is_test_file(file) and PurePosixPath(file).stem in stems:
                found.add(file)
                yield Reach(file, "tests_of", call.path, 1)
        for reach in _text(index, FindText(call.name)):
            if is_test_file(reach.at.file) and reach.at.file not in found:
                yield reach
