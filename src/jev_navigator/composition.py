"""Small search compositions and reserved call shares. No caller domain lives here."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .directives.find_all import FindAllResult, find_all_async, find_all_text_async
from .directives.frontier import STAGE_ORDER, WHOLE_FRONTIER, Policy
from .index.code_index import CodeIndex
from .index.units import Anchor, RangeAnchor, Reading
from .judgments.judge import CheckResult, Judge
from .mentions import names_from_text
from .sources import (
    ANCHORS,
    CALLEES,
    CALLERS,
    CLIENT_CALLS,
    DEFINITIONS,
    FILE_WORDS,
    FILES,
    IMPORTS,
    LITERALS,
    MODELS,
    NAMED_FILES,
    NAMES,
    REFERENCES,
    SPELLINGS,
    TEXT_FILE_NAMES,
    TEXT_NAMED_FILES,
    TEXT_NAMES,
    Source,
)


def reserve_calls(judge: Judge, allowances: Mapping[str, int]) -> dict[str, Judge]:
    """Reserve independent stage caps before any stage starts. Parent accounting remains shared.

    Unused calls stay reserved for their stage. A stage can subdivide its share for a later
    composition. An existing parent cap must have room for every reservation.
    """
    return judge.reserve_calls(allowances)


@dataclass(frozen=True)
class SearchConfiguration:
    """Code and text searches with independent call shares. Compose continuation using reserve_calls.

    Text includes named files and name hits by default, without scanning every text file.
    Both searches use the existing unit builder, masker, frontier and judging owner.
    """

    name: str
    code_calls: int
    text_calls: int
    policy: Policy = STAGE_ORDER
    code_sources: tuple[Source, ...] = (ANCHORS, FILES, NAMES)
    text_sources: tuple[Source, ...] = (ANCHORS, FILES, TEXT_FILE_NAMES, TEXT_NAMED_FILES, TEXT_NAMES)

    async def search(
        self,
        index: CodeIndex,
        judge: Judge,
        targets: Mapping[str, str],
        *,
        files: Sequence[str] = (),
        anchors: Sequence[Anchor] = (),
        names: Sequence[str] = (),
        delivered: Sequence[RangeAnchor] = (),
    ) -> tuple[FindAllResult, FindAllResult]:
        stages = reserve_calls(judge, {"code": self.code_calls, "text": self.text_calls})
        extracted = [names_from_text(text) for text in targets.values()]
        names = tuple(dict.fromkeys([*names, *(name for mentions in extracted for name in mentions.code)]))
        options = dict(files=files, anchors=anchors, names=names, delivered=delivered, policy=self.policy)
        code, text = await asyncio.gather(
            find_all_async(index, stages["code"], targets, sources=self.code_sources, **options),
            find_all_text_async(index, stages["text"], targets, sources=self.text_sources, **options),
        )
        return code, text


FRONTIER_SOURCES = (
    FILE_WORDS,
    NAMED_FILES,
    TEXT_NAMED_FILES,
    ANCHORS,
    FILES,
    NAMES,
    SPELLINGS,
    LITERALS,
    DEFINITIONS,
    REFERENCES,
    IMPORTS,
)
FRONTIER_HOPS = (
    CALLEES,
    CALLERS,
    MODELS,
    CLIENT_CALLS,
    NAMED_FILES,
    TEXT_NAMED_FILES,
    IMPORTS,
    DEFINITIONS,
    REFERENCES,
    LITERALS,
)
DEFAULT_FRONTIER_CALLS = 4096


@dataclass(frozen=True)
class FrontierConfiguration:
    """One Jev budget for the entire reached code and text frontier, in real search order.

    Code discovery has no call or unit limit. A unit is expanded once after it has been judged,
    regardless of its relevance answer. Calls are packed at 16 units and stop only at exhaustion,
    cancellation, a failure or this caller-selected budget. No target settles early.
    """

    max_calls: int = DEFAULT_FRONTIER_CALLS
    sources: tuple[Source, ...] = FRONTIER_SOURCES
    hops: tuple[Source, ...] = FRONTIER_HOPS

    async def search(
        self,
        index: CodeIndex,
        judge: Judge,
        targets: Mapping[str, str],
        *,
        files: Sequence[str] = (),
        anchors: Sequence[Anchor] = (),
        names: Sequence[str] = (),
        delivered: Sequence[RangeAnchor] = (),
        completed: Mapping[str, Sequence[CheckResult]] | None = None,
    ) -> FindAllResult:
        if self.max_calls < 0:
            raise ValueError("max_calls must be non-negative")
        scoped = judge.scope()
        scoped.max_calls = self.max_calls
        scoped.items_per_request = 16
        extracted = (name for text in targets.values() for name in names_from_text(text).code)
        names = tuple(dict.fromkeys([*names, *extracted]))
        return await find_all_async(
            index,
            scoped,
            targets,
            files=files,
            anchors=anchors,
            names=names,
            delivered=delivered,
            completed=completed,
            sources=self.sources,
            hops=self.hops,
            reading=Reading.MIXED,
            policy=WHOLE_FRONTIER,
        )
