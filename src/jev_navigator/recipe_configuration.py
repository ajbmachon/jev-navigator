"""Named and versioned source configurations with caller-owned ranking and confirmation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from .index.code_index import CodeIndex
from .index.units import Reading, Unit, UnitReader
from .sources import Reach, Seeds, Source


@dataclass(frozen=True)
class RecipeCandidates:
    units: tuple[Unit, ...]
    reached_by: dict[str, tuple[Reach, ...]]
    unresolved: tuple[tuple[Reach, str], ...]


@dataclass(frozen=True)
class SearchRecipe:
    """Configuration is data. Follow-up hops accept only caller-confirmed units.

    Ranking is injected by the caller, allowing selection blocks to be shared.
    Without it, stable source order is an explicitly simple placeholder.
    """

    name: str
    version: int
    sources: tuple[Source, ...]
    hops: tuple[Source, ...] = ()

    def gather(
        self,
        index: CodeIndex,
        seeds: Seeds,
        *,
        box_chars: int,
        rank: Callable[[Sequence[Unit]], Iterable[Unit]] | None = None,
    ) -> RecipeCandidates:
        return self._gather(index, seeds, self.sources, box_chars, rank)

    def continue_from(
        self,
        index: CodeIndex,
        confirmed: Sequence[Unit],
        *,
        box_chars: int,
        rank: Callable[[Sequence[Unit]], Iterable[Unit]] | None = None,
    ) -> RecipeCandidates:
        # No names/files/anchors from the original query leak into this hop.
        return self._gather(index, Seeds(units=tuple(confirmed)), self.hops, box_chars, rank)

    def _gather(self, index, seeds, sources, box_chars, rank) -> RecipeCandidates:
        units: dict[str, Unit] = {}
        provenance: dict[str, list[Reach]] = {}
        unresolved = []
        resolved = {}
        for source in sources:
            for reach in source.reach(index, seeds):
                # A reader lives for one place only, keeping parser/source memory
                # bounded instead of accumulating a repository's materialized units.
                if reach.at not in resolved:
                    reader = UnitReader(index, box_chars, listed_only=False, reading=Reading.MIXED)
                    if isinstance(reach.at, str):
                        listing = reader.list_files([reach.at])
                        found = listing.units
                        problem = listing.unlisted.get(reach.at, "")
                    else:
                        found, problem = reader.resolve(reach.at)
                    resolved[reach.at] = (found, problem)
                found, problem = resolved[reach.at]
                if problem:
                    unresolved.append((reach, problem))
                for unit in found:
                    units.setdefault(unit.id, unit)
                    provenance.setdefault(unit.id, []).append(reach)
        ordered = tuple(units.values())
        result = RecipeCandidates(
            ordered, {key: tuple(value) for key, value in provenance.items()}, tuple(unresolved)
        )
        if rank is None:
            return result
        ranked = tuple(rank(ordered))
        if len(ranked) != len(ordered) or {unit.id: unit for unit in ranked} != units:
            raise ValueError("a recipe ranker must reorder every candidate exactly once")
        return replace(result, units=ranked)
