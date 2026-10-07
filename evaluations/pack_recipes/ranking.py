"""Recipe adapter for the shared case-1 selection blocks, with frozen default weights."""

from dataclasses import dataclass

from jev_navigator.index.units import read_ranges
from jev_navigator.selection import ScentIndex, graph_from_index, rank, rank_features, scent_document
from jev_navigator.selection.rank import RankWeights


@dataclass(frozen=True)
class RecipeRanker:
    index: object
    query: str
    seeds: tuple
    weights: RankWeights = RankWeights()

    def __call__(self, units):
        documents = ScentIndex(
            scent_document(
                unit.id,
                unit.path,
                unit.symbol,
                read_ranges(self.index, unit.path, unit.ranges),
                test=unit.test,
            )
            for unit in units
        )
        # The declared recipe uses current code relations, not co-change history.
        graph = graph_from_index(self.index, units, cochange_limit=0)
        seed_weights = {
            unit.id: 1.0
            for unit in units
            if any(
                unit.path == seed.path
                and any(a <= d and c <= b for a, b in unit.ranges for c, d in seed.ranges)
                for seed in self.seeds
            )
        }
        features = rank_features(documents, self.query, graph, seed_weights)
        by_id = {unit.id: unit for unit in units}
        return tuple(by_id[key] for key in rank(list(by_id), features, self.weights))
