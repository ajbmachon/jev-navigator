"""Sparse code graph and random walk with restart. File relations use virtual file nodes."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..index.code_index import CodeIndex
from ..index.languages import language_of
from ..index.spans import Span
from ..index.units import Unit
from ..mentions import paths_in


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: str
    weight: float = 1.0


class CodeGraph:
    def __init__(self, nodes: Iterable[str], edges: Iterable[GraphEdge] = ()):
        self.adjacency: dict[str, dict[str, float]] = {node: {} for node in nodes}
        self.edge_counts: dict[str, int] = defaultdict(int)
        self.unresolved: dict[str, int] = defaultdict(int)
        for edge in edges:
            self.add(edge)

    def add(self, edge: GraphEdge) -> None:
        if not math.isfinite(edge.weight) or edge.weight <= 0:
            raise ValueError("Graph edge weight must be positive and finite")
        if edge.source == edge.target:
            return
        for source, target in ((edge.source, edge.target), (edge.target, edge.source)):
            neighbours = self.adjacency.setdefault(source, {})
            neighbours[target] = neighbours.get(target, 0) + edge.weight
        self.edge_counts[edge.kind] += 1

    def degrees(self) -> dict[str, float]:
        return {node: sum(neighbours.values()) for node, neighbours in self.adjacency.items()}


def random_walk(
    graph: CodeGraph,
    seeds: Mapping[str, float],
    *,
    restart: float = 0.25,
    hub_penalty: float = 0.5,
    tolerance: float = 1e-7,
    max_iterations: int = 80,
) -> dict[str, float]:
    """Personalized probability mass, divided by degree**hub_penalty to suppress hubs.

    Dangling mass restarts at the same seeds. Iteration count bounds numerical work, never search
    time. Returned penalized scores need not sum to one; unpenalized scores do.
    """
    if not 0 < restart <= 1 or hub_penalty < 0 or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Invalid random-walk parameters")
    if any(not math.isfinite(weight) or weight < 0 for weight in seeds.values()):
        raise ValueError("Seed weights must be nonnegative and finite")
    mass = {node: weight for node, weight in seeds.items() if node in graph.adjacency and weight > 0}
    total = sum(mass.values())
    if not total:
        return {node: 0.0 for node in graph.adjacency}
    prior = {node: weight / total for node, weight in mass.items()}
    probabilities = dict(prior)
    degrees = graph.degrees()
    for _ in range(max_iterations):
        dangling = sum(value for node, value in probabilities.items() if not degrees[node])
        updated = {node: (restart + (1 - restart) * dangling) * value for node, value in prior.items()}
        for node, value in probabilities.items():
            if not degrees[node]:
                continue
            scale = (1 - restart) * value / degrees[node]
            for neighbour, weight in graph.adjacency[node].items():
                updated[neighbour] = updated.get(neighbour, 0) + scale * weight
        error = sum(abs(updated.get(node, 0) - probabilities.get(node, 0)) for node in graph.adjacency)
        probabilities = updated
        if error < tolerance:
            break
    return {
        node: probabilities.get(node, 0) / max(1.0, degrees[node]) ** hub_penalty for node in graph.adjacency
    }


def graph_from_index(index: CodeIndex, units: Sequence[Unit], *, cochange_limit: int = 5) -> CodeGraph:
    """Resolved calls/references, imports, named files, same-file and git co-change relations.

    Only admitted unit locations are nodes. Unknown bindings are counted, never joined by name.
    One file's parser facts are requested at a time; no repository parser output is materialized.
    File nodes replace complete bipartite expansions of file relations.
    """
    graph = CodeGraph(unit.id for unit in units)
    by_file: dict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        by_file[unit.path].append(unit)
        graph.add(GraphEdge(unit.id, _file_node(unit.path), "same_file"))
    filenames: dict[str, list[str]] = defaultdict(list)
    for path in by_file:
        filenames[PurePosixPath(path).name].append(path)
    cochanged = set()
    for path, file_units in by_file.items():
        for imported in index.imports(path):
            if imported in by_file:
                graph.add(GraphEdge(_file_node(path), _file_node(imported), "import"))
        if cochange_limit:
            for changed, count in index.co_changed_files(path, cochange_limit):
                pair = tuple(sorted((path, changed)))
                if changed in by_file and pair not in cochanged:
                    cochanged.add(pair)
                    graph.add(GraphEdge(_file_node(path), _file_node(changed), "cochange", math.log1p(count)))
        if language_of(path):
            _bound_edges(index, graph, path, by_file)
        for unit in file_units:
            source = "\n".join(index.read_slice(Span(path, start, end)).text for start, end in unit.ranges)
            for named in dict.fromkeys(paths_in(source)):
                candidates = [named] if named in by_file else filenames.get(PurePosixPath(named).name, [])
                if len(candidates) == 1 and candidates[0] != path:
                    graph.add(GraphEdge(unit.id, _file_node(candidates[0]), "named_file"))
    return graph


def _bound_edges(index: CodeIndex, graph: CodeGraph, path: str, by_file: Mapping[str, list[Unit]]) -> None:
    facts = index.facts_in_files([path]).get(path)
    if facts is None:
        return
    for kind, uses in (("call", facts.calls), ("reference", facts.references)):
        for use in uses:
            source = _holding(by_file[path], use.line)
            if source is None:
                continue
            role = getattr(use, "role", None)
            binding = index.binding_of(path, use.line, use.name, use.receiver, role)
            if not binding.proven or binding.target is None:
                graph.unresolved[kind] += 1
                continue
            target = _holding(by_file.get(binding.target.file, []), binding.target.start)
            if target is not None:
                graph.add(GraphEdge(source.id, target.id, kind))


def _holding(units: Sequence[Unit], line: int) -> Unit | None:
    holding = (unit for unit in units if any(start <= line <= end for start, end in unit.ranges))
    return min(holding, key=lambda unit: sum(end - start + 1 for start, end in unit.ranges), default=None)


def _file_node(path: str) -> str:
    return "file:" + path
