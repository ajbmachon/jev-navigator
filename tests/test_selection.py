"""Selection's real contracts: visible summaries, bound graph links, ordering and oracle admission."""

from dataclasses import replace

import pytest

from jev_navigator.index.units import Reading, list_units
from jev_navigator.judgments.answers import ChoiceAnswer
from jev_navigator.selection import (
    ActivePolicy,
    CodeGraph,
    GraphEdge,
    Observation,
    OutlineLimits,
    RankFeatures,
    RankWeights,
    ScentIndex,
    active_search,
    graph_from_index,
    outline,
    outline_choice,
    random_walk,
    rank,
    scent_document,
    selected_file,
    words,
)


def test_outline_reads_real_definitions_imports_and_matching_lines(sample_index):
    file = outline(sample_index, "app/validation.py", ["orders.max_items"])
    assert ("validate_order", 4, 7) in file.definitions
    assert any("orders.max_items" in text for _, text in file.matches)
    imported = outline(sample_index, "app/orders.py", ["validateOrder"])
    assert any("app.validation" in text for _, text in imported.imports)
    clipped = outline(
        sample_index,
        "app/validation.py",
        ["order"],
        limits=OutlineLimits(definitions=1, matches=1, line_chars=8),
    )
    assert clipped.omitted["definitions"] >= 1
    assert clipped.omitted["matches"] >= 1
    assert clipped.clipped_lines
    request = outline_choice("Which code limits order size?", [file, imported])
    assert request["state"]["files"]["1"]["path"] == "app/validation.py"
    assert set(request["questions"]["read_file"]["criteria"]) == {"1", "2", "none"}
    answer = ChoiceAnswer.from_probabilities({"1": 0.9, "2": 0.05, "none": 0.05})
    assert selected_file(answer, [file, imported], min_confidence=0.8) == "app/validation.py"
    assert selected_file(answer, [file, imported], min_confidence=0.99) is None
    assert selected_file(replace(answer, choice="none"), [file, imported], min_confidence=0.8) is None
    with pytest.raises(ValueError, match="not offered"):
        selected_file(replace(answer, choice="3"), [file, imported], min_confidence=0)


def test_scent_matches_case_parts_plurals_and_weights_file_names():
    assert {"http", "request", "policy"} <= set(words("HTTPRequests policies"))
    documents = [
        scent_document("owner", "order_policy.py", "limit", "def limit(): return 10"),
        scent_document("body", "other.py", "describe", 'def describe(): return "order policy"'),
        scent_document("comment", "unrelated.py", "other", "def other(): return 0 # order policy"),
    ]
    scores = ScentIndex(iter(documents)).scores("Order policies", filename_weight=3)
    assert scores["owner"] > scores["body"] > scores["comment"]
    assert not any(ScentIndex(documents).scores("unmentionedToken").values())


def test_graph_uses_real_binding_and_file_relations(sample_index):
    units = list_units(
        sample_index, ["app/orders.py", "app/validation.py"], box_chars=50_000, reading=Reading.CODE
    ).units
    graph = graph_from_index(sample_index, units, cochange_limit=0)
    caller = next(unit for unit in units if unit.symbol.endswith("place"))
    validator = next(unit for unit in units if unit.symbol == "validate_order")
    assert validator.id in graph.adjacency[caller.id]
    assert graph.edge_counts["import"] >= 1
    assert graph.edge_counts["same_file"] == len(units)
    walk = random_walk(graph, {caller.id: 1})
    assert walk[validator.id] > 0


def test_walk_conserves_mass_and_penalizes_hubs_and_handles_dangling_nodes():
    graph = CodeGraph(
        ["seed", "hub", "leaf", "isolated"],
        [GraphEdge("seed", "hub", "call"), GraphEdge("seed", "leaf", "call")]
        + [GraphEdge("hub", str(i), "call") for i in range(10)],
    )
    probabilities = random_walk(graph, {"seed": 1}, hub_penalty=0)
    assert sum(probabilities.values()) == pytest.approx(1)
    penalized = random_walk(graph, {"seed": 1}, hub_penalty=1)
    assert penalized["leaf"] > penalized["hub"]
    assert random_walk(graph, {"isolated": 1})["isolated"] == pytest.approx(1)
    assert not any(random_walk(graph, {"outside": 1}).values())
    with pytest.raises(ValueError):
        random_walk(graph, {"seed": -1})


def test_rank_retains_search_order_on_ties():
    features = {"first": RankFeatures(scent=1), "second": RankFeatures(scent=1), "path": RankFeatures(path=1)}
    assert rank(["first", "second", "path"], features, RankWeights(walk=0)) == ["path", "first", "second"]


def test_active_search_propagates_confirmed_labels_then_honors_request_guard():
    candidates = [str(i) for i in range(48)]
    graph = CodeGraph(candidates, [GraphEdge("0", "47", "call")])

    class Oracle:
        def judge(self, batch):
            return {id: Observation(float(id == "0"), id == "0") for id in batch}

    result = active_search(
        candidates,
        {id: 1 for id in candidates},
        graph,
        Oracle(),
        policy=ActivePolicy(max_requests=2, min_expected_gain=0),
    )
    assert result.batches[0] == tuple(candidates[:16])
    assert result.batches[1][0] == "47"
    assert result.stopped_by == "request guard"
    assert len(result.pending) == 16


def test_active_missing_answers_remain_unknown_and_stop_without_negative_labels():
    class Oracle:
        def judge(self, batch):
            return {batch[0]: Observation(0, False)}

    candidates = [str(i) for i in range(40)]
    result = active_search(candidates, dict.fromkeys(candidates, 1), CodeGraph(candidates), Oracle())
    assert result.stopped_by == "missing oracle answers"
    assert len(result.observations) == 1
    assert len(result.unjudged) == 15
    assert len(result.pending) == 24


def test_active_marginal_rule_stops_after_observed_low_yield():
    class Oracle:
        def judge(self, batch):
            return dict.fromkeys(batch, Observation(0, False))

    candidates = [str(i) for i in range(40)]
    result = active_search(
        candidates,
        dict.fromkeys(candidates, 1),
        CodeGraph(candidates),
        Oracle(),
        policy=ActivePolicy(min_expected_gain=0.6),
    )
    assert len(result.batches) == 1
    assert result.stopped_by == "marginal value"
    assert result.expected_gain == pytest.approx(0.5)
