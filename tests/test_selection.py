"""Selection's real contracts: visible summaries, bound graph links, ordering and oracle admission."""

import pickle
from dataclasses import replace
from pathlib import Path

import pytest
from git_repos import git, write_files

from jev_navigator.index.code_index import CodeIndex
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


def test_cochange_keeps_a_relation_present_only_in_the_other_files_top_list(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    write_files(
        tmp_path,
        {"a.py": "def a(): return 0\n", "b.py": "def b(): return 0\n", "z.py": "def z(): return 0\n"},
    )
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    for revision in range(1, 4):
        write_files(
            tmp_path, {"a.py": f"def a(): return {revision}\n", "b.py": f"def b(): return {revision}\n"}
        )
        git(tmp_path, "commit", "-qam", "a and b change together")
    write_files(tmp_path, {"a.py": "def a(): return 4\n", "z.py": "def z(): return 4\n"})
    git(tmp_path, "commit", "-qam", "a and z change together")
    index = CodeIndex.from_git(tmp_path)
    units = list_units(index, ["a.py", "z.py"], box_chars=50_000).units
    graph = graph_from_index(index, units, cochange_limit=1)
    assert graph.edge_counts["cochange"] == 1
    assert "file:z.py" in graph.adjacency["file:a.py"]


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


@pytest.mark.parametrize(
    "literal",
    [
        '"https://example.test/#billing_policy"',
        "'https://example.test/#billing_policy'",
        '"""first line\n#billing_policy\nlast line"""',
        'r"https://example.test/#billing_policy"',
        'f"https://example.test/#{billing_policy}"',
        '"escaped \\" #billing_policy"',
    ],
)
def test_python_scent_keeps_hashes_in_strings_beside_real_comments(literal):
    document = scent_document(
        "unit", "other.py", "describe", f"def describe():\n    return {literal} # hidden_comment\n"
    )
    index = ScentIndex([document])
    assert index.scores("billing policy")["unit"] > 0
    assert index.scores("hidden comment")["unit"] == 0


@pytest.mark.parametrize(
    "scores",
    [
        {"worst": -0.1, "preferred": -0.01},
        {"worst": -0.1, "preferred": -0.01, "positive": 0.2},
        {"worst": -0.1, "preferred": 0},
        {"worst": 0.01, "preferred": 0.1},
        {"worst": -0.1, "preferred": -0.1},
        {"worst": 0, "preferred": 0},
    ],
)
def test_active_search_preserves_signed_score_order_at_the_oracle(scores):
    candidates = list(scores)
    received = []

    class Oracle:
        def judge(self, batch):
            received.append(batch)
            return dict.fromkeys(batch, Observation(0, False))

    result = active_search(candidates, scores, CodeGraph(candidates), Oracle())
    assert received == [tuple(sorted(candidates, key=lambda id: -scores[id]))]
    assert 0 <= result.expected_gain <= len(candidates)


@pytest.mark.parametrize(
    "source",
    [
        'def describe():\n    return f"{"#billing_policy"}" # hidden_comment\n',
        "def describe():\n    return f'''{\n        1 # hidden_comment\n    } billing_policy'''\n",
    ],
    ids=["nested-same-quote-string", "multiline-expression-comment"],
)
def test_python_fstring_scent_distinguishes_expression_comments_from_strings(source):
    document = scent_document("unit", "other.py", "describe", source)
    index = ScentIndex([document])
    assert index.scores("billing policy")["unit"] > 0
    assert index.scores("hidden comment")["unit"] == 0


@pytest.mark.parametrize("path", ["other.js", "other.ts", "other.tsx"])
def test_template_scent_distinguishes_nested_strings_and_expression_comments(path):
    source = "const describe = () => `nested ${`#billing_policy`} ${1 /* hidden_comment */}`;"
    index = ScentIndex([scent_document("unit", path, "describe", source)])
    assert index.scores("billing policy")["unit"] > 0
    assert index.scores("hidden comment")["unit"] == 0


def test_saved_scent_index_answers_as_the_built_one(tmp_path):
    index = ScentIndex(
        [
            scent_document("owner", "order_policy.py", "limit", "def limit(): return 10", test=False),
            scent_document("body", "other.py", "describe", 'def describe(): return "order"', test=True),
            scent_document("empty", "empty.py", "", ""),
        ],
        k1=1.5,
        b=0.5,
    )
    path = tmp_path / "scent file.sqlite"
    index.save(path)
    assert not list(tmp_path.glob("*.partial"))
    saved = ScentIndex.load(path)
    assert (saved.ids, saved.tests, saved.lengths, saved.average_length, saved.k1, saved.b) == (
        index.ids,
        index.tests,
        index.lengths,
        index.average_length,
        index.k1,
        index.b,
    )
    for text in ("order policy", "limit describe", "unmentionedToken", ""):
        assert saved.signals(text) == index.signals(text)
        assert saved.scores(text) == index.scores(text)


def test_saved_scent_index_named_relatively_keeps_reading_after_the_directory_changes(tmp_path, monkeypatch):
    index = ScentIndex([scent_document("owner", "order_policy.py", "limit", "def limit(): return 10")])
    index.save(tmp_path / "scent.sqlite")
    monkeypatch.chdir(tmp_path)
    saved = ScentIndex.load(Path("scent.sqlite"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert saved.signals("limit") == index.signals("limit")


def test_compact_graph_answers_as_the_plain_one():
    edges = [
        GraphEdge("a", "b", "call"),
        GraphEdge("c", "a", "call", 2.5),
        GraphEdge("b", "c", "reference"),
        GraphEdge("a", "b", "reference", 0.5),
        GraphEdge("a", "file:x", "same_file"),
    ]
    plain = CodeGraph(["a", "b", "c", "alone"], edges)
    compact = CodeGraph(["a", "b", "c", "alone"], edges)
    compact.compact()
    assert list(compact.adjacency) == list(plain.adjacency)
    for node, neighbours in plain.adjacency.items():
        assert list(compact.adjacency[node].items()) == list(neighbours.items())
        assert dict(compact.adjacency[node]) == neighbours
    assert "file:x" in compact.adjacency and "missing" not in compact.adjacency
    assert compact.adjacency["a"]["c"] == 2.5
    assert "b" in compact.adjacency["a"] and "alone" not in compact.adjacency["a"]
    assert compact.adjacency.get("missing", {}) == {}
    assert compact.degrees() == plain.degrees()
    assert random_walk(compact, {"a": 1}) == random_walk(plain, {"a": 1})
    assert pickle.loads(pickle.dumps(compact)).degrees() == plain.degrees()
    with pytest.raises(AttributeError):
        compact.add(GraphEdge("a", "alone", "call"))
    with pytest.raises(TypeError):
        compact.degrees()["a"] = 100
