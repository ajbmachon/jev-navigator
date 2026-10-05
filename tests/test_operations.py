from __future__ import annotations

from pathlib import Path

import pytest

from jev_navigator import operations
from jev_navigator.index.code_index import CodeIndex


def test_slice_around_returns_the_enclosing_function(sample_index: CodeIndex) -> None:
    # Act
    code = operations.slice_around(sample_index, "app/validation.py", 11)

    # Assert
    assert code.span.name == "check_limits"
    assert code.text.startswith("def check_limits(order):")


def test_slice_around_falls_back_to_a_window_at_module_level(sample_index: CodeIndex) -> None:
    # Act
    code = operations.slice_around(sample_index, "app/validation.py", 1, radius=2)

    # Assert
    assert (code.span.start, code.span.end) == (1, 3)


def test_callers_of_file_collects_calls_from_other_files_for_every_function(sample_index: CodeIndex) -> None:
    # Act
    sites = operations.callers_of_file(sample_index, "app/validation.py")

    # Assert
    assert [(site.file, site.caller.name) for site in sites] == [("app/orders.py", "place")]


def test_trace_callers_walks_outward_hop_by_hop(sample_index: CodeIndex) -> None:
    # Act
    steps = operations.trace_callers(sample_index, "check_limits", depth=2)

    # Assert
    assert [(step.hop, step.function.name, step.reached_from) for step in steps] == [
        (1, "validate_order", "check_limits"),
        (2, "place", "validate_order"),
    ]


def test_trace_without_depth_reaches_the_static_fixed_point_beyond_three_hops(
    tmp_path: Path,
) -> None:
    source = (
        "\n\n".join(f"def step_{number}():\n    return step_{number + 1}()" for number in range(5))
        + "\n\ndef step_5():\n    return 'done'\n"
    )
    (tmp_path / "chain.py").write_text(source)
    index = CodeIndex.from_directory(tmp_path)

    # Act
    steps = operations.trace_callees(index, "step_0")

    # Assert
    assert [(step.hop, step.function.name) for step in steps] == [
        (1, "step_1"),
        (2, "step_2"),
        (3, "step_3"),
        (4, "step_4"),
        (5, "step_5"),
    ]


def test_trace_keeps_every_neighbour_and_frontier_member(tmp_path: Path) -> None:
    branches = "\n".join(f"    branch_{number}()" for number in range(17))
    functions = "\n\n".join(
        f"def branch_{number}():\n    return leaf_{number}()\n\ndef leaf_{number}():\n    return {number}"
        for number in range(17)
    )
    (tmp_path / "wide.py").write_text(f"def root():\n{branches}\n\n{functions}\n")
    index = CodeIndex.from_directory(tmp_path)

    # Act
    steps = operations.trace_callees(index, "root", depth=2)
    graph = operations.trace_graph(index, index.find_definition("root"))

    # Assert
    assert len([step for step in steps if step.hop == 1]) == 17
    assert len([step for step in steps if step.hop == 2]) == 17
    assert len(graph.functions) == 35
    assert graph.stop == "fixed_point"


def test_trace_rejects_negative_caller_depth(sample_index: CodeIndex) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        operations.trace_callers(sample_index, "check_limits", depth=-1)


def test_trace_callees_follows_only_functions_defined_in_scope(sample_index: CodeIndex) -> None:
    # Act
    steps = operations.trace_callees(sample_index, "validate_order", depth=2)

    # Assert
    assert [(step.hop, step.function.name) for step in steps] == [(1, "check_limits")]


def test_similar_functions_keeps_only_functions_sharing_calls_or_name_words(sample_index: CodeIndex) -> None:
    # Act
    names = [span.name for span in operations.similar_functions(sample_index, "handleOrder")]

    # Assert
    assert "parseOrder" in names
    assert "registerRoutes" not in names
    assert "handleOrder" not in names


def test_code_named_in_doc_finds_mentioned_functions(sample_index: CodeIndex) -> None:
    # Act
    spans = operations.code_named_in_doc(
        sample_index, "Orders are checked by `validate_order`, which calls check_limits."
    )

    # Assert
    assert [span.name for span in spans] == ["validate_order", "check_limits"]


def test_comment_above_a_function_describes_the_whole_function(sample_index: CodeIndex) -> None:
    # Act
    code = operations.code_described_by_comment(sample_index, "app/comments.py", 24)

    # Assert
    assert (code.span.start, code.span.end, code.span.name) == (25, 26, "Basket")


def test_comment_above_a_decorated_function_includes_the_decorator(sample_index: CodeIndex) -> None:
    # Act
    code = operations.code_described_by_comment(sample_index, "app/comments.py", 3)

    # Assert
    assert (code.span.start, code.span.end, code.span.name) == (4, 6, "charge")
    assert code.text.startswith("@functools.cache")


def test_comment_followed_by_blank_lines_reaches_the_next_symbol(sample_index: CodeIndex) -> None:
    # Act
    code = operations.code_described_by_comment(sample_index, "app/comments.py", 9)

    # Assert
    assert (code.span.start, code.span.end, code.span.name) == (11, 12, "normalise")


def test_trailing_comment_on_a_simple_statement_describes_that_line(sample_index: CodeIndex) -> None:
    # Act
    code = operations.code_described_by_comment(sample_index, "app/comments.py", 16)

    # Assert
    assert (code.span.start, code.span.end) == (16, 16)


def test_trailing_comment_on_a_block_opener_describes_the_whole_block(sample_index: CodeIndex) -> None:
    # Act
    code = operations.code_described_by_comment(sample_index, "app/comments.py", 17)

    # Assert
    assert (code.span.start, code.span.end) == (17, 19)


def test_code_named_in_doc_finds_constants_and_classes(sample_index: CodeIndex) -> None:
    # Act
    spans = operations.code_named_in_doc(sample_index, "OrderService reads LIMITS_KEY before placing.")

    # Assert
    assert [span.name for span in spans] == ["OrderService", "LIMITS_KEY"]


def test_trace_steps_carry_the_binding_of_the_linking_call(sample_index: CodeIndex) -> None:
    # Act
    steps = operations.trace_callers(sample_index, "check_limits", depth=2)

    # Assert
    assert [(step.function.name, step.binding.status) for step in steps] == [
        ("validate_order", "resolved"),
        ("place", "resolved"),
    ]


def test_trace_graph_keeps_registration_and_unresolved_call_links(tmp_path: Path) -> None:
    (tmp_path / "workflow.py").write_text(
        "REGISTRY = {'orders': handle}\n\n"
        "def handle(value):\n"
        "    transformed = normalize(value)\n"
        "    missing_sink(transformed)\n\n"
        "def normalize(value):\n"
        "    return value.strip()\n"
    )
    index = CodeIndex.from_directory(tmp_path)
    root = index.find_definition("handle")[0]

    # Act
    graph = operations.trace_graph(index, [root])

    # Assert
    assert graph.stop == "fixed_point"
    assert {span.name for span in graph.functions} == {"handle", "normalize"}
    registration = next(link for link in graph.links if link.relation == "collection")
    assert registration.source is None
    assert registration.target == root
    assert registration.binding is not None and registration.binding.proven
    unresolved = next(link for link in graph.links if link.name == "missing_sink")
    assert unresolved.target is None
    assert unresolved.binding is not None and unresolved.binding.status == "unresolved"


def test_trace_graph_stops_at_the_callers_cancellation_boundary(sample_index: CodeIndex) -> None:
    root = sample_index.find_definition("check_limits")[0]

    # Act
    graph = operations.trace_graph(sample_index, [root], cancelled=lambda: True)

    # Assert
    assert graph.stop == "cancelled"
    assert graph.functions == (root,)
    assert graph.links == ()


def test_trace_links_one_call_site_once_even_from_the_declaration_holding_its_function(
    tmp_path: Path,
) -> None:
    """The declaration `load` holds its arrow function: Trace links the declaration to the function
    and the caller to the function, never the caller to the declaration as well."""
    # Arrange
    (tmp_path / "jobs.ts").write_text(
        "export const load =\n  async () => {\n    return 1;\n  };\n\n"
        "export function run() {\n  return load();\n}\n"
    )
    index = CodeIndex.from_directory(tmp_path)
    declaration = next(span for span in index.find_definition("load") if span.start == 1)
    function = next(span for span in index.find_definition("load") if span.start == 2)

    # Act
    graph = operations.trace_graph(index, [declaration])

    # Assert
    calls = [link for link in graph.links if link.relation == "call" and link.line == 7]
    assert [(link.source.name, link.target) for link in calls] == [("run", function)]
    contains = [link for link in graph.links if link.relation == "contains"]
    assert [(link.source, link.target) for link in contains] == [(declaration, function)]


@pytest.mark.parametrize("use", ["return handle(value)", "return [handle]"])
def test_trace_does_not_follow_references_bound_to_a_different_same_named_function(
    tmp_path: Path, use: str
) -> None:
    (tmp_path / "orders.py").write_text("def handle(value):\n    return value\n")
    (tmp_path / "payments.py").write_text("def handle(value):\n    return value\n")
    (tmp_path / "entry.py").write_text(f"from payments import handle\n\ndef entry(value):\n    {use}\n")
    index = CodeIndex.from_directory(tmp_path)
    root = next(span for span in index.find_definition("handle") if span.file == "orders.py")

    graph = operations.trace_graph(index, [root])

    assert graph.functions == (root,)
    assert graph.links == ()

    actual_target = next(span for span in index.find_definition("handle") if span.file == "payments.py")
    actual_graph = operations.trace_graph(index, [actual_target])
    assert {span.file for span in actual_graph.functions} == {"payments.py", "entry.py"}
    assert any(link.target == actual_target for link in actual_graph.links)
