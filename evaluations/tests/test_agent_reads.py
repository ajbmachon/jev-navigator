"""Regressions for receipt shell boundaries and verified source delivery."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_reads import bind_output, requests  # noqa: E402


@pytest.fixture(scope="module")
def discovery():
    path = Path(
        os.environ.get(
            "JVN_DISCOVERY_ANALYZE",
            str(Path.home() / ".local/share/jvn-takeover/2026-10-03/discovery/analyze.py"),
        )
    )
    spec = importlib.util.spec_from_file_location("discovery_parser_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("separator", ["\n", "\n\n", "; ", " &&\n"])
def test_shell_statement_boundaries_keep_each_requested_file_range(discovery, separator):
    command = (
        "nl -ba enginepy/host/route_plan.py | sed -n '44,49p'"
        + separator
        + "nl -ba enginepy/host/fake_scan_runtime.py | sed -n '16,25p'"
    )
    assert requests(discovery.pipelines(command)) == [
        {"file": "enginepy/host/route_plan.py", "start": 44, "end": 49, "numbered": True},
        {"file": "enginepy/host/fake_scan_runtime.py", "start": 16, "end": 25, "numbered": True},
    ]


def test_newline_after_pipe_continues_the_same_read(discovery):
    command = "nl -ba route.py |\n sed -n '3,8p'\ncat adapter.py"
    assert requests(discovery.pipelines(command)) == [
        {"file": "route.py", "start": 3, "end": 8, "numbered": True},
        {"file": "adapter.py", "start": 1, "end": None, "numbered": False},
    ]


def test_quoted_script_newlines_do_not_invent_shell_reads(discovery):
    command = "python3 -c 'print(1)\nprint(2)'\nnl -ba route.py | sed -n '3,8p'"
    pipelines = discovery.pipelines(command)
    assert pipelines[0] == [["python3", "-c", "print(1)\nprint(2)"]]
    assert requests(pipelines) == [
        {"file": "route.py", "start": 3, "end": 8, "numbered": True},
    ]


def test_truncated_output_earns_only_emitted_source_not_requested_range():
    reads = [{"file": "route.py", "start": 1, "end": 3, "numbered": True}]
    source = {"route.py": ["def route():", "    return 1", ""]}
    observed, ambiguous = bind_output("     1\tdef route():\n[output truncated]", reads, source)
    assert observed == {"route.py": {1}}
    assert ambiguous == 0


def test_repeated_numbered_lines_bind_only_with_unambiguous_source_context():
    source = {"a.py": ["def a():", "    return 1", ""], "b.py": ["def b():", "    return 2", ""]}
    reads = [{"file": file, "start": 1, "end": 3, "numbered": True} for file in source]
    observed, ambiguous = bind_output("     3\t", reads, source)
    assert not observed
    assert ambiguous == 1
    observed, ambiguous = bind_output("     2\t    return 1\n     3\t", reads, source)
    assert observed == {"a.py": {2, 3}}
    assert ambiguous == 0
