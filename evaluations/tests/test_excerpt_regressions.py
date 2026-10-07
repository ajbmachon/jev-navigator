"""Regression evidence for branch context and variable origins through the real JVN parser."""

import subprocess
import sys
from pathlib import Path

import pytest
from test_excerpt_rules import repo as repo
from test_excerpt_rules import source_of

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from census import classify, resolve_unregistered_holder  # noqa: E402
from excerpt_rules import Structure, render_excerpt  # noqa: E402

from jev_navigator.index.units import Reading, UnitReader  # noqa: E402


@pytest.mark.parametrize("suffix", ["js", "ts"])
def test_return_in_else_keeps_the_alternative_branch_header(repo, suffix):
    name = f"branch.{suffix}"
    path = repo / name
    path.write_text("""function decide(flag) {
  if (flag) {
    return 1;
  }
  else {
    return 2;
  }
}
""")
    subprocess.run(["git", "add", name], cwd=repo, check=True)
    facts = Structure(repo, [name]).facts(name)
    excerpt = render_excerpt(source_of(path), facts, "", "D")
    kept = {line for start, end in excerpt["runs"] for line in range(start, end + 1)}
    assert {2, 3, 5, 6}.issubset(kept)
    assert "condition line" in classify(5, source_of(path), facts, "")["categories"]
    assert "condition line" not in classify(6, source_of(path), facts, "")["categories"]


def test_property_write_does_not_replace_the_variable_value_origin(repo):
    path = repo / "origin.py"
    path.write_text("""def use():
    value = factory()
    value.unrelated = 1
    return value
""")
    subprocess.run(["git", "add", "origin.py"], cwd=repo, check=True)
    facts = Structure(repo, ["origin.py"]).facts("origin.py")
    # The factory is outside this repository, so E cannot independently keep
    # its call. F must find the local definition from the returned value.
    excerpt = render_excerpt(source_of(path), facts, "", "F")
    kept = {line for start, end in excerpt["runs"] for line in range(start, end + 1)}
    assert 2 in kept
    assert 3 not in kept
    assert 4 in kept


def test_withheld_test_line_resolves_its_whole_holding_function(repo):
    path = repo / "case_test.py"
    path.write_text("""def test_case():
    expected = 1
    actual = 1
    assert actual == expected
""")
    subprocess.run(["git", "add", "case_test.py"], cwd=repo, check=True)
    structure = Structure(repo, ["case_test.py"])
    reader = UnitReader(structure.index, 2_000_000_000, False, Reading.MIXED)
    ranges, symbol, kind = resolve_unregistered_holder(reader, structure.index, "case_test.py", 4)
    assert symbol == "test_case"
    assert kind == "function"
    assert ranges == [[1, 4]]


def test_unmarked_flow_file_uses_recovered_grammar_for_exact_header_and_return(repo):
    path = repo / "typed.js"
    path.write_text("""function use(value: string): string {
  const noise = 1;
  const unused = 2;
  const ignored = 3;
  return value;
}
""")
    subprocess.run(["git", "add", "typed.js"], cwd=repo, check=True)
    facts = Structure(repo, ["typed.js"]).facts("typed.js")
    assert facts["language"] == "flow"
    assert not facts["refused"]
    excerpt = render_excerpt(source_of(path), facts, "", "D")
    assert excerpt["runs"] == [[1, 1], [5, 6]]
    assert "... ELIDED lines 2-4 (3 lines) ..." in excerpt["body"]
