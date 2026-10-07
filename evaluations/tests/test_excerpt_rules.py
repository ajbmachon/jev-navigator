"""Behavioral proof at the real JVN parser-to-excerpt boundary."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excerpt_rules import Structure, render_excerpt  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Study",
            "-c",
            "user.email=study@local",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def source_of(path):
    return dict(enumerate(path.read_text().splitlines(), 1))


def test_python_condition_chain_and_single_step_value_source(repo):
    path = repo / "sample.py"
    path.write_text("""def save(value):
    origin = value + 1
    unrelated = 8
    if value:
        if origin > 4:
            target = origin
            print(target)
    return None
""")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    facts = Structure(repo, ["sample.py"]).facts("sample.py")
    source = source_of(path)
    c = render_excerpt(source, facts, "target", "C")
    f = render_excerpt(source, facts, "target", "F")
    assert 4 in {n for a, b in c["runs"] for n in range(a, b + 1)}
    assert 1 in {n for a, b in c["runs"] for n in range(a, b + 1)}
    assert 2 in {n for a, b in f["runs"] for n in range(a, b + 1)}
    assert "... ELIDED lines 2-3 (2 lines) ..." in c["body"]
    assert 8 in {n for a, b in f["runs"] for n in range(a, b + 1)}


@pytest.mark.parametrize("suffix", ["ts", "js"])
def test_script_delegation_and_exact_elisions(repo, suffix):
    name = f"sample.{suffix}"
    path = repo / name
    path.write_text("""function helper(x) { return x; }
function main(x) {
  const noise = 2;
  const quiet = 3;
  const ignored = 4;
  const result = helper(x);
  return result;
}
""")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    facts = Structure(repo, [name]).facts(name)
    source = {n: t for n, t in source_of(path).items() if n >= 2}
    a = render_excerpt(source, facts, "", "A")
    e = render_excerpt(source, facts, "", "E")
    assert a["runs"] == [[2, 2], [8, 8]]
    assert "... ELIDED lines 3-7 (5 lines) ..." in a["body"]
    assert e["runs"] == [[2, 2], [6, 8]]
    assert facts["calls"][0]["delegates"][0]["name"] == "helper"
    assert render_excerpt(source, facts, "", "G15")["omitted_lines"] == 0


def test_unsupported_text_keeps_only_lexical_evidence_and_counts_gaps():
    source = {
        10: "first",
        11: "skip",
        12: "skip",
        13: "skip",
        14: "skip",
        15: "skip",
        16: "target",
        17: "skip",
        18: "skip",
        19: "skip",
    }
    result = render_excerpt(source, {"language": None}, "target", "F")
    assert result["runs"] == [[10, 10], [14, 18]]
    assert "... ELIDED lines 11-13 (3 lines) ..." in result["body"]
    assert "... ELIDED lines 19-19 (1 lines) ..." in result["body"]
