"""Real rarity, parser and rendered-source boundaries for selective focus."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excerpt_rules import Structure, render_selection  # noqa: E402
from selective_rules import owners, rank_terms, select_lines  # noqa: E402
from test_excerpt_rules import repo as fixture_repo  # noqa: E402
from test_excerpt_rules import source_of

repo = fixture_repo


def test_rarest_present_terms_use_existing_index_and_first_mention_ties():
    scent, _ = owners()
    index = scent.ScentIndex(
        [
            scent.scent_document("one", "one.py", "", "common raretoken peer"),
            scent.scent_document("two", "two.py", "", "common steady"),
            scent.scent_document("three", "three.py", "", "common steady"),
        ]
    )
    ranked, absent = rank_terms("missing peer raretoken steady common", index)
    assert [(r["term"], r["df"]) for r in ranked] == [
        ("peer", 1),
        ("raretoken", 1),
        ("steady", 2),
        ("common", 3),
    ]
    assert absent == ["missing"]


def test_python_focus_keeps_multiline_call_and_own_return_not_nested_return(repo):
    path = repo / "sample.py"
    path.write_text("""def outer(flag):
    if flag:
        result = delegated(
            "needle",
            flag,
        )
        def inner():
            return "decoy"
        quiet = 1
        unused = 2
        return result
""")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    facts = Structure(repo, ["sample.py"]).facts("sample.py")
    source = source_of(path)
    chosen, _ = select_lines(source, facts, ("needle",), [], "S3")
    assert {1, 2, 3, 4, 5, 6, 11} <= chosen
    assert not {8, 9, 10} & chosen
    body = render_selection(source, chosen)["body"]
    assert "... ELIDED lines 8-10 (3 lines) ..." in body


@pytest.mark.parametrize("suffix", ["ts", "js"])
def test_script_focus_does_not_add_unrelated_returns(repo, suffix):
    name = f"sample.{suffix}"
    path = repo / name
    path.write_text("""function outer(flag) {
  if (flag) {
    const result = delegated("needle");
    function inner() {
      return "decoy";
    }
    const quiet = 1;
    return result;
  }
}
""")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    facts = Structure(repo, [name]).facts(name)
    chosen, _ = select_lines(source_of(path), facts, ("needle",), [], "S3")
    assert {1, 2, 3, 8, 9, 10} <= chosen
    assert 5 not in chosen and 7 not in chosen


def test_function_cap_preserves_exact_citation_and_prices_all_omissions(repo):
    path = repo / "sample.py"
    path.write_text(
        "def example():\n" + "".join(f"    item{i} = needle\n" for i in range(1, 81)) + "    return None\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    facts = Structure(repo, ["sample.py"]).facts("sample.py")
    source = source_of(path)
    chosen, focus = select_lines(source, facts, ("needle",), [[70, 70]], "C3_20")
    assert len(chosen) == 20 and {1, 70} <= chosen
    assert focus["cited"] == [[70, 70]]
    rendered = render_selection(source, chosen)
    assert rendered["shown_lines"] == 20 and rendered["omitted_lines"] == 62
    assert "... ELIDED lines 20-69 (50 lines) ..." in rendered["body"]
    assert "... ELIDED lines 71-82 (12 lines) ..." in rendered["body"]


def test_agent_window_is_asymmetric_and_clipped_with_exact_markers():
    source = {n: "quiet" for n in range(1, 161)}
    source[80] = "requests"
    chosen, _ = select_lines(source, {"language": None}, ("request",), [], "W3")
    assert chosen == {1, *range(40, 129)}
    body = render_selection(source, chosen)["body"]
    assert "... ELIDED lines 2-39 (38 lines) ..." in body
    assert "... ELIDED lines 129-160 (32 lines) ..." in body
