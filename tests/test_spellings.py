"""The spelling map: a lookup of one word meets every real spelling of it in scope, whatever its case,
separators or plural, rarest first, and each file content's words are read once."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator import housekeeping
from jev_navigator.cli import main
from jev_navigator.confirmation import today
from jev_navigator.directives.find_all import CODE_SOURCES, find_all
from jev_navigator.index import spellings
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spellings import SpellingPlace, SpellingTable
from jev_navigator.index.units import LineAnchor
from jev_navigator.judgments.judge import Judge
from jev_navigator.sources import NAMES, SPELLINGS, TEXT_SPELLINGS, Seeds, SpellingSource
from jev_navigator.testing import ScriptedJevClient

SITES = {
    "src/queries/website.ts": (
        "export async function createWebsite(data) {\n"
        "  return db.websites.insert(data);\n"
        "}\n"
        "export const WEBSITE_ID = 'main';\n"
    ),
    "app/web_site.py": "def update_web_site(row):\n    return row\n",
    "config/app.yaml": "websites:\n  default: main\n",
    "docs/notes.md": "# Notes\nThe Website table holds one row per site.\n",
    "src/orders.ts": "export function placeOrder() {\n  return 1;\n}\n",
}
TARGET = {"writes": "code that writes a row of the website table"}


def indexed(root: Path, files: Mapping[str, str] = SITES) -> CodeIndex:
    commit_files(root, files)
    return CodeIndex.from_git(root)


def spelled(index: CodeIndex, term: str) -> set[tuple[str, bool]]:
    return {(spelling.word, spelling.file_name) for spelling in index.names(term)}


def test_a_lookup_meets_every_spelling_of_a_word_whatever_its_case_separators_or_plural(
    tmp_path: Path,
) -> None:
    # Arrange
    index = indexed(tmp_path)

    # Act
    found = spelled(index, "website")

    # Assert
    assert found == {
        ("createWebsite", False),
        ("websites", False),
        ("WEBSITE_ID", False),
        ("update_web_site", False),
        ("Website", False),
        ("website.ts", True),
        ("web_site.py", True),
    }


def test_spellings_come_rarest_first_each_naming_its_files_and_lines(tmp_path: Path) -> None:
    # Arrange
    index = indexed(tmp_path)

    # Act
    found = index.names("websites")

    # Assert
    counts = [spelling.files for spelling in found]
    assert counts == sorted(counts)
    [plural] = [spelling for spelling in found if spelling.word == "websites"]
    assert plural.files == 2
    assert plural.places == (
        SpellingPlace("config/app.yaml", (1,)),
        SpellingPlace("src/queries/website.ts", (2,)),
    )


@pytest.mark.parametrize(
    ("term", "spelling"),
    [
        ("status", "statuses"),
        ("statuses", "status"),
        ("policy", "policies"),
        ("classes", "class"),
        ("box", "boxes"),
        ("case", "cases"),
        ("alias", "aliases"),
        ("aliases", "alias"),
        ("userId", "user_ids"),
    ],
)
def test_a_singular_and_its_plural_meet(tmp_path: Path, term: str, spelling: str) -> None:
    # Arrange
    words = "status statuses policies class boxes cases alias aliases user_ids state"
    index = indexed(tmp_path, {"words.txt": f"{words}\n"})

    # Act
    found = spelled(index, term)

    # Assert
    assert (spelling, False) in found
    assert ("state", False) not in found


def test_a_file_name_is_found_by_its_stem_or_whole_name_never_by_its_suffix_alone(tmp_path: Path) -> None:
    # Arrange
    index = indexed(tmp_path)

    # Act
    by_name, by_suffix = spelled(index, "website.ts"), spelled(index, "ts")

    # Assert
    assert ("website.ts", True) in by_name
    assert not any(file_name for _, file_name in by_suffix)


def test_an_env_template_spells_only_its_keys_and_env_files_and_lockfiles_spell_nothing(
    tmp_path: Path,
) -> None:
    # Arrange
    files = {
        ".env.example": "WEBSITE_URL=https://hidden-website-host.example\n",
        ".env": "WEBSITE_TOKEN=abc\n",
        "package-lock.json": '{"website-kit": "1.0.0"}\n',
    }
    index = indexed(tmp_path, files)

    # Act
    found = spelled(index, "website")

    # Assert
    assert found == {("WEBSITE_URL", False)}


def test_max_files_caps_the_files_each_spelling_names_and_counts_the_rest(tmp_path: Path) -> None:
    # Arrange
    index = indexed(tmp_path)

    # Act
    [plural] = [spelling for spelling in index.names("websites", max_files=1) if spelling.word == "websites"]

    # Assert
    assert (plural.files, len(plural.places), plural.capped) == (2, 1, 1)


def test_a_second_index_reads_each_unchanged_contents_words_from_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    indexed(tmp_path / "first").names("website")
    changed = {**SITES, "src/orders.ts": "export function placeWebsiteOrder() {}\n"}
    second = indexed(tmp_path / "second", changed)
    read: list[str] = []
    real_words_in = spellings.words_in

    def recorded(file: str, content: bytes) -> dict[str, tuple[int, ...]]:
        read.append(file)
        return real_words_in(file, content)

    monkeypatch.setattr(spellings, "words_in", recorded)

    # Act
    found = spelled(second, "website")

    # Assert
    assert read == ["src/orders.ts"]
    assert ("placeWebsiteOrder", False) in found


def test_spelling_rows_go_once_their_contents_went_unconfirmed_for_thirty_days(tmp_path: Path) -> None:
    # Arrange
    indexed(tmp_path / "kept").names("website")
    indexed(tmp_path / "gone", {"lib/other.py": "def elsewhere(x):\n    return x\n"}).names("x")
    with sqlite3.connect(SpellingTable().path) as database:
        database.execute("update words set confirmed = ?", (today() - 40,))
    CodeIndex.from_git(tmp_path / "kept").names("website")

    # Act
    sweep = housekeeping.prune()

    # Assert
    assert sweep.forgotten["spellings"] == 1
    assert SpellingTable().confirmations(before=today()).held == len(SITES)


def test_the_spelling_source_reaches_each_spelling_rarest_first_by_the_word_it_spells(tmp_path: Path) -> None:
    # Arrange: placeOrder is a request name; website is a word of the description
    index = indexed(tmp_path)
    seeds = Seeds(names=("placeOrder",), texts=("code that writes the website table",))

    # Act
    reaches = SPELLINGS.reach(index, seeds)

    # Assert
    assert {(reach.at, reach.seed) for reach in reaches} == {
        (LineAnchor("src/orders.ts", 1), "placeOrder"),
        (LineAnchor("src/queries/website.ts", 1), "website"),
        (LineAnchor("src/queries/website.ts", 2), "website"),
        (LineAnchor("src/queries/website.ts", 4), "website"),
        (LineAnchor("app/web_site.py", 1), "website"),
        ("src/queries/website.ts", "website"),
        ("app/web_site.py", "website"),
    }
    assert all(reach.names == {reach.seed} and reach.distance == 3 for reach in reaches)
    assert (reaches[0].at, reaches[-1].at) == (
        LineAnchor("src/queries/website.ts", 4),
        LineAnchor("src/queries/website.ts", 2),
    )


def test_the_spelling_source_leaves_out_a_spelling_more_files_hold_than_its_cap(tmp_path: Path) -> None:
    # Arrange: three files spell common_word, one spells rareWord
    files = {f"src/m{n}.py": "common_word = 1\n" for n in range(3)} | {"src/r.py": "rareWord = 2\n"}
    index = indexed(tmp_path, files)
    seeds = Seeds(texts=("the common_word and the rareWord",))

    # Act
    capped = SpellingSource(max_files=2).reach(index, seeds)
    every = SpellingSource(max_files=None).reach(index, seeds)

    # Assert
    assert [reach.at for reach in capped] == [LineAnchor("src/r.py", 1)]
    assert [reach.at for reach in every] == [
        LineAnchor("src/r.py", 1),
        *(LineAnchor(f"src/m{n}.py", 1) for n in range(3)),
    ]


def test_the_text_spelling_source_keeps_only_text_files_that_are_not_lockfiles(tmp_path: Path) -> None:
    # Arrange
    index = indexed(tmp_path, {**SITES, "package-lock.json": '{"website-kit": "1.0.0"}\n'})

    # Act
    reaches = TEXT_SPELLINGS.reach(index, Seeds(texts=("the website table",)))

    # Assert
    assert {reach.at for reach in reaches} == {
        LineAnchor("config/app.yaml", 1),
        LineAnchor("docs/notes.md", 2),
    }


def test_find_all_judges_the_units_the_spelling_source_reaches_only_when_the_caller_adds_it(
    tmp_path: Path,
) -> None:
    # Arrange
    index = indexed(tmp_path)

    # Act
    default = find_all(index, Judge(ScriptedJevClient()), TARGET, names=["placeOrder"])
    composed = find_all(
        index, Judge(ScriptedJevClient()), TARGET, names=["placeOrder"], sources=(*CODE_SOURCES, SPELLINGS)
    )

    # Assert
    assert {unit.symbol for unit in default.units} == {"placeOrder"}
    assert {unit.symbol: composed.entered_by[unit.id] for unit in composed.units} == {
        "placeOrder": NAMES.name,
        "createWebsite": SPELLINGS.name,
        "update_web_site": SPELLINGS.name,
        "<top level>": SPELLINGS.name,
    }


def test_jvn_names_lists_each_spelling_with_its_files_and_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    commit_files(tmp_path, SITES)

    # Act
    text_exit = main(["names", "createWebsite", "--repo", str(tmp_path)])
    text = capsys.readouterr().out
    json_exit = main(["--json", json.dumps({"command": "names", "target": "website", "repo": str(tmp_path)})])
    request = json.loads(capsys.readouterr().out)

    # Assert
    assert (text_exit, json_exit) == (0, 0)
    assert text.splitlines()[1] == "createWebsite  [1 file]  src/queries/website.ts:1"
    assert request["keys"] == ["website", "websitee"]
    assert {(s["word"], s["file_name"]) for s in request["spellings"]} == spelled(
        CodeIndex.from_git(tmp_path), "website"
    )
