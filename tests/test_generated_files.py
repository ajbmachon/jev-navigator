"""Jev's generated-file judgment on real git repositories: what each flagged file's entry holds, who
imports it, and how answers and refusals map back to files. Answers come from the scripted client;
everything else is the real scope, index, masker and judge."""

from __future__ import annotations

import tracemalloc
from collections.abc import Mapping
from pathlib import Path

import pytest
from git_repos import commit_all, write_files

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.file_shape import shape_of
from jev_navigator.index.scope import ResolvedScope, Scope, resolve_scope
from jev_navigator.judgments.generated_files import (
    EXCERPT_CHARS,
    GENERATED_FILE,
    MAX_IMPORTERS,
    MAX_NAMED_BY,
    NAMING_LINE_CHARS,
    NOT_JUDGED_SECRET,
    files_naming,
    generated_file_entry,
    importers_of,
    judge_generated_files,
)
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.secrets import SecretMasker, mask_request
from jev_navigator.testing import ScriptedJevClient

BUNDLE = "".join(f"var a{number}=function(){{return {number}}};" for number in range(800)) + "\n"
SECRET_MARK = "LOOKS-LIKE-A-KEY"


def _repository(root: Path, files: dict[str, str]) -> Path:
    root.mkdir()
    write_files(root, files)
    commit_all(root)
    return root


def _awaiting(repo: Path) -> tuple[CodeIndex, Mapping]:
    resolved = resolve_scope(
        Scope(
            repo=repo,
            with_tests=False,
            with_generated=False,
            with_vendored=False,
            with_docs=False,
            max_files=200,
        )
    )
    assert isinstance(resolved, ResolvedScope)
    return CodeIndex(repo, resolved.files), resolved.awaiting_generated_judgment


def test_a_flagged_file_reaches_jev_as_its_path_measured_facts_and_two_excerpts(tmp_path: Path) -> None:
    text = BUNDLE.removesuffix("\n")
    repo = _repository(tmp_path / "repo", {"web/bundle.js": text})
    index, awaiting = _awaiting(repo)
    shape = shape_of(repo, "web/bundle.js")
    middle_start = (len(text) - EXCERPT_CHARS) // 2

    entry = generated_file_entry(index, "web/bundle.js", awaiting["web/bundle.js"], naming=())

    assert entry == {
        "file": "web/bundle.js",
        "size_bytes": shape.size_bytes,
        "line_count": shape.line_count,
        "longest_line_bytes": shape.longest_line,
        "average_line_bytes": round(shape.chars_per_line, 1),
        "importers": [],
        "importer_count": 0,
        "named_by": [],
        "named_by_count": 0,
        "opening": text[:EXCERPT_CHARS],
        "middle": text[middle_start : middle_start + EXCERPT_CHARS],
    }


def test_importers_are_the_files_whose_imports_resolve_to_it_capped_with_the_true_count(
    tmp_path: Path,
) -> None:
    importers = {f"web/use{number:02}.ts": 'import { a0 } from "./bundle";\n' for number in range(12)}
    mention_only = {"web/notes.ts": "// the bundle is rebuilt nightly\nexport const n = 1;\n"}
    repo = _repository(tmp_path / "repo", {"web/bundle.js": BUNDLE, **importers, **mention_only})
    index, _ = _awaiting(repo)

    entry = generated_file_entry(index, "web/bundle.js", shape_of(repo, "web/bundle.js"), naming=())

    assert entry["importers"] == sorted(importers)[:MAX_IMPORTERS]
    assert entry["importer_count"] == 12


def test_finding_importers_runs_no_fact_scan(tmp_path: Path) -> None:
    repo = _repository(tmp_path / "repo", {"web/bundle.js": BUNDLE, "web/use.ts": 'import "./bundle";\n'})
    scans: list[tuple[str, str, int]] = []
    fact_cache = tmp_path / "facts"
    index = CodeIndex(
        repo,
        ["web/bundle.js", "web/use.ts"],
        scan_observer=lambda *scan: scans.append(scan),
        fact_cache_dir=fact_cache,
    )

    found = importers_of(index, "web/bundle.js")

    assert found == ("web/use.ts",)
    assert scans == []
    assert not fact_cache.exists() or not any(fact_cache.rglob("*"))


def test_each_flagged_file_gets_one_question_naming_it_and_its_own_answer(tmp_path: Path) -> None:
    repo = _repository(tmp_path / "repo", {"web/a.js": BUNDLE, "web/b.js": "/*b*/" + BUNDLE})
    index, awaiting = _awaiting(repo)
    by_file = {"web/a.js": 0.93, "web/b.js": 0.12}
    client = ScriptedJevClient(nouls=lambda question_id, question, state: _scripted(question, state, by_file))

    judgments = judge_generated_files(Judge(client), index, awaiting)

    assert {path: result.probability for path, result in judgments.judged.items()} == by_file
    assert dict(judgments.not_judged) == {}
    [(state, questions)] = client.requests
    assert list(state) == ["files"]
    assert sorted(question["instructions"] for question in questions.values()) == [
        GENERATED_FILE.to_question(f"files[{slot}]")["instructions"] for slot in range(2)
    ]


def test_a_file_the_secret_scanner_refuses_stays_named_as_not_judged(tmp_path: Path) -> None:
    repo = _repository(
        tmp_path / "repo", {"web/a.js": BUNDLE, "web/keyed.js": f"/* {SECRET_MARK} */" + BUNDLE}
    )
    index, awaiting = _awaiting(repo)
    client = ScriptedJevClient(default_noul=0.9)

    judgments = judge_generated_files(Judge(client, scanner=_MarkScanner()), index, awaiting)

    assert list(judgments.judged) == ["web/a.js"]
    assert dict(judgments.not_judged) == {"web/keyed.js": NOT_JUDGED_SECRET}
    assert all(SECRET_MARK not in str(state) for state, _ in client.requests)


def test_files_that_name_a_flagged_path_reach_jev_non_test_files_first_capped_with_the_true_count(
    tmp_path: Path,
) -> None:
    test_namers = {
        f"app/t{number}.test.mjs": 'const copy = join(dir, "web/bundle.js");\n' for number in range(5)
    }
    lookalikes = (
        "// web/bundle.json, lib/web/bundle.js, https://x.test/main/web/bundle.js\nexport const n = 1;\n"
    )
    repo = _repository(
        tmp_path / "repo",
        {
            "web/bundle.js": "// web/bundle.js\n" + BUNDLE,
            "scripts/build.mjs": 'writeFileSync("./web/bundle.js", out);\nlog("wrote web/bundle.js");\n',
            "README.md": "The build rewrites [/web/bundle.js](web/bundle.js).\n",
            "web/notes.ts": lookalikes,
            **test_namers,
        },
    )
    index, awaiting = _awaiting(repo)

    naming = files_naming(repo, list(awaiting))
    entry = generated_file_entry(index, "web/bundle.js", awaiting["web/bundle.js"], naming["web/bundle.js"])

    assert entry["named_by"] == [
        {"file": "README.md", "line": 1, "text": "The build rewrites [/web/bundle.js](web/bundle.js)."},
        {"file": "scripts/build.mjs", "line": 1, "text": 'writeFileSync("./web/bundle.js", out);'},
        *(
            {"file": file, "line": 1, "text": 'const copy = join(dir, "web/bundle.js");'}
            for file in sorted(test_namers)[: MAX_NAMED_BY - 2]
        ),
    ]
    assert entry["named_by_count"] == 7


def test_a_long_naming_line_reaches_jev_as_a_window_that_keeps_the_path(tmp_path: Path) -> None:
    line = "alpha " * 80 + 'copy("web/bundle.js") ' + "omega " * 80
    repo = _repository(tmp_path / "repo", {"web/bundle.js": BUNDLE, "scripts/copy.mjs": line + "\n"})
    index, awaiting = _awaiting(repo)

    naming = files_naming(repo, list(awaiting))
    [named] = generated_file_entry(
        index, "web/bundle.js", awaiting["web/bundle.js"], naming["web/bundle.js"]
    )["named_by"]

    assert NAMING_LINE_CHARS - len("alpha") <= len(named["text"]) <= NAMING_LINE_CHARS
    assert 'copy("web/bundle.js")' in named["text"]
    assert named["text"] in line


def test_one_line_naming_two_flagged_files_names_each_of_them(tmp_path: Path) -> None:
    repo = _repository(
        tmp_path / "repo",
        {
            "web/a.js": BUNDLE,
            "web/b.js": "/*b*/" + BUNDLE,
            "scripts/pack.mjs": 'pack("web/a.js", "web/b.js");\n',
        },
    )
    _, awaiting = _awaiting(repo)

    naming = files_naming(repo, list(awaiting))

    assert {path: [(hit.file, hit.line) for hit in hits] for path, hits in naming.items()} == {
        "web/a.js": [("scripts/pack.mjs", 1)],
        "web/b.js": [("scripts/pack.mjs", 1)],
    }


def test_a_path_with_regex_characters_is_found_as_written(tmp_path: Path) -> None:
    # Arrange: Next.js route folders put brackets and parentheses in paths, which a regex would read
    # as a character class and a group
    dynamic, grouped = "app/[id]/page.tsx", "app/(shop)/page.tsx"
    repo = _repository(
        tmp_path / "repo",
        {
            dynamic: "export default function Page() {}\n",
            grouped: "export default function Shop() {}\n",
            "scripts/routes.mjs": f'build("{dynamic}");\nbuild("{grouped}");\n',
        },
    )

    # Act
    naming = files_naming(repo, [dynamic, grouped])

    # Assert
    assert {path: [(hit.file, hit.line) for hit in hits] for path, hits in naming.items()} == {
        dynamic: [("scripts/routes.mjs", 1)],
        grouped: [("scripts/routes.mjs", 2)],
    }


def test_a_secret_on_a_naming_line_keeps_the_named_file_unsent(tmp_path: Path) -> None:
    repo = _repository(
        tmp_path / "repo",
        {
            "web/a.js": BUNDLE,
            "web/named.js": "/*n*/" + BUNDLE,
            "scripts/build.mjs": f'writeFileSync("web/named.js", "{SECRET_MARK}");\n',
        },
    )
    index, awaiting = _awaiting(repo)
    client = ScriptedJevClient(default_noul=0.9)

    judgments = judge_generated_files(Judge(client, scanner=_MarkScanner()), index, awaiting)

    assert list(judgments.judged) == ["web/a.js"]
    assert dict(judgments.not_judged) == {"web/named.js": NOT_JUDGED_SECRET}
    assert all(SECRET_MARK not in str(state) for state, _ in client.requests)


def test_nothing_is_sent_when_no_file_awaits_a_judgment(tmp_path: Path) -> None:
    repo = _repository(tmp_path / "repo", {"app/orders.py": "def run():\n    return 1\n"})
    index, awaiting = _awaiting(repo)
    client = ScriptedJevClient()

    judgments = judge_generated_files(Judge(client), index, awaiting)

    assert (dict(judgments.judged), dict(judgments.not_judged), client.requests) == ({}, {}, [])


class _MarkScanner:
    """A host's stronger scanner, which the judge accepts by design: it finds one marked string."""

    def findings(self, text: str) -> list[str]:
        return [SECRET_MARK] if SECRET_MARK in text else []


def _scripted(question: Mapping, state: Mapping, by_file: Mapping[str, float]) -> float:
    slot = int(question["instructions"].split("`files[")[1].split("]")[0])
    return by_file[state["files"][slot]["file"]]


def test_a_one_line_bundle_naming_a_path_reaches_python_only_as_a_window(tmp_path: Path) -> None:
    # Arrange: jvn-verifier's shape, a 20 MB one-line bundle naming the path 600,000 times beside
    # 200 small files that name it once each
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "dist").mkdir()
    (repo / "src/util.js").write_text("export const util = 1;\n")
    (repo / "dist/bundle.js").write_text("var a=require('./src/util.js');a.util(1);" * 600_000 + "\n")
    for number in range(200):
        (repo / f"src/m{number}.js").write_text(
            f"import {{ util }} from './src/util.js';\nexport const m{number} = util;\n"
        )
    tracemalloc.start()

    # Act
    naming = files_naming(repo, ["src/util.js"])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Assert
    [bundle_hit] = [hit for hit in naming["src/util.js"] if hit.file == "dist/bundle.js"]
    assert len(naming["src/util.js"]) == 201
    assert peak < 10 * 2**20
    assert bundle_hit.line == 1 and "src/util.js" in bundle_hit.text
    assert len(bundle_hit.text) <= 2 * NAMING_LINE_CHARS + len("src/util.js") + 2


def test_a_sentence_ending_with_the_path_names_it(tmp_path: Path) -> None:
    # Arrange: the full stop after the path starts no file extension
    repo = _repository(
        tmp_path / "repo",
        {"web/bundle.js": BUNDLE, "README.md": "Rebuild web/bundle.js.\nSee web/bundle.json.\n"},
    )

    # Act
    naming = files_naming(repo, ["web/bundle.js"])

    # Assert
    assert [(hit.file, hit.line) for hit in naming["web/bundle.js"]] == [("README.md", 1)]


def test_a_minified_line_cut_on_both_sides_still_shows_the_path(tmp_path: Path) -> None:
    # Arrange: no whitespace or quote stands between either cut and the path
    line = "a=" + "b" * 300 + "=web/gen.js;c=" + "d" * 300 + "\n"
    repo = _repository(tmp_path / "repo", {"web/gen.js": BUNDLE, "web/min.js": line})

    # Act
    texts = _sent_naming_texts(repo, "web/gen.js")

    # Assert
    assert texts == ["web/gen.js"]


# Made-up values with a secret's shape; none was ever a credential.
OPAQUE_VALUE = "Zq8mKx2LpR7vWn4TsB9cHd3FgJ6aE1yU" * 4
EDGE_TOKENS = ("Q7xK2mZp9LwR4vTn8YsB3cHd6FgJ1aE5", "Yt5Rw2Nq8Lm3Kp7Vx4Bz9Cs6Dh1Fj0Gk")


def _sent_naming_texts(repo: Path, path: str) -> list[str]:
    """The naming text of each file naming ``path``, as the judge would send it, masked."""
    index, awaiting = _awaiting(repo)
    entries = [
        generated_file_entry(index, path, awaiting[path], (hit,))["named_by"][0]
        for hit in files_naming(repo, [path])[path]
    ]
    masked, _, _ = mask_request({"named_by": entries}, {}, SecretMasker())
    return [entry["text"] for entry in masked["named_by"]]


def _pieces(value: str, length: int) -> set[str]:
    return {value[start : start + length] for start in range(len(value) - length + 1)}


def test_a_secret_whose_key_falls_outside_the_naming_window_stays_masked(tmp_path: Path) -> None:
    # Arrange: jvn-verifier's shape; masked whole, the line hides the value, but the 200-character
    # window around the path holds the value without its key
    line = f'{{"apiKey": "{OPAQUE_VALUE}", "padding": "{"x" * 40}", "output": "web/gen.js"}}'
    repo = _repository(tmp_path / "repo", {"web/gen.js": BUNDLE, "app/build.js": line + "\n"})

    # Act
    [text] = _sent_naming_texts(repo, "web/gen.js")

    # Assert
    assert "web/gen.js" in text
    assert not any(piece in text for piece in _pieces(OPAQUE_VALUE, 6))


def test_a_token_cut_at_either_edge_of_the_naming_window_leaves_no_piece(tmp_path: Path) -> None:
    # Arrange: the window keeps NAMING_LINE_CHARS around the path, so with these fillers its left
    # edge sweeps across the head token and its right edge across the tail token, one character per
    # naming line, leaving fragments of every length shorter than a token
    head, tail = EDGE_TOKENS
    namers = {
        f"scripts/s{shift}.mjs": f'{head} {"w" * (56 + shift)} copy("web/gen.js") {"w" * (90 - shift)} {tail}'
        + "\n"
        for shift in range(31)
    }
    repo = _repository(tmp_path / "repo", {"web/gen.js": BUNDLE, **namers})

    # Act
    texts = _sent_naming_texts(repo, "web/gen.js")

    # Assert
    assert len(texts) == 31 and all("web/gen.js" in text for text in texts)
    leaked = {
        piece for text in texts for token in EDGE_TOKENS for piece in _pieces(token, 4) if piece in text
    }
    assert leaked == set()


def test_a_token_cut_where_the_search_stops_reading_leaves_no_piece(tmp_path: Path) -> None:
    # Arrange: four-byte characters make the search's 200 bytes of context fewer than
    # NAMING_LINE_CHARS characters, so the edges where the search stopped reading reach Jev; each
    # line shifts both edges one character further into a token
    head, tail = EDGE_TOKENS
    namers = {
        f"scripts/s{shift}.mjs": f'{head} {"🙂" * 40}{"w" * shift} copy("web/gen.js") {"w" * (32 - shift)}'
        + f"{'🙂' * 40} {tail}\n"
        for shift in range(1, 27)
    }
    repo = _repository(tmp_path / "repo", {"web/gen.js": BUNDLE, **namers})

    # Act
    texts = _sent_naming_texts(repo, "web/gen.js")

    # Assert
    assert len(texts) == 26 and all("web/gen.js" in text and len(text) < NAMING_LINE_CHARS for text in texts)
    leaked = {
        piece for text in texts for token in EDGE_TOKENS for piece in _pieces(token, 4) if piece in text
    }
    assert leaked == set()


# Made-up values with a JWT's and a password's shape; neither was ever a credential.
SPLIT_VALUES = (
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0LXVzZXIiLCJyb2xlIjoibm9uZSJ9.c2lnbmF0dXJlLW9ubHktYS10ZXN0",
    "p@ss!w0rd#Kq8$Lm2%Zx7^Vn4&Rt9*Wb3(Yc6)Hd1-Fg5+Jk0=",
)


@pytest.mark.parametrize("value", SPLIT_VALUES, ids=["jwt", "password-with-symbols"])
def test_a_split_value_leaves_no_piece_whatever_its_characters(tmp_path: Path, value: str) -> None:
    # Arrange: each naming line puts the 200-character cut one character further inside the value
    namers = {f"app/c{inside}.json": _line_cut_inside(value, inside) for inside in range(1, len(value))}
    repo = _repository(tmp_path / "repo", {"web/gen.js": BUNDLE, **namers})

    # Act
    texts = _sent_naming_texts(repo, "web/gen.js")

    # Assert
    assert len(texts) == len(value) - 1 and all("web/gen.js" in text for text in texts)
    assert {piece for text in texts for piece in _pieces(value, 4) if piece in text} == set()


def _line_cut_inside(value: str, inside: int) -> str:
    """A naming line whose last ``NAMING_LINE_CHARS`` characters start ``inside`` into ``value``."""
    before_pad, after_pad = '", "pad": "', '", "output": "web/gen.js"}'
    pad = "x" * (NAMING_LINE_CHARS - (len(value) - inside) - len(before_pad) - len(after_pad))
    return '{"token": "' + value + before_pad + pad + after_pad + "\n"
