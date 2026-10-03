from __future__ import annotations

from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.entry import FILE_READ_CAP, choose_initial_candidates
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.client import JEV_INPUT_BOX_CHARS
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import ScriptedJevClient

TARGET = "the function that computes how concentrated a set of probabilities is"


def _index(tmp_path: Path, files: dict[str, str]) -> CodeIndex:
    commit_files(tmp_path / "repo", files)
    return CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "fact-cache")


def _root_options(index: CodeIndex) -> dict[str, str]:
    client = ScriptedJevClient()
    choose_initial_candidates(index, Judge(client), TARGET)
    _state, questions = client.requests[0]
    return dict(next(iter(questions.values()))["criteria"])


def _option_for(options: dict[str, str], prefix: str) -> str:
    return next(text for text in options.values() if text.startswith(prefix))


LIBRARY = {
    "src/pkg/__init__.py": "",
    "src/pkg/adapters/__init__.py": "",
    "src/pkg/adapters/routes.py": "def route(name):\n    return name\n",
    "src/pkg/answers.py": (
        '"""Typed answers."""\n\ndef distribution_confidence(probabilities):\n    return 1.0\n\n'
        "class Answer:\n    def to_json(self):\n        return {}\n"
    ),
    "src/pkg/zeta.py": "def last():\n    return 1\n",
    "tests/test_answers.py": "def test_answers():\n    pass\n",
}


def test_a_directory_option_lists_its_subfolders_with_counts_and_its_files_relative_to_it(
    tmp_path: Path,
) -> None:
    options = _root_options(_index(tmp_path, LIBRARY))

    src = _option_for(options, "directory src/ ")

    assert "(5 code files)" in src
    assert "All of it is under src/pkg/." in src
    assert "Subfolders: adapters/ (2)" in src
    assert "answers.py" in src and "adapters/routes.py" in src


def test_a_directory_option_shows_the_top_level_symbols_of_its_main_files_not_empty_package_markers(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path, {**LIBRARY, "src/answers.py": LIBRARY["src/pkg/answers.py"]})
    options = _root_options(index)

    src = _option_for(options, "directory src/ ")

    assert "answers.py: distribution_confidence, Answer" in src
    assert "to_json" not in src, "methods are nested, not top-level"
    assert "__init__" not in src.split("Main files:")[1]


def test_a_file_option_shows_its_first_doc_line_and_its_symbol_names(tmp_path: Path) -> None:
    index = _index(tmp_path, {"answers.py": LIBRARY["src/pkg/answers.py"], "util/helpers.py": "x = 1\n"})
    options = _root_options(index)

    answers = _option_for(options, "file answers.py")

    assert answers == "file answers.py: Typed answers. Symbols: distribution_confidence, Answer"


def test_the_option_set_is_the_same_directories_and_files_as_before(tmp_path: Path) -> None:
    options = _root_options(_index(tmp_path, {**LIBRARY, "setup.py": "def setup():\n    pass\n"}))

    kinds_and_paths = sorted(" ".join(text.split()[:2]).rstrip(":") for text in options.values())

    assert kinds_and_paths == ["directory src/", "directory tests/", "file setup.py"]


def test_symbols_are_read_for_at_most_the_file_read_cap_per_request_shared_round_robin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folders = {f"area{number:02d}/mod.py": f"def thing_{number:02d}():\n    pass\n" for number in range(40)}
    index = _index(tmp_path, folders)
    read: list[str] = []
    original = CodeIndex.symbols_in

    def counting(self: CodeIndex, file: str):
        read.append(file)
        return original(self, file)

    monkeypatch.setattr(CodeIndex, "symbols_in", counting)

    reads_for_the_root_request: list[int] = []

    class Recording(ScriptedJevClient):
        def send(self, state, questions):
            reads_for_the_root_request.append(len(read))
            return super().send(state, questions)

    client = Recording()
    choose_initial_candidates(index, Judge(client), TARGET)
    options = dict(next(iter(client.requests[0][1].values()))["criteria"])

    assert reads_for_the_root_request[0] == FILE_READ_CAP
    with_symbols = [text for text in options.values() if "thing_" in text]
    assert len(with_symbols) == FILE_READ_CAP
    assert "thing_00" in _option_for(options, "directory area00/")
    assert "thing_39" not in _option_for(options, "directory area39/")


def test_a_directory_option_shows_one_main_file_for_every_subfolder_not_only_the_alphabetical_first(
    tmp_path: Path,
) -> None:
    files = {
        f"src/{folder}/{module}.py": f"def {name}():\n    pass\n"
        for folder, module, name in (
            ("adapters", "routes", "route_name"),
            ("index", "scan", "scan_tree"),
            ("judgments", "answers", "distribution_confidence"),
        )
    }
    options = _root_options(_index(tmp_path, {**files, "tests/test_x.py": "def test_x():\n    pass\n"}))

    src = _option_for(options, "directory src/ ")

    for name in ("route_name", "scan_tree", "distribution_confidence"):
        assert name in src


def test_all_options_of_a_request_fit_the_character_box_however_many_there_are(tmp_path: Path) -> None:
    crowded = {
        f"area{number:03d}/{'module' * 6}{file}.py": f"def {'symbol' * 5}{file}():\n    pass\n"
        for number in range(190)
        for file in range(4)
    }
    options = _root_options(_index(tmp_path, crowded))

    assert len(options) == 190
    assert sum(len(text) for text in options.values()) <= JEV_INPUT_BOX_CHARS


def test_anonymous_functions_and_calls_are_not_listed_as_symbols(
    tmp_path: Path,
) -> None:
    source = (
        "describe('x', () => {\n  test('y', () => { return 1 })\n})\n"
        "export function admit() { return 1 }\n[1].map(() => 2)\n"
    )
    options = _root_options(_index(tmp_path, {"checks.ts": source, "other/readme.py": "x = 1\n"}))

    checks = _option_for(options, "file checks.ts")

    assert checks == "file checks.ts: Symbols: admit"


TYPESCRIPT_WITH_LATE_DOC = (
    'import type { A } from "./a.js";\n'
    'import type {\n  B,\n  C,\n} from "./b.js";\n\n'
    "/**\n * The desired-state seam: what the hub says should be running.\n *\n * More detail.\n */\n"
    "export function project() { return 1 }\n"
)


def test_a_doc_comment_after_the_import_block_is_the_files_doc_line(tmp_path: Path) -> None:
    index = _index(tmp_path, {"seam.ts": TYPESCRIPT_WITH_LATE_DOC, "other/readme.py": "x = 1\n"})

    seam = _option_for(_root_options(index), "file seam.ts")

    assert (
        seam == "file seam.ts: The desired-state seam: what the hub says should be running. Symbols: project"
    )


def test_a_file_of_only_types_still_shows_its_doc_line(tmp_path: Path) -> None:
    types_only = "// Admission catalog contract.\nexport interface Catalog {\n  find(): void\n}\n"
    index = _index(tmp_path, {"catalog.ts": types_only, "other/readme.py": "x = 1\n"})

    catalog = _option_for(_root_options(index), "file catalog.ts")

    assert catalog == "file catalog.ts: Admission catalog contract."


@pytest.mark.parametrize("directive", ["// eslint-disable-next-line x", "# noqa: E501", "// @ts-nocheck"])
def test_tool_directives_are_not_taken_for_a_doc_line(tmp_path: Path, directive: str) -> None:
    source = f"{directive}\nexport function real() {{ return 1 }}\n"
    index = _index(tmp_path, {"code.ts": source, "other/readme.py": "x = 1\n"})

    code = _option_for(_root_options(index), "file code.ts")

    assert code == "file code.ts: Symbols: real"
