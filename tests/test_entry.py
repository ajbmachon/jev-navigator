from __future__ import annotations

import json
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.entry import FILE_READ_CAP, MAX_OPTIONS, choose_initial_candidates
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.client import JEV_INPUT_BOX_CHARS
from jev_navigator.judgments.judge import Judge, request_exceeds_input_budget
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


def test_files_are_read_one_per_option_first_then_round_robin_up_to_the_read_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folders = {f"area{number:02d}/mod.py": f"def thing_{number:02d}():\n    pass\n" for number in range(40)}
    index = _index(tmp_path, folders)
    read: list[str] = []
    original = CodeIndex.top_level_symbols

    def counting(self: CodeIndex, file: str):
        read.append(file)
        return original(self, file)

    monkeypatch.setattr(CodeIndex, "top_level_symbols", counting)

    reads_for_the_root_request: list[int] = []

    class Recording(ScriptedJevClient):
        def send(self, state, questions):
            reads_for_the_root_request.append(len(read))
            return super().send(state, questions)

    client = Recording()
    choose_initial_candidates(index, Judge(client), TARGET)
    options = dict(next(iter(client.requests[0][1].values()))["criteria"])

    assert reads_for_the_root_request[0] == max(FILE_READ_CAP, len(options)) == 40
    assert all("thing_" in text for text in options.values()), "every option gets its first main file read"


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


SECRET = "Zq9xK2mP7vL4nB8wR3tY6uH1"


def _sent_and_recorded(index: CodeIndex, target: str = TARGET) -> tuple[ScriptedJevClient, dict]:
    client = ScriptedJevClient()
    selection = choose_initial_candidates(index, Judge(client), target)
    return client, selection.to_json()


def _assert_secret_nowhere(client: ScriptedJevClient, receipt: dict) -> None:
    assert client.requests, "the scenario must send a request"
    assert SECRET not in json.dumps(client.requests)
    assert SECRET not in json.dumps(receipt)
    assert SECRET[:12] not in json.dumps(client.requests)
    assert SECRET[:12] not in json.dumps(receipt)


@pytest.mark.parametrize(
    "first_lines",
    [
        f'# api_token = "{SECRET}"\n',
        f"// password = '{SECRET}'\n",
        f'/* secret = "{SECRET}" */\n',
        f'"""api_token = "{SECRET}""""\n',
        f'# {"n" * 90} api_token = "{SECRET}"\n',
        f'"""\napi_key = "{SECRET}"\n"""\n',
    ],
    ids=["hash", "slashes", "block", "docstring", "cut-inside-the-token", "docstring-next-line"],
)
def test_a_commented_credential_never_reaches_the_request_or_the_receipt(
    tmp_path: Path, first_lines: str
) -> None:
    index = _index(
        tmp_path, {"config.py": f"{first_lines}def load():\n    return 1\n", "other/readme.py": "x = 1\n"}
    )

    client, receipt = _sent_and_recorded(index)

    _assert_secret_nowhere(client, receipt)


def test_a_credential_cut_by_the_span_preview_never_reaches_the_request_or_the_receipt(
    tmp_path: Path,
) -> None:
    source = (
        "def load():\n"
        f'    padding = "{"p" * 300}"\n'
        f'    api_token = "{SECRET}"\n'
        "\n\ndef other():\n    return 2\n"
    )
    index = _index(tmp_path, {"config.py": source})

    client, receipt = _sent_and_recorded(index)

    _assert_secret_nowhere(client, receipt)


def test_the_closing_quote_of_a_comment_is_kept_and_only_a_matching_closer_is_removed(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path,
        {
            "quoted.py": """# note: 'a' and "b"\ndef f():\n    return 1\n""",
            "block.ts": "/* Header comment */\nexport function g() { return 1 }\n",
            "triple.py": '"""Say "hi" now"""\n\ndef h():\n    return 1\n',
            "other/readme.py": "x = 1\n",
        },
    )

    options = _root_options(index)

    assert _option_for(options, "file quoted.py").startswith("""file quoted.py: note: 'a' and "b" """)
    assert _option_for(options, "file block.ts").startswith("file block.ts: Header comment Symbols")
    assert _option_for(options, "file triple.py").startswith('file triple.py: Say "hi" now Symbols')


def test_a_docstring_that_starts_on_the_line_after_the_quotes_is_the_doc_line(tmp_path: Path) -> None:
    source = '"""\nDistribution confidence for choice answers.\n"""\n\ndef f():\n    return 1\n'
    index = _index(tmp_path, {"answers.py": source, "other/readme.py": "x = 1\n"})

    answers = _option_for(_root_options(index), "file answers.py")

    assert answers == "file answers.py: Distribution confidence for choice answers. Symbols: f"


def test_every_option_of_a_crowded_request_still_gets_symbols_from_its_main_file(tmp_path: Path) -> None:
    folders = {f"area{number:02d}/mod.py": f"def thing_{number:02d}():\n    pass\n" for number in range(40)}
    index = _index(tmp_path, folders)

    options = _root_options(index)

    assert len(options) == 40
    assert all("thing_" in text for text in options.values())


def test_the_options_of_a_request_shrink_by_the_options_shown_not_by_every_entry(tmp_path: Path) -> None:
    doc = "d" * 110
    files = {
        f"mod{number:03d}.py": f"# {doc}\ndef function_{number:03d}():\n    pass\n" for number in range(600)
    }
    index = _index(tmp_path, files)
    client = ScriptedJevClient()

    choose_initial_candidates(index, Judge(client), TARGET)

    descriptions = [
        text
        for _, questions in client.requests
        for question in questions.values()
        for text in question["criteria"].values()
        if text.startswith("file mod")
    ]
    assert descriptions, "the 200-option request must be sent"
    assert max(len(text) for text in descriptions) > 140, "600 entries must not shrink the 200 shown"


def test_a_long_target_leaves_the_request_inside_the_character_box(tmp_path: Path) -> None:
    files = {
        f"area{number:03d}/{'module' * 6}{file}.py": f"def {'symbol' * 5}{file}():\n    pass\n"
        for number in range(150)
        for file in range(8)
    }
    index = _index(tmp_path, files)
    client = ScriptedJevClient()

    choose_initial_candidates(index, Judge(client), "t" * 20_000)

    assert client.requests
    for state, questions in client.requests:
        assert not request_exceeds_input_budget(state, questions)


def test_a_level_wider_than_one_request_reads_only_the_files_of_the_options_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: 450 folders at the root, more than two requests of options
    folders = {f"area{number:03d}/mod.py": f"def thing_{number:03d}():\n    pass\n" for number in range(450)}
    index = _index(tmp_path, folders)
    parsed: list[int] = []
    real = CodeIndex.functions_in_files

    def counted(self: CodeIndex, files):
        parsed.append(len(files))
        return real(self, files)

    monkeypatch.setattr(CodeIndex, "functions_in_files", counted)

    # Act
    choose_initial_candidates(index, Judge(ScriptedJevClient()), TARGET)

    # Assert: the group request reads nothing; the request that shows options reads at most theirs
    assert max(parsed) <= MAX_OPTIONS
    assert sum(parsed) <= MAX_OPTIONS + FILE_READ_CAP


def test_every_option_gets_its_first_main_file_read_before_any_option_gets_a_second(tmp_path: Path) -> None:
    # Arrange: 20 options, each with three subfolders, so each has three main files; the budget is 30
    files = {
        f"area{number:02d}/{folder}/mod.py": f"def {folder}_{number:02d}():\n    pass\n"
        for number in range(20)
        for folder in ("alpha", "beta", "gamma")
    }

    # Act
    options = _root_options(_index(tmp_path, files))

    # Assert
    assert len(options) == 20
    assert all("alpha_" in text for text in options.values())
    assert sum("beta_" in text for text in options.values()) == FILE_READ_CAP - 20


def test_a_package_marker_with_code_is_never_a_main_file(tmp_path: Path) -> None:
    index = _index(tmp_path, {**LIBRARY, "src/pkg/__init__.py": "def package_marker():\n    pass\n"})

    src = _option_for(_root_options(index), "directory src/ ")

    assert "package_marker" not in src
    assert "__init__" not in src.split("Main files:")[1]


def test_a_secret_shaped_file_name_is_masked_in_the_request_and_in_the_descriptions_recorded(
    tmp_path: Path,
) -> None:
    # The receipt still names the selected file by its path: it is local evidence, never sent.
    name = "AKIAIOSFODNN7EXAMPLE"
    index = _index(tmp_path, {f"keys/{name}.py": "def load():\n    return 1\n", "app.py": "x = 1\n"})

    client, receipt = _sent_and_recorded(index)

    descriptions = [
        option["description"] for decision in receipt["decisions"] for option in decision["options"]
    ]
    assert client.requests
    assert name not in json.dumps(client.requests)
    assert any("[MASKED]" in text for text in descriptions)
    assert not any(name in text for text in descriptions)


def test_options_with_non_ascii_docs_and_names_keep_every_request_inside_the_box(tmp_path: Path) -> None:
    # Arrange: a non-ASCII character costs its whole escape in the request, six characters
    doc = "计算一组概率的集中程度并返回置信度" * 8
    files = {
        f"mod{number:03d}.py": f'"""{doc}"""\ndef 函数_{number:03d}():\n    pass\n' for number in range(200)
    }
    client = ScriptedJevClient()

    # Act
    choose_initial_candidates(_index(tmp_path, files), Judge(client), TARGET)

    # Assert
    assert client.requests
    for state, questions in client.requests:
        assert not request_exceeds_input_budget(state, questions)
