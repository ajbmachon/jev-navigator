"""Every cut the cut-before-mask audit found (verifier-work/cut-before-mask-audit.md, rows 1 to 7 and
12), run through its real cut function: a value the masker finds anywhere in the whole file never
reaches a request, however the slice, window, preview or line cut falls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from git_repos import commit_files, write_files
from secret_shapes import COPY_VALUE, SECRET_SHAPES, sent_pieces

from jev_navigator.adapters.routes import DREX_INPUT_LIMITS
from jev_navigator.directives import find_code, places, shown, trace
from jev_navigator.directives.entry import _file_description, _preview
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.secrets import MASK, SecretMasker
from jev_navigator.operations import TraceLink
from jev_navigator.testing import ScriptedJevClient

SHOWN_CHECK = Check(
    name="shown",
    instructions="Does `{item}` send an order?",
    yes=Criterion("it calls post"),
    no=Criterion("it does not call post"),
)


def _index(tmp_path: Path, files: dict[str, str]) -> CodeIndex:
    write_files(tmp_path / "repo", files)
    return CodeIndex(tmp_path / "repo", list(files), fact_cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("value", SECRET_SHAPES.values(), ids=SECRET_SHAPES.keys())
def test_the_line_cut_never_splits_a_secret_out_of_its_masking(tmp_path: Path, value: str) -> None:
    # Arrange: row 1, one line per position of the 240-character cut across the value
    lines = [
        f'const cfg{pad} = {{ name: "{"x" * pad}", apiKey: "{value}", retries: 3 }};'
        for pad in range(120, 240)
    ]
    index = _index(tmp_path, {"app/config.ts": "\n".join(lines) + "\n"})

    # Act
    cut = shown.shown_slice(index.read_slice(Span("app/config.ts", 1, len(lines))), lambda _: True)

    # Assert
    assert cut is not None and shown.LINE_CUT_MARK in cut.text
    assert sent_pieces(value, {"slice": {"file": "app/config.ts", "code": cut.text}}) == set()


@pytest.mark.parametrize("value", SECRET_SHAPES.values(), ids=SECRET_SHAPES.keys())
def test_finds_first_cut_never_splits_a_secret_out_of_its_masking(tmp_path: Path, value: str) -> None:
    # Arrange: the verifier's line_cut probe, one line per position of the 240-character cut across the value
    lines = [
        f'const cfg{pad} = {{ name: "{"x" * pad}", apiKey: "{value}", retries: 3 }};'
        for pad in range(120, 240)
    ]
    index = _index(tmp_path, {"app/config.ts": "\n".join(lines) + "\n"})

    # Act
    cut = find_code.shown_for_target(
        index.read_slice(Span("app/config.ts", 1, len(lines))), "the retry setting", DREX_INPUT_LIMITS
    )

    # Assert
    assert cut is not None and shown.LINE_CUT_MARK in cut.text
    assert sent_pieces(value, {"slice": {"file": "app/config.ts", "code": cut.text}}) == set()


def test_a_candidate_preview_and_signature_hide_a_copy_whose_key_line_is_below_them(tmp_path: Path) -> None:
    # Arrange: rows 2 and 3, the key line falls below the preview's first lines
    source = (
        f'function send(order) {{ return post("{COPY_VALUE}", order); }}\n'
        + "".join(f"step{number}();\n" for number in range(10))
        + f'const API_TOKEN = "{COPY_VALUE}";\n'
    )
    index = _index(tmp_path, {"app/hooks.js": source})
    place = places.function_place(index, Span("app/hooks.js", 1, 1, name="send"), "a candidate")

    # Act
    state = find_code._candidate_state(place, find_code.SearchBudget())

    # Assert
    assert "post(" in state["preview"] and "send" in state["signature"]
    assert sent_pieces(COPY_VALUE, {"candidates": [state]}) == set()


def test_a_slice_cut_to_fit_hides_a_copy_whose_key_line_is_below_the_cut(tmp_path: Path) -> None:
    # Arrange: row 4
    source = (
        "def send(order):\n"
        + "".join(f'    post("{COPY_VALUE}", order{number})\n' for number in range(5))
        + f'    API_TOKEN = "{COPY_VALUE}"\n    return order\n'
    )
    index = _index(tmp_path, {"app/hooks.py": source})
    code = index.read_slice(Span("app/hooks.py", 1, source.count("\n"), name="send"))

    # Act
    cut = shown.shown_slice(code, lambda shown_code: shown_code.text.count("\n") < 6)

    # Assert
    assert cut is not None and "API_TOKEN" not in cut.text and "post(" in cut.text
    assert sent_pieces(COPY_VALUE, {"slice": {"file": "app/hooks.py", "code": cut.text}}) == set()


def test_a_window_hides_a_copy_whose_key_line_is_outside_it(tmp_path: Path) -> None:
    # Arrange: row 5, the hit on line 20, the key on line 35, outside lines 10 to 30
    lines = [f"x{number} = {number}" for number in range(1, 41)]
    lines[19] = f'post("{COPY_VALUE}")'
    lines[34] = f'WEBHOOK_TOKEN = "{COPY_VALUE}"'
    index = _index(tmp_path, {"app/settings.py": "\n".join(lines) + "\n"})

    # Act
    window = places.window_place(index, "app/settings.py", 20, "mentions a key").open()

    # Assert
    assert "post(" in window.text and "WEBHOOK_TOKEN" not in window.text
    assert sent_pieces(COPY_VALUE, {"slice": {"file": "app/settings.py", "code": window.text}}) == set()


def test_a_window_hides_the_lines_of_a_multi_line_secret_whose_first_line_is_outside_it(
    tmp_path: Path,
) -> None:
    # Arrange: row 5 and check 3, a key block opening on line 8, outside lines 10 to 30
    body = [f"{COPY_VALUE}{number:08d}" for number in range(6)]
    lines = [f"x{number} = {number}" for number in range(1, 41)]
    lines[7 : 7 + len(body) + 2] = ["-----BEGIN RSA PRIVATE KEY-----", *body, "-----END RSA PRIVATE KEY-----"]
    index = _index(tmp_path, {"deploy/key.txt": "\n".join(lines) + "\n"})

    # Act
    window = places.window_place(index, "deploy/key.txt", 20, "mentions a key").open()

    # Assert: the window keeps its own lines, and none of the block's
    assert window.span.start == 10 and window.text.count("\n") == window.span.end - window.span.start
    assert [line for line in body if line[:12] in window.text] == []


def test_a_function_slice_hides_a_copy_of_a_module_level_key(tmp_path: Path) -> None:
    # Arrange: row 6
    source = f'API_TOKEN = "{COPY_VALUE}"\n\n\ndef send(order):\n    return post("{COPY_VALUE}", order)\n'
    index = _index(tmp_path, {"app/client.py": source})

    # Act
    opened = places.function_place(index, Span("app/client.py", 4, 5, name="send"), "a function").open()

    # Assert
    assert "post(" in opened.text
    assert sent_pieces(COPY_VALUE, {"slice": {"file": "app/client.py", "code": opened.text}}) == set()


def test_a_trace_link_line_hides_a_copy_whose_key_is_elsewhere_in_its_file(tmp_path: Path) -> None:
    # Arrange: row 7
    source = f'API_TOKEN = "{COPY_VALUE}"\n\n\ndef send(order):\n    return post("{COPY_VALUE}", order)\n'
    index = _index(tmp_path, {"app/client.py": source})
    link = TraceLink(1, Span("app/client.py", 4, 5, name="send"), None, "calls", "post", "app/client.py", 5)

    # Act
    item = trace._link_item(index, link, own_key="")

    # Assert
    assert "post(" in item
    assert sent_pieces(COPY_VALUE, {"links": [item]}) == set()


def test_an_entry_preview_hides_a_copy_whose_key_line_is_below_it(tmp_path: Path) -> None:
    # Arrange: row 12, three preview lines, the key on line 5
    source = f'def send(order):\n    return post("{COPY_VALUE}", order)\n\n\nAPI_TOKEN = "{COPY_VALUE}"\n'
    index = _index(tmp_path, {"app/client.py": source})

    # Act
    preview = _preview(index, "app/client.py", 1)

    # Assert
    assert "post(" in preview
    assert sent_pieces(COPY_VALUE, {"options": [preview]}) == set()


def test_an_entry_file_description_hides_a_copy_whose_key_line_is_past_the_doc_lines(tmp_path: Path) -> None:
    # Arrange: row 12, the doc line quotes the value; its key sits past the 40 lines scanned for a doc
    source = f'"""Posts with {COPY_VALUE} for now."""\n' + "x = 1\n" * 45 + f'API_TOKEN = "{COPY_VALUE}"\n'
    index = _index(tmp_path, {"app/client.py": source})

    # Act
    description = _file_description(index, "app/client.py", {})

    # Assert
    assert "Posts with" in description
    assert sent_pieces(COPY_VALUE, {"options": [description]}) == set()


@dataclass(frozen=True)
class _HostMasker:
    """A host's own masker, which the index accepts by design: it hides one marked word."""

    def mask(self, text: str, path: str | None = None) -> str:
        return text.replace(HOST_MARK, MASK)

    def masked_values(self, text: str, path: str | None = None) -> list[str]:
        return [HOST_MARK] if HOST_MARK in text else []


HOST_MARK = "host-only-mark"


BUILDS = {
    "from_git": lambda root, masker: CodeIndex.from_git(root, masker=masker),
    "from_directory": lambda root, masker: CodeIndex.from_directory(root, masker=masker),
    "at_commit": lambda root, masker: CodeIndex.at_commit(root, "HEAD", masker=masker),
}


@pytest.mark.parametrize("build", BUILDS.values(), ids=BUILDS.keys())
def test_an_index_built_with_a_hosts_masker_masks_its_slices_with_it(tmp_path: Path, build) -> None:
    # Arrange
    commit_files(tmp_path / "repo", {"app/a.py": f'def send():\n    return "{HOST_MARK}"\n'})

    # Act
    index = build(tmp_path / "repo", _HostMasker())

    # Assert
    assert index.read_slice(Span("app/a.py", 1, 2)).text == f'def send():\n    return "{MASK}"'


@dataclass(frozen=True)
class _EngineShapedMasker:
    """A host masker of its own, as the Engine passes its Judge: the built-in rules behind another object."""

    def mask(self, text: str, path: str | None = None) -> str:
        return SecretMasker().mask(text, path)

    def masked_values(self, text: str, path: str | None = None) -> list[str]:
        return SecretMasker().masked_values(text, path)


@pytest.mark.parametrize("masker", [None, _EngineShapedMasker()], ids=["judge-default", "judge-own-masker"])
def test_a_copy_in_one_file_of_a_value_keyed_in_another_is_hidden_in_the_sent_request(
    tmp_path: Path, masker
) -> None:
    # Arrange: the index masks settings.py as a whole, so the request never sees the keyed value raw
    index = _index(
        tmp_path,
        {
            "app/settings.py": f'WEBHOOK_TOKEN = "{COPY_VALUE}"\n',
            "app/hooks.py": f'def send(order):\n    return post("{COPY_VALUE}", order)\n',
        },
    )
    items = [
        {"file": file, "code": index.read_slice(Span(file, 1, last)).text}
        for file, last in (("app/settings.py", 1), ("app/hooks.py", 2))
    ]
    client = ScriptedJevClient()
    judge = Judge(client) if masker is None else Judge(client, masker=masker)

    # Act
    judge.check_each(SHOWN_CHECK, items)

    # Assert
    assert [request for request in client.requests if COPY_VALUE[:12] in str(request)] == []
