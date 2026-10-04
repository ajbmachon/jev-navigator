from __future__ import annotations

import json
from pathlib import Path

from git_repos import commit_files

from jev_navigator.cli import create_evidence_pack
from jev_navigator.directives.find_code import SearchBudget
from jev_navigator.testing import ScriptedJevClient

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "evidence-pack"


def _shape(value: object) -> object:
    """The keys a manifest has, nested; a list is described by its first entry."""
    if isinstance(value, dict):
        return {key: _shape(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return type(value).__name__


def _same_shape(written: object, example: object) -> bool:
    if isinstance(written, dict) and isinstance(example, dict):
        return written.keys() == example.keys() and all(_same_shape(written[k], example[k]) for k in written)
    if isinstance(written, list) and isinstance(example, list):
        return not written or not example or _same_shape(written[0], example[0])
    return True


def test_the_public_example_has_the_manifest_and_report_lines_the_pack_command_writes(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "orders"
    commit_files(
        repository,
        {
            "app/orders.py": "from .policy import admit\n\ndef handle(items):\n    return admit(items)\n",
            "app/policy.py": "def admit(items):\n    return len(items) <= 3\n",
        },
    )
    client = ScriptedJevClient(
        nouls=lambda question_id, question, state: (
            0.96 if "len(items) <= 3" in state["slice"]["code"] else 0.04
        ),
        choices={"open_first": {"0": 1.0}},
    )

    written = create_evidence_pack(
        repository,
        ("app/",),
        "the check that limits how many items an order may have",
        ("app/orders.py:4",),
        tmp_path / "pack",
        SearchBudget(max_depth=2, max_steps=3, max_calls=3, beam_width=1),
        client,
        fact_cache_dir=tmp_path / "facts",
    )

    example = json.loads((EXAMPLE / "manifest.json").read_text())
    assert _same_shape(json.loads(json.dumps(written)), example), _shape(example)
    report_lines = {line.split(":")[0] for line in (tmp_path / "pack" / "report.md").read_text().splitlines()}
    example_lines = {line.split(":")[0] for line in (EXAMPLE / "report.md").read_text().splitlines()}
    assert {line for line in report_lines if line.startswith("- ")} <= example_lines
