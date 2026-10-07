"""Trace translation and literal output attribution, independent of the local discovery data."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

from jev_navigator.index.code_index import CodeIndex

spec = importlib.util.spec_from_file_location(
    "pack_case3_replay", Path(__file__).parents[1] / "measurements/pack_case3/replay.py"
)
replay = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replay
spec.loader.exec_module(replay)


class CharacterCounter:
    def encode(self, text, **_):
        return text


def test_multiline_shell_and_disjoint_sed_ranges_emit_the_requested_lines(tmp_path):
    (tmp_path / "one.txt").write_text("one\ntwo\nthree\nfour\nfive\nsix\n")
    (tmp_path / "two.txt").write_text("a\nb\nc\nd\ne\nf\n")
    record = {
        "commands": [
            {
                "step": 1,
                "started_line": 1,
                "receipt_line": 2,
                "output": str(tmp_path),
                "command": ("nl -ba one.txt | sed -n '1,2p;5,6p'\nnl -ba two.txt | sed -n '3,4p'"),
            }
        ]
    }
    parser = SimpleNamespace(shell=lambda text: text)
    with CodeIndex.from_directory(tmp_path) as index:
        moves, rejected = replay.translate(record, index, parser)
        measured, printed = replay.replay(index, moves, "", False, 4000, io.StringIO(), CharacterCounter())
    assert not rejected
    assert set(printed) == {
        ("one.txt", 1),
        ("one.txt", 2),
        ("one.txt", 5),
        ("one.txt", 6),
        ("two.txt", 3),
        ("two.txt", 4),
    }
    assert measured["calls"] == 1


def test_fragments_of_two_operations_never_share_a_reassembly_buffer(tmp_path):
    (tmp_path / "one.txt").write_text("🙂" * 2000)
    (tmp_path / "two.txt").write_text("漢" * 2000)
    moves = [
        replay.Move(replay.Operation("show", file=file), 1, 1, 2, "recorded read")
        for file in ("one.txt", "two.txt")
    ]
    with CodeIndex.from_directory(tmp_path) as index:
        measured, printed = replay.replay(index, moves, "", False, 2400, io.StringIO(), CharacterCounter())
    assert measured["calls"] > 1
    assert set(printed) == {("one.txt", 1), ("two.txt", 1)}
    assert not measured["errors"]
