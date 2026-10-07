"""Original citation and persisted feature-resume contracts of the measurement collector."""

import json
import sqlite3
import subprocess

import collect
import numpy as np
import pytest

from jev_navigator.judgments.questions import content_hash


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ({"file": "source.py", "line": 2}, [("source.py", 2)]),
        ({"file": "source.py", "line": 2, "first_line": 5}, [("source.py", 5)]),
        ({"file": "source.py", "line": 2, "start": 5}, [("source.py", 5)]),
        ({"file": "source.py", "line": 2, "lines": [5, 6]}, [("source.py", 5)]),
        ({"file": "source.py"}, [("source.py", None)]),
    ],
)
def test_original_evidence_anchor_precedence(evidence, expected):
    row = {"claim": {"statement": "The handler drops requests.", "evidence": [evidence]}}
    assert collect.input_anchors(row) == expected


@pytest.fixture
def measurement(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    first = "def first():\n    return 1"
    second = "def second():\n    return 2"
    (root / "source.py").write_text(first + "\n\n" + second + "\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "Fixture source",
        ],
        check=True,
    )
    db = sqlite3.connect(":memory:")
    db.executescript(
        "create table documents(root text,id text,path text,code text,binding text);"
        "create table candidates(case_id text,id text,position integer,request_number integer);"
    )
    ids = []
    for position, code in enumerate((first, second)):
        key = content_hash({"file": "source.py", "code": code})
        ids.append(key)
        db.execute("insert into documents values (?,?,?,?,null)", (str(root), key, "source.py", code))
        for cid in ("trial:one", "trial:two"):
            db.execute("insert into candidates values (?,?,?,0)", (cid, key, position))
    cases = [{"case_id": cid, "set_name": "fixture", "labels": []} for cid in ("trial:one", "trial:two")]
    inputs = {
        case["case_id"]: {
            "repository": str(root),
            "claim": {
                "statement": "The handler drops requests.",
                "evidence": [{"file": "source.py", "line": 2}],
            },
        }
        for case in cases
    }
    out = tmp_path / "features"
    out.mkdir()
    # Resource floors protect large lab jobs, independently of citation and cache behavior.
    monkeypatch.setattr(collect, "resources", lambda _: {})
    yield db, cases, inputs, out, ids
    db.close()


@pytest.mark.parametrize("resume_boundary", ["complete_root", "individual_case"])
def test_changed_citation_rebuilds_features_and_current_receipts_resume(measurement, resume_boundary):
    db, cases, inputs, out, ids = measurement
    initial = collect.build_features(db, cases, inputs, out)
    assert len(initial) == 2
    receipt = json.loads((out / "trial_one.json").read_text())
    assert receipt["anchors"] == [["source.py", 2]]
    assert set(receipt["seeds"]) == {receipt["identities"][ids[0]]}
    with np.load(out / "trial_one.npz") as features:
        first_walk = features["features"][:, 1].copy()
    assert first_walk[0] > first_walk[1]

    inputs["trial:one"]["claim"]["evidence"][0]["line"] = 5
    if resume_boundary == "individual_case":
        # An incomplete sibling forces the root build to reach individual cache checks.
        (out / "trial_two.npz").unlink()
        expected = {"trial:one", "trial:two"}
    else:
        sibling_receipt = (out / "trial_two.json").read_bytes()
        expected = {"trial:one"}
    rebuilt = collect.build_features(db, cases, inputs, out)
    assert {item["case"] for item in rebuilt} == expected
    receipt = json.loads((out / "trial_one.json").read_text())
    assert receipt["anchors"] == [["source.py", 5]]
    assert set(receipt["seeds"]) == {receipt["identities"][ids[1]]}
    with np.load(out / "trial_one.npz") as features:
        second_walk = features["features"][:, 1]
        assert second_walk[1] > second_walk[0]
    if resume_boundary == "complete_root":
        assert (out / "trial_two.json").read_bytes() == sibling_receipt

    # JSON list serialization and anchor order/duplicates do not invalidate equivalent seeds.
    inputs["trial:one"]["claim"]["evidence"] *= 2
    saved = {path.name: path.read_bytes() for path in out.glob("trial_*")}
    assert collect.build_features(db, cases, inputs, out) == []
    assert {path.name: path.read_bytes() for path in out.glob("trial_*")} == saved
