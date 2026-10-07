"""Native owner regression for filtered original groups and canonical source aliases.

Run with the same pinned owners as paid_pack.py, using unittest (no paid client).
"""

import copy
import subprocess
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from enginepy.workflows.document_analysis import evidence_pack as owner
from enginepy.workflows.document_analysis.code_relations import CodeRelations
from jev_navigator.judgments.profiles import ROLES_V2
from paid_pack import (
    answered_observations,
    filter_observations,
    native_scope,
    pack_lab,
    pack_native,
    score_windows,
)
from paid_shape import prepare

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import items_to_judge, list_units, read_ranges
from jev_navigator.judgments.questions import content_hash


class OriginalGroupPackingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        sources = {
            "floor.py": "def floor():\n    return 1\n",
            "config.py": "FIRST = 1\n\ndef between():\n    return 2\n\nLAST = 3\n",
            "plan.py": "def planned():\n    return " + repr("v" * 31000) + "\n",
            "unasked.py": "def unasked():\n    return 5\n",
            "enginepy/protocol/generated/remediation_contract.py": "def remediation():\n    return 6\n",
            "enginepy/protocol/generated/event_contract.py": "def event():\n    return 7\n",
            "private.py": "def private():\n    return 8\n",
            "untracked.py": "def untracked():\n    return 9\n",
        }
        for file, source in sources.items():
            (root / file).parent.mkdir(parents=True, exist_ok=True)
            (root / file).write_text(source)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "rm", "--cached", "-q", "untracked.py"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Pack test",
                "-c",
                "user.email=pack@test",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        self.index = CodeIndex(root, sources)
        self.addCleanup(self.index.close)
        self.relations = CodeRelations(str(root), ({}, {}))
        self.row = {
            "repository": str(root),
            "withheld": [],
            "claim": {
                "id": "fixture",
                "statement": "Check the behavior",
                "evidence": [{"file": "floor.py", "line": 2}],
            },
        }
        listing = list_units(self.index, ["config.py", "plan.py", "unasked.py"], box_chars=70000)
        top = next(unit for unit in listing.units if unit.path == "config.py" and unit.kind == "top_level")
        plan = next(unit for unit in listing.units if unit.path == "plan.py")
        unasked = next(unit for unit in listing.units if unit.path == "unasked.py")
        self.candidates, entries = [], []
        for unit, flags in (
            (top, {"plan": False, "ranking": True}),
            (plan, {"plan": True, "ranking": False}),
            (unasked, {"plan": True, "ranking": True}),
        ):
            items = [
                {**asdict(item), "code": read_ranges(self.index, item.file, item.ranges)}
                for item in items_to_judge(unit)
            ]
            self.candidates.append(
                {
                    "unit": asdict(replace(unit, id=f"canonical:{unit.path}")),
                    "items": items,
                    "source_flags": flags,
                }
            )
            if unit is not unasked:
                entries.extend(
                    ({"file": item["file"], "code": item["code"]}, next(iter(items_to_judge(unit))))
                    for item in items
                )
        requests, membership, refusals = prepare(
            ROLES_V2.questions("p0"),
            entries,
            {"targets": {"p0": "Check the behavior"}},
        )
        self.assertFalse(refusals)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        digest = content_hash({"state": request["state"], "questions": request["questions"]})
        members = []
        for member in membership[0]:
            candidate = next(c for c in self.candidates if c["unit"]["path"] == member["file"])
            members.append(
                {**member, "unit_id": candidate["unit"]["id"], "source_flags": candidate["source_flags"]}
            )
        self.prepared = [{"ordinal": 1, "request": request, "request_sha256": digest, "members": members}]
        self.responses = [
            {
                "ordinal": 1,
                "request_sha256": digest,
                "usd": "0.00042",
                "seconds": 1.25,
                "response": {
                    "model": "jev-1.13.0",
                    "usage": {"input_tokens": 456},
                    "answers": {
                        qid: {"type": "noul", "noul": 0.9 if qid.endswith("#0") else 0.6}
                        for qid in request["questions"]
                    },
                },
            }
        ]

    def test_original_answers_filter_before_actual_lab_and_native_owner_packing(self):
        original = copy.deepcopy(self.prepared)
        observations = answered_observations(self.prepared, self.responses, self.candidates)
        self.assertEqual(len(observations), 2)  # The unasked source receives no probability.
        labels = [
            {"file": "config.py", "first_line": 1, "last_line": 1},
            {"file": "config.py", "first_line": 3, "last_line": 3},
            {"file": "plan.py", "first_line": 2, "last_line": 2},
            {"file": "unasked.py", "first_line": 2, "last_line": 2},
        ]
        for view, expected in (
            ("union", [True, False, True, False]),
            ("ranking-only", [True, False, False, False]),
            ("plan-only", [False, False, True, False]),
        ):
            selected = filter_observations(observations, view, 4)
            for room in (7200, 20000, 36000):
                native = pack_native(self.row, self.relations, self.index, selected, room)
                lab = pack_lab({"floor_windows": []}, selected, room)
                room_expected = [*expected]
                if room == 7200:
                    room_expected[2] = False  # This whole source unit exceeds the legacy room.
                for windows in (native["consumer_windows"], lab["windows"]):
                    self.assertEqual([r["delivered"] for r in score_windows(windows, labels)], room_expected)
                self.assertLessEqual(native["packet_chars"], room * 4)
                self.assertEqual(native["provider_calls"], 0)
                self.assertAlmostEqual(native["original_usd"], 0.00042)
                self.assertNotIn("def unasked", native["packet"])
        self.assertEqual(self.prepared, original)

    def test_identity_and_original_question_companions_are_required(self):
        changed = copy.deepcopy(self.prepared)
        changed[0]["request"]["state"]["items"].reverse()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            answered_observations(changed, self.responses, self.candidates)
        changed_response = copy.deepcopy(self.responses)
        changed_response[0]["response"]["answers"].pop(next(iter(changed_response[0]["response"]["answers"])))
        with self.assertRaisesRegex(ValueError, "incomplete original answers"):
            answered_observations(self.prepared, changed_response, self.candidates)

    def test_physical_request_checkpoint_and_native_no_anchor(self):
        prepared, responses = copy.deepcopy(self.prepared), copy.deepcopy(self.responses)
        prepared[0]["ordinal"] = responses[0]["ordinal"] = 5
        observations = answered_observations(prepared, responses, self.candidates)
        self.assertEqual(filter_observations(observations, "union", 4), ())
        self.assertEqual(len(filter_observations(observations, "union", 8)), 2)
        row = {**self.row, "claim": {**self.row["claim"], "evidence": []}}
        result = pack_native(row, self.relations, self.index, observations, 20000)
        self.assertTrue(result["no_anchor"])
        self.assertEqual(result["consumer_windows"], [])

    def test_explicit_union_scope_renders_generated_brothers_and_reports_ineligible_sources(self):
        generated = "enginepy/protocol/generated/remediation_contract.py"
        brother = "enginepy/protocol/generated/event_contract.py"
        files = [generated, brother, "private.py", "untracked.py"]
        row = {**self.row, "withheld": [str(Path(self.row["repository"]) / "private.py")]}
        candidates, entries = [], []
        for unit in list_units(self.index, files, box_chars=70000).units:
            items = [
                {**asdict(item), "code": read_ranges(self.index, item.file, item.ranges)}
                for item in items_to_judge(unit)
            ]
            candidates.append({"unit": asdict(unit), "items": items, "source_flags": ["ranking"]})
            entries.extend(
                ({"file": item["file"], "code": item["code"]}, place)
                for item, place in zip(items, items_to_judge(unit), strict=True)
            )
        requests, members, refusals = prepare(
            ROLES_V2.questions("p0"), entries, {"targets": {"p0": "Check the behavior"}}
        )
        self.assertFalse(refusals)
        request = requests[0]
        digest = content_hash({"state": request["state"], "questions": request["questions"]})
        membership = [
            {**member, "unit_id": member["id"], "source_flags": ["ranking"]} for member in members[0]
        ]
        prepared = [{"ordinal": 1, "request": request, "members": membership, "request_sha256": digest}]
        responses = [
            {
                **self.responses[0],
                "request_sha256": digest,
                "response": {
                    "model": "jev-1.13.0",
                    "answers": {qid: {"type": "noul", "noul": 0.9} for qid in request["questions"]},
                },
            }
        ]
        observations = answered_observations(prepared, responses, candidates)
        self.assertFalse(self.relations.readable(generated))  # The real native inventory rejects this class.
        normal_index = owner.pack_index(self.relations, row["repository"])
        self.addCleanup(normal_index.close)
        with self.assertRaisesRegex(ValueError, "outside the index scope"):
            normal_index.lines(generated)  # Original live failure at the actual index boundary.
        relations, index, scope = native_scope(row, files)
        self.addCleanup(index.close)
        self.assertEqual(set(scope["added_files"]), {generated, brother})
        self.assertEqual({item["file"] for item in scope["excluded_files"]}, {"private.py", "untracked.py"})
        result = pack_native(row, relations, index, observations, 20000)
        self.assertEqual(result["excluded_observation_count"], 2)
        self.assertEqual(
            {item["file"] for item in result["excluded_observations"]}, {"private.py", "untracked.py"}
        )
        self.assertEqual(
            {w["file"] for w in result["consumer_windows"] if w["file"] in files}, {generated, brother}
        )
        self.assertEqual({w["file"] for w in result["asked_windows"]}, set(files))


if __name__ == "__main__":
    unittest.main()
