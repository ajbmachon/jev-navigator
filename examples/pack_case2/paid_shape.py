"""Free rehearsal on the pinned profile owner, with actual ranked trial bodies."""

import itertools
import json
import sys
from pathlib import Path

from jev_navigator.judgments.profiles import ROLES_V2
from offline_mask import OfflineMasker
from request_batches import judge_in_order

from jev_navigator.index.units import Item
from jev_navigator.judgments.answers import JevResponse, NoulAnswer
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import content_hash


class Capture:
    model = "jev-latest"

    def __init__(self):
        self.requests = []

    def ask(self, state, questions):
        self.requests.append({"model": self.model, "state": state, "questions": questions})
        # These neutral answers rehearse shape only. They never score delivery.
        return JevResponse({name: NoulAnswer(0.5) for name in questions}, self.model)


def entries(path):
    with path.open() as stream:
        for line in stream:
            for item in json.loads(line)["items"]:
                yield (
                    {"file": item["file"], "code": item["code"]},
                    Item(item["id"], item["file"], tuple(map(tuple, item["ranges"]))),
                )


def main():
    source, out = (Path(arg).expanduser() for arg in sys.argv[1:3])
    records = []
    for folder in sorted((out / "cases").iterdir()):
        if not (folder / "ranked-candidates.jsonl").exists():
            continue
        context = json.loads((source / "cases" / folder.name / "planner-context.json").read_text())
        capture = Capture()
        refusals = []
        judge = Judge(capture, items_per_request=16, masker=OfflineMasker())
        for _ in judge_in_order(
            judge,
            ROLES_V2.questions("p0"),
            itertools.islice(entries(folder / "ranked-candidates.jsonl"), 128),
            {"targets": {"p0": context["finding"]}},
            refusals,
        ):
            pass
        with (folder / "prepared-requests.jsonl").open("w") as stream:
            for request in capture.requests[:8]:
                stream.write(json.dumps(request, ensure_ascii=False) + "\n")
        records.append(
            {
                "folder": folder.name,
                "requests": len(capture.requests[:8]),
                "items": [len(r["state"]["items"]) for r in capture.requests[:8]],
                "bytes": [len(json.dumps(r).encode()) for r in capture.requests[:8]],
                "refusals": [str(refusal) for refusal in refusals],
            }
        )
        if not (out / "guard-candidate.json").exists():
            full = next((r for r in capture.requests if len(r["state"]["items"]) == 16), None)
            if full:
                candidate = {
                    "case_id": folder.name,
                    "revision_id": "case2-paid-ranked-six-role-16-v1",
                    "request": full,
                    "intended_uses": {
                        name: "Retain the raw role probability. Code ranks by the maximum "
                        "of decide, guard, value, effect and satisfied, and retains "
                        "the best unit per role. Delegates selects forwarding only. "
                        "This is candidate relevance, not a finding verdict."
                        for name in full["questions"]
                    },
                }
                (out / "guard-candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
                (out / "guard-candidate-hash.txt").write_text(content_hash(full) + "\n")
    (out / "request-shape-summary.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
