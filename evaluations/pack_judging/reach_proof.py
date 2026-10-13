"""Prove unchanged sixteen-item dispatch and source delivery at each breadth prefix."""

import asyncio

from paid_run import scopes

from evaluations.pack_judging.frozen import load
from evaluations.pack_judging.packing import native_pack, native_rooms
from evaluations.pack_judging.prepare import write
from evaluations.pack_judging.reach import DESIGN, OUT, PREFIXES
from evaluations.pack_judging.repack import offline_owner
from jev_navigator.judgments.answers import JevResponse, NoulAnswer
from jev_navigator.judgments.judge import Judge


async def main():
    owner = offline_owner()
    full = load(OUT / "cases/P1.json")
    relations, index = scopes(full["context"], {})
    proofs = []
    for prefix in PREFIXES:
        case = {**full, "groups": full["groups"][:prefix]}
        group = next(g for g in reversed(case["groups"]) if g["eligible"])
        submitted = []

        class Port:
            model = "independent-integration-fixture"

            async def ask(self, state, questions, submitted=submitted):
                submitted.append({"state": state, "questions": questions})
                return JevResponse(
                    {qid: NoulAnswer(0.99 if qid.endswith("#15") else 0.01) for qid in questions}, self.model
                )

        judge = Judge(Port(), masker=None, max_calls=1)
        pack, result = await native_pack(
            relations,
            index,
            case,
            full["bindings"][: prefix * 16],
            judge,
            design=DESIGN,
            requests={group["ordinal"]: group["request"]},
        )
        assert result.failure is None and judge.calls == 1
        assert submitted == [{k: group["request"][k] for k in ("state", "questions")}]
        assert all(len(answers) == 16 for answers in result.judged.values())
        winner = group["places"]["15"]
        assert result.judged["p0"][-1].place.id == winner["id"]
        destination = OUT / "proof" / str(prefix)
        destination.mkdir(parents=True, exist_ok=True)
        rooms = native_rooms(pack, relations, case["context"], case["labels"], destination)
        literal = owner.reader((destination / "native-36000.txt").read_text())
        assert (winner["file"], winner["ranges"][0][0]) in literal, (prefix, winner)
        assert all(room["within_room"] for room in rooms.values())
        proofs.append(
            {
                "prefix": prefix,
                "fixture_calls": judge.calls,
                "provider_calls": 0,
                "sixteen_items_unchanged": True,
                "winner_delivered": True,
                "winner": winner,
            }
        )
    write(OUT / "engine-proof.json", {"status": "PASS", "provider_calls": 0, "receipts": proofs})
    print("Engine proof: PASS, all prefixes, zero provider calls")


if __name__ == "__main__":
    asyncio.run(main())
