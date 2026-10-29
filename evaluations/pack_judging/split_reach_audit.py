"""Join every fresh physical attempt and probability to its original frozen slots."""

from collections import Counter
from decimal import Decimal

from paid_client import JEV_RATE
from paid_support import Ledger, load, write

from jev_navigator.judgments.client import JEV_INPUT_LIMITS
from jev_navigator.judgments.questions import request_sha256

from .split_reach import CAP, OUT, REACH, digest, group_folder, records


def main():
    ledger = Ledger(OUT / "spend-ledger.jsonl", cap=CAP)
    entries, seen, counts, models = ledger.latest(), set(), Counter(), set()
    for record in records():
        if record["eligible"]:
            assert not group_folder(record).exists(), "Fresh work touched an unchanged group"
            counts["unchanged_groups_reused"] += 1
            continue
        base = group_folder(record)
        observation = load(base / "observation.json")
        assert observation["failure"] is None
        assert observation["request_sha256"] == record["request_sha256"]
        assert not observation["refused_slots"]
        expected = []
        success = {}
        for path in (base / "http").glob("reach_*/receipt.json"):
            receipt = load(path)
            assert receipt["id"] not in seen
            seen.add(receipt["id"])
            wire_path = path.with_name("request.json")
            wire = load(wire_path)
            assert digest(wire_path) == receipt["request_sha256"]
            assert not JEV_INPUT_LIMITS.exceeded_by(wire["state"], wire["questions"])
            assert wire["state"]["targets"] == record["request"]["state"]["targets"]
            size = len(wire["state"]["items"])
            matches = [
                start
                for start in range(17 - size)
                if record["request"]["state"]["items"][start : start + size] == wire["state"]["items"]
            ]
            assert len(matches) == 1, (record["case"], record["group"], matches)
            offset = matches[0]
            assert wire["questions"] == {
                key: value
                for key, value in record["request"]["questions"].items()
                if int(key.split("#")[1]) < size
            }
            event = entries[receipt["id"]]
            counts[f"status:{receipt['status']}"] += 1
            raw_path = path.with_name("response.json")
            if receipt["status"] == 200:
                raw = load(raw_path)
                assert event["status"] == "settled"
                assert Decimal(event["usd"]) == Decimal(raw["usage"]["input_tokens"]) * JEV_RATE
                models.add(raw["model"])
                key = request_sha256(wire["state"], wire["questions"])
                assert key not in success, "An exact answered half was paid twice"
                success[key] = raw
                for qid, answer in raw["answers"].items():
                    assert answer["type"] == "noul"
                    expected.append((offset + int(qid.split("#")[1]), qid, key, answer["noul"]))
            else:
                assert event["status"] == "reserved"
                if receipt["status"] == 529:
                    assert load(raw_path)["detail"]["error_type"] == "system_overloaded"
                else:
                    assert receipt["status"] is None and not raw_path.exists()
                    assert receipt["failure"].startswith("TypeSafeAPIConnectionError:")
        actual = [
            (row["slot"], row["raw_question_id"], row["answer_request_sha256"], row["score"])
            for row in observation["rows"]
        ]
        assert sorted(actual) == sorted(expected)
        assert sorted(row["slot"] for row in observation["rows"]) == list(range(16))
        counts["supplemented_groups"] += 1
        counts["successful_halves"] += len(success)
    assert seen == set(entries)
    assert counts["unchanged_groups_reused"] == 1709 and counts["supplemented_groups"] == 259
    balance = ledger.balance()
    assert balance >= 0
    write(
        OUT / "send-audit.json",
        {
            "status": "PASS",
            "provider_calls_during_audit": 0,
            "counts": dict(counts),
            "models": sorted(models),
            "new_step_counted_usd": str(ledger.cap - balance),
            "remaining_usd": str(balance),
            "reach_requests_sha256": digest(REACH / "requests.jsonl.gz"),
        },
    )
    print(dict(counts))


if __name__ == "__main__":
    main()
