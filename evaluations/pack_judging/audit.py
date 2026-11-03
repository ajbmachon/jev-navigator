"""Audit completed observations and paired line delivery without provider calls."""

import json
import statistics
from collections import Counter, defaultdict

from .frozen import hits, load, sha256
from .prepare import DEFAULT_CASE1, write
from .refresh import ADMITTED
from .run import OUT, folder, records


def main():
    cases = load(ADMITTED / "frozen-cases.json")
    baseline = {r["case"]: r for r in load(DEFAULT_CASE1 / "per-finding-summary.json")}
    results = load(OUT / "per-finding-results.json")
    finalists = load(OUT / "finalists.json")
    paired = defaultdict(Counter)
    choice = defaultdict(Counter)
    attempts = Counter()
    failures = []
    timing = defaultdict(list)
    pilot = load(ADMITTED / "pilot.json")["groups"]
    for name in (*finalists["jev"], finalists["llm"]):
        prepared = {(r["case"], r["group"]): r for r in records(name)}
        for case in cases:
            cid, dataset = case["context"]["case"], case["context"]["dataset"]
            sent = []
            assert baseline[cid]["labels"] == case["labels"]
            for group in case["groups"]:
                record = prepared[cid, group["ordinal"]]
                request = record["request"]
                state = (
                    request["state"]
                    if "state" in request
                    else json.loads(request["messages"][0]["content"].split("STATE_JSON:\n", 1)[1])
                )
                assert state == group["request"]["state"]
                observation = load(
                    OUT / "answers" / name / folder(cid) / str(group["ordinal"]) / "observation.json"
                )
                assert observation["request_sha256"] == record["request_sha256"]
                attempts[name] += 1
                if group["ordinal"] not in pilot.get(cid, []):
                    sent.append(observation)
                if observation["failure"]:
                    failures.append(
                        {
                            "arm": name,
                            "case": cid,
                            "group": group["ordinal"],
                            "failure": observation["failure"],
                            "deciding_lines_in_group": sum(
                                any(hits(p, label) for p in group["places"].values())
                                for label in case["labels"]
                            ),
                        }
                    )
                if name.startswith("J4"):
                    for point in group["request"]["state"]["targets"]:
                        answer = next(r for r in observation["rows"] if r["point"] == point)
                        stats = choice[dataset]
                        stats["questions"] += 1
                        winner = answer["winner"]
                        confident = answer["choice_confidence"] >= 0.70
                        stats["confident_questions"] += confident
                        stats["none"] += winner == "none"
                        stats["confident_none"] += winner == "none" and confident
                        if winner != "none":
                            deciding = any(hits(group["places"][winner], label) for label in case["labels"])
                            stats["item_picks"] += 1
                            stats["cited_deciding_picks"] += deciding
                            stats["confident_item_picks"] += confident
                            stats["confident_cited_deciding_picks"] += confident and deciding
            timing[f"{dataset}:{name}"].append(
                {
                    "case": cid,
                    "new_groups": len(sent),
                    "continuation_elapsed_seconds": max(r["finished_at"] for r in sent)
                    - min(r["started_at"] for r in sent),
                }
            )
    packet_count, max_chars = 0, 0
    for row in results:
        old = baseline[row["case"]]["arms"]["code"]
        for path in ("lab", "native"):
            for room in ("7200", "20000", "36000"):
                before = old["lab_rooms" if path == "lab" else "rooms"][room]["delivered"]
                after = (
                    [r["delivered"] for r in row[path][room]["labels"]]
                    if path == "lab"
                    else row[path][room]["delivered"]
                )
                assert len(before) == len(after) == row["metrics"]["registered_lines"]
                for b, a in zip(before, after, strict=True):
                    paired[f"{row['dataset']}:{row['arm']}:{path}:{room}"][
                        "kept" if a and b else "gained" if a else "lost" if b else "neither"
                    ] += 1
                if path == "native":
                    text = (
                        OUT / "packs" / row["arm"] / folder(row["case"]) / f"native-{room}.txt"
                    ).read_text()
                    assert len(text) == row[path][room]["chars"] <= int(room) * 4
                    packet_count += 1
                    max_chars = max(max_chars, len(text))
    write(
        OUT / "final-audit.json",
        {
            "status": "PASS",
            "attempts": dict(attempts),
            "failures": failures,
            "original_states_and_companions": "exact",
            "native_packets": packet_count,
            "native_packets_within_room": packet_count,
            "maximum_packet_chars": max_chars,
            "choice_confidence_070": {k: dict(v) for k, v in choice.items()},
            "full_continuation_timing": {
                k: {
                    "mean_elapsed_seconds": statistics.mean(r["continuation_elapsed_seconds"] for r in v),
                    "median_elapsed_seconds": statistics.median(r["continuation_elapsed_seconds"] for r in v),
                    "cases": v,
                }
                for k, v in timing.items()
            },
            "paired_line_delivery": {k: dict(v) for k, v in paired.items()},
        },
    )
    # Index immutable raw evidence separately from the generated summaries.
    index = {}
    for root in (
        ADMITTED,
        OUT / "answers",
        OUT / "guards",
        OUT / "failed-schema-transport",
        OUT / "canary-json-object",
        OUT / "canary-typed-json-object",
        OUT / "packs",
    ):
        for file in sorted(root.rglob("*")):
            if file.is_file():
                index[str(file.relative_to(OUT.parent))] = {
                    "sha256": sha256(file),
                    "bytes": file.stat().st_size,
                }
    write(OUT / "artifact-index.json", index)
    print("PASS", dict(attempts), packet_count, "packets;", len(index), "hashed artifacts")


if __name__ == "__main__":
    main()
