"""Offline literal packet replay on the original roles16 judgment groups.

Run with the proof Engine interpreter and pinned harness on PYTHONPATH. This
uses the original allocator, never infers answers for newly selected groups,
and renders only archived delivery source from the shared frozen bindings.
Masked request states remain the binding owner for the original answers.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

PROOF = Path("/Users/andremachon/.local/share/system-one-proof/jvn-eval-2026-10-03")
PLANNING = Path("/Users/andremachon/.local/share/jvn-takeover/2026-10-03")
FROZEN_SHA = "f3685c220818bff335a326c6c096edd488f5b00edaba42df81ea9e3c57e02576"


def load(path):
    return json.loads(path.read_text())


def write_tsv(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deny_network(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("This packet replay permits zero network/provider calls.")


def numbered_source(code, runs):
    numbers = [n for a, b in runs for n in range(a, b + 1)]
    lines = code.splitlines()
    assert len(numbers) == len(lines), (len(numbers), len(lines), runs)
    return dict(zip(numbers, lines, strict=True))


def render_runs(source, selected_runs):
    """Use inline markers for every omitted range, including boundary cuts."""
    from excerpt_rules import ranges

    selected = {n for a, b in selected_runs for n in range(a, b + 1)}
    assert selected <= source.keys()
    parts = []
    for a, b in ranges(source):
        cursor = a
        while cursor <= b:
            shown = cursor in selected
            end = cursor
            while end < b and ((end + 1 in selected) == shown):
                end += 1
            if shown:
                parts.extend(f"{n}: {source[n]}" for n in range(cursor, end + 1))
            else:
                parts.append(f"... ELIDED lines {cursor}-{end} ({end - cursor + 1} lines) ...")
            cursor = end + 1
    return {
        "runs": selected_runs,
        "body": "\n".join(parts),
        "shown_lines": len(selected),
        "omitted_lines": len(source) - len(selected),
    }


def historical_prefixes(metadata, role_root):
    from find_eval.frozen_inputs import historical_case_folder

    root = role_root.parent / "real-pack-e733ea0c-jvn0a73a59c-value-full"
    historical = load(root / "dev110-corrected/windows.json")
    result = {}
    for cid, case in metadata["cases"].items():
        pack = (historical_case_folder(root, cid, historical) / "pack.txt").read_text()
        ranked = pack.find("\nRANKED ")
        prefix = pack[:ranked] if ranked >= 0 else pack.rstrip("\n")
        assert len(prefix) == case["pack_chars"] - case["ranked_capacity_chars"], cid
        cursor = 0
        positions = []
        for region in case["floor_windows"]:
            header = (
                f"{region['role'].upper()} {region['name']} "
                f"{region['file']}:{region['first_line']}-{region['last_line']}\n"
            )
            start = prefix.find(header, cursor)
            assert start >= 0, (cid, header)
            cursor = start + len(header)
            positions.append((start, region, header))
        positions.sort(key=lambda item: item[0])
        prose = prefix[: positions[0][0]].rstrip("\n") if positions else prefix
        floors = []
        for index, (start, region, header) in enumerate(positions):
            end = positions[index + 1][0] if index + 1 < len(positions) else len(prefix)
            section = prefix[start:end].rstrip("\n")
            source = {int(n): text for n, text in re.findall(r"(?m)^(\d+): (.*)$", section)}
            assert set(source) == set(range(region["first_line"], region["last_line"] + 1))
            number_start = re.search(r"(?m)^\d+: ", section).start()
            floors.append(
                {
                    "region": region,
                    "header": header.rstrip("\n"),
                    "facts": section[len(header) : number_start],
                    "section": section,
                    "source": source,
                }
            )
        assert "\n".join([prose, *(f["section"] for f in floors)]) == prefix, cid
        result[cid] = {"prefix": prefix, "prose": prose, "floors": floors}
    return result


def rank_positions(pairs, units, observations, evidence_type):
    """Recover renderer reason labels from the original per-point score order."""
    by_point = collections.defaultdict(dict)
    for pair, meta in sorted(pairs.items(), key=lambda item: item[1]["order"]):
        unit = units[meta["unit_key"]]
        extent = (unit["file"], tuple(map(tuple, unit["extent"])))
        score = evidence_type("", observations[pair]).relevance
        old = by_point[meta["point_id"]].get(extent)
        if old is None or score > old:
            by_point[meta["point_id"]][extent] = score
    return {
        point: {
            extent: (index, len(entries))
            for index, (extent, _) in enumerate(sorted(entries.items(), key=lambda i: -i[1]), 1)
        }
        for point, entries in by_point.items()
    }


def verify_historical_whole(metadata, by_case, observations, reducer, required, historical_cases):
    """Bind the retained whole baseline without adding historical selective arms."""
    from find_eval.simulate import composed_case

    totals = {}
    for factor in [1, 3, 5]:
        delivered = 0
        for cid, original in metadata["cases"].items():
            cap = original["pack_chars"] * factor
            fixed_chars = original["pack_chars"] - original["ranked_capacity_chars"]
            case = {**original, "ranked_capacity_chars": cap - fixed_chars}
            result = composed_case(
                case,
                by_case[cid],
                metadata["units"],
                {pair: observations[pair] for pair in by_case[cid]},
                reducer,
                required,
            )
            expected = historical_cases[str(factor)]["cases"][cid]
            assert result["selected"] == expected["selected"], (cid, factor, "original selection")
            assert result["labels"] == expected["labels"], (cid, factor, "original labels")
            delivered += sum(label["delivered"] for label in result["labels"])
        assert delivered == {1: 80, 3: 115, 5: 126}[factor]
        totals[f"x{factor}"] = delivered
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, default=PROOF)
    parser.add_argument("--planning", type=Path, default=PLANNING)
    parser.add_argument("--rules", nargs="+", required=True)
    parser.add_argument(
        "--selective", action="store_true", help="Compare selective rules in separate artifacts."
    )
    args = parser.parse_args()
    # No import-time global audit hooks: importing this module stays harmless.
    sys.addaudithook(deny_network)
    import excerpt_rules
    from enginepy.workflows.document_analysis.evidence_pack import PackUnit, RankedUnit
    from find_eval.simulate import _covers, _windows, composed_case, pair_probabilities, proposal_reducer

    selective = None
    extra_hashes = {}
    if args.selective:
        import selective_rules

        selective = selective_rules
        if not set(args.rules) <= set(selective.RULES):
            parser.error("--rules must name existing selective rules")
        owner_modules = selective.owners()
        for module in (selective, *owner_modules):
            path = Path(module.__file__)
            extra_hashes[path] = digest(path)
    elif not set(args.rules) <= set(excerpt_rules.RULES):
        parser.error("--rules must name existing excerpt rules")
    helper_hash = digest(Path(excerpt_rules.__file__))
    script_hash = digest(Path(__file__))
    role_root = args.proof / "runs/roles-compare-20261006"
    out = args.planning / "search-design/excerpts"
    if args.selective:
        out /= "selective"
    out.mkdir(parents=True, exist_ok=True)
    prior_controls = {}
    if args.selective:
        for line in (out.parent / "pack-cases.jsonl").open():
            row = json.loads(line)
            if row["variant"] in {"whole", "D"} and row["budget"].startswith("uniform"):
                prior_controls[(row["variant"], row["budget"], row["case"])] = row
    metadata = load(role_root / "step-3/metadata.json")
    assert digest(role_root / "frozen-unit-points.jsonl") == metadata["pairs_sha256"] == FROZEN_SHA
    assert len(metadata["cases"]) == 110
    policy = load(role_root / "step-2/selection-policy.json")
    reducer = proposal_reducer(role_root / "step-1b/roles-PROPOSAL.md")
    observations, models = pair_probabilities(
        role_root / "step-3/paid/roles16/answer-table.jsonl",
        load(role_root / "step-3/roles16-candidate.json")["request"]["questions"],
    )
    assert set(observations) == set(metadata["pairs"]) and models == {"jev-1.13.0"}
    excerpt_rules.STOP_WORDS = set(load(role_root / "efficiency/ordering-method.json")["stop_words"]) | {
        "py",
        "json",
        "yaml",
        "md",
        "mjs",
        "test",
        "tests",
    }
    hashes = load(role_root / "step-3/final-artifact-hashes.json")
    for relative, expected in hashes.items():
        assert digest(role_root / relative) == expected, relative
    spend_hash = digest(role_root / "spend.json")
    by_case = collections.defaultdict(dict)
    sources, points = {}, {}
    for text in (role_root / "frozen-unit-points.jsonl").open():
        row = json.loads(text)
        meta = metadata["pairs"][row["hash"]]
        by_case[meta["case_id"]][row["hash"]] = meta
        unit_key = meta["unit_key"]
        source = numbered_source(row["state"]["unit"]["code"], metadata["units"][unit_key]["runs"])
        assert unit_key not in sources or sources[unit_key] == source
        sources[unit_key] = source
        cid = meta["case_id"]
        assert cid not in points or points[cid] == row["state"]["point"]
        points[cid] = row["state"]["point"]
    masked_sources = sources
    delivery_file = args.planning / "search-design/fitting/delivery-lines.json"
    sources = {key: {int(n): line for n, line in body.items()} for key, body in load(delivery_file).items()}
    assert set(sources) == set(metadata["units"])
    delivery_hash = digest(delivery_file)
    different_masked_costs = 0
    for key, unit in metadata["units"].items():
        archived = render_runs(sources[key], unit["runs"])
        assert len(archived["body"]) == unit["numbered_chars"], key
        different_masked_costs += len(render_runs(masked_sources[key], unit["runs"])["body"]) != len(
            archived["body"]
        )
    assert different_masked_costs == 403
    del masked_sources
    floor_data = historical_prefixes(metadata, role_root)
    historical_cases = load(role_root / "funnel/pack-curves.json")["roles16"]
    baseline_verification = None
    if args.selective:
        baseline_verification = verify_historical_whole(
            metadata, by_case, observations, reducer, policy["required_roles"], historical_cases
        )
        print("Original whole baselines verified:", baseline_verification, flush=True)
    windows = load(role_root / "line-windows/render-bindings.json")
    roots, files = {}, collections.defaultdict(set)
    for cid in metadata["cases"]:
        # Open exactly the original judged dev110 IDs, never heldout manifests.
        case = load(args.planning / "search-design/case1/data" / (cid.replace(":", "_") + ".json"))
        roots[cid] = case["root"]
        files[case["root"]].update(metadata["units"][p["unit_key"]]["file"] for p in by_case[cid].values())
        files[case["root"]].update(f["region"]["file"] for f in floor_data[cid]["floors"])
    structures = {root: excerpt_rules.Structure(root, sorted(paths)) for root, paths in files.items()}
    selectors = {}
    if selective is not None:
        queries, citations, input_paths = selective.load_case_inputs(args.proof, metadata["cases"])
        catalogue = args.planning / "search-design/case1/data/catalogue.sqlite"
        for path in [catalogue, *input_paths]:
            extra_hashes[path] = digest(path)
        for root in files:
            cases = [cid for cid in metadata["cases"] if roots[cid] == root]
            scent = selective.repository_scent(root, catalogue)
            selectors[root] = selective.Selector(
                scent, {cid: queries[cid] for cid in cases}, {cid: citations[cid] for cid in cases}
            )
    print(
        "Bound original masked request states, archived delivery bodies, prefixes, and parser roots.",
        flush=True,
    )
    ranks = {
        cid: rank_positions(pairs, metadata["units"], observations, reducer[0])
        for cid, pairs in by_case.items()
    }
    summary, case_rows, line_rows, packet_rows, compression_rows = [], [], [], [], []
    variants = ["whole", "D", *args.rules] if args.selective else ["whole", "window6", *args.rules]
    budgets = [] if args.selective else [(f"x{factor}", factor, None) for factor in [1, 3, 5]]
    budgets += [(f"uniform{tokens}", None, tokens) for tokens in [7200, 20000, 36000]]

    for variant in variants:
        adjusted, details, rendered_floors = {}, {}, {}
        for cid, pairs in by_case.items():
            adjusted[cid], details[cid], rendered_floors[cid] = {}, {}, []
            structure = structures[roots[cid]]
            for meta in pairs.values():
                key = meta["unit_key"]
                if key in details[cid]:
                    continue
                unit = metadata["units"][key]
                if variant == "whole":
                    rendered = render_runs(sources[key], unit["runs"])
                elif variant == "window6":
                    rendered = render_runs(sources[key], windows["ranked"]["6"][cid][key]["runs"])
                elif selective is not None and variant in selective.RULES:
                    rendered = selectors[roots[cid]].render(
                        sources[key], structure.facts(unit["file"]), cid, variant, file=unit["file"]
                    )
                else:
                    rendered = excerpt_rules.render_excerpt(
                        sources[key], structure.facts(unit["file"]), points[cid], variant
                    )
                details[cid][key] = rendered
                adjusted[cid][key] = {
                    **unit,
                    "runs": rendered["runs"],
                    "numbered_chars": len(rendered["body"]),
                }
            for index, floor in enumerate(floor_data[cid]["floors"]):
                if variant == "whole":
                    rendered = render_runs(
                        floor["source"], [[floor["region"]["first_line"], floor["region"]["last_line"]]]
                    )
                elif variant == "window6":
                    rendered = render_runs(
                        floor["source"], windows["floor"]["6"][cid][index]["window"]["runs"]
                    )
                elif selective is not None and variant in selective.RULES:
                    file = floor["region"]["file"]
                    rendered = selectors[roots[cid]].render(
                        floor["source"], structure.facts(file), cid, variant, file=file
                    )
                else:
                    rendered = excerpt_rules.render_excerpt(
                        floor["source"], structure.facts(floor["region"]["file"]), points[cid], variant
                    )
                literal = {int(n): text for n, text in re.findall(r"(?m)^(\d+): (.*)$", rendered["body"])}
                assert literal == {
                    n: floor["source"][n] for a, b in rendered["runs"] for n in range(a, b + 1)
                }
                body = floor["header"] + "\n" + floor["facts"] + rendered["body"]
                if variant == "whole":
                    assert body == floor["section"]
                rendered_floors[cid].append({**floor, "rendered": body, "excerpt": rendered})
        compression = collections.defaultdict(collections.Counter)
        for cid, rendered in details.items():
            selected_samples = {
                f"whole_selected_x{factor}": {
                    by_case[cid][pair]["unit_key"]
                    for pair in historical_cases[str(factor)]["cases"][cid]["selected"]
                }
                for factor in [1, 3, 5]
            }
            for unit_key, excerpt in rendered.items():
                unit = metadata["units"][unit_key]
                holding = any(
                    _covers(_windows(unit), label["file"], label["first_line"], label["last_line"])
                    for label in metadata["cases"][cid]["labels"]
                )
                samples = [
                    "original_judged_pool",
                    *(name for name, keys in selected_samples.items() if unit_key in keys),
                ]
                for sample in samples:
                    for population in ["all", "deciding_holding" if holding else "other"]:
                        counts = compression[(sample, population)]
                        counts["unit_pieces"] += 1
                        counts["whole_body_chars"] += unit["numbered_chars"]
                        counts["excerpt_body_chars"] += len(excerpt["body"])
                        counts["whole_source_lines"] += len(sources[unit_key])
                        counts["shown_source_lines"] += excerpt["shown_lines"]
            for floor in rendered_floors[cid]:
                counts = compression[("all_fixed_source_regions", "all")]
                counts["unit_pieces"] += 1
                counts["whole_body_chars"] += len(
                    render_runs(
                        floor["source"], [[floor["region"]["first_line"], floor["region"]["last_line"]]]
                    )["body"]
                )
                counts["excerpt_body_chars"] += len(floor["excerpt"]["body"])
                counts["whole_source_lines"] += len(floor["source"])
                counts["shown_source_lines"] += floor["excerpt"]["shown_lines"]
        for (sample, population), counts in compression.items():
            compression_rows.append(
                {
                    "variant": variant,
                    "sample": sample,
                    "population": population,
                    **counts,
                    "body_share": counts["excerpt_body_chars"] / counts["whole_body_chars"],
                    "source_line_share": counts["shown_source_lines"] / counts["whole_source_lines"],
                }
            )
        for budget_name, factor, uniform_tokens in budgets:
            key = f"{variant}/{budget_name}"
            totals = collections.Counter()
            sizes, allowances = [], []
            for cid, original in metadata["cases"].items():
                cap = int(original["pack_chars"] * factor) if factor else uniform_tokens * 4
                fixed = floor_data[cid]["prose"]
                prose_trimmed = len(fixed) > cap
                if prose_trimmed:
                    fixed = fixed.splitlines()[0]
                floor_windows, all_floor, skipped = [], [], 0
                for floor in rendered_floors[cid]:
                    region_windows = [
                        {"file": floor["region"]["file"], "first_line": a, "last_line": b}
                        for a, b in floor["excerpt"]["runs"]
                    ]
                    all_floor.extend(region_windows)
                    if len(fixed) + 1 + len(floor["rendered"]) <= cap:
                        fixed += "\n" + floor["rendered"]
                        floor_windows.extend(region_windows)
                    else:
                        skipped += 1
                assert len(fixed) <= cap
                if variant == "whole" and factor:
                    assert fixed == floor_data[cid]["prefix"]
                case = {**original, "floor_windows": floor_windows, "ranked_capacity_chars": cap - len(fixed)}
                result = composed_case(
                    case,
                    by_case[cid],
                    adjusted[cid],
                    {p: observations[p] for p in by_case[cid]},
                    reducer,
                    policy["required_roles"],
                )
                packets, displayed = [fixed], list(floor_windows)
                for pair in result["selected"]:
                    meta = metadata["pairs"][pair]
                    unit = adjusted[cid][meta["unit_key"]]
                    extent = (unit["file"], tuple(map(tuple, unit["extent"])))
                    rank, count = ranks[cid][meta["point_id"]][extent]
                    probability = reducer[0]("", observations[pair]).relevance
                    ranked = RankedUnit(
                        unit["place"],
                        unit["file"],
                        tuple(map(tuple, unit["runs"])),
                        tuple(map(tuple, unit["extent"])),
                        unit["name"],
                        probability,
                        1,
                    )
                    body = details[cid][meta["unit_key"]]["body"]
                    packet = PackUnit(
                        ranked, (f"{meta['point_id']} at P={probability:.3f}, rank {rank} of {count}",), body
                    ).render()
                    literal = {int(n): t for n, t in re.findall(r"(?m)^(\d+): (.*)$", packet)}
                    expected = {
                        n: sources[meta["unit_key"]][n] for a, b in unit["runs"] for n in range(a, b + 1)
                    }
                    assert literal == expected
                    packets.append(packet)
                    displayed.extend(_windows(unit))
                if variant == "whole" and factor:
                    expected = historical_cases[str(factor)]["cases"][cid]
                    assert result["selected"] == expected["selected"], (key, cid, "original selection")
                    assert result["labels"] == expected["labels"], (key, cid, "original labels")
                packet = "\n".join(packets)
                assert len(packet) == len(fixed) + result["ranked_chars"] <= cap, (key, cid)
                if args.selective and variant in {"whole", "D"}:
                    prior = prior_controls[(variant, budget_name, cid)]
                    assert result["selected"] == prior["selected_pairs"], (key, cid, "control selection")
                    assert hashlib.sha256(packet.encode()).hexdigest() == prior["body_sha256"], (
                        key,
                        cid,
                        "literal control packet",
                    )
                selected_whole = [
                    w
                    for p in result["selected"]
                    for w in _windows(metadata["units"][by_case[cid][p]["unit_key"]])
                ]
                for label in result["labels"]:
                    file, a, b = label["file"], label["first_line"], label["last_line"]
                    assert _covers(displayed, file, a, b) == label["delivered"]
                    lost = not label["delivered"]
                    was_floor = _covers(original["floor_windows"], file, a, b)
                    line_rows.append(
                        {
                            "variant": variant,
                            "budget": budget_name,
                            "case": cid,
                            **label,
                            "ranked_excerpt_cut": lost and _covers(selected_whole, file, a, b),
                            "floor_excerpt_cut": lost and was_floor and not _covers(all_floor, file, a, b),
                            "floor_budget_skip": lost
                            and was_floor
                            and _covers(all_floor, file, a, b)
                            and not _covers(floor_windows, file, a, b),
                        }
                    )
                delivered = sum(label["delivered"] for label in result["labels"])
                totals["delivered"] += delivered
                totals["ranked_units"] += len(result["selected"])
                totals["floor_regions_skipped"] += skipped
                totals["prose_trimmed_cases"] += prose_trimmed
                sizes.append(len(packet) / 4)
                allowances.append(cap / 4)
                case_rows.append(
                    {
                        "variant": variant,
                        "budget": budget_name,
                        "case": cid,
                        "delivered": delivered,
                        "total_labels": len(result["labels"]),
                        "allowance_chars": cap,
                        "body_chars": len(packet),
                        "body_sha256": hashlib.sha256(packet.encode()).hexdigest(),
                        "fixed_chars": len(fixed),
                        "selected_pairs": result["selected"],
                        "floor_regions_skipped": skipped,
                    }
                )
                # Save one deterministic literal example per rule/budget, with source elisions.
                if cid == next(iter(metadata["cases"])):
                    relative = f"pack-packets/{variant}-{budget_name}.txt"
                    path = out / relative
                    path.parent.mkdir(exist_ok=True)
                    path.write_text(packet + "\n")
                    packet_rows.append(
                        {
                            "variant": key,
                            "case": cid,
                            "path": relative,
                            "body_chars": len(packet),
                            "allowance_chars": cap,
                            "sha256": digest(path),
                        }
                    )
            row = {
                "variant": variant,
                "budget": budget_name,
                **totals,
                "total_labels": 201,
                "mean_rendered_tokens_estimate": statistics.mean(sizes),
                "median_rendered_tokens_estimate": statistics.median(sizes),
                "mean_allowance_tokens_estimate": statistics.mean(allowances),
            }
            summary.append(row)
            print(key, totals["delivered"], round(statistics.mean(sizes)), flush=True)
            if variant == "whole" and factor:
                assert totals["delivered"] == {1: 80, 3: 115, 5: 126}[factor], key
    for relative, expected in hashes.items():
        assert digest(role_root / relative) == expected, relative
    assert digest(role_root / "spend.json") == spend_hash
    assert digest(delivery_file) == delivery_hash
    assert digest(Path(excerpt_rules.__file__)) == helper_hash
    assert digest(Path(__file__)) == script_hash
    for path, expected in extra_hashes.items():
        assert digest(path) == expected, str(path)
    for name, data in [
        ("pack-summary.json", summary),
        ("pack-packet-manifest.json", packet_rows),
        ("pack-compression.json", compression_rows),
    ]:
        (out / name).write_text(json.dumps(data, indent=2) + "\n")
    for name, rows in [("pack-cases.jsonl", case_rows), ("pack-lines.jsonl", line_rows)]:
        (out / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_tsv(out / "pack-summary.tsv", summary)
    write_tsv(out / "pack-lines.tsv", line_rows)
    write_tsv(out / "pack-compression.tsv", compression_rows)
    provenance = {
        "script_sha256": script_hash,
        "excerpt_rules_sha256": helper_hash,
        "source_roots": sorted(files),
        "selective_owner_and_input_hashes": {str(path): value for path, value in extra_hashes.items()},
        "selective_corpus_documents": {root: selector.documents for root, selector in selectors.items()},
        "selective_seed_source": (
            "Original finder claim.statement and exact claim.evidence integer line points; "
            "no expanded floor ranges or deciding labels"
        )
        if args.selective
        else None,
        "provider_calls": 0,
        "spend_usd": 0,
        "spend_sha256": spend_hash,
        "frozen_sha256": FROZEN_SHA,
        "answer_models": sorted(models),
        "exact_original_judged_pairs": len(observations),
        "selected_rules": args.rules,
        "physically_verified_packets": len(case_rows),
        "labels_verified": len(line_rows),
        "whole_baseline": [80, 115, 126],
        "historical_whole_verification": baseline_verification,
        "prior_whole_D_literal_controls": len(prior_controls),
        "hard27": "unknown: no original roles16 groups available",
        "tokens": "4 characters per token estimate",
        "window6": "Original stored radius-six geometry; inline exact elisions counted in physical budget",
        "source": "Archived fitting/delivery-lines.json and exact historical floor packets",
        "delivery_sha256": delivery_hash,
        "delivery_binding_owner": "fitting/study.py: frozen source-bindings by file, runs, raw-entry hash",
        "different_masked_request_costs": different_masked_costs,
        "request_context": "Original frozen masked states bind answers; delivery uses archived shown source",
        "limitation": (
            "Answers judged whole units in their original groups; excerpt judgment/comprehension unknown"
        ),
    }
    (out / "pack-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("All literal packet budgets and deciding-line delivery verified. Zero calls.", flush=True)


if __name__ == "__main__":
    main()
