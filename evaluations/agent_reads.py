"""Measure recorded reads in deciding-line files using the existing discovery and JVN parsers.

No commands from receipts are executed. Only dev110 and P1-P7/U1-U6 are admitted.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import statistics
from pathlib import Path

from excerpt_rules import node_range, syntax_rules

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.languages import parse_language, sgconfig_of
from jev_navigator.index.spans import Span
from jev_navigator.index.tools import ast_grep_rules

HARD_CASES = {f"P{i}" for i in range(1, 8)} | {f"U{i}" for i in range(1, 7)}
NUMBERED = re.compile(r"^\s*(\d+)\t(.*)$")


def ranges(numbers):
    result = []
    for number in sorted(set(numbers)):
        if result and number == result[-1][1] + 1:
            result[-1][1] = number
        else:
            result.append([number, number])
    return result


def summary(numbers):
    return (
        {
            "count": len(numbers),
            "median": statistics.median(numbers),
            "min": min(numbers),
            "max": max(numbers),
        }
        if numbers
        else {"count": 0}
    )


def requests(pipelines):
    """Recognise ordinary shell reads with the discovery parser; never execute shell text."""
    result = []
    for pipeline in pipelines:
        if not pipeline:
            continue
        words = pipeline[0]
        tool = words[0]
        numbered = tool == "nl" and "-ba" in words
        if tool == "nl" or tool == "cat":
            files = [word for word in words[1:] if not word.startswith("-")]
            start, end = 1, None
        elif tool == "sed" and "-n" in words:
            position = words.index("-n") + 1
            match = re.fullmatch(r"(\d+)(?:,(\d+|\$))?p", words[position])
            if not match:
                continue
            start = int(match[1])
            end = None if match[2] == "$" else int(match[2] or match[1])
            files = words[position + 1 :]
        else:
            continue
        for stage in pipeline[1:]:
            if stage[0] == "sed" and "-n" in stage:
                match = re.fullmatch(r"(\d+)(?:,(\d+|\$))?p", stage[stage.index("-n") + 1])
                if match:
                    start = int(match[1])
                    end = None if match[2] == "$" else int(match[2] or match[1])
                else:
                    files = []
            else:
                files = []  # Other pipeline transformations cannot promise source lines.
        result.extend(
            {"file": file.removeprefix("./"), "start": start, "end": end, "numbered": numbered}
            for file in files
        )
    return result


def bind_output(output, reads, source):
    """Bind only emitted source text. Ambiguous numbered lines never earn delivery."""
    observed = collections.defaultdict(set)
    rows = output.splitlines()
    candidates = []
    for row in rows:
        match = NUMBERED.match(row)
        if not match:
            candidates.append(None)
            continue
        number, text = int(match[1]), match[2]
        files = {
            read["file"]
            for read in reads
            if read["numbered"]
            and read["start"] <= number <= read["end"]
            and source[read["file"]][number - 1] == text
        }
        candidates.append((number, files))
    # Blank and repeated source lines bind through adjacent unambiguous output in the same
    # consecutive source run. This does not bridge truncation messages or numbering resets.
    changed = True
    while changed:
        changed = False
        for i, candidate in enumerate(candidates):
            if candidate is None or len(candidate[1]) <= 1:
                continue
            number, files = candidate
            neighbours = []
            for j, wanted in ((i - 1, number - 1), (i + 1, number + 1)):
                if 0 <= j < len(candidates) and candidates[j] is not None:
                    previous_number, previous_files = candidates[j]
                    if previous_number == wanted and len(previous_files) == 1:
                        neighbours.append(previous_files)
            if neighbours:
                narrowed = files.intersection(*neighbours)
                if len(narrowed) == 1:
                    candidates[i] = (number, narrowed)
                    changed = True
    ambiguous = 0
    for candidate in candidates:
        if candidate is None:
            continue
        number, files = candidate
        if len(files) == 1:
            observed[next(iter(files))].add(number)
        elif len(files) > 1:
            ambiguous += 1
    for read in reads:
        if read["numbered"]:
            continue
        block = "\n".join(source[read["file"]][read["start"] - 1 : read["end"]])
        # Require a full requested block and multiple nonblank lines, avoiding one-token matches.
        if sum(bool(line.strip()) for line in block.splitlines()) >= 2 and block in output:
            observed[read["file"]].update(range(read["start"], read["end"] + 1))
    return observed, ambiguous


def grep_hits(output, file):
    return {
        int(match[1]) for match in re.finditer(r"(?m)^(?:\./)?" + re.escape(file) + r"[:-](\d+)[:-]", output)
    }


def signature_headers(root, files):
    """Reuse the excerpt study's signature rules through JVN's existing grammar owner."""
    grouped = collections.defaultdict(list)
    headers = collections.defaultdict(dict)
    refusals = {}
    for file in files:
        language = parse_language(file, (root / file).read_bytes())
        if language:
            grouped[language].append(file)
    for language, paths in grouped.items():
        rules = "\n---\n".join(
            rule for rule in syntax_rules(language).split("\n---\n") if rule.startswith("id: signature_")
        )
        for match in ast_grep_rules(rules, paths, root, config=sgconfig_of(language), refused=refusals):
            body = match.get("metaVariables", {}).get("single", {}).get("PART")
            if body:
                start, end = node_range(match)
                body_start = body["range"]["start"]["line"] + 1
                headers[match["file"]][(start, end)] = [
                    start,
                    max(start, body_start - 1 if language == "python" else body_start),
                ]
    return headers, refusals


def function_coverage(functions, lines, headers=None):
    covered = []
    for function in functions:
        overlap = lines.intersection(range(function.start, function.end + 1))
        if not overlap:
            continue
        header = (headers or {}).get((function.start, function.end))
        covered.append(
            {
                "header": header,
                "header_only": bool(header)
                and min(overlap) >= header[0]
                and max(overlap) <= header[1]
                and len(overlap) < function.size(),
                "name": function.name,
                "range": [function.start, function.end],
                "read_lines": len(overlap),
                "whole": len(overlap) == function.size(),
                "start_only_proxy": function.start in overlap
                and max(overlap) <= function.start + 2
                and len(overlap) != function.size(),
            }
        )
    return covered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planning-root", type=Path, default=Path.home() / ".local/share/jvn-takeover/2026-10-03"
    )
    args = parser.parse_args()
    planning = args.planning_root
    discovery = planning / "discovery"
    module_spec = importlib.util.spec_from_file_location("discovery_analyze", discovery / "analyze.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    commands = json.loads((discovery / "commands.json").read_text())
    ledger = [json.loads(line) for line in (discovery / "lines.jsonl").read_text().splitlines()]
    admitted = [row for row in ledger if row["slice"] == "replay-dev110" or row["case"] in HARD_CASES]
    if len(admitted) != 229 or {r["case"] for r in admitted if r["slice"] != "replay-dev110"} != HARD_CASES:
        raise ValueError("Unexpected admitted population; expected dev201 and P/U28 deciding lines")
    cases = collections.defaultdict(list)
    for row in admitted:
        cases[row["case"]].append(row)
    roots = {}
    indices = {}
    source = {}
    headers_by_root = {}
    parser_refusals = {}
    manifests = {}
    boundaries = []
    records = []
    file_records = []
    ambiguous_total = 0
    unbound = []
    for case, lines in sorted(cases.items()):
        manifest_path = planning / "search-design/case1/data" / (case.replace(":", "_") + ".json")
        manifest = json.loads(manifest_path.read_text())
        manifests[str(manifest_path)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        root = Path(manifest["root"])
        roots[case] = str(root)
        files = sorted({line["file"] for line in lines})
        root_key = str(root)
        if root_key not in indices:
            all_files = sorted({row["file"] for row in admitted if row["slice"] == lines[0]["slice"]})
            # Only deciding files are parsed. Other receipt files are read solely for output identity.
            root_files = [file for file in all_files if (root / file).is_file()]
            indices[root_key] = CodeIndex(root, root_files)
            headers_by_root[root_key], parser_refusals[root_key] = signature_headers(root, root_files)
        index = indices[root_key]
        functions = {}
        for file in files:
            facts = index.facts_in_files([file]).get(file)
            functions[file] = facts.structure.functions if facts else ()
            source[(root_key, file)] = (root / file).read_text().splitlines()
        for line in lines:
            holders = [f for f in functions[line["file"]] if f.contains(line["line"])]
            holder = min(holders, key=Span.size, default=None)
            boundaries.append(
                {
                    "case": case,
                    "slice": line["slice"],
                    "file": line["file"],
                    "line": line["line"],
                    "function": {"name": holder.name, "range": [holder.start, holder.end]}
                    if holder
                    else None,
                    "has_agent_trace": case in commands,
                }
            )
        history = commands.get(case, {}).get("commands", [])
        cumulative = collections.defaultdict(set)
        prior_hits = collections.defaultdict(set)
        for command in history:
            parsed_reads = requests(module.pipelines(command["command"]))
            valid_reads = []
            command_source = {}
            for read in parsed_reads:
                file = read["file"]
                path = root / file
                if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                    continue
                key = (root_key, file)
                if key not in source:
                    source[key] = path.read_text(errors="replace").splitlines()
                command_source[file] = source[key]
                read = {**read, "end": min(read["end"] or len(source[key]), len(source[key]))}
                if read["start"] <= read["end"]:
                    valid_reads.append(read)
            observed, ambiguous = bind_output(command["output"], valid_reads, command_source)
            ambiguous_total += ambiguous
            for file in files:
                file_requests = [r for r in valid_reads if r["file"] == file]
                if not file_requests:
                    prior_hits[file].update(grep_hits(command["output"], file))
                    continue
                read_lines = observed[file]
                if not read_lines:
                    unbound.append(
                        {
                            "case": case,
                            "file": file,
                            "step": command["step"],
                            "requests": file_requests,
                            "command": command["command"],
                        }
                    )
                coverage = function_coverage(functions[file], read_lines, headers_by_root[root_key][file])
                requested_lines = {n for r in file_requests for n in range(r["start"], r["end"] + 1)}
                grep_windows = [
                    {"range": [a, b], "hit": hit, "lines_before": hit - a, "lines_after": b - hit}
                    for a, b in ranges(read_lines)
                    for hit in sorted(prior_hits[file])
                    if a <= hit <= b
                ]
                record = {
                    "case": case,
                    "file": file,
                    "step": command["step"],
                    "receipt_line": command["receipt_line"],
                    "command": command["command"],
                    "requested_ranges": ranges(requested_lines),
                    "observed_ranges": ranges(read_lines),
                    "observed_lines": len(read_lines),
                    "requested_lines": len(requested_lines),
                    "functions": coverage,
                    "earlier_grep_hits_inside_read": sorted(prior_hits[file].intersection(read_lines)),
                    "grep_hit_windows": grep_windows,
                    "deciding_lines_read": sorted(
                        {r["line"] for r in lines if r["file"] == file}.intersection(read_lines)
                    ),
                    "output_has_truncation_notice": "truncated" in command["output"].lower(),
                }
                records.append(record)
                cumulative[file].update(read_lines)
                prior_hits[file].update(grep_hits(command["output"], file))
        for file in files:
            deciding = {row["line"] for row in lines if row["file"] == file}
            file_records.append(
                {
                    "case": case,
                    "slice": lines[0]["slice"],
                    "file": file,
                    "source_lines": len(source[(root_key, file)]),
                    "source_sha256": hashlib.sha256((root / file).read_bytes()).hexdigest(),
                    "deciding_lines": sorted(deciding),
                    "observed_ranges": ranges(cumulative[file]),
                    "unique_read_lines": len(cumulative[file]),
                    "deciding_lines_read": sorted(deciding.intersection(cumulative[file])),
                    "has_agent_trace": bool(history),
                    "functions": function_coverage(
                        functions[file], cumulative[file], headers_by_root[root_key][file]
                    ),
                }
            )
    successful = [r for r in records if r["observed_lines"]]
    dev_files = [r for r in file_records if r["slice"] == "replay-dev110"]
    with_whole = sum(any(f["whole"] for f in r["functions"]) for r in successful)
    with_partial = sum(any(not f["whole"] for f in r["functions"]) for r in successful)
    counters = {
        "provider_calls": 0,
        "usd": 0,
        "population": {
            "dev_findings_in_manifest": 110,
            "dev_findings_with_deciding_lines": len(
                {r["case"] for r in admitted if r["slice"] == "replay-dev110"}
            ),
            "dev_deciding_line_incidences": 201,
            "hard_cases": 13,
            "hard_deciding_line_incidences": 28,
            "hard_cases_with_agent_trace": 0,
            "dev_case_file_pairs": len(dev_files),
        },
        "read_command_file_pairs": len(records),
        "verified_read_command_file_pairs": len(successful),
        "unbound_read_command_file_pairs": len(unbound),
        "ambiguous_numbered_output_rows": ambiguous_total,
        "reads_covering_at_least_one_whole_function": with_whole,
        "reads_cutting_at_least_one_function": with_partial,
        "reads_whole_only": sum(
            bool(r["functions"]) and all(f["whole"] for f in r["functions"]) for r in successful
        ),
        "reads_partial_only": sum(
            bool(r["functions"]) and all(not f["whole"] for f in r["functions"]) for r in successful
        ),
        "reads_without_function_overlap": sum(not r["functions"] for r in successful),
        "reads_with_header_only_function_overlap": sum(
            any(f["header_only"] for f in r["functions"]) for r in successful
        ),
        "reads_all_function_overlap_header_only": sum(
            bool(r["functions"]) and all(f["header_only"] for f in r["functions"]) for r in successful
        ),
        "reads_with_start_only_proxy": sum(
            any(f["start_only_proxy"] for f in r["functions"]) for r in successful
        ),
        "reads_containing_earlier_grep_hit": sum(
            bool(r["earlier_grep_hits_inside_read"]) for r in successful
        ),
        "grep_hit_window_lines_before": summary(
            [w["lines_before"] for r in successful for w in r["grep_hit_windows"]]
        ),
        "grep_hit_window_lines_after": summary(
            [w["lines_after"] for r in successful for w in r["grep_hit_windows"]]
        ),
        "observed_lines_per_read": summary([r["observed_lines"] for r in successful]),
        "requested_lines_per_read": summary([r["requested_lines"] for r in records]),
        "unique_observed_lines_per_deciding_file": summary([r["unique_read_lines"] for r in dev_files]),
        "dev_deciding_lines_seen_in_verified_read": sum(len(r["deciding_lines_read"]) for r in dev_files),
        "boundary_counts": {
            slice_name: dict(
                collections.Counter(
                    "inside_function" if r["function"] else "outside_function"
                    for r in boundaries
                    if r["slice"] == slice_name
                )
            )
            for slice_name in ("replay-dev110", "hard27-tuning")
        },
        "signature_parser_refusals": parser_refusals,
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                discovery / "commands.json",
                discovery / "lines.jsonl",
                discovery / "analyze.py",
                Path(__file__).resolve(),
                Path(__file__).with_name("excerpt_rules.py").resolve(),
            ]
        },
        "manifest_sha256": manifests,
    }
    output = planning / "search-design/excerpts"
    output.mkdir(exist_ok=True)
    (output / "agent-reads.json").write_text(
        json.dumps(
            {
                "reads": records,
                "files": file_records,
                "deciding_boundaries": boundaries,
                "unbound_reads": unbound,
                "roots": roots,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "agent-reads-summary.json").write_text(json.dumps(counters, indent=2) + "\n")
    write_report(output, counters)
    print(json.dumps({k: v for k, v in counters.items() if not k.endswith("sha256")}, indent=2))


def write_report(output, counts):
    n = counts["verified_read_command_file_pairs"]
    text = f"""# What the recorded agents actually read

This is an offline measurement of source shown by ordinary file-read commands in files holding deciding lines.
It uses dev110 and P1-P7/U1-U6 only. No provider calls ran. Cost: $0.

The discovery ledger contains 201 dev deciding-line incidences across 89 findings with traces. The dev set has
110 findings; the other 21 have no deciding-line rows in this ledger. The 13 P/U tuning cases contribute 28 truth
lines and no agent command histories. They inform function-boundary counts only, not claims about agent behaviour.
Repeated lines in separate findings count separately. A file means a case/file pair, not a globally unique path.

| Measurement | Count or distribution |
|---|---|
| Dev deciding files | {counts["population"]["dev_case_file_pairs"]} case/file pairs |
| File reads requested in deciding files | {counts["read_command_file_pairs"]} command/file pairs |
| File reads with verified emitted source | {n} command/file pairs |
| Reads containing a complete JVN function | {counts["reads_covering_at_least_one_whole_function"]} / {n} |
| Reads cutting through a JVN function | {counts["reads_cutting_at_least_one_function"]} / {n} |
| Reads with complete functions only | {counts["reads_whole_only"]} / {n} |
| Reads with partial functions only | {counts["reads_partial_only"]} / {n} |
| Reads with no function overlap | {counts["reads_without_function_overlap"]} / {n} |
| Reads containing a hit from an earlier grep command | {counts["reads_containing_earlier_grep_hit"]} / {n} |
| Reads whose only function overlaps are parser-derived headers | {counts["reads_all_function_overlap_header_only"]} / {n} |
| Reads containing at least one header-only function overlap | {counts["reads_with_header_only_function_overlap"]} / {n} |
| Reads exposing a function start and at most its next two lines | {counts["reads_with_start_only_proxy"]} / {n} |
| Dev deciding lines shown in verified file reads | {counts["dev_deciding_lines_seen_in_verified_read"]} / 201 |

Whole and cut counts can overlap: a broad read may show two complete functions and part of a third. These are literal
coverage counts, not guesses about intent. A read containing an earlier grep hit is consistent with following search
results, but the trace does not establish that the agent centred the read on that hit. Header ranges come from the excerpt study's signature rules through JVN's grammar owner. Header-only means that the
read's overlap with a function is confined to that header, at line granularity. A broad read can clip only the header
of its last function while showing bodies elsewhere. The separate all-header measure avoids conflating those shapes.
The function-start measure is a proxy only and is not used to classify header-only reads.

Observed lines per verified command/file pair: `{json.dumps(counts["observed_lines_per_read"])}`.
Requested lines per command/file pair: `{json.dumps(counts["requested_lines_per_read"])}`.
Unique observed lines per dev deciding file, summed across its history: `{json.dumps(counts["unique_observed_lines_per_deciding_file"])}`.
These distributions measure only deciding files, not the agent's total reading across every file.
Context before each earlier grep hit inside a verified observed range:
`{json.dumps(counts["grep_hit_window_lines_before"])}`. Context after:
`{json.dumps(counts["grep_hit_window_lines_after"])}`. Multiple hits in one range each contribute an observation;
these are hit/window pairs, not independent reads.

## Method and limits

`evaluations/agent_reads.py` reuses `discovery/analyze.py`'s shell parser and JVN CodeIndex's existing
function facts and `evaluations/excerpt_rules.py`'s signature rules. The discovery parser preserves unquoted shell newlines as statement separators.
Quoted script newlines remain inside their original argument. It never executes a receipt command.
It records requested source ranges separately from emitted
ranges. Numbered output must match the source line exactly. Repeated and blank lines bind through neighbouring
consecutive, unambiguous output; ambiguity does not earn delivery. Unnumbered reads earn delivery only when their
complete requested block, with at least two nonblank lines, occurs verbatim in the output. A requested block truncated
in an unnumbered output can therefore be undercounted. All original requests and notices remain inspectable.

{counts["unbound_read_command_file_pairs"]} requested command/file pairs had no verified output;
{counts["ambiguous_numbered_output_rows"]} numbered output rows remained ambiguous across files.
Source comes from the case-1 manifest's pinned roots. Exact output matching catches disagreement with those roots.
Signature-parser refusals: `{json.dumps(counts["signature_parser_refusals"])}`.
Function boundaries use the current JVN parser. Unsupported languages, module code, schemas and omitted inputs may
have no function boundary. This study does not measure comprehension or a newly assembled pack's effectiveness.

`agent-reads.json` holds every requested and observed range, receipt command and line, overlapping function coverage,
case/file totals and each deciding line's containing function. `agent-reads-summary.json` holds exact denominators
and input hashes. P/U boundaries: `{json.dumps(counts["boundary_counts"]["hard27-tuning"])}`.
Dev boundaries: `{json.dumps(counts["boundary_counts"]["replay-dev110"])}`.

## Shell-parser correction and earlier attribution limits

The original discovery lexer treated unquoted shell newlines as whitespace. A command containing several
`nl | sed` reads could inherit the last range for the first file and omit later files. The owning parser in
`discovery/analyze.py` now separates shell statements while preserving quoted scripts and pipeline continuations.
Focused regression coverage is in `evaluations/tests/test_agent_reads.py`: four checks fail with the pre-fix
parser for this defect; all eight checks pass after the fix.

On the allowed dev traces, 89 of 1,216 commands across 18 findings parse differently. Recomputing descriptions
at the original exposure steps changes 10 first-line and 3 first-file description records. These counts measure
description changes, not revised term-origin labels. The existing `discovery/lines.jsonl` and generated discovery
reports remain unchanged and retain the earlier parser's attribution limitations. This study uses their deciding
locations, and derives reads afresh from command output. Its corrected file-read count is 169 deciding lines;
the initial run with the faulty parser counted 155. No broader populations were scanned to assess this defect.

## Reproduce

From `/Users/andremachon/Projects/jev-navigator-excerpts`:

```sh
PYTHONPATH=src /Users/andremachon/.local/share/system-one-proof/jvn-eval-2026-10-03/checkouts/engine-e733ea0c/.venv/bin/python evaluations/agent_reads.py
```

The default planning root is `/Users/andremachon/.local/share/jvn-takeover/2026-10-03`; override it with
`--planning-root` only for a copy of the same allowed data. This command makes no network or provider requests.
"""  # noqa: E501
    (output / "AGENT-READS.md").write_text(text)


if __name__ == "__main__":
    main()
