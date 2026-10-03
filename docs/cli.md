# jvn command-line guide

Locate code by describing the behavior you need to inspect. `find` follows code relationships and
uses Jev to judge concrete source evidence. It stops on a match; it does not promise all matches or
a complete end-to-end trace. Use a concrete question such as “where do we reject evidence quotes
that are absent from the source?” rather than “find everything important”.

## Contents

- [Start with one command](#start-with-one-command)
- [Discover commands and request fields](#discover-commands-and-request-fields)
- [Every find option](#every-find-option)
- [Continue after a call limit](#continue-after-a-call-limit)
- [JSON requests](#json-requests)
- [Results, progress and exit status](#results-progress-and-exit-status)
- [Agent workflow](#agent-workflow)
- [Find All function search](#find-all-function-search)
- [Workflow trace](#workflow-trace)
- [Structural measurements](#structural-measurements)

## Start with one command

```sh
jvn find "the check that limits how many items an order may have"
```

The current directory is the search root. Git is optional and uncommitted changes are included.
The command chooses an entry point and creates `./jvn-results/<directory>-<timestamp>/` in the
directory where you invoked it. You do not need to supply a scope, starting line or budget.

Credentials come from `TYPESAFE_API_KEY` and `TYPESAFE_BASE_URL` in the process environment, then
from `~/.config/jvn/env`. The file uses dotenv syntax and is not executed. Help and schema discovery
need no key and make no model calls.

## Discover commands and request fields

```sh
jvn help                 # same general help as jvn --help
jvn help find            # same command help as jvn find --help
jvn schema find          # JSON Schema: fields, types, defaults and required properties
```

The schema and JSON parser read the same command option definitions. Agents should discover that
schema instead of inventing arguments. It describes the request, not the evidence-pack manifest.

## Every find option

Each example below is complete. The numbers demonstrate optional controls; they are not a required
configuration. Live calls stop at 24 unless you set another cap; depth, steps and neighbour counts are
unlimited unless you set a limit.

| Option | Default and purpose | Example |
|---|---|---|
| `--repo PATH` | Current directory. Select another search root. | `jvn find "the order limit" --repo /path/to/repository` |
| `--prefix PATH` | Whole source inventory. Limit scope to a file or directory, relative to the search root. Repeat for multiple scopes. | `jvn find "the order limit" --prefix app/ --prefix tests/` |
| `--start PATH:LINE` | Automatic entry selection. Start from a known caller or entry point; repeat for multiple starts. Lines are 1-based, paths are relative to the search root. | `jvn find "the order limit" --start app/orders.py:42 --start app/routes.py:18` |
| `--out PATH` | A unique directory under `./jvn-results/`. Choose another new or empty directory. | `jvn find "the order limit" --out ./order-evidence` |
| `--resume PATH` | Off. Continue a budget-stopped or cancelled evidence pack into a new output directory. | `jvn find "the order limit" --resume ./order-evidence` |
| `--max-depth N` | Unlimited. Maximum relationship hops from the starting places; `0` opens only those places. | `jvn find "the order limit" --max-depth 3` |
| `--max-steps N` | Unlimited. Maximum distinct code openings during navigation; entry selection is separate. | `jvn find "the order limit" --max-steps 8` |
| `--max-calls N\|none` | `24`. Maximum model requests, including automatic entry selection; each is a paid request. `none` lifts the cap. One request may contain many questions. This is not a token or monetary cap. A search that reaches it ends with outcome `budget` and its unexplored places in `not_inspected`. | `jvn find "the order limit" --max-calls 8` |
| `--beam-width N` | `3`. Places opened together in a navigation round. `1` makes navigation sequential. A wider round may do more work before a match stops the search. | `jvn find "the order limit" --beam-width 1` |
| `--neighbours-per-kind N` | Unlimited. Retain at most this many candidates per relationship kind from each opened place. Explicitly omitted candidates stay visible in the result. | `jvn find "the order limit" --neighbours-per-kind 8` |
| `--preview-lines N` | `8`. Leading source lines shown with a neighbour candidate's signature; `0` omits its code preview. | `jvn find "the order limit" --preview-lines 12` |
| `--max-slice-chars N` | `12000`. Character allowance for an opened code slice, ending on a line boundary. This does not bound the entire request, its candidate previews or its questions. | `jvn find "the order limit" --max-slice-chars 24000` |
| `--max-line-chars N` | `240`. Clip long lines in opened source, previews and signatures shown to the model. Source files are not edited. | `jvn find "the order limit" --max-line-chars 480` |
| `--verbose` | Off. Print expanded masked requests on stderr as they are sent. Concise phase/request/elapsed progress is already on by default. | `jvn find "the order limit" --verbose` |
| `-h`, `--help` | Print help and exit without searching. | `jvn find --help` |

Limits and context sizes affect how much evidence the search can inspect. Read `search.outcome`,
`search.not_inspected` and the recorded source spans before interpreting coverage. `max_calls=8`
means up to eight requests, not eight questions. Reducing previews or clipping source can remove
information that matters to the judgment.

A supplied `--start` is navigation context: the library records its judgment in `search.starts` and
continues looking for a target reached from it. Supply a caller or entry point rather than the
function you already believe is the answer. Automatic entry selection can find a target directly.

An opening with many neighbours uses separate requests when the provider input allowance requires
it. Every neighbour keeps its preview and independent judgment; each actual request counts toward
the call allowance. The global `open_first` Choice is a scheduling hint. If its complete request or
option set exceeds provider capability, history records `open_first.unavailable` and the search
orders the assessed neighbours by their individual probabilities. Completed answers remain in the
store and journal when a call stop interrupts the opening, so continuation can reuse them.

## Continue after a call limit

In an interactive terminal, `find` and `findall` save a partial pack when the live-call allowance is used up,
then asks whether to continue with another allowance of the same size. Answer `y` or `yes` to
continue; any other answer or end of input leaves the saved pack intact. Each continuation creates
a new pack beside the previous one and reuses its saved frontier and answers. The final printed path
identifies the newest pack. No input is required until another paid call would need an allowance.

JSON mode and redirected or piped input/output never prompt. An explicit `--max-calls 0` also
returns immediately. Trace returns its recorded partial results at a call stop without prompting.

When `search.outcome` is `budget` or `cancelled`, the pack contains `resume.json`. Supply that pack to a follow-up
invocation with the same target, repository, prefixes and starts:

```sh
jvn find "the order limit" --repo /path/to/repository --out ./first-pack
jvn find "the order limit" --repo /path/to/repository --resume ./first-pack --out ./continued-pack
```

The follow-up invocation gets a fresh allowance by default (24 live calls for Find, 48 for Find All); `--max-calls N` or
`--max-calls none` changes that allowance. Stored answers and the journal carry forward, so replayed
answers cost no live calls. The new manifest combines earlier and new visits, history and call counts;
`search.calls_this_invocation` records only the new requests, while `search.calls` is cumulative;
the previous pack remains intact. A cap reached during automatic entry selection saves that stage,
and the next invocation replays its stored decisions before continuing. Resume requires unchanged
source and scope, the same thresholds and requested model. If the source changed, start a new search.

## JSON requests

Three equivalent input forms are supported:

```sh
jvn --json '{"target":"the order limit"}'
jvn --json request.json
cat request.json | jvn --json -
```

A minimal request is:

```json
{"target": "the check that limits how many items an order may have"}
```

All the options in the longer example can also be supplied as one object:

```json
{
  "command": "find",
  "target": "the check that limits how many items an order may have",
  "repo": "/path/to/repository",
  "prefix": ["app/"],
  "start": ["app/orders.py:42"],
  "out": "./order-evidence",
  "max_depth": 3,
  "max_steps": 8,
  "max_calls": 8,
  "beam_width": 1,
  "neighbours_per_kind": 8,
  "preview_lines": 8,
  "max_slice_chars": 12000,
  "max_line_chars": 240,
  "verbose": false
}
```

Omit fields you do not need. `command` defaults to `find`. JSON field names use underscores in place
of flag hyphens. `prefix` and `start` are arrays even for one item. Numbers and booleans are JSON
values, not strings. Unknown fields are errors. `null` is accepted for `out` and the optional limits
`max_depth`, `max_steps` and `neighbours_per_kind`, whose defaults are unset, and for `max_calls`,
where it lifts the default cap of 24.

Paths are relative to the invocation directory, not the JSON file's directory. Prefixes and start
paths are relative to `repo`. Use `--json` on its own and put search settings inside the request.

## Results, progress and exit status

JSON mode writes one result object to stdout. It contains:

| Field | Meaning |
|---|---|
| `output_directory` | Absolute path to the saved evidence pack. |
| `manifest` | Absolute path to the complete `manifest.json`. |
| `report` | Absolute path to the readable `report.md`. |
| `search` | Outcome, matched spans, source code, decisions, request counts and coverage details. |
| `provider` | Requested/served model and `input_tokens`, the sum of the counts the provider reported. `responses_without_usage` counts responses that reported none (null when resumed from an older pack), so 0 tokens with a non-zero count means unknown, not free. |
| `resume` | Evidence pack path to pass to `--resume` when the outcome is `budget` or `cancelled`; otherwise `null`. |

Progress, expanded requests and errors go to stderr, so stdout remains parseable. For example:

```sh
jvn --json '{"target":"the order limit"}' > result.json
jq '{outcome: .search.outcome, matches: .search.found, report}' result.json
```

Check the command's exit status before reading a result file:

| Exit code | Meaning |
|---|---|
| `0` | A search finished and wrote its result. Read `search.outcome`; this does not guarantee a match. |
| `1` | Search, configuration, filesystem or provider failure. Read stderr. |
| `2` | Invalid command or request. Read stderr. |
| `130` | Cancelled with Ctrl-C. Existing journal records remain available. |

The evidence directory contains `report.md`, `manifest.json`, `journal.jsonl` and `answers.jsonl`.
Budget-stopped and cancelled packs also contain `resume.json`.
The manifest retains the full record even if a pipeline selects only a few output fields. Journal
records preserve request/response evidence; inspect their exact-capture flags when auditing bytes.

## Agent workflow

1. Read `jvn schema find` to discover accepted fields and defaults.
2. Describe the behavior, not an assumed symbol name. Add a scope only when it follows from the task.
3. Send one JSON request. Use `start` only when a caller or entry point is already known.
4. Inspect the exit code, outcome, matched source and coverage. A successful command alone is not
   evidence that all relevant code was inspected.
5. Use the manifest/report paths for deeper evidence instead of repeatedly rerunning the same search.

Schema discovery and structured calls take inspiration from the
[Google Workspace CLI's agent guidance](https://github.com/googleworkspace/cli/blob/main/CONTEXT.md).
The Python library remains the interface for composing a custom planner or a broader workflow.

## Find All function search

Use `jvn findall "functions enforcing the order item limit"` with the same scope, output and
request-display options as `find`. `jvn schema findall` describes the JSON input. Find All defaults
to 48 live model calls, twice the 24-call `find` default. Use `--max-calls N` to change it or
`--max-calls none` (JSON `null`) to remove it. Parsing, graph traversal and cached answers are free. Depth/step/neighbour and preview options affect seed discovery only; the
function enumeration reads complete bodies. The call allowance is shared across seed discovery and enumeration. A budget stop writes completed
judgments and remaining coverage to a partial evidence pack.

The result includes `seed_search`, and `search` records `found`, `unsure`, `searched`, source hashes,
request identities, the static graph and coverage gaps. `functions_examined` means every enumerated
function was judged; it does not prove the model found every behavior. `scope_incomplete` retains
unsupported, unparsed or unavailable files. See the library composition in [extending.md](extending.md#compose-a-seed-first-find-all-search).

`seed_search.calls` counts seed discovery; `search.enumeration_calls` counts the following enumeration.
Their sum is `search.calls`, the whole workflow's actual model-request count.
Continue a budget-stopped pack with the same query and scope:

```sh
jvn findall "functions enforcing the order item limit" --resume ./jvn-results/previous-pack
jvn --json '{"command":"findall","target":"functions enforcing the order item limit","resume":"./jvn-results/previous-pack"}'
```

A stop during seed selection resumes that stage first. Once enumeration has begun, continuation
restores the seed and all completed judgments, including negative and uncertain answers, and
examines only the outstanding functions. Static graph reconstruction reuses parser caches; it is
not a paid call. `calls_this_invocation` shows new calls; `calls` and `enumeration_calls` are cumulative.
Source, scope, target, thresholds, model and containment-question identity must still match.

## Workflow trace

Start from a known function and inspect the connected workflow:

```sh
jvn trace "how an order request becomes an HTTP result" --start app/orders.py:42
jvn --json '{"command":"trace","target":"order request to HTTP result","start":["app/orders.py:42"]}'
jvn help trace
jvn schema trace
```

`--start PATH:LINE` is required and repeatable; paths are relative to `--repo` (current directory by
default). Use `find` first when the entry point is unknown. `--prefix` narrows scope, `--out` chooses
a new output directory, and `--verbose` displays masked requests. Without `--out`, packs go under
`./jvn-results/` in the invocation directory.

Static relationships drive traversal. Jev receives batched, independent questions about input
origin, transformation, handoff, outcome and relevant branches. The manifest retains all walked
functions and links, including uncertain bindings. Positive evidence does not prove a complete
end-to-end path or model accuracy.

There is no default depth or model-call limit for trace. Set `--max-depth N` or `--max-calls N`
explicitly when needed; `--max-calls none` is unlimited. Outcomes preserve `depth`, `budget`,
`cancelled` and `completed` separately. A budget stop retains completed answer batches and leaves
unexamined evidence unresolved. Saved continuation is available for Find and Find All, not Trace.

JSON stdout contains `output_directory`, `manifest`, `report`, `trace`, `provider` and `resume`
(`null` for trace). Progress and requests stay on stderr. See `trace.outcome`, its obligations and
`unresolved_links` before interpreting coverage. Ctrl-C stops the command with exit 130; an abrupt
interruption can leave the journal and answer store without a final manifest. The library also
offers cooperative cancellation between traversal steps and model batches that writes a partial
pack.

## Structural measurements

Use parser facts for counting and physical line sizes. No model, API key, or natural-language
judgment is needed:

```sh
jvn stats                                      # counts, largest symbols and line ranges
jvn stats --kind function --limit 1             # largest function, including methods
jvn stats --operation count --kind class        # class totals and counts per file
jvn stats --operation range --kind function --min-lines 20 --max-lines 80
jvn --json '{"command":"stats","kind":["function"],"limit":1}'
jvn schema stats
```

The command writes `statistics.json` and `statistics.md` under a unique `./jvn-results/` directory.
`--repo`, repeatable `--prefix`, and `--out` work as for search. Repeat `--operation` to choose
`count`, `largest`, or `range` sections; repeat `--kind` for `function` and `class`. Both kinds and
all sections are included by default. `--top-level` excludes methods/nested symbols from rankings
and ranges; counts still cover all parsed symbols of the requested kinds.

`--limit` restricts the displayed ranking, never the measured inventory. All tied largest symbols
remain identified in `largest.biggest`. `--min-lines` and `--max-lines` filter only the range listing,
inclusively; they do not change counts or the largest-symbol ranking. Sizes include both boundary
lines. Reports retain skipped, unreadable and partially parsed files; recovered counts are not a
claim that unsupported or malformed code was completely understood. Inspect `coverage` in JSON.

These are explicit structural commands. `find` still accepts a semantic target; it does not yet
route arbitrary natural-language arithmetic questions into `stats` automatically.
