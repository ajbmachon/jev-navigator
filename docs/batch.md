# Batched code operations

`jvn batch` is a transport for explicit operations over one `CodeIndex`. The caller chooses the
sequence. It does not plan a search or manufacture semantic answers.

```sh
jvn batch '{"operations":[{"op":"named_files","patterns":["*orders*"]},{"op":"refs","query":"cancel_order","window":2}]}' --repo .
jvn batch request.json --repo . --max-chars 24000
cat request.json | jvn batch - --repo .
jvn help batch
```

The same JSON accepts `"command":"batch"` through `jvn --json`. Top-level fields are
`operations` (required), `repo`, `max_chars`, `max_calls` and `replay`. Unknown fields are errors.
Facts create no model client. Ranking requires an explicit `--max-calls N` for live requests, or
`--replay` for exact stored answers with no live requests. Missing answers stay `not_judged`.

## Operations

Operation objects use the following fields. Paths are relative to the index. `scopes` restricts
paths or directories. `patterns` is an array of glob patterns, with `!` exclusions. File and basename
globs are supported. `limit` bounds records on a page, default 80.

| Operation | Input | Result |
| --- | --- | --- |
| `outline` | optional `file`, `scopes`, `patterns` | files, resolved import paths, symbol names and ranges |
| `names` | `name`, optional `scopes`, `patterns` | observed existing spelling variants and text-hit counts, rarest first |
| `def` | `name`, optional `file`, `line` | definitions; a file selects its own definition or its imported binding |
| `refs` | `name`, optional owning `file`, or `query` with `regex`, `window`, `scopes`, `patterns` | syntactic references and calls with binding status, or literal source matches with context |
| `callers` | `name`, optional owning `file` | call sites, including imported aliases of that owner |
| `callees` | `file`, `line` inside a symbol | outgoing calls and bindings resolved through imports and reexports |
| `named_files` | `query` containing path tokens, or `patterns` / `scopes` | files resolved from those inputs |
| `show` | `file`, `line` (default 1), optional `end`, `window` | exact source blocks with inclusive line and end numbers |
| `cochange` | `file` | neighbors and shared commit counts in the index's last 200 commits |
| `tests_of` | `file` or `name`, optional `scopes` | candidate tests with reasons: sibling name, imports source, mentions name |
| `rank` | `query`, `candidates` array of `file`, `start`, `end` | ranked candidates, raw answer probabilities and request hashes; partial and unjudged status |

`names` uses the existing spelling-variant generator. It is not the complete spelling map planned
in the architecture. Binding status remains explicit when static resolution is uncertain. An owning
file filters proven different definitions but retains uncertain candidates. `tests_of` suggests tests;
it does not prove coverage. Regex syntax is Python's. A show window is added on either side of the
requested range. Source blocks contain at most 60 lines before response paging.

## Counts and continuations

The complete compact JSON response fits `max_chars`, default 24,000 characters. Each operation gets
an equal display share. The minimum is 1,024 characters per operation plus 128 for the response.
Every page has `number` (operation position), `operation`, `total` (all matching records), numbered
`items`, `next` and `error`. A failed operation appears beside successful operations. The CLI returns
1 for an operation failure and 2 for an invalid request.

A non-null `next` is a cursor object with zero-based `row` and `character`. Resubmit the same
operation with that `cursor` to continue. Keep the inputs and source revision fixed. When one record
cannot fit, its JSON is split into `row_json` fragments, carrying `row_json_offset` and
`row_json_chars`. Concatenate fragments for that operation and row number, then decode the resulting
JSON. This preserves even a single long source line. `total` counts source blocks, not source lines;
source blocks carry their exact inclusive ranges. No page silently drops records or characters.

Rank continuations replay the original answer store and never send another live request. A library
Judge without a store cannot continue a rank page and returns an explicit error. For the CLI, pass
`--replay` on continuations to avoid loading a live transport. Exact cache identity includes code,
question, ordered batch mates and served model. Changing these inputs requires new judgment. Partial
results from a call cap retain completed probabilities; missing probabilities remain null.

## Library and async hosts

```python
from pathlib import Path

from jev_navigator.batch import Operation, run_batch, run_batch_async
from jev_navigator.index.code_index import CodeIndex

with CodeIndex.from_directory(Path(".").resolve()) as index:
    result = run_batch(index, [Operation("outline", file="app/orders.py")])
    print(result.render())

# In an async host, use its existing index and optional Judge:
result = await run_batch_async(index, operations, judge=judge,
                               checks=checks, shared=shared, max_chars=24000)
```

Pass a `Path` to `CodeIndex.from_directory` when the host requires a resolved directory. Mechanical
operations run in at most four threads and parse files lazily. Async judging uses the client's native
async method. Ranking operations run in caller order and share the supplied Judge's parent cap.
Candidate order stays unchanged in requests of at most 16 items. Oversized candidates are explicitly
refused by the existing Judge rather than silently cut. The response reports live `calls` and
`replayed_answers` for this batch only. Supply a store on the Judge when ranking may need paging.

By default rank reuses the existing query-match Check. Hosts can pass their own Check list and
shared state. Generic `score` is the largest supplied probability, with judged candidates ordered
before missing ones; hosts needing another composition use the raw `answers`. The batch block has
no caller-specific packet policy, role vocabulary or model route.
