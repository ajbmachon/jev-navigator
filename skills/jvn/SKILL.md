---
name: jvn
description: Use JVN to locate semantically described code, enumerate matching function bodies, or trace relationships from a known function. Choose exact text and compiler reference tools when the symbol is already known.
---

# Search code with JVN

Choose the operation by the evidence needed:

| Need | Tool |
|---|---|
| Function/class counts, largest functions, physical line-size ranges | `jvn stats` (parser facts, no model) |
| Exact identifier, literal or filename | `rg` / `rg --files` |
| Exact callers or references of a resolved symbol | Compiler/reference tooling or JVN's `CodeIndex`; model judgment is unnecessary |
| Locate an implementation described by behavior | `jvn find` |
| Enumerate functions implementing a specified behavior, including disconnected implementations | `jvn findall` |
| Follow relationships around a known function and assess workflow evidence | `jvn trace` |
| Independently score source chunks against a property | `jgrep`; inspect its scope/omissions separately |

Ask a concrete question: “where is a quote rejected when absent from the source?” works better than
“find the complete algorithm” or “find bad code.” Separate distinct behaviors when one query hides
several judgments. Jev judges; parser facts and code perform counting and arithmetic.

```sh
jvn stats --kind function --limit 1
jvn find 'where an evidence quote is rejected as absent from the source'
jvn findall 'functions that implement rejection of unsupported evidence quotes'
jvn trace 'how the source quote becomes an accepted or rejected claim' --start app/evidence.py:42
```

Run in the source directory, or add `--repo /path/to/repo`. Dirty trees and non-Git directories work.
Every file under it is searched whether git tracks it or not; ignored files (a .gitignore, or outside
Git ripgrep's ignore files) are not, and the report lists them as not indexed with the reason.
Output defaults to a unique run folder under `~/.local/share/jev-navigator/runs/` whose path the run prints; nothing is written into the project. Trace starts must be repository-relative
`PATH:LINE` values inside a function or method, not a class declaration. Unknown entry? Find first,
inspect the returned function, then trace it. Quote the entire natural-language argument once.
Do not edit files in scope while a search runs: a file that changes is reported unavailable when the index reads it again, and a
search that finds nothing then ends `scope_incomplete` instead of `nothing_left`. A file too large to
parse even alone (a large minified bundle, or many megabytes of generated code) is never parsed: it is reported
unavailable with the reason "too large to parse", and it ends a not-found search the same way.

For agents and pipelines, discover the current contract with `jvn schema find`, `jvn schema findall`
or `jvn schema trace`; use `jvn help COMMAND` for examples. Pass JSON inline, as a file path or `-`:

```sh
jvn --json '{"command":"find","target":"where source quotes are rejected","repo":"."}'
```

Read `report.md` and `manifest.json`; they name code as `path:start-end` with file hashes (neighbours
as `path:line name`, key mentions as `mentions a key (path:line)`), so open it there.
`--keep-requests` (JSON `"keep_requests": true`) also keeps code and exact request text; use it only
for your own or open-source code. Resume works without it. `provider.input_tokens` adds only the
counts the provider reported; `responses_without_usage` counts responses that reported none (null
when resumed from an older pack), so 0 tokens with a non-zero count means unknown, not free.
Findall covers indexed function bodies; uncertain, unsupported and unexamined code remain gaps.
Trace expands the whole connected component: not a precise data-flow slice, nor proof the path is
complete. Its five judgments cover input, transformation, handoff, outcome and relevant branches.

Read Find's outcome before claiming anything; no outcome proves the code is absent:

- `found`: the reported span crossed the yes bar. Claim that location, nothing wider.
- `unsure_only`: the best candidates stayed unsure. Open and check them yourself.
- `scope_incomplete` ("not found: Jev judged code in N of M files; K more were read only to list
  links; U never reached"): claim only that the places Jev judged, in N files, did not show it.
- `nothing_left`: all files were read, Jev judged code in N. Claim nothing worth opening was left.
- `budget`, `cancelled` or `failed`: unfinished. Resume it; claim nothing about the rest. `failed`
  exits 1 and names the error: fix its cause first.

Find defaults to 24 live requests, Findall to 48. Only requests sent to the provider count: answers
replayed from the answer store and local work are free, and `--max-calls none` removes the cap. After
the cap refuses a request, Find stops at the first round no stored answer covers and saves the
unopened places. Trace has no default cap and no saved continuation. A budget-stopped Find or Findall
offers another allowance in a terminal (never in JSON or pipes); continue with the same target and
`--resume /path/to/previous-pack`; completed Findall judgments remain. Ctrl+C cancels; a cancelled
Trace may leave a journal without a manifest; keep its output.

Find All and Trace judge at most 16 functions per request and send their requests in parallel; a Find
opening still asks about all its neighbours in one request. Every answer goes to one shared answer
store, `$XDG_CACHE_HOME/jev-navigator/answers-v2.sqlite` (`~/.cache` when unset or relative), which
holds hashes, locations and answers, never code, and forgets a request no run reused for 30 days. A later run at the same commit replays from it after the
requests that learn the served model: one for Find All and Trace, one per place a Find's first round
opens. Give each experiment or eval arm its own store with `--answer-store PATH` (or
`JEV_NAVIGATOR_ANSWER_STORE`) outside the cache folder so arms never reuse each other's answers;
stderr names the store in use, and a store inside the cache folder is refused with exit status 2.
`jvn trace` reports `replayed_answers` beside its live `calls`.

The first search in a scope parses every file in it once and records each name and its lines in the
name table under the same cache folder; later searches over unchanged files look names up there
without searching or parsing. The table holds names and line numbers, never code or string literals.
JVN prunes its own caches and run folders as runs end (rules: `docs/cli.md`, Disk use and
housekeeping); `jvn cache status` shows disk use. Copy a run folder you want to keep, or use `--out`.

Live searches send source to the configured provider: reuse the user's source and spend authorization,
and never print credentials (the environment, the `.env` of a jev-navigator checkout `jvn` runs
from, or `~/.config/jvn/env`). Progress is stderr, JSON stdout.
Check the exit status first: 0 completed, 1 failed, 2 invalid input, 130 cancelled.

Library compositions and maintained options: `docs/extending.md` and `docs/cli.md` in the
[JVN repository](https://github.com/ajbmachon/jev-navigator). Static candidates remain candidates;
use narrow batched judgments only for the property static facts cannot establish.
