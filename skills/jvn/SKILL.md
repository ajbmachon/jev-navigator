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
Output defaults to a unique `./jvn-results/` directory. Trace starts must be repository-relative
`PATH:LINE` values inside a function or method, not a class declaration. Unknown entry? Find first,
inspect the returned function, then trace it. Quote the entire natural-language argument once.

For agents and pipelines, discover the current contract with `jvn schema find`, `jvn schema findall`
or `jvn schema trace`; use `jvn help COMMAND` for examples. Pass inline/file/stdin JSON:

```sh
jvn --json '{"command":"find","target":"where source quotes are rejected","repo":"."}'
jvn --json request.json
```

Read `report.md`, `manifest.json` and the request journal. `provider.input_tokens` adds only the
counts the provider reported, and `responses_without_usage` counts responses that reported none (null
when resumed from an older pack), so 0 tokens with a non-zero count means unknown, not free. Find
stops at a match. Findall describes coverage of indexed function bodies; uncertain, unsupported and
unexamined code remain gaps.
Trace currently expands the bidirectional connected component, which can be broad: it is not a
precise data-flow slice or a proof that the requested path is complete. Its five atomic judgments
cover input, transformation, handoff, outcome and relevant branches.

Find defaults to 24 live model requests; Findall defaults to 48. Only requests sent to the provider count;
answers replayed from the answer store and local code work are free. `--max-calls none` removes it.
Trace has no default request/depth cap. A budget-stopped
Find and Findall offer another allowance in an interactive terminal after saving their work. JSON and piped
commands never prompt. Continue either search with the same target and `--resume /path/to/previous-pack`;
completed Findall judgments remain available across the stop. Trace has no saved continuation. Ctrl+C cancels; a cancelled Trace may have a
journal without a finished manifest. Preserve the diagnostic and existing output.

Find All and Trace judge at most 16 functions per request and send their requests in parallel; a Find
opening still asks about all its neighbours in one request. Every answer goes to
one shared answer store, `~/.cache/jev-navigator/answers.sqlite`, which holds hashes, locations and
answers, never code. A later run at the same commit replays from it after one live request. Give each
experiment or eval arm its own store with `--answer-store PATH` (or `JEV_NAVIGATOR_ANSWER_STORE`) so
arms never reuse each other's answers; stderr names the store in use. `jvn trace` reports
`replayed_answers` beside its live `calls`.

Live searches send selected source to the configured provider. Reuse the user's existing source
and spend authorization. Credentials come from environment or `~/.config/jvn/env`; never print them.
Progress is stderr; JSON results are stdout. Check exit status and recorded outcome before claiming
success, complete coverage, or an absence of matches.

Library compositions and maintained options: `docs/extending.md` and `docs/cli.md` in the
[JVN repository](https://github.com/ajbmachon/jev-navigator). Static candidates remain candidates;
use narrow batched judgments only for the property static facts cannot establish.
