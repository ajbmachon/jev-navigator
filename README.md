# jev-navigator

Find code with small, typed AI judgments. jev-navigator is a Python library for code navigation in
three layers. The first works without any model: an index over a narrowed set of files (definitions,
callers, callees, references, text, imports, git history). The second adds judgments from
[Jev](https://docs.typesafe.ai), TypeSafe's System One model, as small building blocks: yes/no
checks, picks from a list code built, and scores, each returned with its raw probabilities. The third
composes both into searches such as "find the code this sentence describes". An optional piece,
`LlmStep`, lets you add an LLM call to your own function.

> **Jev gets concrete state and one closed judgment; code owns goals, loops and stopping.**

Coding agents can load the focused [JVN skill](skills/jvn/SKILL.md) for choosing a search,
using JSON requests, reading evidence and handling partial results. Install it by linking or copying
the `skills/jvn` directory into your agent harness's skills directory.

The library finds code. Judging that code (is a comment accurate, does a rule hold everywhere) is a
layer you build on top; [docs/extending.md](docs/extending.md) shows how to compose one.

For structural questions, use code directly: `jvn stats --kind function --limit 1` finds the largest
function without model calls. `jvn stats` reports counts and line ranges; see the
[structural command examples](docs/cli.md#structural-measurements).

## Trace a known workflow

```sh
jvn trace "how an order request becomes an HTTP result" --start app/orders.py:42
jvn --json '{"command":"trace","target":"order request to HTTP result","start":["app/orders.py:42"]}'
jvn help trace
```

Trace follows static relationships and batches atomic evidence judgments. It preserves uncertain
bindings, partial coverage and request evidence in `./jvn-results/`. A positive judgment is evidence,
not proof of a complete path. Use `find` first if the starting function is unknown.
See [trace options and outputs](docs/cli.md#workflow-trace).

## Find every matching function

```bash
jvn findall "functions that reject an order exceeding the item limit"
jvn --json '{"command":"findall","target":"functions that reject an order exceeding the item limit"}'
jvn help findall
jvn schema findall
```

`find` locates an implementation; `findall` finds a seed, examines related functions, then checks
remaining function bodies for disconnected implementations. It uses batched Jev judgments and
defaults to 48 live model calls (twice `find`); `--max-calls none` lifts that cap. There is no file cap. Reports, source provenance and request journals go to a unique
`./jvn-results/` directory. `functions_examined` describes coverage of function bodies, not a proof
of semantic equivalence or completeness across arbitrary code fragments. Uncertain answers and
unreadable or unsupported source stay visible. At a call stop, the terminal offers another allowance.
For a later invocation or an agent pipeline, pass `--resume ./jvn-results/previous-pack` with the same
Find All query and scope. Completed judgments and the seed are retained; only unfinished work spends
new model calls.

For an engineer-authored library composition and its limits, see
[Extending: seed-first Find All](docs/extending.md#compose-a-seed-first-find-all-search).

## Install

```sh
uv add "jev-navigator[typesafe] @ git+https://github.com/ajbmachon/jev-navigator"
```

Install the command globally with uv:

```sh
uv tool install "jev-navigator[typesafe] @ git+https://github.com/ajbmachon/jev-navigator"
jvn --help
```

Needs Python 3.11 or newer, and `ast-grep`, `rg` (ripgrep) and `git` on the PATH. The `typesafe` extra adds
the official SDK for live calls; set `TYPESAFE_API_KEY`. Everything else, including the tests, runs
offline.

## Live evidence-pack command

Start in the directory you want to search:

```sh
jvn find "the check that limits how many items an order may have"
```

In a terminal, reaching the call budget offers another allowance without losing the saved search.
JSON and piped commands return partial results without prompting; continue them with `--resume`.

That is enough. `jvn` chooses an entry point and creates a unique evidence pack under
`./jvn-results/`. It works with uncommitted changes and ordinary directories outside Git.
`find` follows code relationships to locate a match; it does not promise every matching function
or a complete end-to-end trace.

To search another directory, add just `--repo`:

```sh
jvn find "where do we reject evidence quotes that are absent from the source?" --repo /path/to/repository
```

Explicit `TYPESAFE_API_KEY` and `TYPESAFE_BASE_URL` process values win independently. Otherwise
`jvn` reads those settings from `~/.config/jvn/env` with a dotenv parser; it does not execute that
file or print the values. `TYPESAFE_BASE_URL` is the API root before `/v1/systemone`, such as
`http://127.0.0.1:4777/jvn` for a gateway serving `/jvn/v1/systemone`.

### JSON input for agents and pipelines

Put a request in `request.json`:

```json
{
  "target": "the check that limits how many items an order may have"
}
```

Then run:

```sh
jvn --json request.json
```

Or send the same request on stdin:

```sh
printf '%s\n' '{"target":"the check that limits how many items an order may have"}' | jvn --json -
```

`command` defaults to `find`, `repo` defaults to the current directory, and output goes to
`./jvn-results/` unless you supply `out`. JSON mode prints one result object on stdout with
`output_directory`, `manifest`, `report`, `search` and `provider`. Progress and errors stay on stderr.
For example, pipe the command's output to `jq '.search.found'` to read the matching source spans.
The report and manifest paths refer to the saved evidence pack. Failed invocations return a nonzero
exit status; check it before consuming stdout. Ctrl-C cancels the search.

Optional fields use CLI names with underscores instead of hyphens. Repeatable options are arrays,
numeric options are numbers, and `verbose` is a boolean:

```json
{
  "command": "find",
  "target": "the check that limits how many items an order may have",
  "repo": "/path/to/repository",
  "prefix": ["app/"],
  "start": ["app/orders.py:42"],
  "verbose": false
}
```

JSON and flags use the same defaults, option validation and search workflow. Unknown fields and
wrong value types are errors. Omit options you do not need; `null` also lifts the `max_calls` default.
Paths are relative to the invocation directory, including when the JSON
file lives elsewhere. Use `--json` on its own; put any search options inside the request.

### Optional search controls

Use `--prefix app/` to narrow the scope, `--start app/orders.py:42` to supply a known caller or entry
point, and `--out /path/to/new-pack` to select the result directory. Prefixes and starts are repeatable.
Without a start, `jvn` uses typed Jev judgments to select entry candidates from the source inventory.

Every live call is a paid request, so `--max-calls` defaults to 24 for the whole run, choosing an entry
point included; a search that reaches it ends with outcome `budget` and a resumable `not_inspected`
frontier (or a saved entry-selection stage if the cap arrives earlier). Resume with another `jvn find`
invocation using `--resume /path/to/previous-pack`; it gets a fresh call allowance and writes a new pack
while keeping the earlier evidence. `--max-calls none` lifts the cap. Depth and step limits are unset by default. If you want an
explicit allowance for a particular search, you can supply one:

```sh
jvn find "the check that limits how many items an order may have" \
  --repo /path/to/repository \
  --prefix app/ \
  --max-calls 8
```

Every option has its default, purpose and a complete example in the [CLI guide](docs/cli.md#every-find-option).
Use `jvn help find` for grouped help and examples, or `jvn schema find` for a machine-readable request
schema. Agents can also pass an inline JSON object: `jvn --json '{"target":"the order limit"}'`.
These controls are optional tuning, not prerequisites.

An explicitly selected output directory must be new or empty. Each evidence pack contains:

- `manifest.json`: schema version, navigator build fingerprint and source revision, inspected
  repository revision, explicit budget and thresholds, requested and served model, elapsed time,
  versioned code locations (`path:start-end` with file hashes; neighbours as `path:line name`), raw
  probabilities, full search history, uninspected frontier, and unparsed files.
- `report.md`: a readable outcome, source table, found locations, and coverage caveat.
- `journal.jsonl`: request hashes and exact provider responses as the run progresses.
- `answers.jsonl`: reusable typed answers keyed by source and request hashes. Every answer is also
  written to the machine's shared answer store (`$XDG_CACHE_HOME/jev-navigator/answers.sqlite`,
  `~/.cache` when the variable is unset, or `JEV_NAVIGATOR_ANSWER_STORE`), which holds no code; a later run at the same commit asking the
  same questions replays from it after one live request that learns the served model (Find All and
  Trace items carry the commit and file hashes, so a new commit asks again), and copies what it replays into its own
  `answers.jsonl`. `jvn trace` reports those answers as `replayed_answers` beside its live `calls`.
  `--answer-store PATH` points a run at another store file; each run prints the store it uses.
- `resume.json` (budget-stopped or cancelled runs): the frontier as locations; Resume re-reads the
  code from the unchanged repository.

By default the manifest, report, journal and resume state hold no source code, only locations and
hashes; a relation that quotes a mentioned key reads `mentions a key (path:line)`. `--keep-requests` (JSON `"keep_requests": true`) also keeps the code and full neighbour
signatures in the manifest and report and the exact request text in the journal; use it only for
your own or open-source code. The repository includes only a small public-format sample under
[`examples/evidence-pack`](examples/evidence-pack).

## Layer 1: index, operations and comments (no model)

```python
from jev_navigator.index.code_index import CodeIndex
from jev_navigator import operations, comments

index = CodeIndex.from_git(repo_root, prefixes=("app/", "web/"))
old = CodeIndex.at_commit(repo_root, "abc123", prefixes=("app/",))  # from git objects, checkout untouched
index.find_definition("LIMITS_KEY")  # functions, classes, constants, assignments, types, enums
index.find_callers("validate_order")  # CallSite(file, line, caller, binding), found by name
index.callee_edges(span)  # CallEdge(name, line, binding); find_callees gives names only
index.find_references("send_invoice")  # Reference(name, file, line, role, holder, binding): non-call uses
index.references_in(span)  # names a function passes on without calling (callbacks, registries)
index.enclosing_symbol(file, line)
index.symbols_in(file)
index.read_slice(span)
index.read_window(file, line, radius=10)
index.search_text("orders.max_items")  # ripgrep over the narrowed files only
index.imports(file)
index.dependents(file)
index.co_changed_files(file)

operations.slice_around(index, file, line)  # the enclosing function, or a window
operations.code_described_by_comment(index, file, line)  # the whole next symbol or block
operations.callers_of_file(index, path)
operations.trace_callers(index, symbol)  # and trace_callees; optional depth, otherwise fixed point
operations.trace_graph(index, index.find_definition(symbol))  # calls and non-call references
operations.similar_functions(index, symbol)
operations.code_named_in_doc(index, text)

comments.find_comments(index, files)  # FoundComments(kept, dropped) of CommentBlock
comments.comments_in_diff(index, base, head)  # changed comments, and comments above changed code
comments.code_above_comment(index, file, line)  # CodeAbove(code or None, reason)
```

Imports are read per statement, so an import spanning several lines counts like any other.
Script specifiers resolve in TypeScript's order, from what the repository declares:

- a relative path, with TypeScript's suffix rules: a specifier naming compiled output (`.js`, `.jsx`,
  `.mjs`, `.cjs`) names its TypeScript source when that exists, and a folder names its `index` file;
- the path aliases (`compilerOptions.paths` and `baseUrl`) of the nearest tsconfig.json, or
  jsconfig.json in a folder without one, following relative `extends`. Like TypeScript, an exact alias
  wins, otherwise the wildcard with the longest prefix; only its targets are tried, then `baseUrl`;
- a `#` specifier through the `imports` field of the importer's nearest package.json;
- any other bare specifier through the repository's own package of that name, whatever tool manages
  the workspace: its `exports` field (patterns, fallback lists and conditions, `types` before runtime
  conditions, since a bundle may serve every subpath), else its entry fields. When several
  package.json files claim a name, the one the importer lies in wins, else the one sharing the most
  leading folders with it; a tie resolves to neither.

A declared target the repository does not contain is build output. It resolves to the path under
`rootDir` when the package's tsconfig puts it in `outDir` or `declarationDir`, else to the same path
with leading folders dropped under the package's `src` folder or the package. A specifier none of
these explain stays unresolved. Comments and trailing commas in configs are fine. A named import
follows transitive `export * from` barrel files inside the index; cycles terminate, and more than one
matching definition remains a `candidate`. A config or package.json that is a symbolic link, or a
config that extends or points outside the index root, is not read: its aliases stay unknown and those
bindings stay `candidate`. Package `extends` and tsconfig `references` are not followed.
package.json files are found in the folders that hold scope files. `CodeIndex.at_commit` brings the
commit's tsconfig, jsconfig and package.json files along, outside the scope.
`CodeIndex.imports()` and `dependents()` include suggested repository package paths for navigation,
including source paths inferred from build output. A package.json mapping proves a call only when both
the package and the file are certain. The package is certain for a `#` import, a package's own name
through its `exports`, or the only package of a name the importer depends on through a `workspace:`
range; a `#` import or a dependency counts only when the importer's package.json is the nearest one on
disk. The file is certain when every declared target that exists names it and it is not a declaration
file. Any other package mapping (a version range that could install a published copy, a `workspace:`
alias, a name several packages claim, targets that differ by condition, a `.d.ts` file, a source
inferred from build output, a `#` target naming another `#` import) makes the call `candidate`, with
the mapping named as its reason. Relative imports, Python imports and declared
script-config paths keep their resolved bindings. Package redirects follow acyclic chains of any length
and stop when a specifier repeats.
File lists come from git with NUL separators, so
names with non-ASCII characters enter the scope as they are on disk, and lines split at newlines only,
as the parser counts them. A line that is not valid UTF-8 is read with its invalid bytes replaced, the
same way by `search_text` and by every other lookup. A scope path that is a symbolic link, or that leads
out of the root (through a linked directory or `..`), raises `UnsafePathError` when the index is built,
before any tool reads it.

The index extracts symbols, declarations, calls and non-call references together in one ast-grep
pass over the files a lookup actually needs. The pass runs a few hundred files per ast-grep process
and turns each match into its fact as ast-grep prints it, so memory holds the facts, never the
parser's output, and no command line outgrows the system's argument limit. Facts that start on the
same line are ordered by their position in the line, so every run returns them in the same order.
Exact-name lookups first use ripgrep to narrow the candidate files, and `prefetch_names` narrows
several names with one ripgrep; opening a known span parses its file directly. The resulting
per-file facts are cached by source bytes, language, ast-grep version, the rule text and the source
of the code that reads the matches, in `$XDG_CACHE_HOME/jev-navigator/facts` (`~/.cache` when the
variable is unset), so a new index can reuse facts without treating changed source or changed
parser rules as current. A file that changes on disk after the index first read it is
reported as unavailable rather than read in its new form. Each call
site's binding is computed once, and `search_text` and `co_changed_files` each run their tool once
per argument for the life of the index. The index keeps the lines of a bounded number of recently
read files (`LINE_CACHE_FILES`). There is no default file-count refusal or parser timeout, and no requested file is silently
omitted.

Before that pass, `.js` files whose leading comments (before any code, after an optional byte-order
mark or shebang) carry the `@flow` pragma are separated from plain JavaScript. They ride on the tsx
grammar — the closest available superset — through a `languageGlobs` sgconfig written outside the
scanned repository. Plain JavaScript keeps the JavaScript grammar unchanged. The tsx grammar is not
a Flow parser: unsupported constructs such as exact object types `{| |}`, `export opaque type`,
variance annotations, `?T` in static property types and inexact objects `...` remain visible as
ERROR nodes.

The facts include grammar ERROR nodes. A language's parser may recover only part of such a file
(a Flow-only construct, for example), so what it swallowed must not silently count as
indexed; the symbols it did recover still count. What it swallowed is unknown, not absent. The facts
keep the lines each ERROR node spans, and a definition names what it defines, so only a name those
lines mention can be hidden there: a call to such a name has status `unknown`, with the files in its
reason, unless a definition in another file, not imported from one of them, settles it. A completed
search reports `scope_incomplete` instead of `nothing_left`; a budget-limited result reports which
fact scans completed and which remain pending. A file that disappears after the working-directory
inventory was built is reported separately as unavailable. So is a file too large to parse safely:
`tools.ast_grep_rules`, the one door every parse passes through, never hands ast-grep a file whose
estimated parse peak (from the length of each line, `index/file_shape.py`) is over 250 MB, about
70,000 characters on one line, and `unavailable_files` gives the estimated peak and the longest line.
The file keeps its path in import relations, a name that may be defined in it binds `unknown`, and
`find_comments` lists it in `refused_files`. Any ast-grep or ripgrep failure other
than that verified disappearance still fails the lookup that triggered it.

Calls are found by name in the syntax tree, which is not a resolved binding. Every call carries a
`Binding(status, reason, target)`: `resolved` when a module-level definition in the same file, or one
an import names, proves the target, `candidate` when only the name matches (a method on an unknown receiver, or a
definition elsewhere with no import), `unresolved` when nothing in scope defines it, and `unknown` when
the definition may sit in lines the index could not parse. References carry a binding too. A
binding counts only the definitions its site can name: a type, a class or a declaration a type can
name, such as an interface; an export, any definition; and a call or any other reference (an
argument, receiver, condition or decorator), a function, class or declaration a value can name, such
as a module constant. A host with a real resolver (a code-intelligence service, a TypeScript alias
resolver, an LSP) passes it as
`binding_resolver=`; its answer wins. Trace steps and search neighbours carry the binding, so a
candidate edge is never presented as a proven call. Script constructor expressions such as `new
MemoryAdapter()` are calls too. A bound method passed as an argument (`bus.on(self.handler)`) is
indexed under its member name, so navigation can offer the method definition, and is bound like a
method call on an unknown receiver: a function of that name in the same file or an import never
proves it.

Every `CodeSlice` records its source: `slice.source()` gives the file, line range, commit (with
`+worktree` when the file had uncommitted changes) and how it was reached.

Comment kinds are `docstring`, `jsdoc`, `header`, `tool_directive`, `declaration`, `inline` and
`block`. Adjacent line comments of the same syntax are merged; a JSDoc block and the `//` lines
after it stay separate. Both finders return `FoundComments(kept, dropped)`. Filtering is yours: pass
`drop=` any rule that returns a reason to set a block aside, or None to keep it
(`comments.noise_reason` sets aside dividers, licence headers and bare tool directives). Every
dropped block comes back in `dropped` with its reason, so the total count stays known. Facts come from
pluggable rules (`jev_navigator.facts`): each `Fact` has a name, offsets, line and matched text. A rule
is `FactRule(name, pattern, keep=None)`; the shipped `DEFAULT_COMMENT_RULES` (TODO without owner,
commented-out code, date, ticket reference) are examples. `outside_names` is an optional filter that
skips matches inside paths, file names or identifiers: `DATE.with_filter(outside_names)`.

### Static trace graphs

`operations.trace_graph(index, roots)` follows both callers and callees, including non-call
references such as callback registrations. It makes no model calls. Choose concrete root spans
from the index when several functions share a name; resolved imports retain their actual target.

```python
roots = [span for span in index.find_definition("handle") if span.file == "app/orders.py"]
stop_requested = False  # your host can set this when the user cancels
graph = operations.trace_graph(index, roots, cancelled=lambda: stop_requested)
for link in graph.links:
    print(link.source, link.target, link.relation, link.binding)
print(graph.stop)  # fixed_point, depth, or cancelled
```

With no `depth`, traversal visits each reachable function once and stops when the frontier is
empty. An explicit `depth=2` limits traversal to two hops; cancellation is checked between functions.
There is no hidden depth, neighbour or frontier cap. Missing endpoints and uncertain bindings stay
visible in `graph.links`. `fixed_point` means the available static graph is exhausted; it does not
prove that runtime dispatch is resolved or that every stage relevant to your question is covered.
For the Jev-backed trace CLI, see [Workflow trace](docs/cli.md#workflow-trace).

## Layer 2: judgments

```python
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion, Pick, Rate
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.judgments.thresholds import Thresholds
from jev_navigator.adapters.typesafe import TypeSafeJevClient

judge = Judge(TypeSafeJevClient(), store=JsonlAnswerStore(path), thresholds=Thresholds.from_env())
judge.check_each(check, items, shared_state)  # one Noul per item, batched
judge.pick(pick, options, state)  # one Choice over options code built
judge.ask_all(state, checks=[...], picks=[(pick, options)], scores=[rate])  # one request
judge.choose_call(route, offers, state)  # function calling: operation plus its input
```

Every one of these has an async form (`check_each_async`, `pick_async`, `ask_all_async`,
`choose_call_async`, `ask_async`), and `find_code_async` is the async search. They take any
`AsyncJevClient` (an object with `model` and `async ask(state, questions)`, optionally an async
`send`), such as a host's own orchestrator; a sync client also works there and runs in a worker
thread. Both paths share one core: masking, the secret scan, the hash, the store lookup, the call
budget, the journal and the recording are the same steps, and only the send differs (a direct call,
or an awaited one). Batches of `check_each_async` and the places of each `find_code_async` round are
sent with `asyncio.gather` — except that the first batch of a `check_each_async` whose served model
is still unknown and which has an answer store goes out alone. Its live answer pins the served model,
so the remaining batches can replay from the store. The sync `check_each`, `check_every` and their
`iter_` forms send their batches on a thread pool, at most `Judge(max_concurrency=N)` at once
(default 16), with the same first-batch rule; the `iter_` forms yield each batch as it completes. The
call cap stays exact under concurrency, and after a failure or cancellation no batch sends a new
request, while answers already received still yield. A sync method given an async client raises
`TypeError`. Offline tests use `testing.AsyncScriptedJevClient`.

Budgets: `judge.calls` counts requests sent (store hits are free; `judge.replayed_answers` counts
the answers the store gave instead). `Judge(max_calls=N)` caps a judge
together with every `judge.scope()` made from it, and a scope counts its own calls; `find_code` runs
on its own scope, so searches sharing one judge never use up each other's budget.

- **Raw values are always kept.** `CheckResult.probability` is Jev's P(yes). `PickResult` keeps
  `probabilities` and `confidence`, and `ScoreResult` keeps `score`, `probabilities` and `confidence`.
  Verdicts are a convenience; apply any band you like to the raw values.
- **Thresholds.** A Noul is yes at 0.80 or above, no at 0.20 or below, and unsure in between; unsure
  is a result, never rounded. A Choice counts as confident when its `confidence` (from the whole
  distribution, not the winner's probability) is at least 0.70. Precedence: library defaults, then
  `JEV_NAVIGATOR_CHOICE_MIN_CONFIDENCE`, `JEV_NAVIGATOR_NOUL_YES_AT` and `JEV_NAVIGATOR_NOUL_NO_AT` (via
  `Thresholds.from_env()` at the edge), then a directive's defaults, then per-call overrides
  (`judge.effective(directive, call)`).
- **Secrets.** `SecretMasker` masks private keys, token shapes, secret-named assignments and
  high-entropy assignments in every request, by content: a value hidden in one place is hidden
  everywhere in the request, for example where a relation text or another candidate quotes it.
  The complete candidate set is masked once, before packing, so copied values stay hidden across
  batches; the final scan still runs on every request before it is sent.
  `SecretScanner` refuses to send a request that still contains a secret, and a masked value
  left in a key is refused too. Both are on by default; a host passes its own (a masker offers
  `mask(text)` and `masked_values(text)`), or turns one off explicitly with `None`.
- **Batches.** A batched request carries at most `Judge(items_per_request=N)` items (default 16)
  and closes early when the next item would not fit the size budget. Batches form over every item in
  an order fixed by each unit's file and lines (by content for an item without them), so the same
  units form the same batches whatever order a caller passes them in, and a request carries all
  its batch mates even when some of their questions were answered before.
- **Answer store.** Every answer is stored with the served model and the thresholds in force. Jev's
  answer about one item changes with the other items in its request, so an item answer is reused
  only when the item, its batch mates, the shared state, the question with its wording hash and the
  served model all match. A route's refusal of an exact request for its input size is stored too, so
  a replay splits that request again without sending it; until the first live answer of a run the served model is unknown, and
  unknown counts as a miss (or pass `served_model=`); with a store, a first `check_each_async` then sends its
  first batch alone, and the batches after that answer replay as usual. `ReplayOnlyClient` replays
  from the store and never calls Jev.
  `JsonlAnswerStore` is one run's pack. `SqliteAnswerStore(path)` is one insert-only store shared by
  every run on a machine, so a repeated run at the same commit asks nothing again but the request
  that learns the served model. It never holds
  code, state or question text: only hashes, unit locations, batch member ids, the batching rule and
  size, the model, raw answers and timestamps. Its location and retention (no expiry) are provisional;
  `LayeredAnswerStore(pack, shared)` reads the pack first, copies every answer it finds only in the
  shared store into the pack, and writes new answers to both, so the pack alone still replays the run.
- **Journal, separate from the store.** Pass `journal=` (any object with `record_request(request) ->
  request_id`, `record_response(request_id, response)` and `record_failure(request_id, error,
  response)`). The judge records the masked request before dispatch and the raw response before
  parsing, as a `RawResponse(body, status, content_type, decoded)`: the body bytes as received, the HTTP
  status, the content type and `input_tokens`, the count the provider reported or the text
  `not reported`; a missing count is never written as 0. Transport errors and responses that fail to
  parse are recorded as failures. Clients that offer `send` and `parse` return that `RawResponse`; the TypeSafe adapter
  captures the exact bytes from its HTTP transport. A client that only parses is journaled with its
  decoded JSON and `exact=False`. `request_sha256` never includes the model; cache reuse checks the
  served model separately. A request holds code, and the library cannot know whose code it is, so
  `JsonlJournal` keeps only the request hash, the question ids and a state hash by default; pass
  `keep_request_text=True` only for your own or open-source code.
- **No client code in the store.** Request text is kept only with `keep_requests=True`, which is for
  your own or open-source code (for example a frozen evaluation set). Such a record keeps the request
  twice: `request`, written with sorted keys for reading, and the body as it was sent
  (`sent_body_base64`, with `sent_exact` true when the TypeSafe adapter captured the wire bytes).
  Jev can answer the two orders differently, so ask a stored request again only from
  `record.sent_request()`, with a judge that has no store. `JsonlJournal(keep_request_text=True)`
  likewise keeps the body as handed to the client (`body_base64`) and the wire bytes when captured
  (`sent_body_base64`), and `export_for_review` keeps the order the request is sent in. By default the store keeps
  hashes, question wording, and each item's file, lines and commit, so
  `rebuild_request(record, CodeIndex.at_commit(...), shared)` can rebuild the exact request and prove
  it matches, or name the part that differs.

## Layer 3: directives

`find_code(index, judge, target_description, start, *, budget=SearchBudget(), thresholds=None)` is
the central search. Use it only when the target is described by meaning; anything code can decide
(the callers of X) is an operation. For each opened place, a request asks "Does `slice.code`
contain the code described in `target.description`?" and, per neighbour code lists (callers, with
callers in test files after the others; callees, proven production targets first and then the ones
called from fewest places; code that
refers to it or that it passes on without a call, as an argument, collection entry, assignment,
decorator, export, return, method receiver or type; the modules it imports, re-exports or requires
(module-level code takes its whole file's imports): the definitions of the names it takes from each,
and the start of a module it takes whole or takes names from that it does not define itself; the
other functions of its file, nearest first; lines anywhere in scope (docs and config too) that
mention its environment variables or its quoted keys (six characters or more with a dot,
underscore, colon, slash or dash), the
rarest key first, skipping a key found on more than 30 lines; co-changed files; and the lines before and
after it), whether the target could be inside it. Places that open the same lines of the same file are
listed once, whatever move found them, and a place wholly inside the opened code is not listed;
identical code in two files stays two places. A line outside any function opens its class or
module-level declaration when that has at most 120 lines; in a longer one it opens the window around the
line under the definition's name. Either way the moves can follow that name. Callees and passed-on
definitions are also offered from anonymous functions and windows. For an anonymous nested function,
same-file navigation first offers the nearest named containing symbol, else the nearest containing
one (a callback inside a test's callback offers that test). By default the finite,
deduplicated frontier decides when the search is complete: depth, step, call and per-move neighbour
limits are `None`. A caller can set any of those fields on `SearchBudget` when it has an explicit
operational limit. Each round opens
`beam_width` places concurrently: start places first, then the neighbours Jev picked to open next, in
the order it picked them, then the other neighbours by falling `could_contain` probability, with a
visited set and a content cache. An `open_first` Choice picks the neighbour to open next, with the
option "None of the entries is likely to contain it."; every pick but "none" waits ahead of the scored
neighbours, whatever its confidence and its own score, and the history's `used` says that it was queued.
Large openings split independent neighbour questions through the same Judge batching owner without
discarding candidates or previews. The global pick is optional: when its full request or option set
exceeds provider capability, `open_first.unavailable` records why and individual neighbour scores
still order the complete frontier. Every live sub-request counts toward the selected call allowance.
Once the allowance has refused a request, the search keeps opening places only while the answer store
still answers them; the first round that gets no answer at all ends the search as `budget`, and the
places it did not open stay in `not_inspected` for Resume.
Only HTTP 400 with `detail.error_type` equal to `max_tokens_exceeded` is a size refusal;
mentions of that text in question IDs or unrelated error messages do not trigger splitting.
A low neighbour score only lowers that neighbour's priority; it is never treated as proof that the code
is not there. The search runs out of places when no start or pick waits and no neighbour scores
above the no bar (0.20 by default). It then ends as `nothing_left` only if its own moves parsed every
code file in scope without a grammar error; otherwise it ends as `scope_incomplete`. The remaining
files are never parsed just to choose the label. Of `FindResult.code_files`, `files_judged` counts the
files in which Jev judged code (the opened places, not whole files) and `files_read` adds the files
read only to list neighbours; the CLI prints all three, for example `scope_incomplete (not found: Jev
judged code in 1 of 7 files; 4 more were read only to list links; 2 never reached)`. A start place is judged but never ends the search as found, because
the caller already had it; `FindResult.starts` keeps each start with its verdict. Each neighbour's
signature names its file and lines: a function quotes its first line; a window around a call, reference
or key outside any function gives its line range and quotes that line; a stretch chosen by position (the
lines before or after, the start of a co-changed or imported file) gives its range and quotes its first
line of code, past blank lines, comments, a license banner, a `'use strict'` directive or a module
docstring. The outcome is `found`, `stop_rule`, `budget`, `nothing_left`, `unsure_only` or
`scope_incomplete`, and the result keeps three sets: `found`; `searched` and `unsure` (bodies actually
judged, start places apart in `starts`); and `not_inspected`, each entry with its reason (`budget`,
`deprioritized`, `capped` or `depth`) and its `QueueTier`: `START`, `PICK` or `MOVE`. Resume
preserves that role, so waiting starts still open before picks and are never reported as new finds.
`searched` means "opened and judged at or below the no bar, probability kept", and `nothing_left`
means "nothing left worth opening in a scope the search parsed whole"; neither proves that the code does not exist, because one "no" about
one place can be wrong. When nothing reaches the yes bar, rank the opened places by their
`contains_target` probability: the best-scored place is the likeliest one. Pass the result back as
`resume=` to continue from that frontier with a fresh budget. Pass `commit=` to require that the index
holds exactly that revision (use `CodeIndex.at_commit` for history); a mismatch raises
`RevisionMismatchError`. Nothing escalates on its own. Library depth, steps, and calls are unlimited
by default, with a beam of 3. The CLI sets a default allowance of 24 model requests for Find and
48 for Find All. Everything is a parameter: `SearchBudget` also sets
`neighbours_per_kind`, `preview_lines`, `max_line_chars` (240: longer lines and signatures are cut and
marked "[line cut]") and `max_slice_chars` (12,000: an opened place is cut on a line boundary with a
note, and `Visit.code` ends at the last shown line). If the first line cannot fit, the place stays
`not_inspected` with reason `budget`; Resume with a larger slice budget inspects that same source.
`questions=SearchQuestions(found=...,
could_contain=..., open_first=None)` replaces the wording. `moves=` chooses how neighbours are listed: the default
`places.MOVES` maps each move's name (`callers`, `callees`, `referenced_by`, `passed_on`, `imported`,
`same_file`, `keys_mentioned`, `co_changed`, `lines_before`, `rest_of_file`) to a function of the
index and the opened code that returns places. Pass a subset, or add a function of your own; `MOVES`
itself is read-only. `FindResult.moves` and the final `stop` step name the moves a search used, and
`context_for_comment` takes `moves=` too. The directives take their check (`check=`) as a parameter too.

Directives on top: `context_for_comment` and `find_similar_code`. The library finds code; answering
questions about that code (is a comment accurate, does a claim hold) is a layer you build on top. A new
use case is your own function of 30 to 60 lines that composes these pieces; see
`directives/similar.py` and `docs/extending.md` for the pattern.

## History: typed steps, read through named sections

`jev_navigator.history.History` is an append-only list of `HistoryStep(operation, arguments, fetched,
judgments, decision)`; each `FetchedSpan` keeps its source. Jev never gets the list itself: a check
selects named sections and `history.state_for(names)` builds exactly that state.

| Section | Holds |
| --- | --- |
| `fetched` (default for history checks) | every code body fetched, with its file, lines and commit, and nothing else |
| `history` | `{"steps": [...]}`: each step's operation, arguments and fetched code, without judgments or decisions |
| `decisions` | every step without code: judgments with probabilities, candidates, choices, places set aside |
| `previous_judgments` | the last answer of each history check, with its probability |
| your own | declared with `History(sections={"subject": ..., "shown_code": ...})`, updated with `set_section` |

The default is `fetched`, so a history check never leans on the search's own verdicts; the
`history` section carries no verdicts either. A check that is meant to read them selects `decisions`
explicitly. An unknown name raises `UnknownSectionError`. Each section has its own `SectionLimit(max_entries, max_chars)`
(newest entries kept, long text cut; defaults in `DEFAULT_LIMITS`), applied before the character budget.
Text limits also apply inside nested lists and mappings. Rendering a limited view preserves the
complete code and judgments in the append-only record.
The budget is a character box, capped at Jev's documented 32,000 tokens for state plus the longest
question times 2.4 characters per token (the Engine's `REQUEST_CHARS_PER_TOKEN`), 76,800 characters
(the Engine measured 32,883 tokens accepted and about 33,200 refused on 27.09.2026). A whole request
may reach the documented 64k tokens, 153,600 characters. The batching owner (`check_each`,
`check_every`) and the `find_code` opening questions measure the same boxes before sending and split
what would exceed them; a direct `Judge.ask` sends what it is given and relies on the provider's
refusal. When the selected sections still do not fit, the
pluggable `evict` policy trims them; the default `drop_oldest_code` replaces the oldest code bodies with
`[evicted]` and records each eviction in `history.evictions`. A check that reads no code never evicts.
Pass `recorder=` (for example a `JsonlJournal`) to record every appended step; the recorder gets each
step without code bodies, only their sources and hashes.

Whether the history holds what you need is your own concrete check, asked with
`judge_history(judge, history, check, shared, sections=("fetched",), exhausted=False)`: yes is `found`,
no is `searched_not_found`, and unsure is `continue`, or `not_inspected` once your budget is exhausted,
never "absent". Name a concrete property ("Does `fetched` contain code that compares the number of
items with a limit?"), never "is it enough". `judge_sections(judge, history, {name: HistoryCheck(check,
sections)})` asks several checks: those selecting the same sections share one request, different
selections run in parallel. `find_code(..., stop_rule=StopRule(check, shared, sections=..., context=...))`
applies such a check after each round (off by default); `context` adds your own sections, and the
history always declares `subject` (the target description). The docs warn that accuracy falls as
unrelated state grows, so measure first: `ceiling_curve(judge, recorded_steps, check, sections=...)`
replays a recorded search with a growing history and reports the probability at each size.

`find_code` always records its own history in `FindResult.history`, with or without a stop rule; it
costs no calls. A `choose_next` step lists the places opened next, each with its priority and reason
(`start`, `open_first` for a place Jev picked, or `queue_score`). An `open` step holds the
code, the `contains_target` probability and verdict, every neighbour offered with its `could_contain`
probability, the `open_first` pick, and places set aside (`capped` or `depth`). A final `stop` step
names the outcome, the not-inspected frontier with reasons, and the last stop check, so the history
and the result agree. Without a stop rule nothing reads the history; with one, the stop check reads the
sections it selects (by default only the fetched code). `HistoryStep` is generic: append your own steps (an agent's tool call and result) the same way.

## LlmStep: an LLM call you add yourself

Nothing in the library calls an LLM. `LlmStep` is a building block a user adds to their own
directive, and the user defines all four parts:

```python
from jev_navigator.llm_step import LlmGuard, LlmStatus, LlmStep, PickFromOptions
from jev_navigator import connectors

phrase_step = LlmStep(
    name="phrase_fallback",
    when=lambda result: result.confidence < 0.70,  # when: the phrase Choice was unsure
    context=lambda result: {"comment": comment, "slice": code_state, "options": phrases},
    answer=PickFromOptions("Which phrase names what the code does?", answer_field="phrase"),
    connector=connectors.pi("your-model"),  # any CLI or OpenAI-compatible endpoint
    guard=LlmGuard(store_path=path, max_calls=5),
)
call = phrase_step.run(judge.pick(phrase_pick, phrases, state))
if call.status == LlmStatus.ANSWERED:
    selected_phrase = call.answer
```

`answer` can be `PickFromOptions`, `JsonContract(instructions, required={"accurate": bool})`, or any
object with `render(context, parse_error)` and `parse(reply, context)`. A reply that does not parse
is retried once with the error. Connectors: `hermes`, `pi`, `OpenAICompatibleConnector`,
`CommandConnector`, and `claude`. The result distinguishes `not_requested`, `budget_exhausted`,
`answered`, and `parse_failed`; `attempts` counts actual provider calls. Provider exceptions propagate
after their failure is recorded.

The guard masks context, refuses a prompt with a secret, and enforces an optional budget. When a store
is configured, it records the attempt identity, connector, model, and prompt hash before dispatch;
the exact reply before parsing or retry; and each parse outcome or provider failure. A final summary
records the result and attempt count. The first malformed reply survives a successful retry.

> **Warning:** `connectors.claude()` runs Claude headless. Every run spends from your Claude plan or API
> budget, and an automation can start many; enable it deliberately and set `LlmGuard(max_calls=...)`.

## Before any live call: review the exact request

Run your function once with `CapturingJevClient` (from `jev_navigator.judgments.review`); it never
calls Jev and answers every question neutrally. Write the captured, already-masked request with
`export_for_review(state, questions, intended_uses, path, case_id=..., group_id=...)`: one JSON file
with the request and, per question, what code does with the answer. Read it, or pass it to a
question-review tool, before any paid call, and pilot a small set of cases first.

## Register a benchmark before running it

`judgments.round` freezes the ordered case IDs, exact questions and scoring rule before
model calls. Keep the generated `FROZEN.txt` in a trusted versioned record; the local hash
chain detects edits beneath that anchor, not replacement of the entire chain.

```python
from pathlib import Path
from jev_navigator.judgments.round import RoundRegistration, freeze, verify, registered_request_sha256

registration = RoundRegistration(
    case_ids=("order-limit",),
    questions={"found": {"type": "noul", "instructions": "Does the supplied code enforce the order limit?"}},
    rule={"yes_at": 0.9},
    library_commit="",  # Supply the verified navigator revision when known; empty means unknown.
)
round_dir = Path("jvn-results/order-limit-round")
freeze(round_dir, registration)  # Creates the directory; refuses to overwrite a frozen round.
verify(round_dir, registration)  # Call before scoring stored answers.
request_hash = registered_request_sha256(registration, {"code": "..."})
# store.by_request(request_hash) retrieves the answer for this exact state/question identity.
```

Pass `rule_source=inspect.getsource(rule_function)` when code implements the scoring rule.
An optional `verifier_report=Path(...)` is copied into the round and its content is checked by
`verify`; a missing supplied report is an error. `library_commit` is caller-supplied provenance;
`checkout_commit` and `uncommitted_changes` describe the working directory at freeze time.
Request identity uses the same built-in secret masking as `Judge`. If the caller supplies a
custom masker or disables masking, pass that same `masker` to `registered_request_sha256`.
Neither a question hash nor a frozen manifest proves model quality or dataset completeness.

## Tests

`uv run pytest --basetemp=<scratch dir>`. Tests run offline against small real git repositories and
`ScriptedJevClient`. The suite retains the ten Express/Next.js and FastAPI/GraphQL graph
capability regressions and verifies request/response capture through a real local HTTP socket.
The TypeSafe adapter's tests run only with the extra installed:
`uv run --extra typesafe pytest`. Run `uv run ruff check src tests` and
`uv run ruff format --check src tests` before pushing. Local checks are the normal validation
path for this small library; pushes and pull requests do not launch hosted CI. The `tests`
workflow is available through GitHub Actions **Run workflow** when an explicit cross-version
check is needed (Python 3.11 and 3.13, each with and without the TypeSafe extra).

## License

MIT, see [LICENSE](LICENSE).
