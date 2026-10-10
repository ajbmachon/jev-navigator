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

## Architecture: blocks, mini-workflows and configurations

JVN is a library of building blocks for searching code. Each level composes the one below it, and
every use of JVN, the `jvn` command included, is a composition of the same blocks.

1. **Code primitives** establish facts without a model: the files in scope, definitions, callers,
   callees, references, imports, text hits, units and git history
   ([Layer 1](#layer-1-index-operations-and-comments-no-model)).
2. **Mini-workflows** compose primitives and Jev judgments into one kind of search: `find`
   (`find_code`), `find_all`, `trace`, `find_text` and `find_all_text`
   ([Layer 3](#layer-3-directives)).
3. **Configurations** (being built) compose mini-workflows into a larger workflow. A configuration is
   typed: it names the mini-workflows, their order, their inputs and their budgets. `jvn search`, also
   being built, is to be the default configuration.

Between two stages a configuration can place a **Jev step**, one bounded decision such as a yes or no
check or a pick from a list code built ([Layer 2](#layer-2-judgments)), or an **LLM step**, generation
over an open space ([`LlmStep`](#llmstep-an-llm-call-you-add-yourself)). Which steps run is
configuration: a recipe the caller passes as data names them, never an environment or deploy flag.

| Block | Status |
| --- | --- |
| Index, operations, units and scope (`CodeIndex`, `operations`, `index.units`, `resolve_scope`) | built |
| Jev judgments (`Check`, `Pick`, `Rate`, asked through `Judge`) | built |
| Mini-workflows `find_code`, `find_all` and `trace` | built |
| The frontier: the order a search judges what its sources reach, a named policy, `STAGE_ORDER` or `VALUE` (per-target queues and shares, settling after one step of hops); see [Sources, the frontier and each workflow's composition](#sources-the-frontier-and-each-workflows-composition) | built |
| `LlmStep` | built |
| Text search: the mini-workflows `find_text` and `find_all_text` | built |
| Sources: one contract (`sources.Source`) for every primitive that reaches candidates; `find_all`, `find_all_text` and `find_text` are compositions of them | built |
| `find` and `trace` as compositions of sources | not yet: they keep their own moves and call graph |
| The spelling map | being built |
| Static context configuration: one-hop proven calls, directly named files and structural excerpts (`selection.context`) | built |
| Typed configurations | being built |
| `jvn search` | being built |

The spelling map is an index block. It splits every identifier, file name, config key and string
literal into word parts and normalises case, separators and plural, so all spellings of one name
share a key: `Website`, `website`, `websites`, `web_site` and `website.ts` meet. A name lookup then
returns every real spelling and its locations, rarest first.

**Sources** are the primitives a search's candidates come from, all under one contract, and a
mini-workflow is a composition of them: the sources that start it, the hop sources a unit that clears
a target's bar expands through, the frontier's policy and shares, and Jev judging in queue order. See
[Sources, the frontier and each workflow's composition](#sources-the-frontier-and-each-workflows-composition).

Three rules hold for every change:

- A capability that is not about one caller's domain is a block that any caller can use. A caller
  configures blocks and passes its own inputs; it never reimplements a block.
- Nothing specific to findings, themes or the Analysis Engine lives in JVN. The Engine's evidence
  pack, a theme agent's search tool and a coding agent's search are each a configuration plus that
  caller's inputs.
- An experiment compares named configurations on an evaluation set, never tweaks inside one call.

Until typed configurations exist, a composition is a plain function of the blocks; see
[docs/extending.md](docs/extending.md).

## Trace a known workflow

```sh
jvn trace "how an order request becomes an HTTP result" --start app/orders.py:42
jvn --json '{"command":"trace","target":"order request to HTTP result","start":["app/orders.py:42"]}'
jvn help trace
```

Trace follows static relationships and batches atomic evidence judgments. It preserves uncertain
bindings, partial coverage and request evidence in its [run folder](#where-jvn-keeps-runs-and-caches). A positive judgment is evidence,
not proof of a complete path. Use `find` first if the starting function is unknown.
See [trace options and outputs](docs/cli.md#workflow-trace).

## Find every matching function

```bash
jvn findall "functions that reject an order exceeding the item limit"
jvn --json '{"command":"findall","target":"functions that reject an order exceeding the item limit"}'
jvn help findall
jvn schema findall
```

`find` locates an implementation; `findall` finds a seed, then judges every unit in scope (each
function, method, Prisma schema block and file's top-level code), the units holding the seed's found code in its first
wave of requests. It uses batched Jev judgments and defaults to 48 live model calls (twice `find`);
`--max-calls none` lifts that cap. There is no file cap. Reports, source provenance and request journals go to a unique
[run folder](#where-jvn-keeps-runs-and-caches). `units_examined` describes coverage of the units in
scope, not a proof of semantic equivalence or completeness. Uncertain answers and
unreadable or unsupported source stay visible. At a call stop, the terminal offers another allowance.
For a later invocation or an agent pipeline, pass `--resume` with the folder the earlier run printed, and the same
Find All query and scope. Completed judgments and the seed are retained; only unfinished work spends
new model calls. `findall` judges code only. The files JVN does not parse, such as YAML, JSON,
Markdown, TOML and config files, are searched only when a caller asks for them, through the library's
`find_all_text` and `find_text` ([extending.md](docs/extending.md#text-units)); env files are never
read, though an env template (`.env.example`, `.env.sample`, `.env.template`) is read masked.

For an engineer-authored library composition and its limits, see
[Extending: judge every unit with Find All](docs/extending.md#judge-every-unit-with-find-all).

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

JVN limits the child processes it starts. Each host reserves 1,024 MB for its ast-grep, ripgrep,
git and command-line connector processes together, with an 8,192 MB shared child-process ceiling.
It waits up to two minutes for room, then raises `MemoryLimitReachedError`. The embedding host or
CLI launcher owns the Python process's total memory, including indexes and other caches. JVN cannot
attribute a shared process's footprint to individual libraries. See
[Memory limit](docs/cli.md#memory-limit) for the settings and refusal behavior.

## Live evidence-pack command

Start in the directory you want to search:

```sh
jvn find "the check that limits how many items an order may have"
```

In a terminal, reaching the call budget offers another allowance without losing the saved search.
JSON and piped commands return partial results without prompting; continue them with `--resume`.

That is enough. `jvn` chooses an entry point and creates a unique evidence pack in its
[run folder](#where-jvn-keeps-runs-and-caches), never inside your project. It works with uncommitted
changes and ordinary directories outside Git.
`find` follows code relationships to locate a match; it does not promise every matching function
or a complete end-to-end trace.

To search another directory, add just `--repo`:

```sh
jvn find "where do we reject evidence quotes that are absent from the source?" --repo /path/to/repository
```

Explicit `TYPESAFE_API_KEY` and `TYPESAFE_BASE_URL` process values win independently. Otherwise
`jvn` reads those settings from `~/.config/jvn/env`, a file in dotenv syntax that it parses itself;
it does not execute that file or print the values. `TYPESAFE_BASE_URL` is the API root before
`/v1/systemone`, such as `http://127.0.0.1:4777/jvn` for a gateway serving `/jvn/v1/systemone`.

When its code runs from a jev-navigator source checkout (`uv run jvn` there, or an editable
install), `jvn` first fills what is missing from that checkout's `.env` (see `.env.example`). Any
install into site-packages (`uv tool install`, `pipx`, a non-editable `pip install`) reads no
`.env`, and when the directory it runs in holds one, it says on stderr that it did not read it. It
never reads a `.env` from the directory or repository it searches, unless that is the checkout
its own code runs from. A settings file can set only
`jvn`'s own `TYPESAFE_*`, `JEV_NAVIGATOR_*` and `SYSTEM_ONE_*` names; `jvn` names on stderr any
other name it ignores, never its value. The `JEV_NAVIGATOR_*` settings hold the judgment thresholds
only; a search's budget comes from its flags or the request's JSON fields.

### Decision-model routes

`SYSTEM_ONE_ROUTES` names the decision models `jvn` asks, in order. With
`SYSTEM_ONE_ROUTES=drex,jev`, every request goes to Drex first, and to Jev only when Drex fails; the
journal records each attempt with the route that made it. A size refusal is not a failure: it goes back
to the judge, which splits the request, because the next route would get the same request. Each route
reads `SYSTEM_ONE_<NAME>_ENDPOINT`, `SYSTEM_ONE_<NAME>_MODEL` and `SYSTEM_ONE_<NAME>_API_KEY`; `drex`
and `jev` also take `SYSTEM_ONE_<NAME>=1` for their hosted endpoint and model. Only the `jev` route
falls back to `TYPESAFE_API_KEY`: every other route needs its own key, so your TypeSafe key never goes
to Drex or to a server you configured, and a table without a `jev` route, such as
`SYSTEM_ONE_ROUTES=drex`, runs with no `TYPESAFE_API_KEY` at all. A route missing its endpoint, model
or key stops the command before any request, naming the route and the setting. Without
`SYSTEM_ONE_ROUTES`, `jvn` uses the default Jev client described above.

Drex accepts a stricter request than Jev, so the `drex` route sends what the judge built in Drex's
form: each structured criterion goes as its JSON text, its fields still labelled, and a zero-width
space goes after the `data` of any base64 data URL (`data:image/png;base64,…`), which Drex would
otherwise refuse as media. The judge hashes and stores the request it built; the journal's sent body
records what Drex received. Every other route gets the request exactly as built.

Each route has an input limit for the state plus the longest question: Drex accepts 8,192 tokens and
Jev 32,000, as Analysis Engine measured them; `jvn` turns tokens into characters at the one rate
`REQUEST_CHARS_PER_TOKEN` in `judgments/client.py`. Any other
route sets its own with `SYSTEM_ONE_<NAME>_INPUT_TOKENS`, or the command stops naming that setting.
`jvn` packs every request to the smallest limit in the table, so whichever route answers can take it,
and remembers a size refusal under the limit of the route that refused. Point Drex at `jvn` through
the route table: the default client always packs to Jev's limit.

Each route also has its own concurrency: how many requests it receives in flight at once. Drex admits
2 (it answers HTTP 429 to a third) and Jev takes 32, as Analysis Engine measured them; any other route
sets `SYSTEM_ONE_<NAME>_CONCURRENCY`, or the command stops naming that setting. A request waits for a
free slot of the route it goes to before it is sent, so waiting never counts against its timeout, and a
request that falls back to Jev is not held back by Drex's limit.

One difference under routes: Ctrl-C cannot abort a request already in flight, so the command waits for
those requests to finish, keeps their answers, and then stops with a resumable pack. Without routes,
Ctrl-C aborts requests in flight.

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

`command` defaults to `find`, `repo` defaults to the current directory, and output goes to a new
[run folder](#where-jvn-keeps-runs-and-caches) unless you supply `out`. JSON mode prints one result object on stdout with
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
Each file option shows the file's first doc line and up to eight names: the functions and classes the
module names or exports through CommonJS and each module-level constant whose call or `new` builds a
function, as `run` in `export const run = Effect.fn("run")(function* ...)` or `userRouter` for a
router, but not one that builds data through a callback, such as `items.map((item) => item.id)` or
`new Map(...)`, then each function of an object a module-level variable holds,
as `api.list`, each function of an object a module-level call or `new` is passed, as `errorFormatter` in
`create({ errorFormatter() {} })`, and each member of a namespace, in file order within each group. A
file holding none of these, such as one of types only or one whose functions are all callbacks, shows
no names.

Every live call is a paid request, so `--max-calls` defaults to 24 for the whole run, choosing an entry
point included; a search that reaches it ends with outcome `budget` and a resumable `not_inspected`
frontier (or a saved entry-selection stage if the cap, Ctrl-C or a failed request arrives earlier). Resume with another `jvn find`
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

Time is never a budget. Two runaway guards, set in `runaway_guards.py` (André, 06.10.2026), stop only a
search that would otherwise hang: a file whose parse runs past 30 s is read as text instead (see
[Choosing the files a search covers](#choosing-the-files-a-search-covers)), and a `find_code` search
still running after 7 minutes ends `runaway`, with every unopened place listed and resumable like
`budget`. A library caller can pass its own `ceiling_seconds` to `find_code` and `find_code_async`.

An explicitly selected output directory must be new or empty. Each evidence pack contains:

- `manifest.json`: schema version, navigator build fingerprint and source revision, inspected
  repository revision, explicit budget and thresholds, requested and served model, elapsed time,
  versioned code locations (`path:start-end` with file hashes; neighbours as `path:line name`), raw
  probabilities, full search history, uninspected frontier, and unparsed files.
- `report.md`: a readable outcome, source table, found locations, and coverage caveat.
- `journal.jsonl`: request hashes and exact provider responses as the run progresses.
- `answers.jsonl`: reusable typed answers keyed by source and request hashes. Every answer is also
  written to the machine's shared answer store (`$XDG_CACHE_HOME/jev-navigator/answers-v2.sqlite`,
  `~/.cache` when the variable is unset or relative, or `JEV_NAVIGATOR_ANSWER_STORE`), which holds no code; a later run at the same commit asking the
  same questions replays from it after the live requests that learn the served model (one for Find
  All and Trace, one per place a Find's first round opens, up to `--beam-width`; Find All and
  Trace items carry the commit and file hashes, so a new commit asks again), and copies what it replays into its own
  `answers.jsonl`. Every pack reports those answers as `provider.replayed_answers` beside its live calls.
  `--answer-store PATH` points a run at another store file; each run prints the store it uses.
- `resume.json` (budget-stopped, cancelled or failed runs): the frontier as locations; Resume re-reads the
  code from the unchanged repository.

By default the manifest, report, journal and resume state hold no source code, only locations and
hashes; a relation that quotes a mentioned key reads `mentions a key (path:line)`. `--keep-requests` (JSON `"keep_requests": true`) also keeps the code and full neighbour
signatures in the manifest and report and the exact request text in the journal; use it only for
your own or open-source code. The repository includes only a small public-format sample under
[`examples/evidence-pack`](examples/evidence-pack).

### Where JVN keeps runs and caches

JVN never writes into the project it searches or the directory you start it in, unless you name a
folder with `--out`. Without `--out`, a run's evidence pack goes to its own run folder,
`$XDG_DATA_HOME/jev-navigator/runs/<directory>-<timestamp>` (`~/.local/share` when the variable is
unset), and the run prints that path.

Caches live in `$XDG_CACHE_HOME/jev-navigator` (`~/.cache` when unset): the fact cache (`facts/`), the
name table (`names/`) and the shared answer store (`answers-v2.sqlite`). A host that keeps each
tenant's caches apart sets `JEV_NAVIGATOR_CACHE_HOME` to that tenant's folder, which then holds
them directly; a relative path in either variable is ignored. Caches are the data JVN
values most, but only while they represent real files, so JVN cleans up after itself:

- Facts or a name table another JVN version wrote, which this version can never read, go once no
  JVN version has used them for 3 days. Versions in use side by side keep theirs. A default answer
  store in an older layout holds paid-for answers, so it stays until unused for 30 days.
- A cached file's facts or names go once no run has met that exact file content for 30 days.
- An answer in the default shared store goes once no run has reused it for 30 days, with its item
  answers and refusals. A store you name with `--answer-store` or `JEV_NAVIGATOR_ANSWER_STORE` keeps
  every answer and is never touched; it must lie outside the cache folder, so a run naming a store
  inside it stops with exit status 2.
- A run folder goes 14 days after its run started, or 30 days while it can still be resumed (it holds
  `resume.json`). A folder you name with `--out` is never touched.
- Above the disk budget, 5 GB unless `JEV_NAVIGATOR_DISK_BUDGET` says otherwise (`750MB`, `20GB` or
  plain bytes), the oldest run folders go first, then other versions' facts and name tables, then the
  least recently confirmed facts and names, and answers last, older layouts first.

Every `find`, `findall`, `trace` and `stats` run applies these rules as it ends, at most once a day,
deleting at most 2,000 files per run; a failure to clean up is a notice on stderr and never fails the
run, and Ctrl-C during the cleanup, which starts only once the run has ended, stops it with one
notice and exit status 130. Nothing outside these two folders is ever deleted, and links are never followed.
`jvn cache status` shows what each store holds and what each rule would remove; `jvn cache prune`
applies every rule now.

A program that uses JVN as a library (`CodeIndex`, `find_code` and the other blocks) fills the same
caches but never cleans them up, because only a CLI run ends with housekeeping. Such a host runs
`jvn cache prune` itself on its own schedule. A host serving several tenants composes the two
variables: it gives each tenant's searches `JEV_NAVIGATOR_CACHE_HOME=<that tenant's folder>`, and
prunes each folder with the same variable and the share of disk it allows that tenant in
`JEV_NAVIGATOR_DISK_BUDGET`:

```bash
JEV_NAVIGATOR_CACHE_HOME=/volume/jvn-cache/tenant-a JEV_NAVIGATOR_DISK_BUDGET=10GB jvn cache prune
```

## Layer 1: index, operations and comments (no model)

```python
from jev_navigator.index.code_index import CodeIndex
from jev_navigator import operations, comments
from jev_navigator.index import units

index = CodeIndex.from_directory(repo_root, prefixes=("app/", "web/"))  # tracked or not, minus ignored
index.not_indexed_files  # {"node_modules/": "ignored", ...}: every file or folder left out, with the reason
tracked = CodeIndex.from_git(repo_root, ["app/orders.py"])  # only what git tracks; the rest is not_indexed
old = CodeIndex.at_commit(repo_root, "abc123", prefixes=("app/",))  # from git objects, checkout untouched
old.close()  # removes at_commit's private copy; `with CodeIndex.at_commit(...) as old:` closes it too
index.find_definition("LIMITS_KEY")  # functions, classes, constants, assignments, types, enums
index.find_callers("validate_order")  # CallSite(file, line, caller, binding), found by name
index.callee_edges(span)  # CallEdge(name, line, binding); find_callees gives names only
index.find_references("send_invoice")  # Reference(name, file, line, role, holder, binding): non-call uses
index.references_in(span)  # names a function passes on without calling (callbacks, registries)
index.enclosing_symbol(file, line)
index.symbols_in(file)
index.decorator_starts_in(file)  # each decorated function's span and its first decorator line
index.stubs_in(file)  # functions whose body is only ..., pass, a docstring or raise NotImplementedError
index.read_slice(span)
index.read_window(file, line, radius=10)
# ripgrep over the narrowed files only, every hit in file and line order
index.search_text("orders.max_items")
index.search_text("orders.max_items", max_hits=30)  # only the first 30 hits
index.imports(file)
index.dependents(file)
index.co_changed_files(file)

# outermost functions and methods, Prisma schema blocks, top-level code; room: docs/extending.md
units.list_units(index, files, box_chars=room)
# the text units of the files JVN does not parse
units.list_units(index, files, box_chars=room, reading=units.Reading.TEXT)
# the units holding lines or line ranges
units.resolve_anchors(index, [units.LineAnchor(file, line)], box_chars=room)

operations.slice_around(index, file, line)  # the enclosing function, or a window
operations.code_described_by_comment(index, file, line)  # the whole next symbol or block
operations.callers_of_file(index, path)
operations.trace_callers(index, symbol)  # and trace_callees; optional depth, otherwise fixed point
operations.trace_graph(index, index.find_definition(symbol))  # calls and non-call references
operations.similar_functions(index, symbol)
operations.code_named_in_doc(index, text)  # definitions of the names mentions.code_names_in finds
# the scope files texts or anchor files name by path or run with python -m: NamedFiles(code, text, named_by)
operations.files_named_by(index, texts, anchor_files)

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
pass. The pass runs a few hundred files per ast-grep process
and turns each match into its fact as ast-grep prints it, so memory holds the facts, never the
parser's output, and no command line outgrows the system's argument limit. Calls are ordered by
where they start in the file, and of two calls starting at one place (`new Foo(a).bar()` and
`new Foo(a)`) the outer comes first, so every run returns them in the same order; symbols spanning
the same lines keep the order they start in.
Exact-name lookups (definitions, callers, call counts and references) read the persistent name table
in `$XDG_CACHE_HOME/jev-navigator/names`, which ties every name to the lines it sits on in each file
content. A file's content is identified by its git blob id, taken from the Git listing for a clean
tracked file and hashed from its bytes otherwise (also when its bytes differ from the listed blob, as
on a checkout that converts line endings), so a new index maps its files to table rows without
reading them, and a warm lookup starts no text search and parses no file. The first name lookup of an
index covers its whole scope: each file the table lacks is read from the fact cache, or parsed, and
its rows are written. A changed file gets new rows under its new content, a file deleted before the
first lookup answers none, one deleted later is reported unavailable and proves nothing, and a change
to the parser or to any language's rules starts a new table. When the table is warm but the fact
cache is not (after a change to the fact rules, or after housekeeping pruned it), callers and
references load the facts their bindings read, the files of the uses and of the definitions, in one
scan instead of one per file. `definitions_in(file)` reads one file's
definitions from the table. The table holds names and line numbers, never code. A file counts as read
in a Find's counts only when navigation reached it, never because the table covered it. A call's or
argument's receiver, in the table and in the cached facts alike, is kept only when it is a plain chain
of names such as `this.store`; any other receiver (`client("k").fetch`, `cfg["token"].get`) is
recorded as `<expression>`, so no string literal is ever stored. Opening a known span parses its file directly. ripgrep, which
`search_text` runs, always runs with `--no-config`, so a `RIPGREP_CONFIG_PATH` file can neither
change what the index sees nor run a preprocessor over the searched repository. The resulting
per-file facts are cached by source bytes, language, ast-grep version, the rule text and the source
of the code that runs ast-grep and reads its matches, in `$XDG_CACHE_HOME/jev-navigator/facts` (`~/.cache` when the
variable is unset or relative), so a new index can reuse facts without treating changed source or changed
parser rules as current. A file that changes on disk after the index first read it is
reported as unavailable when the index reads it again, and its code still reads as the text the
index first read, the text its SHA-256 names, never in its new form. The index keeps each file's
first read compressed for the run, about 2 MB per 1,000 files of Heedvane's web app. Each call
site's binding is computed once, and `search_text` and `co_changed_files` each run their tool once
per argument for the life of the index. The index keeps the lines of a bounded number of recently
read files (`LINE_CACHE_FILES`). Every cache an index keeps lives in the index itself and none holds
it back, so a dropped index, with its facts and first reads, is freed at once. There is no default file-count refusal or parser timeout, and no requested file is silently
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
inventory was built, or changes after the index first read it, is reported separately as
unavailable. So is a file too large to parse safely. `tools.ast_grep_rules`, the one door every
parse passes through, estimates each file's parse peak (`index/file_shape.py`): 80 MB per MB of the
file, every byte counted as code, plus the square of the punctuation `{}();,[]` on each line,
which a minified bundle of a few tens of kilobytes on one line drives up. Files estimated at up to
250 MB are parsed side by side. A file over that, but within the single-file limit
(`MemoryLimit.single_parse_mb`: the child-process allowance less 270 MB of parser headroom, so 754 MB at the
default), is parsed alone on one thread, one at a time, with no other file beside it. A file
over the single-file limit is never handed to ast-grep, and neither is a large file that cannot be read
to measure it. `CodeIndex.refused_files` and `unavailable_files` give the reason, with the estimated
peak, the limit it is over, and the longest line in bytes. A file that ast-grep itself skips without
parsing (it prints nothing for a file that is not valid UTF-8, or for one of more than 3,000,000
bytes and 200,000 lines, which a file parsed alone can be) is refused too, as `not parsed`, and is
never taken for a file without functions. So is a file whose parse runs past the 30 s runaway guard
(`runaway_guards.PARSE_GUARD_SECONDS`): tree-sitter-python takes quadratic time on a long run of `#`
comment lines after a statement (66 s in ast-grep at 40,000 lines), while normal files parse in well under a second.
An ast-grep run of files side by side that passes the guard is stopped, and each of its files that had
not finished is parsed again alone under the same guard, so only the runaway file is cut. A refused file is never recorded as parsed: it stays readable and
searchable as text, it keeps its path in import relations (also as a re-export target), a name its
bytes mention binds `unknown`, so does any name imported from it, whether or not its bytes say the
name (a default export never needs the word `default`), `jvn stats` names it as never scanned, and `find_comments` lists it in
`refused_files`. Any ast-grep or ripgrep failure other than that verified disappearance still fails the
lookup that triggered it.

Calls are found by name in the syntax tree, which is not a resolved binding. Every call carries a
`Binding(status, reason, target)`: `resolved` when a module-level definition in the same file, or one
an import names, proves the target, `candidate` when only the name matches (a method on an unknown receiver, or a
definition elsewhere with no import), `unresolved` when nothing in scope defines it, and `unknown` when
the definition may sit in lines the index could not parse. Inside a TypeScript namespace a use first
names a member of the innermost namespace around it that defines the name, exported or not, so
`config` in `namespace B` is B's own and never namespace A's, nor an import's; outside it, a member is
no module-level definition. Lines are the unit, so a use on the namespace's first or last line stays a
candidate, and one namespace split over two blocks is not merged. Lines the parser lost inside the
namespace that mention the name leave the use `unknown`; lost lines elsewhere never pass the member over. A call `jwt.verify()` where module-level
code binds `jwt` to a whole module of the scope (`import * as jwt`, `const jwt = require('./jwt')`,
in Python `import app.jwt as jwt`, `from app import jwt` and `from . import jwt as tokens`, and
`app.jwt.verify()` after `import app.jwt`, all read from the syntax tree) binds to the `verify` that
module, or one it re-exports from, defines; only that module's facts are read. Python's `from app
import jwt` takes the package's own `jwt` before it imports the module `app.jwt`, so it holds the
module only when `app/__init__.py` binds nothing named `jwt` (a definition, an assignment, a loop or
`with` target), imports nothing else under that name,
has no star import and no lost line that mentions it; otherwise `jwt.verify()` stays a `candidate`.
In Python the alias must also be its module's one binding of the name: module-level code that
assigns `jwt`, defines a function or class `jwt`, loops, opens or catches into it, or deletes it, or a
function that declares it `global`, leaves `jwt.verify()` a `candidate`; after `import app.jwt` the
name is `app`. A script module alias is held the same way: a second declaration of `jwt` outside every
function (a `require` inside a block included), a loop over it, a module-level function `jwt`, or an
assignment to it anywhere leaves the call a `candidate`. A name a function binds for its own body (a parameter, a local
variable, a caught error or a loop variable) replaces any module-level definition or import of that
name inside the function: `db.query()` with a parameter `db`, or `stop()` with a parameter `stop`,
binds to no import; it is a `candidate` whose local value is not resolved. The one exception is a
function's own `const db = require('./db')` when it is the function's only binding of `db`: a `const`
is never bound again, so there `db.query()` binds through `./db` like a module alias, and `db()` to the
module's default export (`module.exports = ...`). A `const` is block-scoped, so it holds the module
from its own line to the end of its block, or of the function when it sits in the function's body;
before it, after its block, or in a `switch` case, the call stays a `candidate`. A `let` or `var` may
be bound again and holds no module. A function's own `const` that unpacks or reads a member of a plain
name, `const { insert } = client`, `const { insert: write } = client` or `const utc = client.toUtc`,
holds that member on the same terms, so `insert()`, `write()` and `utc()` bind as `client.insert()` and
`client.toUtc()` would; when `client` is itself the function's own value, such as a parameter, they stay
a `candidate`. A function counts from its first line, so on
`stream(c, async (stream) => ...)` the outer call counts as inside the callback.
Types are looked up apart from values, so a local value never replaces a type. A call `halt()` where
`halt` imports a definition under another name (`import { stop as halt }`, `const { stop: halt } =
require(...)`, `from m import stop as halt`) binds the same way to `stop`, unless the file defines
`halt` itself. A default import, under any local name, takes the module's default export; the default's own name is no named export, so `import { make }`, `defaults.make()` and `const { solo } = require(...)` of a default reach nothing. When the default export is an object literal, `export default { insert, utc: toUtc }`, a default import reaches its members: `client.insert()` binds to the module's `insert`, and `client.utc()` to its `toUtc`; `import { insert }` still reaches nothing, since the object's members are no named exports. Every import, by name, under another
name, as a default or through a module alias, is decided the same way from the module it names and
the modules that one re-exports the name from: one definition proves the target, several leave a
`candidate`, any of these modules that could not be parsed where it mentions the name, or that
vanished, leaves it `unknown`, and a module with no definition exported under the name leaves a
`candidate` that says so. A Python module passes on each name it imports by that name, as a package's
`__init__.py` does with `from .check import check` or `from .rules import *`, so `from pkg import check`
reaches `pkg/check.py`; a name it imports under another name (`from .legacy import old as new`) is not
followed, since its module exports it under the first. A function or class
held by another function, a class or an object literal, or assigned to a property (`foo.bar =
function () {}`), is no module-level definition, and neither is a function or class expression's own
name (`run(function handler() {})`), which is bound only inside it. One assigned to `exports.x` or
`module.exports.x`,
or listed in `module.exports = {...}`, is a CommonJS export: an import names it, its own module does not.
An import reaches only what its module exports. A Python module exports its whole module scope. A
script module exports the definitions an `export` statement or its own list names, under the name the
list gives them (`export { inner as outer }` exports `inner` as `outer`, never a private `outer`),
its default export
(`export default build`, `module.exports = build`), and its CommonJS exports (`exports.query = query`,
`module.exports = { log }`, and under another name `exports.parse = urlParse`); a module that exports
`new Logger()` exports no `log`, and an unexported helper stays its own module's. Each name an exported
destructuring binds, as `a` and `c` in `export const { a, b: c } = ...`, is an export. References carry
a binding too. A
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

### Choosing the files a search covers

`resolve_scope` decides which files a search covers from paths, git and the first lines of files. It
parses nothing, and it is not yet wired into the `jvn` commands. A `Scope` carries the request's
`scope` object: the four `with_` switches and `max_files` are required, because the request schema
owns their defaults (switches off, a cap of 200), and `Scope` repeats none of them.

```python
from jev_navigator.index.scope import ResolvedScope, Scope, resolve_scope

scope = Scope(
    repo=repo_root,
    include=("app/",),
    languages=("python",),
    with_tests=False,
    with_generated=False,
    with_vendored=False,
    with_docs=False,
    max_files=200,
)
resolved = resolve_scope(scope)
if isinstance(resolved, ResolvedScope):
    resolved.files  # the files in scope; resolved.filters names every filter applied
else:
    resolved.counts_by_folder, resolved.counts_by_language  # a ScopeRefusal: over max_files
```

- Only files JVN reads (Python, TypeScript, TSX, JavaScript, and Prisma schemas, whose blocks a
  scanner reads) enter a scope and count toward the cap, plus markup files with `with_docs`;
  `filters["supported_languages"]` names them.
- Left out unless asked for: tests (`with_tests`), generated code (`with_generated`: a true
  `linguist-generated` attribute, or a comment line holding `@generated` or `do not edit`, in any case,
  in the first 10 lines), vendored code (`with_vendored`: a true `linguist-vendored` attribute, or a
  `vendor`, `third_party` or `node_modules` folder) and docs (`with_docs`: a `docs` folder or a markup
  file). A false linguist attribute keeps a file the path or header rule would leave out.
- A file none of those rules decides may still look generated by its shape: a line over 10,000
  characters, dense lines, or a very large file (the triggers of `file_shape.shape_of`). Under an
  output folder (`dist`, `build`, `generated`, `__generated__` or one starting with `generated-`) such
  a file is left out before the count, listed in `resolved.set_aside` with "left out as generated:
  under dist/" and its measured facts. Anywhere else it stays in `files`, counted toward the cap, and
  is listed in `resolved.awaiting_generated_judgment` with its measured facts, for Jev to judge. With
  `with_generated` nothing is measured and nothing awaits a judgment.
- `judgments.generated_files.judge_generated_files(judge, index, resolved.awaiting_generated_judgment)`
  asks Jev about those files, one question each: is the file generated, meaning no person edits it as
  source? Each file is sent as its path, its measured facts, up to 10 files that import it with their
  true count, up to 5 files that name its path with the naming line (at most 200 characters around
  the path; files outside the scope count, non-test files come first; a path written relative to the
  naming file, such as `../src/a.js`, or joined to a variable folder, such as `$root/src/a.js`, is not
  found) and their true count, and two 2,000-character excerpts (the opening and the middle). A file
  the secret scan would refuse is never sent and comes back in `not_judged` with the reason. Nothing calls it yet: the
  search that acts on the answers lands with Find v2's round controller.
- `include` and `exclude` entries without `*`, `?` or `[` are folders or files. Other entries are
  globs over the whole path: `**` crosses folders, and a glob without `/` matches the file name at any
  depth unless a leading `/` anchors it at the root.
- `changed_since` keeps the files that differ from a git ref in the working tree, untracked files
  included; `filters["changed_since_commit"]` records the commit the ref named.
- More files than `max_files` returns a `ScopeRefusal` instead of files: the count, the cap, the
  filters, and counts per language and per folder one level below the folder the files share. Each
  folder label (`src/`, or `/src/*` for files directly in `src`), used as an `include` entry with the
  same other filters, keeps exactly the files it counts. Raise the cap with `max_files`.
- An unusable field raises `InvalidScopeError` naming it (`/scope/languages`, `/scope/changed_since`,
  `/scope/repo`). `checked_root(scope)` runs the `/scope/repo` check alone, for a caller that reads
  files inside the folder before it resolves the scope.

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
`choose_call_async`, `ask_async`, and `iter_check_every_async`, which yields each wave's answers as
the wave settles), and `find_code_async`, `find_all_async` and `find_all_text_async` are the async
searches. They take any
`AsyncJevClient` (an object with `model` and `async ask(state, questions)`, optionally an async
`send`), such as a host's own orchestrator; a sync client also works there and runs in a worker
thread. Both paths share one core: masking, the secret scan, the hash, the store lookup, the call
budget, the journal and the recording are the same steps, and only the send differs (a direct call,
or an awaited one). Batches of `check_each_async` go out concurrently, at most
`Judge(max_concurrency=N)` at once (default 16), and the places of each `find_code_async` round are
sent with `asyncio.gather`; the first batch of a `check_each_async` whose served model is still
unknown and which has an answer store goes out alone. Its live answer pins the served model, so the
remaining batches can replay from the store. The sync `check_each`, `check_every` and their `iter_`
forms send their batches on a thread pool under the same `max_concurrency` and first-batch rule; the `iter_` forms yield each batch as it completes. The
call cap stays exact under concurrency, and after a failure or cancellation no batch sends a new
request, while answers already received still yield. A sync method given an async client raises
`TypeError`. Offline tests use `testing.AsyncScriptedJevClient`.

Places: `check_each`, `check_every` and their `iter_` and async forms take `places`, one
`index.units.Item` per item, when items are code units. A place orders the batches (file, then
lines) and goes into the stored record, never into the state, so each item carries only the fields a
question reads; each `CheckResult` names its `place`.

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
- **Secrets.** `SecretMasker` masks secret values and keeps code (rules in `judgments/secret_shapes.py`,
  `secret_structures.py` and `secret_values.py`). It hides private keys, token shapes, password hashes
  (bcrypt, argon2), Bearer values, passwords and secret query values in URLs, and values under
  secret-named keys: shell and env-file words, quoted values (with escapes, prefixes such as `b"..."`,
  triple quotes, across lines, or never closed), YAML block and continued values, nested values that
  hold a literal, plain words, fallbacks after a reference, and literal arguments to secret-named
  calls that look like key material; plus high-entropy quoted values that are not identifier words.
  A key holds a secret when a secret word (including `pass`, `pwd` and `credentials`) is one of its
  parts; `max_tokens`, `tokenizer` and `bypass` hold none. Under a key the secret word ends
  (`DB_PASSWORD`, `authToken`, `db_pass`) every literal is hidden, in tables and blocks too. Under a key
  a naming word ends (`SECRET_ENV`, `token_url`, `CREDENTIAL_PATTERNS`) only a credential-looking word
  is hidden: one word of eight or more characters that is not a name, a path or a URL. Under any other
  suffix (`SECRET_KEY_BASE`, `GH_TOKEN_RO`) every string literal is hidden, sentences and passphrases
  too, except an environment variable's name, a path or a URL; that includes a block scalar and the
  literals concatenated onto one, while a nested table's inner keys are judged on their own. Under a
  key a message word ends (`PASSWORD_ERROR`, `TOKEN_HELP_TEXT`) a sentence is kept. A value that
  repeats its key (`PASS: "PASS"`) is kept, unless it is a common default password such as
  `password`. A long unquoted run of letters and digits is a value, not a reference.
  The masker reads a slice as its file type: in a config file (`.yml`, `.yaml`, `.env`, `.ini`, `.cfg`,
  `.conf`, `.properties`, `.toml`, a Dockerfile) or in text from no file, an unquoted value under a
  secret key is masked too (`POSTGRES_PASSWORD: example`), unless it is empty, a boolean or a whole
  `${VAR}`, `$VAR` or `${{ ... }}` reference; in code it stays (`token: str`). A request mapping's
  `file` names the file of the strings inside it, and a candidate's signature names its file the same
  way (the signature builders in `directives/places.py` write it, and `located_file` beside them parses
  exactly that grammar); a signature that names no file, or whose file is ambiguous, reads as config. An upper-case environment assignment is a value
  wherever it stands on a shell, Makefile or CI line (`run: API_TOKEN=... npm test`), unless it is a
  usage placeholder (`KEY=...`, `KEY=<credential>`). A secret flag on a command line
  (`psql --password=...`, `deploy --api-token ...`) and a Stripe secret key anywhere are values too.
  `is_high_entropy`, `HIGH_ENTROPY_MIN_CHARS` and `TOKEN_CHARACTER_CLASS` are public, for callers that
  judge a lone token. A reference stays code: an identifier, dotted path, call, a whole `${...}`, a
  command substitution `$(...)`, or `$NAME` outside single quotes, so `secret: process.env.AUTH_SECRET`
  reaches Jev unchanged, and so does nested metadata such as a Kubernetes `secret:` volume. Every rule
  scans in time linear in the text length. The analysis engine's audit-masker corpus is shared in
  `tests/test_secret_shape_corpus.py`. Masking works by content: a value hidden in one place is hidden
  everywhere in the request, for example where a relation text or another candidate quotes it; a value
  of 8 or more characters wherever it appears, a shorter one as a whole word, and a number of at most four characters or a value without letters
  or digits only where it stands. JVN's own question wording (instructions, and the criteria of a
  question that is not a choice) keeps its words, and a key of the request equal to a short masked
  value does not refuse it. A name that holds a hidden copy
  (`x-runs-[MASKED]`) is still a name, and a string a copy changed is masked once more, so the
  request sent is always one the rules leave as it is and the final scan refuses only what a
  host's own scanner finds.
  A quoted value may run across lines (triple quotes, template literals, text blocks) or sit in
  parentheses with its joined parts; a quote that closes a string the key sat in opens no value; a string
  that is not hidden is read again for the secret assignments inside it; and a value inside another
  quote's string ends where that string closes.
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
  `JsonlAnswerStore` is one run's pack. `SqliteAnswerStore(path)` is one store shared by
  every run on a machine, so a repeated run at the same commit asks nothing again but the requests
  that learn the served model (one batch, or the places of a Find's first round). It never holds
  code, state or question text: only hashes, unit locations, batch member ids, the batching rule and
  size, the model, raw answers and timestamps. Each request records the day a run last stored or reused
  it; the default store forgets a request unused for 30 days, and a store at a path you name keeps every
  answer ([housekeeping](#where-jvn-keeps-runs-and-caches)).
  `LayeredAnswerStore(pack, shared)` reads the pack first, copies every answer it finds only in the
  shared store into the pack, and writes new answers to both, so the pack alone still replays the run.
- **Journal, separate from the store.** Pass `journal=` (any object with `record_request(request) ->
  request_id`, `record_response(request_id, response)` and `record_failure(request_id, error,
  response)`). The judge records the masked request before dispatch and the raw response before
  parsing, as a `RawResponse(body, status, content_type, decoded)`: the body bytes as received, the HTTP
  status, the content type and `input_tokens`, the count the provider reported or `null`; a missing
  count is never written as 0, and the count is on the response line only. A replay from the store
  sends nothing, is marked `from_store` and carries no count. Transport errors and responses that fail to
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
  hashes, question wording, and each item's ids, file, lines, commit and names, or its place's file
  and runs, so `rebuild_request(record, old, shared)`, with `old` an open `CodeIndex.at_commit(...)`, rebuilds a request from
  the code at that commit and proves it matches, or names the part that differs. A request whose items carried a
  field that can quote code, such as a Trace link line or a Find signature, keeps that field withheld,
  so it does not rebuild exactly; the mismatch then names the withheld fields first.

## Layer 3: directives

`find_code(index, judge, target_description, start, *, budget=SearchBudget(), thresholds=None)` is
the central search. Use it only when the target is described by meaning; anything code can decide
(the callers of X) is an operation. For each opened place, a request asks "Does `slice.code`
contain the code described in `target.description`?" and, per neighbour code lists (callers, with
callers in test files after the others; callees, proven production targets first and then the ones
called from fewest places; code that
refers to it or that it passes on without a call, as an argument, collection entry, assignment,
decorator, export, return, method receiver, type or base class (also a qualified one, `pkg.Base`); the
modules it imports, re-exports
or requires (module-level code takes its whole file's imports): the definitions of the names it
takes from each, and the start of a module it takes whole or takes names from that it does not
define itself; the other functions of its file, nearest first; lines anywhere in scope (docs and
config too) that mention its environment variables or its quoted keys (six characters or more with
a dot, underscore, colon, slash or dash), the
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
code file in scope without a grammar error and read the blocks of every Prisma schema in scope; otherwise
it ends as `scope_incomplete`. The remaining
files are never parsed just to choose the label. Of `FindResult.code_files`, `files_judged` counts the
files in which Jev judged code (the opened places, not whole files) and `files_read` adds the files
read only to list neighbours; the CLI prints all three, for example `scope_incomplete (not found: Jev
judged code in 1 of 7 files; 4 more were read only to list links; 2 never reached)`. A start place is judged but never ends the search as found, because
the caller already had it; `FindResult.starts` keeps each start with its verdict. Each neighbour's
signature names its file and lines: a function quotes its first line; a window around a call, reference
or key outside any function gives its line range and quotes that line; a stretch chosen by position (the
lines before or after, the start of a co-changed or imported file) gives its range and quotes its first
line of code, past blank lines, comments, a license banner, a `'use strict'` directive or a module
docstring. The outcome is `found`, `stop_rule`, `budget`, `runaway`, `cancelled`, `failed`, `nothing_left`,
`unsure_only` or `scope_incomplete`, and the result keeps three sets: `found`; `searched` and `unsure`
(bodies actually judged, start places apart in `starts`); and `not_inspected`, each entry with its
reason (`budget`, `runaway`, `cancelled`, `failed`, `deprioritized`, `capped` or `depth`) and its `QueueTier`:
`START`, `PICK` or `MOVE`. A request that fails, such as a provider error or a full disk while
storing its answer, ends `find_code` and `find_code_async` as `failed`: `failure` holds that same error object, the answers
its round did get stay merged, and the failed place waits in `not_inspected` with reason `failed`.
A request Ctrl-C stopped is `cancelled` instead. Resume
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
marked "[line cut]"). An opened place goes to Jev whole when its requests fit the input box of the
judge's client (Jev's 32,000 tokens are 76,800 characters, `judgments.client.JEV_INPUT_LIMITS`): the
request asking whether it is the target, and, when the opening is split, the request asking about each
neighbour alone. Larger code is cut on a line boundary with a visible note, and `Visit.code` ends at
the last shown line. A cut never grows back: under `neighbours_per_kind` a shorter cut can list a
small neighbour in place of a large one, so the opening keeps that cut and the neighbours listed for it.
Every opening starts from the code that fits the request asking whether it is the target, and
`find_code.shown_for_target(code, target, input_limits, found=FOUND, masker=DEFAULT_MASKER)` gives
exactly that, for a caller that must show what Find shows. Every size check measures a request as
the judge's masker leaves it (`judge.masked_request_fits`), since masking can make it longer. If not even its first line fits, the place stays
`not_inspected` with reason `budget`; Resume on a route with a larger box inspects that same source.
`questions=SearchQuestions(found=...,
could_contain=..., open_first=None)` replaces the wording. `moves=` chooses how neighbours are listed: the default
`places.MOVES` maps each move's name (`callers`, `client_calls`, `callees`, `queried_models`,
`referenced_by`, `passed_on`, `imported`, `same_file`, `keys_mentioned`, `co_changed`, `lines_before`,
`rest_of_file`) to a function of the
index and the opened code that returns places. Pass a subset, or add a function of your own; `MOVES`
itself is read-only. `FindResult.moves` and the final `stop` step name the moves a search used, and
`context_for_comment` takes `moves=` too. The directives take their check (`check=`) as a parameter too.

### Sources, the frontier and each workflow's composition

A **source** ([`sources.py`](src/jev_navigator/sources.py)) is a primitive that reaches candidates
without a model call. It takes `Seeds`: the request's names and the targets' descriptions (`texts`),
the caller's files and anchors, or units a search already judged. It returns `Reach` records: a place,
which is a file (every unit listed in it) or an anchor (the unit holding it), with its provenance, which
is the source's name, the seed it came from, a distance and the request names it was reached by. A
source never builds units and never scores them. The search turns places into units with its own room
and reading, so units, `unlisted` files, `unresolved` anchors and each name's counts (`names`) have
one owner. The frontier measures every unit's code the same way whichever source reached it, so two
sources reaching one unit never score it differently; only the distance is the source's own, and when
several sources reach one unit the smallest counts. A source is any object with a `name`, a `label`
(how coverage counts the units it left unjudged) and `reach(index, seeds)`.

| Source | Reads | Reaches | Distance |
| --- | --- | --- | --- |
| `ANCHORS` | anchors | the unit each anchor names | 0 |
| `FILES` | files | every unit of each file | 1 for an anchor's file or a file it imports, else 2 |
| `NAMES`, `TEXT_NAMES` | names | the unit holding each line a name is on, rarest name first; `TEXT_NAMES` only in text files, never a lockfile | 3 |
| `DEFINITIONS` | names | the units defining each name | 1 |
| `REFERENCES` | names | the units using each name other than by a call | 2 |
| `NAMED_FILES`, `TEXT_NAMED_FILES` | texts, anchors | every unit of the code (or text) files they name by path or run as a module | 1 |
| `IMPORTS`, `IMPORTERS` | anchors, units | every unit of the files their files import, or that import their files | 1 |
| `CALLERS`, `CALLEES` | units | the functions calling each function unit, or that it calls | 1 |
| `MODELS`, `CLIENT_CALLS` | units | the Prisma models a unit queries, or the lines querying a model block | 1 |

The **frontier** ([`directives/frontier.py`](src/jev_navigator/directives/frontier.py)) is the order
in which a search judges what its sources reached. Under a call cap whatever is ranked last is lost,
so the order is a named policy. `STAGE_ORDER`, the default, judges source by source in the order the
composition lists them. `VALUE` scores every unit by code before the first call: the request's names
its code holds as whole words, each weighted by how rare it is, whether the unit or its file is named
like one, the distance and whether it is a test. Each target has its own queue, ranked by the names its
description spells out, and a share of the item slots in every batch: equal by default, and a caller
overrides it with `shares={"limit": 3}`. Under `VALUE`, a target with a unit clearing the Judge's
yes bar draws only its pending one-step hops. It settles when none is left to judge. Without a yes
answer, it keeps drawing its ordinary queue until the call cap or exhaustion. Supplied lines seed
discovery hops separately and provide no relevance answer for a target. A settled target draws nothing more, its share flows to the targets
still open, and when every target has settled the search ends `settled`. Every unit drawn is still
asked every target's question. `Policy("value_all", ranked=True)` keeps the queues and shares without
settling.

Each mini-workflow's default composition (a caller replaces any part with `sources=`, `hops=`,
`policy=` and `shares=`):

| Workflow | Starts from | Hops (settling policy only) | Policy | Ends |
| --- | --- | --- | --- | --- |
| `find_all` | `ANCHORS`, `FILES`, `NAMES` (`CODE_SOURCES`) | `CALLERS`, `CALLEES` (`HOP_SOURCES`) | `STAGE_ORDER` | scope examined, call cap, or every target settled |
| `find_all_text` | `ANCHORS`, `FILES`, `TEXT_NAMES` (`TEXT_SOURCES`) | `HOP_SOURCES` | `STAGE_ORDER` | as `find_all` |
| `find_text` | `TEXT_SOURCES` | `HOP_SOURCES` | `STAGE_ORDER` | the first wave with a yes |
| `find` (`find_code`) | not a composition of sources yet: its own neighbour moves (`places.MOVES`) | | | |
| `trace` | not a composition of sources yet: the static call graph (`operations.trace_graph`) | | | |

A new source feeds a workflow through `sources=` or `hops=`, with no change to the workflow. The
spelling map's source (`spelling`, every spelling of a name) and handler following (`handler`) are
being built under the same contract. A source joins a workflow's default composition only when a
measurement without model calls shows it reaches more of the deciding units at an equal or better
rank, without more Jev calls; until then a caller adds it. This composition, run by
[`tests/test_readme_examples.py`](tests/test_readme_examples.py), adds the definitions of the names
and the Prisma models a clearing unit queries, ranks by value and gives `limit` three slots for every
one of `refund`'s:

<!-- example: frontier composition -->
```python
from jev_navigator.directives.find_all import CODE_SOURCES, HOP_SOURCES, find_all
from jev_navigator.directives.frontier import VALUE
from jev_navigator.index.units import LineAnchor
from jev_navigator.sources import DEFINITIONS, MODELS

result = find_all(
    index,
    judge,
    {
        "limit": "the check that refuses an order over the item limit",
        "refund": "the code that refunds an order",
    },
    anchors=[LineAnchor("orders/service.py", 5)],
    files=index.files,
    names=["MAX_ITEMS", "check_limit"],
    sources=(*CODE_SOURCES, DEFINITIONS),
    hops=(*HOP_SOURCES, MODELS),
    policy=VALUE,
    shares={"limit": 3},
)
for target in result.targets:
    best = result.ranked(target)[0]
    print(target, best.unit.path, best.unit.symbol, round(best.probability, 2))
print(result.stopped_by, result.settled)
```

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
The budget is a character box: the client's input limit for state plus the longest question, less
the longest question asked, measured together with the shared state, and within `budget_chars` when
the history sets one. Each client declares its
limits as `input_limits` (`InputLimits`, in characters at the rate `REQUEST_CHARS_PER_TOKEN`); a
client that declares none is taken to be Jev, 32,000 tokens for state plus the longest question and
64k tokens for a whole request (the Engine measured 32,883 tokens accepted and about 33,200 refused on 27.09.2026). The
batching owner (`check_each`, `check_every`) and the `find_code` opening questions measure the same
limits before sending and split what would exceed them; a direct `Judge.ask` sends what it is given
and relies on the provider's refusal. When the selected sections still do not fit, the
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
and the result agree. Each Jev judgment in a step names the answer behind it in `answered_by` (a place
`choose_next` opens names the answer that scored it in `scored_by`): the request's `request_sha256`, the
`question_id` it was asked under, and `from_store`. The journal's `request` row with that hash lists the
question id, and that row's `response` holds the answer, also for an opening split into several requests;
packs written before these fields resume as before. Each automatic entry selection decision in the
manifest's `entry_selection`, and each Find All verdict in `found`, `unsure` and `searched`, names its
answer the same way. Without a stop rule nothing reads the history; with
one, the stop check reads the sections it selects (by default only the fetched code). `HistoryStep` is generic: append your own steps (an agent's tool call and result) the same way.

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
round_dir = Path("rounds/order-limit")
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
Plain `uv run pytest` installs the TypeSafe extra with the dev group, so every test runs. A skipped
test did not run, so a run with a skip fails and names it, unless the test declares a platform it
cannot run on with a `skipif` condition. To show the core works without the extra, run
`uv run --no-dev --with pytest --with pytest-timeout pytest --without-typesafe`: only there may the
TypeSafe tests skip, and it refuses to start when the extra is installed. Every run needs
pytest-timeout (`required_plugins`), so a missing plugin stops the run instead of dropping its
time limits. Run `uv run ruff check src tests` and `uv run ruff format --check src tests` before
pushing.

Locally, run only the test files that cover or import what you changed (`uv run pytest
--basetemp=<scratch dir> tests/<file>`); never the whole suite on the shared Mac (André,
05.10.2026). Pushes and pull requests do not launch hosted CI, so once a head is the one to merge,
start the `tests` workflow on its branch (`gh workflow run tests.yml -R ajbmachon/jev-navigator
--ref <branch>`). It runs the whole suite on Python 3.11 and 3.13, each with and without the
TypeSafe extra; judge it by its log. A whole-suite run that must happen outside CI goes to GX10
nr3 over SSH with a memory cap, never to this Mac.

## License

MIT, see [LICENSE](LICENSE).

## Compose code and named text with reserved calls

The general discovery sources reach files whose path components match the request's words,
named code and text files, imports, definitions, references, callers and owner-qualified callees.
`FILE_WORDS` places files with more matching words first. Exact text searches are batched and
retain overlapping matches. Callers compose these sources explicitly through `sources=` and
`hops=`. Selection blocks can rank the reached units before the caller chooses what to judge.

The existing `VALUE` policy follows one step from units that clear the Judge's relevance bar or
whose code the caller already supplies through `delivered`. Hop sources receive the seed unit's
code, spelled identifiers and file, so named paths and exact literals can be followed without
judging supplied code again. Supplied lines seed discovery only: they provide no target-specific
relevance answer and cannot close a target's ordinary queue. Hop results do not recursively expand.
`STAGE_ORDER` keeps its existing source-by-source population without relevance hops. Neither policy
adds a new request allowance. The Judge's caller-selected cap still owns the judging budget.

`mentions.names_from_text(text)` returns `TextNames(code=..., paths=...)`, using the existing
mention rules. Bare `copy_sandbox_tree` and `copySandbox` are code names, while `pyproject.toml`
remains a whole path. `sources.TEXT_FILE_NAMES` reaches a text file by its basename or stem even
when its body never repeats that name; ambiguous basenames retain all matches.

```python
from jev_navigator.composition import SearchConfiguration, reserve_calls
from jev_navigator.directives.frontier import VALUE

configuration = SearchConfiguration("code-and-named-text", code_calls=36, text_calls=12, policy=VALUE)
code, text = await configuration.search(index, judge, {"p": description}, files=scope, anchors=anchors)
```

The configuration calls `find_all_async` and `find_all_text_async` with separate stage allowances.
Text uses named paths, file names and name hits, under its own share. Each result retains its own
units, raw answers, cuts and provenance. The single-target `find_text` block is also available to a
caller that wants to stop at its first matching text unit.

`reserve_calls(judge, {"discovery": 36, "continuation": 12})` returns scoped judges whose caps are
reserved before either stage starts. Every call still counts against the parent. Unused calls remain
reserved; a host can deliberately assign unused calls in a later composition. These are explicit
allowances, not claims that one split is optimal. `VALUE` settles after a relevant unit and its
pending one-step hops have been judged; `STAGE_ORDER` examines its population up to the call cap.

### J1 ranks, roles label selected pieces

Code and text population searches use the admitted **J1-3** local-match question as their one ranking
profile. Its unchanged wording, yes/no criteria and contrasting examples live in
`judgments/local_match_question.json`; `J1` in `judgments/profiles.py` binds only the item and target
state paths. `match_check(target)` returns that Check. Each request carries `targets` and `items`,
each item containing `file` and `code`. Ranking consumes that Check's raw P(yes). Cut units retain
their best matching piece. Search has no role maxima, required-role retention or role prerequisites.
The bare match profile and six-role ranking profile are removed.

After the caller fits its packet, it passes **only the selected pieces and their exact visible text**
to `label_roles` or `label_roles_async`. Neither operation reads surrounding source or changes the
selection. The six unchanged questions remain owned by `judgments/role_questions.json`: `decide`,
`guard`, `value`, `effect`, `delegates` and `satisfied`. All six run together for each supplied piece
and point, in supplied order, at most sixteen pieces per request. The existing Judge handles size
splitting by item, masking, final scanning, call accounting and the configured answer store.
A single piece over the declared box remains explicitly unlabelled in `refusals` and is not sent.

```python
from jev_navigator.judgments.role_labels import LabelPiece, label_roles_async

# selected contains the final packet pieces, with their source Item and displayed text.
pieces = [LabelPiece(piece.place, piece.code) for piece in selected]
labels = await label_roles_async(label_judge, pieces, targets)
for labelled in labels.pieces:
    print(labelled.piece.place.id, labelled.probabilities)
```

`RoleLabellingResult.pieces` preserves supplied order. Each `PieceRoles.answers[point][role]` is
its original `CheckResult`, retaining probability, request hash, question hash, source place and
whether the answer came from the store. `probabilities` exposes the raw numeric dictionary without
combining roles. Refused pieces have empty answers and are unknown. The caller displays these
labels alongside each ranked region and chooses its display policy; labels never alter J1 ranks.
Identical complete labelling batches can replay from the store. Changing batch companions changes
the judgment context and requires new answers.

Pass labelling a Judge with its own caller-owned allowance, or reserve a separate stage share from
an uncapped parent. Its requests still count against that parent. A ranking allowance already used
up cannot fund the labelling step. The twelve-group ranking default is an **Engine setting**, not a
JVN default or an environment flag. No provider calls are made by preparing pieces or questions.

The [frozen selection report](measurements/selection/REPORT.md) and its summary record the
historical comparison and pinned revisions. The caller-specific reproduction harness and its tests
live outside JVN at `~/.local/share/jvn-takeover/2026-10-03/search-design/case1/selection-harness/`.
That harness reproduces historical controls at their recorded pins; JVN retains only the report and
summary, with no six-role ranking code or recipes.
