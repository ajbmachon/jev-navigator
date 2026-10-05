# Extending jev-navigator

The library is a set of small pieces you import and compose in your own functions. There is no plugin
system, registry or base class: a new use case is a plain function of 30 to 60 lines.

## The pieces

| Piece | What it gives you |
| --- | --- |
| `resolve_scope` | the files a search covers from folders, patterns, languages and a git ref, with tests, generated, vendored code and docs left out by default; a file only its shape marks as possibly generated is set aside under an output folder (`dist`, `build`, `generated`), and otherwise awaits Jev's generated judgment with its measured facts; a scope over its cap is refused with counts per folder and language (README, "Choosing the files a search covers") |
| `judge_generated_files` | Jev's generated-file judgment for the files a scope left undecided: one question per file over its path, measured facts, up to 10 importers and up to 5 files naming its path, each with their true count, and two excerpts; a file the secret scan refuses is named as not judged (README, "Choosing the files a search covers") |
| `masked_lines` | the text a request may show: `CodeIndex.lines` and every `read_slice` give a file's lines masked as one text by the index's masker (`DEFAULT_MASKER` unless you pass another), with the line count kept, so a slice, window or line cut never holds a value masked anywhere in its file; `plain_lines` is for analysis only. For text from outside the index, mask the whole file with `masked_lines` before you cut it; never cut unmasked source text into a request |
| `CodeIndex` | mechanical lookups over a narrowed scope: definitions, callers, callees, references, text, imports, git history |
| `index.units` | the units a search judges (functions, methods, each file's top-level code), cut into 60-line pieces only when larger than their room in a request, and the one resolver of lines and line ranges to units |
| `operations` | ready-made combinations of lookups: slices, traces, similar functions, code named in a doc |
| `Check`, `Pick`, `Rate` | one closed question each: yes or no, one option of a list, a level on a scale |
| `Judge` | asks questions with masking, a secret scan, a cache, budgets and a journal; returns raw probabilities |
| `find_code` | a best-first search that opens places until the code a description names is found |
| `find_all` | judges every unit of a population (anchored lines, files, and each hit of named texts, rarest name first) against described targets, one question per unit per target |
| `places.MOVES` | the ways a search lists the neighbours of an opened place; pick a subset or add your own |
| `StopRule`, `History` | your own stop check over a search's history, reading only the sections you select |
| `LlmStep` | an opt-in LLM call for the cases where Jev's answer is not clear enough |

Request masking reuses identical text within each masking operation. Its temporary memoization ends
with that operation: discovered secret values never carry over into a later request. Cross-field
masking and the final pre-send secret scan still apply to the complete request.

## The rule for questions

Code holds the goal, the loop and the stopping. Jev gets concrete state and one closed judgment:

1. Say what code will do with each answer, and what the costly error is.
2. Do everything mechanical in code: which functions exist, who calls whom, which files changed.
3. Ask one `Check` per item about a concrete property of supplied code. Add yes and no criteria
   when the instructions alone leave the boundary open; a `Check` takes both or neither.
4. Never ask whether something is false, wrong or contradicts something; ask for the concrete
   property instead, and let code combine the answers.
5. Handle every outcome: yes, no, unsure, and low confidence.

## Worked example: which route handlers write an audit entry?

```python
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.thresholds import NoulVerdict

WRITES_AUDIT_ENTRY = Check(
    name="writes_audit_entry",
    instructions=(
        "Does `{item}.code`, or one of the functions in `{item}.called`, write an entry to the audit log?"
    ),
    yes=Criterion(
        "A line in `{item}.code` or in one of `{item}.called` writes an audit log entry, "
        "for example `audit_log.write(...)`."
    ),
    no=Criterion(
        "No line in `{item}.code` or in `{item}.called` writes an audit log entry. "
        "A name that only mentions auditing does not count."
    ),
)


def handler_item(index: CodeIndex, handler) -> dict:
    """The handler's code plus the bodies of the functions it calls, one level deep."""
    called = []
    for edge in index.callee_edges(handler):
        targets = [edge.binding.target] if edge.binding.target else index.find_definition(edge.name)
        for span in targets:
            code = index.read_slice(span).text
            called.append({"name": edge.name, "binding": edge.binding.status, "code": code})
    return {
        "file": handler.file,
        "lines": [handler.start, handler.end],
        "code": index.read_slice(handler).text,
        "called": called,
    }


def handlers_writing_audit_entries(index: CodeIndex, judge: Judge, handlers):
    items = [handler_item(index, handler) for handler in handlers]
    results = judge.check_each(WRITES_AUDIT_ENTRY, items)
    return {
        "writes": [result.item for result in results if result.verdict == NoulVerdict.YES],
        "does_not": [result.item for result in results if result.verdict == NoulVerdict.NO],
        "unsure": [result.item for result in results if result.verdict == NoulVerdict.UNSURE],
    }
```

Why it is built this way:

- Code, not Jev, decides which handlers exist and which functions each one calls.
- A handler that delegates the write to a helper is still judged correctly, because the helper's body
  is in `{item}.called`; Jev is never asked to guess what an unseen call does. This holds for one level
  only: a write two calls down is not in the state, so read "no" as "no write within one call", or
  expand further in code. Each entry in `called` keeps its binding status, so a call the index could
  not resolve stays visible instead of looking like a proven one. A yes can rest on the body of a
  function the index could not prove (`candidate`, `unresolved` or `unknown`); when you need proof,
  keep only the entries whose binding is `resolved`.
- Each handler is its own question, so one handler cannot hide another.
- Unsure stays unsure: it is reported, never counted as "does not write".

Test it offline with `ScriptedJevClient` and AAA tests, including the unsure path, before any paid call.

## Searching instead of listing

### Judge every unit with Find All

This is an ordinary function composition, not a workflow interpreter. `find_all` judges each unit of
a population against one or more described targets: one question per unit per target, all targets
asked in the same request. The population is the units the caller's line and range anchors name,
then the units of the named files, then the units holding each hit of the named texts, names with
fewer hits first, so a common word never decides which hits of a rare name are seen:

```python
from jev_navigator.directives.find_all import find_all
from jev_navigator.index.units import RangeAnchor

seed = index.find_definition("check_limits")[0]
result = find_all(
    index,
    judge,
    {"limit": "the check that limits items per order"},
    files=index.files,
    anchors=[RangeAnchor(seed.file, seed.start, seed.end)],
    names=["max_items"],
)
for score in result.scores("limit"):
    print(score.unit.path, score.unit.ranges, score.probability)
```

Each item Jev reads holds only the unit's file and code. The targets sit in the shared state, and each
question reads "Look only at `items[n]`. Does that code match the description in `targets.<name>`?"
(`match_check`). A unit larger than its room in a request (`result.room`) is judged by its pieces and
scored by its best one; a piece still too large is named in `not_judged`, and so is a unit or piece
whose request asking every target does not fit the client's limits once masked, since masking can
lengthen code past the room (`Judge.fits_alone`). One unit never fails the search. Every unit is one
a listing lists, so a hit inside a nested function names the function holding it. No code step is capped: the
Judge's call cap is the only budget. The population goes to the Judge in waves of `batches_per_wave`
requests' worth (16 by default). Its order holds between waves, and exactly only at one batch per
wave. Like `items_per_request`, the wave size shapes the batches and so the answer store's keys.
`delivered` names the line ranges the caller already shows: a unit or piece whose every line lies
in them is named `already delivered by the caller` and not judged, while one with a line outside them
is judged. `find_all_async` takes the same arguments for an
async client, such as a host's orchestrator; it lists and reads code in a worker thread, reads
`cancelled` between waves, and keeps every answer a wave received before a failure. Ranking and any
bar belong to the caller:
`scores(target)` gives every judged unit's answer, and `names` each name's hits found, reached and
naming no unit.

`units_examined` means every unit of the population was judged, not that every semantic answer is
correct. Uncertain answers, parser failures, unlisted files and unresolved anchors remain visible.
Pass `completed=previous.judged` to continue with a fresh Judge allowance: a place answered for every
target is not asked again. Reuse is valid only for the same source bytes, scope, targets and
thresholds; the CLI verifies those identities in its saved pack. Cancellation keeps coverage partial
and parses nothing it has not reached. Ctrl-C during judging ends it `cancelled`, and a failed request
ends it `failed` with `failure` holding the same error; both keep every answer that arrived in
`judged`, so `completed=` resumes it. Retained journal receipts describe the work actually performed.

The CLI composes entry selection and `find_code` with this function: the seed search's found code
becomes range anchors, and the population is every file in scope, so a seed-search miss still judges
the whole scope. Use `jvn findall "functions that enforce the order item limit"` or
`jvn --json '{"command":"findall","target":"functions that enforce the order item limit"}'`.
The evidence pack retains the seed search, per-unit answers and raw request identities.

### Find one location

When code cannot list the candidates, search: `find_code(index, judge, description, start)` opens
places (starts, then Jev's picks, then the best-scored neighbours) and returns `found`, `searched`,
`unsure`, `starts` and `not_inspected` with reasons. A start place never counts as found. Add a
`StopRule` with your own concrete check when the target is spread over several places.

Read `searched` and the outcome `nothing_left` as "opened and judged unlikely", never as "the code does
not exist": one "no" about one place can be wrong. When nothing is found, rank the opened places by
their probability and treat the best one as the likeliest place. A search that ran out of places
before parsing every code file in scope ends as `scope_incomplete`. `files_judged`, `files_read` and
`code_files` say in how many files Jev judged code, how many the search read, and how many are in scope.

Documents and other files without a supported code grammar remain searchable as text and can
participate in text-based moves. Syntax operations return no symbols, calls or references for them;
they are never sent to ast-grep with an empty language rule. This does not claim their text was
parsed as code.

The CLI creates a unique run folder under `$XDG_DATA_HOME/jev-navigator/runs/` when `--out` is
omitted. Each run retains its report, manifest and request journal. An `--out` folder inside the
searched directory is excluded from the CLI's source inventory so repeated searches do not search
their own evidence. Library callers can similarly pass `exclude_paths` to `CodeIndex.from_directory`.

The CLI indexes every file under the search root whether git tracks it or not, minus ignored ones: in a
Git worktree by git's ignore rules, including the lines of an enclosing repository's .gitignore that
match the root, and outside Git by ripgrep's ignore files. Each file or folder left out (ignored, a
separate git repository, a symbolic link) is named with its reason in `search.not_indexed_files`,
`trace.not_indexed_files` or the statistics pack's `coverage.not_indexed`; the reports count them by
reason and top folder, so thousands of ignored build outputs stay one row. A left-out folder that holds
no indexed file is named once, ending in `/`. A file that is new or edited since the last commit is
read from the disk, and its slices carry the commit plus `+worktree`.

Agents can pass the same CLI request as JSON with `jvn --json request.json`, an inline JSON object,
or `jvn --json -` for stdin. `jvn schema find` emits its JSON Schema without model calls. The CLI parser remains the single owner of options, types and defaults. `target` is
required; `command` defaults to `find`. JSON mode emits the result and evidence-pack paths on
stdout while progress stays on stderr. See the README for the request and result fields.

The CLI report distinguishes candidates not independently opened from text already included inside
a larger opened span. `target_found` means the search stopped after finding a match; `budget` means an
actual configured limit stopped an otherwise viable candidate. The recorded candidate score is the
entry Choice probability for an initial alternative or the could-contain Noul probability for a
navigation neighbour; those are different judgments. Navigation elapsed time excludes indexing and
entry selection. Journal `exact` flags, not the presence of JSON, determine whether responses are
wire captures or re-encoded SDK data.

## Choosing how the search moves

A move is a plain function of the index and the opened code that returns places. `places.MOVES` maps
each built-in move's name to its function (callers, callees, references, code passed on, imported
modules, the same file, quoted keys and environment variables, co-changed files, the lines before
and after) and is read-only. Pass `moves=` to `find_code`, `find_code_async` or
`context_for_comment` to use a subset,
for example `{name: MOVES[name] for name in ("callers", "callees")}`, or add a function of your own.
`FindResult.moves` and the final `stop` step of the history name the moves the search used, so every
result says how it was found. Your move's places go through the same filter as the built-in ones:
places that open the same lines of the same file are kept once, the first move that listed them wins,
a place wholly inside the opened code is dropped, and each move's cap counts only places no earlier
move kept. Order the places a move returns by how likely they are to matter, because the cap keeps the
first ones.

## Stopping on your own check

`find_code(..., stop_rule=StopRule(check))` asks your check after every round. `sections` chooses what
the check sees: only the `fetched` code with its sources (the default), each step's operation,
arguments and code (`history`), the search's own verdicts (`decisions`), or sections you add with
`StopRule(context={"shown_code": ...})`. Only `decisions` carries verdicts, so a check reads them only
when you select that section.
Pick the smallest view the question needs; the fewer unrelated fields, the steadier the answer.

## Adding your own steps to a history

`HistoryStep(operation, arguments, fetched, judgments, decision)` is generic. An agent's tool call and
its result can be appended next to a search's own steps, and every check reads them through the same
sections.

## When Jev's answer is not enough

Nothing in the library escalates on its own. If a low-confidence pick should go to an LLM, add an
`LlmStep` to your function: you define when it runs, what context it gets, the answer contract and the
connector.

## Async hosts

Every judge method has an `*_async` form, and `find_code_async` is the same search. Pass any object
with `model` and `async ask(state, questions)`; masking, the store, budgets and the journal behave
exactly as on the sync path.

## Testing

Test every function offline with `ScriptedJevClient` (or `AsyncScriptedJevClient`) in Arrange, Act,
Assert form. Script the unsure and low-confidence answers too, and assert what your code does with
them. `CapturingJevClient` records a function's exact first request, so you can review the question
text before any paid call.


### Parser facts and naming

Function and class names come from the syntax tree, never from a physical line. A declaration is
named by its own name node. A function or class expression is named by the declarator, class field,
object key or assignment that holds it, looking through parentheses and type casts
(`export const load = (async () => ...) satisfies PageLoad` is `load`), and only then by its own
name. A callback passed to a call (`it("works", () => ...)`) is held by no name and stays
`<anonymous>`. This keeps a method on a one-line TypeScript class distinct from its enclosing class,
and keeps test and framework callbacks from sharing the names `it`, `describe` or `expect`. A
callback spanning exactly a named symbol's lines (`xs.map((x) => x.id)` on the one line of `ids`) is
the same place, so it is left out rather than listed as a second, anonymous symbol.
Persistent facts are keyed by source bytes, language, parser version, the ast-grep rule text a scan
of that language sends, and the source of the modules that build the rules and turn matches into
facts (`fact_cache._MODULES_THAT_READ_MATCHES`). Changing a rule or the code that reads matches
reparses existing cached results by itself; there is no version string to bump. A new module that
shapes facts belongs in that tuple.

Name lookups read `name_table.NameTable`: one SQLite file per `table_identity()`, which hashes
`fact_cache.facts_identity()` (the parser version and every language's rules) with the source of
`name_table.py`. Rows are written only from facts, at `CodeIndex._remember_facts`, keyed by the git
blob id of the file content, and hold names, kinds, lines, roles and receivers as the facts hold them:
`scope_scan.receiver_of` keeps a receiver only as a plain chain of names and records anything else,
which could quote a string literal, as `OPAQUE_RECEIVER`. `CodeIndex` records the files navigation
reaches apart from the table's coverage, and `parsed_files`, `parser_scans_pending` and
`observed_unparsed_files` read only the reached files. Two processes may write the table at once: a new file is created whole and
linked into place (`shared_database.open_shared_database`), and each content's rows are written in
one transaction. A bidirectional
trace prepares the scoped fact inventory in one batch before walking incoming and outgoing links;
it does not launch one repository search for every encountered name.

## Structural measurements without model calls

Use `directives.statistics` for function/class counts and inclusive physical line sizes. It reuses
one batched parser-fact inventory; Jev is not needed for arithmetic.

```python
from pathlib import Path
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.directives.statistics import count_symbols, largest_functions

index = CodeIndex.from_directory(Path("."))
counts = count_symbols(index)
print(counts.total, counts.coverage)

ranking = largest_functions(index)  # includes methods and nested functions
for symbol in ranking.biggest:  # retains all ties for the largest function
    print(symbol.span, symbol.size)

between = [symbol for symbol in ranking.measured if 20 <= symbol.size <= 50]
function_counts = count_symbols(index, kinds=("function",))
```

A `limit` restricts the displayed ranking, not the measured inventory or counts. Inspect coverage
for unreadable, unsupported and partially parsed files; recovered symbol counts are not proof that
all source parsed successfully. An unrequested symbol kind is omitted, not represented as zero. `counts.count_of(path)` returns
`None` when that file was not measured; only a measured empty file has zero counts.
Same-line nesting can have no known holder because the current index records line spans rather
than AST parent identities. The `jvn stats` CLI writes these measurements as JSON and Markdown; see [the CLI guide](cli.md#structural-measurements).

## Units

`index.units` lists what a search judges and names the units that hold a caller's lines, with no
model:

```python
from jev_navigator.index.units import LineAnchor, RangeAnchor, items_to_judge, list_units, read_ranges, resolve_anchors

room = judge.input_limits.box_chars - beside_the_unit  # the characters one unit's text may take in a request
listing = list_units(index, index.files, box_chars=room)
for unit in listing.units:
    print(unit.id, unit.kind, unit.symbol, unit.content_sha256[:12])
print(listing.unlisted)  # files that gave no units, each with its reason
for item in items_to_judge(listing.units[0]):  # the unit, or its pieces that fit the box
    print(item.id, item.ranges, read_ranges(index, item.file, item.ranges)[:60])

resolved = resolve_anchors(index, [LineAnchor("app/routes.py", 21), RangeAnchor("app/orders.py", 5, 7)], box_chars=room)
print([unit.id for unit in resolved.units], resolved.unresolved)
```

`box_chars` is the room a unit's text has in one request, counted as escaped JSON like every request
(`judgments.questions.serialized_chars`): the client's box, `judge.input_limits.box_chars` (Jev's is
76,800 characters), less what the request carries beside the unit, such as its shared state and its
longest question. Passing the whole box would let a unit just under it through, and the request
carrying it would be refused.

A unit is one function, one method, or one file's top-level code. Its id is the location
`path:start-end`; top-level code is `path:top`. `list_units` lists the functions and methods no
other function holds, and each file's top-level code, so every line of code sits in a listed unit
once: a nested function or callback is inside its holder's text and is not listed. A unit's
`symbol` names every holder, `OrderService.place`, and names an anonymous function by the line it
starts on, `<anonymous:4>`. `content_sha256` hashes the unit's own text, so
an unchanged function keeps its hash when other lines of its file change. The record holds locations
and hashes, never code; its `ranges` are its (start, end) line pairs in file order, one for a
function and one per run for top-level code. `read_ranges(index, file, ranges)` is the one reader of
that code, joining the ranges in order with a newline.

A function's or method's unit starts at its first decorator, so a route such as
`@app.route("/orders")` or NestJS `@Get()` is judged with its handler and is not top-level code.
Only the unit's `ranges` reach back to the decorator: its id, like the index's span
(`CodeIndex.decorator_starts_in`), still starts at the function's own first line. Python and
TypeScript put decorators before the function node; JavaScript's parser already starts a method at
its decorators. A class's decorators stay with the class head in the top-level code. A stub, a Python
function whose body is only `...`, `pass`, a docstring or `raise NotImplementedError`
(`CodeIndex.stubs_in`), is no unit of its own: its lines are top-level code, so a Protocol is judged
whole.

Top-level code is a file's lines outside every function and method, class bodies included, kept as
runs of lines in order (`ranges`) without the blank lines at their edges. A file whose top-level code
is only imports, comments, directives such as `"use client"`, lines of closing brackets and blank
lines lists no top-level unit. A file in a language JVN does not parse gives no units and is named
in `unlisted` with `language not supported`, as is a file that disappeared after the inventory, and a
file outside the index's scope with the index's own reason (`no file at this path`) or `not in the index scope`.

A unit whose text fits `box_chars` is one item, whatever its length. Only a larger unit is cut into
`pieces` of at most 60 lines, in order, with no overlap and never across two runs of top-level code;
each piece has its own range, hash and size. A piece still larger than `box_chars` is
`too_large_to_judge`: it keeps its range and size, and `judged_pieces` leaves it out. A cut unit
stays one unit: `unit_score` gives it its best piece's score, and `best_piece` names that piece's
lines as the place to read. `items_to_judge(unit)` gives what a request judges: the whole unit as
one `Item(id, file, ranges)`, or each piece that fits, with the id `unit.piece_id(piece)`, the unit
id plus `#p<index>`.

Spans are lines, so functions on the same lines are one unit named by the first named one, and a
callback that shares a line with top-level code (`app.post("/orders", (req, res) => ...)`) takes
that line: its unit's text holds the registration.

`resolve_anchors` is the one way to turn lines into units. A line names the innermost unit holding it:
a function, its decorators included, or the file's top-level code when the line lies outside every
function, stubs included, even top-level code the listing leaves out. A line inside a nested
function names that function, which the listing leaves out; its `nested_in` names the function that
holds its text. A range names each unit its non-blank lines touch, without the units nested in
another one it names. Each unit comes back once, in the order first named. With `listed_only=True`
every unit named is one `list_units` lists, for a caller that judges only listed units: a nested
function gives way to the outermost function holding it, and lines of only top-level code the
listing leaves out (imports, comments, directives, brackets) are reported. A file outside the scope,
a file in a language JVN does not parse, a line outside its file, a reversed range, and a blank line
in a file with no top-level code are reported in `unresolved` with their problem, and a file is
parsed only after its anchor is known to point inside it.

## Trace a workflow and retain its evidence

`directives.trace.trace_workflow` composes the existing static graph walk with five independent
checks: input origin, transformation, handoff, observable outcome and relevant branch. All checks
share batches through `Judge.iter_check_every`; graph connectivity remains a static fact and a
positive model answer never promotes a candidate binding to a proved connection.

```python
from jev_navigator.directives.trace import trace_workflow

result = trace_workflow(
    index,
    judge,
    "How does an order request become an HTTP result?",
    index.find_definition("handle_order"),
)
for obligation in result.obligations:
    print(obligation.name, obligation.status, obligation.examined)
```

An evidence-backed obligation has at least one positive judgment; it does not prove the whole path
or every relevant branch is present. Negative judgments mean no evidence in the supplied component.
Uncertain and unexamined items remain unresolved. `result.graph` retains every walked function and
link, including uncertain bindings; `included` is only a presentation backbone, not a deletion of
the remaining component.

The callable `cli_trace.create_trace_evidence_pack` takes repository, question, `PATH:LINE` starts,
output directory and client. It writes JSON, Markdown, the request journal and stored answers.
The manifest retains the full static graph so resolved connections can be inspected after exit.
Outcomes distinguish completion, an explicit depth boundary, call budget and cancellation.
Cancellation is cooperative between static steps and live model batches; already answered batches
are retained in full, and a request already in flight is not aborted by the callback.
`answers_from` with the prior served-model identity reuses identical stored answers without calls. A
store that kept its requests' text (written with `keep_requests=True`) seeds only a pack that keeps
them too, and is refused otherwise before the output directory is made, so a default pack holds no code.
Use `jvn trace "order request to HTTP result" --start app/orders.py:42` for the same pack from the
CLI. `jvn schema trace` describes JSON input; [the CLI guide](cli.md#workflow-trace) explains options.
