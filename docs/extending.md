# Extending jev-navigator

The library is a set of small pieces you import and compose in your own functions. There is no plugin
system, registry or base class: a new use case is a plain function of 30 to 60 lines.

## The pieces

| Piece | What it gives you |
| --- | --- |
| `resolve_scope` | the files a search covers from folders, patterns, languages and a git ref, with tests, generated, vendored code and docs left out by default; a file only its shape marks as possibly generated is set aside under an output folder (`dist`, `build`, `generated`), and otherwise awaits Jev's generated judgment with its measured facts; a scope over its cap is refused with counts per folder and language (README, "Choosing the files a search covers") |
| `judge_generated_files` | Jev's generated-file judgment for the files a scope left undecided: one question per file over its path, measured facts, up to 10 importers and up to 5 files naming its path, each with their true count, and two excerpts; a file the secret scan refuses is named as not judged (README, "Choosing the files a search covers") |
| `CodeIndex` | mechanical lookups over a narrowed scope: definitions, callers, callees, references, text, imports, git history |
| `index.units` | the units a search judges (functions, methods, Prisma schema blocks, each file's top-level code; in a text reading, the text units of the files JVN does not parse), cut into 60-line pieces only when larger than their room in a request, and the one resolver of lines and line ranges to units |
| `index.text_blocks` | how a file JVN does not parse splits into blocks without a parser: Markdown by heading section, YAML, JSON and TOML by top-level key, anything else whole |
| `index.prisma_schema` | a Prisma schema's model, view, enum and composite type blocks with their lines, and the client accessor a model or view is queried through (`model WebsiteEvent` is `prisma.websiteEvent`) |
| `operations` | ready-made combinations of lookups: slices, traces, similar functions, code named in a doc, files a text names by path or runs as a module (`files_named_by`) |
| `mentions` | what a text mentions, one owner each: the path tokens it spells out (`paths_in`), the modules a `python -m` command runs (`python_modules_in`), whether a span names a file rather than code (`is_file_path`), the code names it spells out (`code_names_in`, which `code_named_in_doc` reads), and the members a symbol list names (`member_names`) |
| `Check`, `Pick`, `Rate` | one closed question each: yes or no, one option of a list, a level on a scale |
| `Judge` | asks questions with masking, a secret scan, a cache, budgets and a journal; returns raw probabilities |
| `find_code` | a best-first search that opens places until the code a description names is found |
| `sources` | the primitives a search's candidates come from, under one contract: a source reaches places (files or anchors) from seeds with their provenance and no model call; anchors, files, name hits, definitions, references, named files, imports, importers, callers, callees and Prisma model links are built in |
| `find_all` | judges every unit its sources reach (by default anchored lines, files, and each hit of named texts) against described targets, one question per unit per target, in the order a `frontier` policy gives |
| `frontier` | the order a search judges its population in: `STAGE_ORDER` (source by source) or `VALUE` (each target's own queue by code features, ties by content hash, an equal or caller-set share of every batch, and a target settling once a unit clears its bar and the units its hop sources reach from that unit are judged) |
| `find_all_text`, `find_text` | the same for text units only, the files JVN does not parse: judge every one, or stop once one is found |
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
lengthen code past the room (`Judge.fits_alone`). A unit or piece whose request is refused anyway, by
the provider for its size with no smaller split or by the final secret scan, is named `REFUSED` in
`not_judged`, its error is kept in `refusals`, and the search goes on. One unit never fails the
search. Every unit is one a listing lists, so a hit inside a nested function names the function
holding it. No code step is capped: the Judge's call cap is the only budget. The population goes to
the Judge in waves of `batches_per_wave` requests' worth (16 by default). Its order holds between waves, and exactly only at one batch per
wave. Like `items_per_request`, the wave size shapes the batches and so the answer store's keys.
`delivered` names the line ranges the caller already shows: a unit or piece whose every line lies
in them is named `already delivered by the caller` and not judged, while one with a line outside them
is judged. `find_all_async` takes the same arguments for an
async client, such as a host's orchestrator; it lists and reads code in a worker thread, reads
`cancelled` between waves, and keeps every answer a wave received before a failure. Ranking and any
bar belong to the caller:
`scores(target)` gives every judged unit's answer, and `names` each request name's places: how many
the sources reached by the name, how many the search resolved before it stopped, and how many named no
unit. `sources` lists the sources the search used, its hop sources last, and `entered_by` gives the
name of the source each unit entered the population by (`anchor`, `file` or `name` by default, the
first when several reached it; `caller` or `callee` for a unit a settling search pushed).

The population comes from sources (`jev_navigator/sources.py`): `sources=` names the ones that start
the search (`CODE_SOURCES` by default: `ANCHORS`, `FILES`, then `NAMES`) and `hops=` the ones a unit
that clears a target's bar expands through under a settling policy (`HOP_SOURCES`: `CALLERS` and
`CALLEES`). Every source reaches places from the same seeds, the caller's anchors, files and names plus
the targets' descriptions, and the search resolves them into units. A source of your own is any object
with a `name`, a `label` and `reach(index, seeds)` returning `Reach` records; the README's
[source table](../README.md#sources-the-frontier-and-each-pipelines-composition) lists the built-in ones.

`policy` (`frontier`) decides the order, and under a call cap whatever is ranked last is what gets
lost. `STAGE_ORDER`, the default, is the order above: anchors, files, then name hits rarest name first,
each wave's batches sorted by place. `VALUE` lists every source and resolves every name's hits before
the first call, scores each unit by code, and judges the best first, its batches kept in that order
(`Judge.iter_check_every(..., keep_order=True)`). A unit's `features` are facts of any language: the
request's names its code holds as whole words, each worth `1 / log2(2 + places)` over the places the
sources reached by that name, whether the unit is named like a name, whether its file is, its distance
(the smallest any source reached it at: an anchor's unit 0; a unit of an anchor's file, or of a file it
imports where the index resolves that language's imports, 1; any other file's unit 2; a unit only a
name hit reached 3), and whether its file is a test. `Weights` turns them
into one value; a test is ranked lower by that value, never dropped. Ties go to the content hash, never
to the path, so renaming a folder changes nothing. Code that repeats a unit already queued is judged
once: `repeat_of` names the unit judged in its place, and the repeat shares its answers in `scores` and
its reasons in `not_judged`. `ranked(target)` gives the answers best first, a tie going to the unit
worth more by code, then to the content hash. Under `VALUE` every unit is admitted before the first
wave, so the units never reached are counted, and the answer store's keys follow the new batches, so
answers stored under the stage order are asked again once.

Under `VALUE` each target has its own queue: its features count the names its description spells out
as whole words (all the request's names when it spells out none), so `features` is kept per target.
Each batch's item slots go to the targets in proportion to their shares, by smooth weighted round
robin: equal by default, set with `shares={"limit": 3}` (a target not named has 1). A unit drawn for
one target is asked every target's question. A unit clears a target's bar when its answer is yes by the
Judge's thresholds; it then pushes the units the hop sources reach from it (by default its callers
and callees, `operations.caller_functions` and `callee_functions`, a function or method only), which
that target judges before anything else, and a pushed unit pushes nothing, so the hops go one step
deep. The target settles once none of its pushed
units is left to judge (a refused, too large or delivered one counts as done), draws no more slots, and
its share flows to the targets still open. `pushed` names what each clearing unit pushed and `settled`
the targets that settled; when every target has, the search ends `settled`. `STAGE_ORDER` has one
queue and judges every unit until the call cap, so `shares` with it is refused. A `Policy` with
`ranked=True` and `settles=False` keeps the queues and shares and judges every unit.

`search_coverage.point_results(rounds, bar)` turns one search's rounds into each point's result:
`found` when a unit's answer reaches the bar, `none_among_judged` over the units judged, or `unknown`
when no unit was judged or a round failed, never "none". Its coverage counts units once over the
rounds: the units considered, the ones judged for the point, and the rest cut by source and reason. A
unit left unjudged for several reasons counts under the first of not reached, refused, too large and
delivered. A caller composing rounds names the source of a round it fed, such as
`Round(callees, sources.CALLEES)`; each source counts under its `label`. `PointResult.render(bar)` gives the fact line, for example
`audit: none at the bar 0.80 among 16 unit(s) judged, best P=0.100; 284 not reached (not negative proof)`.

`units_examined` means every unit of the population was judged, not that every semantic answer is
correct. Uncertain answers, parser failures, unlisted files and unresolved anchors remain visible.
Pass `completed=previous.judged` to continue with a fresh Judge allowance: a place answered for every
target is not asked again. Reuse is valid only for the same source bytes, scope, targets and
thresholds; the CLI verifies those identities in its saved pack. Cancellation keeps coverage partial
and parses nothing it has not reached. Ctrl-C during judging ends it `cancelled`, and a failed request
ends it `failed` with `failure` holding the same error; both keep every answer that arrived in
`judged`, so `completed=` resumes it. Retained journal receipts describe the work actually performed.

`find_all` judges code units only, and `find_all_text` (with `find_all_text_async`) judges only the
text units of the files JVN does not parse, with the same arguments, question, masker and request
guard. A caller decides when to include text and gives a text search its own Judge, so it never
spends code search's budget. A name's hits in code files and lockfiles are left out of a text
search, so a common word never floods it with a lockfile's pieces; a lockfile named in `files` or
`anchors` is judged. A code file named in a text search, like a text file named in `find_all`, is
named in `unlisted`. `find_text(index, judge, description, files=..., names=...)` searches the same
population for one target, `FIND_TEXT_TARGET`, and ends `found` after the first wave in which a
unit's answer is yes by the Judge's thresholds. See [Text units](#text-units).

The CLI composes entry selection and `find_code` with this function: the seed search's found code
becomes range anchors, and the population is every file in scope, so a seed-search miss still judges
the whole scope. Use `jvn findall "functions that enforce the order item limit"` or
`jvn --json '{"command":"findall","target":"functions that enforce the order item limit"}'`.
The evidence pack retains the seed search, per-unit answers and raw request identities.

A population can start from the files a text names by path. `operations.files_named_by(index,
texts, anchor_files)` reads `texts` and the whole text of each anchor file for path tokens
(`mentions.paths_in`: a file name with a suffix, perhaps under folders, without a leading `./`, `../`
or `/`). A token names each scope file whose path is the token or ends with `/` and the token, so
`jobs/sweep.py` names `web/jobs/sweep.py` but never `xjobs/sweep.py`, and a bare `ci.yml` names every
`ci.yml`. A module a `python -m` command runs (`mentions.python_modules_in`) names the file Python
runs, resolved like an import from the repository root or `src/`: `uv run python -m app.jobs` names
`src/app/jobs.py`, and a package names its `__main__.py`. Options before `-m` are skipped, also
those with a value (`python -W ignore -m app.jobs`, `python -X dev -m app.jobs`). Anchor files are never named, an anchor file
outside the scope is never read, and a named file is not read for further names. The result splits `code` files, for `find_all`'s `files`, from
`text` files, and `named_by` keeps the token that named each one. No model is called. As a source,
`sources.NAMED_FILES` (or `TEXT_NAMED_FILES` for a text search) reaches the files the targets'
descriptions and the anchors' files name, so a search can start from them through `sources=`.

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
each built-in move's name to its function (callers, the code querying a Prisma model through its
client, callees, the Prisma models the code queries, references, code passed on, imported modules, the
same file, quoted keys and environment variables, co-changed files, the lines before and after) and is
read-only. Pass `moves=` to `find_code`, `find_code_async` or
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
reaches apart from the table's coverage (a Prisma schema is reached once `schema_blocks_in` reads it),
and `parsed_files`, `parser_scans_pending` and `observed_unparsed_files` read only the reached files. Two processes may write the table at once: a new file is created whole and
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
from jev_navigator.index.units import (
    LineAnchor,
    RangeAnchor,
    items_to_judge,
    list_units,
    read_ranges,
    resolve_anchors,
)

room = judge.input_limits.box_chars - beside_the_unit  # the characters one unit's text may take in a request
listing = list_units(index, index.files, box_chars=room)
for unit in listing.units:
    print(unit.id, unit.kind, unit.symbol, unit.content_sha256[:12])
print(listing.unlisted)  # files that gave no units, each with its reason
for item in items_to_judge(listing.units[0]):  # the unit, or its pieces that fit the box
    print(item.id, item.ranges, read_ranges(index, item.file, item.ranges)[:60])

resolved = resolve_anchors(
    index, [LineAnchor("app/routes.py", 21), RangeAnchor("app/orders.py", 5, 7)], box_chars=room
)
print([unit.id for unit in resolved.units], resolved.unresolved)
```

`box_chars` is the room a unit's text has in one request, counted as escaped JSON like every request
(`judgments.questions.serialized_chars`): the client's box, `judge.input_limits.box_chars` (Jev's is
76,800 characters), less what the request carries beside the unit, such as its shared state and its
longest question. Passing the whole box would let a unit just under it through, and the request
carrying it would be refused.

A unit is one function, one method, one Prisma schema block, or one file's top-level code, in a code
reading (`Reading.CODE`, the default); a text reading lists [text units](#text-units). Its id is
the location `path:start-end`; top-level code is `path:top`. `list_units` lists the functions and
methods no other function holds, the blocks of each schema, and each file's top-level code, so every
line of code sits in a listed unit once: a nested function or callback is inside its holder's text and is not listed. A unit's
`symbol` names every holder, `OrderService.place`, and names an anonymous function by the line it
starts on, `<anonymous:4>`. A function a module-level constant's call builds goes by the name the
entry text gives it, `CodeIndex.constant_function_names`: `run` for `export const run =
Effect.fn("run")(function* ...)` and `userRouter.list` for a router's procedure, so a callback inside
one is `run.<anonymous:2>`. A callback that builds data, as in `items.map((item) => item.id)` or
`new Map(...)`, gets no constant's name. `content_sha256` hashes the unit's own text, so
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

A `.prisma` file has no parser grammar, so `index.prisma_schema.schema_blocks` scans it: each `model`,
`view`, `enum` and `type` block is one unit of kind `schema_block`, from its header line to the line of
the brace that closes it, named `model Website`. Braces in strings and after `//` never count, and a
header whose brace never closes is no block. A `generator` or `datasource` block holds settings, so
its lines are the schema's top-level code. `CodeIndex.schema_blocks_in(file)` gives a schema's blocks
and `CodeIndex.schema_files` the schemas in scope. A scope keeps schemas under the language `prisma`
(`languages.language_read`). A model's code lives where it is queried, so a caller that starts at a
schema adds each block's `client_call_text` as a name (`.website.` for `model Website`, from its
`client_accessor`) to reach the functions that read and write it. Find follows the same link both
ways: a line in a block opens the whole block, the same-file move offers a schema's other blocks,
`client_calls` offers the code holding a model's client call text, and `queried_models` offers the
blocks of the models opened code queries. Both are text matches: `.website.` also matches a
`session.website.domain` relation read.

Top-level code is a file's lines outside every function, method and schema block, class bodies
included, kept as runs of lines in order (`ranges`) without the blank lines at their edges. A file
whose top-level code is only imports, comments, directives such as `"use client"`, lines of closing
brackets and blank lines lists no top-level unit. In a code reading, a file JVN does not parse gives
no units and is named in `unlisted` with `language not supported`; in a text reading a code file is
named with `code, which a text search leaves to find_all`, and a text file left out with its reason.
So is a file that disappeared after the inventory, and a file outside the index's scope with the
index's own reason (`no file at this path`) or `not in the index scope`.

A unit whose text fits `box_chars` is one item, whatever its length. Only a larger unit is cut into
`pieces` of at most 60 lines (a YAML or JSON text block at its keys, see below), in order, with no
overlap and never across two runs of top-level code;
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
a file the reading leaves out, a line outside its file, a reversed range, and a blank line in a
file with no top-level code are reported in `unresolved` with their problem, and a file is parsed
only after its anchor is known to point inside it.

### Text units

A text reading (`list_units(..., reading=Reading.TEXT)`, and the same argument to `resolve_anchors`)
lists only the files JVN does not parse, and a code reading never lists them, so find and find_all
never see a text unit. A text file splits into blocks by its format (`index.text_blocks`), each one
unit of kind `text` in the language `text`, its id the block's lines `path:start-end` without the
blank lines at its edges:

- Markdown (`.md`, `.mdx`, `.markdown`) at each `#` heading outside a code fence and front matter, named
  by its heading path (`Install > macOS`); the lines before the first heading are a block of their own.
- YAML at each top-level key, JSON at each top-level key of an object whose keys start on lines of
  their own, and TOML at each root key and table header, named by the key path (`tool.ruff`). The
  lines before the first key belong to the first block, and comment lines right above a key to its
  block. A minified or invalid JSON file, or a list, is one block.
- Any other file is one block, named `<top level>`.

A block larger than its room is cut into 60-line pieces like any unit, except a YAML or JSON block,
which is cut one level deeper, at its value's keys (`text_blocks.child_blocks`), so one CI job stays in
one piece. Neighbouring keys are packed together while they fit 60 lines and the room, a longer key that
fits the room is one piece, and a key over the room is cut into 60-line pieces. A value without keys,
such as a list, is cut into 60-line pieces. `scope.text_files_left_out`
leaves out an env file (`.env`, `.env.*`, `*.env`) whatever its content, a binary file (a NUL byte in
its first 8,000 bytes, as git decides), and a vendored or generated file by the rules a scope applies
by default. An env template (`.env.example`, `.env.sample`, `.env.template`) is read, masked like a
config file, and a lockfile (`scope.is_lockfile`) is never left out as generated. `CodeIndex` decides each
file once (`text_files_left_out`) and reads its blocks once (`text_blocks_in`). A text unit is masked
like any request: YAML, TOML, `.conf`, `.ini`, `.properties` and Dockerfiles count as config, so an
unquoted value under a secret key is hidden too. `resolve_scope` still keeps only files JVN parses,
plus markup with `with_docs`.

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
Uncertain and unexamined items remain unresolved. A span whose request is refused stays unjudged
(`result.refusals` keeps it with its error, and the pack's `trace.refused` and report.md name it), so
every obligation stays unexamined, and the trace goes on. `result.graph` retains every walked function and
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
