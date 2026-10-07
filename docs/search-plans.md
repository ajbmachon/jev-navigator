# Typed search plans

`jev_navigator.search_plan` executes a caller's plan of at most ten independent,
ranked lookups without a provider call. A plan is data, so a recipe, an LLM step or
an ordinary program can supply it. The library contains no caller domain policy.

```python
from jev_navigator.search_plan import decode_plan, execute_plan, PlanSource

plan = decode_plan('''{"approaches": [{
  "rank": 1,
  "call": {"operation": "definitions", "name": "check", "owner": "limits.py"},
  "provenance": [{"term": "check", "source": "supplied code"}],
  "reason": "Read the named restriction"
}]}''')
result = execute_plan(index, plan, box_chars=70000)
# Supply sources=(PlanSource(result),) to an existing mini-workflow.
# Keep result.outcomes to report invalid targets, empty searches and exclusions.
```

The ten operation families are `find_text`, `matching_files`, `file_units`,
`definitions`, `callers`, `callees`, `references`, `named_imports`, `config_key`
and `tests_of`. `plan_schema()` returns the JSON schema for the same runtime
contract. Unknown fields and operations, duplicate ranks and more than ten
approaches are rejected. Paths and definition owners are exact index-relative
files. Scope paths and globs use JVN's existing scope matching semantics.

Independent calls start together. Identical calls execute once and retain each
approach's provenance. Results merge by canonical unit identity in approach rank
order, independent of thread completion. Each candidate retains its approach
objects, input arguments, copied-term provenance and underlying source routes.
An invalid path or name is reported and never corrected to a guessed target.
Unresolved parser bindings keep the existing sources' semantics.

`PlanSource` feeds the same places into JVN's existing source contract. The
mini-workflow remains the owner of reading, ranking, grouping, Jev questions,
budgets, coverage and stopping. Code and text mini-workflows can each consume the
source; a mixed workflow can consume both. The host retains outcomes even if an
oversized candidate population first needs another selection step.

`jev_navigator.plan_outline.outline_lines` streams paths, actual declarations and
resolved imports one file at a time. The host chooses repository, directory or
query scopes and measures its tokenizer. It does not assign meanings to folders
or silently truncate a map to fit a prompt.

The development composition and zero-provider replay are in
`examples/pack_case2`. A recorded plan is a surrogate for a future planner, and
candidate reach is distinct from delivery. Changing ordered batch companions
invalidates stored Jev answers.
