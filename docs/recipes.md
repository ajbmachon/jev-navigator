# Small searches configured by callers

`SearchRecipe` is a named, versioned configuration of existing `Source` blocks.
It gathers mixed code and text units through `UnitReader`, keeps every source
witness, and reports unresolved places. It makes no model request.

```python
from jev_navigator.recipe_configuration import SearchRecipe
from jev_navigator.recipe_blocks import OWNER_DEFINITIONS
from jev_navigator.sources import ANCHORS, CALLEES, CALLERS, NAMED_FILES, TEXT_NAMED_FILES, Seeds

recipe = SearchRecipe(
    "named-code", 1,
    sources=(ANCHORS, OWNER_DEFINITIONS, NAMED_FILES, TEXT_NAMED_FILES),
    hops=(CALLEES, CALLERS),
)
candidates = recipe.gather(index, Seeds(names=("read",), files=("api.py",)), box_chars=76_800)
# The caller judges these candidates through the existing Judge/frontier.
# It supplies only the units confirmed relevant to this one follow-up hop.
next_candidates = recipe.continue_from(index, confirmed, box_chars=76_800)
```

A caller can inject `rank=` into either method. The ranker must reorder every
candidate exactly once. Stable source order is the placeholder when no ranker
is supplied. Candidate selection, request limits and judging remain the caller's
configuration, with cuts counted there. The pack-specific local, named and
convention recipes live in `evaluations/pack_recipes/configurations.py`.

`guard_chain(index, entry)` walks resolved calls and passed callbacks downward.
Decorator expressions travel with the entry. It returns static evidence
candidates with links, unknown bindings and an explicit depth boundary. A
`ChainAttachment` adds a router middleware, wrapper, decorator or schema binding
from a wiring map, including its registration location and certainty. Only proven
attachments are expanded. The chain does not classify every callee as a guard or
prove that a dynamic framework applies all of them before the handler. Supply the
framework's entry and wiring, then use the existing judgments to classify the
visible evidence. No upward caller search establishes the chain.

`value_source(index, key, aliases=(), files=())` locates setting occurrences in
code, configuration and documentation. The containing units include assignments,
defaults, environment reads and overrides where the names occur. Explicit aliases
connect a public key with an internal setting name. This is lexical reach, with
the value's semantic role left to the caller's judgment.

`presence_check(index, literal, places)` checks an exact literal in an explicit
list of files or inclusive spans. It reports matches and the number of places
actually checked. The aggregate is present if any place contains the literal,
absent among checked places only when every supplied place was checked, and
unknown when any place could not be checked or the list is empty. Individual
unknown places remain visible even when another place is present. A missing file,
an excluded source, or an invalid range cannot prove absence. Literal presence
does not establish semantic enforcement or the absence of aliases.

The development evaluation keeps registered deciding lines outside every search
argument. It reports reach and request positions separately from strict cached
delivery. The exact ordered batch and complete questions must match before a
stored answer can be consumed. Selecting whole historical batches with their
original companions is a separate frozen-group development measure.
