# Open conflicts

Each entry names two patterns that disagree, why both still exist, and what would settle it. None is
resolved; each waits for a ruling.

## Contents

- [Three pairs of similar sources](#three-pairs-of-similar-sources)
- [`moves=` against `sources=` and `hops=`](#moves-against-sources-and-hops)
- [Trace does not fit the source contract](#trace-does-not-fit-the-source-contract)

## Three pairs of similar sources

Recorded 06.10.2026 on branch `claude/find-sources`, where find's moves became sources. Each pair reaches
similar code, and each stays two sources because their rules differ. Merging a pair changes `find` or
`find_all`, so it needs a measurement first.

1. `CALL_SITES` (find's callers) and `CALLERS` (find_all's hop). `CALL_SITES` keeps calls outside every
   function and calls bound to any definition the opened lines overlap (`falls_inside`), which a window
   inside a class needs. `CALLERS` keeps only calls bound to that exact definition (`names_exactly`) and
   reaches the calling function's first line, not the call.
2. `REFERRERS` (find's "referenced by") and `REFERENCES`. `REFERRERS` starts from the opened code's own name
   and keeps a use only where its binding may reach those lines. `REFERENCES` starts from the request's
   names and keeps every use.
3. `IMPORTED_CODE` (find's "imported") and `IMPORTS`. `IMPORTED_CODE` reaches the definitions the opened code
   imports by name, or the start of a module it imports whole, counting only a function's own import
   lines. `IMPORTS` reaches every unit of every file the anchors' or spans' files import.

## `moves=` against `sources=` and `hops=`

`find_code` takes `moves=`, a mapping from a move name to a source. `find_all`, `find_all_text` and
`find_text` take `sources=` and `hops=`, sequences of sources. Both hold the same kind of object. The move
names stay because `FindResult.moves`, the history's stop step and resume packs record them. Renaming
`moves=` would touch about sixty test call sites and the persisted names; it waits for a ruling.

## Trace does not fit the source contract

Trace (`operations.trace_graph`) returns links, not places. Each link has both ends, the line the call or
use sits on, the name, and a call whose target is outside scope is kept as a gap with no place. A `Reach`
carries neither the link's line nor a link with no place. André rules between:

- (a) Extend `Reach` with the line of the link, and let a reach have no place for an unresolved link. Trace's
  five edge kinds (contains, calls out, calls in, uses out, uses in) become sources, and handler following
  a sixth. This changes the contract other branches build on.
- (b) Keep trace's own walk and add handler following as a new edge kind inside it.
- (c) Ship find on sources first, then decide (a) or (b) as its own change.

The builder recommended (c), then (a).
