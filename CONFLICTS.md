# Conflicts and duplicates

Two implementations of one idea, or two patterns that contradict each other, are listed here until one
owner is chosen. Each entry names both sides and a proposal. Nothing here is decided until the entry says so.

## Open

### 1. Packing pieces into a token budget

- **Here:** nothing yet. The planned slicing call (`directives/slice_code.py`) needs to pack kept code
  units into a token budget.
- **Elsewhere:** the private evaluation harness (navlab) has `packing.pack_in_order`. It takes each piece
  in order if it fits in what is left, and skips a piece whose lines a packed piece already covers. Its
  budget rule (packed tokens plus tokens left equal the budget) is proven in Lean.
- **Proposal:** the library owns `jev_navigator.packing` (`Piece`, `Packed`, `pack_in_order`, `covered`),
  and the harness imports it. The proof's reference follows the code.

### 2. Estimating tokens

- **Here:** `history.estimate_tokens` counts characters / 3 + 1.
- **Elsewhere:** the harness's `packing.tokens_of` counts characters / 3.5, rounded up.
- **Why it matters:** the two differ by about 17%, so the same budget holds different code.
- **Proposal:** one estimator, owned here. Results measured with the other one say which they used.

### 3. The same-file functions a unit uses

- **Here:** nothing yet. The slicing call adds, as glue, the same-file definitions a kept unit uses.
- **Elsewhere:** the harness's `evidence_pool.same_file_callees` does the same from the index's bindings.
- **Proposal:** the library owns it, next to the index's call and reference lookups.

### 4. Merging overlapping code of one file

- **Here:** nothing yet. The slicing call merges overlapping or touching gathered slices of a file into
  one region before cutting units.
- **Elsewhere:** the harness's `evidence_pool.merged_units` merges overlapping pool units.
- **Proposal:** the library owns it.

### 5. One wording for "does this code contain the target"

- **Here:** `find_code`'s `contains_target` asks it about `slice.code`. The slicing call asks the same
  question about `units[i].code`.
- **Proposal:** build both from one wording with the subject path as a parameter. A change to the wording
  then happens in one place, and both questions get a new id.
