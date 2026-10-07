# Pack judging, stage 2

This is the zero-cost design slice for the 7 October 09:59 ruling. It freezes
Case 1's fresh code arm, builds alternative judgments over the same physical
groups, runs free Meta preparation on concrete design prototypes, and prices
the proposed calls. There is no paid execution mode. Stop for the lead's design
check before adding or invoking paid dispatch.

The prepared artifact directory is
`~/.local/share/jvn-takeover/2026-10-03/search-design/judging/design-20261007-actual-consumer`.
`REPORT.md`, `TRIAL.md` and `DESIGN-REVIEW.json` beside it own the review packet.
Earlier preparation directories are incomplete drafts and are not admitted inputs.

Use the existing Case 1 environment and pinned profile, Engine and find-eval
runtime. This branch started at Case 1's `a7bb65f44374bec240093aa2a961f754218854c4`;
it does not copy the newer Judge/profile implementation into evaluation code.

```sh
export PACK_JUDGING_RUNTIME=/Users/andremachon/Projects/jev-navigator-role-profile/src:/Users/andremachon/Projects/analysis-engine-role-profile:/Users/andremachon/.local/share/system-one-proof/jvn-eval-2026-10-03/jev-lab-roles/find-eval/src:/Users/andremachon/Projects/jev-navigator-judging
PYTHONPATH="$PACK_JUDGING_RUNTIME" /Users/andremachon/Projects/jev-navigator-case1/.venv/bin/python -m pytest evaluations/pack_judging/test_contract.py -q
/Users/andremachon/Projects/jev-navigator-case1/.venv/bin/ruff check evaluations
/Users/andremachon/Projects/jev-navigator-case1/.venv/bin/ruff format --check evaluations
```

Preparation requires the current saved EU model catalogue in
`search-design/judging/provider-model.json` and `system-one-meta-builder` 0.6.0.
Use a new output directory; preparation refuses to overwrite frozen inputs.

```sh
PYTHONPATH="$PACK_JUDGING_RUNTIME" /Users/andremachon/Projects/jev-navigator-case1/.venv/bin/python -m evaluations.pack_judging.prepare --out /absolute/path/to/a/fresh/directory
```

`designs.py` composes JVN Check and Pick contracts. `consume.py` uses the existing
async Judge and preserves primitive-specific observations. `frozen.py` joins
actual sent baseline requests to raw provider answers and source addresses.
`prepare.py` generates every candidate group and retains canonical identities,
admissibility and prices. It prepares two full-group prototypes per Jev variant,
one on each dataset. Prototype review is not review of every changing state.
The packet supplies the real consumer. Meta cannot resolve its dynamic JVN
answer reads, so workflow coverage remains explicitly unreviewed. Local tests
execute that submitted code with JVN typed responses; SDK-field snippets are
not substitutes for this interface.

The paid continuation must reuse Case 1's reservation ledger and receipt
transport. It must scan the complete semantic LLM input with `scan_llm_request`
immediately before reservation and dispatch. A JSON serialization is not a new
source representation to mask. Missing, malformed or refused answers remain
failed attempts, never zero scores. Exact responses must be saved before parsing.

The score consumers are implemented and tested. The lab and native packing
bridge for the new scalar observations is still a continuation task, to be
proved at the actual Engine entry point before the pilot. Reuse Case 1's
allocator and Engine recipe rather than implement another allocator or Judge.
