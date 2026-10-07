# Case 2 development recipe

This recipe measures a planner contract, recorded approaches and an explicitly
authorized planner trial. The reusable executor and outline belong to `jev_navigator`; finding inputs,
trace interpretation, labels, replay and packet policy belong here.

Run the candidate study with development-only dependencies:

```sh
uv run --with ijson --with tiktoken python examples/pack_case2/replay.py OUTPUT
uv run --with ijson --with tiktoken python examples/pack_case2/replay.py CORRECTED_OUTPUT --correct-extraction
```

The local proof data lives under `~/.local/share/system-one-proof/jvn-eval-2026-10-03`.
Discovery inputs live under `~/.local/share/jvn-takeover/2026-10-03/discovery`.
Only dev110 and hard27 P1 to P7 and U1 to U6 are selected. Parsing is lazy and
per file. Repository outlines contain paths, declarations and imports rather than
source bodies. Token counts use `cl100k_base` as a proxy, not the planner tokenizer.
Paged prompts disclose omitted outline files. Context construction reads claims,
cited code and admissible repository facts; labels enter only the score.

The supplied first-ten arm preserves original approach ranks and arguments.
The separately labelled corrected-extraction arm reparses shell redirections and
command separators from original command receipts. It only changes arguments
when the original operation and pattern identify one unambiguous search. Genuine
unknown paths remain invalid. Original ledgers are never edited or executed.

The retrospective arm admits all recorded arguments. The strict arm requires
argument components in current context, with exact file paths already observed.
Regex syntax and case/separator changes have explicit provenance. It starts with
the actual paged prompt, then retries pending approaches against code returned by
completed executor rounds. It records blocked approaches after at most three
development rounds. This is a conservative reconstruction, not a cheap model
prediction and not access to the historical agent's unseen context.

The frozen profiles and packet replay are separate prerequisites: JVN #148 at
`677ef3e1` and Engine #1475 at `91f57ab4`. They are not present in the #149 base.
Run `judging.py` and `native.py` with those owners and the frozen `find_eval`,
`navlab` and Engine harness on `PYTHONPATH`, using the Engine virtual environment.
These stages do not edit prerequisite worktrees. Both deny network access.

`judging.py` streams planned items in real search order, with 16 per group and the
final tail. The real Judge still masks, packs and splits oversized groups. Neutral
answers only rehearse physical request shape and never enter delivery. New queues
stay unjudged unless the entire request exactly matches a stored request.
The separate historical-groups development arm retains an original group only
when every original companion body occurs in the planned results. It reuses
that complete historical request, its answers and actual receipt price. It is not
delivery for a reordered planned queue.

`native.py` uses the pinned Engine's actual packet builder and consumer rendering.
An evaluation source supplies candidate anchors; source configuration is restored
after the run. Missing exact answers follow the owner's normal stop and retain
the floor. Packet windows, request bodies and hashes are retained. The offline
masker memoizes the existing pure masker and changes no rules.

`planner.py` renders the prompt and the executor's typed schema for
`sference/deepseek-v4-flash-0731`, then validates replies with the same contract.
Its inputs are the finding, cited code, outline, attempted searches and coverage.
The trial covers our own Analysis Engine code and open-source hard27 repositories.
An approved production LLM route remains an open prerequisite.

The authorized trial uses `paid_planner.py` with the frozen prompt manifest and
Requesty's EU route. It preserves exact requests, responses and reported costs.
`spend.py` reserves each attempt durably before dispatch; interrupted calls stay
reserved until their receipts are reconciled. Completed planner receipts are
skipped on restart. Malformed approaches are logged individually, without changing
their proposed arguments or paying for a repair call.

`paid_execute.py` runs retained plans without provider calls. It reads the pushed
Case 1 scent implementation from a pinned source snapshot, ranks within each plan
rank and scores reach only after execution. It needs the development dependency
`ijson`. `paid_shape.py` runs on the same #148 profile owner as `judging.py`, preparing
up to eight actual 16-item requests per finding, including size splits and tails.
Its neutral answers serve shape rehearsal only. Source places are retained per
physical request occurrence, including identical bodies at different places.

The resumed trial has a $3 total cap, including the existing planner and guard
spend. `union_prepare.py` reuses Case 1's frozen features and held-out weights to
compare plan results, the census and their canonical union at equal source-unit
counts. Unknown source bindings stay unavailable. Outside-census plan units use
a declared ordinal fallback rather than invented scent or graph scores.

`paid_trial.py prepare` runs with the pinned #148 profile on `PYTHONPATH`. It
prepares the first 384 pieces through the real Judge, retaining at most 24 physical
requests per finding. `guard` reviews an exact union request; `judge` dispatches
the immutable groups with a durable reservation before each physical send,
explicitly disabled SDK retries, a final secret scan and exact wire receipts.
Rounds complete four requests across all findings before eight, sixteen and
twenty-four. A cap stop preserves completed lower checkpoints and marks the rest
pending. Recorded guard scores are advisory under André's resumed authorization.

`paid_pack.py` runs on the pinned lab and Engine owners. It decodes the original
six-role answers and filters observations into union, plan-only and ranking-only
views, without asking new groups. It packs each checkpoint into rooms of about
7,200, 20,000 and 36,000 tokens. The lab room is ranked-code allowance plus the
original floor; the native room is the total rendered packet allowance. The
legacy 7,200 native room uses the owner's lower-level explicit-character
allocator. The other native rooms use `EvidencePack.packet()`. Conditional subset
views retain the original union companions and do not claim independent trials.

The spend ledger separates planner, Meta review and candidate Jev charges. The
first-stage receipts and the union-stage receipts remain separate, with no paid
planner repetition. `trial_report.py` derives the marginal curve, per-finding
usage and deciding-line provenance from receipts and actual consumer windows.

Results and the separately priced trial proposal belong in
`~/.local/share/jvn-takeover/2026-10-03/search-design/case2/REPORT.md` and `TRIAL.md`.
