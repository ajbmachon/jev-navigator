# Pack judging, stage 2

The 7 October 09:59 study compares Jev and Flash on Case 1's exact shortlisted
source groups. The lead's 10:30 design check authorizes execution with eight
changes and a $7 cap, including guards. No merge or production profile change.

Use Case 1's environment with the pinned runtime on PYTHONPATH:

- /Users/andremachon/Projects/jev-navigator-role-profile/src
- /Users/andremachon/Projects/analysis-engine-role-profile
- ~/.local/share/system-one-proof/jvn-eval-2026-10-03/jev-lab-roles/find-eval/src
- ~/.local/share/system-one-proof/jvn-eval-2026-10-03/jev-lab-roles/navigator-evals/src
- /Users/andremachon/Projects/jev-navigator-case1/measurements/selection
- /Users/andremachon/Projects/jev-navigator-judging

The frozen requests are in the planning repository's
search-design/judging/design-20261007-admitted-v4. Every archive has 984 original
physical groups. J3 omits inadmissible groups and compares its parent on the
same subset. Earlier unsent archives are superseded preparation drafts.

Run with Case 1's .venv/bin/python -m evaluations.pack_judging.run followed by
proof, guards, pilot, rich-guards, rich-pilot, freeze, full, or report. Hand
inspection receipts must precede finalist freeze and candidate calls. TRIAL.md
owns the exact selection order. Modes resume from exact saved observations.
They reuse the existing Judge, Case 1's JevSession and Ledger, lab allocator and
actual async Engine builder. Reservations precede every physical send.

packing.py binds source ids to JVN CheckResult and FindAllResult. Engine owns
ranking, floor and packet assembly; saved answers incur no provider calls.
Meta copies selected source into unlocated state fragments. LocatedReviewScanner
restores the exact body's original file path while still rejecting literal
secrets, using the existing scanner. No submitted bytes are changed.

Focused verification: pytest evaluations/pack_judging -q, ruff check
evaluations/pack_judging, and ruff format --check evaluations/pack_judging.
The Engine integration test uses the retained real P1 group and controlled
independent answers, with no model calls. Do not run the whole suite locally.
