# Offline evidence-excerpt study

These scripts measure source presentation. They are development code, not a
library excerpt API. They use JVN's existing parsers, CodeIndex, units and
spelling map. They do not call providers.

The inputs are the pinned 3 October proof archive and its planning repository:

- `~/.local/share/system-one-proof/jvn-eval-2026-10-03`
- `~/.local/share/jvn-takeover/2026-10-03`

Run the census and recorded-read analysis from this worktree:

```sh
proof_root="$HOME/.local/share/system-one-proof/jvn-eval-2026-10-03"
proof_python="$proof_root/checkouts/engine-e733ea0c/.venv/bin/python"
PYTHONPATH=src "$proof_python" evaluations/census.py
PYTHONPATH=src "$proof_python" evaluations/agent_reads.py
PYTHONPATH=src "$proof_python" evaluations/selective_census.py
```

`excerpt_rules.py` owns the shared measurement rules. `census.py` resolves all
201 development and 28 hard tuning deciding lines against the case-1 manifest.
`agent_reads.py` measures literal output from recorded commands without executing
those commands. It imports the corrected shell parser from the planning repo's
`discovery/analyze.py`.

`pack_replay.py` reuses the frozen line-window allocator and stored answers on
their original groups. Its pinned harness paths and command are in the planning
repo's `search-design/excerpts/pack-REPRODUCE.md`. The fitting study owns archived
delivery source bindings; this script reuses those bindings. It verifies the
whole-unit selections before comparing B, D and G25.

The follow-up `selective_rules.py` reuses case 1's `ScentIndex`, tokenizer and
canonical source-unit owner from `~/Projects/jev-navigator-case1`. It ranks
present finding terms by document frequency and seeds exact original finder
citations, then compares local structure, function caps and wider windows.
`selective_census.py` writes a separate census under `excerpts/selective/`.
`pack_replay.py --selective` runs its eighteen variants beside whole and D at
the three uniform rooms. Use the same pinned PYTHONPATH as the original replay.
The new run checks the prior whole/D literal packet hashes as controls and
leaves the original artifacts intact.

Reports, compact TSV evidence and complete local JSON ledgers are written to
the planning repo's `search-design/excerpts/`. The report distinguishes survival
inside known holding units from delivery inside allocated packets, and neither
measurement establishes model comprehension.

Focused verification uses real parser and rendering boundaries:

```sh
PYTHONPATH=src "$proof_python" -m pytest -q evaluations/tests
"$proof_python" -m ruff check evaluations
"$proof_python" -m ruff format --check evaluations
```

Do not run the entire local JVN suite for this study. The project reserves the
full suite for GitHub Actions on a library head intended to merge.
