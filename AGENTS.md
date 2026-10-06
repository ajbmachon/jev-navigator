# jev-navigator agent rules

JVN is a library of building blocks for searching code. These rules are always loaded; the detail, and
which blocks are built or being built, lives in the README's
[Architecture](README.md#architecture-primitives-pipelines-and-workflows) section.

- **Levels (André, 06.10.2026).** Primitives gather candidate code by code alone. Pipelines (`find`,
  `find_all`, `trace`, `find_text`, `find_all_text`) compose primitives and Jev. Workflows (`jvn search`,
  or a caller's own composition) compose pipelines through typed configurations (being built).
- **Compositions.** A pipeline gathers through sources under one contract, then Jev judges; a new
  source joins through `sources=`, `hops=` or `find`'s `moves=`. Defaults per pipeline:
  [README](README.md#sources-the-frontier-and-each-pipelines-composition).
- **Default.** `jvn search`, being built, is to be the default workflow configuration; every other use
  is another named configuration.
- **Optional steps.** A Jev step (a bounded decision) or an LLM step (generation over an open space) can
  sit between stages. Which steps run is configuration, a recipe the caller passes as data; never an
  environment or deploy flag.
- **Blocks, not copies.** A capability that is not about one caller's domain is a block any caller can
  use. Callers configure blocks; they never reimplement one.
- **No caller domain.** Nothing finding-, theme- or Engine-specific lives in JVN.
- **Experiments** compare named configurations, never tweaks inside one call.
- **Checks (André, 05.10.2026).** Locally run only the tests that cover or import changed files and ruff;
  never the whole suite on this Mac. The full suite runs on GitHub Actions (`gh workflow run tests.yml`
  on the branch) once the head is the one to merge, judged by its log. Detail: [Tests](README.md#tests).
- **One PR per coherent change (André, 05.10.2026).** While it is open, the next step goes onto it as
  commits; a stack collapses into its top PR, and the full suite runs once, on the head to merge.
