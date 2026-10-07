# Case 3 development replay

Run from a clean committed checkout:

```sh
uv run --with tiktoken python measurements/pack_case3/replay.py \
  --out ~/.local/share/jvn-takeover/2026-10-03/search-design/case3/replay-final-24k \
  --capacity 24000
```

This reads only dev110, its masked pack inputs and existing discovery receipts. No recorded shell
command is executed. Provider sockets are rejected after loading the free reference tokenizer.
Outputs keep all 110 workload entries, including cases without traces. Each actual batch response,
page, operation-to-command mapping and literal deciding-line location is recorded. No reference
label chooses an operation or its input. Source, fixture and ledger hashes are captured at startup.

Compression batches up to ten recorded operations using future trace inputs. Conservative replay
starts another group when later inputs have not appeared in the supplied claim or previous output.
Both exhaust every continuation before the next group. These are development proxies, not a new
agent policy, an absence proof or a measured Engine consumer. Unsupported commands remain listed.
Broad grep replay can return more text than the original shell pipe's head limit. JVN records the
complete match count and all continuations rather than silently treating that limit as all matches.

The report compares total history calls with literal source delivery in the first five calls.
First-delivery call numbers are diagnostic label scoring, not a stopping condition supplied to an
agent. `cl100k_base` token counts compare returned text, and are not the DeepSeek or Jev billing
counter. Cold compression runs before warm conservative replay; wall time is local operation time,
not agent reasoning time or HTTP time. No held-out data or paid model is used.

Freeze a proposed trial sample and representative rank request without a provider:

```sh
uv run python measurements/pack_case3/prepare_trial.py --out /path/to/new/trial-folder
system-one-meta-builder prepare /path/to/new/trial-folder/rank-candidate.json \
  > /path/to/new/trial-folder/meta-prepared.json
```

The deterministic stratified sample contains six local searches, eight cross-boundary searches,
four searches needing terms learned from earlier reads and two cases without recorded traces.
SHA256 of `case3-agent-trial-20261007:<case id>` orders each pool. Existing discovery annotations
select strata; labels and historical answers never enter agent inputs or candidate request state.
The rank preparation is an illustrative first batch over cited-file symbols in source order,
with long symbols split into 60-line ranges. Its stand-in probabilities are not saved or measured.
Future adaptive batches are new exact requests and require their own review.

The requested cheap-agent trial is priced and documented in the takeover Case 3 report. It uses
`sference/deepseek-v4-flash-0731`, at most five batched tool calls per finding, a 24,000-character
response cap and an optional total of eight Jev requests per finding. Calls and cumulative billed
tokens are guards, never elapsed time. No guard or paid agent trial is part of these scripts.

The separately authorized adaptive pilot uses `trial.py`. It reads the frozen sample and prompt,
continues the existing spend ledger, and exposes only the documented batched tool to the pinned
Requesty EU model. The provider model catalog is fetched freely before dispatch. Each agent request
reserves the catalog's complete context-window input bill plus its output allowance, then settles
reported uncached, cached and output tokens. Unknown usage retains its reservation and stops new
spend. Optional rank uses the existing Judge, one in-flight group, an isolated persistent answer
store and exact SDK request and response bytes. No ranking question is changed.

```sh
uv run --env-file /path/to/requesty.env --env-file /path/to/typesafe.env \
  --extra typesafe --with tiktoken python measurements/pack_case3/trial.py \
  --case-folder ~/.local/share/jvn-takeover/2026-10-03/search-design/case3 \
  --out /path/to/new/authorized-trial-folder
```

This command makes paid calls. Run it only under the specific recorded approval. The trial reads no
labels. Each search saves every agent and tool request and response, actual literal source lines,
its final response, reported token usage, separate agent and Jev dollars and complete wall time.
Post-run scoring joins those literal lines to the already frozen deciding references.

After every search has ended, score the saved literal source against the existing dev110 references:

```sh
uv run python measurements/pack_case3/score_trial.py \
  --case-folder ~/.local/share/jvn-takeover/2026-10-03/search-design/case3 \
  --run /path/to/completed/trial-folder
```

The scorer verifies actual source text and unchanged source revision, reconciles provider-reported
agent dollars, and writes per-finding and per-line comparisons beside the lab and native baselines.
It makes no paid call. Reference locations never enter the search runner.

The 7 October pilot corrected an unintended 2,000-token per-response ceiling by continuing only the
eight affected transcripts under their original remaining 8,000-token case allowance. The repaired
runner uses that entire remaining allowance. The `--resume-output-caps` switch preserves prior tool
calls, billing and literal source receipts. It is not permission to repeat completed searches.
Elapsed resumed wall time includes the operator repair pause; active wall time is reported separately.
The nominal input envelope is checked against reported usage after a response, so the report retains
any last-request overshoot. The hard dollar cap is reserved before every paid request.
