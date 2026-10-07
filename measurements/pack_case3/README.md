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
