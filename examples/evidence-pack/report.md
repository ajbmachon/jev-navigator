# Jev navigator evidence pack

- Schema: `jev-navigator.evidence-pack/v1`
- Navigator: `0.1.0` at `0000000000000000000000000000000000000000`
- Revision: `1111111111111111111111111111111111111111`
- Scope: `app/`
- Target: the check that limits how many items an order may have
- Outcome: **found**
- Search: 2 opened places, 2 live calls
- Provider: requested `jev-scripted`, served `jev-scripted`
- Responses without usage: 0
- Requests without a response: 0
- Input tokens: 200
- Navigation elapsed: 0.043 seconds (indexing and entry selection excluded)
- Coverage caveat: 2 candidates were not independently opened; 0 files failed a completed parser scan. Pending parser scans: none.
- Files unavailable (disappeared or changed on disk, or refused by the parser): 0.

## Opened code

| Set | Probability | Verdict | Source |
| --- | ---: | --- | --- |
| found | 0.960 | yes | `app/policy.py:1-2` |
| starts | 0.040 | no | `app/orders.py:3-4` |

## Found spans

### `app/policy.py:1-2`

Raw P(contains target): **0.960**. Reached by `called by handle`.

## Candidates not independently opened

This records separate candidate evaluations, not unseen text. Some candidates were already included in a larger opened span; that does not give them an independent model judgment.

| Why no separate opening | Recorded candidate score | Code coverage | Place |
| --- | ---: | --- | --- |
| Search stopped after finding a match | 0.960 | No containing opened span recorded | `app/orders.py:1-4` |
| Candidate score did not exceed the opening threshold | 0.040 | No containing opened span recorded | `app/orders.py:1-2` |

The complete source spans, raw probabilities, decisions, and history are in `manifest.json`; provider response records and request hashes are in `journal.jsonl`. Each response record’s `exact` flag distinguishes wire capture from SDK-decoded data.

This is an illustrative public-format sample, written by the pack command on a two-file repository.
Repository, revision and navigator provenance are replaced by placeholders.
