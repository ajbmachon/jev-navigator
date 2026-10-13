# Decision models: Jev and Microsoft-Decision-1

JVN asks a decision model narrow yes-or-no questions about code. The two models in use are not interchangeable: they
take different settings, and their probabilities behave differently. This page records how they differ, the settings
that work for each, and how to tune JVN for a model. Every number on this page was measured; each row names where.
`README.md` owns how routes work in general; this page owns what each model needs.

## Contents

| Section | What it answers |
|---|---|
| [At a glance](#at-a-glance) | The facts that differ between the two models |
| [Settings](#settings) | A working route for each model |
| [How their answers differ](#how-their-answers-differ) | What changes in ranking, probabilities and speed |
| [Tuning JVN for a model](#tuning-jvn-for-a-model) | The procedure, in order, for a new model or setting |
| [Evidence](#evidence) | Where each measurement lives |

## At a glance

| | Jev (`jev-latest`) | Microsoft-Decision-1 (`decision-1-eu`) |
|---|---|---|
| Served by | TypeSafe, `api.typesafe.ai` | Our own Azure AI Services deployment |
| Region | Unverified (Cloudflare, undisclosed origin): development only | EU Data Zone (enters Germany West Central): production |
| Answers as | `jev-1.x` (a release of `jev-latest`) | `microsoft-decision-1` |
| Input limit | 32,000 tokens; above it HTTP 400 `max_tokens_exceeded` | 32,768 tokens including its decision token; above it HTTP 422 naming the limit |
| Requests in flight | 32 (no HTTP 429 up to 128) | 1 at 60 requests a minute (HTTP 429 above it, with `Retry-After`) |
| **Items per request** | **16** | **4** |
| Key | TypeSafe API key | The Azure resource key, or an Entra token (lasts about an hour) |
| Price | $0.042 per million input tokens, output free (third-party sources) | Same figure in press reports; Azure's price sheet not yet checked |
| Fine-tuning | Not offered | Not offered |
| Lifetime | Current family alias | Preview; Microsoft retires it on 4 February 2027 |

JVN reads both size refusals (`input_budget_error` in `judgments/client.py`) and splits the request; neither is a
failure. The answer store keys answers by the served model, so the two models' answers never mix.

## Settings

Jev needs only its key; its limits are JVN's defaults:

```sh
export SYSTEM_ONE_ROUTES=jev
export SYSTEM_ONE_JEV=1
export TYPESAFE_API_KEY=...   # from your env file; never print it
```

Decision-1 needs every limit set, because JVN knows nothing about a route it did not measure:

```sh
export SYSTEM_ONE_ROUTES=decision1
export SYSTEM_ONE_DECISION1_ENDPOINT=https://<resource>.services.ai.azure.com/providers/microsoft
export SYSTEM_ONE_DECISION1_MODEL=decision-1-eu
export SYSTEM_ONE_DECISION1_INPUT_TOKENS=32000
export SYSTEM_ONE_DECISION1_CONCURRENCY=1
export SYSTEM_ONE_DECISION1_ITEMS_PER_REQUEST=4
export SYSTEM_ONE_DECISION1_API_KEY=...   # the resource key; or an Entra token for one run
```

- **Key.** The resource's own key works with the `Authorization: Bearer` header JVN sends. An Entra token
  (`az account get-access-token --resource https://cognitiveservices.azure.com`) also works but expires after
  about an hour, so a long search can fail midway; prefer the resource key for anything longer than a few minutes.
- **Name Decision-1 alone** when the work must stay in the EU, so no request falls back to Jev.
- **Raise `CONCURRENCY`** once the deployment's quota is raised; at 60 requests a minute even one request in flight
  meets HTTP 429s, which the client waits out (a `jvn findall` of 150 requests met 25 and lost none).

## How their answers differ

**Batch size is the lever for Decision-1, not question wording.** The table shows the same 39 recorded composed
searches, replayed with only the judge changed. "Truth shown" is the share of labelled deciding code each search
showed its agent.

| Judge | Items per request | Truth shown | Requests per search | Input tokens per search | Against Jev by theme (mean; 90% lower bound) |
|---|---:|---:|---:|---:|---|
| Jev | 16 | 20.6% | 5.4 | 46k | |
| Decision-1 | 16 | 13.8% | 6.6 | 54k | −6.8; −10.3 |
| **Decision-1** | **4** | **22.4%** | 13.6 | 51k | **+1.7; −0.4** |
| Jev | 4 | 22.5% | 11.0 | 48k | +1.3; +0.4 |

- **Decision-1 at 16 items barely separates relevant code from the rest.** Over 5,507 labelled point-unit pairs, its
  mean probability was 0.29 for relevant pairs and 0.26 for the rest (AUC 0.53). At 4 items it was 0.18 against 0.11
  (AUC 0.58); Jev at 4 was 0.20 against 0.13 (AUC 0.57).
- **Decision-1 agrees with Jev more as the batch shrinks.** The median Spearman correlation of a point's ranking with
  Jev's, over 347 points, was 0.47 at 16 items, 0.65 at 4, and 0.77 at 1 (each against Jev asked the same way).
- **Rewording did not help.** Several question designs for the same judgment left Decision-1's ranking where it was;
  only the batch size moved it.
- **Absolute probabilities do not transfer between models.** Decision-1's answers sit lower than Jev's for the same
  pairs. A threshold, band or weight fitted on one model's probabilities (a 0.5 or 0.8 bar, a keep cut, a same-defect
  band) must be refitted on the other's answers before it acts. Rankings within one model are what transfer.
- **The existence question can be answered in code.** Taking the best J1-3 answer over a point's shortlist instead of
  asking Decision-1 a separate existence question kept parity with Jev (21.4% against 20.6%) at a quarter fewer requests
  and tokens.
- **Decision-1 is slower per search** at its current quota: 7.4 s median judge time per search at 4 items, against
  1.8 s for Jev at 16. Cost per search stays near $0.002 at the quoted price, so time, not money, is its budget.

## Tuning JVN for a model

Change one thing at a time, on recorded work with labelled truth, and compare by theme against the reference model.
Report the mean difference with its 90% bootstrap lower bound, never a mean alone.

1. **Measure the route's limits first.** Find the input size it refuses and the shape of the refusal; check that
   `input_budget_error` reads it. Find the concurrency and rate at which it answers HTTP 429. Set
   `INPUT_TOKENS` and `CONCURRENCY` from those measurements, never from a vendor page alone.
2. **Sweep items per request** (16, 8, 4, 2, 1) with everything else fixed. JVN keeps a search's rounds at 16 items and
   sends a capped model the same work in smaller requests (`Judge.sent_per_request()`), so the sweep changes only what
   the model sees at once. Take the largest batch that reaches parity; smaller batches cost more requests.
3. **Then the budget.** A capped model's budget already counts requests of 16 items, so it gets the same rounds as Jev.
   Give it more rounds only if a measured search budget improves truth shown.
4. **Refit every fitted number on the new model's answers.** That covers bars, bands, keep cuts and logistic weights,
   on the same split and by the same procedure as the original fit. A consumer that acts on a fitted number names the
   model it was fitted on and refuses to act on another model's answers.
5. **Keep question wording fixed** unless a measured design study shows a gain. A reworded question is a new question,
   so every stored answer for the old wording stops being reused.

## Evidence

- Replays, pair statistics and the question-design study: heedvane-evals
  `docs/research/theme-agent-jvn-search-2026-10-10.md` (FlowConAi/heedvane-evals#55), scripts in
  `experiments/theme_agent/f_agent_search/` (`judge_replay.py`, `judge_questions.py`).
- Jev and Drex route limits: Analysis Engine `docs/reference/RUNTIME-CONTRACTS.md`, measured 2026-09-26/27.
- Decision-1 route facts and region: the same Analysis Engine contract, checked 2026-10-10.
