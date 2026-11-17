# Observe mode: measure before you serve (1.5.0)

A semantic cache saves money only when your users repeat themselves, and only you have your users' traffic. Observe
mode answers "how much would Sulci save us, and would it ever serve a wrong answer?" on that traffic, inside your own
network, before a single cached answer reaches a user.

## Turn it on

```python
from sulci import Cache

cache = Cache(backend="qdrant", threshold=0.85, observe=True)                  # records to ./sulci_observe.db
cache = Cache(backend="sqlite", observe=True, observe_path="/var/lib/app/sulci_observe.db")
```

Everything else stays as you would run it live: the same backend, embedder, threshold, `personalized`, sessions and
`context_window`. `AsyncCache(observe=True, …)` passes it through, and so do the LangChain, LlamaIndex, LiteLLM and MCP
integrations, which take a `Cache` or its arguments.

For an app in another language, or one you would rather not change, run the proxy and point the app's OpenAI or
Anthropic base URL at it:

```bash
sulci-proxy --backend qdrant --observe --observe-path /var/lib/app/sulci_observe.db
```

`observe=True` is for self-hosted backends. On Sulci Cloud (`backend="sulci"`) it raises `TypeError`: switch the API
key to Observe mode in the dashboard (Savings preview) instead, which records the same things there.

## What it does

- **Every lookup runs, and is then turned away.** `get` returns `(None, similarity, depth)`, `cached_call` calls your
  LLM, and the proxy forwards upstream. Your users get your LLM's answer, every time.
- It counts as a miss everywhere: `stats()` (hits stay 0), telemetry, and your event sink, whose `miss` event carries
  `metadata={"observe": True}`. `stats()["observe"]` holds the observed counts.
- **The lookup searches down to a floor of 0.75** (or your threshold, if lower), so the report can show what a lower
  or higher threshold would do. Each call is labelled against your threshold: `would_hit`, `would_miss`, or
  `would_refuse` when your own `accept=` check said no (it is still run, and never acted on).
- **After a would-be hit, the `set` that follows stores nothing,** as after a live hit, which stores nothing. So the
  cache grows as it would live. The set completes a review pair instead (below). Any other `set` stores as before,
  follow-up rule included.
- The token counts of the call that answered (from `set(..., usage=)`, a `sulci.with_usage(...)` answer, or the
  proxy's upstream reply) are recorded on the lookup: they are what a hit would have saved.
- A would-be hit records no conversation turn (it was a miss); the `set` that follows records your LLM's answer as the
  turn, so the next lookup blends what the user actually saw.

## What is stored, and where

One SQLite file (no extra dependency), created readable by its owner only (0600), in WAL mode so several worker
processes on one host can write it. Nothing is sent to Sulci or anyone else.

| Table | What | Text? |
|---|---|---|
| `observe_run` | one row per `Cache` started: the sulci version, backend, embedder, threshold, context settings | No |
| `observe_events` | one row per lookup and per `set`: label, similarity, thresholds, timing, token counts, SHA-256 hashes of the question and the scope | No |
| `observe_pending` | a would-be hit's cached answer, kept up to 10 minutes for the `set` that answers it | Yes, until matched or expired |
| `observe_pairs` | review pairs: the question, the answer the cache would have served, your LLM's answer, the similarity, and the rating | **Yes** |

Review pairs are kept only for would-be hits at similarity 0.80 or above, **at most 400 per file, as a uniform random
sample** of them; a rated pair is never replaced. The file is not encrypted by the library: keep it where you keep the
traffic it measures.

**Erasure.** `cache.delete_user(user_id)` also deletes that user's pairs and pending answers (its result gains
`"observe_pairs"`); `cache.clear()` deletes every pair and pending answer; `python -m sulci.observe forget FILE` does
the same from the command line and compacts the file. The measured numbers (no text) stay. An erasure that cannot be
confirmed raises; recording that fails is logged (its type only) and never breaks the call.

Several hosts write several files: `report` and `summary` take them all and add them up.

## Rate, report, summarise

```bash
python -m sulci.observe status  sulci_observe.db
python -m sulci.observe rate    sulci_observe.db --reviewer alice
python -m sulci.observe export-pairs   sulci_observe.db --csv pairs.csv     # or rate in a spreadsheet…
python -m sulci.observe import-ratings sulci_observe.db --csv pairs.csv     # …and read the ratings back
python -m sulci.observe report  sulci_observe.db
python -m sulci.observe summary sulci_observe.db host2.db --out run.json
python -m sulci.observe forget  sulci_observe.db
```

(`sulci-observe` is the same command, installed with the package.)

**Rating.** For each pair your reviewer sees the question, the answer the cache would have served and your LLM's
answer, in random order and without the similarity, and answers one question: *would you send the cached answer to
this asker, unchanged?* **Acceptable**, **Not acceptable** or **Can't tell**. An answer that would need editing is not
acceptable. Can't tell is reported on its own and never counted as acceptable. About 100 rated pairs is enough for a
useful bound.

**The report** (Markdown) shows, at thresholds 0.75 to 0.95:
- would-be hits: the requests saved, and the hit rate over all calls and over the calls that were looked up;
- input and output tokens saved, with the share of would-be hits whose token counts were known;
- people's ratings, and the 95% upper bound on the wrong-hit rate (Wilson score interval);
- calls not counted, by reason: streams the proxy never looked up, refusals by your `accept`, follow-ups not stored;
- the run's settings and window, and whether it reached the minimums (500 looked-up calls, 100 rated pairs).

It states measured numbers only, with no prices and no verdict.

**The summary** (`run.json`) holds the same numbers and **no text and no hashes**: not a question, an answer, a user id
or a hash of any of them. It is yours to keep, or to share with Sulci if you want the Savings preview report (savings
at your prices, a tuning review and a verdict) without sending your traffic.

## Read the numbers with care

- Similarity comes from your embedder and backend; the numbers describe this setup. The report names both.
- Entries live as long as your `ttl_seconds` (default 24 h), so only repeats within that window count.
- The hit rate at a threshold below your own is what a lower threshold would have served; its wrong-hit bound comes
  only from pairs rated in that range. Pairs are kept from 0.80 up, so below 0.80 there are no ratings.
- Requests saved assume one LLM call per lookup; if a call carries several questions, count accordingly.
