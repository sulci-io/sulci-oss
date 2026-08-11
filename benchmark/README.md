# Sulci Benchmark Suite

Reproducible benchmarks for [Sulci](https://github.com/sulci-io/sulci-oss) — semantic caching for LLM apps.

Three progressive modes: synthetic (no dependencies), real embeddings, and real Claude API calls.

---

## Quick Start

```bash
# Zero dependencies — runs anywhere
python benchmark/run.py

# With real MiniLM embeddings (recommended)
pip install "sulci[sqlite]"
python benchmark/run.py --use-sulci

# With real Claude API calls on misses
pip install "sulci[sqlite]" anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python benchmark/run.py --use-sulci --use-claude --fresh
```

---

## Benchmark Modes

### Mode 1 — Synthetic (default)

No install required. Uses a built-in TF-IDF cosine engine to simulate
sentence-transformer embeddings across a ~200-word domain vocabulary.
Correctness is scored by query group labels. Fast (~30s for 5k queries).

```bash
python benchmark/run.py
python benchmark/run.py --no-sweep --queries 1000   # fast CI version (~5s)
```

### Mode 2 — Real embeddings (`--use-sulci`)

Swaps the TF-IDF engine for `sulci.Cache` with SQLite + all-MiniLM-L6-v2.
Hit/miss decisions use real 384-dimensional sentence-transformer embeddings.
No API key required. Takes 2–5 minutes for 5k queries (model load on first run).

```bash
pip install "sulci[sqlite]"
python benchmark/run.py --use-sulci
python benchmark/run.py --use-sulci --context   # + context-aware benchmark
```

> **Always use `--fresh` with `--use-sulci`** to prevent stale benchmark DB
> inflation across consecutive runs. See [Stale DB Warning](#stale-db-warning) below.

### Mode 3 — Real Claude API (`--use-claude`)

Requires `--use-sulci`. On cache misses, calls the Claude API to get real
responses and records actual API round-trip latency. On cache hits, calls
Claude to semantically verify the cached response against a live answer.

```bash
pip install "sulci[sqlite]" anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Recommended: full verified run
python benchmark/run.py --use-sulci --use-claude --fresh \
  --queries 1000 --no-sweep --claude-max-calls 1000
```

**Cost:** approximately $0.90 per 1,000-query run with Haiku
(~$0.80/1M input + $4.00/1M output tokens).

### Mode 4 — Agent workload (`--agent`)

Simulates a realistic mixed-workload agent's LLM dispatch pattern across
50 sessions × 200 dispatches per session = 10,000 total dispatches.
Measures the per-session deduplication rate that maps directly to the
"X dispatches → Y LLM calls" framing on the homepage.

Workload mix (calibrated to public agent-traffic measurements):

| Category | Weight | Examples |
|---|---|---|
| Structural | 45% | Planner, reflector, system-prompt-like prompts. High cacheability — small param pools, frequent semantic repetition. |
| Semi-structural | 35% | Tool-call decisions, intermediate reasoning. Moderate cacheability — parameterized template-bound prompts. |
| Novel | 20% | Task-specific reasoning, user-input-derived prompts. Low cacheability — large param pools, novel content per dispatch. |

```bash
# Fast synthetic mode (TF-IDF, no dependencies) — CI baseline
python benchmark/run.py --agent

# Real-MiniLM mode — produces the conservative number cited externally
pip install "sulci[sqlite]"
python benchmark/run.py --agent --use-sulci

# Real-Anthropic mode — for blog-post / whitepaper anchor
pip install "sulci[sqlite]" anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python benchmark/run.py --agent --use-sulci --use-claude \
  --claude-max-calls 5000
```

**Outputs:** `agent_summary.json` + `agent_per_session.csv` (per-session
hit/miss/hit-rate). Cold→warm→hot session progression is the primary
visual — sessions 1-5 show cache filling; sessions 40-50 show
steady-state hit rate.

**Two scaling axes** (both have flags for tuning):
- `--agent-sessions N` — default 50. More sessions = more saturation
  (steady-state hit rate increases).
- `--agent-dispatches N` — default 200. Maps to the homepage "200 calls
  per session" framing. Larger = each session takes longer to ramp up.

**Engine choice matters significantly:**

| Mode | Aggregate | Cold session | Warm session | Notes |
|---|---|---|---|---|
| TF-IDF (default) | 95.0% | 43% | 97.9% | CI baseline, `baseline.json`. |
| Real MiniLM (`--use-sulci`) | **95.0%** | **27%** | **99.4%** | MEASURED 2026-08-04. |
| Real Claude (`--use-claude`) | as above | as above | as above | Adds real-LLM latency and cost-saved numbers; does not change hit rates. |

⚠️ **This table previously said real MiniLM gives "~60-75% aggregate" and
called it "the number to cite externally". That was never measured.** Run on
2026-08-04 it gives 95.0% — the same as the TF-IDF baseline it was described as
being conservative against. That one line is where the 60-75% figure in the
site hero, the SI pitch deck and the email signature came from.

The "upper bound" framing went with it: both engines score ~95% here, so TF-IDF
is not an upper bound on the agent workload.

⚠️ **The agent aggregate is not a good external number in either engine.** The
`novel` category — dispatches the generator labels as new work — hits 92.6% on
MiniLM, because `max_unique_combinations ≈ 150` across 2,025 dispatches means
each "novel" prompt recurs about thirteen times. The aggregate largely measures
template reuse.

**The cold→warm progression is the real result**: 27% → 99.4%. An agent loop
re-asking the same things converges fast, and that is structurally true rather
than an artifact of the generator.

---

## Results (v0.4.0)

All results produced with `--use-sulci --use-claude --fresh --queries 1000 --no-sweep --claude-max-calls 1000`.

### Stateless benchmark

| Metric | Value | Notes |
|--------|-------|-------|
| Hit rate | **94.0%** | Real MiniLM embeddings, clean DB |
| Hit latency p50 | **~0.74ms** | Pure cache lookup, no verification overhead |
| Miss latency p50 | **~2,700ms** | Real Haiku API (unqueued baseline) |
| False positive rate | **6.04%** | Group-label correctness across 986 queries |
| Cost saved / 1k queries | **$4.63** | At $0.005/LLM call |
| Semantic accuracy | **51.3%** | Lower bound — see note below |

**Domain breakdown:**

| Domain | Hit Rate | FP Rate | Cost Saved |
|--------|----------|---------|------------|
| customer_support | 89.6% | 0.0% | $0.94 |
| developer_qa | 98.9% | 10.8% | $0.93 |
| product_faq | 92.4% | 0.6% | $0.91 |
| medical_information | 95.3% | 0.0% | $0.92 |
| general_knowledge | 94.4% | 18.8% | $0.93 |

**On semantic accuracy (51.3%):** This is a lower bound, not the true false
positive rate. The benchmark cache stores short synthetic responses (~12 words).
Claude returns verbose paragraph answers (~150 words). TF-IDF cosine similarity
between short and long text is structurally low regardless of meaning — the token
distribution dilutes overlap. The 6.04% group-label false positive rate is the
more reliable correctness signal for this corpus.

### Context-aware benchmark

| Metric | Stateless | Context-Aware | Delta |
|--------|-----------|---------------|-------|
| Hit rate | 64.0% | 81.6% | +17.6pp |
| Resolution accuracy | 56.8% | 77.6% | +20.8pp |

Domain resolution accuracy:

| Domain | Stateless | Context-Aware | Delta |
|--------|-----------|---------------|-------|
| customer_support | 32% | 88% | +56pp |
| developer_qa | 80% | 96% | +16pp |
| medical_information | 40% | 60% | +20pp |

---

## All CLI Options

```
python benchmark/run.py [OPTIONS]

Core:
  --queries N           Test query count (default: 5000; warmup = equal count)
  --threshold F         Stateless similarity cutoff (default: 0.85)
  --no-sweep            Skip threshold sweep — faster, use for CI
  --out DIR             Results directory (default: benchmark/results)

Embedding engine:
  --use-sulci           Use sulci.Cache + MiniLM instead of built-in TF-IDF
  --fresh               Delete existing benchmark DBs before running
                        (prevents stale-cache hit rate inflation with --use-sulci)
  --seed N              Corpus RNG seed (default 42). Varies which groups are
                        held out, so a result can be checked across draws.

Claude API:
  --use-claude          Call Claude on misses + verify hits against live responses
                        Requires: ANTHROPIC_API_KEY, pip install anthropic
  --claude-model MODEL  Model to use (default: claude-haiku-4-5-20251001)
  --claude-max-calls N  Hard cap on API calls to bound cost (default: 500)

Context benchmark:
  --context             Run context-aware benchmark (125 follow-ups, see below)
  --context-window N    Turns per session (default: 4)
  --context-threshold F Context similarity cutoff (default: 0.58)
  --context-holdout N   Sessions per domain left UNWARMED (default: 1)
  --context-followups N Follow-ups per session (default: 5, clamped by the pools)
  --context-sweep       Sweep --query-weight, recording accuracy AND false-hit
```

---

## Discrimination metrics — MEASURED 2026-08-04

`summary.json` reports three numbers instead of one hit rate. A single hit rate
cannot distinguish a good cache from one that answers everything: both score
high.

| | |
|---|---|
| `recall` | of queries that SHOULD hit, how many did |
| `false_hit_rate` | of queries that should MISS, how many hit anyway — the harmful case, where a user receives someone else's answer and acts on it |
| `precision` | of all hits, how many matched the right group |

Four corpus draws (`--seed 1 2 3 42`), threshold 0.85, 39% of the test set
having no correct answer cached:

| | MiniLM (shipped) | TF-IDF (default) |
|---|---|---|
| recall | **0.9990** [.9985–.9995] | 0.667 [.641–.684] |
| false-hit | **0.0106** [.0062–.0185] | 0.304 [.274–.365] |
| precision | **0.9377** [.9213–.9469] | 0.284 [.265–.303] |

Per 1,000 queries with no cached answer, MiniLM wrongly answers **11**; TF-IDF
wrongly answers **304**.

⚠️ **2026-08-12 — "four corpus draws" applies unevenly.** True of the stateless
pass. **Was not true of the agent pass** until 2026-08-12: `run.py` omitted
`seed` when calling `run_agent_bench`, so every agent draw used the hardcoded
`1729`. Fixed; the agent figures now vary across seeds and cold measures
`33.4% [30.0–35.0]` against a previously published `27%`.

Still not fully true of the **context** pass. `run_context_bench` receives
`args.seed` and resolves `None` to `99` at `:999`, but `:1029` uses
`random.Random(99)` unconditionally while `:1459–1512` use the module RNG. Two
RNGs, one seedable and one pinned. The context draws do vary across seeds, so it
is partially seeded — but "four corpus draws" claims more independence than the
code delivers. The context-aware family is retired in `CLAIMS.md`, so nothing
published rests on it.


⚠️ **`general_knowledge` is the weak domain at ~25% false positives** — "what
is AI" and "what is machine learning" are adjacent enough to conflate. Reported
rather than hidden.

⚠️ **The bare `hit_rate` is no longer meaningful** and should not be quoted. It
now averages recall and false-hit over a corpus that is 39% should-miss by
design.

**Cost:** roughly 10 minutes per `--use-sulci` run, 20+ with `--context`. This
is a nightly or on-demand job, not a per-PR one.

---

## Output Files

All written to `benchmark/results/` (or `--out` directory):

| File | Description |
|------|-------------|
| `summary.json` | Overall stateless benchmark stats |
| `domain_breakdown.csv` | Per-domain hit rates, FP rates, cost savings. `general_knowledge` is the weak domain — see above. |
| `threshold_sweep.csv` | Hit rate vs threshold 0.70–0.95 |
| `time_series.csv` | Hit rate evolution over query batches |
| `false_positives.csv` | Near-miss analysis (top 100) |
| `context_summary.json` | Context-aware benchmark results (`--context`) |
| `context_accuracy.csv` | Per-domain resolution accuracy (`--context`) |
| `context_alpha_sweep.csv` | Accuracy **and false-hit** vs `query_weight` (`--context-sweep`) |

`*.json` and `*.csv` result files are gitignored. The `results/` directory
contains only a `.gitkeep` in the repository.

---

## Stale DB Warning

When running with `--use-sulci`, the SQLite benchmark database persists
between runs in `benchmark/results/sulci_bench_db`. If you run the benchmark
twice without `--fresh`, the second run's warmup phase writes on top of an
already-populated cache, causing every test query to hit — producing an
artificially inflated hit rate (100%) and zero misses.

⚠️ **The same shape was true of the CORPUS itself until 2026-08-04.** Every
test query was a cosmetic variant of a warmup query and every group was warmed,
so every test query had a same-group twin already cached — "an artificially
inflated hit rate and zero misses", from the corpus rather than the database.
`--fresh` could not help, because the problem was not stale state. Held-out
groups and near-miss pairs fixed it.

**Always use `--fresh` for canonical benchmark runs:**

```bash
python benchmark/run.py --use-sulci --fresh
```

`--fresh` is safe to use at any time. It prints each removed DB path so you
can confirm what was cleared.

---

## Reproducing the Published Numbers

```bash
# Install
pip install "sulci[sqlite]" anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Clone
git clone https://github.com/sulci-io/sulci-oss && cd sulci-oss

# Stateless + context benchmark (synthetic, no API key needed, ~2 min)
python benchmark/run.py --context

# Verified run with real embeddings + real Claude API (~25 min, ~$0.90)
python benchmark/run.py \
  --use-sulci --use-claude --fresh \
  --queries 1000 --no-sweep \
  --claude-max-calls 1000

# Fast CI check (~30s, no install needed)
python benchmark/run.py --no-sweep --queries 500
```

---

## Methodology Notes

- **Warmup:** equal number of queries to `--queries` are used to warm the cache
  before measurement begins, mirroring real production conditions.
- **Correctness (synthetic):** a cache hit is "correct" if the matched entry
  belongs to the same query group as the test query.
- **Correctness (Claude mode):** semantic cosine similarity (TF-IDF) between
  the cached response and the live Claude response, threshold 0.28. This is a
  conservative lower bound — see semantic accuracy note in results above.
- **Threshold sweep:** disabled automatically in `--use-claude` mode to prevent
  excess API calls. Run separately with `--no-sweep` removed if needed.
- **Context benchmark:** uses brute-force cosine scan (not LSH) to avoid false
  negatives on the small context corpus (<300 entries). Threshold 0.58 is
  calibrated for TF-IDF blended vectors; real embeddings may warrant 0.70–0.75.

---

## The context benchmark — size, hold-outs, and the alpha sweep

### It is 125 follow-ups, not 800 pairs

⚠️ The module docstring said "800-pair" from v0.2.0 until 2026-08-06 and the
corpus has never been that size. There are 25 sessions (5 domains × 5) and every
`SESSION_FOLLOWUPS` pool holds exactly 5 entries, so `min(n_followups,
len(pool))` clamps the draw to 5 and the corpus is 125 rows regardless of
`--context-followups`. Setting it higher now prints the clamp rather than
leaving it to be inferred. **Growing this corpus means writing follow-ups.**

The retired `+20.8pp`, `+17.6pp` and `+56pp` figures came from this corpus. They
were 125 samples across 5 domains, and nothing printed the n.

### Hold-outs, and why a resolution-accuracy delta cannot stand alone

`--context-holdout N` leaves N sessions per domain **unwarmed**. They are primed
and queried identically; the only difference is that no correct answer exists
for them. Every hit they produce is a **false hit** — the blended lookup vector
drifted onto a neighbouring session and returned an answer to a question the
user did not ask.

This is the only thing that separates *"context resolved the follow-up"* from
*"context made the cache answer everything"*. Both raise resolution accuracy on
rows where an answer exists. Before 2026-08-06 the context benchmark had no
should-miss rows at all, so `false_hit_rate` did not exist.

`--context-holdout 0` reproduces the pre-2026-08-06 corpus. It reports
`false_hit_rate: null`, **not** `0.0` — an unmeasured rate and a measured zero
are different claims and must not render the same.

⚠️ `resolution_accuracy` is computed over should-hit rows only. Held-out rows can
never be resolved correctly, so folding them in would drag the number down
mechanically and make it incomparable with anything measured before hold-outs
existed. Their contribution is `false_hit_rate`, on its own axis.

### Reading `context_alpha_sweep.csv`

`--context-sweep` runs the context benchmark at eight values of `query_weight`
and records accuracy **and** false-hit at each. Read both columns.

**A lower alpha that raises accuracy and false-hit together has not found more
answers — it has loosened the cache.** Measured on the **built-in TF-IDF engine**
(`--context --context-sweep`, holdout 1, 100 should-hit / 25 should-miss):

| `query_weight` | ctx accuracy | ctx recall | ctx false-hit | ctx precision |
|---|---|---|---|---|
| 0.20 | 97.0% | 97.0% | **92.0%** | 97.5% |
| 0.30 | 97.0% | 97.0% | **92.0%** | 97.5% |
| 0.40 | 94.0% | 97.0% | **92.0%** | 95.0% |
| 0.50 | 94.0% | 97.0% | **92.0%** | 95.0% |
| 0.60 | 80.0% | 85.0% | **64.0%** | 92.1% |
| **0.70 (shipped default)** | 81.0% | 83.0% | **32.0%** | 94.5% |
| 0.80 | 70.0% | 76.0% | **20.0%** | 88.9% |
| 0.90 | 65.0% | 71.0% | **12.0%** | 87.8% |

At 0.20 the cache answers 92% of the questions it has never seen an answer for.
The accuracy gain is bought, not earned.

⚠️ **These are TF-IDF numbers. The equivalent `--use-sulci` sweep has not been
run.** The two engines are not interchangeable and this table must not be quoted
as a MiniLM result. Run:

```bash
python benchmark/run.py --use-sulci --fresh --context --context-sweep --no-sweep --queries 500
```

⛔ **Do not change the shipped `query_weight` default on either table alone.**
125 samples across 5 domains is not enough to move a default that every context
user inherits.

### `--context` without `--use-sulci` was broken until 2026-08-06

`_BuiltinContextCache.__init__` accepted `query_weight` and never stored it;
`_get_session` then read a bare `query_weight`, which is not a global. The
documented no-install path — the one a technical buyer runs first — raised
`NameError` at the first session. Every context figure ever quoted came from the
`--use-sulci` arm. Fixed 2026-08-06.

---

*Apache 2.0 License — Sulci — github.com/sulci-io/sulci-oss*
