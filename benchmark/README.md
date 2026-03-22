# Sulci Benchmark Suite

Reproducible benchmarks for [Sulci](https://github.com/id4git/sulci) — semantic caching for LLM apps.

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

---

## Results (v0.2.1)

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

Claude API:
  --use-claude          Call Claude on misses + verify hits against live responses
                        Requires: ANTHROPIC_API_KEY, pip install anthropic
  --claude-model MODEL  Model to use (default: claude-haiku-4-5-20251001)
  --claude-max-calls N  Hard cap on API calls to bound cost (default: 500)

Context benchmark:
  --context             Run context-aware benchmark (800 conversation pairs)
  --context-window N    Turns per session (default: 4)
  --context-threshold F Context similarity cutoff (default: 0.58)
```

---

## Output Files

All written to `benchmark/results/` (or `--out` directory):

| File | Description |
|------|-------------|
| `summary.json` | Overall stateless benchmark stats |
| `domain_breakdown.csv` | Per-domain hit rates, FP rates, cost savings |
| `threshold_sweep.csv` | Hit rate vs threshold 0.70–0.95 |
| `time_series.csv` | Hit rate evolution over query batches |
| `false_positives.csv` | Near-miss analysis (top 100) |
| `context_summary.json` | Context-aware benchmark results (`--context`) |
| `context_accuracy.csv` | Per-domain resolution accuracy (`--context`) |

`*.json` and `*.csv` result files are gitignored. The `results/` directory
contains only a `.gitkeep` in the repository.

---

## Stale DB Warning

When running with `--use-sulci`, the SQLite benchmark database persists
between runs in `benchmark/results/sulci_bench_db`. If you run the benchmark
twice without `--fresh`, the second run's warmup phase writes on top of an
already-populated cache, causing every test query to hit — producing an
artificially inflated hit rate (100%) and zero misses.

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
git clone https://github.com/id4git/sulci && cd sulci

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

*MIT License — Sulci v0.2.1 — github.com/id4git/sulci*
