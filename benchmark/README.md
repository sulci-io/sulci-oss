# Sulci Benchmark

10,000-query semantic cache benchmark across 5 domains.  
Includes a dedicated **context-aware benchmark** (v0.2.0) that measures follow-up
resolution accuracy with and without conversation history.

Runs entirely without API keys, cloud accounts, or GPU.

---

## Quick start

```bash
# Stateless benchmark — no dependencies required
python benchmark/run.py

# + context-aware benchmark (v0.2.0)
python benchmark/run.py --context

# With real sulci embeddings (sentence-transformers MiniLM)
pip install "sulci[sqlite]"
python benchmark/run.py --use-sulci --context
```

---

## Options

```
--use-sulci              Use sulci.Cache with SQLite + MiniLM instead of built-in TF-IDF
--threshold N            Similarity threshold for stateless benchmark (default: 0.85)
--queries N              Number of test queries (default: 5000, warmup = same)
--no-sweep               Skip threshold sweep (saves ~7 minutes)
--context                Run context-aware benchmark
--context-window N       Turns to remember per session (default: 4)
--context-threshold F    Similarity threshold for context benchmark (default: 0.58)
--out DIR                Output directory (default: benchmark/results)
```

---

## Output files

Written to `benchmark/results/` (gitignored — regenerate locally):

| File | Contents |
|---|---|
| `summary.json` | Overall hit rate, latency, cost savings |
| `domain_breakdown.csv` | Per-domain hit rates and false positive rates |
| `threshold_sweep.csv` | Hit rate vs threshold (0.70–0.95) |
| `time_series.csv` | Hit rate evolution across query batches |
| `false_positives.csv` | Top false positive examples by similarity |
| `context_summary.json` | Context benchmark: stateless vs context-aware accuracy (`--context`) |
| `context_accuracy.csv` | Per-domain context accuracy breakdown (`--context`) |

---

## Stateless benchmark results

Built-in TF-IDF engine, threshold=0.85, 5,000 queries:

| Metric | Value |
|---|---|
| Hit rate | 85.9% |
| Hit latency p50 / p95 | 0.74ms / 3.12ms |
| Miss latency p50 / p95 | 9.55ms / 19.39ms |
| Cost saved (per 10k queries) | ~$21.47 |
| False positive rate | 31.1% (see note) |

**Domain breakdown:**

| Domain | Hit rate | False positive rate | Saved / 10k |
|---|---|---|---|
| customer_support | 88.2% | 10.0% | $4.41 |
| developer_qa | 87.8% | 16.7% | $4.39 |
| product_faq | 85.4% | 26.4% | $4.27 |
| medical_information | 82.4% | 44.7% | $4.12 |
| general_knowledge | 85.6% | 59.2% | $4.28 |

> **Note on false positives:** The built-in TF-IDF engine is sensitive to
> shared prefix tokens in synthetic queries (e.g. "Could you tell me X" matches
> "Could you tell me Y"). With real sentence-transformer embeddings (`--use-sulci`),
> false positive rates drop to 1–5% because MiniLM encodes semantic meaning
> rather than token overlap.

---

## Context-aware benchmark (v0.2.0)

Measures how accurately the cache resolves **ambiguous follow-up queries**
with and without conversation context.

### The problem stateless caches can't solve

Identical follow-ups mean completely different things in different sessions:

```
Session A:  "My Docker container keeps crashing" → "How do I fix it?"
Session B:  "My account has been locked"         → "How do I fix it?"
```

A stateless cache returns whichever entry happens to score highest — often the
wrong domain. A context-aware cache blends the primer query into the lookup
vector, pushing similarity toward the correct domain's cached answer.

### How the blending works

```
lookup_vec = α × embed(query) + (1−α) × Σ wᵢ × embed(turnᵢ)
```

- `α = query_weight` (default 0.70) — current query always dominates
- `wᵢ = decay^i` (default 0.50) — most recent turn w=1.0, each older turn halved
- Only user turns are blended — assistant responses introduce structural noise

### Results

Built-in TF-IDF engine, `context_window=4`, `context_threshold=0.58`, 800 follow-up queries:

| Metric | Stateless | Context-aware | Δ |
|---|---|---|---|
| Hit rate | 64.0% | 81.6% | **+17.6pp** |
| Resolution accuracy | 56.8% | 77.6% | **+20.8pp** |
| Avg hit latency | 1.02ms | 2.35ms | +1.33ms |

**Domain breakdown:**

| Domain | Stateless | Context-aware | Δ |
|---|---|---|---|
| customer_support | 32% | 88% | **+56pp** |
| developer_qa | 80% | 96% | **+16pp** |
| medical_information | 40% | 60% | **+20pp** |
| general_knowledge | 60% | 68% | **+8pp** |
| product_faq | 72% | 76% | **+4pp** |

> **Note on thresholds:** The context benchmark uses a lower threshold (0.58 vs
> 0.85) because blending dilutes the raw cosine score slightly — the blended
> vector is no longer identical to any stored entry, so peak similarity is lower.
> With real sentence-transformer embeddings (`--use-sulci`), you can use a higher
> threshold (0.78–0.82 is a good starting point).

### Running the context benchmark

```bash
# Default (context_window=4, threshold=0.58)
python benchmark/run.py --no-sweep --context

# Tune the window size
python benchmark/run.py --no-sweep --context --context-window 6

# With real embeddings
pip install "sulci[sqlite]"
python benchmark/run.py --no-sweep --context --use-sulci --context-threshold 0.80
```

---

## Threshold sweep

The sweep tests hit rate vs false positive rate across 8 thresholds (0.70–0.95).
Skip with `--no-sweep` for faster runs.

```bash
python benchmark/run.py  # runs sweep by default
```

Typical output:

```
── Threshold sweep ───────────────────────────────────────
  t=0.70  hit=95.2%  fp=51.3%  saved=95.2%
  t=0.75  hit=92.7%  fp=42.1%  saved=92.7%
  t=0.80  hit=89.4%  fp=35.8%  saved=89.4%
  t=0.85  hit=85.9%  fp=31.1%  saved=85.9%
  t=0.88  hit=80.2%  fp=22.4%  saved=80.2%
  t=0.90  hit=73.1%  fp=14.7%  saved=73.1%
  t=0.92  hit=62.8%  fp= 8.3%  saved=62.8%
  t=0.95  hit=44.1%  fp= 3.2%  saved=44.1%
```

For production use, **0.88–0.92** gives the best accuracy/hit-rate tradeoff
with real sentence-transformer embeddings.

---

## Corpus

**Stateless benchmark:** 5 domains × 10 topic groups × ~200 queries each (~10,000 total)

| Domain | Topics |
|---|---|
| customer_support | cancel, password reset, refund, billing, order tracking, account locked, upgrade plan, invoice… |
| developer_qa | async/await, git merge vs rebase, Docker, React hooks, SQL vs NoSQL, REST, JWT, Kubernetes, Big O… |
| product_faq | pricing, free trial, data security, integrations, team collaboration, API access, GDPR… |
| medical_information | blood pressure, diabetes, common cold, COVID vaccine, anxiety, vitamin D, migraine, sleep… |
| general_knowledge | AI, climate change, blockchain, internet, quantum computing, renewable energy, DNA, Bitcoin… |

**Context benchmark:** 5 domains × 5 sessions × 8 follow-up queries = 800 conversation pairs

Each session stores:
- A verbose primer query (e.g. `"My Docker container keeps crashing on startup with exit code 1"`)
- A short keyword-dense entry (e.g. `"docker container error code deploy"`) optimised for TF-IDF cosine matching

Follow-ups are drawn from `SESSION_FOLLOWUPS` — per-session topic-aligned queries
that share vocabulary with the session's domain (e.g. `"How do I fix this docker container error?"`).

---

## Reproduce

```bash
git clone https://github.com/id4git/sulci
cd sulci

# Full run (takes ~15 minutes with sweep)
python benchmark/run.py --context

# Fast CI run (~30 seconds)
python benchmark/run.py --no-sweep --queries 1000 --context

# Real embeddings (requires sulci installed)
pip install "sulci[sqlite]"
python benchmark/run.py --use-sulci --context
```

Results are written to `benchmark/results/` (gitignored).
