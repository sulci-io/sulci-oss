# Sulci Benchmark

10,000-query semantic cache benchmark across 5 domains.
Runs entirely without API keys, cloud accounts, or GPU.

## Quick start

```bash
# No dependencies — uses built-in TF-IDF engine
python benchmark/run.py

# With real sulci embeddings (sentence-transformers MiniLM)
pip install "sulci[sqlite]"
python benchmark/run.py --use-sulci
```

## Options

```
--use-sulci     Use sulci.Cache with SQLite + MiniLM embeddings
--threshold N   Similarity threshold (default: 0.85)
--queries N     Number of test queries (default: 5000, warmup = same)
--no-sweep      Skip threshold sweep (saves ~7 minutes)
--out DIR       Output directory (default: benchmark/results)
```

## Results

Output files written to `benchmark/results/`:

| File | Contents |
|---|---|
| `summary.json` | Overall hit rate, latency, cost savings |
| `domain_breakdown.csv` | Per-domain hit rates and false positive rates |
| `threshold_sweep.csv` | Hit rate vs threshold (0.70–0.95) |
| `time_series.csv` | Hit rate evolution over time |
| `false_positives.csv` | Top false positive examples |

## Pre-computed results (built-in engine, threshold=0.85)

| Metric | Value |
|---|---|
| Hit rate | 84.9% |
| False positive rate | 31.3% (see note below) |
| Hit latency p50 | 0.55ms |
| Hit latency p95 | 2.25ms |
| Miss latency p50 | 1.44ms |
| Cost saved (per 10k queries) | ~$21 |

**Note on false positives:** The built-in TF-IDF engine is vulnerable to
prefix pollution in synthetic queries (e.g. "Could you tell me X" matches
"Could you tell me Y"). With real sentence-transformer embeddings (`--use-sulci`),
false positive rates drop to 1–5% because MiniLM encodes semantic meaning
rather than token overlap.

## Domains

5 domains × 10 topic groups × ~200 queries each:

- **customer_support** — cancel, password, refund, billing, tracking, etc.
- **developer_qa** — async/await, git, Docker, React, SQL, REST, JWT, etc.
- **product_faq** — pricing, trial, security, integrations, API, GDPR, etc.
- **medical_information** — blood pressure, diabetes, vaccines, anxiety, etc.
- **general_knowledge** — AI, climate, blockchain, internet, quantum, DNA, etc.
