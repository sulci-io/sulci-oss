# Sulci

**The AI native context-aware semantic caching for LLM apps — stop paying for the same answer twice**

[![Tests](https://github.com/sulci-io/sulci-oss/actions/workflows/tests.yml/badge.svg)](https://github.com/sulci-io/sulci-oss/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sulci)](https://pypi.org/project/sulci/)
[![Python](https://img.shields.io/pypi/pyversions/sulci)](https://pypi.org/project/sulci/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Sulci is a drop-in Python library that caches LLM responses by **semantic meaning**, not exact string match. When a user asks _"How do I deploy to AWS?"_ and someone else later asks _"What's the process for deploying on AWS?"_, Sulci returns the cached answer instead of calling the LLM again — saving cost and latency.

---

## Why Sulci

| Without Sulci                | With Sulci                                               |
| ---------------------------- | -------------------------------------------------------- |
| Every query hits the LLM API | Semantically similar queries return instantly from cache |
| $0.005 per call, every time  | Cache hits cost ~$0.0001 (embedding only)                |
| 1–3 second response time     | Cache hits return in <10ms                               |
| No memory across sessions    | Context-aware: understands conversation history          |

**Benchmark results (v0.2.5, 5,000 queries):**

- Overall hit rate: **85.9%**
- Hit latency p50: **0.74ms** (vs ~1,840ms for a live LLM call)
- Cost saved per 10k queries: **$21.47**
- Context-aware mode: **+20.8pp resolution accuracy** over stateless

---

## Install

```bash
pip install "sulci[sqlite]"    # SQLite — zero infra, local dev
pip install "sulci[chroma]"    # ChromaDB
pip install "sulci[faiss]"     # FAISS
pip install "sulci[qdrant]"    # Qdrant
pip install "sulci[redis]"     # Redis + RedisVL
pip install "sulci[milvus]"    # Milvus Lite
pip install "sulci[cloud]"     # Sulci Cloud managed backend (Week 2+)
```

> **zsh users:** always wrap extras in quotes — `".[sqlite]"` not `.[sqlite]`.

---

## Sulci Cloud — zero infrastructure option

Get a free API key at **[sulci.io/signup](https://sulci.io/signup)** and switch
to the managed backend with a single parameter change. Everything else stays identical.

```python
# Before — self-hosted (works today)
cache = Cache(backend="sqlite", threshold=0.85)

# After — managed cloud (zero other code changes)
cache = Cache(backend="sulci", api_key="sk-sulci-...", threshold=0.85)

# Or via environment variable — zero code changes at all
# export SULCI_API_KEY=sk-sulci-...
cache = Cache(backend="sulci", threshold=0.85)
```

**Free tier:** 50,000 requests/month. No credit card required.

### sulci.connect()

For apps that want to set the key once at startup and enable optional telemetry:

```python
import sulci

sulci.connect(
    api_key   = "sk-sulci-...",   # or set SULCI_API_KEY env var
    telemetry = True,             # default True — set False to disable reporting
)

cache = Cache(backend="sulci")    # picks up key from connect() automatically
```

**Telemetry is strictly opt-in.** Nothing is sent unless `sulci.connect()` is called.
`_telemetry_enabled = False` until you explicitly connect. Disable per-instance with
`Cache(backend="sulci", telemetry=False)`.

**Key resolution order:**
```
1. Explicit api_key= argument to Cache()
2. SULCI_API_KEY environment variable
3. Key stored by a prior sulci.connect() call
```

---

## Quickstart

### Stateless (v0.1 style)

```python
from sulci import Cache

cache = Cache(backend="sqlite", threshold=0.85)

# store a response
cache.set("How do I deploy to AWS?", "Use the AWS CLI with 'aws deploy'...")

# exact or semantic hit — returns 3-tuple
response, similarity, context_depth = cache.get("What's the process for deploying on AWS?")

if response:
    print(f"Cache hit (sim={similarity:.2f}): {response}")
else:
    # call your LLM here
    pass
```

### Context-aware (v0.2 style)

```python
from sulci import Cache

cache = Cache(
    backend        = "sqlite",
    threshold      = 0.85,
    context_window = 4,     # remember last 4 turns
    query_weight   = 0.70,  # α — weight of current query vs context
    context_decay  = 0.50,  # halve weight per older turn
)

# turn 1
cache.set("What is Python?", "Python is a high-level programming language.", session_id="s1")

# turn 2 — context from turn 1 blended into the lookup vector
response, sim, depth = cache.get("Tell me more about it", session_id="s1")
```

### Drop-in with `cached_call`

```python
import anthropic
from sulci import Cache

cache = Cache(backend="sqlite", threshold=0.85, context_window=4)
client = anthropic.Anthropic()

def call_llm(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

result = cache.cached_call(
    query         = "How do I deploy to AWS?",
    llm_fn        = call_llm,
    session_id    = "user-123",
    cost_per_call = 0.005,
)

print(result["response"])
print(f"Source:  {result['source']}")        # "cache" or "llm"
print(f"Latency: {result['latency_ms']}ms")
print(f"Saved:   ${result['saved_cost']:.4f}")
```

---

## API Reference

### Constructor

```python
cache = Cache(
    backend         = "sqlite",   # sqlite | chroma | faiss | qdrant | redis | milvus | sulci
    threshold       = 0.85,       # cosine similarity cutoff (0–1)
    embedding_model = "minilm",   # minilm | openai
    ttl_seconds     = None,       # None = no expiry
    personalized    = False,      # partition cache per user_id
    db_path         = "./sulci",  # on-disk path for sqlite / faiss
    context_window  = 0,          # turns to remember; 0 = stateless
    query_weight    = 0.70,       # α in blending formula
    context_decay   = 0.50,       # per-turn decay weight
    session_ttl     = 3600,       # session expiry in seconds
    api_key         = None,       # required when backend="sulci"
    telemetry       = True,       # set False to disable per-instance
)
```

### Methods

| Method                                                                                 | Returns                   | Description                                                        |
| -------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------ |
| `cache.get(query, user_id=None, session_id=None)`                                      | `(str\|None, float, int)` | response, similarity, context_depth                                |
| `cache.set(query, response, user_id=None, session_id=None)`                            | `None`                    | Store entry, advance context window                                |
| `cache.cached_call(query, llm_fn, session_id=None, user_id=None, cost_per_call=0.005)` | `dict`                    | response, source, similarity, latency_ms, cache_hit, context_depth |
| `cache.get_context(session_id)`                                                        | `ContextWindow`           | Return session's context window                                    |
| `cache.clear_context(session_id)`                                                      | `None`                    | Reset session history                                              |
| `cache.context_summary(session_id=None)`                                               | `dict`                    | Snapshot of one or all sessions                                    |
| `cache.stats()`                                                                        | `dict`                    | hits, misses, hit_rate, saved_cost, total_queries, active_sessions |
| `cache.clear()`                                                                        | `None`                    | Evict all entries, reset stats and sessions                        |

> **Important:** `cache.get()` returns a **3-tuple** `(response, similarity, context_depth)` — not a 2-tuple like v0.1. Always unpack all three values.

---

## Context-Aware Blending

When `context_window > 0`, Sulci blends the current query vector with recent
conversation history before performing the similarity lookup:

```
lookup_vec = α · embed(query) + (1−α) · Σ(decay^i · turn_i)
```

- `α` = `query_weight` (default **0.70**) — how much the current query dominates
- `decay` = `context_decay` (default **0.50**) — halves weight per older turn
- Only **user query** vectors are stored in context (not LLM responses)
- Raw un-blended vectors stored in cache; blending happens at lookup time only

**Context-aware benchmark results (800 conversation pairs, context_window=4):**

| Domain              | Stateless | Context-aware | Δ           |
| ------------------- | --------- | ------------- | ----------- |
| customer_support    | 32%       | 88%           | **+56pp**   |
| developer_qa        | 80%       | 96%           | +16pp       |
| medical_information | 40%       | 60%           | +20pp       |
| **overall**         | **64.0%** | **81.6%**     | **+17.6pp** |

---

## Backends

| Backend         | ID       | Hit latency | Best for                                |
| --------------- | -------- | ----------- | --------------------------------------- |
| SQLite          | `sqlite` | <8ms        | Local dev, edge, serverless, zero infra |
| ChromaDB        | `chroma` | <10ms       | Fastest path to working, Python-native  |
| FAISS           | `faiss`  | <3ms        | GPU acceleration, massive scale         |
| Qdrant          | `qdrant` | <5ms        | Production, metadata filtering          |
| Redis + RedisVL | `redis`  | <1ms        | Existing Redis infra, lowest latency    |
| Milvus Lite     | `milvus` | <7ms        | Dev-to-prod without code changes        |
| **Sulci Cloud** | `sulci`  | <8ms        | **Zero infra — managed service (NEW)**  |

All self-hosted backends are free tier or self-hostable at zero cost.

---

## Embedding Models

| ID       | Model                  | Dims | Latency | Notes                                        |
| -------- | ---------------------- | ---- | ------- | -------------------------------------------- |
| `minilm` | all-MiniLM-L6-v2       | 384  | 14ms    | **Default** — free, local, excellent quality |
| `openai` | text-embedding-3-small | 1536 | ~100ms  | Requires `OPENAI_API_KEY`                    |

The default `minilm` model runs entirely locally via `sentence-transformers`.
No network calls are made unless you explicitly configure `embedding_model="openai"`.

---

## Project Structure

```
.
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── LOCAL_SETUP.md
├── README.md
├── benchmark
│   ├── README.md               ← benchmark methodology and results
│   └── run.py                  ← benchmark CLI
├── examples
│   ├── anthropic_example.py    ← requires ANTHROPIC_API_KEY
│   ├── basic_usage.py          ← stateless cache demo, no API key needed
│   ├── context_aware.py        ← 4-demo walkthrough, fully offline
│   └── context_aware_example.py← additional context-aware patterns
├── pyproject.toml              ← name="sulci", version="0.2.5"
├── setup.py
├── sulci
│   ├── __init__.py             ← exports Cache, ContextWindow, SessionStore, connect()
│   │                              NEW (Week 2): connect(), _emit(), _flush(), _SDK_VERSION
│   ├── backends
│   │   ├── __init__.py         ← empty — core.py loads backends via importlib
│   │   ├── chroma.py
│   │   ├── cloud.py            ← SulciCloudBackend — NEW (Week 3)
│   │   ├── faiss.py
│   │   ├── milvus.py
│   │   ├── qdrant.py
│   │   ├── redis.py
│   │   └── sqlite.py
│   ├── context.py              ← ContextWindow + SessionStore
│   ├── core.py                 ← Cache engine (context-aware)
│   │                              Week 2: telemetry= param, _emit() in get()
│   │                              Week 3: api_key= param, _load_backend handles sulci
│   └── embeddings
│       ├── __init__.py
│       ├── minilm.py           ← default: all-MiniLM-L6-v2 (free, local)
│       └── openai.py           ← requires OPENAI_API_KEY
└── tests
    ├── test_backends.py        —  9 tests: per-backend contract + persistence
    ├── test_cloud_backend.py   — 25 tests: SulciCloudBackend + Cache wiring — NEW (Week 3)
    ├── test_connect.py         — 32 tests: sulci.connect(), _emit(), _flush(), Cache telemetry flag
    ├── test_context.py         — 27 tests: ContextWindow, SessionStore, integration
    └── test_core.py            — 26 tests: cache.get/set, TTL, stats, personalization

7 directories, 31 files
```

---

## Running Tests

```bash
# full suite — 121 tests total (7 skipped if optional backend deps not installed)
python -m pytest tests/ -v

# by file
python -m pytest tests/test_core.py -v         # 26 tests
python -m pytest tests/test_context.py -v      # 27 tests
python -m pytest tests/test_backends.py -v     #  9 tests (skipped if dep missing)
python -m pytest tests/test_connect.py -v      # 32 tests — sulci.connect() + telemetry
python -m pytest tests/test_cloud_backend.py -v # 25 tests — NEW Week 3, SulciCloudBackend

# single backend only
python -m pytest tests/test_backends.py -v -k sqlite
python -m pytest tests/test_backends.py -v -k chroma

# with coverage
python -m pytest tests/ -v --cov=sulci --cov-report=term-missing
```

> **Week 2:** `test_connect.py` (32 tests) — `sulci.connect()`, `_emit()`, `_flush()`, `Cache(telemetry=)`. Requires `httpx`.

> **Week 3:** `test_cloud_backend.py` (25 tests) — `SulciCloudBackend` construction, `search()`, `upsert()`, `delete_user()`, `clear()`, and `Cache(backend='sulci')` wiring. Requires `httpx`.

Backend tests are **skipped — not failed** when their dependency isn't installed.
Install the backend extra to run its tests: `pip install -e ".[chroma]"`.

See [`LOCAL_SETUP.md`](./LOCAL_SETUP.md) for the full local development guide including
venv setup, backend installation, smoke testing, and troubleshooting.

---

## Benchmark

```bash
# fast run (~30 seconds)
python benchmark/run.py --no-sweep --queries 1000

# with context-aware pass
python benchmark/run.py --no-sweep --queries 1000 --context

# full benchmark
python benchmark/run.py --context
```

See [`benchmark/README.md`](./benchmark/README.md) for full methodology and results.

---

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for branching model, PR process, and coding standards.

---

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE).

Copyright 2026 Kathiravan Sengodan.

This software may be covered by one or more pending patent applications.
The Apache 2.0 license grants you a royalty-free patent license for your
use of this code. See Section 3 of the Apache License and the NOTICE file.

---

## Links

- **Website:** [sulci.io](https://sulci.io)
- **Sign up (free key):** [sulci.io/signup](https://sulci.io/signup)
- **API:** [api.sulci.io](https://api.sulci.io)
- **PyPI:** [sulci](https://pypi.org/project/sulci/)
- **GitHub:** [sulci-io/sulci-oss](https://github.com/sulci-io/sulci-oss)
- **Issues:** [github.com/sulci-io/sulci-oss/issues](https://github.com/sulci-io/sulci-oss/issues)
