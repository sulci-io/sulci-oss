# Sulci — Local Setup Guide

Everything you need to clone the repo, install dependencies, run tests, and verify a working local environment from scratch.

---

## Requirements

- Python **3.9, 3.11, or 3.12** (all three are tested in CI)
- `git`
- A terminal with `pip` available

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/sulci-io/sulci-oss.git
cd sulci-oss
```

> **Active development branch:** All Phase 1 (v0.3.0) work lives on `feature/saas-onramp`.
> Check it out before making any changes:
>
> ```bash
> git checkout feature/saas-onramp
> git branch
> # Should show: * feature/saas-onramp
> ```
>
> `main` stays at v0.2.5 — clean and untouched until Week 4 when v0.3.0 is ready to release.

---

## Step 2 — Create and Activate a Virtual Environment

Always use a virtual environment. Never install Sulci dependencies into your system Python.

```bash
# create
python -m venv .venv

# activate — macOS / Linux
source .venv/bin/activate

# activate — Windows
.venv\Scripts\activate

# confirm you're inside the venv
which python        # should show .venv/bin/python
python --version    # should be 3.9, 3.11, or 3.12
```

---

## Step 3 — Install the Library

Install in editable mode (`-e`) so any changes you make to `sulci-oss/` source code are reflected immediately without reinstalling.

```bash
# base install — editable
pip install -e .

# with the SQLite backend (zero infra, fully offline — recommended for local dev)
pip install -e ".[sqlite]"

# with ChromaDB
pip install -e ".[chroma]"

# with FAISS
pip install -e ".[faiss]"

# multiple backends at once
pip install -e ".[sqlite,chroma,faiss]"
```

> **zsh users:** always wrap extras in quotes — `".[sqlite]"` not `.[sqlite]`.
> Without quotes, zsh treats the brackets as a glob pattern and throws `no matches found`.

Then install pytest and httpx for running the test suite:

```bash
pip install pytest pytest-cov httpx
```

> **Why httpx?** The `test_connect.py` suite (Week 2) mocks `httpx.post` to test
> the telemetry flush — httpx must be installed even though it is only used in tests.

---

## Step 4 — Verify the Install

```bash
python -c "from sulci import Cache, ContextWindow, SessionStore, connect; from sulci.backends.cloud import SulciCloudBackend; print('Import OK')"
```

Expected output:

```
Import OK
```

> **Week 2 addition:** `connect` is now exported from `sulci/__init__.py` as part of
> the Phase 1 SaaS onramp. If you see `ImportError: cannot import name 'connect'`
> you are on the wrong branch — run `git checkout feature/saas-onramp`.

If you see a `ModuleNotFoundError` on a backend (e.g. `chromadb`, `faiss`), that backend's
extra is not installed. Install it with `pip install -e ".[backend_name]"`.

---

## Step 5 — Run the Tests

Always use `python -m pytest` rather than bare `pytest` to avoid PATH issues.

```bash
python -m pytest tests/ -v
```

All **121 tests** should pass across five test files (7 skipped if optional backend deps not installed):

```
tests/test_core.py           — 26 tests  (cache.get/set, thresholds, TTL, stats, personalization)
tests/test_context.py        — 27 tests  (ContextWindow, SessionStore, integration)
tests/test_backends.py       —  9 tests  (per-backend contract + persistence; skipped if dep missing)
tests/test_connect.py        — 32 tests  (sulci.connect(), _emit(), _flush(), Cache telemetry flag)
                                          ↑ added Week 2 — requires httpx
tests/test_cloud_backend.py  — 25 tests  (SulciCloudBackend, Cache(backend='sulci') wiring)
                                          ↑ added Week 3 — requires httpx
```

> **Note:** Original baseline was 71 tests (3 files). Week 2 added `test_connect.py` (+32). Week 3 adds `test_cloud_backend.py` (+25). Total: 121 passed, 7 skipped.

### Targeted test runs

```bash
# core cache logic only
python -m pytest tests/test_core.py -v

# context and session tests only
python -m pytest tests/test_context.py -v

# backend tests only
python -m pytest tests/test_backends.py -v

# telemetry + sulci.connect() tests only (Week 2)
python -m pytest tests/test_connect.py -v

# SulciCloudBackend + Cache wiring tests only (Week 3)
python -m pytest tests/test_cloud_backend.py -v

# single backend by keyword
python -m pytest tests/test_backends.py -v -k sqlite
python -m pytest tests/test_backends.py -v -k chroma

# one specific test by name
python -m pytest tests/test_core.py::TestBasicOperations::test_semantic_hit -v

# stop at first failure
python -m pytest tests/ -v -x

# with line-level coverage report
python -m pytest tests/ -v --cov=sulci --cov-report=term-missing
```

---

## Step 6 — Run the Examples

### No API key required

```bash
# stateless cache demo
python examples/basic_usage.py

# context-aware demo — 4 walkthroughs, fully offline
python examples/context_aware.py

# additional context-aware patterns
python examples/context_aware_example.py
```

### Requires `ANTHROPIC_API_KEY`

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/anthropic_example.py
```

---

## Step 7 — Run the Benchmark

```bash
# fast run — stateless, 1,000 queries (~30 seconds)
python benchmark/run.py --no-sweep --queries 1000

# fast run — with context-aware mode
python benchmark/run.py --no-sweep --queries 1000 --context

# full benchmark — stateless, 5,000 queries
python benchmark/run.py

# full benchmark — with context-aware mode
python benchmark/run.py --context
```

Results are written to `benchmark/results/`. The `.gitignore` in that directory
excludes `*.json` and `*.csv` so result files are never committed.

### All benchmark flags

| Flag                    | Default             | Description                                       |
| ----------------------- | ------------------- | ------------------------------------------------- |
| `--context`             | off                 | Enable context-aware benchmark pass               |
| `--no-sweep`            | off                 | Skip threshold sweep (much faster)                |
| `--queries N`           | 5000                | Number of test queries                            |
| `--threshold F`         | 0.85                | Similarity threshold for stateless pass           |
| `--context-threshold F` | 0.58                | Similarity threshold for context pass             |
| `--context-window N`    | 4                   | Turns per session window                          |
| `--use-sulci`           | off                 | Use real MiniLM embeddings (vs TF-IDF simulation) |
| `--out DIR`             | `benchmark/results` | Output directory for result files                 |

---

## Step 8 — Smoke Test (Quick End-to-End Sanity Check)

Create a file `smoke_test.py` at the repo root and run it to confirm the full
stack is working — import, set, get, semantic hit, context mode, and stats:

```python
# smoke_test.py
from sulci import Cache

# --- stateless mode, SQLite backend, no infrastructure needed ---
cache = Cache(backend="sqlite", threshold=0.85)

# store an entry
cache.set("How do I deploy to AWS?", "Use the AWS CLI with 'aws deploy'...")

# exact match hit
response, sim, ctx_depth = cache.get("How do I deploy to AWS?")
assert response is not None, "FAIL: exact hit returned None"
print(f"Exact hit:    sim={sim:.3f}  ctx={ctx_depth}  ✅")

# semantic match hit
response, sim, ctx_depth = cache.get("What is the process for deploying on AWS?")
if response:
    print(f"Semantic hit: sim={sim:.3f}  ctx={ctx_depth}  ✅")
else:
    print(f"Semantic miss (sim={sim:.3f}) — try lowering threshold")

# stats
s = cache.stats()
print(f"Stats:        hits={s['hits']}  misses={s['misses']}  hit_rate={s['hit_rate']:.1%}")

# --- context-aware mode ---
cache_ctx = Cache(backend="sqlite", threshold=0.85, context_window=4)
cache_ctx.set(
    "What is Python?",
    "Python is a high-level programming language.",
    session_id="s1"
)
response, sim, ctx_depth = cache_ctx.get("Tell me about Python", session_id="s1")
print(f"Context mode: sim={sim:.3f}  ctx_depth={ctx_depth}  ✅")

print("\nAll smoke tests passed.")
```

```bash
python smoke_test.py
```

All four lines should print `✅` and the final line `All smoke tests passed.`

---

## Step 9 — Test sulci.connect() Locally (Week 2)

`sulci.connect()` is the opt-in telemetry gate added in Week 2. The default state
is **silent** — nothing is sent until you explicitly call `connect()`.

### Verify default state

```python
import sulci

# Before connect() — everything is off
print(sulci._telemetry_enabled)    # False
print(sulci._api_key)              # None
print(sulci._event_buffer)         # []
```

### Test connect() with a real key

```python
import sulci

# Option 1 — explicit key
sulci.connect(api_key="sk-sulci-...")
print(sulci._telemetry_enabled)    # True
print(sulci._api_key)              # sk-sulci-...

# Option 2 — from environment variable
# export SULCI_API_KEY=sk-sulci-...
sulci.connect()
print(sulci._api_key)              # sk-sulci-...

# Option 3 — connect but disable telemetry reporting
sulci.connect(api_key="sk-sulci-...", telemetry=False)
print(sulci._telemetry_enabled)    # False (key stored, no reporting)
```

### Disable telemetry per Cache instance

```python
# Even after connect(), an individual Cache can opt out
cache = sulci.Cache(backend="sqlite", telemetry=False)
print(cache._telemetry)            # False
```

### Key resolution order

When `backend="sulci"` is used, the API key is resolved in this order:

```
1. Explicit api_key= argument to Cache()
2. SULCI_API_KEY environment variable
3. Key stored by a prior sulci.connect() call
```

### Run only the connect tests

```bash
python -m pytest tests/test_connect.py -v

# Run a specific class
python -m pytest tests/test_connect.py::TestDefaultState -v
python -m pytest tests/test_connect.py::TestConnect -v
python -m pytest tests/test_connect.py::TestEmit -v
python -m pytest tests/test_connect.py::TestFlush -v
python -m pytest tests/test_connect.py::TestCacheIntegration -v
python -m pytest tests/test_connect.py::TestThreadSafety -v
```

---

## Step 10 — Test SulciCloudBackend Locally (Week 3)

`SulciCloudBackend` is the cloud backend driver added in Week 3. It routes
cache operations to `api.sulci.io` via httpx.

### Verify the import and basic construction

```python
from sulci.backends.cloud import SulciCloudBackend

# Confirm ValueError on missing key
try:
    b = SulciCloudBackend(api_key=None)
except ValueError as e:
    print(f"ValueError ok: {e}")

# Confirm repr
b = SulciCloudBackend(api_key="sk-sulci-testkey1234567")
print(b)
# SulciCloudBackend(url='https://api.sulci.io', key_prefix='sk-sulci-testke', timeout=5.0)
```

### Verify Cache constructor wiring

```python
from unittest.mock import patch
from sulci import Cache

# Explicit key
with patch("sulci.backends.cloud.SulciCloudBackend") as MockBackend:
    MockBackend.return_value = MockBackend
    cache = Cache(backend="sulci", api_key="sk-sulci-testkey1234567")
    print(f"Cache with sulci backend: {cache}")

# Via env var
import os
os.environ["SULCI_API_KEY"] = "sk-sulci-testkey1234567"
with patch("sulci.backends.cloud.SulciCloudBackend") as MockBackend:
    MockBackend.return_value = MockBackend
    cache = Cache(backend="sulci")
    print("Env var resolution ok")
del os.environ["SULCI_API_KEY"]
```

### Key resolution order reminder

```
1. Explicit api_key= argument to Cache()
2. SULCI_API_KEY environment variable
3. Key stored by a prior sulci.connect() call
```

### Run only the cloud backend tests

```bash
python -m pytest tests/test_cloud_backend.py -v

# Run a specific class
python -m pytest tests/test_cloud_backend.py::TestConstruction -v
python -m pytest tests/test_cloud_backend.py::TestSearch -v
python -m pytest tests/test_cloud_backend.py::TestUpsert -v
python -m pytest tests/test_cloud_backend.py::TestDeleteAndClear -v
python -m pytest tests/test_cloud_backend.py::TestCacheWiring -v
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| ----------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------- |
| `pytest: command not found` | pytest not on `PATH` | Use `python -m pytest` |
| `zsh: no matches found: .[sqlite]` | zsh glob expansion | Use quotes: `".[sqlite]"` |
| `ModuleNotFoundError: sulci` | Not installed | Run `pip install -e .` first |
| `ModuleNotFoundError: chromadb` | Backend extra missing | `pip install -e ".[chroma]"` |
| `ModuleNotFoundError: httpx` | httpx not installed | `pip install httpx` — needed for test_connect.py |
| `ImportError: cannot import name 'connect'` | Wrong branch | `git checkout feature/saas-onramp` |
| `ValueError: not enough values to unpack` | v0.1 unpacking style | `cache.get()` returns a **3-tuple** in v0.2 — always unpack as `response, sim, ctx_depth = cache.get(...)` |
| MiniLM takes 2–3s on first call | Model cold load | Normal — subsequent embeds run at ~14ms. Warm the model at app startup, not per-request. |
| `git push` returns 403 | Token auth expired | `git remote set-url origin https://YOUR_USER:TOKEN@github.com/sulci-io/sulci-oss.git` |
| `_telemetry_enabled` is True unexpectedly | connect() called elsewhere | Check if `sulci.connect()` is being called in app code or test fixtures — telemetry is opt-in only |

---

## API Key Notes

The core library and all tests run **without any API key**. The only things that
require a key:

| File | Key needed |
| ------------------------------- | ------------------- |
| `examples/anthropic_example.py` | `ANTHROPIC_API_KEY` |
| `sulci/embeddings/openai.py` | `OPENAI_API_KEY` |
| `sulci.connect()` / `Cache(backend="sulci")` | `SULCI_API_KEY` (Sulci Cloud — optional) |
| All other code | None |

The default embedding model (`minilm`) runs fully locally via `sentence-transformers`.
No network calls are made unless you explicitly configure `embedding_model="openai"`
or use `backend="sulci"` with `sulci.connect()`.

> **`SULCI_API_KEY`** is the environment variable for the Sulci Cloud managed backend.
> Get a free key at [sulci.io/signup](https://sulci.io/signup). Setting this variable
> is optional — the library works fully offline without it.

---

## What a Clean Run Looks Like

```
$ python -m pytest tests/ -v

tests/test_backends.py::TestSQLiteBackend::test_contract PASSED
tests/test_backends.py::TestSQLiteBackend::test_persistence PASSED
tests/test_backends.py::TestChromaBackend::test_contract SKIPPED (chromadb not installed)
tests/test_backends.py::TestFAISSBackend::test_contract SKIPPED (faiss-cpu not installed)
tests/test_backends.py::TestQdrantBackend::test_contract SKIPPED (qdrant-client not installed)
tests/test_backends.py::TestRedisBackend::test_contract_local SKIPPED (redis not installed)
tests/test_backends.py::TestMilvusBackend::test_contract SKIPPED (pymilvus not installed)
tests/test_connect.py::TestDefaultState::test_telemetry_disabled_by_default PASSED
tests/test_connect.py::TestDefaultState::test_api_key_none_by_default PASSED
tests/test_connect.py::TestDefaultState::test_event_buffer_empty_by_default PASSED
tests/test_connect.py::TestDefaultState::test_emit_is_noop_by_default PASSED
tests/test_connect.py::TestConnect::test_connect_enables_telemetry_with_key PASSED
tests/test_connect.py::TestConnect::test_connect_without_key_does_not_enable_telemetry PASSED
tests/test_connect.py::TestConnect::test_connect_telemetry_false_sets_key_but_not_telemetry PASSED
tests/test_connect.py::TestConnect::test_connect_reads_key_from_env PASSED
tests/test_connect.py::TestConnect::test_connect_emits_startup_event PASSED
tests/test_connect.py::TestConnect::test_connect_twice_starts_flush_thread_only_once PASSED
...
tests/test_connect.py::TestEmit::test_emit_buffers_event_when_enabled PASSED
tests/test_connect.py::TestEmit::test_emit_noop_when_disabled PASSED
tests/test_connect.py::TestEmit::test_emit_never_raises_on_exception PASSED
...
tests/test_connect.py::TestFlush::test_flush_sends_aggregated_payload PASSED
tests/test_connect.py::TestFlush::test_flush_sends_correct_auth_header PASSED
tests/test_connect.py::TestFlush::test_flush_swallows_http_exception PASSED
tests/test_connect.py::TestFlush::test_flush_uses_3s_timeout PASSED
...
tests/test_connect.py::TestCacheIntegration::test_cache_constructor_accepts_telemetry_false PASSED
tests/test_connect.py::TestCacheIntegration::test_cache_get_does_not_emit_when_telemetry_false PASSED
tests/test_connect.py::TestCacheIntegration::test_cache_get_emits_when_telemetry_enabled PASSED
tests/test_connect.py::TestThreadSafety::test_concurrent_emits_do_not_lose_events PASSED
tests/test_context.py::TestContextWindow::test_empty_window_returns_query_vec PASSED
tests/test_context.py::TestContextWindow::test_blend_pulls_toward_history PASSED
tests/test_context.py::TestContextWindow::test_blend_is_normalised PASSED
tests/test_context.py::TestContextWindow::test_decay_weights_recent_more PASSED
...
tests/test_context.py::TestSessionStore::test_same_session_id_returns_same_window PASSED
tests/test_context.py::TestSessionStore::test_different_sessions_are_isolated PASSED
tests/test_context.py::TestSessionStore::test_ttl_eviction PASSED
...
tests/test_context.py::TestCacheContextIntegration::test_context_depth_increases_on_follow_up PASSED
tests/test_context.py::TestCacheContextIntegration::test_sessions_are_isolated PASSED
tests/test_context.py::TestCacheContextIntegration::test_clear_context_resets_depth PASSED
...
tests/test_core.py::TestBasicOperations::test_import PASSED
tests/test_core.py::TestBasicOperations::test_version PASSED
tests/test_core.py::TestBasicOperations::test_miss_on_empty_cache PASSED
tests/test_core.py::TestBasicOperations::test_set_then_exact_get PASSED
tests/test_core.py::TestBasicOperations::test_semantic_hit PASSED
tests/test_core.py::TestBasicOperations::test_ttl_expiry PASSED
...
tests/test_core.py::TestCachedCall::test_miss_calls_llm PASSED
tests/test_core.py::TestCachedCall::test_hit_skips_llm PASSED
tests/test_core.py::TestCachedCall::test_result_has_all_fields PASSED
...
tests/test_core.py::TestStats::test_initial_stats PASSED
tests/test_core.py::TestStats::test_hit_increments_hits PASSED
tests/test_core.py::TestStats::test_saved_cost_accumulates PASSED
...
tests/test_core.py::TestThreshold::test_strict_threshold_rejects_paraphrase PASSED
tests/test_core.py::TestPersonalization::test_user_scoped_hit PASSED
tests/test_core.py::TestPersonalization::test_user_scoped_miss_for_other_user PASSED

========== 121 passed, 7 skipped in ~280s ==========
```

> **Backend tests are skipped — not failed — when the dependency isn't installed.** This is expected.
> Install a backend extra (e.g. `pip install -e ".[chroma]"`) to run its tests.

---

## Project Structure (Reference)

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
    ├── test_backends.py        —  9 tests: per-backend contract + persistence (skipped if dep missing)
    ├── test_cloud_backend.py   — 25 tests: SulciCloudBackend + Cache wiring — NEW (Week 3)
    ├── test_connect.py         — 32 tests: sulci.connect(), _emit(), _flush(), Cache telemetry flag
    ├── test_context.py         — 27 tests: ContextWindow, SessionStore, integration
    └── test_core.py            — 26 tests: cache.get/set, TTL, stats, personalization

7 directories, 31 files
```

---

## Related Docs

- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — branching model, PR process, coding standards
- [`CHANGELOG.md`](./CHANGELOG.md) — version history
- [`benchmark/README.md`](./benchmark/README.md) — benchmark methodology and results
- [PyPI: sulci](https://pypi.org/project/sulci/)
- [GitHub: sulci-io/sulci-oss](https://github.com/sulci-io/sulci-oss)

---

## Branch Reference

| Branch | Purpose | Status |
|---|---|---|
| `main` | Stable release — v0.2.5 | Do not push directly |
| `feature/saas-onramp` | Phase 1 / v0.3.0 development — Weeks 1–3 complete | **Current — work here** |
| `feature/context-aware` | v0.2.0 context-aware library | Merged |
| `feature/benchmark-context-aware` | v0.2.5 benchmark suite | Merged |

All Phase 1 changes (Steps 4–13, targeting v0.3.0) live on `feature/saas-onramp`.
Merge to `main` and tag `v0.3.0` happens at the end of Week 4.

### What's on feature/saas-onramp right now

```
sulci/__init__.py             ← connect(), _emit(), _flush()             Week 2 ✅
sulci/core.py                 ← telemetry= + api_key= + _load_backend    Week 2+3 ✅
sulci/backends/cloud.py       ← SulciCloudBackend via httpx              Week 3 ✅
tests/test_connect.py         ← 32 tests — telemetry opt-in              Week 2 ✅
tests/test_cloud_backend.py   ← 25 tests — SulciCloudBackend + wiring    Week 3 ✅
README.md                     ← updated for Weeks 2 + 3                  Week 3 ✅
LOCAL_SETUP.md                ← updated for Weeks 2 + 3                  Week 3 ✅
```

### Coming in Week 4

```
pyproject.toml                ← add httpx>=0.27.0, bump version to 0.3.0
CHANGELOG.md                  ← add v0.3.0 entry
sulci.io                      ← signup form posting to api.sulci.io/v1/activate
→ then tag v0.3.0 → push → publish.yml triggers → PyPI release
```
