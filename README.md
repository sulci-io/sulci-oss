# Sulci Cache

**The AI native context-aware semantic caching for LLM apps — stop paying for the same answer twice**

[![Patent Pending](https://img.shields.io/badge/Patent-Pending-blue.svg)](https://sulci.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/sulci-io/sulci-oss/actions/workflows/tests.yml/badge.svg)](https://github.com/sulci-io/sulci-oss/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sulci)](https://pypi.org/project/sulci/)
[![Python](https://img.shields.io/pypi/pyversions/sulci)](https://pypi.org/project/sulci/)
[![Downloads](https://pepy.tech/badge/sulci/month)](https://pepy.tech/project/sulci)
[![Downloads](https://pepy.tech/badge/sulci)](https://pepy.tech/project/sulci)

Sulci Cache is a drop-in Python library that caches LLM responses by **semantic meaning**, not exact string match. When a user asks _"How do I deploy to AWS?"_ and someone else later asks _"What's the process for deploying on AWS?"_, Sulci Cache returns the cached answer instead of calling the LLM again — saving cost and latency.

---

## Why Sulci Cache

| Without Sulci Cache          | With Sulci Cache                                         |
| ---------------------------- | -------------------------------------------------------- |
| Every query hits the LLM API | Semantically similar queries return instantly from cache |
| $0.005 per call, every time  | Cache hits cost ~$0.0001 (embedding only)                |
| 1–3 second response time     | Cache hits return in <10ms                               |
| No memory across sessions    | Context-aware: understands conversation history          |

**Benchmark results (5,000 queries):**

- Overall hit rate: **85.9%**
- Hit latency p50: **0.74ms** (vs ~1,840ms for a live LLM call)
- Cost saved per 10k queries: **$21.47**
- Context-aware mode: **+20.8pp resolution accuracy** over stateless

---

## Install

**Step 1 — Install Sulci Cache with a backend:**

```bash
pip install "sulci[sqlite]"    # SQLite — zero infra, local dev (start here)
pip install "sulci[chroma]"    # ChromaDB
pip install "sulci[faiss]"     # FAISS
pip install "sulci[qdrant]"    # Qdrant
pip install "sulci[redis]"     # Redis + RedisVL
pip install "sulci[milvus]"    # Milvus Lite
pip install "sulci[cloud]"     # Sulci Cloud managed backend
```

**LangChain integration:**

```bash
pip install "sulci[sqlite,langchain]"   # + LangChain integration
```

**LlamaIndex integration:**

```bash
pip install "sulci[sqlite,llamaindex]"  # + LlamaIndex native integration
```

**AsyncCache (non-blocking async wrapper):**

```bash
pip install "sulci[sqlite]"   # AsyncCache is included — no extra install needed
```

**Step 2 — Install your LLM SDK** (required for `cached_call` with a live model):

```bash
pip install anthropic           # for Anthropic / Claude
pip install openai              # for OpenAI
```

> **zsh users:** always wrap extras in quotes — `"sulci[sqlite]"` not `sulci[sqlite]`.

> **Cloud backend, since v0.6.3:** `pip install sulci` (no extras) already pulls `httpx` as a mandatory dependency, so the `[cloud]` extra is now a back-compat no-op. New installs can do `pip install sulci` and use `Cache(backend="sulci", api_key="sk-sulci-...")` directly. The `[cloud]` extra is kept for users who pinned the extras-bearing install command in their pyproject.toml or requirements.txt — installing it does no harm, just no longer required.

---

## LangChain Integration

Sulci Cache is the only LangChain cache that implements context-aware lookup
vector blending — blending prior conversation turns into the similarity lookup,
not just matching the current prompt in isolation.

```python
from langchain_core.globals import set_llm_cache
from sulci.integrations.langchain import SulciCache

# Stateless semantic — drop-in for GPTCache
set_llm_cache(SulciCache(backend="sqlite"))

# Context-aware — chatbot / agent (+56pp hit rate in customer support)
set_llm_cache(SulciCache(backend="sqlite", context_window=4, threshold=0.75))

# Managed Sulci Cloud
set_llm_cache(SulciCache(backend="sulci", api_key="sk-sulci-..."))
```

Install: `pip install "sulci[sqlite,langchain]"`

---

## LlamaIndex Integration

`SulciCacheLLM` is a native LLM-level semantic cache for LlamaIndex with
no LangChain dependency required. It wraps any LlamaIndex-compatible LLM (`OpenAI`, `Anthropic`, `Ollama`, `HuggingFaceLLM`, etc.) — `complete()` and `chat()` are cached, streaming passes through uncached, async methods use `run_in_executor`.

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from sulci.integrations.llamaindex import SulciCacheLLM

# Stateless — drop-in for any LlamaIndex LLM
Settings.llm = SulciCacheLLM(
    llm       = OpenAI(model="gpt-4o"),
    backend   = "sqlite",
    threshold = 0.85,
)

# Context-aware — RAG chatbot / agent (+56pp hit rate in customer support)
Settings.llm = SulciCacheLLM(
    llm            = OpenAI(model="gpt-4o"),
    backend        = "sqlite",
    threshold      = 0.75,
    context_window = 4,
)

# Managed Sulci Cloud
Settings.llm = SulciCacheLLM(
    llm     = OpenAI(model="gpt-4o"),
    backend = "sulci",
    api_key = "sk-sulci-...",
)
```

Install: `pip install "sulci[sqlite,llamaindex]"`

**Via LangChain (alternative — works today, no extra install):**

```python
from langchain_core.globals import set_llm_cache
from sulci.integrations.langchain import SulciCache
from llama_index.llms.langchain import LangChainLLM
from langchain_openai import ChatOpenAI

set_llm_cache(SulciCache(backend="sqlite", context_window=4))

from llama_index.core import Settings
Settings.llm = LangChainLLM(llm=ChatOpenAI(model="gpt-4o"))
```

Install: `pip install "sulci[sqlite,langchain]" llama-index-llms-langchain langchain-openai`

---


## AsyncCache — non-blocking async wrapper

`AsyncCache` wraps `sulci.Cache` with `asyncio.to_thread()` so every cache
operation yields the event loop. The correct pattern for FastAPI, LangChain
async chains, LlamaIndex async agents, and any asyncio-based application.

```python
from sulci import AsyncCache

cache = AsyncCache(backend="sqlite", context_window=4)

# FastAPI endpoint — event loop never blocked
@app.post("/chat")
async def chat(query: str, session_id: str):
    response, sim, depth = await cache.aget(query, session_id=session_id)
    if response:
        return {"response": response, "source": "cache", "sim": sim}
    response = await call_llm(query)
    await cache.aset(query, response, session_id=session_id)
    return {"response": response, "source": "llm"}

# All Cache parameters work identically
cache = AsyncCache(
    backend        = "sqlite",
    threshold      = 0.85,
    context_window = 4,
    query_weight   = 0.70,
    api_key        = "sk-sulci-...",   # for Sulci Cloud
)
```

**Async methods:** `aget()`, `aset()`, `acached_call()`, `aget_context()`,
`aclear_context()`, `acontext_summary()`, `astats()`, `aclear()`

**Sync passthrough:** All sync methods (`get`, `set`, `stats`, `clear`) also
available — `AsyncCache` works in mixed sync/async codebases without switching types.

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

### One-line setup — telemetry just works (v0.7.0+)

As of v0.7.0, passing `api_key` to `Cache()` enables the Sulci Cloud
dashboard automatically. **One line is enough** — no separate `sulci.connect()`
call required for the dashboard panels (TrendChart, AuditEventsTable,
DeploymentsTable, Active SDKs) to populate.

```python
from sulci import Cache

cache = Cache(
    backend = "sulci",                # or "sqlite"/"chroma"/etc. — see below
    api_key = "sk-sulci-...",         # or set SULCI_API_KEY env var
)

cache.get("hello")                    # populates the entire dashboard
```

This works for every tier and every backend choice:

| Persona | Backend | api_key role |
|---|---|---|
| **Pro / Business (paid managed)** | `"sulci"` | cache auth + telemetry |
| **OSS-Connect (free, self-host + dashboard)** | `"sqlite"` / `"chroma"` / `"qdrant"` / etc. | **telemetry only** — cache lives locally |
| **Pure self-hosted (no Sulci account)** | local backend | omit `api_key` — no telemetry, no cloud |

**One rule covers all three:**

> If `api_key` is present anywhere (kwarg, `SULCI_API_KEY` env, or prior
> `sulci.connect()`) AND `telemetry=True` (the default), telemetry flows
> to Sulci. Backend choice is independent.

**Telemetry remains strictly opt-in.** No api_key anywhere → no telemetry,
ever. Pass `telemetry=False` to override: `Cache(backend="sulci",
api_key="sk-sulci-...", telemetry=False)` uses the managed cache without
emitting telemetry (useful in compliance-restricted environments or
internal staging where dev traffic should not pollute production
dashboards).

### sulci.connect() — advanced flows

`sulci.connect()` is still the canonical entry point for two patterns that
benefit from explicit ordering:

**1. OSS-Connect device-code onboarding (browser-based auth)**

```python
import sulci

sulci.connect(prompt=True)            # opens browser, registers key in ~/.sulci/config
cache = Cache(backend="sqlite")       # cache lives locally; telemetry already wired
```

**2. Register key at boot, construct Cache later (multi-worker apps)**

```python
# At app startup, before workers spawn:
import sulci
sulci.connect(api_key="sk-sulci-...")

# Later, in a worker thread / lazy init:
from sulci import Cache
cache = Cache(backend="sulci")        # picks up the already-registered key
```

Both flows short-circuit the v0.7.0 auto-connect logic by setting
`sulci._telemetry_enabled` to its intended state before `Cache()` runs.
The auto-connect block respects that and does nothing — your explicit
`connect()` choice (including `telemetry=False` if you passed it) survives.

**Key resolution order** (first match wins):

```
1. Explicit api_key= argument to sulci.connect() or Cache()
2. SULCI_API_KEY environment variable
3. ~/.sulci/config (persisted by a prior successful sulci.connect() call)
4. Browser-based OSS-Connect device-code flow — only if prompt=True
```

All four are equivalent opt-in signals per the §5.2 trust-boundary spec.
Pre-v0.7.0 only paths (1) via `connect()`, (2), and (3) flipped the
telemetry flag; v0.7.0 makes path (1) via `Cache(api_key=...)` equivalent
to the others, which is what eliminates the historic footgun.

Step 3 (config persistence) ships in **v0.5.3**. After your first successful
`sulci.connect(api_key="sk-sulci-...")`, the key is persisted to
`~/.sulci/config` (mode 0600) and subsequent `sulci.connect()` calls with
no arguments will pick it up automatically.

Step 4 (device-code flow) ships **latent** in v0.5.3. The SDK code is in
place, but the gateway endpoints and dashboard page need to deploy
end-to-end before it's usable. The `prompt` parameter defaults to `False`
in v0.5.3+:

```python
# v0.5.3+ default — safe everywhere:
sulci.connect()
# - Steps 1-3 work normally
# - Step 4 is skipped (prompt=False default)
# - If no key found, connect() returns silently (no telemetry enabled)

# Once your environment has OSS-Connect end-to-end deployed
# (gateway + dashboard), opt in:
sulci.connect(prompt=True)
# - First-run: prints "Visit https://app.sulci.io/oss-connect and enter code: WXYZ-2345"
# - User authorizes via browser → SDK gets api_key and persists to ~/.sulci/config
# - Subsequent runs: step 3 short-circuits, no browser needed
```

**A future release** will flip the `prompt` default to `True` once the full
OSS-Connect chain (gateway endpoints + dashboard page) is announced as
publicly available. v0.6.0 was originally pencilled in for this flip; it
shipped (2026-05-11) focused on the cloud transport rewrite instead, so
`prompt` is still `False`-by-default in v0.6.x and v0.7.0. **Setting
`prompt=True` against an environment that hasn't announced OSS-Connect
availability is user error** — wait for the release announcement.

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

> **Requires:** `pip install "sulci[sqlite]" anthropic`
>
> ```bash
> export ANTHROPIC_API_KEY=sk-ant-...
> ```

```python
import anthropic
from sulci import Cache

cache = Cache(backend="chroma", threshold=0.85)
client = anthropic.Anthropic()

def call_llm(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

result = cache.cached_call(
    query  = "How do I deploy to AWS?",
    llm_fn = call_llm,
)

print(result["response"])
print(f"Source:  {result['source']}")       # "cache" or "llm"
print(f"Latency: {result['latency_ms']:.1f}ms")
```

Run it a second time with the same (or similar) question — `source` switches to `"cache"` and latency drops from ~2,000ms to under 10ms.

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
| `cache.get(query, *, tenant_id=None, user_id=None, session_id=None, plan=None)`                   | `(str\|None, float, int)` | response, similarity, context_depth (tenant_id added in v0.4.0; plan added in v0.5.6) |
| `cache.set(query, response, *, tenant_id=None, user_id=None, session_id=None, metadata=None, plan=None)` | `None`                    | Store entry, advance context window (plan added in v0.5.6)                            |
| `cache.cached_call(query, llm_fn, *, tenant_id=None, user_id=None, session_id=None, cost_per_call=0.005, plan=None)` | `dict`        | response, source, similarity, latency_ms, cache_hit, context_depth (plan added in v0.5.6) |
| `cache.get_context(session_id)`                                                        | `ContextWindow`           | Return session's context window                                    |
| `cache.clear_context(session_id)`                                                      | `None`                    | Reset session history                                              |
| `cache.context_summary(session_id=None)`                                               | `dict`                    | Snapshot of one or all sessions                                    |
| `cache.stats()`                                                                        | `dict`                    | hits, misses, hit_rate, saved_cost, total_queries, active_sessions |
| `cache.clear()`                                                                        | `None`                    | Evict all entries, reset stats and sessions                        |

> **Important:** `cache.get()` returns a **3-tuple** `(response, similarity, context_depth)` — not a 2-tuple like v0.1. Always unpack all three values.

### v0.5.0 additions

Two additive constructor kwargs for advanced deployments:

```python
from sulci import Cache, RedisSessionStore, TelemetrySink

cache = Cache(
    backend        = "sqlite",
    context_window = 4,
    session_store  = RedisSessionStore(redis_client),         # horizontal-scale sessions
    event_sink     = TelemetrySink("https://your.endpoint"),  # privacy-firewalled events
)
```

- `session_store=` accepts any `sulci.sessions.SessionStore` impl. Default `None` uses the legacy in-process manager (unchanged from v0.4.x).
- `event_sink=` accepts any `sulci.sinks.EventSink` impl. Default `None` uses `NullSink()` (no-op). Shipped sinks (`TelemetrySink`, `RedisStreamSink`) enforce a strict field allowlist — query text, response text, and embeddings never leave the process.
- `SyncCache` is now exported as a naming-symmetric alias for `Cache` (parallel to `AsyncCache`).

### v0.5.2 additions

Connected-OSS telemetry — opt-in, anonymous, never enabled by default. Pairs with the Sulci dashboard at [sulci.io](https://sulci.io) to give you persistent stats, multi-machine fingerprinting, and savings reports without sending any query content.

```python
import sulci

# Opt in — sends aggregate counts + a stable per-deployment fingerprint
sulci.connect(api_key="sk-sulci-...")

cache = sulci.Cache(backend="sqlite", threshold=0.85)
# cache.get() / cache.set() now buffer aggregated metrics for /v1/telemetry
```

What's new at the wire level:

- **Per-deployment fingerprint** — `blake2b(machine_id || backend || embedding_model || threshold || context_window)` truncated to 24 hex chars. The `machine_id` is a fresh `uuid4` generated once and persisted at `~/.sulci/config` (mode 0600). No PII, no MAC, no hostname. Switching backends produces a new fingerprint, which the dashboard treats as a new deployment.
- **`cache.set` events** are now buffered and POSTed alongside `cache.get` events — gives the dashboard a write-vs-read picture of your cache.
- **Privacy firewall is unchanged.** Wire payload is locked to nine allowlisted fields (`event`, `backend`, `hits`, `misses`, `avg_latency_ms`, `sdk_version`, `python_version`, `fingerprint`). The gateway uses `extra='forbid'` server-side as a hard rejection.

New top-level modules:

- **`sulci.config`** — small persistent SDK config at `~/.sulci/config`. `load()`, `save()`, `update()`, `get_machine_id()`. Atomic write, silent fallback on corruption.
- **`sulci.telemetry`** — fingerprint helper + wire-field allowlist for the `connect()` emit pipe. Distinct from `sulci.sinks.telemetry` (the per-event `EventSink` implementation from v0.5.0).

Passive nudge:

After 100 cached queries on a `Cache` instance that hasn't been connected, `cache.stats()` prints a one-line nudge to stderr suggesting `sulci.connect()`. One-shot per process. Silence with `SULCI_QUIET=1` or by calling `sulci.connect()`.

```bash
export SULCI_QUIET=1   # silences the nudge globally
```

### v0.5.3 additions

OSS-Connect device-code SDK client (D12). The flow ships **latent** —
SDK code is in place, but `prompt` defaults to `False` because the
gateway endpoints and dashboard page need to deploy end-to-end before
the flow is usable. **A future release will flip `prompt` to `True` once
the full chain ships.** Setting `prompt=True` against an environment that
hasn't announced OSS-Connect availability is user error.

> **2026-05-11 note:** v0.6.0 was originally pencilled in for the
> `prompt`-default flip. v0.6.0 shipped focused on the cloud-transport
> rewrite instead (umbrella sulci-oss #63 — see the v0.6.0 additions
> section above); the OSS-Connect prompt flip remains deferred to a
> future release with no committed version target yet.

```python
import sulci

# v0.5.3 default — completely safe (still the default in v0.6.x):
sulci.connect(api_key="sk-sulci-...")     # the v0.5.x flow, unchanged
sulci.connect()                            # falls through args/env/config; no browser

# After the Sulci team announces OSS-Connect availability:
sulci.connect(prompt=True)                 # browser-based onboarding
```

What's new at the SDK level:

- **`sulci.oss_connect`** — RFC 8628 device-code flow client. Lazy-imported only on the no-key-found path so `import sulci` cost is unchanged for users who never trigger it.
- **Four-step `sulci.connect()` resolution** — `arg → env → ~/.sulci/config → device-code flow`. The third step (config-persisted key) is new in v0.5.3 — your first successful `sulci.connect(api_key=...)` persists the key, and subsequent `sulci.connect()` calls with no arguments pick it up automatically.
- **`prompt: bool = False`** — keyword parameter. Default flip to `True` deferred to a future release (was originally targeted at v0.6.0; v0.6.0 shipped cloud-transport instead).
- **`SULCI_GATEWAY` env var** — overrides the gateway base URL (default `https://api.sulci.io`). Used for staging / local-dev. **In v0.5.5+ a single value drives both telemetry POSTs and the device-code client.** In v0.5.0-v0.5.4 this env var only redirected the device-code flow; telemetry stayed pinned to `api.sulci.io` regardless. See the v0.5.5 additions section below.

### v0.5.4 additions

D7 enabler bundle — five paper-cut fixes that ship alongside sulci-platform's dashboard `/oss-connect` page work. No new public API surface; one observable behavior change called out below.

What's new at the SDK level:

- **Startup events on the wire** — `sulci.connect()`'s `_emit("startup", {})` now reaches `/v1/telemetry` instead of being drained on the floor. One POST per flush cycle that contains any startup event; backend is sniffed from any non-startup event in the same batch (or `""` if cache traffic hasn't started yet — gateway accepts both, fingerprint dedupes the dashboard row). Result: a fresh deployment shows up on the dashboard before its first `cache.get`/`cache.set`.
- **`Cache.stats()` now reflects raw `.get()`/`.set()` users.** Previously `_stats["hits"]/["misses"]` only incremented inside `cached_call()`, so anyone using the raw API saw `{"hits": 0, "misses": 0}` regardless of activity. The increments moved into `Cache.get()` itself; `cached_call()` no longer increments them (it goes through `.get()`, so existing hit/miss counts from `cached_call()`-only callers are identical to before). `saved_cost` stays a `cached_call()`-only metric. **Behavior change to flag:** assertions that assumed raw `.get()` was a stats no-op need updating.
- **Examples are idempotent across re-runs.** `basic_usage.py`, `anthropic_example.py`, `context_aware.py`, and `context_aware_example.py` switched from `./sulci_db` (default, polluted the working tree) and hardcoded `/tmp/sulci_ctx_demo*` paths to per-run `tempfile.mkdtemp(prefix="sulci_<demo>_")`. `async_example.py` and `llamaindex_example.py` already used this pattern.
- **Examples fail fast on rejected API keys.** `anthropic_example.py` and `async_example.py` catch `anthropic.AuthenticationError` and `openai.AuthenticationError` on first call, print a one-line "key rejected — verify at <provider URL>" message, and fall back to mock LLM for the rest of the demo. Previously a stale or wrong key surfaced as a raw `HTTPStatusError` traceback mid-output.
- **PyPI metadata: `authors` block + `Changelog` URL** in `pyproject.toml`. After the next release, `pip show sulci` surfaces `Author:` and the PyPI sidebar shows the Changelog link.

### v0.5.5 additions

One-line fix that makes `SULCI_GATEWAY` actually redirect telemetry POSTs — closing the comment-vs-code gap that's been silently in place since v0.5.0. No new public API; default behavior unchanged for anyone not setting the env var.

What's new at the SDK level:

- **`SULCI_GATEWAY` redirects telemetry now.** `_TELEMETRY_URL` is derived from `_GATEWAY_BASE` instead of being a separate hardcoded literal. Setting `SULCI_GATEWAY=https://staging.example.com` redirects both the v0.6.0 device-code flow *and* the v0.5.x telemetry pipeline; previously it only redirected the former, contradicting the in-source comment that claimed otherwise. Concretely:

  ```python
  # v0.5.4
  SULCI_GATEWAY=https://staging.example.com python -c "import sulci; print(sulci._TELEMETRY_URL)"
  # → https://api.sulci.io/v1/telemetry              (env var ignored — bug)

  # v0.5.5
  SULCI_GATEWAY=https://staging.example.com python -c "import sulci; print(sulci._TELEMETRY_URL)"
  # → https://staging.example.com/v1/telemetry      (env var honored — fixed)
  ```

  This unblocks staging-gateway smoke tests where the published wheel needs to point at a non-prod gateway (e.g. the Railway staging URL pre-DNS-cutover). See LOCAL_SETUP.md Step 9 for the full local + staging walkthrough.

- **6 new regression tests** in `tests/test_telemetry_gateway_override.py` covering default URL, env override, trailing-slash normalization, localhost-for-local-dev, and end-to-end verification that `_post()` honors the resolved URL.

- **Out-of-scope follow-up.** `sulci/backends/cloud.py` (the `Cache(backend="sulci")` HTTP backend) still hardcodes `CLOUD_URL = "https://api.sulci.io"` and only honors a programmatic `gateway_url=` kwarg, not `SULCI_GATEWAY`. Tracked separately for a future minor — `Cache(backend="sulci")` users today should pass `gateway_url=os.environ["SULCI_GATEWAY"]` explicitly if they want symmetry.

### v0.5.6 additions

Additive `plan` field on the v0.5.0 `CacheEvent` dataclass + matching keyword
argument on `Cache.get` / `Cache.set` / `Cache.cached_call`, so callers who
know a tenant's plan tier at emit time can attribute it onto the event without
monkey-patching the dataclass or doing a join at consume time. Backward-
compatible per ADR 0005's "additive kwarg with default" rule — pre-0.5.6
callers see no behavior change; emitted events default to `plan=None`.

```python
from sulci import Cache

cache = Cache(backend="sqlite", context_window=4)

# When the caller knows the tenant's plan tier, attribute it onto the event:
response, sim, depth = cache.get(query, session_id="s1", plan="pro")
cache.set(query, response, session_id="s1", plan="pro")
```

What's new at the SDK level:

- **`CacheEvent.plan: Optional[str] = None`.** New field on the privacy-
  firewalled event surface, sitting alongside `tenant_id`. Carries the
  customer plan tier (`'free' | 'pro' | 'business' | 'enterprise' |
  'oss_connect'`) when the caller knows it. Defaults to `None` so users of
  the OSS library who don't have plan context don't have to thread anything
  through.
- **`plan: Optional[str] = None`** added as a keyword-only argument to
  `Cache.get`, `Cache.set`, and `Cache.cached_call`. When supplied, it is
  forwarded onto the emitted `CacheEvent.plan`. `cached_call` threads it
  through both its internal `.get()` and `.set()` calls so the miss-then-set
  path emits two events that both carry plan.
- **`"plan"` added to `_ALLOWED_FIELDS`** in `sulci/sinks/telemetry.py` so it
  survives the privacy firewall and reaches `TelemetrySink` /
  `RedisStreamSink` consumers. The allowlist's docstring now articulates the
  three-criteria rule for future additions: a candidate field must be (a)
  low-cardinality, (b) already known to the recipient via auth context, and
  (c) explicitly billing- or routing-relevant.

See the v0.5.6 entry in [`CHANGELOG.md`](./CHANGELOG.md) for the full privacy
review note and compatibility section.

### v0.5.7 additions

Cloud-backend route fix — `SulciCloudBackend` now POSTs to the gateway's
canonical paths (`/v1/cache/get` + `/v1/cache/set`) rather than the legacy
`/v1/get` + `/v1/set` paths it had been sending to since v0.3.0. Closes
sulci-oss #57.

The pre-v0.5.7 surface was a silent failure: the gateway returned `404 Not
Found` and the cloud backend's outer `except Exception: pass` clause
swallowed the error, returning `(None, 0.0)` to the caller — indistinguishable
from a genuine cache miss. Cloud-tier customers had **never** seen a real
cache hit since the cloud backend was introduced; the silent route mismatch
guaranteed it. v0.5.7 fixed the routes; v0.6.0 then fixed the deeper
contract mismatch sitting one layer below (see below).

What's new at the SDK level:

- **Three string changes** in `sulci/backends/cloud.py` (lines 101, 150, 179):
  `/v1/get` → `/v1/cache/get`, `/v1/set` → `/v1/cache/set`, plus the matching
  fix in `delete_user()`.
- **`TestCanonicalGatewayPaths` regression-guard class** (4 new tests in
  `tests/test_cloud_backend.py`) — pins each URL-bearing method to the
  gateway's canonical path plus a static-source check that scans `cloud.py`
  for legacy `/v1/get` / `/v1/set` strings. Replaces the pre-existing
  tautological `TestSearch.test_sends_correct_payload` test that asserted
  `call_args[0][0] == "/v1/get"` (verifying what the SDK did rather than what
  the gateway expected — the exact pattern that let this slip through CI for
  14 months).

### v0.6.0 additions

**Cloud transport finally works end-to-end.** The largest customer-facing
change since v0.3.0. After 14 months of silent `(None, 0.0)` misses,
`Cache(backend="sulci")` returns real cache hits against the production
gateway. Three coordinated PRs under umbrella sulci-oss
[#63](https://github.com/sulci-io/sulci-oss/issues/63):

```python
# This now actually works (it didn't, v0.3.0 through v0.5.7)
cache = sulci.Cache(backend="sulci", api_key="sk-sulci-...")
cache.set("What is Python?", "A programming language.")
response, sim, depth = cache.get("What is Python?")
# → response='A programming language.'  sim=1.000  depth=0
```

What's new at the SDK level:

- **Native `Embedder` and `Backend` instance injection in `Cache.__init__`.**
  `Cache(embedding_model=..., backend=...)` now accepts either a string
  (`"minilm"`, `"sqlite"`, etc. — the v0.5.x path, unchanged) or a
  pre-constructed instance:
  ```python
  from sulci.embeddings.openai import OpenAIEmbedder
  from sulci.backends.qdrant import QdrantBackend

  cache = sulci.Cache(
      embedding_model = OpenAIEmbedder(model="text-embedding-3-small"),
      backend         = QdrantBackend(url="https://my-cluster.qdrant.io"),
  )
  ```
  Closes sulci-oss #34 sub-issues C1c (Embedder injection) and C1d (Backend
  injection). Enables advanced deployments — connection pooling, custom
  client configuration, multi-tenant Backend instances — without subclassing.

- **Cloud transport short-circuits local embedding.** When the backend is
  the cloud transport, `Cache.get` / `Cache.set` forward the raw query string
  to the gateway instead of embedding locally. The gateway-side library does
  the embedding via `EmbedServiceEmbedder`, then runs the ANN search and
  emits the billing event. Closes sulci-oss
  [#62](https://github.com/sulci-io/sulci-oss/issues/62). Detection is
  capability-based (`hasattr(backend, "remote_get")`), so the cloud module's
  `httpx` import stays lazy.

- **`SulciCloudBackend` is now a transport, not a `Backend` impl.**
  `.search()` / `.store()` / `.upsert()` removed; replaced with
  `remote_get(query, threshold, *, user_id, session_id) -> (response,
  similarity, context_depth)` and `remote_set(query, response, *, user_id,
  session_id, ttl_seconds) -> None`. Wire payloads now match the gateway's
  pydantic models (`CacheGetRequest` / `CacheSetRequest`) exactly. Vendored
  the gateway pydantic contract into the test fixture with a sync comment so
  the SDK payload and gateway contract can't silently drift again.

- **20 new tests across `TestRemoteGet` / `TestRemoteSet` /
  `TestCloudTransportShortCircuit`** in `tests/test_cloud_backend.py`,
  replacing the pre-v0.6.0 broken `TestSearch` / `TestUpsert` classes.

**Self-hosted backends (chroma, qdrant, faiss, redis, sqlite, milvus) are
completely unaffected.** All v0.5.x usage patterns continue to work
identically; the conditional only fires when the backend exposes the
`remote_get` / `remote_set` duck-type protocol that v0.6.0 introduced.

### v0.6.1 additions

Cloud-only install path fix. Closes sulci-oss
[#60](https://github.com/sulci-io/sulci-oss/issues/60). Discovered during the
v0.6.0 release smoke test: `pip install "sulci[cloud]==0.6.0"` followed by
`Cache(backend="sulci", api_key=...)` crashed at construction with
`ImportError: sentence-transformers not found`. The `sulci[cloud]` extra
correctly declares only `httpx>=0.27` (no sentence-transformers, since the
cloud transport doesn't do local embedding), but `Cache.__init__` eagerly
loaded `MiniLMEmbedder` regardless of which backend was selected.

After v0.6.1, the minimal cloud-only install works end-to-end:

```bash
pip install "sulci[cloud]"        # httpx only — no sentence-transformers
```
```python
cache = sulci.Cache(backend="sulci", api_key="sk-sulci-...")
cache.set("What is Python?", "A programming language.")
# All cloud calls round-trip through the gateway. Works.
```

What's new at the SDK level:

- **`Cache.__init__` defers local `Embedder` load on the cloud transport.**
  Construction order is now (1) load backend, (2) detect remote transport
  via `hasattr(backend, "remote_get")`, (3) skip embedder load if the flag
  is set. `self._embedder` stays `None` on the cloud path; every read of it
  is already gated behind `self._is_remote_transport` or sits inside the
  self-hosted `else:` branch.
- **`cached_call` hit-record session path now skips on remote transport** —
  mirrors the same-pattern guard already present in `Cache.set`. Without
  this gate, a cache hit on a session-aware cloud `Cache`
  (`backend="sulci"` + `context_window > 0`) would crash on `None.embed()`.
- **Friendlier `ImportError`** when constructing a cloud `Cache` without
  `httpx` installed — re-raises with `pip install "sulci[cloud]"` advice
  instead of the bare `ModuleNotFoundError: No module named 'httpx'`.
- **New `TestCloudTransportNoLocalEmbedder` class** (4 tests in
  `tests/test_core.py`) — verifies the embedder stays `None` on a fake
  remote-transport backend, and that `Cache.get` / `Cache.set` /
  `cached_call` all route through the transport without touching the
  embedder.

Self-hosted backends preserve identical behavior to v0.6.0.

---

## Context-Aware Blending

When `context_window > 0`, Sulci Cache blends the current query vector with recent
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
| **Sulci Cloud** | `sulci`  | <8ms        | **Zero infra — managed service**        |

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
├── Makefile                    ← make smoke, make test, make test-all, make verify
├── NOTICE
├── README.md
├── benchmark
│   ├── README.md               ← benchmark methodology and results
│   └── run.py                  ← benchmark CLI (--context for context-aware pass)
├── examples
│   ├── anthropic_example.py    ← Anthropic Claude, context-aware, requires ANTHROPIC_API_KEY
│   ├── basic_usage.py          ← stateless cache demo, no API key needed
│   ├── context_aware.py        ← 4-demo walkthrough, fully offline
│   ├── context_aware_example.py← additional context-aware patterns
│   ├── langchain_example.py    ← LangChain integration, OpenAI/Anthropic/mock
│   ├── llamaindex_example.py   ← LlamaIndex integration, OpenAI/Anthropic/mock
│   └── async_example.py        ← AsyncCache demo, OpenAI/Anthropic/mock    (v0.3.7)
├── pyproject.toml              ← name="sulci", version="0.5.6"
├── setup.py
├── setup.sh                    ← one-shot setup: venv + install + smoke tests
├── smoke_test.py               ← core smoke test
├── smoke_test_langchain.py     ← LangChain integration smoke test
├── smoke_test_llamaindex.py    ← LlamaIndex integration smoke test
├── smoke_test_async.py         ← AsyncCache smoke test                     (v0.3.7)
├── sulci
│   ├── __init__.py             ← exports Cache, SyncCache, AsyncCache, ContextWindow,
│   │                              SessionStore (legacy), InMemorySessionStore,
│   │                              RedisSessionStore, EventSink, NullSink,
│   │                              TelemetrySink, RedisStreamSink, CacheEvent, connect()
│   │                              _SDK_VERSION = __version__   # derived from pyproject.toml
│   ├── backends
│   │   ├── __init__.py         ← empty — core.py loads backends via importlib
│   │   ├── chroma.py
│   │   ├── cloud.py            ← SulciCloudBackend (backend="sulci")
│   │   ├── faiss.py
│   │   ├── milvus.py
│   │   ├── qdrant.py
│   │   ├── redis.py
│   │   └── sqlite.py
│   ├── async_cache.py          ← AsyncCache non-blocking wrapper         (v0.3.7)
│   ├── context.py              ← ContextWindow + legacy SessionStore manager
│   ├── core.py                 ← Cache engine + B1 adapter (v0.5.0)
│   │                              telemetry= param, api_key= param,
│   │                              session_store= + event_sink= kwargs (v0.5.0)
│   ├── embeddings
│   │   ├── __init__.py
│   │   ├── minilm.py           ← default: all-MiniLM-L6-v2 (free, local)
│   │   └── openai.py           ← requires OPENAI_API_KEY
│   ├── sessions                ← v0.5.0 — SessionStore protocol package
│   │   ├── __init__.py
│   │   ├── protocol.py         ← public stable SessionStore protocol
│   │   ├── memory.py           ← InMemorySessionStore (default)
│   │   └── redis.py            ← RedisSessionStore (multi-replica)
│   ├── sinks                   ← v0.5.0 — EventSink protocol package
│   │   ├── __init__.py
│   │   ├── protocol.py         ← public stable EventSink + CacheEvent
│   │   ├── null.py             ← NullSink (default no-op)
│   │   ├── telemetry.py        ← TelemetrySink (HTTPS POST, allowlist-scrubbed)
│   │   └── redis_stream.py     ← RedisStreamSink (XADD, allowlist-scrubbed)
│   └── integrations
│       ├── __init__.py
│       ├── langchain.py        ← SulciCache(BaseCache) for LangChain  (v0.3.3)
│       └── llamaindex.py       ← SulciCacheLLM(LLM) for LlamaIndex    (v0.3.5)
└── tests
    ├── test_backends.py                —   9 tests: per-backend contract + persistence
    ├── test_cloud_backend.py           —  28 tests: SulciCloudBackend + Cache wiring
    ├── test_connect.py                 —  40 tests: sulci.connect(), _emit(), _flush()
    ├── test_context.py                 —  35 tests: ContextWindow, legacy SessionStore
    ├── test_core.py                    —  41 tests: cache.get/set, TTL, stats (incl. raw-get/set), personalization, tenant_id, CacheEvent.plan (v0.5.6)
    ├── test_integrations_langchain.py  —  27 tests: SulciCache LangChain adapter
    ├── test_integrations_llamaindex.py —  29 tests: SulciCacheLLM LlamaIndex wrapper
    ├── test_async_cache.py             —  25 tests: AsyncCache non-blocking wrapper       (v0.3.7)
    ├── test_qdrant_tenant_isolation.py —  11 tests: tenant_id partition isolation         (v0.4.0)
    ├── test_sessions.py                —  24 tests: SessionStore protocol + tenant isol.  (v0.5.0)
    ├── test_sinks.py                   —  20 tests: EventSink protocol + privacy allowlist (v0.5.0; +plan scrub tests v0.5.6)
    ├── test_session_store_injection.py —  12 tests: Cache(session_store=, event_sink=)    (v0.5.0)
    ├── test_config.py                  —  20 tests: ~/.sulci/config — load/save/0600 perms (v0.5.2)
    ├── test_telemetry.py               —  28 tests: fingerprint helper + flush wire shape (incl. startup-events) (v0.5.2 / v0.5.4)
    ├── test_nudge.py                   —  13 tests: 100-query nudge in Cache.stats()       (v0.5.2)
    ├── test_oss_connect.py             —  17 tests: RFC 8628 device-code client            (v0.5.3)
    ├── test_telemetry_gateway_override.py —  6 tests: SULCI_GATEWAY redirect for telemetry (v0.5.5)
    └── compat/                         —  Backend + Embedder conformance suites

Plus: sulci/tests/compat/ — SessionStore + EventSink conformance suites (v0.5.0)
```

---

## Running Tests

```bash
# full suite — 385 tests total (skipped backend tests if optional deps not installed)
python -m pytest tests/ -v

# by file
python -m pytest tests/test_core.py -v                       # 41 tests
python -m pytest tests/test_context.py -v                    # 35 tests
python -m pytest tests/test_backends.py -v                   #  9 tests (skipped if dep missing)
python -m pytest tests/test_connect.py -v                    # 40 tests — sulci.connect() + telemetry
python -m pytest tests/test_cloud_backend.py -v              # 28 tests — SulciCloudBackend
python -m pytest tests/test_integrations_langchain.py -v     # 27 tests — LangChain integration
python -m pytest tests/test_integrations_llamaindex.py -v    # 29 tests — LlamaIndex integration
python -m pytest tests/test_async_cache.py -v                # 25 tests — AsyncCache wrapper

# single backend only
python -m pytest tests/test_backends.py -v -k sqlite
python -m pytest tests/test_backends.py -v -k chroma

# with coverage
python -m pytest tests/ -v --cov=sulci --cov-report=term-missing
```

### Make targets

```bash
make smoke              # all smoke tests (core + LangChain + LlamaIndex)
make smoke-core         # core smoke test only
make smoke-langchain    # LangChain smoke test only
make smoke-llamaindex   # LlamaIndex smoke test only
make smoke-async        # AsyncCache smoke test only
make test               # core pytest suite
make test-integrations  # LangChain + LlamaIndex integration tests
make test-async         # AsyncCache tests only
make test-all           # full suite (212 tests)
make test-cov           # full suite with coverage
make verify             # smoke + test-all (run before committing)
```

`test_connect.py` (32 tests) — `sulci.connect()`, `_emit()`, `_flush()`, `Cache(telemetry=)`. Requires `httpx`.

`test_cloud_backend.py` (28 tests) — `SulciCloudBackend` construction, `search()`, `upsert()`, `delete_user()`, `clear()`, and `Cache(backend='sulci')` wiring. Requires `httpx`.

`test_integrations_langchain.py` (27 tests) — `SulciCache(BaseCache)` LangChain adapter. Requires `langchain-core`.

`test_integrations_llamaindex.py` (29 tests) — `SulciCacheLLM(LLM)` LlamaIndex wrapper. Requires `llama-index-core`.

Backend tests are **skipped — not failed** when their dependency isn't installed.
Install the backend extra to run its tests: `pip install -e ".[chroma]"`.

See [`LOCAL_SETUP.md`](./LOCAL_SETUP.md) for the full local development guide including
venv setup, backend installation, smoke testing, and troubleshooting.

---

## Examples

```bash
python examples/basic_usage.py          # stateless cache — no API key needed
python examples/context_aware.py        # context-aware — no API key needed
python examples/anthropic_example.py    # requires ANTHROPIC_API_KEY
python examples/langchain_example.py    # OpenAI or Anthropic or mock fallback
python examples/llamaindex_example.py   # OpenAI or Anthropic or mock fallback
python examples/async_example.py        # AsyncCache demo, OpenAI/Anthropic/mock
```

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

## Troubleshooting

### `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`

Conda environments often have a stale `huggingface_hub` that conflicts with `sentence-transformers`. Fix by upgrading all three together:

```bash
pip install --upgrade huggingface_hub datasets sentence-transformers
```

Or use a clean venv (avoids conda transitive dependency conflicts entirely):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "sulci[sqlite]" anthropic
python your_script.py
```

### `huggingface/tokenizers: The current process just got forked...` warning

Harmless — suppress it with:

```bash
export TOKENIZERS_PARALLELISM=false
```

### `anthropic.OverloadedError: Error code: 529`

Transient API congestion — not a Sulci Cache issue. Wait a moment and retry, or check [status.anthropic.com](https://status.anthropic.com).

### `zsh: no matches found: sulci[chroma]`

Wrap extras in quotes:

```bash
pip install "sulci[chroma]"    # ✓
pip install sulci[chroma]      # ✗ — zsh glob expansion breaks this
```

### `pytest: command not found`

```bash
python -m pytest tests/ -v
```

---

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for branching model, PR process, and coding standards.

---

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE).

Copyright 2026 Kathiravan Sengodan.

U.S. Patent Application No. 64/018,452 (pending) covers the
context-aware semantic caching algorithm. Apache 2.0 grants users
a royalty-free patent license for use of this code.

---

## Links

- **Website:** [sulci.io](https://sulci.io?utm_source=github&utm_medium=readme&utm_campaign=oss)
- **Sign up (free key):** [sulci.io/signup](https://sulci.io/signup?utm_source=github&utm_medium=readme&utm_campaign=oss)
- **API:** [api.sulci.io](https://api.sulci.io)
- **PyPI:** [sulci](https://pypi.org/project/sulci/)
- **GitHub:** [sulci-io/sulci-oss](https://github.com/sulci-io/sulci-oss)
- **Issues:** [github.com/sulci-io/sulci-oss/issues](https://github.com/sulci-io/sulci-oss/issues)
