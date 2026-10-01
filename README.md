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
| $0.005 per call, every time  | Cache hits need a local embedding, not an LLM call       |
| 1–3 second response time     | Cache hits: ~87 ms p50 on the benchmark machine (below)  |
| No memory across sessions    | Context-aware: understands conversation history          |

**Benchmark results — measured on the shipped MiniLM embedder**, 5,000-query
corpus, `threshold=0.85`. Ranges are across four seeded draws
(`--seed 1 2 3 42`):

| | measured | range | n |
|---|---|---|---|
| Repeat questions served from cache | **99.9%** | `[99.85–99.95%]` | 4,000 |
| Correctly declined when it has no answer | **98.9%** | `[98.15–99.38%]` | 2,588 |
| Of all cache hits, share that are correct | **93.8%** | `[92.13–94.69%]` | 4,014 |
| Cache hit, p50 | **87.19ms** | `[86.96–87.73ms]` | 4,014 |

Reproduce:

```bash
pip install -e ".[sqlite]"
python3 benchmark/run.py --use-sulci --fresh --no-sweep --seed 1
```

⚠️ **87.19ms is environment-dependent** — it is a MiniLM forward pass on the
benchmark machine, not a property of the cache. Measure it on your own hardware
before planning against it.

**Context-aware mode** blends prior conversation turns into the lookup vector so
follow-ups resolve against the session rather than the whole cache. How much
that helps depends entirely on your query mix, and **this README does not carry
a number for it.** Run `python3 benchmark/run.py --use-sulci --fresh --no-sweep
--context` and read the harness's own note: the context corpus is 125 follow-ups
across 25 sessions, which is small.

> Figures previously in this block — `85.9%`, `0.74ms`, `$21.47`, and a
> `56.8% → 77.6% (+20.8pp)` context claim on an "800-pair" corpus — were
> **retired 2026-08-12** and replaced above. They were measured on the built-in
> TF-IDF engine, which ships in no product, and the corpus was 125 follow-ups
> rather than 800. `pyproject.toml` makes this file the PyPI project page, so
> they were the most widely-read copy in the estate. See
> `sulci-platform docs/marketing/CLAIMS.md` for the register.

**Agent workloads** are the headline result, measured on real MiniLM across
four corpus draws (`--seed 1 2 3 42`, n=10,000):

| | measured |
|---|---|
| aggregate per-session hit rate | **95.01%** `[94.99–95.05]` |
| cold session | **33.4%** `[30.0–35.0]`, n=200 |
| warm session | **99.74%** `[99.69–99.81]`, n=200 |

Reproduce with `benchmark/run.py --agent --use-sulci --fresh --seed 1`.

> The **~60–75%** figure this paragraph used to carry was never measured. It
> came from one line in `benchmark/README.md` that asserted it and called it
> "the number to cite externally." Retired 2026-08-04; `CLAIMS.md` recorded
> the correction as landed, and it had not landed here. Fixed 2026-08-11.

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
pip install sulci              # Sulci Cloud managed backend — no extra needed (httpx is included)
```

**LangChain integration:**

```bash
pip install "sulci[sqlite,langchain]"   # + LangChain integration
```

**LlamaIndex integration:**

```bash
pip install "sulci[sqlite,llamaindex]"  # + LlamaIndex native integration
```

**MCP server, caching proxy, LiteLLM** (Python 3.10+):

```bash
pip install "sulci[sqlite,mcp]"          # + MCP server (sulci-mcp)
pip install "sulci[sqlite,proxy]"        # + OpenAI/Anthropic caching proxy (sulci-proxy)
pip install "sulci[sqlite,litellm]"      # + LiteLLM cache layer
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

# Context-aware — chatbot / agent
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

# Context-aware — RAG chatbot / agent
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

**Alternative — route LlamaIndex through the LangChain adapter:**

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


## MCP Server

Agents that run in containers — Copilot CLI, Claude Code, Codex, GitHub
Agentic Workflows — cannot `pip install sulci` into their own process. They
can call an MCP server.

```bash
pip install "sulci[sqlite,mcp]"
sulci-mcp --backend sqlite --db-path ./sulci_db
```

Tools: `cache_lookup` and `cache_stats` (both annotated `readOnlyHint`), and
`cache_store` (annotated as a write). `SULCI_MCP_READ_ONLY=1` registers only
the read-only pair. Transports: `stdio` (default), `sse`, `streamable-http`.

Requires **mcp >= 2.0.0**. mcp 1.x is not supported — the `FastMCP` entry
point it used was removed in 2.0.

See `examples/gh_aw_sulci_mcp.md` for a GitHub Agentic Workflow that mounts
the store on gh-aw's `cache-memory` so it survives between runs.


## Caching Proxy (OpenAI / Anthropic compatible)

The zero-code-change door. Point any SDK or CLI at it and every call is
cached — no import, no adapter, no cooperation from the caller.

```bash
pip install "sulci[sqlite,proxy]"
sulci-proxy --backend sqlite --db-path ./sulci_db --port 8787

export OPENAI_BASE_URL=http://localhost:8787/v1
export ANTHROPIC_BASE_URL=http://localhost:8787
```

Serves `POST /v1/chat/completions` and `POST /v1/messages`, plus `/healthz`
and `/stats`. Every response carries `x-sulci-cache: hit | miss |
miss-uncacheable | bypass`; hits also carry `x-sulci-similarity`.

Scope a lookup with the `x-sulci-tenant-id` and `x-sulci-session-id` headers.

⚠️ **A cache hit is not authenticated.** Your `Authorization` header is
forwarded upstream only on a **miss**; a hit is served from the store without
any credential check, so an invalid or expired key still returns 200 for a
question already cached. This is inherent to every caching proxy, and it is
why `--host` defaults to `127.0.0.1`. Read access to the proxy is read access
to the cache contents — put your own auth in front of it before binding it to
anything but localhost.

⚠️ **Give it a dedicated `--db-path`.** `./sulci_db` is sulci's *default*
path, shared with anything else that omits `db_path`. Pointing the proxy at it
means unrelated entries can satisfy a lookup.

**Not cached, by design:** streaming requests (`"stream": true`) pass through
untouched; tool-call responses are never stored, because their arguments are
state-dependent and a stale one sends an agent to the wrong file. Cached
replies report `usage.total_tokens == 0` — no tokens were consumed upstream,
and fabricating the original counts would corrupt billing reconciliation.


## LiteLLM

Sulci as the cache layer inside a LiteLLM deployment, giving it the
context-aware cache it does not have.

```bash
pip install "sulci[sqlite,litellm]"
```

```python
from sulci.integrations.litellm import install
install(backend="sqlite", context_window=4)

import litellm
litellm.completion(model="gpt-4o", messages=[...])   # now cached
```

LiteLLM has no `custom` cache type — `install()` replaces the inner
implementation on `litellm.cache`, which is how LiteLLM's own semantic
caches are wired. Pass the conversation id as
`metadata={"sulci_session_id": ...}` to get context blending.


## ⚠️ The MCP, proxy and LiteLLM extras need Python 3.10+

Sulci itself supports Python 3.9. `mcp`, `litellm` and `fastapi` all declare
`requires-python >=3.10`, so `pip install "sulci[mcp]"` (or `[proxy]`,
`[litellm]`) will not resolve on 3.9. The core library, the backends and the
LangChain adapter are unaffected. Same constraint as `sulci[llamaindex]`.


## ⚠️ Tenant isolation is enforced by Qdrant and Sulci Cloud only

`tenant_id` is accepted by every backend. Two enforce isolation (see
[`docs/API-SURFACE.md`](docs/API-SURFACE.md)):

| backend | isolation |
|---|---|
| qdrant | **enforced** — filters on `tenant_id` |
| sulci (cloud) | **enforced server-side** by the Sulci Cloud gateway, keyed off the API key (one key per tenant). Its `ENFORCES_TENANT_ISOLATION` flag reads `False` only because the OSS test suite cannot reach the gateway to verify it. |
| chroma, faiss, milvus, redis, sqlite | not enforced |

On the other five the argument is accepted and ignored, so entries from
different tenants can be served for each other. This matters most for the
three surfaces above, which all offer scoping as a safety property —
`--tenant-id`, `namespace_by_model=True`, per-model proxy scoping. On the
default `sqlite` backend **each of those is a no-op**, and each will emit a
`ScopeNotEnforcedWarning` saying so rather than pretending otherwise.

If you need real isolation today: use `backend="qdrant"` or the managed `sulci` backend, or give each scope
its own `db_path`, or turn the feature off and share deliberately.

**`user_id` is a separate scope.** It takes effect only with `personalized=True`
— with the default `personalized=False` it is ignored on every backend. With
`personalized=True`, pass `user_id` on both `set` and `get`: a lookup with a
`user_id` matches only that user's entries, while a lookup without one is
unscoped and, on every backend except `qdrant`, can match any user's entry.
On `sqlite` and `milvus`, versions before 0.9.2 did not reliably keep users
apart; see the [CHANGELOG](CHANGELOG.md).

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

**Sync passthrough:** All sync methods (`get`, `set`, `cached_call`, `stats`, `clear`) also
available — `AsyncCache` works in mixed sync/async codebases without switching types.

**Partition & per-call kwargs:** every method — async **and** sync passthrough —
forwards the same set of kwargs `Cache` accepts. Keyword-only, all default `None`: `tenant_id`
(multi-tenant isolation) and `plan` (plan tier on the emitted `CacheEvent`) on
every method, `metadata` on `aset`/`set`, and the per-call `threshold` on
`aget`/`acached_call` and their passthroughs (`tenant_id`/`plan`/`metadata`
parity v0.8.1; passthrough `threshold` parity v0.8.2):

```python
resp, sim, depth = await cache.aget("...", tenant_id="acme", plan="pro", threshold=0.9)
await cache.aset("...", "...", tenant_id="acme", plan="pro", metadata={"src": "kb"})
```

> **The mirror is of *which* kwargs are forwarded, not of *how* they are passed.**
> On `aget` and `acached_call`, `threshold` — along with `user_id`, `session_id`,
> `cost_per_call` — is positional-or-keyword, whereas `Cache.get` makes everything
> after `query` keyword-only. So `await cache.aget(q, uid, sid, 0.9)` is legal
> while `cache.get(q, uid, sid, 0.9)` is a `TypeError`. Not a bug, but if you read
> "100% mirror" as full signature parity you will be surprised.
> `set` deliberately has no `threshold`, because `Cache.set` has none — the mirror
> is faithful, not a superset. Full measured surface:
> [`docs/API-SURFACE.md`](docs/API-SURFACE.md).

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

See [sulci.io](https://sulci.io) for current plans.

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

Step 4 (device-code flow) shipped **latent** in v0.5.3 — the SDK code was in
place but the gateway endpoints and dashboard page had not deployed. **That
is no longer the case: the full chain (SDK + gateway `/v1/oss-connect/*` +
dashboard `/oss-connect`) has been live end-to-end since 2026-05-08.**

`prompt` nonetheless still defaults to `False`, and as of 2026-07-06 that is
a sustained decision rather than a stale promise — see
`sulci.connect`'s docstring (`sulci/__init__.py`) for the three reasons.
Step 4 is opt-in per call:

```python
# v0.5.3+ default — safe everywhere:
sulci.connect()
# - Steps 1-3 work normally
# - Step 4 is skipped (prompt=False default)
# - If no key found, connect() returns silently (no telemetry enabled)

# Opt in to the browser flow (live since 2026-05-08):
sulci.connect(prompt=True)
# - First-run: prints "Visit https://dashboard.sulci.io/oss-connect and enter code: WXYZ-2345"
# - User authorizes via browser → SDK gets api_key and persists to ~/.sulci/config
# - Subsequent runs: step 3 short-circuits, no browser needed
```

**`prompt=False` is the permanent default.** v0.6.0 was once pencilled in to
flip it; v0.6.0 shipped (2026-05-11) focused on the cloud transport rewrite
instead, and the flip was subsequently decided against outright. The reasoning
is recorded in `sulci.connect`'s docstring and dated 2026-07-06:

1. A non-interactive default is safe for library callers with no tty or
   browser — LangChain / LlamaIndex agents, FastAPI handlers, LangGraph
   nodes, CI runners. `prompt=True` at import time would block those on a
   15-minute device-code timeout with no visible cause.
2. v0.7.0's `Cache()` auto-connect already covers the ergonomic path:
   passing `api_key=` to the constructor attaches telemetry, with no
   blocking browser prompt as an import-time side effect.
3. Explicit `prompt=True` remains first-class — one keyword argument, opt
   in per call on an interactive machine.

Treat this as settled. Do not re-file it as pending work.

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

A blended vector scores lower against stored entries than an exact-match lookup,
so with `context_threshold` unset, context lookups are compared against
`threshold` and Sulci emits a `UserWarning` saying so. Set `context_threshold=`
to a value calibrated on your own follow-up queries; sulci ships no default
because no measurement supports one. See
[`docs/context-threshold.md`](docs/context-threshold.md).

### Drop-in with `cached_call`

> **Requires:** `pip install "sulci[sqlite]" anthropic`
>
> ```bash
> export ANTHROPIC_API_KEY=sk-ant-...
> ```

```python
import anthropic
from sulci import Cache

cache = Cache(backend="sqlite", threshold=0.85)
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

Run it a second time with the same (or similar) question — `source` switches to `"cache"` and the LLM round-trip disappears: latency drops from the model's response time (typically seconds) to the local embedding lookup (~87 ms p50 on the benchmark machine; measure your own).

---

## API Reference

### Constructor

```python
cache = Cache(
    backend         = "chroma",     # chroma | sqlite | faiss | qdrant | redis | milvus | sulci
    threshold       = 0.85,         # cosine similarity cutoff (0–1)
    embedding_model = "minilm",     # minilm | mpnet | bge | openai
    ttl_seconds     = 86400,        # 24h. None = no expiry
    personalized    = False,        # scope lookups to the user_id passed on get/set
    db_path         = "./sulci_db", # on-disk path for sqlite / faiss
    context_window  = 0,            # turns to remember; 0 = stateless
    query_weight    = 0.70,         # α in blending formula
    context_decay   = 0.50,         # per-turn decay weight
    session_ttl     = 3600,         # session expiry in seconds
    telemetry       = True,         # set False to disable per-instance
    api_key         = None,         # required when backend="sulci"; also SULCI_API_KEY
    gateway_url     = "",           # cloud gateway override; also SULCI_GATEWAY (v0.7.4)
    session_store   = None,         # custom SessionStore implementation
    event_sink      = None,         # custom EventSink implementation
    cost_per_call   = 0.005,        # $ per LLM call, for saved-cost stats
    context_threshold = None,       # threshold for context-blended lookups; None = use threshold (warns)
)
```

### Methods

| Method                                                                                 | Returns                   | Description                                                        |
| -------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------ |
| `cache.get(query, *, threshold=None, tenant_id=None, user_id=None, session_id=None, plan=None)`                   | `(str\|None, float, int)` | response, similarity, context_depth (tenant_id added in v0.4.0; plan added in v0.5.6) |
| `cache.set(query, response, *, tenant_id=None, user_id=None, session_id=None, metadata=None, plan=None)` | `None`                    | Store entry, advance context window (plan added in v0.5.6)                            |
| `cache.cached_call(query, llm_fn, *, threshold=None, tenant_id=None, user_id=None, session_id=None, cost_per_call=None, plan=None)` | `dict`        | response, source, similarity, latency_ms, cache_hit, context_depth (plan added in v0.5.6) |
| `cache.get_context(session_id)`                                                        | `ContextWindow`           | Return session's context window                                    |
| `cache.clear_context(session_id)`                                                      | `None`                    | Reset session history                                              |
| `cache.context_summary(session_id=None)`                                               | `dict`                    | Snapshot of one or all sessions                                    |
| `cache.stats()`                                                                        | `dict`                    | hits, misses, hit_rate, saved_cost, total_queries, active_sessions |
| `cache.clear()`                                                                        | `None`                    | Evict all entries, reset stats and sessions                        |

> **Important:** `cache.get()` returns a **3-tuple** `(response, similarity, context_depth)` — not a 2-tuple like v0.1. Always unpack all three values.

**That is the whole public surface — eight methods, not nine.** The constructor
defaults and method signatures above match an AST measurement of `sulci/core.py`
at 0.9.1 (`python3 scripts/check_api_surface.py --show`), not memory. Re-run it
before trusting any restatement of this table found elsewhere; the full measured
surface is in [`docs/API-SURFACE.md`](docs/API-SURFACE.md).

**`delete_user` is not on `Cache`.** It is a method on `SulciCloudBackend`
(`sulci/backends/cloud.py:233`), reached through the backend or by calling
`DELETE /v1/cache/user/{id}` directly. `Cache` does not proxy it — `cache.delete_user(...)`
raises `AttributeError` — and the six self-hosted backends do not implement it,
because it is not in the `Backend` protocol. Whole-cache erasure via
`cache.clear()` works on every backend and is the portable path.

### Advanced constructor options

**Custom session store and event sink.** `session_store=` accepts any
`sulci.sessions.SessionStore` implementation (default `None` = the in-process
manager); `event_sink=` accepts any `sulci.sinks.EventSink` (default `None` = a
no-op `NullSink`). The shipped sinks enforce a strict field allowlist — query
text, response text and embeddings never leave the process.

```python
from sulci import Cache, RedisSessionStore, TelemetrySink

cache = Cache(
    backend        = "sqlite",
    context_window = 4,
    session_store  = RedisSessionStore(redis_client),         # horizontal-scale sessions
    event_sink     = TelemetrySink("https://your.endpoint"),  # privacy-firewalled events
)
```

When the caller knows a tenant's plan tier, pass `plan=` to `get` / `set` /
`cached_call`; it is attributed onto the emitted `CacheEvent`.

**Instance injection.** `embedding_model=` and `backend=` accept a
pre-constructed instance as well as a string — useful for connection pooling or
custom client configuration:

```python
from sulci.embeddings.openai import OpenAIEmbedder
from sulci.backends.qdrant import QdrantBackend

cache = Cache(
    embedding_model = OpenAIEmbedder(model="text-embedding-3-small"),
    backend         = QdrantBackend(url="https://my-cluster.qdrant.io"),
)
```

**Managed cloud backend.** With `backend="sulci"` the query text is sent to the
gateway, which embeds and searches server-side, so no local embedding model is
loaded and `pip install sulci` alone is enough. The gateway URL resolves as
`gateway_url=` → `SULCI_GATEWAY` → `https://api.sulci.io`.

`SyncCache` is exported as an alias of `Cache`, parallel to `AsyncCache`.

### Telemetry and privacy

Telemetry is opt-in (see *Sulci Cloud* above for the one rule). When enabled:

- **Eight fields, nothing else.** The wire payload is limited to `event`,
  `backend`, `hits`, `misses`, `avg_latency_ms`, `sdk_version`,
  `python_version` and `fingerprint`. No query or response text, no
  embeddings. The gateway rejects extra fields server-side.
- **Per-deployment fingerprint** — a hash of a random `machine_id` plus
  backend, embedding model, threshold and context window. The `machine_id` is a
  `uuid4` generated once and stored in `~/.sulci/config` (mode 0600); no
  hostname, MAC or path.
- **Stale keys are refused.** A key persisted in `~/.sulci/config` more than
  90 days ago, or with no write timestamp, is ignored with a warning; pass
  `api_key=` once or re-run `sulci.connect(prompt=True)` to refresh it.
- **Passive nudge.** After 100 queries on an unconnected `Cache`,
  `cache.stats()` prints a one-line suggestion to connect, once per process.
  Silence it with `SULCI_QUIET=1`.
- **Redirecting the gateway.** `SULCI_GATEWAY` redirects telemetry, the
  device-code flow and the cloud backend together. It is read when `sulci` is
  imported, so set it in the environment first. Contributors: see
  [`LOCAL_SETUP.md`](./LOCAL_SETUP.md) Step 9 — never test against production.

### Release history

Every release, with its rationale and compatibility notes, is in
[`CHANGELOG.md`](./CHANGELOG.md). Earlier editions of this README carried
per-version "additions" sections; they stopped at v0.6.1 and have been folded
into the reference above and the changelog.

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

**Context-aware benchmark:** run it rather than reading a number here, and
run it on the shipped engine — `--use-sulci` is opt-in, and without it this
measures a built-in TF-IDF engine that is in no product.

```bash
pip install -e ".[sqlite]"
python3 benchmark/run.py --use-sulci --fresh --no-sweep --context
```

The corpus is **125 follow-ups across 25 sessions**, which is small — the
harness says so on every run. A per-domain gain table used to sit here
(+56pp customer support, +17.6pp overall) and was withdrawn 2026-08-11: it
was measured on the built-in TF-IDF engine rather than the shipped MiniLM
embedder, carried no n, described the corpus as 800 pairs when
`benchmark/run.py` says 125, and had stopped reproducing even on TF-IDF.

---

## Backends

| Backend         | ID       | Best for                                | Tenant isolation |
| --------------- | -------- | --------------------------------------- | ---------------- |
| SQLite          | `sqlite` | Local dev, edge, serverless, zero infra | no |
| ChromaDB        | `chroma` | Fastest path to working, Python-native  | no |
| FAISS           | `faiss`  | GPU acceleration, massive scale         | no |
| Qdrant          | `qdrant` | Production, metadata filtering          | **yes** |
| Redis + RedisVL | `redis`  | Existing Redis infra                    | no |
| Milvus Lite     | `milvus` | Dev-to-prod without code changes        | no |
| **Sulci Cloud** | `sulci`  | **Zero infra — managed service**        | **yes** (server-side) |

All self-hosted backends are free tier or self-hostable at zero cost. For
latency on your own hardware, run the benchmark (below); a hit is dominated by
the embedding step, not the backend search.

---

## Embedding Models

| ID       | Model                  | Dims | Notes                                        |
| -------- | ---------------------- | ---- | -------------------------------------------- |
| `minilm` | all-MiniLM-L6-v2       | 384  | **Default** — free, local; the benchmarked engine |
| `mpnet`  | all-mpnet-base-v2      | 768  | Local; slower, better quality                |
| `bge`    | BAAI/bge-base-en-v1.5  | 768  | Local; slower                                |
| `openai` | text-embedding-3-small | 1536 | Requires `OPENAI_API_KEY`                    |

The local models run via `sentence-transformers`. The first load downloads the
model from Hugging Face; later loads check it for updates unless
`HF_HUB_OFFLINE=1` is set. Beyond that, no network calls are made unless you
configure `embedding_model="openai"` or use the cloud backend.

---

## Project Structure

```
.
├── .envrc                      ← direnv: activates .venv (tracked in git)
├── .github/workflows/          ← CI: tests.yml (3 OS × Python 3.9–3.12), publish.yml, benchmark.yml
├── CHANGELOG.md  CONTRIBUTING.md  LICENSE  NOTICE  README.md  SECURITY.md
├── LOCAL_SETUP.md              ← this guide
├── Makefile                    ← smoke, test-*, checkin / checkin-fast, benchmark-verify, …
├── pyproject.toml              ← name="sulci", version, extras, console scripts
├── setup.py
├── setup.sh                    ← quick start only: installs a SUBSET of the Step 3 extras
│                                 and uses plain `python3` — follow Steps 2–3 for development
├── smoke_test.py  smoke_test_async.py  smoke_test_langchain.py  smoke_test_llamaindex.py
├── benchmark/
│   ├── README.md               ← methodology and results
│   ├── baseline.json           ← pinned TF-IDF regression baseline
│   ├── run.py                  ← benchmark CLI (Step 7)
│   └── results/                ← tfidf/ (scratch) and minilm/ (committed published draws)
├── docs/
│   ├── API-SURFACE.md          ← public API, checked by scripts/check_api_surface.py
│   ├── context-threshold.md    ← why context_threshold has no default
│   ├── protocols.md  multi_tenancy_and_isolation.md  OSS_BOUNDARY_POLICY.md
│   └── architecture/           ← ADRs (e.g. 0002 smoke-fast CPU mode)
├── examples/                   ← see Step 6 and API Key Notes
│   ├── basic_usage.py  context_aware.py  context_aware_example.py
│   ├── anthropic_example.py  langchain_example.py  llamaindex_example.py  async_example.py
│   ├── agent_example_langgraph.py  agent_example_crewai.py
│   ├── mcp_example.py  litellm_example.py  proxy_example.py  gh_aw_sulci_mcp.md
│   └── extending_sulci/        ← reference implementations of the protocols
├── scripts/                    ← see scripts/README.md
│   ├── run_tests_per_file.py  run_examples.py  verify_integration_examples.py
│   ├── verify_benchmark.py  check_agent_draws.py  check_api_surface.py
│   └── check_ci_test_coverage.py  check_release_ready.py  check_tag_version.py
├── sulci/
│   ├── __init__.py             ← exports Cache, AsyncCache, ContextWindow, SessionStore, connect()
│   ├── core.py                 ← Cache engine (context-aware)
│   ├── async_cache.py          ← AsyncCache wrapper
│   ├── context.py              ← ContextWindow + in-process session handling
│   ├── config.py               ← ~/.sulci/config persistence
│   ├── telemetry.py  oss_connect.py   ← opt-in telemetry, device-code flow (Step 9)
│   ├── backends/               ← chroma, faiss, milvus, qdrant, redis, sqlite (free)
│   │                             + cloud (backend="sulci", managed); protocol.py
│   ├── embeddings/             ← minilm (default, local), openai; protocol.py
│   ├── integrations/           ← langchain, llamaindex, litellm, mcp_server
│   ├── proxy/                  ← sulci-proxy (OpenAI-compatible caching proxy)
│   ├── sessions/               ← SessionStore: memory, redis; protocol.py
│   ├── sinks/                  ← EventSink: null, telemetry, redis_stream; protocol.py
│   └── tests/compat/           ← SessionStore + EventSink conformance suites
└── tests/                      ← test_*.py per feature, plus compat/ (backend + embedder
                                  conformance) and integration/flows/
```

---

## Running Tests

The full contributor setup — Python version, the exact extras, and the offline
switches that make the suite run in about a minute — is in
[`LOCAL_SETUP.md`](./LOCAL_SETUP.md). In short, from a clone:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[sqlite,redis,qdrant,langchain,llamaindex,mcp,litellm,proxy,dev]" pytest-timeout
export HF_HUB_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True   # after the first, online run
python -m pytest tests/ -q -rs
```

At v0.9.1 that is 689 passed and 41 skipped. Skips are backends or servers you
don't have; failures or collection errors are not expected. Some missing extras
cause failures rather than skips, which is why the install line is long.

### Make targets

```bash
make smoke              # all four smoke tests (core, LangChain, LlamaIndex, AsyncCache)
make smoke-fast         # same, forcing CPU inference (recommended on Apple Silicon)
make test               # core pytest suite
make test-integrations  # LangChain + LlamaIndex integration tests
make test-async         # AsyncCache tests only
make test-all           # full suite
make test-cov           # full suite with coverage
make checkin            # pre-PR check: smoke + per-file tests + examples + benchmark verify
make checkin-fast       # same with CPU smoke mode — use this on macOS
```

---

## Examples

```bash
python examples/basic_usage.py              # stateless cache — no API key needed
python examples/context_aware.py            # context-aware — no API key needed
python examples/context_aware_example.py    # more context-aware patterns — no API key needed
python examples/anthropic_example.py        # requires ANTHROPIC_API_KEY
python examples/langchain_example.py        # OpenAI or Anthropic or mock fallback
python examples/llamaindex_example.py       # OpenAI or Anthropic or mock fallback
python examples/async_example.py            # AsyncCache demo, OpenAI/Anthropic/mock
python examples/mcp_example.py              # MCP server — no API key needed
python examples/litellm_example.py          # LiteLLM cache layer — OpenAI or mock fallback
python examples/agent_example_langgraph.py  # LangGraph agent demo — needs langgraph, langchain-anthropic
python examples/agent_example_crewai.py     # CrewAI agent demo — needs crewai
python examples/proxy_example.py            # needs a running sulci-proxy and OPENAI_API_KEY
```

The agent examples need frameworks that aren't sulci extras; installing `crewai`
downgrades `mcp` below the 2.x the `mcp` extra needs, so re-run
`pip install "mcp>=2.0.0"` afterwards (see `LOCAL_SETUP.md` Step 8.5).

---

## Benchmark

```bash
# fast run (~10 seconds)   # TF-IDF, fast; not the shipped engine, do not cite
python3 benchmark/run.py --no-sweep --queries 1000 --out /tmp/sulci-bench-q1000

# with context-aware pass  # TF-IDF, fast; not the shipped engine, do not cite
python3 benchmark/run.py --no-sweep --queries 1000 --context --out /tmp/sulci-bench-q1000

# full benchmark on the SHIPPED engine — the only form worth citing (~10 min)
pip install -e ".[sqlite]"
python3 benchmark/run.py --use-sulci --fresh --no-sweep --context
```

> **`--use-sulci` is opt-in.** Without it the benchmark runs a built-in TF-IDF
> engine that ships in no product; the shipped engine is `all-MiniLM-L6-v2`.
> The choice does not scale a number, it can invert a conclusion. Output is
> written to `benchmark/results/tfidf/` or `benchmark/results/minilm/`
> accordingly, so the two can never be confused for each other.

Give non-default runs their own `--out`: otherwise they share
`benchmark/results/tfidf/` with the regression check, which then refuses to
overwrite them.

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
python3.12 -m venv .venv          # Python 3.9–3.12; 3.13+ is untested
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

Start with [`LOCAL_SETUP.md`](./LOCAL_SETUP.md) to get a working environment, then see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for adding a backend, pre-publish review, and releasing.

---

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE).

Copyright 2026 Sulci Labs Inc.

U.S. Patent Application No. 64/018,452 (pending) covers the
context-aware semantic caching algorithm. The application is currently
assigned to Kathiravan Sengodan, not to Sulci Labs Inc. Apache 2.0
grants users a royalty-free patent license for use of this code.

## Trademarks

Sulci&trade;, the Sulci logo, and related marks are common-law (unregistered)
trademarks of Sulci Labs Inc. Apache 2.0 does not grant trademark rights
(Section 6). Nominative fair use is fine —
you may accurately say your product uses Sulci — but the name and marks may
not be used to imply endorsement, sponsorship, or affiliation without prior
written permission. See [`NOTICE`](./NOTICE).

---

## Links

- **Website:** [sulci.io](https://sulci.io?utm_source=github&utm_medium=readme&utm_campaign=oss)
- **Sign up (free key):** [sulci.io/signup](https://sulci.io/signup?utm_source=github&utm_medium=readme&utm_campaign=oss)
- **API:** [api.sulci.io](https://api.sulci.io)
- **PyPI:** [sulci](https://pypi.org/project/sulci/)
- **GitHub:** [sulci-io/sulci-oss](https://github.com/sulci-io/sulci-oss)
- **Issues:** [github.com/sulci-io/sulci-oss/issues](https://github.com/sulci-io/sulci-oss/issues)
