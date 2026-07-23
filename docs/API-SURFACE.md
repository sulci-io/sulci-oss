# Sulci — the measured public API surface

**Measured:** 2026-07-22 against `sulci` **0.8.2**
**Method:** AST parse of `sulci/core.py` and `sulci/async_cache.py`. Not memory, not the README.

---

## Why this file exists

Six documents in three repos have carried a wrong description of this API at
some point, and the wrongness survived for months each time. The specific
failures were: four constructor defaults stated wrongly (two of them
behavioural), a `metadata` kwarg attributed to a method that has never had it,
and a method attributed to `Cache` that has never been on `Cache` — that last one
reached a published privacy policy and a GDPR claim.

The defect is always the same shape: **a document restating a fact it does not
own, correct when written, with no mechanism to notice when the code moved.**

The cheap structural defence is to record the command that measures the claim, so
a reader can re-run it in ten seconds instead of trusting a sentence. That is
what this file is. **When you restate anything here in another document, link
here instead — or if you must restate it, paste the command alongside.**

---

## Regenerate this file

```bash
cd ~/code/sulci-oss
python3 - <<'PY'
import ast
for f in ("sulci/core.py", "sulci/async_cache.py"):
    tree = ast.parse(open(f).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            print(f"\n### {f} :: class {node.name}")
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if m.name.startswith("_") and m.name != "__init__":
                        continue
                    a = m.args
                    print(f"  {m.name}("
                          f"pos={[x.arg for x in a.args]}, "
                          f"kwonly={[x.arg for x in a.kwonlyargs]})")
PY
grep -n '^version' pyproject.toml
```

If the output disagrees with anything below, **the output wins** — update this
file and date it, then grep the estate for whatever else restated the old value.

---

## `Cache.__init__` — the real defaults

Parameter order below is the real declaration order.

```python
from sulci import Cache

cache = Cache(
    backend         = "chroma",    # chroma|sqlite|faiss|qdrant|redis|milvus|sulci
    threshold       = 0.85,
    embedding_model = "minilm",    # minilm|mpnet|bge|openai
    ttl_seconds     = 86400,       # 24h — entries DO expire by default
    personalized    = False,
    db_path         = "./sulci_db",
    context_window  = 0,           # 0 = stateless
    query_weight    = 0.7,
    context_decay   = 0.5,
    session_ttl     = 3600,
    telemetry       = True,
    api_key         = None,        # also SULCI_API_KEY
    gateway_url     = "",          # empty; resolved downstream, not defaulted here
    session_store   = None,
    event_sink      = None,
    cost_per_call   = 0.005,
)
```

`backend` and `embedding_model` accept either a string **or** a pre-constructed
instance (the v0.4.0 protocols).

### The four that were documented wrongly, and are easy to get wrong again

| Kwarg | Commonly stated as | Actually | Why it matters |
|---|---|---|---|
| `backend` | `"sqlite"` | **`"chroma"`** | A reader following the docs installs the wrong extra |
| `ttl_seconds` | `None` | **`86400`** | **Behavioural.** Entries expire after 24h by default; "never expires" is wrong |
| `db_path` | `"./sulci"` | **`"./sulci_db"`** | Points at a directory that will not exist |
| `gateway_url` | `"https://api.sulci.io"` | **`""`** | The URL is resolved downstream (see below), not defaulted here |

`sulci-web`'s `src/pages/two/docs/Configuration2.jsx:8-27` has carried the correct
values — with the measurement and a "do not fix `ttl_seconds` back to `None`"
warning — since months before the README was corrected in PR #114. That file is
the model to copy.

### Gateway URL resolution (v0.7.4)

Three tiers, first match wins:

1. explicit `gateway_url=` kwarg
2. `SULCI_GATEWAY` environment variable
3. `https://api.sulci.io`

The env var is read **at instance construction**, not at import. Empty or
whitespace values fall through to the next tier.

---

## `Cache` — exactly eight public methods

There are eight. Not nine.

| Method | Keyword-only kwargs | Returns |
|---|---|---|
| `get(query, *, …)` | `threshold`, `tenant_id`, `user_id`, `session_id`, `plan` | `(response \| None, similarity, context_depth)` |
| `set(query, response, *, …)` | `tenant_id`, `user_id`, `session_id`, `metadata`, `plan` | `None` |
| `cached_call(query, llm_fn, *, …)` | `threshold`, `tenant_id`, `user_id`, `session_id`, `cost_per_call`, `plan` | `{response, source, similarity, latency_ms, cache_hit, context_depth}` |
| `get_context(session_id)` | — | `ContextWindow` |
| `clear_context(session_id)` | — | `None` |
| `context_summary(session_id=None)` | — | `dict` |
| `stats()` | — | `dict` |
| `clear()` | — | `None` |

Everything after `query` on `get` / `set` / `cached_call` is **keyword-only**.
`SyncCache` is an alias for `Cache`. `tenant_id` is a hard partition boundary
(v0.4.0).

### `metadata` is on `set` only

`metadata` exists on `set` and `aset`. **It is not on `cached_call`** and never
has been, despite having been documented there. It is also not on `get`.

### `delete_user` is not a `Cache` method

This one has caused real damage, so it gets its own section.

```python
cache.delete_user(user_id)           # AttributeError
cache.backend.delete_user(user_id)   # AttributeError — `.backend` is the *string*
cache._backend.delete_user(user_id)  # works — cloud backend only, private attribute
```

- `delete_user` is defined in exactly one place: `sulci/backends/cloud.py:233`.
- `class Cache:` (`core.py:140`) has no base class and inherits nothing.
- There is no `__getattr__` or delegation anywhere in `core.py`.
- It is not re-exported from `sulci/__init__.py`.
- `self.backend` (`core.py:218`) is the **string**; the instance is
  `self._backend` (`core.py:235`), private.

**It is also a capability gap, not just a naming one.** `delete_user` exists on
one of seven backends and is **not in the `Backend` protocol** — `protocol.py`
declares `clear`, not `delete_user`. So even a naive `Cache.delete_user()` proxy
would `AttributeError` on the six local backends, while `personalized=True` is
documented as partitioning per `user_id`. Any fix needs either a `hasattr` guard
or a protocol addition, and that is a design decision, not a one-line patch.

**Why it survived so long:** `cache.clear()` *does* work — `core.py:960` proxies
to `self._backend.clear()` — and the two are documented as a pair. Testing the
pair by checking `clear()` shows green. `delete_user` never got the proxy.

**Recorded for the irony:** CHANGELOG `[0.6.2]` is titled *"GDPR-adjacent fix:
`cache.clear()` and `cache.delete_user()` now actually delete"* and calls the old
behaviour *"a success-shaped no-op for a GDPR-relevant operation."* That fix made
the backend method really delete. Nobody noticed the documented call site was
never reachable. The no-op became an exception.

**Correct as written, leave alone:** `README.md:648` and `:955` attribute the
method to `SulciCloudBackend`; both CHANGELOGs describe it as a backend method in
past-tense release notes. Those are right.

**Open decision.** Whether to add a guarded `Cache.delete_user()` proxy, add
`delete_user` to the `Backend` protocol, or keep per-user deletion an
HTTP-and-cloud-backend-only capability and say so everywhere. Until that is
decided, **documentation must describe the capability that exists**, which is:
`cache.clear()` on any backend, `DELETE /v1/cache/user/{id}` on the hosted tier.

---

## `AsyncCache` — a complete mirror of *which* kwargs, not of *how*

As of 0.8.2 the forwardable-kwarg surface is a complete mirror:

| Method | Keyword-only |
|---|---|
| `aget` | `tenant_id`, `plan` |
| `aset` | `tenant_id`, `plan`, `metadata` |
| `acached_call` | `tenant_id`, `plan` |
| `get` (sync passthrough) | `threshold`, `tenant_id`, `plan` |
| `set` (sync passthrough) | `tenant_id`, `plan`, `metadata` |
| `cached_call` (sync passthrough) | `threshold`, `tenant_id`, `plan` |

**The mirror is of which kwargs are forwarded, not of how they are passed.** On
`aget` and `acached_call`, `threshold` — along with `user_id`, `session_id`,
`cost_per_call` — is **positional-or-keyword**, whereas `Cache.get` makes
everything after `query` keyword-only. So this is legal:

```python
await cache.aget(q, uid, sid, 0.9)      # positional threshold — fine
cache.get(q, uid, sid, 0.9)             # TypeError — Cache.get is keyword-only
```

Not a bug. But the changelog's "100% mirror" language is about *forwarding*, and
a reader taking it as full signature parity would be wrong. Measured:

```
aget(pos=['self','query','user_id','session_id','threshold'], kwonly=['tenant_id','plan'])
get (pos=['self','query'],  kwonly=['threshold','tenant_id','user_id','session_id','plan'])
```

`set` deliberately did **not** grow a `threshold` in 0.8.2, because `Cache.set`
has none. The mirror is faithful, not a superset.

---

## Backends

| Backend | ID | Native vectors | Tenant isolation |
|---|---|---|---|
| ChromaDB | `chroma` | ✓ | label only |
| Qdrant | `qdrant` | ✓ | **enforced** — payload `Filter`, `ENFORCES_TENANT_ISOLATION = True` |
| Redis + RedisVL | `redis` | manual | label only |
| FAISS | `faiss` | ✓ | label only |
| SQLite | `sqlite` | manual | label only |
| Milvus Lite | `milvus` | ✓ | label only |
| Sulci Cloud | `sulci` | ✓ | **enforced server-side** |

Six free backends plus the managed one. `httpx>=0.27` has been a mandatory
dependency since v0.6.3, so the `sulci[cloud]` extra is a back-compat no-op.

Only `qdrant` and `sulci` enforce tenant isolation. The other five accept
`tenant_id` and use it as a label — which is a real difference and should not be
flattened into "all backends support multi-tenancy" in any marketing surface.

---

## Embedding models

| ID | Model | Dims |
|---|---|---|
| `minilm` | all-MiniLM-L6-v2 | 384 |
| `mpnet` | all-mpnet-base-v2 | 768 |
| `bge` | BAAI/bge-base-en-v1.5 | 768 |
| `openai` | text-embedding-3-small | 1536 |

Four, not two. `embedding_model` was documented as `minilm | openai` until
PR #114.

---

*Measured 2026-07-22 at `sulci` 0.8.2. Re-run the command at the top before
trusting any line of this.*
