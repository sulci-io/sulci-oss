# Sulci Protocols Reference

> Public extension points for sulci. Introduced in v0.4.0.

This document describes the two public protocols that govern how
sulci's pluggable components are shaped: the **Backend** protocol
(for vector storage drivers) and the **Embedder** protocol (for
text-embedding models). A short section at the end covers the
**SessionStore** protocol, which 0.9.3 changed.

> Current to **sulci 0.9.3** (re-checked 2026-10-14 against `v0.9.3`).

If you want to plug a new vector store, embedding model, or storage
medium into sulci, this is the contract you implement.

---

## Why protocols?

Sulci ships seven backends (sqlite, chroma, qdrant, faiss, redis, milvus, sulci-cloud)
and two embedder classes (`MiniLMEmbedder`, which serves the `minilm`, `mpnet` and
`bge` models, and `OpenAIEmbedder`). Each implementation predates v0.4.0
and was working in production. The protocols introduced in v0.4.0 do not
change those implementations — they formalize the surface they already
expose, so that:

1. **Custom implementations** can satisfy a stable, documented contract
2. **A conformance test suite** can validate any implementation against
   the contract programmatically
3. **Multi-tenant isolation** can be guaranteed at the protocol level,
   not as an opt-in feature

Both protocols are `typing.Protocol` definitions decorated with
`@runtime_checkable`, so `isinstance(my_backend, Backend)` returns
`True` if your class has the required methods. The conformance suite
(`tests/compat/`) goes further — it checks parameter names, kinds,
defaults, and behavioral round-trip semantics.

📌 0.10.0: `delete_user` is one of the required methods, so `isinstance(b, Backend)` is `False` for a backend
written for 0.9.3 until it implements `delete_user()` (below). `Cache` does not check `isinstance`.

---

## Backend protocol

Defined in `sulci/backends/protocol.py`. Three methods every backend
must implement, plus one class-level annotation.

### `store()`

```python
def store(
    self,
    key: str,
    query: str,
    response: str,
    embedding: list[float],
    *,
    tenant_id: Optional[str] = None,
    user_id:   Optional[str] = None,
    expires:   Optional[float] = None,
    metadata:  Optional[dict] = None,
) -> None:
```

Insert or replace one cache entry.

| Parameter | Semantics |
|---|---|
| `key` | Deterministic identifier. `Cache.set` derives it from the query text alone, so it does **not** identify the scope — see the upsert requirement below. |
| `query` | Original query text. Stored for debugging only — not used in retrieval. |
| `response` | The LLM response text being cached. |
| `embedding` | L2-normalized vector representation of `query`. Length must match the embedder's `dimension`. |
| `tenant_id` | Optional partition key. `None` is stored as the literal sentinel `"global"`. See *Multi-tenancy* below. |
| `user_id` | Optional within-tenant partition key. Same `None` → `"global"` semantics as `tenant_id`. |
| `expires` | Unix timestamp after which this entry is stale. `None` means no expiry. |
| `metadata` | Arbitrary application-defined fields, stored alongside the entry. |

The kwargs after `*` are **keyword-only**. Calling code must use
`store(..., tenant_id=...)`, never positional. This is enforced at
the protocol level so callers can't accidentally swap arguments.

📌 0.9.3: `metadata` must be a mapping whose keys are not the entry's own field names
(`sulci._metadata.RESERVED_METADATA_KEYS`). `Cache.set` refuses both before calling `store`; a backend that keeps
`metadata` also calls `sulci._metadata.checked_metadata(metadata)` before it writes anything and lets the
`ValueError` / `TypeError` propagate, so a direct `store` cannot hold what `Cache.set` refuses.

**Behavioral requirements:**

- MUST upsert by `(key, tenant_id, user_id)` — calling `store` twice
  with the same key **and** the same scope replaces the existing entry
  rather than creating a duplicate. The same key under a different
  `tenant_id` or `user_id` is a different entry and MUST NOT replace
  or be merged with the first. (Before 0.9.2 this said "upsert by
  `key`".)
- MUST normalize `tenant_id=None` and `user_id=None` to `"global"`
  before storage, so that subsequent searches with `None` find them.
- SHOULD log, not raise, on transient backend errors. A failed write
  should never crash the calling application.

### `search()`

```python
def search(
    self,
    embedding: list[float],
    threshold: float,
    *,
    tenant_id: Optional[str] = None,
    user_id:   Optional[str] = None,
    now:       Optional[float] = None,
) -> tuple[Optional[str], float]:
```

Approximate nearest-neighbor search by cosine similarity, restricted
by tenant and user partition.

Returns `(response, similarity)` if a hit at or above `threshold`,
otherwise `(None, 0.0)`.

| Parameter | Semantics |
|---|---|
| `embedding` | The L2-normalized query vector. |
| `threshold` | Minimum cosine similarity (0.0–1.0) to count as a hit. |
| `tenant_id` | Optional partition filter. `None` matches the `"global"` partition only — never other named tenants. |
| `user_id` | Optional within-tenant filter. Same partition semantics as `tenant_id`. |
| `now` | Unix timestamp for TTL comparison. Defaults to `time.time()`. Provided for test determinism. |

**Behavioral requirements:**

- MUST honor tenant isolation as a hard boundary. Entries from a
  different `tenant_id` MUST NOT be returned, even when their
  similarity exceeds `threshold`. This is the protocol's strongest
  guarantee. See `tests/test_qdrant_tenant_isolation.py` for the
  scenarios that pin this behavior.
- MUST honor `user_id` partitioning the same way, layered within
  tenant. Different-tenant + same-user_id MUST miss.
- MUST treat `None` and `"global"` symmetrically on the read path.
  This pairs with `store()`'s sentinel normalization. Any asymmetry
  here is a cross-tenant data leak (caught and fixed in v0.4.0
  phase 1.4).
- MUST NOT raise on connectivity errors. Return `(None, 0.0)` and
  log instead. A broken cache must degrade to "no cache" silently —
  the user's application must never crash because of cache trouble.
- MUST exclude expired entries (entries with `expires` set and
  `expires < now`).

### `clear()`

```python
def clear(self) -> None:
```

Remove all entries. Destructive, idempotent.

Typically used in tests; production should prefer TTL-based expiry.
Per-user or per-tenant deletion of entries is *not* part of the
`Backend` protocol (0.9.3). The managed backend has its own
`delete_user()`; for a self-hosted backend it is the implementer's
own extension.

For backends like Qdrant where `clear` could either delete the
underlying schema or just the data: the protocol semantics are
"empty the cache" — implementations should preserve the underlying
storage structure (collections, indexes, etc.) so subsequent
operations work without rebuilding.

### `delete_user()` — 📌 0.10.0

```python
def delete_user(self, user_id: str, *, tenant_id: Optional[str] = None) -> int:
```

Added to the protocol in 0.10.0 (ADR 0005 Amendment 2); the 0.9.3 paragraph under `clear()` above is the record of
what came before. Remove every entry filed under (`tenant_id or "global"`, `user_id`), expired ones too, and
nothing else; return how many were removed. `Cache.delete_user` is the only caller: it validates the ids, refuses a
cache that is not personalized, erases the user's sessions from the session store, and wraps a failure here in
`sulci.errors.SulciErasureError`.

- MUST leave the same user in another tenant, other users of the tenant, and the shared user `"global"` intact.
- MUST return a true count. A client that does not report one: count the matches, then delete those ids.
- MUST be idempotent (a second call returns 0) and MUST NOT validate the ids: file a missing tenant or user as
  `"global"`, as `store` does.
- MUST raise on failure — unlike `search` — and never remove part of the user's entries and report success.
- MAY cost time linear in the number of entries; it is not on the lookup path.

A backend without `delete_user` still works with `Cache`, but `Cache.delete_user` raises `NotImplementedError` on it.

### `ENFORCES_TENANT_ISOLATION` class attribute

```python
class MyBackend:
    ENFORCES_TENANT_ISOLATION: bool = True
```

A class-level boolean declaring whether this backend honors the
tenant isolation contract on the read path.

- `True` — `search()` filters out entries from other tenants and users.
  The conformance suite runs `TestTenantIsolation` against this backend.
- `False` — the backend does not filter on `tenant_id`. The conformance
  suite still accepts it and skips the isolation tests, but no shipped
  local backend uses it any more; a custom backend is expected to
  isolate and declare `True`.

The shipped backends declare (0.9.3):

| Backend | `ENFORCES_TENANT_ISOLATION` |
|---|---|
| QdrantBackend, ChromaBackend, SQLiteBackend, RedisBackend, FAISSBackend, MilvusBackend | `True` — `search()` filters on `tenant_id` AND `user_id`, `None` meaning the shared scope `"global"`, inside the search |
| SulciCloudBackend | `False` — enforced server-side by the gateway; the OSS conformance suite can't reach a gateway to verify it |

Before 0.9.3 only `QdrantBackend` declared `True`; the other five accepted
`tenant_id` and ignored it.

Custom backends must declare this attribute. Conformance fails if it's missing.

---

## Embedder protocol

Defined in `sulci/embeddings/protocol.py`. Three members.

### `dimension` property

```python
@property
def dimension(self) -> int:
```

Vector dimensionality. MUST be:

- A positive integer
- Stable for the lifetime of the embedder instance
- Exposed as a `@property` (not a plain attribute, not a method)

The `Cache` uses this at construction time to size backend collections.

### `embed()`

```python
def embed(self, text: str) -> list[float]:
```

Embed a single text string. Returns a list of `dimension` floats,
**L2-normalized** (i.e. `||v|| ≈ 1.0`).

The L2 normalization requirement is critical. Sulci backends use
cosine similarity assuming unit-length vectors. An unnormalized
embedder produces incorrect similarity scores and silently breaks
cache-hit semantics.

### `embed_batch()`

```python
def embed_batch(self, texts: list[str]) -> list[list[float]]:
```

Embed a list of texts. Returns a list of `dimension`-length vectors
in the same order as the inputs, each L2-normalized.

This is the hot path for benchmark workloads and bulk imports.
Implementations should batch internally where the underlying model
supports it (e.g. sentence-transformers' `batch_size` argument).

---

## Verifying conformance

Sulci ships a public conformance test suite at `tests/compat/`. It
parametrizes across every class in two registries:
`BACKEND_CLASSES` and `EMBEDDER_CLASSES`. Both are defined in
`tests/compat/conftest.py`.

### Adding your class to the registry

```python
# tests/compat/conftest.py

from my_package.backends import MyCustomBackend

BACKEND_CLASSES = [
    cls for cls in [
        # ... existing backends ...
        MyCustomBackend,
    ]
    if cls is not None
]
```

Add a corresponding clause to `_try_construct_backend` so the suite
knows how to build a live instance for behavioral tests:

```python
if name == "MyCustomBackend":
    try:
        return cls(my_required_arg="...")
    except Exception:
        return None  # behavioral tests will skip; structural still run
```

### What the suite checks

Three test groups, each with different infrastructure requirements:

| Group | Checks | Runs when |
|---|---|---|
| `TestStructural` | `isinstance(b, Backend)`, parameter names + kinds + defaults via `inspect.signature`, presence of `ENFORCES_TENANT_ISOLATION` | Always |
| `TestRoundTrip` | `store()` + `search()` returns the stored response; `clear()` empties the backend | When `_try_construct_backend` returns a live instance |
| `TestTenantIsolation` | Cross-tenant search misses; same-tenant search hits | When `ENFORCES_TENANT_ISOLATION = True` AND backend is constructable |
| `TestUserErasure` — 📌 0.10.0 | `delete_user` signature; the count includes expired entries; nothing left for the user; the same user in another tenant, other users and the shared user intact; a second call returns 0 | When the class has `delete_user` AND the backend is constructable |

### Run

```bash
pytest tests/compat/ -v
```

Structural tests are fast (sub-second per backend) and will catch
signature mismatches immediately. Behavioral tests are slower and
require infrastructure for backends like Qdrant, Redis, Milvus.

---

## Where this fits in the architecture

The Backend and Embedder protocols are the OSS extension surface.
They define the contract between Sulci's `Cache` class and any
storage/embedding driver — that's their full scope. Two important
boundaries to be aware of:

**Calling code is responsible for `tenant_id` and `user_id`.** The
protocol exposes both as kwargs on `store()` and `search()`, and
makes no assumption about where they come from. A solo developer
using sulci as a personal cache passes neither. A multi-tenant SaaS
embedding sulci passes both, derived from whatever identity layer
the application uses.

**Sulci Cloud (the managed product) sits one layer above.** If
you're using `Cache(backend="sulci")`, the gateway derives
`tenant_id` from your API key before the request reaches a backend
driver. End-user application code in that case never touches
`tenant_id` directly. This is platform behavior, not protocol
behavior — see [Multi-Tenancy and Data Isolation](multi_tenancy_and_isolation.md)
for the OSS-layer trust model.

---

## Stability and versioning

Both protocols are declared **STABLE API** in their respective
source files (`sulci/backends/protocol.py`,
`sulci/embeddings/protocol.py`). This means:

- Removing a method or kwarg is a **breaking change**. Custom
  implementations would no longer satisfy the contract.
- Adding a new optional kwarg is **backwards-compatible** if the
  default preserves existing behavior. Existing implementations
  satisfying the old contract still satisfy the new one.
- Changing the *semantics* of an existing method (e.g. what
  `tenant_id=None` means on the read path) is a breaking change
  even if the signature is unchanged. v0.4.0 phase 1.4 was this kind
  of change, and it landed in a minor release because there were
  no shipped consumers of v0.4.0's behavior yet.

Per ADR 0005 (kept in sulci-platform:
`docs/architecture/adrs/0005-multi-backend-commitment.md`), protocol
modifications require a superseding ADR before merge. Its Amendment 1
(2026-10-11) records that 0.9.3 changed the Backend and SessionStore
protocols without a deprecation cycle, deliberately, and that the
deprecation rule resumes at the first official release.

📌 0.10.0: the addition of `Backend.delete_user` (see `delete_user()` above) is to be recorded as Amendment 2,
written in sulci-platform before 0.10.0 is released.

---

## SessionStore protocol (context-aware caching)

Defined in `sulci/sessions/protocol.py`; shipped as `InMemorySessionStore`
and `RedisSessionStore`. Methods: `get(session_id)`, `append(session_id,
vector, max_turns=8)`, `clear(session_id)`, `summary(session_id=None)`, and —
**required since 0.9.3** — `clear_scope(scope) -> int` and `clear_all() -> int`.

- `Cache` hands a store **scoped** session ids
  (`sulci.sessions.scoped_session_id(tenant_id, user_id, session_id)`), so a
  session belongs to one tenant and, when personalized, one user.
  `clear_scope` removes every session whose id starts with a tenant's or a
  user's scope prefix (`sulci.sessions.scope.tenant_scope` / `user_scope`), and
  must not scan another tenant's keys to do it when the store is shared.
- `Cache.clear()` calls `clear_all()`. `Cache` refuses, at construction, an
  injected store that lacks either method.
- A `SessionStore` holds embedding vectors of the user's turns, not text. (The built-in
  in-process manager `Cache` uses when no store is injected also keeps the turns' text.)
- `RedisSessionStore` (0.9.3) defaults to the prefix `sulci:sessions:`, needs
  Redis 6.2+, and does not support Redis Cluster.

---

## See also

- **`examples/extending_sulci/custom_backend.py`** — A worked
  reference implementation. ~150 lines, in-memory dict-based,
  satisfies the protocol fully. Run it directly to see a self-test
  pass.
- **`docs/multi_tenancy_and_isolation.md`** — The customer-facing
  scenarios and trust model that motivate the protocol's tenant
  isolation requirements.
- **`tests/compat/`** — The conformance test suite itself.
- **`tests/test_qdrant_tenant_isolation.py`** — End-to-end isolation
  scenarios, framed as customer-support product cases.
