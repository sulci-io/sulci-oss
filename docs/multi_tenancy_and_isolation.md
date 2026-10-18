# Multi-Tenancy and Data Isolation in Sulci

> How sulci enforces partition boundaries between tenants and users,
> what guarantees the OSS library makes, and how to wire it into your
> application correctly.

This document covers the *why* and *how* of sulci's tenant isolation
model. For the strict protocol reference (method signatures, the
conformance test contract, kwarg semantics), see
[docs/protocols.md](protocols.md). For the customer-support test
scenarios that pin behavior at the implementation level, see
[tests/test_qdrant_tenant_isolation.py](../tests/test_qdrant_tenant_isolation.py).

---

## Why this matters

A semantic cache sits between an LLM application and the LLM provider.
When a query comes in, the cache asks: *"have I already seen something
semantically similar?"* If yes, return the cached response and skip the
LLM call. This saves cost and latency.

The semantic part is also the dangerous part.

Consider a SaaS support chatbot company — call it HelpDesk AI — that
sells the same chatbot to multiple customers. Acme Corp uses it for
warranty questions. Globex Health uses it for patient billing
(HIPAA-regulated). Initech Software uses it for technical support.

If HelpDesk AI runs all three customers' caches against the same
embedding space *without partition isolation*, then a Globex agent's
question about a patient's billing history could be returned to an
Initech caller asking a semantically-similar question. Globex's
patient PHI ends up in Initech's response. That's a HIPAA breach,
a lawsuit, and probably the end of HelpDesk AI as a company.

The fix is partition isolation: every cached entry is tagged with
the tenant it belongs to, and searches are restricted to entries
matching the caller's tenant. Sulci's `Backend` protocol takes this
as a hard requirement — entries from other tenants MUST NOT be
returned, even when their similarity exceeds threshold. Threshold
is a quality knob; tenant filtering is a security boundary.

This document explains how that boundary works in the OSS library,
what its limits are, and how to wire it into your application.

---

## The two axes of isolation

There are two independent ways one customer's data can be kept apart
from another's. They are often conflated in conversations about
"multi-tenancy" but they're conceptually distinct:

### Axis 1 — Logical isolation (within a single deployment)

How does *one running cache cluster* keep tenant A's data away from
tenant B's data? This is what `tenant_id` in the protocol enforces.

It's a property of the code: every `store()` and `search()` call
filters by tenant, and the protocol's behavioral requirements pin
that filtering as mandatory. Logical isolation runs on every
request, lives forever, doesn't depend on deployment topology.

### Axis 2 — Physical isolation (across deployments)

Does tenant A's data physically sit on the same hardware/network/
cluster as tenant B's? Or are they in entirely separate processes,
machines, or networks?

This is a property of the *infrastructure*, decided at procurement
time. It's outside the scope of the OSS library — the OSS library
runs wherever you put it. Some customers are happy with shared
infrastructure where logical isolation is the only boundary. Others
require dedicated infrastructure for regulatory reasons (HIPAA,
FedRAMP, SOC 2). The Sulci Cloud product offers both shared and
dedicated options; see [sulci.io/enterprise](https://sulci.io/enterprise)
for the deployment options Sulci offers commercially.

This document focuses on Axis 1 — the logical isolation that the
OSS library enforces in code. Axis 2 is a deployment decision the
operator makes, not a protocol concern.

---

## The OSS trust model

Sulci OSS is a Python library that runs *inside* the user's
application. Its trust assumptions:

- **The calling code is trusted.** The application embedding sulci
  is responsible for authenticating its users and deciding what
  `tenant_id`/`user_id` to pass on each request. If the application
  passes the wrong tenant_id, sulci will dutifully return entries
  for the wrong tenant. The library cannot validate a tenant_id
  against an external authentication source — that's not its job.
- **The backend is trusted.** Whichever vector store sulci is
  configured to use (Qdrant, Chroma, etc.) is in the trust path.
  A compromised backend can return any data it wants. Sulci's
  filter logic is a defense against accidental leakage from
  application bugs, not a defense against a malicious backend.
- **The protocol's filter is the last line of defense.** Above sulci,
  there might be a gateway, a load balancer, a service mesh, an
  auth layer — those are *your* concerns, not the library's. Below
  sulci, the backend's data files. The Backend protocol's `tenant_id`
  filter is the boundary the library is responsible for. If that
  filter has a bug, no upstream care can fix it; if it works, no
  downstream chaos can bypass it from within the same process.

This is what makes the conformance suite (`tests/compat/`) and the
customer-scenario tests (`tests/test_qdrant_tenant_isolation.py`)
matter so much. They verify the library's one job is done correctly.

---

## What's enforced where

The OSS library's filter logic is implemented in each Backend's
`search()` method. The protocol prescribes the contract; each
implementation declares that it honors it with the
`ENFORCES_TENANT_ISOLATION` class attribute.

**Since 0.9.3 every shipped backend enforces both `tenant_id` and
`user_id`** on every lookup, with the scope inside the search so other
scopes' copies of a question cannot crowd the caller's own out. `None`
means the shared scope (stored as `"global"`) and nothing else.

| Backend | `ENFORCES_TENANT_ISOLATION` | How (0.9.3) |
|---|---|---|
| QdrantBackend | `True` | Qdrant payload `Filter(must=[...])` on `tenant_id` and `user_id` at query time |
| ChromaBackend | `True` | `where` filter on `tenant_id` and `user_id` in the query; entries stored under a scoped id (`sulci.backends._scope.scoped_id`) |
| SQLiteBackend | `True` | `WHERE tenant_id = ? AND user_id = ?`; a row is unique on `(key, tenant_id, user_id)` since 0.9.2 |
| RedisBackend | `True` | each entry stores `tenant_id` and `user_id` under a scoped key; a lookup reads only matching entries |
| FAISSBackend | `True` | a per-scope position list; a lookup searches only its own scope's vectors, exactly |
| MilvusBackend | `True` | filter expression on `tenant_id` and `user_id`, every value escaped |
| SulciCloudBackend | `False` — enforced **gateway-side** | the managed gateway scopes by the API key's tenant and the request's `user_id`; the flag reads `False` only because the OSS conformance suite cannot reach a gateway to verify it |

Before 0.9.3 only Qdrant (and the managed gateway) enforced
`tenant_id`; the other five accepted it and ignored it, and a lookup
without `user_id` read every user's entries. Entries written before
0.9.3 on Redis, FAISS and Milvus carry no tenant and match no scope;
`clear()` those stores after upgrading (see the 0.9.3 *Security*
entries in the [CHANGELOG](../CHANGELOG.md)).

> Unreleased on `main` (not on PyPI; CHANGELOG `[Unreleased]`): Milvus
> files each entry under a digest-based scope key
> (`sulci.sessions.scope.user_scope`) and filters on that key alone,
> instead of on the ids.

The conformance test suite (`tests/compat/`) runs the
`TestTenantIsolation` group against every backend declaring
`ENFORCES_TENANT_ISOLATION = True` that it can construct locally. A
custom backend is expected to declare `True` and pass that group.

### Ids the library refuses (0.9.3)

`Cache.get` / `Cache.set` (and so `cached_call`, `AsyncCache` and the
integrations) raise `ValueError` before anything is read, written or
sent for a `tenant_id` — or, with `personalized=True`, a `user_id` —
that is `"global"`, any falsy value other than `None` (`""`, `0`,
`False`, `b""`), or text that is not valid Unicode. `"global"` is the
shared scope's sentinel; a caller passing it would read and write the
shared scope as its own. Pass `None` for "no tenant" / "no user".

### Conversation history is scoped too (0.9.3)

With `context_window > 0`, the `Cache` hands its session store a
`session_id` scoped by tenant and, when personalized, by user
(`sulci.sessions.scoped_session_id`), so the same `session_id` from
another tenant or user is another conversation. `get_context`,
`clear_context` and `context_summary` take `tenant_id=` / `user_id=`
to name a session; `Cache.clear()` removes every session from any
store.

---

## Customer scenarios

Concrete walkthroughs of how the model behaves in real situations.
These are also the names of test methods in
`tests/test_qdrant_tenant_isolation.py` — when one of those tests
fails, the failure tells you which user-impacting scenario is broken.

### Scenario 1 — Acme agents share each other's cache (the value prop)

> **Acme agent #1:** "What's the warranty on the Acme P500 pump?"

The cache misses, the LLM is called, the response gets stored:

```python
from sulci import Cache

cache = Cache(backend="qdrant")          # personalized=False (the default): user_id is ignored

cache.set(
    query     = "What's the warranty on the Acme P500 pump?",
    response  = "The Acme P500 has a 5-year limited warranty.",
    tenant_id = "acme_corp",
)
```

> **Acme agent #2 (different agent, same company):** "How long is
> the P500 covered under warranty?"

The cache **hits** — same tenant, semantically-similar query. The
response is returned without calling the LLM. This is sulci's
core value: cache hits within a tenant save cost and latency.

```python
cache.get(
    query     = "How long is the P500 covered under warranty?",
    tenant_id = "acme_corp",
)
# → ("The Acme P500 has a 5-year limited warranty.", <similarity>, 0)
```

(Agents of one company share answers here because the `Cache` is not
personalized. With `personalized=True` and a `user_id` on each call,
agent #2 would miss — Scenario 3.)

**What could go wrong:** if the application forgets to pass
`tenant_id`, all entries land in the `"global"` partition. That works
fine for single-tenant apps but breaks the cross-tenant isolation
guarantee the moment the app starts handling multiple tenants. Always
pass `tenant_id` in multi-tenant deployments.

### Scenario 2 — Globex PHI must not leak to Initech

> **Globex agent:** "What's the bill code for John Smith's surgery
> on March 15?"

```python
cache.set(
    query     = "Bill code for John Smith's surgery on March 15?",
    response  = "Patient John Smith MRN 4471823, code 49505. Balance $1,247.83.",
    tenant_id = "globex_health",
)
```

> **Initech caller (different company, semantically-similar query):**
> "What's the policy for billing the patient surgery on March 15?"

The two queries are semantically close. Without
tenant isolation, this would match — and Initech would receive PHI
for a Globex patient.

With tenant isolation:

```python
resp, sim, depth = cache.get(
    query     = "What's the policy for billing the patient surgery on March 15?",
    tenant_id = "initech_software",
)
# → (None, 0.0, 0)   ← clean miss
```

The Globex entry is filtered out before similarity is even
considered. The result is a clean miss; Initech's LLM is called
fresh with their (very different) actual context.

**What could go wrong:** an asymmetry between the store path's
sentinel handling and the search path's filter logic. If `store()`
writes `tenant_id="global"` for `None` but `search()` with
`tenant_id=None` doesn't filter, the unscoped read would silently
return entries from named tenants. This was a real bug in v0.4.0
phase 1.4, caught by
`test_named_tenant_entry_does_not_match_global_search`. The fix is
documented in the [protocols reference](protocols.md#search).

### Scenario 3 — Within-tenant user partitioning

Same company, two agents with different access permissions.

> **Acme enterprise-sales agent (agent_001):** "What pricing tier
> does Northrop Industries get?"

```python
from sulci import Cache

# Its own db_path: a local Qdrant store can be open in only one client per process.
cache = Cache(backend="qdrant", personalized=True, db_path="./sulci_db_agents")   # user_id now scopes lookups

cache.set(
    query     = "What pricing tier does Northrop Industries get?",
    response  = "Northrop is Tier-3 enterprise: 22% discount, net-60 terms.",
    tenant_id = "acme_corp",
    user_id   = "agent_001_enterprise",
)
```

> **Acme residential-sales agent (agent_002):** "What pricing does
> Northrop get?"

```python
resp, sim, depth = cache.get(
    query     = "What pricing does Northrop get?",
    tenant_id = "acme_corp",
    user_id   = "agent_002_residential",
)
# → (None, 0.0, 0)
```

agent_002 doesn't have enterprise-customer access in their CRM.
They miss the cache and the LLM is called fresh, producing an
answer based on agent_002's own permissions (likely "I don't have
access to Northrop's pricing details"). The privileged response
that agent_001 saw never leaks to agent_002.

**Whether to set `user_id`** is a deployment decision. Some
applications have role-based access controls where different users
must see different responses to the same query — those should build
the `Cache` with `personalized=True` and pass `user_id` on both `set`
and `get`. Others have shared knowledge bases where all users in a
tenant should hit the same cache — those leave `personalized=False`
(the default), and `Cache` then ignores `user_id`. With
`personalized=True`, a lookup without a `user_id` sees only entries
stored without one.

### Scenario 4 — Solo-developer single-tenant deployment

> **Solo dev:** "How do I deploy a Python app to AWS Lambda?"

```python
cache.set(
    query    = "How do I deploy a Python app to AWS Lambda?",
    response = "Package code as a .zip, upload via aws lambda create-function.",
    # No tenant_id, no user_id
)
```

> **Solo dev later:** "What's the process for getting a Python app
> onto Lambda?"

```python
resp, sim, depth = cache.get(
    query = "What's the process for getting a Python app onto Lambda?",
    # No tenant_id, no user_id
)
# → ("Package code as a .zip, upload via aws lambda create-function.", <similarity>, 0)
```

This is the simplest deployment. No tenants, no users. Everything
goes into the `"global"` partition (sulci's internal sentinel for
"un-scoped"). The cache works exactly as it did before v0.4.0
introduced tenant_id; backwards-compatible by design.

**Most OSS users start here.** The library is approachable because
multi-tenancy is opt-in, not required.

---

## For implementers: wiring tenant_id and user_id from your application

The library doesn't dictate where these values come from. They're
yours to set on each call. Here are common patterns.

### From a JWT / session token

```python
from sulci import Cache
import jwt

cache = Cache(backend="qdrant", personalized=True)

def handle_chat_request(request, http_headers):
    token  = http_headers.get("Authorization", "").removeprefix("Bearer ")
    claims = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    # Derive partition keys from authenticated identity.
    tenant_id = claims["org_id"]      # the customer's organization
    user_id   = claims["sub"]         # the individual end-user

    response, sim, _ = cache.get(
        query     = request.query,
        tenant_id = tenant_id,
        user_id   = user_id,
    )
    if response is None:
        response = call_llm(request.query, system_prompt_for(tenant_id))
        cache.set(
            query     = request.query,
            response  = response,
            tenant_id = tenant_id,
            user_id   = user_id,
        )
    return response
```

The principle: **`tenant_id` and `user_id` come from authenticated
identity, never from request body.** Letting a caller pass their own
`tenant_id` in a request payload would let any caller spoof another
tenant's cache. Always derive from a token your app issued.

### From an API key map

```python
TENANT_FROM_API_KEY = {
    "sk_acme_xxxxxxxxxxxx":   "acme_corp",
    "sk_globex_yyyyyyyyyy":   "globex_health",
    "sk_initech_zzzzzzzzz":   "initech_software",
}

def handle_request(request, headers):
    api_key   = headers["Authorization"].removeprefix("Bearer ")
    tenant_id = TENANT_FROM_API_KEY.get(api_key)
    if tenant_id is None:
        raise Unauthorized()
    user_id = headers.get("X-Customer-User-Id") or None   # optional; map "" to None (0.9.3 refuses "")

    response, sim, _ = cache.get(
        query     = request.query,
        tenant_id = tenant_id,
        user_id   = user_id,
    )
```

The API key is the source of truth for `tenant_id`. The customer's
own user identifier (which they pass through in a header) becomes
`user_id`, but only as a partition label — sulci doesn't validate
those values, that's the customer's concern within their own tenant.

### From an LLM framework integration (LangChain, LlamaIndex)

The integrations scope by tenant in their own ways: `sulci-mcp
--tenant-id` (or `SULCI_MCP_TENANT_ID`), the proxy's
`x-sulci-tenant-id` header (combined with the model unless
`--share-across-models`), and LiteLLM's `namespace_by_model=True`. Since
0.9.3 each of these scopes for real on every backend. Consult each
integration's docstring for its exact form.

---

## FAQ

### Does sulci hash tenant_id before storing it?

Not in the cache entry. The `tenant_id` and `user_id` you pass are
stored verbatim in each backend's metadata or payload fields (Chroma
and Redis also derive a hashed storage id from them, and conversation
history keys use SHA-256 digests of the ids, but the entry still
carries the raw values). If you're concerned about
tenant identifiers being readable by anyone with backend access,
hash them in your application before passing to sulci.

### Can I rotate tenant_ids?

The protocol doesn't have a rename-or-merge primitive. To rotate a
tenant_id from `old_id` to `new_id`, you'd need to clear (or
selectively delete) entries under `old_id` and let new entries land
under `new_id`. Cache loss during rotation is acceptable for most
use cases — sulci is a cache, not a system of record.

### Can a tenant see how many entries another tenant has?

`cache.stats()` returns this `Cache` instance's own counters (hits,
misses, saved cost, active sessions) across every tenant it served —
not per-tenant, and not the backend's contents. There's no protocol
method that exposes per-tenant counts to a tenant. If you need per-tenant
metrics for billing or capacity planning, that's an operator-side
concern (you can query the backend directly), not a tenant-facing
API.

### What happens if I pass an empty string as tenant_id?

Since 0.9.3, `Cache.get` / `Cache.set` raise `ValueError` for
`tenant_id=""` (and for `"global"` and any other falsy value but
`None`) — every backend filed `""` under the shared `"global"` scope,
so it was not a partition of its own. Pass `None` or a real tenant
identifier. (A backend called directly, without `Cache`, is pinned by
`test_empty_string_does_not_match_none` in
`tests/test_qdrant_tenant_isolation.py`.)

### What about deletion? GDPR / CCPA right-to-erasure?

Since 0.10.0, `Cache.delete_user(user_id, tenant_id=...)` erases one
user's entries and conversation history on every shipped backend
(`Backend.delete_user` is part of the protocol) and returns
`{"entries": N, "sessions": M}`. It needs `personalized=True`; a failure
raises `sulci.errors.SulciErasureError`, and repeating the call completes
it. On the managed backend it calls `DELETE /v1/cache/user/{user_id}`
(the API key decides the tenant). `Cache.clear()` removes everything,
every conversation session included. Per-tenant deletion of entries is
not a `Cache` method: give each tenant its own store, or `clear()` it.
The session store's `clear_scope(scope)` removes a tenant's or user's
conversation history on its own. See
[sulci.io/enterprise](https://sulci.io/enterprise) for the managed
product.

### Is tenant_id encrypted at rest?

That depends entirely on which backend you're using. Sulci doesn't
add encryption — it stores `tenant_id` as a normal field in your
chosen backend's storage. If your backend (Qdrant, Postgres-backed
SQLite, encrypted Redis, etc.) provides at-rest encryption, the
field is encrypted. If not, it isn't. Check your backend's
documentation.

### Can I use sulci OSS without ever thinking about tenant_id?

Yes. If your application has one tenant — or no concept of tenants
at all — never pass `tenant_id`. Everything goes into the `"global"`
partition automatically. The library works exactly as it did before
v0.4.0. The kwarg is optional precisely so single-tenant
deployments don't pay any complexity tax.

---

## See also

- [docs/protocols.md](protocols.md) — strict technical reference
  for the Backend and Embedder protocols
- [examples/extending_sulci/custom_backend.py](../examples/extending_sulci/custom_backend.py) —
  a worked reference implementation showing how to build a custom
  backend that satisfies the tenant isolation contract
- [tests/test_qdrant_tenant_isolation.py](../tests/test_qdrant_tenant_isolation.py) —
  the customer-scenario tests that pin behavior at the
  implementation level
- [tests/compat/](../tests/compat/) — the conformance test suite
  that validates any custom backend against the protocol
- [sulci.io/enterprise](https://sulci.io/enterprise) — for
  deployment options beyond the OSS library (managed cloud, VPC,
  air-gapped enterprise installations)
