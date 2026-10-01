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
implementation honors it (or doesn't, declared via the
`ENFORCES_TENANT_ISOLATION` class attribute).

| Backend | Enforces tenant isolation? | How |
|---|---|---|
| QdrantBackend | Yes | Qdrant payload `Filter(must=[FieldCondition(...)])` applied at query time |
| ChromaBackend | No (label-only) | `tenant_id` stored in metadata; not filtered on read |
| SQLiteBackend | No | `tenant_id` stored and part of the row's identity since 0.9.2 (the same query in two tenants is two rows); not filtered on read |
| RedisBackend | No | `tenant_id` accepted and ignored — not stored |
| FAISSBackend | No | Same as Redis |
| MilvusBackend | No | Same as Redis |
| SulciCloudBackend | (gateway-side) | Cloud gateway enforces via API key → tenant mapping; OSS conformance opts out because there's no local gateway to verify against |

Backends with `ENFORCES_TENANT_ISOLATION = False` accept the
`tenant_id` kwarg without raising — they're protocol-conformant — but
their `search()` doesn't filter on it. Those backends are appropriate
for single-tenant deployments where the application has no concept of
multiple tenants. **Don't use them for multi-tenant SaaS.**

The conformance test suite (`tests/compat/`) automatically runs the
`TestTenantIsolation` group only against backends declaring
`ENFORCES_TENANT_ISOLATION = True`. Custom backends that handle
multiple tenants should declare `True` and pass that group's tests.

---

## Customer scenarios

Concrete walkthroughs of how the model behaves in real situations.
These are also the names of test methods in
`tests/test_qdrant_tenant_isolation.py` — when one of those tests
fails, the failure tells you which user-impacting scenario is broken.

### Scenario 1 — Acme agents share each other's cache (the value prop)

> **Acme agent #1:** "What's the warranty on the Acme P500 pump?"

The cache misses, the LLM is called, the response gets stored:

```pythoncache.set(
query    = "What's the warranty on the Acme P500 pump?",
response = "The Acme P500 has a 5-year limited warranty.",
tenant_id = "acme_corp",
user_id   = "agent_001",
)

> **Acme agent #2 (different agent, same company):** "How long is
> the P500 covered under warranty?"

The cache **hits** — same tenant, semantically-similar query. The
response is returned without calling the LLM. This is sulci's
core value: cache hits within a tenant save cost and latency.

```pythoncache.get(
query     = "How long is the P500 covered under warranty?",
tenant_id = "acme_corp",
user_id   = "agent_002",
)
→ returns ("The Acme P500 has a 5-year limited warranty.", 0.92, 0)

**What could go wrong:** if the application forgets to pass
`tenant_id`, all entries land in the `"global"` partition. That works
fine for single-tenant apps but breaks the cross-tenant isolation
guarantee the moment the app starts handling multiple tenants. Always
pass `tenant_id` in multi-tenant deployments.

### Scenario 2 — Globex PHI must not leak to Initech

> **Globex agent:** "What's the bill code for John Smith's surgery
> on March 15?"

```pythoncache.set(
query     = "Bill code for John Smith's surgery on March 15?",
response  = "Patient John Smith MRN 4471823, code 49505. Balance $1,247.83.",
tenant_id = "globex_health",
)

> **Initech caller (different company, semantically-similar query):**
> "What's the policy for billing the patient surgery on March 15?"

The vector similarity between these two queries is ~0.85. Without
tenant isolation, this would match — and Initech would receive PHI
for a Globex patient.

With tenant isolation:

```pythonresp, sim, depth = cache.get(
query     = "What's the policy for billing the patient surgery on March 15?",
tenant_id = "initech_software",
)
→ returns (None, 0.0, 0)   ← clean miss

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

```pythoncache.set(
query     = "What pricing tier does Northrop Industries get?",
response  = "Northrop is Tier-3 enterprise: 22% discount, net-60 terms.",
tenant_id = "acme_corp",
user_id   = "agent_001_enterprise",
)

> **Acme residential-sales agent (agent_002):** "What pricing does
> Northrop get?"

```pythonresp, sim, depth = cache.get(
query     = "What pricing does Northrop get?",
tenant_id = "acme_corp",
user_id   = "agent_002_residential",
)
→ returns (None, 0.0, 0)

agent_002 doesn't have enterprise-customer access in their CRM.
They miss the cache and the LLM is called fresh, producing an
answer based on agent_002's own permissions (likely "I don't have
access to Northrop's pricing details"). The privileged response
that agent_001 saw never leaks to agent_002.

**Whether to set `user_id`** is a deployment decision. Some
applications have role-based access controls where different users
must see different responses to the same query — those should pass
`user_id`. Others have shared knowledge bases where all users in a
tenant should hit the same cache — those can leave `user_id`
unset. Sulci's protocol supports both.

### Scenario 4 — Solo-developer single-tenant deployment

> **Solo dev:** "How do I deploy a Python app to AWS Lambda?"

```pythoncache.set(
query    = "How do I deploy a Python app to AWS Lambda?",
response = "Package code as a .zip, upload via aws lambda create-function.",
# No tenant_id, no user_id
)

> **Solo dev later:** "What's the process for getting a Python app
> onto Lambda?"

```pythonresp, sim, depth = cache.get(
query = "What's the process for getting a Python app onto Lambda?",
# No tenant_id, no user_id
)
→ returns ("Package code as a .zip, upload via aws lambda create-function.", 0.94, 0)

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

```pythonfrom sulci import Cache
import jwtcache = Cache(backend="qdrant")def handle_chat_request(request, http_headers):
token  = http_headers.get("Authorization", "").removeprefix("Bearer ")
claims = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])# Derive partition keys from authenticated identity.
tenant_id = claims["org_id"]      # the customer's organization
user_id   = claims["sub"]          # the individual end-userresponse, sim, _ = cache.get(
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

The principle: **`tenant_id` and `user_id` come from authenticated
identity, never from request body.** Letting a caller pass their own
`tenant_id` in a request payload would let any caller spoof another
tenant's cache. Always derive from a token your app issued.

### From an API key map

```pythonTENANT_FROM_API_KEY = {
"sk_acme_xxxxxxxxxxxx":   "acme_corp",
"sk_globex_yyyyyyyyyy":   "globex_health",
"sk_initech_zzzzzzzzz":   "initech_software",
}def handle_request(request, headers):
api_key   = headers["Authorization"].removeprefix("Bearer ")
tenant_id = TENANT_FROM_API_KEY.get(api_key)
if tenant_id is None:
raise Unauthorized()user_id = headers.get("X-Customer-User-Id")  # optional, customer-suppliedresponse, sim, _ = cache.get(
    query     = request.query,
    tenant_id = tenant_id,
    user_id   = user_id,
)

The API key is the source of truth for `tenant_id`. The customer's
own user identifier (which they pass through in a header) becomes
`user_id`, but only as a partition label — sulci doesn't validate
those values, that's the customer's concern within their own tenant.

### From an LLM framework integration (LangChain, LlamaIndex)

Sulci's LangChain and LlamaIndex integrations forward cache kwargs
through their respective adapter classes. The `Cache` constructor
accepts the same `tenant_id`/`user_id` parameters; consult those
integrations' docstrings for framework-specific patterns.

---

## FAQ

### Does sulci hash tenant_id before storing it?

No. The `tenant_id` value you pass is stored verbatim in the
backend's metadata or payload field. If you're concerned about
tenant identifiers being readable by anyone with backend access,
hash them in your application before passing to sulci.

### Can I rotate tenant_ids?

The protocol doesn't have a rename-or-merge primitive. To rotate a
tenant_id from `old_id` to `new_id`, you'd need to clear (or
selectively delete) entries under `old_id` and let new entries land
under `new_id`. Cache loss during rotation is acceptable for most
use cases — sulci is a cache, not a system of record.

### Can a tenant see how many entries another tenant has?

The OSS library's `cache.stats()` returns aggregate counts for the
entire backend, not per-tenant. There's no protocol method that
exposes per-tenant counts to a tenant. If you need per-tenant
metrics for billing or capacity planning, that's an operator-side
concern (you can query the backend directly), not a tenant-facing
API.

### What happens if I pass an empty string as tenant_id?

`tenant_id=""` is treated as a literal empty-string partition key.
It's distinct from `tenant_id=None` (which becomes `"global"`).
Behavior is pinned by `test_empty_string_does_not_match_none` in
`tests/test_qdrant_tenant_isolation.py`. We recommend you don't pass
empty strings — pick `None` or a real tenant identifier.

### What about deletion? GDPR / CCPA right-to-erasure?

The OSS protocol's `clear()` removes everything. Per-user or
per-tenant deletion is not in the v0.4.0 OSS protocol surface. For
deployments with regulatory deletion requirements, this is an
enterprise-tier concern; see
[sulci.io/enterprise](https://sulci.io/enterprise) for what the
managed product offers, or contact us if you need help building it
into a custom backend.

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
