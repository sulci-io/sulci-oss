# Sulci — the measured public API surface

**Measured:** 2026-10-19 against `sulci` **0.10.2rc1** — `check_api_surface.py --show`, its output diffed against
the same command at the tag `v0.10.1`: besides the version banner, the differing lines are `Cache.__init__`'s
`context_window` default, `0` → `None`; `standalone` on `Cache.set`, `Cache.cached_call`, `AsyncCache.aset`,
`AsyncCache.acached_call` and the `AsyncCache.set` / `cached_call` passthroughs; and `context_threshold` on
`Cache.get`, `Cache.cached_call`, `AsyncCache.aget`, `AsyncCache.acached_call` and the `AsyncCache.get` /
`cached_call` passthroughs (oss-0037, the 📌 0.10.2 block and rows below). The 0.10.1 measurement follows.

📌 2026-10-19 (oss-0037, unreleased 0.10.2): `Cache.__init__`'s `context_window` default, `0` → `None`;
`standalone` on `Cache.set`, `Cache.cached_call`, `AsyncCache.aset`, `AsyncCache.acached_call` and the
`AsyncCache.set` / `cached_call` passthroughs; and a per-call `context_threshold` on `Cache.get`, `Cache.cached_call`,
`AsyncCache.aget`, `AsyncCache.acached_call` and the `get` / `cached_call` passthroughs — the 📌 0.10.2 block and
rows below — measured with
`check_api_surface.py --show` on the oss-0037 tree. `pyproject.toml` reads 0.10.1 until the 0.10.2 version bump,
so the version above stays.

**Measured:** 2026-10-17 against `sulci` **0.10.1** — `check_api_surface.py --show`, its output diffed against the
same command at the tag `v0.10.1rc1`: only the version banner differs. The 0.10.1rc1 measurement follows.

**Measured:** 2026-10-17 against `sulci` **0.10.1rc1** — `check_api_surface.py --show`, its output diffed against
the same command at the tag `v0.10.0`: besides the version banner, the only differing line is `Cache.__init__`'s
`threshold` default, `0.85` → `None` (oss-0032, the 📌 above the constructor block). The 0.10.0 measurement follows.

**Measured:** 2026-10-16 against `sulci` **0.10.0** — `check_api_surface.py --show`, its output diffed against the
same command at the tag `v0.10.0rc1`: only the version banner differs. The 0.10.0rc1 measurement follows.

**Measured:** 2026-10-16 against `sulci` **0.10.0rc1** — `check_api_surface.py --show`, its output diffed against
the same command at the tag `v0.9.3`: besides the version banner, the only differing lines are `Cache.delete_user`,
`AsyncCache.adelete_user` and `AsyncCache.delete_user` (each `*, tenant_id`), the 📌 0.10.0 rows below that oss-0016
recorded. The 0.9.3 measurement follows.

**Measured:** 2026-10-10 against `sulci` **0.9.3** — `check_api_surface.py --show`; the 0.9.3 additions are the
📌 rows and notes below (context methods' `tenant_id` / `user_id`). The 0.9.2 measurement follows.

📌 2026-10-14 (oss-0016, unreleased 0.10.0): `Cache.delete_user`, `AsyncCache.adelete_user` and the
`AsyncCache.delete_user` passthrough — the 📌 0.10.0 rows below — measured with `check_api_surface.py --show` on the
oss-0016 tree. `pyproject.toml` reads 0.9.3 until the 0.10.0 version bump, so the version above stays.

**Measured:** 2026-10-01 against `sulci` **0.9.2**

📌 **Re-verified, not re-measured.** The surface was measured 2026-07-24 at
version 0.9.0. On 2026-08-30 `check_api_surface.py --show` was run against both
0.9.0 and 0.9.1, and on 2026-10-01 against 0.9.1 (a clean `HEAD` worktree) and
0.9.2, and the outputs diffed: **each time the only differing line is the
version banner itself.** No method set, keyword-only parameter or
`Cache.__init__` default changed. Re-dating this header without that diff would have been a
restatement, which is the habit this file exists to end.
**Method:** AST parse of `sulci/core.py`, `sulci/async_cache.py`,
`sulci/integrations/langchain.py` and `sulci/integrations/llamaindex.py`. Not
memory, not the README.

*Adapter sections (`SulciCache`, `SulciCacheLLM`) added and measured
2026-08-10. Until that date `SOURCES` was `core.py` and `async_cache.py` only,
so the two classes an integration user actually imports had no drift guard —
and neither did the method **sets**, which this file's own checker had claimed
to verify since it was written.*

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

## Check this file

    python3 scripts/check_api_surface.py          # exit 1 on drift
    python3 scripts/check_api_surface.py --show   # the measured surface

Runs in CI on every PR, inside the `changes` job -- so it runs on docs-only PRs
too, where the test matrix is skipped and nothing else would read this file.

It checks the public method set, every keyword-only parameter, every
`Cache.__init__` default, and the version in the header above. It does not
check prose: a checker that fires on wording gets deleted within a week.

If the check disagrees with anything below, **the output wins** -- update this
file and date it, then grep the estate for whatever else restated the old value.

---

## `Cache.__init__` — the real defaults

Parameter order below is the real declaration order.

📌 **0.10.2 (oss-0037): `context_window` defaults to `None`**, which reads as `0` — a local backend is
unchanged. On the managed backend a window you set is sent with every lookup; `None` sends none, so the tenant's
dashboard setting applies. The block directly below is now the current declaration; the two after it are 0.10.1's
and 0.10.0's, kept as written.

```python
from sulci import Cache

cache = Cache(
    backend         = "chroma",    # chroma|sqlite|faiss|qdrant|redis|milvus|sulci
    threshold       = None,        # 0.10.1: 0.85 locally; on "sulci", the tenant's setting
    embedding_model = "minilm",    # minilm|mpnet|bge|openai
    ttl_seconds     = 86400,       # 24h — entries DO expire by default
    personalized    = False,
    db_path         = "./sulci_db",
    context_window  = None,        # 0.10.2: reads as 0; on "sulci", the tenant's setting
    query_weight    = 0.7,
    context_decay   = 0.5,
    session_ttl     = 3600,
    telemetry       = True,
    api_key         = None,        # also SULCI_API_KEY
    gateway_url     = "",          # empty; resolved downstream, not defaulted here
    session_store   = None,
    event_sink      = None,
    cost_per_call   = 0.005,
    context_threshold = None,   # blended lookups only; None = use `threshold`
)
```

📌 **0.10.1 (oss-0032): `threshold` defaults to `None`.** A local backend then applies `DEFAULT_THRESHOLD` (0.85);
the managed backend sends no threshold, so the tenant's dashboard setting applies. The first block below is the
current declaration; the second is 0.10.0's, kept as it was written.

```python
from sulci import Cache

cache = Cache(
    backend         = "chroma",    # chroma|sqlite|faiss|qdrant|redis|milvus|sulci
    threshold       = None,        # 0.10.1: 0.85 locally; on "sulci", the tenant's setting
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
    context_threshold = None,   # blended lookups only; None = use `threshold`
)
```

0.10.0's declaration:

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
    context_threshold = None,   # blended lookups only; None = use `threshold`
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
📌 0.10.0: nine — `delete_user` (the last row) is the ninth.

| Method | Keyword-only kwargs | Returns |
|---|---|---|
| `get(query, *, …)` | `threshold`, `tenant_id`, `user_id`, `session_id`, `plan` | `(response \| None, similarity, context_depth)` |
| `set(query, response, *, …)` | `tenant_id`, `user_id`, `session_id`, `metadata`, `plan` | `None` |
| `cached_call(query, llm_fn, *, …)` | `threshold`, `tenant_id`, `user_id`, `session_id`, `cost_per_call`, `plan` | `{response, source, similarity, latency_ms, cache_hit, context_depth}` |
| `get_context(session_id)` | — | `ContextWindow` |
| `clear_context(session_id)` | — | `None` |
| `context_summary(session_id=None)` | — | `dict` |
| `get_context(session_id, *, …)` — 📌 0.9.3 | `tenant_id`, `user_id` | `ContextWindow` |
| `clear_context(session_id, *, …)` — 📌 0.9.3 | `tenant_id`, `user_id` | `None` |
| `context_summary(session_id=None, *, …)` — 📌 0.9.3 | `tenant_id`, `user_id` | `dict` |
| `stats()` | — | `dict` |
| `clear()` | — | `None` |
| `delete_user(user_id, *, …)` — 📌 0.10.0 | `tenant_id` | `{"entries": N, "sessions": M}` |
| `get(query, *, …)` — 📌 0.10.2 | `threshold`, `tenant_id`, `user_id`, `session_id`, `plan`, `context_threshold` | `(response \| None, similarity, context_depth)` |
| `set(query, response, *, …)` — 📌 0.10.2 | `tenant_id`, `user_id`, `session_id`, `metadata`, `plan`, `standalone` | `None` |
| `cached_call(query, llm_fn, *, …)` — 📌 0.10.2 | `threshold`, `tenant_id`, `user_id`, `session_id`, `cost_per_call`, `plan`, `standalone`, `context_threshold` | `{response, source, similarity, latency_ms, cache_hit, context_depth}` |

Everything after `query` on `get` / `set` / `cached_call` is **keyword-only**.
`SyncCache` is an alias for `Cache`. `tenant_id` is a hard partition boundary
(v0.4.0).

📌 0.9.3: every backend enforces it — and `user_id` when `personalized=True` — on every lookup; `None` is the
shared scope and nothing else (before, only Qdrant and the managed backend filtered on `tenant_id`).

📌 0.9.3: the same holds for conversation history: a `session_id` is scoped by `tenant_id` and, when
`personalized=True`, `user_id`. `get_context`, `clear_context` and `context_summary` gained keyword-only
`tenant_id` and `user_id` (the table above shows them without), and `clear()` removes every session from any
session store. `context_summary()` with a `tenant_id` or `user_id` but no `session_id` raises `TypeError`.

📌 0.10.0: with an injected `session_store`, a failed write raises `sulci.sessions.SulciSessionError` — from
`set` (the entry **was** stored; only the conversation turn is missing) and from `clear_context` (the history
was **not** cleared). A failed read logs a WARNING on the `sulci` logger and continues without history (`get`,
`get_context`, `context_summary`). `cached_call` never raises it: it logs a WARNING and returns the response.
`clear()` raises it too when the store fails after the entries are cleared (0.9.3 let the store's own exception
through). `AsyncCache` behaves the same. Only a user turn with an embedding reaches an injected store; assistant
and text-only turns stay in the window. Through 0.9.3 every one of these failures except `clear()`'s was silent.

📌 0.9.3: `get` and `set` raise `ValueError` for a `tenant_id`, or a `user_id` when `personalized=True`,
that is `"global"` or falsy but not `None` (`""`, `0`, ...) — all of them named the shared scope. Pass
`None` for no tenant or user.

### `metadata` is on `set` only

`metadata` exists on `set` and `aset`. **It is not on `cached_call`** and never
has been, despite having been documented there. It is also not on `get`.

📌 0.9.3: `metadata` may not use the names of the entry's own fields — `key`, `query`, `response`,
`tenant_id`, `user_id`, `expires`, `created` (`sulci._metadata.RESERVED_METADATA_KEYS`). They are dropped,
with one WARNING naming them, and the entry keeps its own values; every other key is stored as before.
📌 Superseded before release: such a `metadata` raises `ValueError` naming the keys (the gateway answers
`422`), and one that is not a mapping raises `TypeError` — on every backend, before anything is stored.

### `delete_user` — a `Cache` method since 0.10.0

```python
from sulci import Cache

cache = Cache(backend="sqlite", db_path="./sulci_erasure_demo", personalized=True)
cache.set("What is my plan?", "You are on Pro.", tenant_id="t1", user_id="u-42")
print(cache.delete_user("u-42", tenant_id="t1"))   # {'entries': 1, 'sessions': 0}
```

What it does, and what it refuses, is listed under *Decided 2026-10-14* at the end of this section. Through 0.9.3
it was not a `Cache` method; the record of that, kept because it caused real damage, follows.

#### The record: before 0.10.0 `delete_user` was not a `Cache` method

At 0.9.3, `cache.delete_user(user_id)` raised `AttributeError`; so did `cache.backend.delete_user(user_id)`
(`.backend` is the *string*); only `cache._backend.delete_user(user_id)` worked, on the cloud backend alone, through
a private attribute.

- `delete_user` is defined in exactly one place: `sulci/backends/cloud.py:233`.
  📌 0.9.3: `delete_user(user_id, *, raise_on_error=False)` and `clear(*, raise_on_error=False)` (the line
  number above has moved). The id is percent-encoded, `/` and `.` included. Both return the gateway's answer
  on success; on failure they log a WARNING and return `None`, or raise `SulciErasureError` with
  `raise_on_error=True`.
  📌 Superseded before release: `delete_user(user_id) -> dict` and `clear() -> dict`, no flag. Any
  erasure the gateway does not confirm raises `SulciErasureError` (nothing is logged), and so does
  `Cache.clear()` on the managed backend, before it resets stats or sessions.
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

**Correct as written at 0.9.3:** the README then attributed the method to
`SulciCloudBackend`; both CHANGELOGs describe it as a backend method in
past-tense release notes, and those stay as written. The README describes
`Cache.delete_user` since oss-0035.

**Open decision.** Whether to add a guarded `Cache.delete_user()` proxy, add
`delete_user` to the `Backend` protocol, or keep per-user deletion an
HTTP-and-cloud-backend-only capability and say so everywhere. Until that is
decided, **documentation must describe the capability that exists**, which is:
`cache.clear()` on any backend, `DELETE /v1/cache/user/{id}` on the hosted tier.

📌 **Decided 2026-10-14, for 0.10.0 (oss-0016): `delete_user` is a `Cache` method and a `Backend` protocol
method.** The section above describes 0.9.3 and is kept as the record.

- `Cache.delete_user(user_id, *, tenant_id=None) -> {"entries": N, "sessions": M}` erases one end user of one
  tenant: the backend's entries filed under (tenant, user), expired ones included, through `Backend.delete_user`,
  then the user's conversation sessions through the session store's `clear_scope(user_scope(tenant, user))` — the
  built-in store or an injected one. Every shipped backend implements `Backend.delete_user` (SQLite, Redis, Chroma,
  Qdrant, Milvus and FAISS, all in 0.10.0). `AsyncCache.adelete_user` and the sync passthrough `delete_user` mirror it.
- It refuses before touching anything: `ValueError` for a `user_id` that is `None`, `""`, `"global"`, falsy or not
  valid Unicode text, or a reserved `tenant_id`; `ValueError` when the cache is not `personalized=True` (nothing is
  filed per user there; `clear()` is the route); `TypeError` for `tenant_id` on the managed backend (the API key
  decides the tenant); `NotImplementedError` when the backend has no `delete_user`.
- A failure raises `SulciErasureError` — now in `sulci.errors` (`sulci.backends.cloud` imports it from there) —
  with `.entries` and `.sessions`: the count of each half that completed, `None` for a half that did not (on the
  managed backend a partial count for a failed half stays in `.body`). The call is idempotent; a retry completes it.
- On the managed backend it sends `DELETE /v1/cache/user/{id}` and maps the gateway's `deleted` and
  `sessions_deleted` onto the same two keys.
- After a complete erasure it emits one `delete_user` event, without the user id.
- `Backend.delete_user(user_id, *, tenant_id=None) -> int` (ADR 0005 Amendment 2) removes exactly that scope's
  entries, returns a true count, is idempotent, does not validate, and raises on failure. SQLite has it from
  oss-0016; Redis, Chroma, Qdrant, Milvus and FAISS follow, one patch each, before 0.10.0 is tagged.
  📌 oss-0017: Redis has it too (a SCAN of the instance's own entry keys: O(keys under the prefix)).
  📌 oss-0018: Chroma has it too (the user's ids, read a page at a time, deleted by id).
  📌 oss-0020: Qdrant has it too (an exact count with the scope's filter, then one delete by it, waited for).
  📌 oss-0021: Milvus has it too (one delete by the scope key; the count is Milvus's answer).
  📌 oss-0024: FAISS has it too (the index rebuilt without the user's entries; a stop part way is settled before the
  files are next used) — every self-hosted backend now has it.
- `cache._backend.delete_user(...)` remains the managed backend's own method; call `Cache.delete_user` instead.
  The specification is `docs/architecture/specs/delete-user-erasure.md` in sulci-platform.

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
| `aget_context` — 📌 0.9.3 | `tenant_id`, `user_id` |
| `aclear_context` — 📌 0.9.3 | `tenant_id`, `user_id` |
| `acontext_summary` — 📌 0.9.3 | `tenant_id`, `user_id` |
| `get_context` (sync passthrough) — 📌 0.9.3 | `tenant_id`, `user_id` |
| `clear_context` (sync passthrough) — 📌 0.9.3 | `tenant_id`, `user_id` |
| `context_summary` (sync passthrough) — 📌 0.9.3 | `tenant_id`, `user_id` |
| `adelete_user` — 📌 0.10.0 | `tenant_id` |
| `delete_user` (sync passthrough) — 📌 0.10.0 | `tenant_id` |
| `aget` — 📌 0.10.2 | `tenant_id`, `plan`, `context_threshold` |
| `aset` — 📌 0.10.2 | `tenant_id`, `plan`, `metadata`, `standalone` |
| `acached_call` — 📌 0.10.2 | `tenant_id`, `plan`, `standalone`, `context_threshold` |
| `get` (sync passthrough) — 📌 0.10.2 | `threshold`, `tenant_id`, `plan`, `context_threshold` |
| `set` (sync passthrough) — 📌 0.10.2 | `tenant_id`, `plan`, `metadata`, `standalone` |
| `cached_call` (sync passthrough) — 📌 0.10.2 | `threshold`, `tenant_id`, `plan`, `standalone`, `context_threshold` |

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

### Three axes, and the third one was wrong until 2026-07-24

"Mirror" can mean three different things and this file previously described two:

| Axis | Status |
|---|---|
| **Which** kwargs are forwarded | Mirrored. Guarded by `TestAsyncSyncParity` since v0.8.1 |
| **How** they are passed | **Not** mirrored, deliberately — positional-or-keyword on the async twins, keyword-only on `Cache`. Documented above, not fixed |
| **What they default to** | Mirrored *as of this entry*. Was wrong, was unguarded |

`acached_call` and the `cached_call` passthrough declared
`cost_per_call: float = 0.005` while `Cache.cached_call` declares
`Optional[float] = None` — where `None` is the sentinel for *"use the instance
value"*. The wrapper therefore overrode `AsyncCache(cost_per_call=…)` on every
single call, and the arithmetic hid it: `Cache.get()` credits the instance
value, then `core.py:828` applies a delta because the forwarded 0.005 differs
from the instance's value, and the two cancel to exactly 0.005. A wrong number
that looks like a default is harder to spot than one that looks like noise.

Measure the third axis the same way as the first:

```bash
cd ~/code/sulci-oss
python3 - <<'PY'
import inspect
from sulci import AsyncCache
from sulci.core import Cache
for a, s in (("aget","get"), ("aset","set"), ("acached_call","cached_call")):
    sp = inspect.signature(getattr(Cache, s)).parameters
    for surface in (a, s):
        ap = inspect.signature(getattr(AsyncCache, surface)).parameters
        for kw in ("threshold","tenant_id","plan","metadata","cost_per_call"):
            if kw in sp and kw in ap and ap[kw].default != sp[kw].default:
                print(f"DRIFT AsyncCache.{surface}.{kw} = {ap[kw].default!r} "
                      f"vs Cache.{s}.{kw} = {sp[kw].default!r}")
print("done")
PY
```

A forwarded kwarg whose default differs from its source is an unconditional
override of whatever the constructor was given. There is no case where that is
what you meant.

---

## `SulciCache` — the LangChain adapter

`sulci/integrations/langchain.py`. Subclasses `BaseCache`; installed with
`set_llm_cache(SulciCache(...))`. Seven public methods.

| Method | Keyword-only | Returns |
|---|---|---|
| `__init__(*, …, **kwargs)` | `namespace_by_llm` | — |
| `lookup(prompt, llm_string)` | — | `list[Generation] \| None` |
| `update(prompt, llm_string, return_val)` | — | `None` |
| `clear(**kwargs)` | — | `None` |
| `alookup(prompt, llm_string)` | — | `list[Generation] \| None` |
| `aupdate(prompt, llm_string, return_val)` | — | `None` |
| `aclear(**kwargs)` | — | `None` |
| `stats()` | — | `dict` |

**`namespace_by_llm` defaults to `True`.** Each distinct `llm_string` gets its
own `Cache` on a `db_path` suffixed with an 8-char MD5 of that string, so two
models never share cached responses. The partitions are created lazily on first
lookup, which means the on-disk footprint is a function of how many LLM configs
the process has seen, not of configuration.

**Everything else in `**kwargs` goes straight to `Cache(**kwargs)`** — backend,
threshold, `context_window`, `ttl_seconds`, all of it. The defaults are
therefore `Cache.__init__`'s defaults, above; this section does not restate
them, and neither should anything else.

⚠️ **`namespace_by_llm=True` is silently downgraded to `False` when
`backend="sulci"`**, with a `logger.warning`. Sulci Cloud isolates server-side,
and per-LLM `db_path` partitions would otherwise spin up phantom cloud backend
instances all pointing at one namespace. A reader who sets both and does not
watch the log gets behaviour the constructor argument denies.

⚠️ **`lookup` and `update` swallow every exception** — by design, so a cache
fault cannot crash the caller's app. The consequence is that a misconfigured or
unreachable backend does not raise: it presents as a cache that never hits.
`stats()` and the warning log are the only evidence. **A hit rate of zero and a
cache that is not running are the same observation from the outside.**

`clear()` evicts the default cache *and* every namespace partition created so
far. Partitions belonging to `llm_string`s this process has not seen are
untouched, because they have not been instantiated.

`stats()` reports the **default partition only**. With `namespace_by_llm=True`
— the default — that is not the aggregate, and on a multi-model app it can be
an empty cache while the real traffic is in the partitions.

---

## `SulciCacheLLM` — the LlamaIndex adapter

`sulci/integrations/llamaindex.py`. Wraps any LlamaIndex `LLM`
(`SulciCacheLLM(llm=inner)`) and delegates. Ten public methods.

| Method | Cached? | Returns |
|---|---|---|
| `metadata` | — (property) | `LLMMetadata`, the wrapped LLM's, unchanged |
| `complete(prompt, formatted=False, **kw)` | **yes** | `CompletionResponse` |
| `chat(messages, **kw)` | **yes** | `ChatResponse` |
| `acomplete(prompt, formatted=False, **kw)` | **yes** | `CompletionResponse` |
| `achat(messages, **kw)` | **yes** | `ChatResponse` |
| `stream_complete(prompt, formatted=False, **kw)` | **no** | `CompletionResponseGen` |
| `stream_chat(messages, **kw)` | **no** | `ChatResponseGen` |
| `astream_complete(prompt, formatted=False, **kw)` | **no** | `CompletionResponseAsyncGen` |
| `astream_chat(messages, **kw)` | **no** | `ChatResponseAsyncGen` |
| `stats()` | — | `dict` |

⛔ **Four of the ten methods are uncached pass-through.** All four streaming
paths hand straight to the wrapped LLM: a generator cannot be reliably stored
mid-stream, so there is no attempt. **An application that streams gets no
caching at all** — not a lower hit rate, none — and `stats()` will show it
truthfully as a cache that is barely used. This is a capability boundary, not a
tuning problem, and it should not be flattened into "works with LlamaIndex" on
any surface that a buyer reads.

**The chat cache key is the last user message**, not the message list. The list
grows every turn with system prompt and history, which would make the key
change on every call and never hit. If no `USER` role is present the fallback
key is every message's content joined with spaces.

`session_id` is popped out of `**kwargs` before the call reaches the wrapped
LLM and passed to `Cache.set` — so context-aware behaviour is available through
the adapter, but only by passing `session_id=` on each call.

`acomplete` / `achat` are the sync methods run in an executor, so they are
cached by the same path; they do not block the event loop.

---

## `SulciLiteLLMCache` — the LiteLLM adapter

Added 2026-08-11. `sulci.integrations.litellm`. Subclasses
`litellm.caching.base_cache.BaseCache`. Extra: `sulci[litellm]`.

| Method | Keyword-only | Notes |
|---|---|---|
| `__init__(cache=None, *, …, **kwargs)` | `namespace_by_model`, `session_key` | — |
| `get_cache(key, **kwargs)` | — | Prompt comes from `kwargs["messages"]` / `["input"]`, **not** from `key` |
| `set_cache(key, value, **kwargs)` | — | Value JSON-serialised with `default=str` |
| `async_get_cache(key, **kwargs)` | — | `run_in_executor` over the sync path |
| `async_set_cache(key, value, **kwargs)` | — | idem |
| `async_set_cache_pipeline(cache_list, **kwargs)` | — | iterates; no batching |
| `disconnect()` | — | no-op, parity with `BaseCache` |
| `stats()` | — | passthrough to `Cache.stats()` |

Seven public methods. Keyword-only constructor args: `namespace_by_model`
(default `True`), `session_key` (default `"sulci_session_id"`).

**`key` is not usable for semantic lookup.** LiteLLM computes it as a hash of
the whole request, so it changes with every prompt. The prompt itself arrives
in `kwargs` — the same contract `RedisSemanticCache._get_prompt_from_kwargs`
reads.

**LiteLLM has no `custom` cache type.** `LiteLLMCacheType` is exactly {local,
redis, redis-semantic, valkey-semantic, s3, disk, qdrant-semantic, azure-blob,
gcs}, measured 2026-08-11 at litellm 1.96.0. The injection point is
`litellm.cache.cache = <BaseCache>`, which is what `install()` does.

---

## `sulci-mcp` and `sulci-proxy` — function surfaces, not classes

Both expose a factory rather than a public class, so neither is in
`CLASSES` in `check_api_surface.py`. **If either grows a public class, add it
there** — that tuple is the only thing standing between an adapter and the
undocumented-drift state these two sections exist to prevent.

| Entry point | Signature |
|---|---|
| `sulci.integrations.mcp_server.build_server` | `(cache=None, *, name, read_only, default_tenant_id, instructions, **cache_kwargs) -> MCPServer` |
| `sulci.integrations.mcp_server.main` | `(argv=None) -> None` — console script `sulci-mcp` |
| `sulci.proxy.build_app` | `(cache=None, *, openai_upstream, anthropic_upstream, share_across_models, timeout, client, **cache_kwargs) -> FastAPI` |
| `sulci.proxy.__main__.main` | `(argv=None) -> None` — console script `sulci-proxy` |

MCP tools: `cache_lookup`, `cache_stats` (`readOnlyHint=True`), `cache_store`.
Proxy routes: `POST /v1/chat/completions`, `POST /v1/messages`, `GET /healthz`,
`GET /stats`.

**Requires `mcp>=2.0.0`.** `mcp.server.fastmcp.FastMCP` was removed in 2.0;
the entry point is `mcp.server.MCPServer`. A `>=1.0.0` pin resolves and then
fails at import, which is why the extra pins 2.

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

✅ 0.9.3: superseded — every backend enforces tenant (and, when `personalized=True`, user) isolation on
lookup; the "label only" column above describes 0.9.2.

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

*First measured 2026-07-24 at version 0.9.0; the `SulciLiteLLMCache`,
`sulci-mcp` and `sulci-proxy` sections measured 2026-08-11; re-verified
unchanged on 2026-08-30. Re-run the command at the top before trusting any line
of this.*

⚠️ **`sulci/backends/` is not in `check_api_surface.py`'s `SOURCES`, so backend
constructors are unguarded.** v0.9.1 added two public keyword arguments to
`QdrantBackend.__init__` (`on_disk`, `quantization`) and neither this file nor
its checker observed it — correctly, since backends were never in scope, but the
gap is worth stating rather than leaving implied. Same shape as the 2026-08-10
note at the top: the adapter classes an integration user actually imports had no
drift guard until someone noticed, and a backend a self-hosting user constructs
by hand is in that category too. Widening `SOURCES` is its own change, not a
line in a feature PR.

⚠️ **Two checkers read the version out of this file with different patterns.**
`check_api_surface.py` anchors on `**Measured:**` and the first **bolded**
version on that line; `check_release_ready.py` matches every `` `sulci` ``
followed by a version, anywhere in the file, and requires all of them to agree.
A header that satisfies one can be invisible to the other — and
`check_api_surface.py` **skips its version check entirely when its pattern finds
nothing**, so that invisibility reads as a pass. Measured on 2026-08-30: a
re-worded header made it report green while the doc still said 0.9.0. **If you
edit the header, run both, and confirm the first still reports a version rather
than `None`.**

📌 **0.10.0 (oss-0027): the two checkers now read the version the same way**, through
`check_api_surface.measured_version`, so the ⚠️ note above describes the scripts
before 0.10.0. The newest measurement is the first `**Measured:**` in this file
that a reader sees, wherever it stands on its line (not in an HTML comment, a fenced
block or a code span); its version is the one bolded
after `` `sulci` ``, `X.Y.Z` or a pre-release (`0.10.0rc1`). Older measurement lines
are history and are not compared. A newest measurement with no version they can read,
or one that text no reader sees contradicts, fails both — `check_api_surface.py` no longer skips its version check. Before this,
`check_release_ready.py` compared every line, so the 0.9.2 measurement kept
`make release-check` red. `tests/test_release_check.py` covers both scripts.
