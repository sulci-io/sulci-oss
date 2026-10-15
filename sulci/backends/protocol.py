# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/backends/protocol.py
==========================
Public Backend protocol — first introduced in v0.4.0.

STABLE API — modifications require api-reviewer approval per ADR 0005.
Changes to this protocol are BREAKING CHANGES for all customer-authored
backend implementations. Do not modify without a superseding ADR.

This protocol formalizes the shape every shipped backend
(chroma, qdrant, sqlite, redis, faiss, milvus, cloud) has, with one
v0.4.0 addition: `tenant_id` for multi-tenant partition isolation.
All shipped backends accept this kwarg. Since 0.9.3 every shipped backend
enforces it, and `user_id`, on every lookup (before, only QdrantBackend
and the managed backend did). See ADR 0005 for the rationale.

0.10.0 (ADR 0005 Amendment 2): `delete_user(user_id, *, tenant_id=None)
-> int` — remove one user's entries in one tenant and return how many.
`Cache.delete_user` calls it; a backend without it cannot erase a user
through `Cache` (NotImplementedError, naming the backend).

Usage
-----
Verify your custom backend conforms:

    from sulci.backends import Backend

    class MyBackend:
        def store(self, key, query, response, embedding, *,
                  tenant_id=None, user_id=None,
                  expires=None, metadata=None):
            ...
        def search(self, embedding, threshold, *,
                   tenant_id=None, user_id=None, now=None):
            ...
        def clear(self):
            ...
        def delete_user(self, user_id, *, tenant_id=None):   # 0.10.0
            ...

    # Duck-typed conformance (Protocol is structural):
    isinstance(MyBackend(), Backend)  # True if methods match by name
    # 0.10.0: delete_user is one of those names — a backend without it is not an instance.
    # NOTE: runtime_checkable only validates method NAMES, not signatures.
    # Run sulci.tests.compat.test_backend_conformance for full validation.

Inject into Cache:

    from sulci import Cache
    cache = Cache(backend=MyBackend(...))

Run the conformance suite against your implementation:

    from sulci.tests.compat import test_backend_conformance
    # Run with: pytest path/to/test_backend_conformance.py --backend-cls=MyBackend

See docs/protocols.md for the complete extension guide.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable, Optional


@runtime_checkable
class Backend(Protocol):
    """
    Protocol every sulci vector-cache backend must satisfy.

    Implementations shipped in v0.4.0:
      - ChromaBackend       sulci/backends/chroma.py
      - QdrantBackend       sulci/backends/qdrant.py
      - SQLiteBackend       sulci/backends/sqlite.py
      - RedisBackend        sulci/backends/redis.py
      - FaissBackend        sulci/backends/faiss.py
      - MilvusBackend       sulci/backends/milvus.py
      - SulciCloudBackend   sulci/backends/cloud.py
    Every one enforces tenant_id and user_id (0.9.3).

    Custom implementations: any class matching this surface.
    Verify conformance: sulci.tests.compat.test_backend_conformance
    """

    def store(
        self,
        key: str,
        query: str,
        response: str,
        embedding: list[float],
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Insert or replace a cache entry.

        Args:
            key:       Deterministic identifier for this cache entry.
                       ``Cache.set`` derives it from the query text
                       alone, so it does NOT identify the scope: a
                       store MUST NOT replace an entry of another
                       (tenant_id, user_id) that has the same key
                       (0.9.3; docs/protocols.md).
            query:     Original query text (for debugging / display).
            response:  The LLM response being cached.
            embedding: Normalized vector representation of `query`.
            tenant_id: Optional tenant (organization) scoping. Entries
                       written with a `tenant_id` are isolated from other
                       tenants — a `search` call without a matching
                       `tenant_id` MUST NOT return them. None = the
                       shared scope "global", which is a scope of its
                       own: entries written with a tenant are not in it.
            user_id:   Optional per-user scoping within a tenant. Layered
                       on top of `tenant_id`: tenant partitions first,
                       then user within tenant. "global" if None.
            expires:   Unix timestamp after which this entry is stale.
                       None or 0.0 = never expires.
            metadata:  Additional fields to store beside the entry: a
                       mapping whose keys are not the entry's own field
                       names (``sulci._metadata.RESERVED_METADATA_KEYS``).
                       ``Cache.set`` refuses others before calling store;
                       a backend that keeps metadata calls
                       ``sulci._metadata.checked_metadata(metadata)``
                       first and lets its ValueError / TypeError
                       propagate (0.9.3).

        Raises:
            Implementation-specific errors are acceptable but should be
            logged; never let a backend error crash the caller's request.
            The recommended pattern is to log and either swallow (store
            failures are not critical) or re-raise (as most backends do).
        """
        ...

    def search(
        self,
        embedding: list[float],
        threshold: float,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        """
        Approximate nearest-neighbor cosine search.

        Args:
            embedding: Query embedding to search with.
            threshold: Minimum cosine similarity (0.0-1.0) for a match.
                       Results below this score are treated as misses.
            tenant_id: Optional tenant filter. When set, only entries
                       stored with the same `tenant_id` are eligible
                       matches. Tenant isolation is a hard boundary:
                       entries from other tenants MUST NOT be returned
                       even if their similarity exceeds `threshold`.
            user_id:   Only match entries stored with exactly this
                       `user_id` within the tenant scope; None = only
                       entries stored without one ("global"). The scope
                       is part of the search, so other scopes' entries
                       cannot crowd this one's out of the candidates
                       (0.9.3).
            now:       Unix timestamp for TTL comparison. Uses time.time()
                       if None. Parameter is for test determinism.

        Returns:
            (response, similarity) on hit above threshold
            (None, 0.0) on miss (empty backend or no match above threshold)

        Must not raise on connectivity errors — log and return (None, 0.0)
        so that cache misses don't crash user applications. This is a core
        sulci invariant: a broken cache degrades to "no cache" quietly.
        """
        ...

    def clear(self) -> None:
        """
        Remove all entries from this backend. Destructive operation.

        Typically used in tests. In production, prefer TTL-based expiry
        and per-user deletion rather than clearing everything.
        """
        ...

    def delete_user(
        self,
        user_id: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> int:
        """
        0.10.0 (ADR 0005 Amendment 2). Remove every entry filed under
        (``tenant_id or "global"``, ``user_id``) — expired ones too — and
        nothing else. Returns how many entries were removed.

        ``Cache.delete_user`` is the only caller. It validates the ids,
        refuses a cache that is not personalized, erases the user's
        conversation history from the session store, and wraps a failure
        here in ``sulci.errors.SulciErasureError``. So this method:

        - removes exactly one scope: the same ``user_id`` in another
          tenant, other users of the same tenant, and the shared user
          ``"global"`` are left intact;
        - returns a true count: the number of entries removed, after which
          the user has none left. A backend whose client does not return a
          count counts the matches first, then deletes them by id, so the
          count and the deletion are about the same entries;
        - is idempotent: a second call returns 0;
        - does not validate: ``Cache`` already refused reserved ids, and a
          backend called directly treats its arguments as given, filing a
          missing tenant or user as ``"global"`` exactly as ``store`` does;
        - raises on failure, its own exception, and never removes part of
          the user's entries and reports success (unlike ``search``, which
          must not raise: a failed erasure must not look like one that
          happened);
        - is not in a hot path: a cost linear in the number of entries is
          acceptable.

        Verify with the conformance suite's ``TestUserErasure``.
        """
        ...