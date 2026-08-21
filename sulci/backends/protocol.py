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
All shipped backends accept this kwarg; QdrantBackend enforces tenant
isolation natively via payload filters, while other backends currently
store `tenant_id` as a labeling field. See ADR 0005 for the rationale.

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

    # Duck-typed conformance (Protocol is structural):
    isinstance(MyBackend(), Backend)  # True if methods match by name
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
      - QdrantBackend       sulci/backends/qdrant.py     (enforces tenant_id)
      - SQLiteBackend       sulci/backends/sqlite.py
      - RedisBackend        sulci/backends/redis.py
      - FaissBackend        sulci/backends/faiss.py
      - MilvusBackend       sulci/backends/milvus.py
      - SulciCloudBackend   sulci/backends/cloud.py      (enforces tenant_id)

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
            key:       Deterministic identifier for this cache entry
                       (typically hash of tenant + user + query text).
            query:     Original query text (for debugging / display).
            response:  The LLM response being cached.
            embedding: Normalized vector representation of `query`.
            tenant_id: Optional tenant (organization) scoping. Entries
                       written with a `tenant_id` are isolated from other
                       tenants — a `search` call without a matching
                       `tenant_id` MUST NOT return them. None = no tenant
                       scoping (single-tenant deployment).
            user_id:   Optional per-user scoping within a tenant. Layered
                       on top of `tenant_id`: tenant partitions first,
                       then user within tenant. "global" if None.
            expires:   Unix timestamp after which this entry is stale.
                       None or 0.0 = never expires.
            metadata:  Arbitrary additional fields to store in the backend.

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
            user_id:   Optional filter — only match entries stored with
                       this `user_id` (or "global" entries if permitted),
                       within the tenant scope.
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