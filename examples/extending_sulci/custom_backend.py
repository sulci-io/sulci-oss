"""
examples/extending_sulci/custom_backend.py
==========================================
A worked example showing how to implement a custom Backend that conforms
to the sulci v0.4.0+ Backend protocol.

This implementation is **deliberately simple** — an in-memory dict with
linear-scan cosine similarity. Real backends (Qdrant, Chroma, FAISS,
etc.) use specialized vector indexes; this example trades performance
for clarity so the protocol contract is the only thing on screen.

Run this file directly to see the backend in action:

    python examples/extending_sulci/custom_backend.py

Validate conformance with the public test suite:

    # In your project's tests/ directory:
    from examples.extending_sulci.custom_backend import InMemoryBackend
    # Then add InMemoryBackend to BACKEND_CLASSES in tests/compat/conftest.py
    # and run: pytest tests/compat/

The protocol is documented in docs/protocols.md. Multi-tenancy semantics
are documented in docs/multi_tenancy_and_isolation.md.
"""
from __future__ import annotations
import math
import time
from typing import Optional


class InMemoryBackend:
    """
    Reference implementation of the Backend protocol.

    Stores entries in a Python list. Every search() does a linear cosine
    scan — O(N) per query. Don't ship this for production; ship Qdrant,
    Chroma, or FAISS. This exists to show the contract.
    """

    #: This backend treats tenant_id as a real partition key — entries
    #: written under one tenant_id are never returned to a search under
    #: a different tenant_id, even when their similarity is high.
    ENFORCES_TENANT_ISOLATION: bool = True

    def __init__(self):
        # Each entry is a dict with: key, query, response, embedding,
        # tenant_id, user_id, expires, metadata. Stored as a list so
        # iteration order is stable for tests.
        self._entries: list[dict] = []

    # ---- Backend protocol methods -------------------------------------------

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

        The protocol stores tenant_id=None and user_id=None as the literal
        sentinel "global" so that searches with None pass through the
        same partition rather than returning anything-and-everything.
        See docs/multi_tenancy_and_isolation.md "operational migration"
        section for why this matters.
        """
        # Normalize sentinels at the storage boundary so search-time
        # filtering can compare against canonical values.
        entry = {
            "key":       key,
            "query":     query,
            "response":  response,
            "embedding": embedding,
            "tenant_id": tenant_id if tenant_id is not None else "global",
            "user_id":   user_id   if user_id   is not None else "global",
            "expires":   expires,
            "metadata":  metadata or {},
        }

        # Upsert by (key, tenant_id, user_id) — docs/protocols.md. Matching on the key alone (as this
        # example did before 0.9.3) lets another tenant or user storing the same question replace this entry.
        for i, existing in enumerate(self._entries):
            if (existing["key"], existing["tenant_id"], existing["user_id"]) == (key, entry["tenant_id"], entry["user_id"]):
                self._entries[i] = entry
                return
        self._entries.append(entry)

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
        Linear-scan cosine search with tenant + user partition filtering.

        Returns (response, similarity) on hit, (None, 0.0) on miss.
        Never raises — a backend that fails a search must degrade
        gracefully to "no cache" rather than crashing the caller.
        """
        # Normalize search-time sentinels to match the storage boundary.
        # Without this, tenant_id=None on read would not match tenant_id="global"
        # on write — a real cross-tenant data leak we caught in v0.4.0 phase 1.4.
        target_tenant = tenant_id if tenant_id is not None else "global"
        target_user   = user_id   if user_id   is not None else "global"
        now = now if now is not None else time.time()

        try:
            best_sim:  float = 0.0
            best_resp: Optional[str] = None

            for entry in self._entries:
                # Tenant isolation: hard filter, applied before similarity.
                if entry["tenant_id"] != target_tenant:
                    continue
                if entry["user_id"] != target_user:
                    continue

                # TTL filter: skip expired entries silently.
                if entry["expires"] is not None and now > entry["expires"]:
                    continue

                sim = _cosine_similarity(embedding, entry["embedding"])
                if sim >= threshold and sim > best_sim:
                    best_sim  = sim
                    best_resp = entry["response"]

            return (best_resp, best_sim) if best_resp is not None else (None, 0.0)
        except Exception:
            # Backend connectivity errors must not propagate — return a
            # clean miss instead. This is a core sulci invariant: a
            # broken cache degrades to "no cache" quietly.
            return (None, 0.0)

    def clear(self) -> None:
        """Remove all entries. Destructive, idempotent."""
        self._entries.clear()


# -----------------------------------------------------------------------------
# Helpers — not part of the protocol, just used by this example's search().
# -----------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors. Assumes inputs are
    already L2-normalized (which sulci embedders guarantee), in which
    case cosine similarity is just the dot product.
    """
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# -----------------------------------------------------------------------------
# Self-test — run this file directly.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("InMemoryBackend self-test")
    print("=" * 50)

    backend = InMemoryBackend()

    # Two L2-normalized vectors. v1 and v2 are nearly identical (high sim);
    # v3 is orthogonal (zero sim).
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.99, 0.14, 0.0]   # ~99% similar to v1
    v3 = [0.0, 0.0, 1.0]     # orthogonal to v1

    # 1. Empty search returns miss
    resp, sim = backend.search(v1, threshold=0.5)
    print(f"\n1. Empty search:           resp={resp!r}  sim={sim:.3f}")
    assert resp is None and sim == 0.0

    # 2. Store and retrieve
    backend.store(
        key="k1", query="hello", response="world",
        embedding=v1, tenant_id="acme",
    )
    resp, sim = backend.search(v1, threshold=0.5, tenant_id="acme")
    print(f"2. Same-tenant exact hit:  resp={resp!r}  sim={sim:.3f}")
    assert resp == "world" and sim > 0.99

    # 3. Cross-tenant must miss (tenant isolation guarantee)
    resp, sim = backend.search(v1, threshold=0.5, tenant_id="globex")
    print(f"3. Cross-tenant search:    resp={resp!r}  sim={sim:.3f}")
    assert resp is None and sim == 0.0, "Tenant isolation breach!"

    # 4. Paraphrased query (high sim) within tenant — hits
    resp, sim = backend.search(v2, threshold=0.85, tenant_id="acme")
    print(f"4. Paraphrase same tenant: resp={resp!r}  sim={sim:.3f}")
    assert resp == "world"

    # 5. Different topic (orthogonal vec) — misses even within tenant
    resp, sim = backend.search(v3, threshold=0.85, tenant_id="acme")
    print(f"5. Different topic:        resp={resp!r}  sim={sim:.3f}")
    assert resp is None

    # 6. Solo-developer mode (no tenant_id) round-trips correctly
    backend.store(
        key="solo", query="aws", response="lambda answer",
        embedding=v1,
    )
    resp, sim = backend.search(v1, threshold=0.5)
    print(f"6. Solo (no tenant):       resp={resp!r}  sim={sim:.3f}")
    assert resp == "lambda answer"

    # 7. clear() empties the backend
    backend.clear()
    resp, sim = backend.search(v1, threshold=0.5, tenant_id="acme")
    print(f"7. After clear():          resp={resp!r}  sim={sim:.3f}")
    assert resp is None

    print("\n" + "=" * 50)
    print("All assertions passed. InMemoryBackend conforms to the protocol.")
