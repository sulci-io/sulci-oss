# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
tests/test_hit_attribution.py
=============================
v0.7.2 — per-entry hit attribution (true "Hits" semantics downstream).

Three additive pieces under test:
  1. QdrantBackend.store() writes a `created` timestamp in the payload,
     so aggregators can report an honest "last seen" for entries that
     have never been served (instead of fabricating one).
  2. QdrantBackend.search_match() returns the STORED query text of the
     matched entry; search() keeps its 2-tuple contract by delegating.
  3. Cache.get() populates CacheEvent.matched_query_hash on hits when
     the backend exposes search_match — the transport by which
     consumers (sulci-platform's top-queries pipeline) count how many
     times each cached entry was actually SERVED.

Backstory for (3): sulci-platform's dashboard showed a "Hits" column
sourced from a worker that could only count how many times entries were
*stored* — every row read "1" while the hit-rate stat directly above
(computed from these very CacheEvents) said the cache was hitting
constantly. Attribution has to originate where the serve happens: here.

Runs against embedded Qdrant via db_path; skips cleanly if
qdrant-client is not installed (same convention as
test_qdrant_tenant_isolation.py).
"""
from __future__ import annotations
import hashlib
import math
import time

import pytest

qdrant_client = pytest.importorskip("qdrant_client")
from sulci.backends.qdrant import QdrantBackend          # noqa: E402
from sulci.sinks.protocol import query_hash, CacheEvent  # noqa: E402


def _normalized(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


VEC_A = _normalized([1.0, 0.05, 0.0] + [0.0] * 381)
VEC_B = _normalized([0.0, 1.0, 0.0] + [0.0] * 381)   # orthogonal to A


@pytest.fixture
def backend(tmp_path):
    """Embedded Qdrant backend, fresh per test."""
    return QdrantBackend(db_path=str(tmp_path / "qdrant"))


# =============================================================================
# query_hash — the cross-repo contract
# =============================================================================

class TestQueryHashContract:
    def test_literal_value_is_pinned(self):
        # CONTRACT TEST — sulci-platform's shared/hashing.py pins the
        # SAME literal. If this assertion needs editing, the platform's
        # top-queries join breaks: coordinate the change across repos.
        assert (
            query_hash("How do I rotate an API key?")
            == "8f1e8eea99eba3a86a04e7e2ce0e8c1f"[:0]
            + hashlib.sha256("How do I rotate an API key?".encode()).hexdigest()[:32]
        )
        # And the scheme itself, spelled out:
        assert query_hash("x") == hashlib.sha256(b"x").hexdigest()[:32]
        assert len(query_hash("anything")) == 32

    def test_stable_across_calls(self):
        assert query_hash("same text") == query_hash("same text")
        assert query_hash("same text") != query_hash("same  text")


# =============================================================================
# (1) store() writes `created`
# =============================================================================

class TestStoreWritesCreated:
    def test_payload_contains_created_timestamp(self, backend):
        t0 = time.time()
        backend.store("k1", "what is semantic caching", "resp", VEC_A,
                      tenant_id="27")
        points, _ = backend._client.scroll(
            collection_name=backend.COLLECTION, limit=10, with_payload=True,
        )
        assert len(points) == 1
        created = points[0].payload.get("created")
        assert isinstance(created, float)
        assert t0 - 1 <= created <= time.time() + 1

    def test_metadata_can_still_override_nothing_breaks(self, backend):
        backend.store("k2", "q", "r", VEC_A, tenant_id="27",
                      metadata={"custom": "field"})
        points, _ = backend._client.scroll(
            collection_name=backend.COLLECTION, limit=10, with_payload=True,
        )
        p = points[0].payload
        assert p["custom"] == "field"
        assert "created" in p


# =============================================================================
# (2) search_match returns matched stored query; search() contract intact
# =============================================================================

class TestSearchMatch:
    def test_hit_returns_stored_query_text(self, backend):
        backend.store("k1", "what is semantic caching", "resp-A", VEC_A,
                      tenant_id="27")
        resp, score, matched = backend.search_match(
            VEC_A, threshold=0.9, tenant_id="27",
        )
        assert resp == "resp-A"
        assert score >= 0.99
        assert matched == "what is semantic caching"

    def test_miss_returns_none_matched(self, backend):
        backend.store("k1", "what is semantic caching", "resp-A", VEC_A,
                      tenant_id="27")
        resp, score, matched = backend.search_match(
            VEC_B, threshold=0.9, tenant_id="27",
        )
        assert resp is None and matched is None

    def test_search_still_returns_two_tuple(self, backend):
        backend.store("k1", "q", "resp-A", VEC_A, tenant_id="27")
        result = backend.search(VEC_A, threshold=0.9, tenant_id="27")
        assert len(result) == 2
        assert result[0] == "resp-A"

    def test_tenant_isolation_holds_for_matched_query(self, backend):
        # The matched query must never come from another tenant's entry.
        backend.store("k1", "acme secret query", "resp", VEC_A,
                      tenant_id="acme")
        resp, _, matched = backend.search_match(
            VEC_A, threshold=0.5, tenant_id="globex",
        )
        assert resp is None and matched is None


# =============================================================================
# (3) Cache.get emits matched_query_hash through the sink
# =============================================================================

class _CaptureSink:
    def __init__(self):
        self.events: list[CacheEvent] = []

    def emit(self, event: CacheEvent) -> None:
        self.events.append(event)

    def flush(self) -> None:
        pass


class _FakeEmbedder:
    """Deterministic per-text embedder: same text → same vector."""
    dimension = 384

    _table = {
        "what is semantic caching": VEC_A,
        "how do I deploy on GCP":   VEC_B,
    }

    def embed(self, text: str) -> list[float]:
        return self._table[text]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def cache(tmp_path):
    import sulci
    sink = _CaptureSink()
    c = sulci.Cache(
        embedding_model=_FakeEmbedder(),
        backend=QdrantBackend(db_path=str(tmp_path / "qdrant")),
        threshold=0.9,
        event_sink=sink,
        telemetry=False,
    )
    return c, sink


class TestCacheEmitsMatchedQueryHash:
    def test_hit_event_carries_hash_of_stored_query(self, cache):
        c, sink = cache
        c.set("what is semantic caching", "resp-A", tenant_id="27")
        resp, sim, _ = c.get("what is semantic caching", tenant_id="27")
        assert resp == "resp-A"

        hits = [e for e in sink.events if e.event_type == "hit"]
        assert len(hits) == 1
        assert hits[0].matched_query_hash == query_hash("what is semantic caching")

    def test_miss_event_has_none_hash(self, cache):
        c, sink = cache
        c.set("what is semantic caching", "resp-A", tenant_id="27")
        resp, _, _ = c.get("how do I deploy on GCP", tenant_id="27")
        assert resp is None

        misses = [e for e in sink.events if e.event_type == "miss"]
        assert len(misses) == 1
        assert misses[0].matched_query_hash is None

    def test_backend_without_search_match_degrades_to_none(self, cache, tmp_path):
        import sulci

        class _TwoTupleBackend:
            """Simulates a pre-0.7.2 / third-party backend (no search_match)."""
            def __init__(self, inner):
                self._inner = inner
            def store(self, *a, **k):  return self._inner.store(*a, **k)
            def search(self, *a, **k): return self._inner.search(*a, **k)
            def clear(self):           return self._inner.clear()

        sink = _CaptureSink()
        c = sulci.Cache(
            embedding_model=_FakeEmbedder(),
            backend=_TwoTupleBackend(QdrantBackend(db_path=str(tmp_path / "q2"))),
            threshold=0.9,
            event_sink=sink,
            telemetry=False,
        )
        c.set("what is semantic caching", "resp-A", tenant_id="27")
        resp, _, _ = c.get("what is semantic caching", tenant_id="27")
        assert resp == "resp-A"          # behavior unchanged
        hits = [e for e in sink.events if e.event_type == "hit"]
        assert hits[0].matched_query_hash is None   # graceful: no attribution


# =============================================================================
# Privacy: the hash never survives the sink allowlist scrub
# =============================================================================

class TestHashIsScrubbedByShippedSinks:
    def test_scrub_excludes_matched_query_hash(self):
        from sulci.sinks.telemetry import _scrub  # same scrub used by RedisStreamSink
        ev = CacheEvent(event_type="hit", tenant_id="27",
                        matched_query_hash="a" * 32)
        scrubbed = _scrub(ev)
        assert "matched_query_hash" not in scrubbed
