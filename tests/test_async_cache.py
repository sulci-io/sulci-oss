# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
tests/test_async_cache.py
=========================
Test suite for sulci.AsyncCache — 25 tests, zero API keys required.

All tests use the SQLite backend (in-memory temp dir) and the MiniLM
embedding model.  No network calls are made.

Test classes
------------
TestConstruction        ( 4) — constructor passthrough, repr, invalid backend
TestAget                ( 5) — hit, miss, session_id, user_id, 3-tuple return
TestAset                ( 3) — stores entry, advances context window, session_id
TestAcachedCall         ( 5) — hit, miss, dict shape, cost_per_call
TestContextMethods      ( 4) — aget_context, aclear_context, acontext_summary,
                               session isolation
TestStats               ( 3) — astats dict shape, aclear resets stats, repr
TestSyncPassthrough     ( 3) — sync get/set still work on AsyncCache instance
"""

import os
import tempfile
import pytest

from sulci import AsyncCache


class _FakeEmbedder:
    @property
    def dimension(self):
        return 2

    def embed(self, text):
        return [1.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_cache(tmp_path):
    """Fresh AsyncCache backed by SQLite in a temp directory."""
    return AsyncCache(
        backend        = "sqlite",
        db_path        = str(tmp_path / "test_cache"),
        threshold      = 0.85,
        embedding_model= "minilm",
        context_window = 4,
        query_weight   = 0.70,
        context_decay  = 0.50,
        session_ttl    = 3600,
    )


@pytest.fixture
def stateless_cache(tmp_path):
    """Stateless (context_window=0) AsyncCache."""
    return AsyncCache(
        backend  = "sqlite",
        db_path  = str(tmp_path / "stateless"),
        threshold= 0.85,
    )


# ── TestConstruction ─────────────────────────────────────────────────────────

class TestConstruction:

    def test_creates_internal_cache(self, tmp_cache):
        from sulci.core import Cache
        assert isinstance(tmp_cache._cache, Cache)

    def test_repr_contains_async_cache(self, tmp_cache):
        r = repr(tmp_cache)
        assert r.startswith("AsyncCache(")

    def test_repr_contains_inner_repr(self, tmp_cache):
        r = repr(tmp_cache)
        # inner Cache repr has hit_rate in it
        assert "hits=0" in r

    def test_invalid_backend_raises(self, tmp_path):
        with pytest.raises(Exception):
            AsyncCache(backend="nonexistent_backend_xyz",
                       db_path=str(tmp_path / "x"))


# ── TestAget ─────────────────────────────────────────────────────────────────

class TestAget:

    @pytest.mark.asyncio
    async def test_miss_returns_none_response(self, tmp_cache):
        response, sim, depth = await tmp_cache.aget("What is quantum computing?")
        assert response is None
        assert sim < 0.85  # similarity should be below threshold on miss
        assert depth == 0

    @pytest.mark.asyncio
    async def test_returns_3_tuple(self, tmp_cache):
        result = await tmp_cache.aget("What is Python?")
        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_hit_after_aset(self, tmp_cache):
        q = "What is semantic caching?"
        a = "Semantic caching stores responses by meaning."
        await tmp_cache.aset(q, a)
        response, sim, depth = await tmp_cache.aget(q)
        assert response == a
        assert sim >= 0.95   # exact match

    @pytest.mark.asyncio
    async def test_session_id_accepted(self, tmp_cache):
        q = "How does LangChain work?"
        a = "LangChain is a framework for LLM apps."
        await tmp_cache.aset(q, a, session_id="s1")
        response, sim, depth = await tmp_cache.aget(q, session_id="s1")
        assert response == a

    @pytest.mark.asyncio
    async def test_user_id_accepted(self, tmp_cache):
        q = "Explain the CAP theorem"
        a = "CAP: Consistency, Availability, Partition tolerance."
        await tmp_cache.aset(q, a, user_id="alice")
        response, sim, depth = await tmp_cache.aget(q, user_id="alice")
        assert response == a


# ── TestAset ─────────────────────────────────────────────────────────────────

class TestAset:

    @pytest.mark.asyncio
    async def test_aset_stores_entry(self, tmp_cache):
        await tmp_cache.aset("What is FAISS?", "FAISS is a vector search library.")
        response, sim, _ = await tmp_cache.aget("What is FAISS?")
        assert response is not None

    @pytest.mark.asyncio
    async def test_aset_advances_context_window(self, tmp_cache):
        sid = "session-ctx-test"
        await tmp_cache.aset("What is Python?", "Python is a language.", session_id=sid)
        _, _, depth = await tmp_cache.aget("What is Python?", session_id=sid)
        # After one set + one get, context window should have at least 1 turn
        ctx = await tmp_cache.aget_context(sid)
        assert ctx.depth >= 1

    @pytest.mark.asyncio
    async def test_aset_with_session_id(self, tmp_cache):
        sid = "session-aset"
        await tmp_cache.aset("What is Redis?", "Redis is an in-memory store.", session_id=sid)
        response, _, _ = await tmp_cache.aget("What is Redis?", session_id=sid)
        assert "Redis" in response


# ── TestAcachedCall ──────────────────────────────────────────────────────────

class TestAcachedCall:

    @pytest.mark.asyncio
    async def test_miss_calls_llm_fn(self, tmp_cache):
        called = []
        def mock_llm(q: str) -> str:
            called.append(q)
            return f"Answer: {q}"

        result = await tmp_cache.acached_call("What is Qdrant?", mock_llm)
        assert len(called) == 1
        assert result["source"] == "llm"

    @pytest.mark.asyncio
    async def test_hit_skips_llm_fn(self, tmp_cache):
        q = "What is ChromaDB?"
        a = "ChromaDB is a vector database."
        await tmp_cache.aset(q, a)

        called = []
        def mock_llm(query: str) -> str:
            called.append(query)
            return "should not be called"

        result = await tmp_cache.acached_call(q, mock_llm)
        assert len(called) == 0
        assert result["source"] == "cache"
        assert result["response"] == a

    @pytest.mark.asyncio
    async def test_result_dict_shape(self, tmp_cache):
        result = await tmp_cache.acached_call(
            "What is SQLite?",
            lambda q: "SQLite is a lightweight database."
        )
        assert "response"      in result
        assert "source"        in result
        assert "similarity"    in result
        assert "latency_ms"    in result
        assert "cache_hit"     in result
        assert "context_depth" in result

    @pytest.mark.asyncio
    async def test_cost_per_call_tracked(self, tmp_cache):
        await tmp_cache.acached_call(
            "What is Milvus?",
            lambda q: "Milvus is a vector database.",
            cost_per_call=0.01,
        )
        s = await tmp_cache.astats()
        assert s["saved_cost"] >= 0.0

    @pytest.mark.asyncio
    async def test_respects_constructor_cost_per_call(self, tmp_path):
        cache = AsyncCache(
            backend="sqlite",
            db_path=str(tmp_path / "async_cost_db"),
            embedding_model=_FakeEmbedder(),
            threshold=0.85,
            cost_per_call=0.003,
        )
        await cache.aset("What is pgvector?", "pgvector adds vector search to Postgres.")

        result = await cache.acached_call("What is pgvector?", lambda _: "unused")

        assert result["source"] == "cache"
        s = await cache.astats()
        assert s["saved_cost"] == pytest.approx(0.003)


# ── TestContextMethods ───────────────────────────────────────────────────────

class TestContextMethods:

    @pytest.mark.asyncio
    async def test_aget_context_returns_context_window(self, tmp_cache):
        sid = "ctx-session"
        await tmp_cache.aset("What is Python?", "Python is a language.", session_id=sid)
        ctx = await tmp_cache.aget_context(sid)
        from sulci.context import ContextWindow
        assert isinstance(ctx, ContextWindow)

    @pytest.mark.asyncio
    async def test_aclear_context_resets_depth(self, tmp_cache):
        sid = "clear-session"
        await tmp_cache.aset("What is Python?", "Python is a language.", session_id=sid)
        await tmp_cache.aclear_context(sid)
        ctx = await tmp_cache.aget_context(sid)
        assert ctx.depth == 0

    @pytest.mark.asyncio
    async def test_acontext_summary_all_sessions(self, tmp_cache):
        await tmp_cache.aset("Q1", "A1", session_id="s1")
        await tmp_cache.aset("Q2", "A2", session_id="s2")
        summary = await tmp_cache.acontext_summary()
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_session_isolation(self, tmp_cache):
        """Context in session-A must not bleed into session-B."""
        await tmp_cache.aset("What is Python?", "Python is a language.", session_id="A")
        ctx_b = await tmp_cache.aget_context("B")
        assert ctx_b.depth == 0


# ── TestStats ────────────────────────────────────────────────────────────────

class TestStats:

    @pytest.mark.asyncio
    async def test_astats_dict_shape(self, tmp_cache):
        s = await tmp_cache.astats()
        assert "hits"            in s
        assert "misses"          in s
        assert "hit_rate"        in s
        assert "saved_cost"      in s
        assert "total_queries"   in s
        assert "active_sessions" in s

    @pytest.mark.asyncio
    async def test_aclear_resets_stats(self, tmp_cache):
        await tmp_cache.aset("What is Python?", "Python is a language.")
        await tmp_cache.aget("What is Python?")
        await tmp_cache.aclear()
        s = await tmp_cache.astats()
        assert s["hits"]   == 0
        assert s["misses"] == 0

    @pytest.mark.asyncio
    async def test_hits_increment_on_cache_hit(self, tmp_cache):
        q = "What is SQLAlchemy?"
        a = "SQLAlchemy is a Python ORM."
        await tmp_cache.acached_call(q, lambda _: a)   # miss — stores entry
        await tmp_cache.acached_call(q, lambda _: a)   # hit
        s = await tmp_cache.astats()
        assert s["hits"] >= 1


# ── TestSyncPassthrough ──────────────────────────────────────────────────────

class TestSyncPassthrough:

    def test_sync_set_and_get(self, tmp_cache):
        q = "What is asyncio?"
        a = "asyncio is Python's async I/O framework."
        tmp_cache.set(q, a)
        response, sim, depth = tmp_cache.get(q)
        assert response == a

    def test_sync_stats_returns_dict(self, tmp_cache):
        s = tmp_cache.stats()
        assert isinstance(s, dict)
        assert "hits" in s

    def test_sync_cached_call_respects_constructor_cost_per_call(self, tmp_path):
        cache = AsyncCache(
            backend="sqlite",
            db_path=str(tmp_path / "sync_cost_db"),
            embedding_model=_FakeEmbedder(),
            threshold=0.85,
            cost_per_call=0.003,
        )
        cache.set("What is pgvector?", "pgvector adds vector search to Postgres.")

        result = cache.cached_call("What is pgvector?", lambda _: "unused")

        assert result["source"] == "cache"
        assert cache.stats()["saved_cost"] == pytest.approx(0.003)
