# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
tests/test_async_cache.py
=========================
Test suite for sulci.AsyncCache — 40 tests, zero API keys required.

All tests use the SQLite backend (in-memory temp dir) and the MiniLM
embedding model.  No network calls are made. The Qdrant tenant-isolation
class skips cleanly when qdrant-client is not installed.

Test classes
------------
TestConstruction        ( 4) — constructor passthrough, repr, invalid backend
TestAget                ( 5) — hit, miss, session_id, user_id, 3-tuple return
TestAset                ( 3) — stores entry, advances context window, session_id
TestAcachedCall         ( 5) — hit, miss, dict shape, per-call cost_per_call,
                               instance cost_per_call honoured (v0.8.3)
TestContextMethods      ( 4) — aget_context, aclear_context, acontext_summary,
                               session isolation
TestStats               ( 3) — astats dict shape, aclear resets stats, repr
TestSyncPassthrough     ( 4) — sync get/set work on AsyncCache; per-call
                               threshold honored through passthrough get (v0.8.2);
                               instance cost_per_call honoured (v0.8.3)
TestAsyncPartitionKwargs( 5) — tenant_id/plan/metadata forwarded onto the
                               emitted CacheEvent through aget/aset/acached_call
                               (v0.8.1, sulci-oss #108); back-compat None
TestAsyncTenantIsolation( 2) — hard cross-tenant isolation through aget on
                               Qdrant (skips without qdrant-client)
TestAsyncSyncParity     ( 5) — signature guard: async + sync-passthrough
                               methods accept the same forwardable kwargs as
                               their sync Cache counterparts, incl. the per-call
                               threshold (v0.8.1 partition kwargs, v0.8.2 threshold),
                               and now with the same DEFAULTS (v0.8.3)
"""

import os
import tempfile
import pytest

from sulci import AsyncCache


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
        """An explicit per-call override lands in saved_cost exactly.

        This used to assert `saved_cost >= 0.0` against a call that MISSED —
        saved_cost only accrues on hits, so the value being asserted was 0.0
        and the assertion could not fail. It passed identically before and
        after the v0.8.3 fix below, which is how a wrong default survived in
        the file that was supposed to be guarding it. Seed first, then assert
        the number.
        """
        q, a = "What is Milvus?", "Milvus is a vector database."
        await tmp_cache.aset(q, a)
        await tmp_cache.acached_call(q, lambda _q: a, cost_per_call=0.01)
        s = await tmp_cache.astats()
        assert s["saved_cost"] == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_instance_cost_per_call_used_when_not_overridden(self, tmp_path):
        """v0.8.3 — omitting cost_per_call uses the CONSTRUCTOR value.

        `acached_call` declared `cost_per_call: float = 0.005` and forwarded
        it unconditionally, so this asserted 0.005 no matter what the instance
        was built with. Two errors compounded into one plausible number:
        `Cache.get()` credits the instance value (core.py #88), then
        `cached_call` applies the per-call delta at core.py:828 — which fired
        precisely because the wrapper had passed an "explicit" 0.005 that
        differed from the instance's 0.02 — and subtracted 0.015 back out.
        The result was not noise, it was exactly 0.005: the default masquerading
        as a measurement.
        """
        cache = AsyncCache(
            backend       = "sqlite",
            db_path       = str(tmp_path / "cost_default"),
            threshold     = 0.85,
            cost_per_call = 0.02,
        )
        q, a = "What is semantic caching?", "Caching by meaning, not by string."
        await cache.aset(q, a)
        result = await cache.acached_call(q, lambda _q: a)
        assert result["cache_hit"] is True, "seeded query must hit for cost to accrue"
        s = await cache.astats()
        assert s["saved_cost"] == pytest.approx(0.02), (
            "AsyncCache(cost_per_call=0.02) must credit 0.02 per hit, not the "
            "0.005 that acached_call used to hardcode as its own default"
        )


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

    def test_sync_passthrough_get_honors_per_call_threshold(self, tmp_cache):
        """v0.8.2: a per-call threshold on the sync passthrough get reaches the
        backend. A near-miss paraphrase that misses at the strict instance
        threshold (0.85) must hit when the passthrough is called with a
        permissive threshold."""
        tmp_cache.set("How do I deploy to AWS?", "Use the deploy CLI.")
        # Strict instance threshold → paraphrase misses.
        strict_resp, _, _ = tmp_cache.get("What's the process for deploying on AWS?")
        # Permissive per-call threshold via the passthrough → same paraphrase hits.
        loose_resp, loose_sim, _ = tmp_cache.get(
            "What's the process for deploying on AWS?", threshold=0.10
        )
        assert loose_resp == "Use the deploy CLI."
        assert loose_sim >= 0.10
        # And the strict call did not spuriously hit at a lower bar.
        assert strict_resp is None or loose_sim >= 0.85

    def test_sync_cached_call_uses_instance_cost_per_call(self, tmp_path):
        """v0.8.3 — the passthrough carried the same hardcoded 0.005 default as
        its async twin. Both surfaces are fixed, so both are pinned: a fix
        applied to one of a pair is how the pair drifts."""
        cache = AsyncCache(
            backend       = "sqlite",
            db_path       = str(tmp_path / "cost_default_sync"),
            threshold     = 0.85,
            cost_per_call = 0.02,
        )
        q, a = "What is HNSW?", "A graph-based ANN index."
        cache.set(q, a)
        result = cache.cached_call(q, lambda _q: a)
        assert result["cache_hit"] is True
        assert cache.stats()["saved_cost"] == pytest.approx(0.02)


# ── TestAsyncPartitionKwargs ─────────────────────────────────────────────────
# v0.8.1 — sulci-oss #108: AsyncCache parity for tenant_id + plan (+ metadata).
#
# The async methods historically dropped tenant_id (Cache.get/set/cached_call
# since v0.4.0) and plan (since v0.5.6). These tests pin that the async path
# now forwards them so they land on the emitted CacheEvent — the exact async
# mirror of tests/test_core.py::TestCacheEventPlan. A recording sink captures
# the real dataclass round-trip (not a mock), matching what RedisStreamSink /
# TelemetrySink would see downstream.


class _RecordingSink:
    """Captures every CacheEvent emitted to it (mirrors test_core._RecordingSink)."""
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)

    def flush(self):
        pass


@pytest.fixture
def recording_async_cache(tmp_path):
    """Fresh SQLite-backed AsyncCache with a recording sink wired in."""
    sink = _RecordingSink()
    cache = AsyncCache(
        backend         = "sqlite",
        threshold       = 0.85,
        embedding_model = "minilm",
        db_path         = str(tmp_path / "rec_db"),
        event_sink      = sink,
    )
    return cache, sink


class TestAsyncPartitionKwargs:

    @pytest.mark.asyncio
    async def test_aget_passes_tenant_id_and_plan_to_event(self, recording_async_cache):
        cache, sink = recording_async_cache
        await cache.aget("any query", tenant_id="t-1", plan="pro")
        assert len(sink.events) == 1
        assert sink.events[0].tenant_id == "t-1"
        assert sink.events[0].plan == "pro"
        assert sink.events[0].event_type == "miss"   # nothing stored yet

    @pytest.mark.asyncio
    async def test_aset_passes_tenant_id_plan_metadata_to_event(self, recording_async_cache):
        cache, sink = recording_async_cache
        await cache.aset("q", "r", tenant_id="t-1", plan="business", metadata={"k": "v"})
        assert len(sink.events) == 1
        assert sink.events[0].tenant_id == "t-1"
        assert sink.events[0].plan == "business"
        assert sink.events[0].event_type == "set"

    @pytest.mark.asyncio
    async def test_acached_call_threads_plan_through_get_and_set(self, recording_async_cache):
        """acached_call delegates to .get() (miss) then .set(); both events
        must carry the plan, otherwise gateway-style async callers would leak
        plan=None into the stream on the miss-then-set path."""
        cache, sink = recording_async_cache

        def stub_llm(query, **_):
            return "fake llm response"

        await cache.acached_call("fresh query", stub_llm, tenant_id="t-1", plan="enterprise")
        assert len(sink.events) == 2
        assert all(e.plan == "enterprise" for e in sink.events), \
            f"acached_call leaked plan: {[e.plan for e in sink.events]}"
        assert all(e.tenant_id == "t-1" for e in sink.events)

    @pytest.mark.asyncio
    async def test_aget_without_plan_emits_none(self, recording_async_cache):
        """Back-compat: pre-0.8.1 async callers (no plan) still see plan=None."""
        cache, sink = recording_async_cache
        await cache.aget("any query", tenant_id="t-1")
        assert len(sink.events) == 1
        assert sink.events[0].plan is None

    @pytest.mark.asyncio
    async def test_aset_without_plan_emits_none(self, recording_async_cache):
        cache, sink = recording_async_cache
        await cache.aset("q", "r", tenant_id="t-1")
        assert sink.events[-1].plan is None


# ── TestAsyncTenantIsolation ─────────────────────────────────────────────────
# v0.8.1 — the *hard boundary* proof through the async path. SQLite treats
# tenant_id as a label (ENFORCES_TENANT_ISOLATION = False), so isolation can
# only be proven on a backend that enforces it. QdrantBackend does, via a
# payload Filter. Skips cleanly when qdrant-client is not installed, exactly
# like tests/test_qdrant_tenant_isolation.py.


class TestAsyncTenantIsolation:

    @pytest.fixture
    def qdrant_async_cache(self, tmp_path):
        pytest.importorskip("qdrant_client")
        return AsyncCache(
            backend         = "qdrant",
            embedding_model = "minilm",
            threshold       = 0.85,
            db_path         = str(tmp_path / "qdrant_async"),
        )

    @pytest.mark.asyncio
    async def test_cross_tenant_entry_not_returned_through_aget(self, qdrant_async_cache):
        """An entry stored under tenant A must NOT be returned to tenant B
        through aget, even though the query text (and thus similarity) is
        identical — similarity must never bypass isolation."""
        cache = qdrant_async_cache
        q = "What is our refund policy?"
        await cache.aset(q, "Tenant-A refund policy: 30 days.", tenant_id="acme")

        # Same query, different tenant → hard miss despite ~1.0 similarity.
        resp, sim, _ = await cache.aget(q, tenant_id="globex")
        assert resp is None, "tenant globex must not see acme's cached entry"

    @pytest.mark.asyncio
    async def test_same_tenant_still_hits_through_aget(self, qdrant_async_cache):
        """Control: the owning tenant still gets its own entry back."""
        cache = qdrant_async_cache
        q = "What is our refund policy?"
        await cache.aset(q, "Tenant-A refund policy: 30 days.", tenant_id="acme")

        resp, sim, _ = await cache.aget(q, tenant_id="acme")
        assert resp == "Tenant-A refund policy: 30 days."


# ── TestAsyncSyncParity ──────────────────────────────────────────────────────
# v0.8.1 — a signature guard so the parity gap can't silently reopen. Mirrors
# tests/test_core.py::TestCacheEventPlan::test_plan_is_keyword_only_on_get_set_cached_call
# but asserts the ASYNC surface (and the sync passthrough on AsyncCache) carry
# the same partition kwargs as their sync Cache counterparts.


class TestAsyncSyncParity:

    def test_async_methods_are_keyword_only_partition_kwargs(self):
        import inspect
        for method_name in ("aget", "aset", "acached_call"):
            sig = inspect.signature(getattr(AsyncCache, method_name))
            for kw in ("tenant_id", "plan"):
                assert kw in sig.parameters, f"AsyncCache.{method_name} missing {kw}"
                p = sig.parameters[kw]
                assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                    f"AsyncCache.{method_name}.{kw} must be KEYWORD_ONLY, got {p.kind}"
                )
                assert p.default is None, (
                    f"AsyncCache.{method_name}.{kw} must default to None, got {p.default!r}"
                )
        # metadata mirrors Cache.set (which has it; Cache.get/cached_call do not).
        p = inspect.signature(AsyncCache.aset).parameters["metadata"]
        assert p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is None

    def test_sync_passthrough_mirrors_partition_kwargs(self):
        import inspect
        for method_name in ("get", "set", "cached_call"):
            sig = inspect.signature(getattr(AsyncCache, method_name))
            for kw in ("tenant_id", "plan"):
                assert kw in sig.parameters, f"AsyncCache.{method_name} (sync) missing {kw}"
                assert sig.parameters[kw].kind == inspect.Parameter.KEYWORD_ONLY

    def test_sync_passthrough_mirrors_threshold(self):
        """v0.8.2: the get / cached_call passthroughs forward the per-call
        threshold (added to their async twins in v0.8.0). Cache.set has no
        threshold, so the set passthrough must NOT grow one — a faithful
        mirror, not a superset."""
        import inspect
        for method_name in ("get", "cached_call"):
            p = inspect.signature(getattr(AsyncCache, method_name)).parameters
            assert "threshold" in p, f"AsyncCache.{method_name} (sync) missing threshold"
            assert p["threshold"].kind == inspect.Parameter.KEYWORD_ONLY
            assert p["threshold"].default is None
        assert "threshold" not in inspect.signature(AsyncCache.set).parameters, \
            "AsyncCache.set must not gain threshold (Cache.set has none)"

    def test_cost_per_call_defaults_mirror_cache(self):
        """v0.8.3 — cost_per_call must default to None on both AsyncCache
        surfaces, as it does on Cache.cached_call.

        None is not cosmetic here: it is the sentinel core.py reads as "use
        the value this Cache was constructed with". Any concrete default in
        the wrapper is an unconditional override of a constructor argument,
        which is what `float = 0.005` was. The general guard below now
        compares defaults across the whole mirrored set; this test names the
        one that was wrong so a future reader finds it by grep.
        """
        import inspect
        from sulci.core import Cache
        assert inspect.signature(Cache.cached_call).parameters[
            "cost_per_call"].default is None
        for surface in ("acached_call", "cached_call"):
            p = inspect.signature(getattr(AsyncCache, surface)).parameters
            assert "cost_per_call" in p, f"AsyncCache.{surface} missing cost_per_call"
            assert p["cost_per_call"].default is None, (
                f"AsyncCache.{surface}.cost_per_call defaults to "
                f"{p['cost_per_call'].default!r}; a concrete default here "
                f"overrides AsyncCache(cost_per_call=…) on every call"
            )

    def test_full_mirror_of_sync_cache_kwargs(self):
        """Every forwardable kwarg the sync Cache method exposes —
        {threshold, tenant_id, plan, metadata, cost_per_call} — must be
        accepted by BOTH its a-prefixed async twin AND its sync passthrough on
        AsyncCache, WITH THE SAME DEFAULT. This is the introspective form of
        "AsyncCache mirrors every Cache method".

        The default check is v0.8.3 and is the half that was missing. Presence
        alone passed green while `cost_per_call` defaulted to 0.005 on the
        wrapper and None on Cache — a mirror of which kwargs exist, not of what
        they do. Note the third axis, HOW they are passed, is deliberately NOT
        asserted: the async twins take several of these positional-or-keyword
        where Cache is keyword-only, which is documented in docs/API-SURFACE.md
        rather than fixed.
        """
        import inspect
        from sulci.core import Cache
        mirrored = {"threshold", "tenant_id", "plan", "metadata", "cost_per_call"}
        # async twin -> (sync passthrough, sync Cache source)
        pairs = {
            "aget":         ("get",         "get"),
            "aset":         ("set",         "set"),
            "acached_call": ("cached_call", "cached_call"),
        }
        for async_name, (passthrough_name, sync_name) in pairs.items():
            sync_sig = inspect.signature(getattr(Cache, sync_name)).parameters
            sync_p = set(sync_sig) & mirrored
            for surface in (async_name, passthrough_name):
                surface_sig = inspect.signature(getattr(AsyncCache, surface)).parameters
                have = set(surface_sig) & mirrored
                missing = sync_p - have
                assert not missing, (
                    f"AsyncCache.{surface} is missing {sorted(missing)} that "
                    f"Cache.{sync_name} accepts"
                )
                for kw in sorted(sync_p):
                    assert surface_sig[kw].default == sync_sig[kw].default, (
                        f"AsyncCache.{surface}.{kw} defaults to "
                        f"{surface_sig[kw].default!r} but Cache.{sync_name}.{kw} "
                        f"defaults to {sync_sig[kw].default!r} — a forwarded kwarg "
                        f"with a different default is an unconditional override"
                    )
