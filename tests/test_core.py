"""
tests/test_core.py
==================
Core Cache unit tests.
Uses SQLite backend — zero external dependencies beyond sentence-transformers.

Run:
    pip install "sulci[sqlite]" pytest
    pytest tests/ -v
"""
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sulci import Cache


@pytest.fixture
def cache(tmp_path):
    """Fresh SQLite-backed cache, isolated per test."""
    return Cache(
        backend         = "sqlite",
        threshold       = 0.85,
        embedding_model = "minilm",
        ttl_seconds     = None,
        db_path         = str(tmp_path / "test_db"),
    )


# ── Basic operations ──────────────────────────────────────────

class TestBasicOperations:

    def test_import(self):
        assert Cache is not None

    def test_version(self):
        import sulci
        assert hasattr(sulci, "_SDK_VERSION")
        assert sulci._SDK_VERSION

    def test_repr(self, cache):
        assert "Cache" in repr(cache)
        assert "sqlite" in repr(cache)

    def test_miss_on_empty_cache(self, cache):
        result, sim, _ = cache.get("What is Python?")
        assert result is None
        assert sim == 0.0

    def test_set_then_exact_get(self, cache):
        cache.set("What is Python?", "Python is a programming language.")
        result, sim, _ = cache.get("What is Python?")
        assert result == "Python is a programming language."
        assert sim >= 0.99

    def test_semantic_hit(self, tmp_path):
        """Near-identical query should reliably hit cache."""
        cache = Cache(backend="sqlite", threshold=0.85,
                      db_path=str(tmp_path / "hit_db"))
        cache.set("What is Python programming language?",
                  "Python is a programming language.")
        result, sim, _ = cache.get("What is the Python programming language?")
        assert result is not None, f"Expected cache hit, got sim={sim:.3f}"
        assert sim >= 0.85

    def test_semantic_miss_different_topic(self, cache):
        """Completely unrelated query should miss."""
        cache.set("What is Python?", "Python is a programming language.")
        result, sim, _ = cache.get("What is the weather in Paris today?")
        assert result is None

    def test_clear_empties_cache(self, cache):
        cache.set("What is Python?", "Python is a programming language.")
        cache.clear()
        result, _, __ = cache.get("What is Python?")
        assert result is None

    def test_multiple_entries(self, tmp_path):
        """Each query should retrieve its own closest match."""
        cache = Cache(backend="sqlite", threshold=0.85,
                      db_path=str(tmp_path / "multi_db"))
        cache.set("What is Python programming language?",
                  "Python is a language.")
        cache.set("What is Docker container platform?",
                  "Docker is a container platform.")
        r1, s1, _ = cache.get("What is the Python programming language?")
        r2, s2, __ = cache.get("What is the Docker container platform?")
        assert r1 is not None, f"Expected Python hit, sim={s1:.3f}"
        assert r2 is not None, f"Expected Docker hit, sim={s2:.3f}"

    def test_ttl_expiry(self, tmp_path):
        import time
        cache = Cache(backend="sqlite", threshold=0.85,
                      ttl_seconds=1, db_path=str(tmp_path / "ttl_db"))
        cache.set("What is Python?", "Python is a language.")
        time.sleep(1.1)
        result, _, __ = cache.get("What is Python?")
        assert result is None, "Entry should have expired"


# ── cached_call ───────────────────────────────────────────────

class TestCachedCall:

    def test_miss_calls_llm(self, cache):
        calls = [0]
        def fake_llm(q): calls[0] += 1; return f"Answer: {q}"
        r = cache.cached_call("What is Python?", fake_llm)
        assert r["source"]    == "llm"
        assert r["cache_hit"] is False
        assert calls[0]       == 1

    def test_hit_skips_llm(self, tmp_path):
        """Near-identical query should hit cache and skip LLM."""
        cache = Cache(backend="sqlite", threshold=0.85,
                      db_path=str(tmp_path / "hit_llm_db"))
        calls = [0]
        def fake_llm(q): calls[0] += 1; return "Python is a language."

        cache.cached_call("What is Python programming language?", fake_llm)
        assert calls[0] == 1

        r = cache.cached_call("What is the Python programming language?", fake_llm)
        assert r["source"]    == "cache"
        assert r["cache_hit"] is True
        assert calls[0]       == 1

    def test_result_has_all_fields(self, cache):
        r = cache.cached_call("test query", lambda q: "test response")
        for field in ["response", "source", "similarity", "latency_ms", "cache_hit", "context_depth"]:
            assert field in r, f"Missing field: {field}"

    def test_latency_is_positive(self, cache):
        r = cache.cached_call("test query", lambda q: "response")
        assert r["latency_ms"] > 0

    def test_similarity_in_range(self, cache):
        r = cache.cached_call("test query", lambda q: "response")
        assert 0.0 <= r["similarity"] <= 1.0

    def test_kwargs_forwarded_to_llm(self, cache):
        received = {}
        def fake_llm(q, model="default", temp=0.5):
            received.update({"model": model, "temp": temp})
            return "response"
        cache.cached_call("test", fake_llm, model="gpt-4", temp=0.9)
        assert received["model"] == "gpt-4"
        assert received["temp"]  == 0.9

    def test_response_stored_after_miss(self, cache):
        cache.cached_call("What is Python?", lambda q: "Python is great.")
        result, sim, _ = cache.get("What is Python?")
        assert result == "Python is great."


# ── Stats ─────────────────────────────────────────────────────

class TestStats:

    def test_initial_stats(self, cache):
        s = cache.stats()
        assert s["hits"]          == 0
        assert s["misses"]        == 0
        assert s["total_queries"] == 0
        assert s["hit_rate"]      == 0.0
        assert s["saved_cost"]    == 0.0

    def test_miss_increments_misses(self, cache):
        cache.cached_call("question", lambda q: "answer")
        assert cache.stats()["misses"] == 1
        assert cache.stats()["hits"]   == 0

    def test_hit_increments_hits(self, tmp_path):
        cache = Cache(backend="sqlite", threshold=0.85,
                      db_path=str(tmp_path / "stats_db"))
        cache.cached_call("What is Python programming language?",
                          lambda q: "Python is great.")
        cache.cached_call("What is the Python programming language?",
                          lambda q: "should not be called")
        s = cache.stats()
        assert s["misses"]   == 1
        assert s["hits"]     == 1
        assert s["hit_rate"] == 0.5

    def test_saved_cost_accumulates(self, tmp_path):
        cache = Cache(backend="sqlite", threshold=0.85,
                      db_path=str(tmp_path / "cost_db"))
        cache.cached_call("What is Python programming language?",
                          lambda q: "Python.", cost_per_call=0.01)
        cache.cached_call("What is the Python programming language?",
                          lambda q: "x",      cost_per_call=0.01)
        assert cache.stats()["saved_cost"] == pytest.approx(0.01)

    def test_stats_reset_on_clear(self, cache):
        cache.cached_call("question", lambda q: "answer")
        cache.clear()
        s = cache.stats()
        assert s["hits"]   == 0
        assert s["misses"] == 0

    # ── #42: raw .get()/.set() users now see non-zero stats() ───────

    def test_raw_get_miss_increments_misses(self, cache):
        """Raw .get() on an empty cache increments misses (issue #42)."""
        cache.get("anything")
        s = cache.stats()
        assert s["misses"]        == 1
        assert s["hits"]          == 0
        assert s["total_queries"] == 1

    def test_raw_set_then_raw_get_increments_hits(self, tmp_path):
        """set() + get() flow shows the hit in stats() (issue #42)."""
        cache = Cache(backend="sqlite", threshold=0.85,
                      db_path=str(tmp_path / "raw_stats_db"))
        cache.set("What is Python programming language?", "Python is great.")
        resp, sim, _ = cache.get("What is the Python programming language?")
        assert resp is not None, (
            "Sanity: paraphrase should hit at threshold=0.85; "
            f"got sim={sim}"
        )
        s = cache.stats()
        assert s["hits"]   == 1
        assert s["misses"] == 0

    def test_no_double_counting_via_cached_call(self, tmp_path):
        """cached_call() goes through .get() but mustn't double-count.

        Regression guard for the #42 fix: when we moved the increment
        into .get(), we removed the matching increments from
        cached_call() — this test fails if either side is restored.
        """
        cache = Cache(backend="sqlite", threshold=0.85,
                      db_path=str(tmp_path / "double_count_db"))
        # First call: miss → llm → set. Total .get() invocations: 1.
        cache.cached_call("What is Python programming language?",
                          lambda q: "Python is great.")
        # Second call: hit. Total .get() invocations: 2.
        cache.cached_call("What is the Python programming language?",
                          lambda q: "should not be called")
        s = cache.stats()
        assert s["misses"] == 1
        assert s["hits"]   == 1
        # If either path double-counted, total would be 3 or 4.
        assert s["total_queries"] == 2

    def test_saved_cost_from_raw_get_when_cost_per_call_set(self, tmp_path):
        """v0.7.1 (#88) — raw .get() populates saved_cost using Cache's
        cost_per_call (default $0.005, overridable on construction).

        Previously raw .get() left saved_cost at $0, which made the metric
        meaningless for LangChain users (set_llm_cache routes through .get(),
        not cached_call). With cost_per_call resolvable from Cache.__init__,
        .get() now contributes — using the instance default — on every hit.
        """
        cache = Cache(backend="sqlite", threshold=0.85,
                      cost_per_call=0.003,
                      db_path=str(tmp_path / "raw_savings_db"))
        cache.set("a question", "an answer")
        cache.get("a question")           # exact hit
        cache.get("a question")           # exact hit (identical query — guaranteed)
        assert cache.stats()["saved_cost"] == pytest.approx(0.006)

    def test_saved_cost_zero_when_cost_per_call_zero(self, tmp_path):
        """v0.7.1 (#88) — Cache(cost_per_call=0) opts out of saved_cost
        accounting on raw .get(), preserving the pre-v0.7.1 semantics for
        users who want only cached_call() to contribute.
        """
        cache = Cache(backend="sqlite", threshold=0.85,
                      cost_per_call=0.0,
                      db_path=str(tmp_path / "raw_savings_off_db"))
        cache.set("a question", "an answer")
        cache.get("a question")            # raw hit, no contribution
        assert cache.stats()["saved_cost"] == 0.0
        # cached_call with explicit override still works
        cache.cached_call("a question", lambda q: "irrelevant",
                          cost_per_call=0.01)
        assert cache.stats()["saved_cost"] == pytest.approx(0.01)


# ── Threshold behaviour ───────────────────────────────────────

class TestThreshold:

    def test_strict_threshold_rejects_paraphrase(self, tmp_path):
        cache = Cache(backend="sqlite", threshold=0.99,
                      db_path=str(tmp_path / "strict"))
        cache.set("What is Python?", "Python is a language.")
        result, _, __ = cache.get("Explain Python to me")
        assert result is None

    def test_lenient_threshold_accepts_loose_match(self, tmp_path):
        cache = Cache(backend="sqlite", threshold=0.40,
                      db_path=str(tmp_path / "lenient"))
        cache.set("What is Python?", "Python is a language.")
        result, _, __ = cache.get("Tell me about software programming")
        assert result is not None

    def test_exact_query_always_hits(self, cache):
        """Exact same string must always exceed any reasonable threshold."""
        cache.set("What is Python?", "Python is a language.")
        result, sim, _ = cache.get("What is Python?")
        assert result is not None
        assert sim >= 0.99


# ── Personalization ───────────────────────────────────────────

class TestPersonalization:

    def test_user_scoped_hit(self, tmp_path):
        cache = Cache(backend="sqlite", threshold=0.85,
                      personalized=True, db_path=str(tmp_path / "personal"))
        cache.set("What is Python?", "Python is a language.", user_id="alice")
        result, _, __ = cache.get("What is Python?", user_id="alice")
        assert result is not None

    def test_user_scoped_miss_for_other_user(self, tmp_path):
        cache = Cache(backend="sqlite", threshold=0.85,
                      personalized=True, db_path=str(tmp_path / "personal2"))
        cache.set("What is Python?", "Python is a language.", user_id="alice")
        result, _, __ = cache.get("What is Python?", user_id="bob")
        assert result is None, "Bob should not see Alice's cached entry"


# ── Tenant ID (v0.4.0+) ────────────────────────────────────────

class TestTenantId:
    """
    Verify tenant_id is plumbed through Cache public API (added in v0.4.0).

    These tests use SQLite (ENFORCES_TENANT_ISOLATION=False), so they only
    verify the kwarg is accepted and forwarded — not enforced. Real
    isolation enforcement is verified in tests/test_qdrant_tenant_isolation.py.
    """

    def test_cache_set_then_get_round_trips_with_tenant_id(self, tmp_path):
        cache = Cache(backend="sqlite", threshold=0.85,
                      db_path=str(tmp_path / "ti1"))
        cache.set("hello", "world", tenant_id="acme")
        resp, sim, _ = cache.get("hello", tenant_id="acme")
        assert resp == "world"
        assert sim >= 0.99

    def test_cache_cached_call_accepts_tenant_id_kwarg(self, tmp_path):
        cache = Cache(backend="sqlite", threshold=0.85,
                      db_path=str(tmp_path / "ti2"))
        def mock_llm(q, **_):
            return f"response to {q}"

        # First call — cache miss, LLM invoked
        result = cache.cached_call("hi", mock_llm, tenant_id="acme")
        assert result["source"] == "llm"
        assert result["response"] == "response to hi"

        # Second call — cache hit, LLM not invoked
        result = cache.cached_call("hi", mock_llm, tenant_id="acme")
        assert result["source"] == "cache"

    def test_cache_set_rejects_positional_after_response(self, tmp_path):
        """
        The ``*,`` separator must enforce keyword-only for partition kwargs
        after the required positional args. Without this guard, a future
        refactor could silently drop ``*,`` and let callers pass partition
        keys positionally — which would let argument-order mistakes go
        undetected and create cross-tenant data leak risks.
        """
        cache = Cache(backend="sqlite", threshold=0.85,
                      db_path=str(tmp_path / "ti3"))
        with pytest.raises(TypeError, match=r"takes \d+ positional argument"):
            cache.set("query", "response", "should-fail-positional")

    def test_cache_signatures_have_keyword_only_partition_kwargs(self):
        """
        Lock the contract: tenant_id, user_id, session_id are keyword-only
        on Cache.get / .set / .cached_call. Locks the v0.4.0 API shape so
        a future refactor can't silently drop ``*,`` and break callers.
        """
        import inspect
        for method_name in ("get", "set", "cached_call"):
            sig = inspect.signature(getattr(Cache, method_name))
            for partition_kw in ("tenant_id", "user_id", "session_id"):
                p = sig.parameters[partition_kw]
                assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                    f"Cache.{method_name}.{partition_kw} must be KEYWORD_ONLY, "
                    f"got {p.kind}"
                )
                assert p.default is None, (
                    f"Cache.{method_name}.{partition_kw} must default to None, "
                    f"got {p.default!r}"
                )

# ─────────────────────────────────────────────────────────────────────
# v0.5.6 — sulci-oss issue #36: plan attribution on CacheEvent
# ─────────────────────────────────────────────────────────────────────


class _RecordingSink:
    """Test double that captures every CacheEvent emitted to it.

    Not a MagicMock so the dataclass round-trip is real and field
    access mirrors what RedisStreamSink / TelemetrySink will see.
    """
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)

    def flush(self):
        pass


@pytest.fixture
def recording_cache(tmp_path):
    """Fresh SQLite-backed Cache with a recording sink wired in."""
    sink = _RecordingSink()
    cache = Cache(
        backend         = "sqlite",
        threshold       = 0.85,
        embedding_model = "minilm",
        ttl_seconds     = None,
        db_path         = str(tmp_path / "test_db"),
        event_sink      = sink,
    )
    return cache, sink


class TestCacheEventPlan:
    """The fix for sulci-oss #36 / sulci-platform #49.

    Pre-0.5.6 the gateway had no way to attribute billing-stream events
    to a plan tier; events landed in Redis with plan=None and downstream
    consumers had to do a per-event Postgres join to recover the plan.
    These tests pin the new contract.
    """

    def test_get_passes_plan_to_event(self, recording_cache):
        cache, sink = recording_cache
        cache.get("any query", tenant_id="t-1", plan="pro")
        assert len(sink.events) == 1
        assert sink.events[0].plan == "pro"
        # Sanity: existing fields still flow.
        assert sink.events[0].tenant_id == "t-1"
        assert sink.events[0].event_type == "miss"  # nothing stored yet

    def test_set_passes_plan_to_event(self, recording_cache):
        cache, sink = recording_cache
        cache.set("q", "r", tenant_id="t-1", plan="business")
        assert len(sink.events) == 1
        assert sink.events[0].plan == "business"
        assert sink.events[0].event_type == "set"

    def test_get_without_plan_emits_none(self, recording_cache):
        """Backward compat: callers who don't pass plan see plan=None.

        This is the shape every pre-0.5.6 caller was running with;
        it must keep working unchanged.
        """
        cache, sink = recording_cache
        cache.get("any query", tenant_id="t-1")
        assert len(sink.events) == 1
        assert sink.events[0].plan is None

    def test_set_without_plan_emits_none(self, recording_cache):
        cache, sink = recording_cache
        cache.set("q", "r", tenant_id="t-1")
        assert sink.events[-1].plan is None

    def test_cached_call_threads_plan_through_get_and_set(self, recording_cache):
        """cached_call delegates to .get() and (on miss) .set().

        Both emitted events must carry the plan from the cached_call
        invocation; otherwise gateway code that uses cached_call would
        still leak plan=None into the stream on miss-then-set paths.
        """
        cache, sink = recording_cache

        def stub_llm(query, **_):
            return "fake llm response"

        cache.cached_call("fresh query", stub_llm, tenant_id="t-1", plan="enterprise")

        # cached_call goes through .get (emits 'miss') then .set (emits 'set').
        # Both must carry plan='enterprise'.
        assert len(sink.events) == 2
        assert all(e.plan == "enterprise" for e in sink.events), \
            f"cached_call leaked plan: {[e.plan for e in sink.events]}"

    def test_plan_is_keyword_only_on_get_set_cached_call(self):
        """Lock plan as KEYWORD_ONLY with default None on all three methods.

        Mirrors the existing keyword-only enforcement for
        tenant_id/user_id/session_id (see
        TestTenantId.test_cache_signatures_have_keyword_only_partition_kwargs),
        so a future refactor can't silently drop ``*,`` and let plan
        slip into positional territory.
        """
        import inspect
        for method_name in ("get", "set", "cached_call"):
            sig = inspect.signature(getattr(Cache, method_name))
            p = sig.parameters["plan"]
            assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"Cache.{method_name}.plan must be KEYWORD_ONLY, got {p.kind}"
            )
            assert p.default is None, (
                f"Cache.{method_name}.plan must default to None, got {p.default!r}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# v0.6.0 — Cache.__init__ accepts native Embedder + Backend protocol instances
# (sulci-oss #34 C1c + C1d, umbrella #63)
#
# Mirrors the test_session_store_injection.py pattern from v0.5.0 (ADR 0007):
# verifies that passing a pre-constructed instance through the existing
# `embedding_model=` / `backend=` parameters is honored end-to-end, while
# the string-based dispatch path is unchanged.
#
# These tests do NOT instantiate MiniLM — they use fake protocol-shaped
# objects, which is the whole point of the feature (decouple Cache from
# the registry of named embedders/backends).
# ═══════════════════════════════════════════════════════════════════════════

class _FakeEmbedder:
    """Minimal Embedder-protocol-shaped fake.

    Returns a deterministic 4-d vector per query — small enough to keep
    fixtures readable, large enough that cosine-similarity is well-defined.
    """
    dimension = 4

    def __init__(self):
        self.embed_calls       = []
        self.embed_batch_calls = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        # Deterministic but non-trivial: hash → 4 floats in [0,1)
        h = hash(text)
        return [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(4)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_batch_calls.append(list(texts))
        return [self.embed(t) for t in texts]


class _FakeBackend:
    """Minimal Backend-protocol-shaped fake.

    In-memory store keyed by `key`. Search returns the first stored entry
    above threshold — deterministic given the FakeEmbedder above.
    """
    ENFORCES_TENANT_ISOLATION = False

    def __init__(self):
        self._entries     = []           # list of (embedding, response, metadata)
        self.search_calls = []
        self.store_calls  = []
        self.clear_calls  = 0

    def search(self, *, embedding, threshold, tenant_id=None, user_id=None, now=None):
        self.search_calls.append({
            "embedding": list(embedding),
            "threshold": threshold,
            "tenant_id": tenant_id,
            "user_id":   user_id,
        })
        # Trivial: return the most recently stored response if any exists.
        # Real similarity scoring would happen here in a non-fake backend;
        # for injection tests we only care that .search was reached.
        if self._entries:
            emb, resp, meta = self._entries[-1]
            return (resp, 1.0)
        return (None, 0.0)

    def store(self, *, key, query, response, embedding,
              tenant_id=None, user_id=None, expires=None, metadata=None):
        self.store_calls.append({
            "key":       key,
            "query":     query,
            "response":  response,
            "embedding": list(embedding),
            "tenant_id": tenant_id,
            "user_id":   user_id,
        })
        self._entries.append((embedding, response, metadata or {}))

    def clear(self):
        self.clear_calls += 1
        self._entries.clear()


class TestInstanceInjection:
    """v0.6.0 — Cache.__init__ accepts pre-constructed Embedder/Backend instances."""

    # ── default-path regression: existing string-based callers unchanged ──

    def test_string_paths_still_work(self, tmp_path):
        """Default Cache(backend='sqlite', embedding_model='minilm') path resolves
        the string dispatch identically to v0.5.x. We mock MiniLMEmbedder here
        to keep the test offline-runnable — the purpose is to verify that the
        STRING IS DISPATCHED through _load_embedder (i.e., not stored as-is
        and not skipped), not to verify the MiniLM model itself loads."""
        from unittest.mock import patch, MagicMock
        fake_embedder_class = MagicMock()
        fake_embedder_class.return_value = MagicMock()
        with patch("sulci.embeddings.minilm.MiniLMEmbedder", fake_embedder_class):
            c = Cache(
                backend         = "sqlite",
                embedding_model = "minilm",
                db_path         = str(tmp_path / "db"),
            )
        # _load_embedder dispatched through the string registry
        fake_embedder_class.assert_called_once_with("minilm")
        # User-visible field preserves the original string argument
        assert c.embedding_model == "minilm"
        assert c.backend         == "sqlite"
        # _embedder is the materialized instance (the mock), not the string
        assert not isinstance(c._embedder, str)
        assert not isinstance(c._backend,  str)

    # ── new path: instance injection ──

    def test_embedder_instance_passes_through(self, tmp_path):
        """Cache(embedding_model=<Embedder instance>) honors the instance directly —
        no MiniLM load, identity-equal to what we passed in."""
        fake = _FakeEmbedder()
        c = Cache(
            backend         = "sqlite",
            embedding_model = fake,
            db_path         = str(tmp_path / "db"),
        )
        # Identity check — not a copy, not a wrapper
        assert c._embedder is fake
        # User-visible field preserves the original argument
        assert c.embedding_model is fake

    def test_backend_instance_passes_through(self, tmp_path):
        """Cache(backend=<Backend instance>) honors the instance directly —
        no SQLite filesystem touch, identity-equal to what we passed in."""
        fake = _FakeBackend()
        c = Cache(
            backend         = fake,
            embedding_model = _FakeEmbedder(),   # also fake so no MiniLM
            db_path         = "/nonexistent/path/that/would/fail/if/touched",
        )
        assert c._backend is fake
        assert c.backend  is fake

    def test_both_instances_injected_together(self, tmp_path):
        """The gateway's use case: inject both protocol instances simultaneously.
        Mirrors what `LibraryBackedCache(sulci.Cache)` does today via subclass
        override of `_load_embedder` + `_load_backend` — but now without the
        subclass workaround."""
        emb = _FakeEmbedder()
        be  = _FakeBackend()
        c = Cache(
            backend         = be,
            embedding_model = emb,
            db_path         = "/nonexistent/should/not/be/touched",
        )
        assert c._embedder is emb
        assert c._backend  is be

    def test_mixed_string_and_instance(self, tmp_path):
        """One string + one instance — both dispatch paths in the same call."""
        be = _FakeBackend()
        # Backend = instance; embedding_model = string would trigger MiniLM,
        # so use the opposite mix: string backend ('sqlite'), instance embedder.
        emb = _FakeEmbedder()
        c = Cache(
            backend         = "sqlite",
            embedding_model = emb,
            db_path         = str(tmp_path / "db"),
        )
        # Backend materialized from string
        assert not isinstance(c._backend, str)
        # Embedder is the injected instance
        assert c._embedder is emb

    # ── end-to-end: injected instances are actually USED by Cache.get/set ──

    def test_injected_embedder_is_called_on_set_and_get(self, tmp_path):
        """Cache.set() and .get() route through the injected embedder's .embed().
        Identity check via call-recording, not just storage."""
        emb = _FakeEmbedder()
        be  = _FakeBackend()
        c = Cache(
            backend         = be,
            embedding_model = emb,
            db_path         = "/nonexistent",
        )
        # set() must call embedder.embed() on the query
        c.set("how do I deploy to AWS?", "use ECS or Fargate")
        assert "how do I deploy to AWS?" in emb.embed_calls
        # get() must also call embedder.embed() on the query
        c.get("how do I deploy to AWS?")
        # at least one embed call per operation
        assert len(emb.embed_calls) >= 2

    def test_injected_backend_is_called_on_set_and_get(self, tmp_path):
        """Cache.set() routes to backend.store(); Cache.get() routes to
        backend.search(). Verifies the engine actually delegates to the
        injected backend rather than constructing a parallel one."""
        emb = _FakeEmbedder()
        be  = _FakeBackend()
        c = Cache(
            backend         = be,
            embedding_model = emb,
            db_path         = "/nonexistent",
        )
        c.set("question", "answer")
        c.get("question")
        # store() was reached
        assert len(be.store_calls) == 1
        assert be.store_calls[0]["query"]    == "question"
        assert be.store_calls[0]["response"] == "answer"
        # search() was reached
        assert len(be.search_calls) == 1


# ─────────────────────────────────────────────────────────────────────────
# v0.6.1 (sulci-oss #60) — cloud-transport construction without a local
# embedder. Constructing Cache with a remote-transport backend should NOT
# trigger a local Embedder load (and therefore must not require
# sentence-transformers to be installed). These tests use a minimal fake
# remote-transport backend — duck-typed via remote_get/remote_set, the
# same protocol that v0.6.0 #65 introduced for SulciCloudBackend.
# ─────────────────────────────────────────────────────────────────────────

class _FakeRemoteTransport:
    """Minimal remote-transport-shaped fake.

    Duck-types the cloud-transport protocol: exposes remote_get + remote_set.
    Cache.__init__'s capability check (hasattr) treats any object with both
    methods as a transport, which is exactly the v0.6.0 #65 contract.

    Records every call so tests can assert that Cache.get/Cache.set routed
    through the transport rather than the self-hosted self._embedder /
    self._backend.search path.
    """

    def __init__(self, get_returns=("hit-response", 0.95, 0)):
        self.remote_get_calls  = []
        self.remote_set_calls  = []
        self._get_returns      = get_returns  # (response, similarity, context_depth)

    def remote_get(self, *, query, threshold, user_id=None, session_id=None):
        self.remote_get_calls.append({
            "query":      query,
            "threshold":  threshold,
            "user_id":    user_id,
            "session_id": session_id,
        })
        return self._get_returns

    def remote_set(self, *, query, response, user_id=None, session_id=None,
                   ttl_seconds=None):
        self.remote_set_calls.append({
            "query":       query,
            "response":    response,
            "user_id":     user_id,
            "session_id":  session_id,
            "ttl_seconds": ttl_seconds,
        })


class TestCloudTransportNoLocalEmbedder:
    """v0.6.1 — Cache(remote_transport_backend) skips local Embedder load."""

    def test_construction_skips_embedder_load(self):
        """Cache with a remote-transport backend keeps self._embedder as None.

        This is the headline fix: cloud-only users (`pip install sulci[cloud]`)
        do not need sentence-transformers installed. The eager embedder load
        in v0.6.0 caused a hard ImportError at Cache.__init__ on the cloud
        path; v0.6.1 defers it.
        """
        transport = _FakeRemoteTransport()
        cache = Cache(backend=transport)

        assert cache._is_remote_transport is True
        assert cache._embedder is None, (
            "Cloud transport should NOT trigger a local embedder load. "
            "If this fires after a code change, audit Cache.__init__ for "
            "an unconditional _load_embedder() call."
        )

    def test_get_routes_through_transport_without_embedder(self):
        """Cache.get forwards the raw query string to remote_get; never
        touches self._embedder (which is None).
        """
        transport = _FakeRemoteTransport(
            get_returns=("Python is great.", 0.92, 0),
        )
        cache = Cache(backend=transport, threshold=0.85)

        resp, sim, depth = cache.get("What is Python?")

        assert resp  == "Python is great."
        assert sim   == 0.92
        assert depth == 0
        # Transport was called once with the raw query, threshold passed through
        assert len(transport.remote_get_calls) == 1
        call = transport.remote_get_calls[0]
        assert call["query"]     == "What is Python?"
        assert call["threshold"] == 0.85
        # Sanity: no embedder ever materialized
        assert cache._embedder is None

    def test_set_routes_through_transport_without_embedder(self):
        """Cache.set forwards (query, response) to remote_set; never touches
        self._embedder.
        """
        transport = _FakeRemoteTransport()
        cache = Cache(backend=transport, ttl_seconds=600)

        cache.set("What is Python?", "A programming language.")

        assert len(transport.remote_set_calls) == 1
        call = transport.remote_set_calls[0]
        assert call["query"]       == "What is Python?"
        assert call["response"]    == "A programming language."
        assert call["ttl_seconds"] == 600
        # Sanity: no embedder ever materialized
        assert cache._embedder is None

    def test_cached_call_hit_with_session_skips_local_embed(self):
        """cached_call() hit-record path must not call self._embedder.embed()
        on a remote-transport Cache, even when session_id + context_window
        are set. Without the v0.6.1 gate this would crash on None.embed().
        """
        transport = _FakeRemoteTransport(
            get_returns=("cached answer", 0.99, 0),
        )
        cache = Cache(
            backend        = transport,
            context_window = 4,   # session-tracking enabled
        )

        result = cache.cached_call(
            query      = "What is Python?",
            llm_fn     = lambda q, **kw: "should not be called",
            session_id = "test-session-001",
        )

        # cached_call should have returned the cached hit without invoking
        # llm_fn and without crashing on None.embed()
        assert result["cache_hit"] is True
        assert result["response"]  == "cached answer"
        assert result["source"]    == "cache"
        # And confirm the embedder really is None (i.e. the test
        # demonstrates the gate, not a side-channel)
        assert cache._embedder is None
