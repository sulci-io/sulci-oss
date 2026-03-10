"""
tests/test_context.py
=====================
Tests for sulci.context — ContextWindow and SessionStore.
Runs without any ML dependencies (uses synthetic float vectors).
"""
import math
import time
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sulci.context import ContextWindow, SessionStore, Turn


# ── helpers ───────────────────────────────────────────────────────────────────

def unit(v: list[float]) -> list[float]:
    """L2-normalise a vector."""
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

DIM = 8

# A few orthogonal unit vectors to use as fake embeddings
VEC_DOCKER  = unit([1, 0, 0, 0, 0, 0, 0, 0])
VEC_BILLING = unit([0, 1, 0, 0, 0, 0, 0, 0])
VEC_QUERY   = unit([0.5, 0.5, 0, 0, 0, 0, 0, 0])   # equidistant


# ── ContextWindow ─────────────────────────────────────────────────────────────

class TestContextWindow:

    def test_empty_window_returns_query_vec(self):
        w = ContextWindow(max_turns=4)
        q = unit([1, 2, 3, 4, 5, 6, 7, 8])
        assert w.blend(q) == q

    def test_depth_zero_on_creation(self):
        w = ContextWindow()
        assert w.depth == 0

    def test_add_turn_increases_depth(self):
        w = ContextWindow(max_turns=4)
        w.add_turn("hello", role="user", embedding=VEC_DOCKER)
        assert w.depth == 1

    def test_max_turns_drops_oldest(self):
        w = ContextWindow(max_turns=2)
        w.add_turn("turn 1", role="user", embedding=VEC_DOCKER)
        w.add_turn("turn 2", role="user", embedding=VEC_BILLING)
        w.add_turn("turn 3", role="user", embedding=VEC_QUERY)
        assert w.depth == 2
        assert w.turns[0].text == "turn 2"
        assert w.turns[1].text == "turn 3"

    def test_clear_resets_turns(self):
        w = ContextWindow(max_turns=4)
        w.add_turn("a", role="user", embedding=VEC_DOCKER)
        w.clear()
        assert w.depth == 0

    def test_blend_pulls_toward_history(self):
        """
        With Docker history, blending an equidistant query vec should
        move the result closer to Docker than to Billing.
        """
        w = ContextWindow(max_turns=4, query_weight=0.70, decay=0.5)
        w.add_turn("Docker issue", role="user", embedding=VEC_DOCKER)

        blended = w.blend(VEC_QUERY)
        sim_docker  = cosine(blended, VEC_DOCKER)
        sim_billing = cosine(blended, VEC_BILLING)
        assert sim_docker > sim_billing, (
            f"Expected blended vec to be closer to Docker history "
            f"(sim_docker={sim_docker:.3f}, sim_billing={sim_billing:.3f})"
        )

    def test_blend_is_normalised(self):
        w = ContextWindow(max_turns=4)
        w.add_turn("x", role="user", embedding=VEC_DOCKER)
        blended = w.blend(VEC_QUERY)
        norm = math.sqrt(sum(v * v for v in blended))
        assert abs(norm - 1.0) < 1e-6, f"Blended vector not unit norm: {norm}"

    def test_query_weight_1_ignores_history(self):
        """query_weight=1.0 should return the original query vec."""
        w = ContextWindow(max_turns=4, query_weight=1.0)
        w.add_turn("history", role="user", embedding=VEC_BILLING)
        blended = w.blend(VEC_DOCKER)
        # Should be identical to VEC_DOCKER (already normalised)
        assert all(abs(a - b) < 1e-6 for a, b in zip(blended, VEC_DOCKER))

    def test_query_weight_0_dominated_by_history(self):
        """query_weight=0.0 should return only the history vector."""
        w = ContextWindow(max_turns=4, query_weight=0.0)
        w.add_turn("history", role="user", embedding=VEC_BILLING)
        blended = w.blend(VEC_DOCKER)
        sim_billing = cosine(blended, VEC_BILLING)
        assert sim_billing > 0.99, f"Expected history to dominate: {sim_billing}"

    def test_role_filter_excludes_assistant(self):
        """roles=('user',) means assistant turns don't affect blending."""
        w = ContextWindow(max_turns=4, query_weight=0.5, roles=("user",))
        w.add_turn("assistant said Billing", role="assistant", embedding=VEC_BILLING)
        blended = w.blend(VEC_DOCKER)
        # No user turns → should return query vec unchanged
        assert all(abs(a - b) < 1e-6 for a, b in zip(blended, VEC_DOCKER))

    def test_decay_weights_recent_more(self):
        """
        Two turns: old=Billing, recent=Docker.
        With decay=0.0 (only most recent matters), result should be near Docker.
        """
        w = ContextWindow(max_turns=4, query_weight=0.5, decay=0.0)
        w.add_turn("old billing turn",  role="user", embedding=VEC_BILLING)
        w.add_turn("recent docker turn", role="user", embedding=VEC_DOCKER)
        blended = w.blend(VEC_QUERY)
        sim_docker  = cosine(blended, VEC_DOCKER)
        sim_billing = cosine(blended, VEC_BILLING)
        assert sim_docker > sim_billing

    def test_lazy_embedding_via_embedder(self):
        """Turns added without embeddings are lazily embedded when blending."""
        class MockEmbedder:
            def embed(self, text):
                return VEC_DOCKER   # always return Docker vec

        w = ContextWindow(max_turns=4, query_weight=0.5)
        w.add_turn("needs lazy embed", role="user")  # no embedding provided
        blended = w.blend(VEC_QUERY, embedder=MockEmbedder())
        sim_docker = cosine(blended, VEC_DOCKER)
        assert sim_docker > 0.5   # history pulled toward Docker

    def test_last_user_query(self):
        w = ContextWindow()
        assert w.last_user_query() is None
        w.add_turn("first",  role="user")
        w.add_turn("answer", role="assistant")
        w.add_turn("second", role="user")
        assert w.last_user_query() == "second"

    def test_summary_structure(self):
        w = ContextWindow(max_turns=3)
        w.add_turn("hello", role="user", embedding=VEC_DOCKER)
        s = w.summary()
        assert s["depth"]     == 1
        assert s["max_turns"] == 3
        assert len(s["turns"]) == 1
        assert s["turns"][0]["role"] == "user"
        assert s["turns"][0]["has_embed"] is True

    def test_repr(self):
        w = ContextWindow(max_turns=5, query_weight=0.8, decay=0.4)
        r = repr(w)
        assert "ContextWindow" in r
        assert "0/5" in r


# ── SessionStore ──────────────────────────────────────────────────────────────

class TestSessionStore:

    def test_creates_new_window_on_get(self):
        store = SessionStore(max_turns=4)
        w = store.get("session-1")
        assert isinstance(w, ContextWindow)
        assert w.depth == 0

    def test_same_session_id_returns_same_window(self):
        store = SessionStore(max_turns=4)
        w1 = store.get("session-A")
        w1.add_turn("hello", role="user", embedding=VEC_DOCKER)
        w2 = store.get("session-A")
        assert w2.depth == 1   # same object

    def test_different_sessions_are_isolated(self):
        store = SessionStore(max_turns=4)
        a = store.get("session-A")
        b = store.get("session-B")
        a.add_turn("A's turn", role="user", embedding=VEC_DOCKER)
        assert b.depth == 0   # B is unaffected

    def test_delete_removes_session(self):
        store = SessionStore(max_turns=4)
        store.get("x").add_turn("hello", role="user", embedding=VEC_DOCKER)
        store.delete("x")
        # After delete, a fresh window is returned
        assert store.get("x").depth == 0

    def test_clear_all(self):
        store = SessionStore(max_turns=4)
        for sid in ["a", "b", "c"]:
            store.get(sid).add_turn("t", role="user", embedding=VEC_DOCKER)
        store.clear_all()
        assert store.active_sessions() == []

    def test_active_sessions_list(self):
        store = SessionStore(max_turns=4)
        store.get("s1")
        store.get("s2")
        active = store.active_sessions()
        assert "s1" in active
        assert "s2" in active

    def test_ttl_eviction(self):
        store = SessionStore(max_turns=4, ttl_seconds=1)
        store.get("old")
        # Manually backdate last_active
        store._last_active["old"] = time.time() - 10
        # Trigger eviction
        store._evict_stale()
        assert "old" not in store.active_sessions()

    def test_no_eviction_when_ttl_none(self):
        store = SessionStore(max_turns=4, ttl_seconds=None)
        store.get("session-X")
        store._last_active["session-X"] = 0   # epoch = ancient
        store._evict_stale()
        assert "session-X" in store.active_sessions()

    def test_summary_structure(self):
        store = SessionStore(max_turns=4, ttl_seconds=3600)
        store.get("s1").add_turn("hi", role="user", embedding=VEC_DOCKER)
        s = store.summary()
        assert s["active_sessions"] == 1
        assert "s1" in s["sessions"]

    def test_repr(self):
        store = SessionStore(ttl_seconds=1800)
        r = repr(store)
        assert "SessionStore" in r
        assert "1800" in r


# ── Integration: Cache with context_window ────────────────────────────────────

class TestCacheContextIntegration:
    """
    End-to-end tests using sulci.Cache with the SQLite backend.
    Uses real sentence embeddings only if available; skips otherwise.
    """

    @pytest.fixture
    def ctx_cache(self, tmp_path):
        try:
            from sulci import Cache
            return Cache(
                backend        = "sqlite",
                threshold      = 0.80,
                context_window = 4,
                query_weight   = 0.70,
                db_path        = str(tmp_path / "test_ctx_db"),
                ttl_seconds    = None,
            )
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_stateless_when_no_session_id(self, ctx_cache):
        def llm(q): return f"answer:{q}"
        r = ctx_cache.cached_call("What is Python?", llm)
        assert r["context_depth"] == 0

    def test_context_depth_zero_on_first_turn(self, ctx_cache):
        def llm(q): return f"answer:{q}"
        r = ctx_cache.cached_call("First question", llm, session_id="s1")
        assert r["context_depth"] == 0

    def test_context_depth_increases_on_follow_up(self, ctx_cache):
        def llm(q): return f"answer:{q}"
        ctx_cache.cached_call("My Docker container is crashing", llm, session_id="s2")
        r = ctx_cache.cached_call("What should I check?", llm, session_id="s2")
        assert r["context_depth"] >= 1

    def test_sessions_are_isolated(self, ctx_cache):
        def llm(q): return f"answer:{q}"
        ctx_cache.cached_call("Topic A question", llm, session_id="sa")
        r = ctx_cache.cached_call("What do you think?", llm, session_id="sb")
        # Session B has no history → depth 0
        assert r["context_depth"] == 0

    def test_clear_context_resets_depth(self, ctx_cache):
        def llm(q): return f"answer:{q}"
        ctx_cache.cached_call("Question 1", llm, session_id="sc")
        ctx_cache.cached_call("Question 2", llm, session_id="sc")
        ctx_cache.clear_context("sc")
        r = ctx_cache.cached_call("What now?", llm, session_id="sc")
        assert r["context_depth"] == 0

    def test_get_context_raises_when_disabled(self, tmp_path):
        from sulci import Cache
        try:
            c = Cache(
                backend        = "sqlite",
                context_window = 0,
                db_path        = str(tmp_path / "nocontext_db"),
            )
            with pytest.raises(RuntimeError, match="context_window"):
                c.get_context("any-session")
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_manual_context_injection(self, ctx_cache):
        ctx = ctx_cache.get_context("sd")
        ctx.add_turn("I am building a Kubernetes cluster", role="user")
        assert ctx.depth == 1
        assert ctx.last_user_query() == "I am building a Kubernetes cluster"

    def test_result_has_context_depth_key(self, ctx_cache):
        def llm(q): return f"answer:{q}"
        r = ctx_cache.cached_call("Any question", llm, session_id="se")
        assert "context_depth" in r

    def test_context_summary_returns_dict(self, ctx_cache):
        def llm(q): return f"answer:{q}"
        ctx_cache.cached_call("Setup question", llm, session_id="sf")
        s = ctx_cache.context_summary("sf")
        assert isinstance(s, dict)
        assert "depth" in s

    def test_context_summary_all_sessions(self, ctx_cache):
        def llm(q): return f"answer:{q}"
        ctx_cache.cached_call("Q", llm, session_id="sg1")
        ctx_cache.cached_call("Q", llm, session_id="sg2")
        s = ctx_cache.context_summary()
        assert s["active_sessions"] >= 2
