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
