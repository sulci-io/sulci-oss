# SPDX-License-Identifier: Apache-2.0
"""
tests/test_integrations_langchain.py
──────────────────────────────────────
Tests for sulci.integrations.langchain (SulciCache).

No real LLM API keys required.  All tests use the SQLite backend with
the tmp_path fixture so each test gets a fresh, isolated cache.

Run from repo root (sulci-oss/):
    python -m pytest tests/test_integrations_langchain.py -v
"""
import asyncio
from unittest.mock import patch

import pytest

# Skip the entire module if langchain-core is not installed.
langchain_core = pytest.importorskip(
    "langchain_core",
    reason="langchain-core not installed — run: pip install langchain-core",
)

from langchain_core.outputs import Generation

from sulci.integrations.langchain import SulciCache


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cache(tmp_path):
    """Single-partition SulciCache backed by SQLite."""
    return SulciCache(
        backend          = "sqlite",
        db_path          = str(tmp_path / "lc"),
        threshold        = 0.85,
        namespace_by_llm = False,
    )


@pytest.fixture
def cache_ns(tmp_path):
    """Per-model namespaced SulciCache."""
    return SulciCache(
        backend          = "sqlite",
        db_path          = str(tmp_path / "lc_ns"),
        threshold        = 0.85,
        namespace_by_llm = True,
    )


LLM   = "openai::gpt-4o::temp=0.0"
QUERY = "What is semantic caching?"
RESP  = "Semantic caching stores LLM responses indexed by meaning."


# ─────────────────────────────────────────────────────────────────────────────
# Core contract: lookup / update / clear
# ─────────────────────────────────────────────────────────────────────────────

class TestContract:

    def test_miss_on_empty_cache(self, cache):
        assert cache.lookup(QUERY, LLM) is None

    def test_update_then_lookup_hit(self, cache):
        cache.update(QUERY, LLM, [Generation(text=RESP)])
        result = cache.lookup(QUERY, LLM)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Generation)
        assert result[0].text == RESP

    def test_lookup_returns_list_of_generation(self, cache):
        # LangChain specifically requires list[Generation], not a raw string
        cache.update(QUERY, LLM, [Generation(text=RESP)])
        result = cache.lookup(QUERY, LLM)
        assert isinstance(result, list)
        assert isinstance(result[0], Generation)

    def test_clear_removes_entries(self, cache):
        cache.update(QUERY, LLM, [Generation(text=RESP)])
        cache.clear()
        assert cache.lookup(QUERY, LLM) is None

    def test_update_empty_return_val_is_noop(self, cache):
        cache.update(QUERY, LLM, [])
        assert cache.lookup(QUERY, LLM) is None

    def test_only_first_generation_stored(self, cache):
        """LangChain may pass multiple Generations; Sulci stores only the first."""
        cache.update(QUERY, LLM, [Generation(text="first"), Generation(text="second")])
        result = cache.lookup(QUERY, LLM)
        assert result is not None
        assert result[0].text == "first"

    def test_multiple_prompts_stored_independently(self, cache):
        q2, r2 = "What is a vector database?", "A vector DB stores embeddings."
        cache.update(QUERY, LLM, [Generation(text=RESP)])
        cache.update(q2,    LLM, [Generation(text=r2)])
        assert cache.lookup(QUERY, LLM)[0].text == RESP
        assert cache.lookup(q2,    LLM)[0].text == r2

    def test_exact_prompt_always_hits(self, cache):
        cache.update(QUERY, LLM, [Generation(text=RESP)])
        assert cache.lookup(QUERY, LLM) is not None

    def test_unrelated_prompt_misses(self, cache):
        cache.update(QUERY, LLM, [Generation(text=RESP)])
        result = cache.lookup("What is the current price of Bitcoin?", LLM)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Per-model namespace isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestNamespacing:

    def test_different_llm_strings_are_isolated(self, cache_ns):
        llm_a = "openai::gpt-4o"
        llm_b = "anthropic::claude-3-5-sonnet"
        cache_ns.update(QUERY, llm_a, [Generation(text="GPT answer")])
        cache_ns.update(QUERY, llm_b, [Generation(text="Claude answer")])
        assert cache_ns.lookup(QUERY, llm_a)[0].text == "GPT answer"
        assert cache_ns.lookup(QUERY, llm_b)[0].text == "Claude answer"

    def test_same_llm_string_reuses_partition(self, cache_ns):
        cache_ns.update(QUERY, LLM, [Generation(text=RESP)])
        assert cache_ns.lookup(QUERY, LLM) is not None

    def test_namespace_false_shares_across_models(self, tmp_path):
        c = SulciCache(
            backend="sqlite",
            db_path=str(tmp_path / "shared"),
            namespace_by_llm=False,
        )
        c.update(QUERY, "model-a", [Generation(text=RESP)])
        # Same entry visible regardless of llm_string
        assert c.lookup(QUERY, "model-b") is not None

    def test_clear_removes_all_partitions(self, cache_ns):
        cache_ns.update(QUERY, "gpt-4o",     [Generation(text="A")])
        cache_ns.update(QUERY, "claude-3-5", [Generation(text="B")])
        cache_ns.clear()
        # MUST check len BEFORE any lookup — lookup calls _cache_for() which
        # recreates namespace entries for any llm_string it encounters.
        assert len(cache_ns._ns_caches) == 0
        # Data is gone — lookups return None (may recreate empty partitions)
        assert cache_ns.lookup(QUERY, "gpt-4o")     is None
        assert cache_ns.lookup(QUERY, "claude-3-5") is None


# ─────────────────────────────────────────────────────────────────────────────
# Silent failure — cache errors must never crash the caller's app
# ─────────────────────────────────────────────────────────────────────────────

class TestSilentFailure:

    def test_lookup_error_returns_none(self, cache):
        with patch.object(
            cache._default_cache, "get", side_effect=RuntimeError("db gone")
        ):
            assert cache.lookup(QUERY, LLM) is None  # miss, not exception

    def test_update_error_is_silent(self, cache):
        with patch.object(
            cache._default_cache, "set", side_effect=RuntimeError("db gone")
        ):
            cache.update(QUERY, LLM, [Generation(text=RESP)])  # no exception

    def test_clear_error_propagates(self, cache):
        """0.9.3: clear() is an erasure, not a cache read: its failure is raised, not logged and
        ignored (on the managed backend, SulciErasureError when the gateway does not confirm it)."""
        with patch.object(
            cache._default_cache, "clear", side_effect=RuntimeError("db gone")
        ):
            with pytest.raises(RuntimeError, match="db gone"):
                cache.clear()

    def test_a_partition_that_fails_to_clear_is_kept(self, cache_ns):
        cache_ns.update(QUERY, "gpt-4o",     [Generation(text="A")])
        cache_ns.update(QUERY, "claude-3-5", [Generation(text="B")])
        failing = cache_ns._ns_caches["claude-3-5"]
        with patch.object(failing, "clear", side_effect=RuntimeError("db gone")):
            with pytest.raises(RuntimeError, match="db gone"):
                cache_ns.clear()
        assert list(cache_ns._ns_caches) == ["claude-3-5"] and cache_ns._ns_caches["claude-3-5"] is failing


# ─────────────────────────────────────────────────────────────────────────────
# Async variants
# ─────────────────────────────────────────────────────────────────────────────

class TestAsync:

    @pytest.mark.asyncio
    async def test_alookup_miss(self, cache):
        assert await cache.alookup(QUERY, LLM) is None

    @pytest.mark.asyncio
    async def test_aupdate_then_alookup_hit(self, cache):
        await cache.aupdate(QUERY, LLM, [Generation(text=RESP)])
        result = await cache.alookup(QUERY, LLM)
        assert result is not None
        assert result[0].text == RESP

    @pytest.mark.asyncio
    async def test_aclear(self, cache):
        await cache.aupdate(QUERY, LLM, [Generation(text=RESP)])
        await cache.aclear()
        assert await cache.alookup(QUERY, LLM) is None

    @pytest.mark.asyncio
    async def test_concurrent_lookups_no_crash(self, cache):
        """Concurrent alookup calls must not raise exceptions.

        The SQLite backend uses a single connection. Under high concurrency some
        reads may return None (miss) due to connection contention — that is
        acceptable. What must never happen is an unhandled exception.
        """
        cache.update(QUERY, LLM, [Generation(text=RESP)])
        results = await asyncio.gather(
            *[cache.alookup(QUERY, LLM) for _ in range(20)],
            return_exceptions=True,
        )
        # No call should raise an exception — None (miss) is fine
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Got exceptions: {exceptions[:3]}"
        # At least one of the 20 calls should have hit the cache
        hits = [r for r in results if r is not None]
        assert len(hits) > 0, "Expected at least one cache hit among 20 concurrent reads"


# ─────────────────────────────────────────────────────────────────────────────
# Stats and repr
# ─────────────────────────────────────────────────────────────────────────────

class TestStats:

    def test_stats_is_dict(self, cache):
        assert isinstance(cache.stats(), dict)

    def test_stats_has_required_keys(self, cache):
        s = cache.stats()
        for k in ("hits", "misses", "hit_rate", "total_queries", "saved_cost"):
            assert k in s, f"stats() missing key: {k}"

    def test_repr_contains_class_name_and_hit_rate(self, cache):
        r = repr(cache)
        assert "SulciCache" in r
        assert "hit_rate"   in r


# ─────────────────────────────────────────────────────────────────────────────
# LangChain global registration
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalRegistration:

    def test_set_and_get_llm_cache(self, cache):
        """SulciCache can be registered as the global LangChain cache."""
        from langchain_core.globals import get_llm_cache, set_llm_cache
        set_llm_cache(cache)
        assert get_llm_cache() is cache
        set_llm_cache(None)   # restore — don't leak between tests


# ─────────────────────────────────────────────────────────────────────────────
# namespace_by_llm warning when backend="sulci"
# ─────────────────────────────────────────────────────────────────────────────

class TestNamespaceByLLMCloudWarning:
    """
    namespace_by_llm=True should warn + be disabled when backend='sulci'.
    Uses patch to avoid needing a real Sulci Cloud API key.
    """

    @pytest.fixture
    def mock_sulci_cache(self):
        """Patch sulci.Cache so SulciCache.__init__ never touches real backends."""
        from unittest.mock import MagicMock
        mock_instance = MagicMock()
        mock_instance.get.return_value = (None, 0.0, 0)
        mock_instance.stats.return_value = {
            "hits": 0, "misses": 0, "hit_rate": 0.0,
            "saved_cost": 0.0, "total_queries": 0,
        }
        with patch("sulci.Cache", return_value=mock_instance):
            yield mock_instance

    def test_warns_and_disables_when_backend_sulci(self, caplog, mock_sulci_cache):
        """namespace_by_llm=True with backend='sulci' logs warning + sets False."""
        import logging
        with caplog.at_level(logging.WARNING, logger="sulci.integrations.langchain"):
            cache = SulciCache(backend="sulci", api_key="sk-test")
        assert "namespace_by_llm=True has no effect" in caplog.text
        assert cache._namespace_by_llm is False

    def test_no_warning_when_namespace_by_llm_false(self, caplog, mock_sulci_cache):
        """namespace_by_llm=False with backend='sulci' produces no warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger="sulci.integrations.langchain"):
            cache = SulciCache(backend="sulci", api_key="sk-test", namespace_by_llm=False)
        assert "namespace_by_llm" not in caplog.text
        assert cache._namespace_by_llm is False

    def test_no_warning_for_non_sulci_backend(self, caplog, mock_sulci_cache):
        """namespace_by_llm=True with a local backend produces no warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger="sulci.integrations.langchain"):
            cache = SulciCache(backend="sqlite")
        assert "namespace_by_llm" not in caplog.text
        assert cache._namespace_by_llm is True
