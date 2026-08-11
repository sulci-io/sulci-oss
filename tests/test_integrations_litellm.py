# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
tests/test_integrations_litellm.py
───────────────────────────────────
Covers sulci.integrations.litellm. Runs offline via FakeEmbedder.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from sulci import Cache
from tests._fake_embedder import FakeEmbedder

ll = pytest.importorskip(
    "sulci.integrations.litellm",
    reason='litellm not installed; pip install "sulci[litellm]"',
)
SulciLiteLLMCache = ll.SulciLiteLLMCache


@pytest.fixture
def cache(tmp_path):
    return Cache(
        backend="sqlite",
        db_path=str(tmp_path / "ll"),
        embedding_model=FakeEmbedder(),
        threshold=0.85,
        telemetry=False,
    )


@pytest.fixture
def adapter(cache):
    return SulciLiteLLMCache(cache, namespace_by_model=False)


MESSAGES = [{"role": "user", "content": "What is semantic caching?"}]
VALUE = {"choices": [{"message": {"role": "assistant", "content": "It caches by meaning."}}]}


# ── contract conformance ─────────────────────────────────────────────────
def test_is_a_litellm_base_cache(adapter):
    from litellm.caching.base_cache import BaseCache

    assert isinstance(adapter, BaseCache)


def test_implements_every_abstract_method(adapter):
    for name in ("get_cache", "set_cache", "async_get_cache", "async_set_cache"):
        assert callable(getattr(adapter, name))


def test_litellm_still_has_no_custom_cache_type():
    # The module docstring asserts this. If LiteLLM ever adds one, this test
    # fails and the adapter should switch to the first-class mechanism
    # instead of monkey-patching `litellm.cache.cache`.
    from litellm.types.caching import LiteLLMCacheType

    assert "custom" not in {e.value for e in LiteLLMCacheType}


# ── prompt extraction ────────────────────────────────────────────────────
def test_prompt_from_chat_messages():
    assert ll._prompt_from_kwargs(messages=MESSAGES) == "What is semantic caching?"


def test_prompt_from_content_blocks():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert ll._prompt_from_kwargs(messages=msgs) == "hello"


def test_prompt_from_responses_api_input():
    assert ll._prompt_from_kwargs(input="summarise this") == "summarise this"


def test_prompt_is_none_when_there_is_nothing_to_key_on():
    assert ll._prompt_from_kwargs() is None
    assert ll._prompt_from_kwargs(messages=[]) is None


# ── round trip ───────────────────────────────────────────────────────────
def test_miss_returns_none(adapter):
    assert adapter.get_cache("k", messages=MESSAGES) is None


def test_set_then_get_round_trips_a_dict(adapter):
    adapter.set_cache("k", VALUE, messages=MESSAGES)
    assert adapter.get_cache("k", messages=MESSAGES) == VALUE


def test_set_then_get_round_trips_a_string(adapter):
    adapter.set_cache("k", "plain text", messages=MESSAGES)
    assert adapter.get_cache("k", messages=MESSAGES) == "plain text"


def test_semantically_unrelated_prompt_misses(adapter):
    adapter.set_cache("k", VALUE, messages=MESSAGES)
    other = [{"role": "user", "content": "zebra migration patterns"}]
    assert adapter.get_cache("k", messages=other) is None


def test_no_prompt_means_no_write(adapter):
    adapter.set_cache("k", VALUE)  # no messages
    assert adapter.get_cache("k", messages=MESSAGES) is None


def test_similarity_is_surfaced_in_metadata_like_litellms_own_caches(adapter):
    adapter.set_cache("k", VALUE, messages=MESSAGES)
    meta: dict = {}
    adapter.get_cache("k", messages=MESSAGES, metadata=meta)
    assert meta["semantic-similarity"] >= 0.85


def test_similarity_is_zero_on_a_miss(adapter):
    meta: dict = {}
    adapter.get_cache("k", messages=MESSAGES, metadata=meta)
    assert meta["semantic-similarity"] == 0.0


def test_unserialisable_value_is_dropped_not_raised(adapter):
    adapter.set_cache("k", {"fn": lambda: 1}, messages=MESSAGES)
    # json.dumps(default=str) stringifies the lambda rather than raising, so
    # this stores; the contract that matters is that it does not explode.
    assert adapter.get_cache("k", messages=MESSAGES) is not None


# ── async parity ─────────────────────────────────────────────────────────
def test_async_round_trip(adapter):
    async def go():
        await adapter.async_set_cache("k", VALUE, messages=MESSAGES)
        return await adapter.async_get_cache("k", messages=MESSAGES)

    assert asyncio.run(go()) == VALUE


def test_async_pipeline_writes_every_entry(cache):
    a = SulciLiteLLMCache(cache, namespace_by_model=False)

    async def go():
        await a.async_set_cache_pipeline(
            [("k1", VALUE)], messages=MESSAGES
        )
        return await a.async_get_cache("k1", messages=MESSAGES)

    assert asyncio.run(go()) == VALUE


# ── scoping ──────────────────────────────────────────────────────────────
def test_namespace_by_model_warns_on_a_backend_that_ignores_it(cache):
    from sulci.integrations._scope import ScopeNotEnforcedWarning

    with pytest.warns(ScopeNotEnforcedWarning):
        SulciLiteLLMCache(cache, namespace_by_model=True)


def test_namespace_off_does_not_warn(cache, recwarn):
    from sulci.integrations._scope import ScopeNotEnforcedWarning

    SulciLiteLLMCache(cache, namespace_by_model=False)
    assert not [w for w in recwarn if w.category is ScopeNotEnforcedWarning]


def test_session_key_is_read_from_metadata(adapter):
    adapter.set_cache(
        "k", VALUE, messages=MESSAGES, metadata={"sulci_session_id": "run-1"}
    )
    assert (
        adapter.get_cache(
            "k", messages=MESSAGES, metadata={"sulci_session_id": "run-1"}
        )
        == VALUE
    )


# ── failure isolation ────────────────────────────────────────────────────
class _Exploding:
    _backend = None

    def get(self, *a, **k):
        raise RuntimeError("down")

    def set(self, *a, **k):
        raise RuntimeError("down")

    def stats(self):
        return {}


def test_lookup_failure_degrades_to_a_miss():
    a = SulciLiteLLMCache(_Exploding(), namespace_by_model=False)
    assert a.get_cache("k", messages=MESSAGES) is None


def test_store_failure_is_swallowed():
    a = SulciLiteLLMCache(_Exploding(), namespace_by_model=False)
    a.set_cache("k", VALUE, messages=MESSAGES)  # must not raise


# ── wiring ───────────────────────────────────────────────────────────────
def test_install_replaces_the_inner_cache_implementation(cache):
    import litellm

    previous = getattr(litellm, "cache", None)
    try:
        adapter = ll.install(cache, namespace_by_model=False)
        assert litellm.cache.cache is adapter
    finally:
        litellm.cache = previous


def test_constructor_rejects_cache_and_kwargs_together(cache):
    with pytest.raises(TypeError):
        SulciLiteLLMCache(cache, backend="sqlite")


def test_stats_passes_through(adapter):
    adapter.set_cache("k", VALUE, messages=MESSAGES)
    adapter.get_cache("k", messages=MESSAGES)
    assert adapter.stats()["hits"] == 1


# ── end-to-end through litellm.completion ────────────────────────────────
# WHY THESE EXIST (added 2026-08-11, after the fact)
#
# Every test above calls adapter.get_cache/.set_cache DIRECTLY. That verifies
# the adapter honours BaseCache, and it verified nothing about whether LiteLLM
# ever CALLS it. The gap was invisible until examples/litellm_example.py
# printed hits=0, misses=0, total_queries=0 while exiting 0 -- a mock that
# replaced litellm.completion had removed the caching path wholesale, because
# the cache is consulted INSIDE that wrapper.
#
# A direct-call test suite cannot catch that. These go through the real
# litellm.completion using its own mock_response kwarg, so the wrapper, the
# cache lookup and the store are all exercised without a network call.

def _completion(messages, **kw):
    import litellm

    return litellm.completion(
        model=kw.pop("model", "gpt-4o-mini"),
        messages=messages,
        mock_response=kw.pop("mock_response", "Cached answer."),
        **kw,
    )


@pytest.fixture
def installed(cache):
    import litellm

    previous = getattr(litellm, "cache", None)
    adapter = ll.install(cache, namespace_by_model=False)
    yield adapter
    litellm.cache = previous


def test_completion_populates_the_cache_on_a_miss(installed):
    _completion(MESSAGES)
    stats = installed.stats()
    assert stats["total_queries"] == 1 and stats["misses"] == 1


def test_second_identical_completion_is_a_cache_hit(installed):
    _completion(MESSAGES)
    _completion(MESSAGES)
    stats = installed.stats()
    assert stats["hits"] == 1, f"cache not consulted by litellm: {stats}"
    assert stats["total_queries"] == 2


def test_cache_is_actually_consulted_at_all(installed):
    # The regression that motivated this block: total_queries stuck at 0.
    _completion(MESSAGES)
    assert installed.stats()["total_queries"] > 0


def test_different_prompt_does_not_hit(installed):
    _completion(MESSAGES)
    _completion([{"role": "user", "content": "zebra migration patterns"}])
    assert installed.stats()["hits"] == 0
