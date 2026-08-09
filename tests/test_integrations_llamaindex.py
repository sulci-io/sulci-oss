# SPDX-License-Identifier: Apache-2.0
"""
tests/test_integrations_llamaindex.py
───────────────────────────────────────
Tests for sulci.integrations.llamaindex (SulciCacheLLM).

No real LLM API keys required. All tests use a MockLLM and the SQLite
backend with tmp_path so each test gets a fresh, isolated cache.

Run from repo root (sulci-oss/):
    python -m pytest tests/test_integrations_llamaindex.py -v
"""
import sys
from typing import Any, Sequence
from unittest.mock import patch

import pytest

# Python 3.9: llama-index-core pulls `banks`, which declares
# requires_python>=3.9 but evaluates `Path | None` in a class body (PEP 604,
# 3.10+). It raises TypeError at IMPORT, which importorskip does not catch, so
# collection of this whole file aborts and takes the LangChain suite with it.
# sulci/integrations/llamaindex.py raises a legible ImportError for this case;
# this skip records the same limitation rather than hiding it.
# Measured 2026-08-09 on CI 3.9: ubuntu, macos and windows, all three.
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="sulci[llamaindex] requires Python 3.10+ (banks uses 3.10 syntax)",
)

# Skip the entire module if llama-index-core is not installed.
llamaindex_core = pytest.importorskip(
    "llama_index.core",
    reason="llama-index-core not installed — run: pip install llama-index-core",
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms.llm import LLM

from sulci.integrations.llamaindex import SulciCacheLLM


# ─────────────────────────────────────────────────────────────────────────────
# External call log — survives Pydantic's internal copy of the LLM instance.
# Each test resets it via the `call_log` fixture.
# ─────────────────────────────────────────────────────────────────────────────

_call_log: list = []


@pytest.fixture(autouse=True)
def call_log():
    """Reset call log before each test and return it for inspection."""
    _call_log.clear()
    yield _call_log
    _call_log.clear()


def n_calls() -> int:
    return len(_call_log)


# ─────────────────────────────────────────────────────────────────────────────
# MockLLM — no API keys needed
# ─────────────────────────────────────────────────────────────────────────────

class MockLLM(LLM):
    """Minimal LLM implementation for tests — returns predictable responses."""

    model_name: str = Field(default="mock-llm")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name)

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        _call_log.append("complete")
        return CompletionResponse(text=f"complete:{prompt}")

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        _call_log.append("chat")
        last = messages[-1].content if messages else ""
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=f"chat:{last}")
        )

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        _call_log.append("stream_complete")
        def gen():
            yield CompletionResponse(text=f"stream:{prompt}", delta=f"stream:{prompt}")
        return gen()

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        _call_log.append("stream_chat")
        def gen():
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="stream_chat"),
                delta="stream_chat",
            )
        return gen()

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, formatted, **kwargs)

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        async def gen():
            yield CompletionResponse(text=f"astream:{prompt}", delta=f"astream:{prompt}")
        return gen()

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        async def gen():
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="astream_chat"),
                delta="astream_chat",
            )
        return gen()


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cache(tmp_path):
    """SulciCacheLLM backed by SQLite in a fresh temp directory."""
    return SulciCacheLLM(
        llm       = MockLLM(),
        backend   = "sqlite",
        db_path   = str(tmp_path / "test_cache"),
        threshold = 0.85,
    )


PROMPT   = "What is semantic caching?"
RESPONSE = f"complete:{PROMPT}"
USER_MSG = ChatMessage(role=MessageRole.USER, content=PROMPT)


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruction:

    def test_wraps_llm(self, cache):
        assert isinstance(cache.llm, MockLLM)

    def test_metadata_delegates_to_wrapped_llm(self, cache):
        assert cache.metadata.model_name == "mock-llm"

    def test_sulci_kwargs_separated_from_llm_kwargs(self, tmp_path):
        c = SulciCacheLLM(
            llm       = MockLLM(),
            backend   = "sqlite",
            db_path   = str(tmp_path / "c2"),
            threshold = 0.75,
        )
        assert c._sulci_kwargs["threshold"] == 0.75
        assert c._sulci_kwargs["backend"]   == "sqlite"

    def test_cache_is_lazily_constructed(self, cache):
        assert cache._sulci_cache is None

    def test_get_cache_constructs_on_first_call(self, cache):
        c = cache._get_cache()
        assert c is not None
        assert cache._sulci_cache is c

    def test_get_cache_returns_same_instance(self, cache):
        assert cache._get_cache() is cache._get_cache()


# ─────────────────────────────────────────────────────────────────────────────
# complete() — cache hit / miss
# ─────────────────────────────────────────────────────────────────────────────

class TestComplete:

    def test_miss_calls_llm(self, cache):
        result = cache.complete(PROMPT)
        assert result.text == RESPONSE
        assert n_calls() == 1

    def test_hit_skips_llm(self, cache):
        cache.complete(PROMPT)          # miss — populates cache
        _call_log.clear()
        result = cache.complete(PROMPT) # hit — should not call LLM
        assert result.text == RESPONSE
        assert n_calls() == 0

    def test_returns_completion_response(self, cache):
        result = cache.complete(PROMPT)
        assert isinstance(result, CompletionResponse)
        assert result.text

    def test_different_prompts_cached_independently(self, cache):
        p2 = "What is a vector database?"
        cache.complete(PROMPT)
        cache.complete(p2)
        assert n_calls() == 2

    def test_cache_error_falls_through_to_llm(self, cache):
        with patch.object(cache._get_cache(), "get", side_effect=RuntimeError("db error")):
            result = cache.complete(PROMPT)
        assert result.text == RESPONSE
        assert n_calls() == 1


# ─────────────────────────────────────────────────────────────────────────────
# chat() — cache hit / miss
# ─────────────────────────────────────────────────────────────────────────────

class TestChat:

    def test_miss_calls_llm(self, cache):
        result = cache.chat([USER_MSG])
        assert n_calls() == 1
        assert result.message.content == f"chat:{PROMPT}"

    def test_hit_skips_llm(self, cache):
        cache.chat([USER_MSG])
        _call_log.clear()
        result = cache.chat([USER_MSG])
        assert n_calls() == 0
        assert result.message.content == f"chat:{PROMPT}"

    def test_returns_chat_response(self, cache):
        result = cache.chat([USER_MSG])
        assert isinstance(result, ChatResponse)
        assert result.message.role == MessageRole.ASSISTANT

    def test_uses_last_user_message_as_key(self, cache):
        """System message prefix must not create a different cache entry."""
        sys_msg       = ChatMessage(role=MessageRole.SYSTEM, content="You are helpful.")
        msgs_with_sys = [sys_msg, USER_MSG]
        cache.chat([USER_MSG])      # populate cache
        _call_log.clear()
        cache.chat(msgs_with_sys)   # same last user message — should hit
        assert n_calls() == 0

    def test_cache_error_falls_through_to_llm(self, cache):
        with patch.object(cache._get_cache(), "get", side_effect=RuntimeError("db error")):
            result = cache.chat([USER_MSG])
        assert result.message.content == f"chat:{PROMPT}"


# ─────────────────────────────────────────────────────────────────────────────
# Streaming — pass-through uncached
# ─────────────────────────────────────────────────────────────────────────────

class TestStreaming:

    def test_stream_complete_passes_through(self, cache):
        gen    = cache.stream_complete(PROMPT)
        tokens = list(gen)
        assert len(tokens) > 0
        assert tokens[0].text == f"stream:{PROMPT}"

    def test_stream_complete_calls_llm(self, cache):
        list(cache.stream_complete(PROMPT))
        assert n_calls() == 1

    def test_stream_chat_passes_through(self, cache):
        gen    = cache.stream_chat([USER_MSG])
        tokens = list(gen)
        assert len(tokens) > 0

    def test_stream_does_not_populate_cache(self, cache):
        """Streaming must not write to cache — subsequent complete() must be a miss."""
        list(cache.stream_complete(PROMPT))
        _call_log.clear()
        cache.complete(PROMPT)   # same prompt — must be a miss (not cached by stream)
        assert n_calls() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Async
# ─────────────────────────────────────────────────────────────────────────────

class TestAsync:

    @pytest.mark.asyncio
    async def test_acomplete_miss(self, cache):
        result = await cache.acomplete(PROMPT)
        assert result.text == RESPONSE
        assert n_calls() == 1

    @pytest.mark.asyncio
    async def test_acomplete_hit(self, cache):
        await cache.acomplete(PROMPT)
        _call_log.clear()
        result = await cache.acomplete(PROMPT)
        assert n_calls() == 0
        assert result.text == RESPONSE

    @pytest.mark.asyncio
    async def test_achat_miss(self, cache):
        result = await cache.achat([USER_MSG])
        assert n_calls() == 1

    @pytest.mark.asyncio
    async def test_astream_complete_passes_through(self, cache):
        gen    = await cache.astream_complete(PROMPT)
        tokens = [r async for r in gen]
        assert len(tokens) > 0


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

    def test_repr_contains_class_name(self, cache):
        assert "SulciCacheLLM" in repr(cache)

    def test_repr_contains_model_name(self, cache):
        assert "mock-llm" in repr(cache)

    def test_repr_contains_hit_rate(self, cache):
        assert "hit_rate" in repr(cache)
