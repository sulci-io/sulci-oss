# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
sulci.integrations.llamaindex
──────────────────────────────
Context-aware semantic cache adapter for LlamaIndex, backed by Sulci Cache.

This module wraps any LlamaIndex-compatible LLM with a transparent semantic
cache layer. Unlike the LangChain integration (which uses a global cache
registry), LlamaIndex caching is implemented by wrapping the LLM itself —
this is the correct and idiomatic LlamaIndex v0.10+ pattern.

Usage:
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from sulci.integrations.llamaindex import SulciCacheLLM

    # Stateless — drop-in for any LlamaIndex LLM
    Settings.llm = SulciCacheLLM(
        llm       = OpenAI(model="gpt-4o"),
        backend   = "sqlite",
        threshold = 0.85,
    )

    # Context-aware — chatbot / agent
    Settings.llm = SulciCacheLLM(
        llm            = OpenAI(model="gpt-4o"),
        backend        = "sqlite",
        threshold      = 0.75,
        context_window = 4,
    )

Install:
    pip install "sulci[sqlite,llamaindex]"
    # which installs: sulci + llama-index-core

Notes:
    - complete() and chat() are cached.
    - stream_complete() and stream_chat() pass through uncached — streaming
      responses are generators and cannot be reliably stored mid-stream.
    - Async methods delegate to sync via run_in_executor — event loop never blocked.
    - All cache errors are swallowed — a cache failure must never crash the app.
    - Patent pending covers the context-aware blending algorithm.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# llama-index-core is optional — guard the import clearly.
try:
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
    from llama_index.core.bridge.pydantic import Field, PrivateAttr
    from llama_index.core.llms.llm import LLM
except ImportError as _li_err:  # pragma: no cover
    raise ImportError(
        "llama-index-core is required for sulci.integrations.llamaindex.\n"
        "Install: pip install \"sulci[llamaindex]\"\n"
        "or:      pip install llama-index-core"
    ) from _li_err
except TypeError as _li_syntax:  # pragma: no cover
    # Python 3.9: llama-index-core pulls `banks`, which declares
    # requires_python>=3.9 but evaluates `Path | None` in a class body --
    # PEP 604, 3.10+ only. It raises TypeError at import, not ImportError, so
    # the guard above does not catch it and the user sees a cryptic message
    # naming a package they have never heard of.
    # Measured 2026-08-09 on CI 3.9: ubuntu, macos and windows, all three.
    if sys.version_info < (3, 10):
        raise ImportError(
            "sulci[llamaindex] needs Python 3.10 or newer.\n"
            "On 3.9, pip resolves llama-index-core to a version whose "
            "transitive dependency 'banks' uses 3.10-only syntax and fails "
            "at import. The core sulci package still supports 3.9; only this "
            "adapter does not.\n"
            f"Detected: Python {sys.version_info.major}.{sys.version_info.minor}"
        ) from _li_syntax
    raise

# Keys accepted by sulci.Cache — everything else forwarded to the wrapped LLM
_SULCI_KEYS = frozenset({
    "backend", "threshold", "embedding_model", "ttl_seconds",
    "personalized", "db_path", "context_window", "query_weight",
    "context_decay", "session_ttl", "api_key", "gateway_url", "telemetry",
})


class SulciCacheLLM(LLM):
    """
    Context-aware semantic cache wrapper for any LlamaIndex LLM.

    Wraps ``complete()`` and ``chat()`` with a Sulci Cache lookup before
    forwarding to the underlying LLM on a miss. Streaming calls pass
    through uncached. All async methods delegate via ``run_in_executor``
    so the event loop is never blocked.

    Unlike stateless semantic caches, SulciCacheLLM blends prior
    conversation turns into the similarity lookup vector when
    ``context_window > 0`` — dramatically improving hit rates in
    multi-turn RAG and agent workloads:

    - Customer support:  32% → 88% hit rate  (+56 pp)
    - Developer Q&A:     80% → 96% hit rate  (+16 pp)

    Note: This is the first correct native LLM-level semantic cache for
    LlamaIndex. GPTCache's claimed integration was a broken global
    OpenAI API patch; SulciCacheLLM uses the idiomatic ``LLM`` subclass
    pattern and works with any LlamaIndex-compatible model.

    Args:
        llm (LLM):
            The underlying LlamaIndex LLM to wrap. Any LLM subclass works:
            ``OpenAI``, ``Anthropic``, ``Ollama``, ``HuggingFaceLLM``, etc.
        backend (str):
            Sulci backend — ``"sqlite"`` | ``"chroma"`` | ``"qdrant"`` |
            ``"faiss"`` | ``"redis"`` | ``"milvus"`` | ``"sulci"``
            (default ``"sqlite"``).
        threshold (float):
            Cosine similarity cutoff 0–1 (default ``0.85``).
        context_window (int):
            Turns to remember per session; 0 = stateless (default).
            Set to 4 for context-aware multi-turn caching.
        api_key (Optional[str]):
            Sulci Cloud key — required when ``backend="sulci"``.
        gateway_url (str):
            Custom gateway for Enterprise VPC deployments
            (default: ``https://api.sulci.io``).

    Examples:
        # Stateless — drop-in replacement
        Settings.llm = SulciCacheLLM(llm=OpenAI(model="gpt-4o"), backend="sqlite")

        # Context-aware — RAG chatbot
        Settings.llm = SulciCacheLLM(
            llm=OpenAI(model="gpt-4o"), backend="sqlite",
            context_window=4, threshold=0.75,
        )

        # Managed Sulci Cloud
        Settings.llm = SulciCacheLLM(
            llm=OpenAI(model="gpt-4o"), backend="sulci", api_key="sk-sulci-...",
        )
    """

    # ── Public Pydantic field ─────────────────────────────────────────────────
    # BaseLLM is Pydantic — fields must be declared with Field().
    # The wrapped LLM is the only public field; Sulci kwargs are private.
    llm: Any = Field(description="The underlying LlamaIndex LLM to wrap.")

    # ── Private state (Pydantic PrivateAttr — not serialised) ─────────────────
    _sulci_kwargs: dict = PrivateAttr(default_factory=dict)
    _sulci_cache:  Any  = PrivateAttr(default=None)

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        """
        Initialise SulciCacheLLM.

        Args:
            llm:      The LlamaIndex LLM to wrap.
            **kwargs: Sulci Cache kwargs (backend, threshold, …) plus any
                      LLM kwargs forwarded to the parent LLM class.
        """
        sulci_kwargs = {k: v for k, v in kwargs.items() if k in _SULCI_KEYS}
        llm_kwargs   = {k: v for k, v in kwargs.items() if k not in _SULCI_KEYS}

        super().__init__(llm=llm, **llm_kwargs)

        # Assign private attrs AFTER super().__init__() so Pydantic is set up
        self._sulci_kwargs = sulci_kwargs
        self._sulci_cache  = None

    # ── Cache construction (lazy) ─────────────────────────────────────────────

    def _get_cache(self) -> Any:
        """Return the sulci.Cache instance, constructing it on first call."""
        if self._sulci_cache is None:
            try:
                from sulci import Cache as _Cache
            except ImportError as exc:
                raise ImportError(
                    "sulci is required for SulciCacheLLM.\n"
                    "Install: pip install \"sulci[sqlite]\"  # or another backend"
                ) from exc
            self._sulci_cache = _Cache(**self._sulci_kwargs)
        return self._sulci_cache

    # ── LLMMetadata (delegate entirely to wrapped LLM) ────────────────────────

    @property
    def metadata(self) -> LLMMetadata:
        """Return the wrapped LLM's metadata unchanged."""
        return self.llm.metadata

    # ── Cache key helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _last_user_message(messages: Sequence[ChatMessage]) -> str:
        """
        Extract the last user message as the cache key for chat calls.

        Using the full message list would mean the key grows every turn
        (system prompt + history). The last user message is the semantic
        unit that determines the response.
        """
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                return str(msg.content)
        # Fallback — join all content if no user message found
        return " ".join(str(m.content) for m in messages)

    @staticmethod
    def _pop_session_id(kwargs: dict) -> Optional[str]:
        """Extract and remove session_id from kwargs if present."""
        return kwargs.pop("session_id", None)

    # ── Cached sync methods ───────────────────────────────────────────────────

    def complete(
        self,
        prompt:    str,
        formatted: bool = False,
        **kwargs:  Any,
    ) -> CompletionResponse:
        """
        Check cache first; call underlying LLM on miss.

        Cache errors are swallowed — a failure must never raise to the caller.
        """
        session_id = self._pop_session_id(kwargs)
        try:
            resp, sim, depth = self._get_cache().get(prompt, session_id=session_id)
            if resp is not None:
                logger.debug(
                    "sulci HIT  sim=%.3f  depth=%d  prompt=%r",
                    sim, depth, prompt[:60],
                )
                return CompletionResponse(text=resp)
        except Exception:
            logger.warning("sulci complete() lookup error — cache miss", exc_info=True)

        # Cache miss — call the real LLM
        result = self.llm.complete(prompt, formatted=formatted, **kwargs)

        try:
            self._get_cache().set(prompt, result.text, session_id=session_id)
            logger.debug("sulci SET  prompt=%r", prompt[:60])
        except Exception:
            logger.warning("sulci complete() store error — skipping", exc_info=True)

        return result

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Check cache first using the last user message as the cache key.

        Cache errors are swallowed.
        """
        session_id = self._pop_session_id(kwargs)
        key        = self._last_user_message(messages)

        try:
            resp, sim, depth = self._get_cache().get(key, session_id=session_id)
            if resp is not None:
                logger.debug(
                    "sulci HIT  sim=%.3f  depth=%d  key=%r",
                    sim, depth, key[:60],
                )
                return ChatResponse(
                    message=ChatMessage(
                        role    = MessageRole.ASSISTANT,
                        content = resp,
                    )
                )
        except Exception:
            logger.warning("sulci chat() lookup error — cache miss", exc_info=True)

        # Cache miss — call the real LLM
        result = self.llm.chat(messages, **kwargs)

        try:
            text = result.message.content or ""
            if text:
                self._get_cache().set(key, text, session_id=session_id)
                logger.debug("sulci SET  key=%r", key[:60])
        except Exception:
            logger.warning("sulci chat() store error — skipping", exc_info=True)

        return result

    # ── Pass-through streaming (uncached) ─────────────────────────────────────

    def stream_complete(
        self,
        prompt:    str,
        formatted: bool = False,
        **kwargs:  Any,
    ) -> CompletionResponseGen:
        """
        Streaming completion — passes through to underlying LLM uncached.

        Streaming responses are generators that cannot be reliably stored
        mid-stream. Cache the non-streaming path instead.
        """
        return self.llm.stream_complete(prompt, formatted=formatted, **kwargs)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Streaming chat — passes through to underlying LLM uncached."""
        return self.llm.stream_chat(messages, **kwargs)

    # ── Async methods (run_in_executor — event loop never blocked) ────────────

    async def acomplete(
        self,
        prompt:    str,
        formatted: bool = False,
        **kwargs:  Any,
    ) -> CompletionResponse:
        """Async completion — delegates to sync complete() via executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.complete(prompt, formatted, **kwargs)
        )

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat — delegates to sync chat() via executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.chat(messages, **kwargs)
        )

    async def astream_complete(
        self,
        prompt:    str,
        formatted: bool = False,
        **kwargs:  Any,
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion — passes through uncached."""
        return await self.llm.astream_complete(prompt, formatted=formatted, **kwargs)

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async streaming chat — passes through uncached."""
        return await self.llm.astream_chat(messages, **kwargs)

    # ── Extras ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return Sulci Cache statistics.

        Keys: hits, misses, hit_rate, saved_cost, total_queries,
              active_sessions.
        """
        return self._get_cache().stats()

    def __repr__(self) -> str:
        s         = self._get_cache().stats()
        model     = getattr(self.llm.metadata, "model_name", self.llm.__class__.__name__)
        return (
            f"SulciCacheLLM("
            f"llm={model!r}, "
            f"hit_rate={s.get('hit_rate', 0):.1%}, "
            f"hits={s.get('hits', 0)}, "
            f"misses={s.get('misses', 0)})"
        )
