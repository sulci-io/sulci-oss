# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/async_cache.py
====================
Non-blocking async wrapper around sulci.Cache.

Delegates all cache operations to a thread pool via asyncio.to_thread()
so the event loop is never blocked during embedding or vector search.

Typical use — FastAPI endpoint
------------------------------
    from sulci import AsyncCache

    cache = AsyncCache(backend="sqlite", threshold=0.85, context_window=4)

    @app.post("/chat")
    async def chat(query: str, session_id: str):
        response, sim, depth = await cache.aget(query, session_id=session_id)
        if response:
            return {"response": response, "source": "cache", "sim": sim}
        response = await call_llm(query)
        await cache.aset(query, response, session_id=session_id)
        return {"response": response, "source": "llm"}

All constructor parameters are identical to sulci.Cache.
All async methods mirror their sync counterparts exactly — same
arguments, same return values.

Sync passthrough methods (get, set, stats, clear) are also available
so AsyncCache can be used in mixed sync/async codebases without
switching types.

Pattern: asyncio.to_thread()
----------------------------
asyncio.to_thread() runs a sync callable in a thread-pool executor and
returns a coroutine that yields to the event loop while waiting.
It is equivalent to loop.run_in_executor(None, fn) but cleaner and
idiomatic for Python 3.9+.  Sulci requires Python 3.9+.

Requires: Python 3.9+
"""

import asyncio
from typing import Any, Callable, Optional

from sulci.core import Cache


class AsyncCache:
    """
    Non-blocking async wrapper around sulci.Cache.

    All ``a*`` methods are async and safe to ``await`` from any
    async framework (FastAPI, Starlette, aiohttp, LangChain async
    chains, LlamaIndex async agents, CrewAI, AutoGen, etc.).

    Sync passthrough methods (``get``, ``set``, ``stats``, ``clear``)
    are provided so the same object works in mixed sync/async code.

    Args:
        **kwargs: All arguments accepted by sulci.Cache — backend,
                  threshold, embedding_model, ttl_seconds, personalized,
                  db_path, context_window, query_weight, context_decay,
                  session_ttl, cost_per_call, api_key, telemetry.

    Examples
    --------
    Stateless::

        cache = AsyncCache(backend="sqlite", threshold=0.85)
        response, sim, depth = await cache.aget("What is Python?")

    Context-aware::

        cache = AsyncCache(backend="sqlite", context_window=4)
        await cache.aset("What is Python?", "...", session_id="s1")
        response, sim, depth = await cache.aget(
            "Tell me more about it", session_id="s1"
        )
        # depth=1 — prior turn blended into lookup

    Drop-in LLM wrapper::

        result = await cache.acached_call(
            "How do I deploy to AWS?",
            my_llm_fn,
            session_id = "user-42",
        )
        print(result["source"])      # "cache" or "llm"
        print(result["latency_ms"])  # <10ms on cache hit
    """

    def __init__(self, **kwargs: Any) -> None:
        self._cache = Cache(**kwargs)

    # ── Async methods ─────────────────────────────────────────────────────────

    async def aget(
        self,
        query:      str,
        user_id:    Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple:
        """
        Async semantic cache lookup.

        Returns
        -------
        (response, similarity, context_depth)
            response      — str | None — cached response, or None on miss
            similarity    — float — cosine similarity (0.0 on miss)
            context_depth — int — prior turns used in blending (0 = stateless)
        """
        return await asyncio.to_thread(
            self._cache.get, query,
            user_id    = user_id,
            session_id = session_id,
        )

    async def aset(
        self,
        query:      str,
        response:   str,
        user_id:    Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Async cache store — saves response and advances the context window.
        """
        return await asyncio.to_thread(
            self._cache.set, query, response,
            user_id    = user_id,
            session_id = session_id,
        )

    async def acached_call(
        self,
        query:         str,
        llm_fn:        Callable[[str], str],
        session_id:    Optional[str]   = None,
        user_id:       Optional[str]   = None,
        cost_per_call: Optional[float] = None,
    ) -> dict:
        """
        Async drop-in LLM wrapper — checks cache first, calls llm_fn on miss.

        Returns
        -------
        dict with keys:
            response      — str
            source        — "cache" | "llm"
            similarity    — float
            latency_ms    — float
            cache_hit     — bool
            context_depth — int
        """
        return await asyncio.to_thread(
            self._cache.cached_call, query, llm_fn,
            session_id    = session_id,
            user_id       = user_id,
            cost_per_call = cost_per_call,
        )

    async def aget_context(self, session_id: str):
        """Async — return the ContextWindow for a session."""
        return await asyncio.to_thread(self._cache.get_context, session_id)

    async def aclear_context(self, session_id: str) -> None:
        """Async — reset conversation history for a session."""
        return await asyncio.to_thread(self._cache.clear_context, session_id)

    async def acontext_summary(self, session_id: Optional[str] = None) -> dict:
        """Async — snapshot of one or all sessions."""
        return await asyncio.to_thread(self._cache.context_summary, session_id)

    async def astats(self) -> dict:
        """
        Async cache statistics.

        Returns
        -------
        dict with keys: hits, misses, hit_rate, saved_cost,
                        total_queries, active_sessions
        """
        return await asyncio.to_thread(self._cache.stats)

    async def aclear(self) -> None:
        """Async — evict all entries, reset stats and sessions."""
        return await asyncio.to_thread(self._cache.clear)

    # ── Sync passthrough ──────────────────────────────────────────────────────
    # Provided so AsyncCache can be used in mixed sync/async codebases
    # without switching types.

    def get(
        self,
        query:      str,
        user_id:    Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple:
        """Sync passthrough — cache.get()."""
        return self._cache.get(query, user_id=user_id, session_id=session_id)

    def set(
        self,
        query:      str,
        response:   str,
        user_id:    Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Sync passthrough — cache.set()."""
        return self._cache.set(
            query, response, user_id=user_id, session_id=session_id
        )

    def cached_call(
        self,
        query:         str,
        llm_fn:        Callable[[str], str],
        session_id:    Optional[str] = None,
        user_id:       Optional[str] = None,
        cost_per_call: Optional[float] = None,
    ) -> dict:
        """Sync passthrough — cache.cached_call()."""
        return self._cache.cached_call(
            query, llm_fn,
            session_id    = session_id,
            user_id       = user_id,
            cost_per_call = cost_per_call,
        )

    def stats(self) -> dict:
        """Sync passthrough — cache.stats()."""
        return self._cache.stats()

    def clear(self) -> None:
        """Sync passthrough — cache.clear()."""
        return self._cache.clear()

    def get_context(self, session_id: str):
        """Sync passthrough — cache.get_context()."""
        return self._cache.get_context(session_id)

    def clear_context(self, session_id: str) -> None:
        """Sync passthrough — cache.clear_context()."""
        return self._cache.clear_context(session_id)

    def context_summary(self, session_id: Optional[str] = None) -> dict:
        """Sync passthrough — cache.context_summary()."""
        return self._cache.context_summary(session_id)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        inner = repr(self._cache)
        return f"AsyncCache({inner})"
