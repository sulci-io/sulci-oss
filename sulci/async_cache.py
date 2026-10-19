# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

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
arguments, same return values. This includes the partition kwargs
``tenant_id`` (v0.4.0) and ``plan`` (v0.5.6), plus ``metadata`` on
``aset`` — all keyword-only, threaded straight through to the wrapped
sync call (v0.8.1, closing the parity gap noted in the 0.8.0 CHANGELOG).

Sync passthrough methods (get, set, cached_call, stats, clear) are also
available so AsyncCache can be used in mixed sync/async codebases without
switching types. As of v0.8.2 the passthrough ``get`` / ``cached_call``
also accept the per-call ``threshold`` (v0.8.0), so the passthrough surface
is now a 100% mirror of ``Cache`` too — not just the ``a``-prefixed methods.

"Mirror" has three axes, and only two of them were ever checked. WHICH
kwargs are forwarded: guarded by ``TestAsyncSyncParity`` since v0.8.1. HOW
they are passed: deliberately NOT a mirror — ``aget`` and ``acached_call``
take ``user_id`` / ``session_id`` / ``threshold`` / ``cost_per_call`` as
positional-or-keyword where ``Cache`` makes everything after ``query``
keyword-only, and that is documented rather than fixed (see
docs/API-SURFACE.md). WHAT THEY DEFAULT TO: unchecked until now, and wrong.
``acached_call`` and the ``cached_call`` passthrough both declared
``cost_per_call: float = 0.005`` and forwarded it unconditionally, so
``AsyncCache(cost_per_call=0.02)`` was silently overridden on every call —
a constructor argument that appeared to work and did not. The defaults are
now mirrored too, and the parity test compares them.

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
                  session_ttl, api_key, telemetry.

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
        threshold:  Optional[float] = None,
        *,
        tenant_id:  Optional[str] = None,
        plan:       Optional[str] = None,
        context_threshold: Optional[float] = None,
    ) -> tuple:
        """
        Async semantic cache lookup.

        Parameters
        ----------
        threshold : float, optional
            Per-call minimum cosine similarity, overriding the instance value.
            ``None`` (default) uses the instance threshold. v0.8.0 (#34).
        tenant_id : str, optional
            Tenant identifier for multi-tenant partition isolation. Mirrors
            ``Cache.get`` (keyword-only since v0.4.0). Forwarded unchanged;
            on backends that enforce isolation (Qdrant, cloud) entries from
            other tenants are never returned, even above threshold. v0.8.1.
        plan : str, optional
            Customer plan tier ('free' | 'pro' | 'business' | 'enterprise' |
            'oss_connect'), forwarded onto the emitted ``CacheEvent.plan``.
            Mirrors ``Cache.get`` (v0.5.6). v0.8.1.

        Returns
        -------
        (response, similarity, context_depth)
            response      — str | None — cached response, or None on miss
            similarity    — float — cosine similarity (0.0 on miss)
            context_depth — int — prior turns used in blending (0 = stateless)
        """
        return await asyncio.to_thread(
            self._cache.get, query,
            threshold  = threshold,
            tenant_id  = tenant_id,
            user_id    = user_id,
            session_id = session_id,
            plan       = plan,
            context_threshold = context_threshold,
        )

    async def aset(
        self,
        query:      str,
        response:   str,
        user_id:    Optional[str] = None,
        session_id: Optional[str] = None,
        *,
        tenant_id:  Optional[str] = None,
        plan:       Optional[str] = None,
        metadata:   Optional[dict] = None,
        standalone: bool          = False,
    ) -> None:
        """
        Async cache store — saves response and advances the context window.

        Parameters
        ----------
        tenant_id : str, optional
            Tenant identifier stored with the entry for partition isolation.
            Mirrors ``Cache.set`` (v0.4.0). v0.8.1.
        plan : str, optional
            Customer plan tier, forwarded onto the emitted ``CacheEvent.plan``.
            Mirrors ``Cache.set`` (v0.5.6). v0.8.1.
        metadata : dict, optional
            Metadata persisted alongside the entry (0.9.3: a mapping whose keys are not
            ``sulci._metadata.RESERVED_METADATA_KEYS``, else TypeError/ValueError). Mirrors
            ``Cache.set``. v0.8.1.
        standalone : bool, optional
            Managed backend: store the entry even when ``session_id`` already holds a turn. Mirrors
            ``Cache.set``. 0.10.2.
        """
        return await asyncio.to_thread(
            self._cache.set, query, response,
            tenant_id  = tenant_id,
            user_id    = user_id,
            session_id = session_id,
            metadata   = metadata,
            plan       = plan,
            standalone = standalone,
        )

    async def acached_call(
        self,
        query:         str,
        llm_fn:        Callable[[str], str],
        session_id:    Optional[str]   = None,
        user_id:       Optional[str]   = None,
        cost_per_call: Optional[float] = None,
        threshold:     Optional[float] = None,
        *,
        tenant_id:     Optional[str]   = None,
        plan:          Optional[str]   = None,
        standalone:    bool            = False,
        context_threshold: Optional[float] = None,
    ) -> dict:
        """
        Async drop-in LLM wrapper — checks cache first, calls llm_fn on miss.

        Parameters
        ----------
        threshold : float, optional
            Per-call minimum cosine similarity, overriding the instance value.
            ``None`` (default) uses the instance threshold. v0.8.0 (#34).
        cost_per_call : float, optional
            Estimated LLM cost per call, for ``stats()['saved_cost']``.
            ``None`` (default) uses the instance value passed to the
            constructor. This defaulted to a hardcoded ``0.005`` until
            v0.8.3 and was forwarded unconditionally, so an ``AsyncCache``
            constructed with any other ``cost_per_call`` had it silently
            overridden on every call — and, because ``Cache.get()`` had
            already credited the instance value, the per-call delta at
            ``core.py:828`` then subtracted the difference, landing on
            exactly the wrong number rather than on noise. Mirrors
            ``Cache.cached_call``.
        tenant_id : str, optional
            Tenant identifier threaded through the underlying ``.get()`` and
            (on miss) ``.set()`` for partition isolation. Mirrors
            ``Cache.cached_call`` (v0.4.0). v0.8.1.
        plan : str, optional
            Customer plan tier threaded onto BOTH emitted events (the 'miss'
            from ``.get()`` and the 'set' from ``.set()``), so miss-then-set
            paths never leak ``plan=None`` into the stream. Mirrors
            ``Cache.cached_call`` (v0.5.6). v0.8.1.
        standalone : bool, optional
            Managed backend: store the answer even when ``session_id`` already holds a turn. Mirrors
            ``Cache.cached_call``. 0.10.2.

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
            threshold     = threshold,
            tenant_id     = tenant_id,
            session_id    = session_id,
            user_id       = user_id,
            cost_per_call = cost_per_call,
            plan          = plan,
            standalone    = standalone,
            context_threshold = context_threshold,
        )

    async def aget_context(self, session_id: str, *, tenant_id: Optional[str] = None,
                           user_id: Optional[str] = None):
        """Async — return the ContextWindow for a session (0.9.3: of this tenant and user)."""
        return await asyncio.to_thread(self._cache.get_context, session_id, tenant_id=tenant_id, user_id=user_id)

    async def aclear_context(self, session_id: str, *, tenant_id: Optional[str] = None,
                             user_id: Optional[str] = None) -> None:
        """Async — reset conversation history for a session (0.9.3: of this tenant and user)."""
        return await asyncio.to_thread(self._cache.clear_context, session_id, tenant_id=tenant_id, user_id=user_id)

    async def acontext_summary(self, session_id: Optional[str] = None, *, tenant_id: Optional[str] = None,
                               user_id: Optional[str] = None) -> dict:
        """Async — snapshot of one or all sessions."""
        return await asyncio.to_thread(self._cache.context_summary, session_id, tenant_id=tenant_id, user_id=user_id)

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
        """Async — evict all entries, reset stats and sessions. Raises as ``Cache.clear()`` does."""
        return await asyncio.to_thread(self._cache.clear)

    async def adelete_user(self, user_id: str, *, tenant_id: Optional[str] = None) -> dict:
        """Async — erase one user of one tenant: entries and conversation history (0.10.0). Returns
        ``{"entries": N, "sessions": M}`` and raises as ``Cache.delete_user()`` does."""
        return await asyncio.to_thread(self._cache.delete_user, user_id, tenant_id=tenant_id)

    # ── Sync passthrough ──────────────────────────────────────────────────────
    # Provided so AsyncCache can be used in mixed sync/async codebases
    # without switching types.

    def get(
        self,
        query:      str,
        user_id:    Optional[str] = None,
        session_id: Optional[str] = None,
        *,
        threshold:  Optional[float] = None,
        tenant_id:  Optional[str] = None,
        plan:       Optional[str] = None,
        context_threshold: Optional[float] = None,
    ) -> tuple:
        """Sync passthrough — cache.get(). threshold (v0.8.2) + tenant_id/plan
        (v0.8.1) mirror Cache.get."""
        return self._cache.get(
            query,
            threshold  = threshold,
            tenant_id  = tenant_id,
            user_id    = user_id,
            session_id = session_id,
            plan       = plan,
            context_threshold = context_threshold,
        )

    def set(
        self,
        query:      str,
        response:   str,
        user_id:    Optional[str] = None,
        session_id: Optional[str] = None,
        *,
        tenant_id:  Optional[str] = None,
        plan:       Optional[str] = None,
        metadata:   Optional[dict] = None,
        standalone: bool          = False,
    ) -> None:
        """Sync passthrough — cache.set(). tenant_id/plan/metadata mirror Cache.set (v0.8.1); standalone (0.10.2)."""
        return self._cache.set(
            query, response,
            tenant_id  = tenant_id,
            user_id    = user_id,
            session_id = session_id,
            metadata   = metadata,
            plan       = plan,
            standalone = standalone,
        )

    def cached_call(
        self,
        query:         str,
        llm_fn:        Callable[[str], str],
        session_id:    Optional[str] = None,
        user_id:       Optional[str] = None,
        cost_per_call: Optional[float] = None,
        *,
        threshold:     Optional[float] = None,
        tenant_id:     Optional[str] = None,
        plan:          Optional[str] = None,
        standalone:    bool          = False,
        context_threshold: Optional[float] = None,
    ) -> dict:
        """Sync passthrough — cache.cached_call(). threshold (v0.8.2) +
        tenant_id/plan (v0.8.1) mirror Cache.cached_call.

        cost_per_call defaults to None (v0.8.3), meaning "use the value this
        AsyncCache was constructed with", exactly as Cache.cached_call does.
        It was a hardcoded 0.005 and was forwarded unconditionally — see the
        module docstring."""
        return self._cache.cached_call(
            query, llm_fn,
            threshold     = threshold,
            tenant_id     = tenant_id,
            session_id    = session_id,
            user_id       = user_id,
            cost_per_call = cost_per_call,
            plan          = plan,
            standalone    = standalone,
            context_threshold = context_threshold,
        )

    def stats(self) -> dict:
        """Sync passthrough — cache.stats()."""
        return self._cache.stats()

    def clear(self) -> None:
        """Sync passthrough — cache.clear()."""
        return self._cache.clear()

    def delete_user(self, user_id: str, *, tenant_id: Optional[str] = None) -> dict:
        """Sync passthrough — cache.delete_user() (0.10.0)."""
        return self._cache.delete_user(user_id, tenant_id=tenant_id)

    def get_context(self, session_id: str, *, tenant_id: Optional[str] = None, user_id: Optional[str] = None):
        """Sync passthrough — cache.get_context()."""
        return self._cache.get_context(session_id, tenant_id=tenant_id, user_id=user_id)

    def clear_context(self, session_id: str, *, tenant_id: Optional[str] = None,
                      user_id: Optional[str] = None) -> None:
        """Sync passthrough — cache.clear_context()."""
        return self._cache.clear_context(session_id, tenant_id=tenant_id, user_id=user_id)

    def context_summary(self, session_id: Optional[str] = None, *, tenant_id: Optional[str] = None,
                        user_id: Optional[str] = None) -> dict:
        """Sync passthrough — cache.context_summary()."""
        return self._cache.context_summary(session_id, tenant_id=tenant_id, user_id=user_id)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        inner = repr(self._cache)
        return f"AsyncCache({inner})"
