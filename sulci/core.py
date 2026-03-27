# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/core.py
================
Core semantic cache engine — backend-agnostic, context-aware.

Supports backends : ChromaDB, Qdrant, FAISS, Redis, SQLite, Milvus
Embedding models  : MiniLM / MPNet / BGE (local, free) or OpenAI API
Context-awareness : sliding window per session, blended embedding lookup
"""
import time
import hashlib
import importlib
import time as _time
from typing import Optional, Callable, Any

from sulci.context import ContextWindow, SessionStore


class Cache:
    """
    Semantic cache for LLM applications with optional context-awareness.

    Stateless (default)
    -------------------
    Each query is looked up independently — identical to a standard
    semantic cache.  No sessions, no overhead::

        cache = Cache(backend="sqlite", threshold=0.85)
        result = cache.cached_call("What is Python?", my_llm)

    Context-aware
    -------------
    Enable by setting ``context_window > 0`` and passing a ``session_id``
    to ``cached_call`` / ``get`` / ``set``.  Sulci blends the current
    query embedding with recent conversation history so that ambiguous
    follow-up queries resolve correctly::

        cache = Cache(backend="sqlite", context_window=6)

        cache.cached_call(
            "My Docker container crashes on startup",
            my_llm,
            session_id="user-42",
        )
        result = cache.cached_call(
            "How do I fix it?",          # resolved in Docker context
            my_llm,
            session_id="user-42",
        )
        print(result["context_depth"])   # 1 — prior turn influenced lookup
        print(result["source"])          # "cache" or "llm"

    Args:
        backend:         Vector store backend. One of:
                         "chroma" | "qdrant" | "faiss" | "redis" | "sqlite" | "milvus"
        threshold:       Cosine similarity threshold (0.0–1.0).
                         0.85 is a good starting point.
        embedding_model: Local: "minilm" (default), "mpnet", "bge"
                         API:   "openai" (requires OPENAI_API_KEY)
        ttl_seconds:     Cache entry time-to-live. None = no expiry.
        personalized:    Scope cache per user_id (prevents cross-user hits).
        db_path:         Local storage path (ChromaDB, SQLite, FAISS).
        context_window:  Number of recent turns to remember per session.
                         0  = stateless (default, original behaviour).
                         4–8 is recommended for conversational apps.
        query_weight:    Current query's weight vs blended history (0.0–1.0).
                         0.70 = query dominates, context nudges direction.
        context_decay:   Exponential decay per turn (older → less weight).
                         0.5 = each older turn contributes half as much.
        session_ttl:     Seconds of inactivity before a session is evicted.
                         None = sessions live forever (not recommended).
    """

    def __init__(
        self,
        backend:         str           = "chroma",
        threshold:       float         = 0.85,
        embedding_model: str           = "minilm",
        ttl_seconds:     Optional[int] = 86400,
        personalized:    bool          = False,
        db_path:         str           = "./sulci_db",
        # context-awareness
        context_window:  int           = 0,
        query_weight:    float         = 0.70,
        context_decay:   float         = 0.50,
        session_ttl:     Optional[int] = 3600,
        telemetry                      = True,
        api_key:         Optional[str] = None,
    ):
        self._telemetry     = telemetry
        self.backend        = backend
        self.threshold      = threshold
        self.ttl_seconds    = ttl_seconds
        self.personalized   = personalized
        self.context_window = context_window
        self._stats         = {"hits": 0, "misses": 0, "saved_cost": 0.0}

        self._embedder = self._load_embedder(embedding_model)
        self._backend  = self._load_backend(backend, db_path, api_key)

        # Session store — created only when context_window > 0
        self._sessions: Optional[SessionStore] = (
            SessionStore(
                max_turns    = context_window,
                query_weight = query_weight,
                decay        = context_decay,
                ttl_seconds  = session_ttl,
            )
            if context_window > 0
            else None
        )

    # ── private helpers ───────────────────────────────────────────

    def _load_embedder(self, name: str):
        if name == "openai":
            from sulci.embeddings.openai import OpenAIEmbedder
            return OpenAIEmbedder()
        from sulci.embeddings.minilm import MiniLMEmbedder
        return MiniLMEmbedder(name)

    def _load_backend(self, name: str, db_path: str, api_key: Optional[str] = None):
        # sulci cloud backend — special construction, needs api_key not db_path
        if name == "sulci":
            import os, sys
            _module_key = None
            _sulci_mod  = sys.modules.get("sulci")
            if _sulci_mod is not None:
                _module_key = getattr(_sulci_mod, "_api_key", None)

            resolved_key = (
                api_key
                or os.environ.get("SULCI_API_KEY")
                or _module_key
            )
            from sulci.backends.cloud import SulciCloudBackend
            return SulciCloudBackend(api_key=resolved_key)

        # all other backends — loaded dynamically via importlib
        registry = {
            "chroma": "sulci.backends.chroma.ChromaBackend",
            "qdrant": "sulci.backends.qdrant.QdrantBackend",
            "faiss":  "sulci.backends.faiss.FAISSBackend",
            "redis":  "sulci.backends.redis.RedisBackend",
            "sqlite": "sulci.backends.sqlite.SQLiteBackend",
            "milvus": "sulci.backends.milvus.MilvusBackend",
        }

        if name not in registry:
            raise ValueError(
                f"Unknown backend '{name}'. "
                f"Choose from: {list(registry.keys()) + ['sulci']}"
            )
        
        module_path, cls_name = registry[name].rsplit(".", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)(db_path=db_path)

    def _context_vec(
        self,
        query:      str,
        session_id: Optional[str],
    ) -> tuple:
        """
        Return (embedding_vector, context_depth).

        context_depth = number of prior turns that influenced the embedding.
        0 means stateless — no blending occurred.
        """
        raw_vec = self._embedder.embed(query)

        if self._sessions is None or not session_id:
            return raw_vec, 0

        window = self._sessions.get(session_id)
        depth  = window.depth
        if depth == 0:
            return raw_vec, 0

        blended = window.blend(raw_vec, embedder=self._embedder)
        return blended, depth

    def _record_turn(
        self,
        session_id: Optional[str],
        query:      str,
        response:   str,
        query_vec:  list,
    ) -> None:
        """Append user + assistant turns to the session window."""
        if self._sessions is None or not session_id:
            return
        window = self._sessions.get(session_id)
        window.add_turn(query,    role="user",      embedding=query_vec)
        window.add_turn(response, role="assistant")   # embedded lazily on next blend

    # ── public API ────────────────────────────────────────────────

    def get(
        self,
        query:      str,
        user_id:    Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple:
        """
        Check cache for a semantically similar query.

        When ``session_id`` is set and ``context_window > 0``, the lookup
        uses a context-blended embedding so ambiguous queries resolve
        correctly within the current conversation.

        Returns:
            (cached_response, similarity_score, context_depth)
            response is None on cache miss.
            context_depth = number of prior turns that influenced lookup.
        """
        _t0 = _time.time()
        vec, depth = self._context_vec(query, session_id)
        resp, sim  = self._backend.search(
            embedding = vec,
            threshold = self.threshold,
            user_id   = user_id if self.personalized else None,
            now       = time.time(),
        )
        try:
            if self._telemetry:
                import sys
                _sulci = sys.modules.get("sulci")
                if _sulci is not None:
                    _sulci._emit("cache.get", {
                            "backend":    self.backend,
                            "hits":       1 if resp is not None else 0,
                            "misses":     0 if resp is not None else 1,
                            "latency_ms": round((_time.time() - _t0) * 1000, 2),
                        })
        except Exception:
            pa

        return resp, sim, depth

    def set(
        self,
        query:      str,
        response:   str,
        user_id:    Optional[str]  = None,
        session_id: Optional[str]  = None,
        metadata:   Optional[dict] = None,
    ) -> None:
        """
        Store a query-response pair in the cache.

        The raw (un-blended) query embedding is stored so entries are
        reusable across different sessions and conversation states.
        Context blending only happens at lookup time.

        When ``session_id`` is provided, the turn is also recorded in
        the session window to inform future context-blended lookups.
        """
        raw_vec = self._embedder.embed(query)
        key     = hashlib.sha256(query.encode()).hexdigest()[:16]
        expires = time.time() + self.ttl_seconds if self.ttl_seconds else None
        self._backend.store(
            key       = key,
            query     = query,
            response  = response,
            embedding = raw_vec,
            user_id   = user_id if self.personalized else None,
            expires   = expires,
            metadata  = metadata or {},
        )
        if session_id and self._sessions:
            self._record_turn(session_id, query, response, raw_vec)

    def cached_call(
        self,
        query:         str,
        llm_fn:        Callable,
        user_id:       Optional[str] = None,
        session_id:    Optional[str] = None,
        cost_per_call: float         = 0.005,
        **llm_kwargs:  Any,
    ) -> dict:
        """
        Drop-in LLM wrapper: checks cache first, calls LLM on miss.

        Context-awareness is automatic when ``session_id`` is supplied
        and ``context_window > 0``.  Every call appends the query and
        response to the session window so the next call benefits from
        the updated context.

        Args:
            query:         User query string.
            llm_fn:        Callable(query, **kwargs) -> str.
            user_id:       For personalized per-user caching.
            session_id:    Conversation ID — enables context-aware lookup.
            cost_per_call: Estimated LLM cost per call (for savings stats).
            **llm_kwargs:  Forwarded to llm_fn on cache miss.

        Returns:
            {
                "response":      str,
                "source":        "cache" | "llm",
                "similarity":    float,
                "latency_ms":    float,
                "cache_hit":     bool,
                "context_depth": int,   # 0 if context unused
            }
        """
        t0              = time.perf_counter()
        hit, sim, depth = self.get(query, user_id=user_id, session_id=session_id)
        ms              = (time.perf_counter() - t0) * 1000

        if hit is not None:
            self._stats["hits"]       += 1
            self._stats["saved_cost"] += cost_per_call
            # Record turn even on hits so future queries see this exchange
            if session_id and self._sessions:
                raw_vec = self._embedder.embed(query)
                self._record_turn(session_id, query, hit, raw_vec)
            return {
                "response":      hit,
                "source":        "cache",
                "similarity":    round(sim, 4),
                "latency_ms":    round(ms, 2),
                "cache_hit":     True,
                "context_depth": depth,
            }

        response = llm_fn(query, **llm_kwargs)
        self.set(query, response, user_id=user_id, session_id=session_id)
        ms = (time.perf_counter() - t0) * 1000
        self._stats["misses"] += 1
        return {
            "response":      response,
            "source":        "llm",
            "similarity":    round(sim, 4),
            "latency_ms":    round(ms, 2),
            "cache_hit":     False,
            "context_depth": depth,
        }

    # ── context management API ────────────────────────────────────

    def get_context(self, session_id: str) -> ContextWindow:
        """
        Return the ContextWindow for a session (creates one if new).

        Use this to manually inject prior turns, inspect history, or
        pre-seed context before the first cached_call::

            ctx = cache.get_context("user-42")
            ctx.add_turn("I am building a Python web scraper", role="user")
            ctx.add_turn("Great! Let me help with that.", role="assistant")

        Raises RuntimeError if context_window == 0.
        """
        if self._sessions is None:
            raise RuntimeError(
                "Context tracking is disabled. "
                "Initialise Cache with context_window > 0 to enable it.\n"
                "Example: Cache(backend='sqlite', context_window=6)"
            )
        return self._sessions.get(session_id)

    def clear_context(self, session_id: str) -> None:
        """
        Clear conversation history for a session without removing cache entries.

        Call this when a user starts a new topic or when a conversation ends.
        """
        if self._sessions:
            self._sessions.delete(session_id)

    def context_summary(self, session_id: Optional[str] = None) -> dict:
        """
        Return a human-readable summary of session context.

        If session_id is None, returns a summary of ALL active sessions.
        """
        if self._sessions is None:
            return {"context_window": 0, "message": "Context tracking disabled."}
        if session_id:
            win = self._sessions.get(session_id)
            return {"session_id": session_id, **win.summary()}
        return self._sessions.summary()

    # ── stats + lifecycle ─────────────────────────────────────────

    def stats(self) -> dict:
        """Return hit/miss statistics for this cache instance."""
        total = self._stats["hits"] + self._stats["misses"]
        base  = {
            **self._stats,
            "total_queries": total,
            "hit_rate": round(self._stats["hits"] / total, 4) if total else 0.0,
        }
        if self._sessions:
            base["active_sessions"] = len(self._sessions.active_sessions())
        return base

    def clear(self) -> None:
        """Remove all cached entries, reset stats, and clear all sessions."""
        self._backend.clear()
        self._stats = {"hits": 0, "misses": 0, "saved_cost": 0.0}
        if self._sessions:
            self._sessions.clear_all()

    def __repr__(self) -> str:
        ctx = f", context_window={self.context_window}" if self.context_window else ""
        return (
            f"Cache(backend={self.backend!r}, "
            f"threshold={self.threshold}"
            f"{ctx}, "
            f"hits={self._stats['hits']}, "
            f"misses={self._stats['misses']})"
        )
