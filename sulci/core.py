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
import os
import time
import hashlib
import importlib
import time as _time
from typing import Optional, Callable, Any, Union

from sulci.context import ContextWindow
from sulci.context import SessionStore as _BuiltinSessionStore

# v0.5.0 — sessions and sinks protocols (additive; see ADR 0004 + ADR 0007)
from sulci.sessions import SessionStore as SessionStoreProtocol
from sulci.sinks import EventSink, NullSink, CacheEvent
from sulci.sinks.protocol import query_hash as _query_hash

# v0.6.0 — Backend + Embedder protocols for instance injection in Cache.__init__
# (sulci-oss #34 C1c + C1d, umbrella #63). Used only in type hints + isinstance
# dispatch; runtime behavior is duck-typed against the protocol's methods.
from sulci.embeddings.protocol import Embedder
from sulci.backends.protocol   import Backend


# ── v0.5.2 nudge state (D15) ──────────────────────────────────────────────────
# Module-level so we nudge once per process, regardless of how many Cache
# instances the user creates. Suppression order in cache.stats():
#   1. SULCI_QUIET=1 in env       → never show
#   2. sulci._telemetry_enabled   → already connected, no nudge needed
#   3. _NUDGE_SHOWN already True  → suppressed (one-shot per process)
#   4. cache._query_count < 100   → not enough usage yet
# All four checks are O(1). Nudge writes to stderr, never raises.

_NUDGE_THRESHOLD = 100
_NUDGE_SHOWN     = False


class _ProtocolAdaptedSessionStore:
    """
    Internal adapter — bridges injected SessionStore-protocol stores
    (sulci.sessions.SessionStore: raw vector storage) to the legacy
    surface Cache uses internally (ContextWindow-returning manager).

    Per ADR 0007: rebuilds a transient ContextWindow on every .get(),
    seeded from the inner store's vectors. The returned window's
    add_turn(role='user', embedding=...) is wrapped to round-trip
    user-vector additions back to the inner store. Assistant turns
    stay window-local (the new protocol doesn't carry response text).

    Not part of the public API.
    """

    def __init__(
        self,
        inner:        SessionStoreProtocol,
        query_weight: float,
        decay:        float,
        max_turns:    int,
    ):
        self._inner     = inner
        self._max_turns = max_turns
        self._cfg       = dict(
            max_turns    = max_turns,
            query_weight = query_weight,
            decay        = decay,
        )

    def get(self, session_id: str) -> ContextWindow:
        """Rebuild a transient window seeded from the inner store."""
        window = ContextWindow(**self._cfg)
        try:
            for v in self._inner.get(session_id):
                window.add_turn("", role="user", embedding=v)
        except Exception:
            # Inner-store read failures degrade to empty context
            pass
        self._wrap_add_turn(window, session_id)
        return window

    def _wrap_add_turn(self, window, session_id):
        original    = window.add_turn
        inner       = self._inner
        max_turns   = self._max_turns
        def wrapped(text, role="user", embedding=None):
            result = original(text, role=role, embedding=embedding)
            # Write-through: only user-role turns with embeddings
            # round-trip to the inner store. Assistant turns and
            # text-only user turns stay window-local.
            if role == "user" and embedding is not None:
                try:
                    inner.append(session_id, embedding, max_turns=max_turns)
                except Exception:
                    # Inner-store write failures must not break Cache calls
                    pass
            return result
        window.add_turn = wrapped

    def delete(self, session_id: str) -> None:
        try:
            self._inner.clear(session_id)
        except Exception:
            pass

    def clear_all(self) -> None:
        # The new protocol doesn't expose enumeration. For multi-replica
        # deployments (the primary motivation for injection), clear-all
        # at the library level is a no-op — operators clear via their
        # own admin tooling. Behavioral parity with legacy in the
        # single-process case is satisfied by the adapter being
        # discarded with the Cache instance.
        pass

    def active_sessions(self) -> list:
        # Same enumeration limitation as clear_all().
        return []

    def summary(self) -> dict:
        try:
            inner_sum = self._inner.summary() or {}
        except Exception:
            inner_sum = {}
        # Translate new-protocol shape to legacy shape
        return {
            "active_sessions": int(inner_sum.get("sessions", 0)),
            "ttl_seconds":     None,
            "sessions":        {},  # per-session breakdown not available
        }


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
        backend:         Union[str, Backend]  = "chroma",
        threshold:       float                = 0.85,
        embedding_model: Union[str, Embedder] = "minilm",
        ttl_seconds:     Optional[int]        = 86400,
        personalized:    bool                 = False,
        db_path:         str                  = "./sulci_db",
        # context-awareness
        context_window:  int           = 0,
        query_weight:    float         = 0.70,
        context_decay:   float         = 0.50,
        session_ttl:     Optional[int] = 3600,
        telemetry                      = True,
        api_key:         Optional[str] = None,
        gateway_url:     str           = "",
        # ── v0.5.0 additions (ADR 0004 + ADR 0007) ──
        session_store:   Optional[SessionStoreProtocol] = None,
        event_sink:      Optional[EventSink]            = None,
        # ── v0.7.1 — issue #88 ──────────────────────────────────────────────
        cost_per_call:   float         = 0.005,
    ):
        self._telemetry     = telemetry
        self.backend        = backend
        self.threshold      = threshold
        self.ttl_seconds    = ttl_seconds
        self.personalized   = personalized
        self.context_window  = context_window
        self.embedding_model = embedding_model
        self._stats          = {"hits": 0, "misses": 0, "saved_cost": 0.0}

        # v0.7.1 (issue #88) — Cache-instance default for the per-call cost
        # used to populate _stats["saved_cost"]. Previously saved_cost only
        # incremented inside cached_call(), which left users of the raw
        # get()/set() API (notably LangChain via set_llm_cache) reporting $0
        # saved indefinitely. Now get() also contributes to saved_cost using
        # this value; opt out with cost_per_call=0.
        self._cost_per_call = cost_per_call

        self._embedder = None  # set conditionally below — see v0.6.1 note
        self._backend  = self._load_backend(backend, db_path, api_key, gateway_url)

        # ── v0.7.0 — auto-connect to the telemetry path ──────────────────────
        # ADR 0001 (sulci-oss) / ADR 0021 (sulci-platform): when an api_key is
        # resolvable from {kwarg, SULCI_API_KEY env, module-level _api_key} and
        # telemetry is left at its default (True), Cache() implicitly calls
        # sulci.connect() so the four telemetry-backed dashboard panels
        # (TrendChart, AuditEventsTable, DeploymentsTable, Active-SDKs counter)
        # populate without a second SDK call.
        #
        # Privacy invariant preserved: per Sulci_UI_Design_and_Dev_Tracker §5.2
        # trust boundaries, presence of api_key (kwarg or SULCI_API_KEY env) is
        # explicit user opt-in. This block makes the kwarg path behave
        # identically to the env path (which was already an authorized opt-in
        # signal per the spec). Pre-0.7.0 the kwarg path silently failed to
        # enable telemetry — a footgun documented in sulci-platform PR #249
        # session notes.
        #
        # Short-circuited when ANY of:
        #   - self._telemetry is False (explicit instance-level opt-out wins)
        #   - sulci._api_key is already set (any prior connect() call ran —
        #     including connect(telemetry=False); the user's explicit choice
        #     survives whether they opted in OR out)
        #   - no api_key resolvable from any of the three sources (pure
        #     self-hosted: no Sulci account, no dashboard, no telemetry)
        #
        # Rationale for the `_api_key is set` short-circuit: module-level
        # `_api_key` is None at import time and is ONLY set non-None by
        # sulci.connect(). So `_api_key is not None` is a reliable proxy for
        # "the user has explicitly invoked connect() and chose some posture."
        # That posture — whether telemetry=True (enabled) or telemetry=False
        # (deliberate opt-out) — is preserved by this check; auto-connect will
        # not second-guess it.
        #
        # Auto-connect failures NEVER crash Cache construction. The cache
        # still serves get/set normally; only the telemetry side stays off,
        # and a WARNING is logged identifying the failure mode.
        if self._telemetry:
            import os, sys
            _sulci_mod = sys.modules.get("sulci")
            if _sulci_mod is not None:
                _user_already_connected = (
                    getattr(_sulci_mod, "_telemetry_enabled", False)
                    or getattr(_sulci_mod, "_api_key", None) is not None
                )
                if not _user_already_connected:
                    resolved_key = (
                        api_key
                        or os.environ.get("SULCI_API_KEY")
                    )
                    if resolved_key:
                        try:
                            _sulci_mod.connect(
                                api_key   = resolved_key,
                                telemetry = True,
                                prompt    = False,
                            )
                        except Exception as exc:
                            import logging
                            logging.getLogger("sulci").warning(
                                "auto-connect from Cache() failed; telemetry "
                                "disabled. Call sulci.connect() explicitly to "
                                "retry. (%s: %s)",
                                type(exc).__name__, exc,
                            )

        # v0.6.0 (sulci-oss #62, umbrella #63) — cloud-transport detection.
        # When self._backend is the cloud transport (SulciCloudBackend), the
        # gateway-side library does ALL engine work (embed + search + emit);
        # the SDK is a thin transport. Cache.get/set short-circuits local
        # embedding when this flag is set.
        #
        # Capability-based detection (hasattr) rather than isinstance against
        # SulciCloudBackend avoids importing httpx at module load — the cloud
        # backend's import is lazy via _load_backend, and that property must
        # be preserved for self-hosted users who don't need httpx (sulci-oss
        # #60). Any backend that implements `remote_get` + `remote_set` is
        # treated as a transport; this future-proofs against alternative
        # transports (custom impls, internal forks).
        self._is_remote_transport = (
            hasattr(self._backend, "remote_get") and
            hasattr(self._backend, "remote_set")
        )

        # v0.6.1 (sulci-oss #60) — defer the local Embedder load when the
        # backend is a remote transport. The gateway-side library does the
        # embedding via EmbedServiceEmbedder, so the SDK's local embedder
        # is never read on the cloud path (every call site is gated by
        # `if self._is_remote_transport` — see Cache.get/Cache.set/cached_call).
        # Loading it eagerly forced cloud-only users to install
        # sentence-transformers (~80MB pulled by `sulci[chroma]`, `sulci[qdrant]`,
        # etc.) even though `sulci[cloud]` correctly declares only `httpx>=0.27`.
        # Pre-v0.6.1 manifestation: `pip install "sulci[cloud]==0.6.0"` +
        # `Cache(backend="sulci", ...)` → ImportError on `sentence_transformers`
        # from `MiniLMEmbedder.__init__`. After this change, _embedder stays at
        # its placeholder None and the cloud-only install path works end-to-end.
        if not self._is_remote_transport:
            self._embedder = self._load_embedder(embedding_model)

        # ── v0.5.2 nudge counter (D15) ──
        # Tracks queries observed by THIS instance, used by .stats() to
        # decide whether to print the one-shot nudge toward sulci.connect()
        # at 100 queries. Kept separate from _stats["hits"]/["misses"]
        # because the nudge is a one-shot trigger, not a cumulative metric
        # — collapsing the two would re-fire the nudge after .clear().
        self._query_count = 0

        # ── v0.5.0 session-store wiring (ADR 0007) ──
        # Three branches, in priority order:
        #   1. session_store= injected → wrap with adapter (B1 approach)
        #   2. context_window > 0      → legacy _BuiltinSessionStore (default)
        #   3. otherwise               → no context tracking
        if session_store is not None:
            self._sessions = _ProtocolAdaptedSessionStore(
                inner        = session_store,
                query_weight = query_weight,
                decay        = context_decay,
                max_turns    = context_window if context_window > 0 else 6,
            )
        elif context_window > 0:
            self._sessions = _BuiltinSessionStore(
                max_turns    = context_window,
                query_weight = query_weight,
                decay        = context_decay,
                ttl_seconds  = session_ttl,
            )
        else:
            self._sessions = None

        # ── v0.5.0 event sink (ADR 0004) ──
        # Default NullSink preserves zero-overhead behavior when no sink injected.
        self._event_sink: EventSink = event_sink if event_sink is not None else NullSink()

    # ── private helpers ───────────────────────────────────────────

    def _load_embedder(self, name_or_instance):
        # v0.6.0 (sulci-oss #34 C1c) — accept a pre-constructed Embedder
        # instance directly. We duck-type rather than runtime-validate against
        # the Embedder protocol; the failure surface is the first .embed() /
        # .embed_batch() call, which is the same shape any wrong-typed value
        # would produce. Any non-str is treated as an instance.
        if not isinstance(name_or_instance, str):
            return name_or_instance
        name = name_or_instance
        if name == "openai":
            from sulci.embeddings.openai import OpenAIEmbedder
            return OpenAIEmbedder()
        from sulci.embeddings.minilm import MiniLMEmbedder
        return MiniLMEmbedder(name)

    def _load_backend(self, name_or_instance, db_path: str, api_key: Optional[str] = None, gateway_url: str = ""):
        # v0.6.0 (sulci-oss #34 C1d) — accept a pre-constructed Backend
        # instance directly. db_path / api_key / gateway_url are ignored when
        # an instance is supplied, since the caller has already configured the
        # backend with its own connection params. Same duck-typing rationale
        # as _load_embedder above.
        if not isinstance(name_or_instance, str):
            return name_or_instance
        name = name_or_instance
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
            # v0.6.1 (sulci-oss #60) — catch the bare ModuleNotFoundError that
            # `import httpx` raises at the top of sulci/backends/cloud.py when
            # the cloud extra wasn't installed, and re-raise with install
            # guidance. Pre-v0.6.1 the user saw a raw "No module named 'httpx'"
            # which is accurate but unhelpful — no pointer to the fix.
            try:
                from sulci.backends.cloud import SulciCloudBackend
            except ModuleNotFoundError as exc:
                if exc.name == "httpx":
                    raise ImportError(
                        "Cloud backend (Cache(backend=\"sulci\", ...)) requires "
                        "httpx. Install with:\n"
                        "    pip install \"sulci[cloud]\"\n"
                        f"(Original error: {exc})"
                    ) from exc
                raise  # any other ModuleNotFoundError is genuinely unexpected
            return SulciCloudBackend(
                api_key     = resolved_key,
                gateway_url = gateway_url,
            )

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
        *,
        threshold:  Optional[float] = None,
        tenant_id:  Optional[str]   = None,
        user_id:    Optional[str]   = None,
        session_id: Optional[str]   = None,
        plan:       Optional[str]   = None,
    ) -> tuple:
        """
        Check cache for a semantically similar query.

        When ``session_id`` is set and ``context_window > 0``, the lookup
        uses a context-blended embedding so ambiguous queries resolve
        correctly within the current conversation.

        Args:
            query:      The query string to look up.
            threshold:  Optional per-call minimum cosine similarity, overriding
                        the instance-wide ``threshold`` set at construction.
                        ``None`` (default) means "use the instance value" — so
                        every existing caller is unaffected.

                        v0.8.0 (sulci-oss #34). This closes a real correctness
                        gap, not just an ergonomic one. Before it existed, a
                        caller who wanted a different threshold had only one
                        option: let the Cache return a hit at ITS threshold and
                        then discard it. That is what sulci-platform's gateway
                        did — and it meant the LIBRARY decided
                        ``CacheEvent.cache_hit`` (at the instance threshold)
                        while the CALLER decided what the customer saw (at the
                        effective threshold). Similarity in
                        ``[instance, effective)`` emitted a **hit** event while
                        returning a **miss**. Billing, usage rollups, and
                        hit-rate dashboards all counted a hit that never
                        happened.

                        The event and the answer are now decided by the same
                        number, in one place. See ADR 0022 Open-7
                        (sulci-platform), and issues #44 / #59 there.
            tenant_id:  Optional tenant identifier for multi-tenant isolation.
            user_id:    Optional user identifier for personalization.
            session_id: Optional session identifier for context-aware lookup.
            plan:       Optional customer plan tier (e.g. 'free', 'pro').
                        Forwarded onto emitted CacheEvent.plan so downstream
                        billing/analytics can route by plan without a join.
                        v0.5.6 (sulci-oss #36); see sinks/protocol.py.

        Returns:
            (cached_response, similarity_score, context_depth)
            response is None on cache miss.
            context_depth = number of prior turns that influenced lookup.
        """
        _t0 = _time.time()
        self._query_count += 1
        matched_query: Optional[str] = None

        # Resolve the effective threshold ONCE, and use it EVERYWHERE below:
        # the backend search, the telemetry payload, and — critically — the
        # CacheEvent, whose `event_type` is derived from whether the backend
        # returned anything. One number, one decision point.
        #
        # `is None`, never `or`: 0.0 is a legitimate threshold (match anything)
        # and must not be silently replaced by the instance default.
        eff_threshold = self.threshold if threshold is None else float(threshold)
        if not 0.0 <= eff_threshold <= 1.0:
            # A cosine similarity threshold outside [0, 1] is a programming
            # error, not a runtime condition — it can never be satisfied (>1) or
            # can never be missed (<0), and silently clamping would hide the bug
            # rather than surface it. Raising here is safe: the value comes from
            # the caller's own code, so it fails immediately in development,
            # never in response to bad data at runtime.
            raise ValueError(
                f"threshold must be between 0.0 and 1.0, got {eff_threshold!r}"
            )

        if self._is_remote_transport:
            # v0.6.0 (sulci-oss #62) — cloud transport short-circuit.
            # The gateway-side library does the embedding, ANN search, and
            # billing event emit. The SDK forwards the raw query string and
            # the gateway-computed context_depth comes back in the response.
            # self._embedder is NOT called; self._context_vec is NOT called.
            resp, sim, depth = self._backend.remote_get(
                query      = query,
                threshold  = eff_threshold,
                user_id    = user_id if self.personalized else None,
                session_id = session_id,
            )
        else:
            # Self-hosted path — library is the engine in-process.
            vec, depth = self._context_vec(query, session_id)
            # v0.7.2 — backends that expose search_match() also report
            # WHICH stored entry was served, so the emitted CacheEvent
            # can carry matched_query_hash for per-entry hit counting
            # (see sinks/protocol.py). Feature-detected so the Backend
            # protocol and custom backends are untouched.
            if hasattr(self._backend, "search_match"):
                resp, sim, matched_query = self._backend.search_match(
                    embedding = vec,
                    threshold = eff_threshold,
                    tenant_id = tenant_id,
                    user_id   = user_id if self.personalized else None,
                    now       = time.time(),
                )
            else:
                resp, sim  = self._backend.search(
                    embedding = vec,
                    threshold = eff_threshold,
                    tenant_id = tenant_id,
                    user_id   = user_id if self.personalized else None,
                    now       = time.time(),
                )
        latency_ms = round((_time.time() - _t0) * 1000, 2)

        # #42 — count every .get() call so users who use the raw .get()/.set()
        # API (not just cached_call()) see non-zero stats(). Previously these
        # counters only incremented inside cached_call(), which left raw users
        # seeing {"hits": 0, "misses": 0} regardless of activity. cached_call()
        # no longer increments these itself — it goes through .get() like
        # everyone else, so there's no double-counting.
        #
        # #88 (v0.7.1) — saved_cost also now contributes here using the
        # instance default self._cost_per_call (set on Cache.__init__).
        # cached_call() still applies a delta when called with an explicit
        # per-call override; see comment block at the cached_call() hit site.
        if resp is not None:
            self._stats["hits"]       += 1
            self._stats["saved_cost"] += self._cost_per_call
        else:
            self._stats["misses"] += 1

        try:
            if self._telemetry:
                import sys
                _sulci = sys.modules.get("sulci")
                if _sulci is not None:
                    _sulci._emit("cache.get", {
                            "backend":         self.backend,
                            "embedding_model": self.embedding_model,
                            # The threshold actually applied to THIS call, not
                            # the instance default — otherwise a per-call
                            # override is invisible in telemetry and the hit
                            # rate looks inexplicable.
                            "threshold":       eff_threshold,
                            "context_window":  self.context_window,
                            "hits":       1 if resp is not None else 0,
                            "misses":     0 if resp is not None else 1,
                            "latency_ms": latency_ms,
                        })
        except Exception:
            pass

        # v0.5.0 — structured event sink (additive; defaults to NullSink)
        try:
            self._event_sink.emit(CacheEvent(
                event_type      = "hit" if resp is not None else "miss",
                tenant_id       = tenant_id,
                user_id         = user_id,
                session_id      = session_id,
                backend_id      = self.backend,
                embedding_model = self.embedding_model,
                similarity      = float(sim) if sim is not None else None,
                latency_ms      = int(latency_ms),
                context_depth   = depth,
                timestamp       = _time.time(),
                plan            = plan,
                # v0.7.2 — identity of the served entry, hits only.
                # Hash, never text (privacy posture; see protocol.py).
                matched_query_hash = (
                    _query_hash(matched_query)
                    if (resp is not None and matched_query) else None
                ),
            ))
        except Exception:
            # Sink failures must not break Cache calls
            pass

        return resp, sim, depth

    def set(
        self,
        query:      str,
        response:   str,
        *,
        tenant_id:  Optional[str]  = None,
        user_id:    Optional[str]  = None,
        session_id: Optional[str]  = None,
        metadata:   Optional[dict] = None,
        plan:       Optional[str]  = None,
    ) -> None:
        """
        Store a query-response pair in the cache.

        The raw (un-blended) query embedding is stored so entries are
        reusable across different sessions and conversation states.
        Context blending only happens at lookup time.

        When ``session_id`` is provided, the turn is also recorded in
        the session window to inform future context-blended lookups.

        ``plan`` (v0.5.6, sulci-oss #36) is forwarded onto the emitted
        CacheEvent.plan; see ``Cache.get`` for the rationale.
        """
        _t0 = _time.time()
        if self._is_remote_transport:
            # v0.6.0 (sulci-oss #62) — cloud transport short-circuit.
            # The gateway-side library handles embedding, Qdrant upsert, and
            # the cache.set billing event. session_id is forwarded so the
            # gateway-side context window updates; local self._sessions is
            # not touched (it would track stale turn state since this Cache
            # instance has no engine-state).
            self._backend.remote_set(
                query       = query,
                response    = response,
                user_id     = user_id if self.personalized else None,
                session_id  = session_id,
                ttl_seconds = self.ttl_seconds,
            )
            raw_vec = None   # not computed locally; session-window record skipped below
        else:
            # Self-hosted path — library is the engine in-process.
            raw_vec = self._embedder.embed(query)
            key     = hashlib.sha256(query.encode()).hexdigest()[:16]
            expires = time.time() + self.ttl_seconds if self.ttl_seconds else None
            self._backend.store(
                key       = key,
                query     = query,
                response  = response,
                embedding = raw_vec,
                tenant_id = tenant_id,
                user_id   = user_id if self.personalized else None,
                expires   = expires,
                metadata  = metadata or {},
            )
        if session_id and self._sessions and raw_vec is not None:
            self._record_turn(session_id, query, response, raw_vec)

        # Telemetry — legacy emit pipe (sulci.connect()).
        # Mirror the cache.get behaviour: only emit when this Cache
        # instance has telemetry enabled AND the user has called
        # sulci.connect(api_key=...). Latency is measured from raw_vec
        # embedding through to backend store completion.
        latency_ms = round((_time.time() - _t0) * 1000, 2)
        try:
            if self._telemetry:
                import sys
                _sulci = sys.modules.get("sulci")
                if _sulci is not None:
                    _sulci._emit("cache.set", {
                        "backend":         self.backend,
                        "embedding_model": self.embedding_model,
                        "threshold":       self.threshold,
                        "context_window":  self.context_window,
                        "latency_ms":      latency_ms,
                    })
        except Exception:
            pass

        # v0.5.0 — structured event sink (additive; defaults to NullSink)
        try:
            self._event_sink.emit(CacheEvent(
                event_type      = "set",
                tenant_id       = tenant_id,
                user_id         = user_id,
                session_id      = session_id,
                backend_id      = self.backend,
                embedding_model = self.embedding_model,
                timestamp       = time.time(),
                plan            = plan,
            ))
        except Exception:
            pass

    def cached_call(
        self,
        query:         str,
        llm_fn:        Callable,
        *,
        threshold:     Optional[float] = None,
        tenant_id:     Optional[str]   = None,
        user_id:       Optional[str]   = None,
        session_id:    Optional[str]   = None,
        cost_per_call: Optional[float] = None,
        plan:          Optional[str]   = None,
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
                           If None (default), uses ``Cache.cost_per_call``
                           (set on construction, default $0.005). Pass an
                           explicit value to override for this call only —
                           the delta is reflected in ``stats()['saved_cost']``.
            plan:          Optional customer plan tier; forwarded onto
                           the underlying .get/.set calls and thence onto
                           emitted CacheEvent.plan. v0.5.6 (sulci-oss #36).
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
        # v0.8.0 (#34) — forward the per-call threshold. cached_call() is the
        # ergonomic front door; a kwarg it silently swallows is a kwarg that
        # doesn't exist to most users.
        hit, sim, depth = self.get(
            query,
            threshold  = threshold,
            tenant_id  = tenant_id,
            user_id    = user_id,
            session_id = session_id,
            plan       = plan,
        )
        ms              = (time.perf_counter() - t0) * 1000

        if hit is not None:
            # _stats["hits"] AND _stats["saved_cost"] are both incremented
            # inside .get() — see #42 / #88 notes there. cached_call() applies
            # only the *delta* when an explicit per-call cost_per_call override
            # was passed, so per-call accounting still works for users who
            # mix instance-default and per-call costs.
            if cost_per_call is not None and cost_per_call != self._cost_per_call:
                self._stats["saved_cost"] += (cost_per_call - self._cost_per_call)
            # Record turn even on hits so future queries see this exchange.
            # v0.6.1 (sulci-oss #60) — skip on remote transport: self._embedder
            # is None there (gateway-side library tracks session state for the
            # cloud path); without this gate, a session-aware cloud Cache would
            # crash here on None.embed(). Mirrors the `raw_vec is not None`
            # guard already present in Cache.set's session-record path.
            if session_id and self._sessions and not self._is_remote_transport:
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
        self.set(query, response, tenant_id=tenant_id, user_id=user_id, session_id=session_id, plan=plan)
        ms = (time.perf_counter() - t0) * 1000
        # _stats["misses"] is incremented inside .get() above — see #42.
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
        # D15 — passive nudge after 100 queries on this instance.
        # Wrapped to never affect the returned stats dict on failure.
        try:
            self._maybe_nudge()
        except Exception:
            pass
        return base

    def _maybe_nudge(self) -> None:
        """Print the one-shot connect() nudge to stderr if conditions match.

        See module docstring near ``_NUDGE_SHOWN`` for the suppression
        order. This method is called from :meth:`stats`; failures are
        swallowed by the caller.
        """
        global _NUDGE_SHOWN

        if _NUDGE_SHOWN:
            return
        if os.environ.get("SULCI_QUIET") == "1":
            return
        if self._query_count < _NUDGE_THRESHOLD:
            return

        # Already connected? Don't nudge — they've taken the action.
        try:
            import sys as _sys
            _sulci = _sys.modules.get("sulci")
            if _sulci is not None and getattr(_sulci, "_telemetry_enabled", False):
                _NUDGE_SHOWN = True   # don't re-check on every stats() call
                return
        except Exception:
            pass

        _NUDGE_SHOWN = True
        import sys as _sys
        _sys.stderr.write(
            f"[sulci] {self._query_count} cached queries — connect to sulci.io "
            "for free, persistent stats: https://sulci.io/connect "
            "(set SULCI_QUIET=1 to silence)\n"
        )

    def clear(self) -> None:
        """Remove all cached entries, reset stats, and clear all sessions."""
        self._backend.clear()
        self._stats = {"hits": 0, "misses": 0, "saved_cost": 0.0}
        if self._sessions:
            self._sessions.clear_all()

        # v0.5.0 — structured event sink (additive; defaults to NullSink)
        try:
            self._event_sink.emit(CacheEvent(
                event_type      = "clear",
                backend_id      = self.backend,
                embedding_model = self.embedding_model,
                timestamp       = time.time(),
            ))
        except Exception:
            pass

    def __repr__(self) -> str:
        ctx = f", context_window={self.context_window}" if self.context_window else ""
        return (
            f"Cache(backend={self.backend!r}, "
            f"threshold={self.threshold}"
            f"{ctx}, "
            f"hits={self._stats['hits']}, "
            f"misses={self._stats['misses']})"
        )
