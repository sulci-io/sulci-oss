# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_session_store_injection.py — v0.5.0 Cache(session_store=, event_sink=) tests.

Verifies the B1 adapter contract from ADR 0007:
  - Default Cache (no session_store kwarg) uses _BuiltinSessionStore — unchanged from v0.4.x
  - Cache(session_store=InMemorySessionStore()) wraps via _ProtocolAdaptedSessionStore
  - Returned ContextWindow's user-vector add_turn() round-trips to inner store
  - EventSink fires on hit/miss/set with the documented field allowlist
"""
from unittest.mock import MagicMock

from sulci import Cache
from sulci.context import SessionStore as LegacySessionStore
from sulci.sessions import InMemorySessionStore
from sulci.sinks import CacheEvent, NullSink


# ═══════════════════════════════════════════════════════════════════════
# Constructor accepts new kwargs without breaking v0.4.x calls
# ═══════════════════════════════════════════════════════════════════════

class TestConstructorAcceptsNewKwargs:
    def test_v04_call_still_works_unchanged(self, tmp_path):
        """Cache(...) with no session_store/event_sink kwargs preserves v0.4.x behavior."""
        c = Cache(backend="sqlite", db_path=str(tmp_path), context_window=4)
        # Internal session manager is the legacy class
        assert isinstance(c._sessions, LegacySessionStore)
        # No event sink injected → NullSink default
        assert isinstance(c._event_sink, NullSink)

    def test_session_store_kwarg_accepted(self, tmp_path):
        """Cache(session_store=...) accepts any sulci.sessions.SessionStore impl."""
        store = InMemorySessionStore()
        c = Cache(
            backend="sqlite", db_path=str(tmp_path),
            context_window=4, session_store=store,
        )
        # Internal manager is the adapter, not the legacy class
        assert not isinstance(c._sessions, LegacySessionStore)
        assert c._sessions._inner is store


# ═══════════════════════════════════════════════════════════════════════
# Default path is byte-for-byte unchanged from v0.4.x
# ═══════════════════════════════════════════════════════════════════════

class TestDefaultPathUnchanged:
    def test_default_session_store_is_legacy_class(self, tmp_path):
        """When session_store=None and context_window>0, use _BuiltinSessionStore."""
        c = Cache(backend="sqlite", db_path=str(tmp_path), context_window=4)
        assert isinstance(c._sessions, LegacySessionStore)

    def test_no_context_window_means_no_session_store(self, tmp_path):
        """context_window=0 → _sessions is None (stateless cache)."""
        c = Cache(backend="sqlite", db_path=str(tmp_path), context_window=0)
        assert c._sessions is None


# ═══════════════════════════════════════════════════════════════════════
# Injected SessionStore: round-trip through the adapter
# ═══════════════════════════════════════════════════════════════════════

class TestInjectedSessionStoreRoundTrip:
    def test_set_pushes_user_vector_to_inner_store(self, tmp_path):
        """cache.set(session_id=...) triggers inner.append() via the adapter wrapper."""
        store = InMemorySessionStore()
        c = Cache(
            backend="sqlite", db_path=str(tmp_path),
            context_window=4, session_store=store,
        )
        c.set("hello there", "hi back", session_id="user-1")
        # Inner store now has one user-query vector for this session
        history = store.get("user-1")
        assert len(history) == 1
        assert all(isinstance(v, float) for v in history[0])  # vector of floats

    def test_get_rebuilds_window_from_inner(self, tmp_path):
        """cache.get(session_id=...) returns a window seeded from inner.get()."""
        store = InMemorySessionStore()
        # Pre-seed inner directly
        store.append("user-2", [0.1] * 384)
        store.append("user-2", [0.2] * 384)
        c = Cache(
            backend="sqlite", db_path=str(tmp_path),
            context_window=4, session_store=store,
        )
        # Adapter rebuilds window with depth=2 from inner's two vectors
        window = c._sessions.get("user-2")
        assert window.depth == 2

    def test_clear_context_clears_inner(self, tmp_path):
        """cache.clear_context(sid) propagates to inner.clear(sid)."""
        store = InMemorySessionStore()
        store.append("user-3", [0.1] * 384)
        c = Cache(
            backend="sqlite", db_path=str(tmp_path),
            context_window=4, session_store=store,
        )
        c.clear_context("user-3")
        assert store.get("user-3") == []


# ═══════════════════════════════════════════════════════════════════════
# EventSink wiring
# ═══════════════════════════════════════════════════════════════════════

class TestEventSink:
    def test_default_event_sink_is_null(self, tmp_path):
        c = Cache(backend="sqlite", db_path=str(tmp_path))
        assert isinstance(c._event_sink, NullSink)

    def test_emits_on_miss(self, tmp_path):
        sink = MagicMock()
        c = Cache(backend="sqlite", db_path=str(tmp_path), event_sink=sink)
        c.get("nothing in cache yet")
        # Sink received exactly one CacheEvent with event_type='miss'
        assert sink.emit.call_count == 1
        evt = sink.emit.call_args[0][0]
        assert isinstance(evt, CacheEvent)
        assert evt.event_type == "miss"
        assert evt.backend_id == "sqlite"

    def test_emits_on_hit(self, tmp_path):
        sink = MagicMock()
        c = Cache(backend="sqlite", db_path=str(tmp_path), event_sink=sink)
        c.set("hello world", "greeting response")
        # set + get = two emit calls; second one should be event_type='hit'
        c.get("hello world")
        assert sink.emit.call_count >= 2
        last_evt = sink.emit.call_args_list[-1][0][0]
        assert last_evt.event_type == "hit"
        assert last_evt.similarity is not None

    def test_emits_on_set(self, tmp_path):
        sink = MagicMock()
        c = Cache(backend="sqlite", db_path=str(tmp_path), event_sink=sink)
        c.set("query x", "response y", tenant_id="acme")
        # First emit is the set event
        first_evt = sink.emit.call_args_list[0][0][0]
        assert first_evt.event_type == "set"
        assert first_evt.tenant_id == "acme"

    def test_sink_failure_does_not_break_cache(self, tmp_path):
        """Per protocol, EventSink failures must never propagate to Cache caller."""
        bad_sink = MagicMock()
        bad_sink.emit.side_effect = RuntimeError("sink exploded")
        c = Cache(backend="sqlite", db_path=str(tmp_path), event_sink=bad_sink)
        # Must not raise even though the sink raises on every emit
        c.set("query", "response")
        result, sim, depth = c.get("query")
        assert result == "response"
