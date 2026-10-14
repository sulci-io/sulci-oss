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
from sulci.sessions import InMemorySessionStore, scoped_session_id
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
        # Inner store now has one user-query vector for this session — under its scoped id (0.9.3)
        assert store.get("user-1") == []
        history = store.get(scoped_session_id(None, None, "user-1"))
        assert len(history) == 1
        assert all(isinstance(v, float) for v in history[0])  # vector of floats

    def test_get_rebuilds_window_from_inner(self, tmp_path):
        """cache.get(session_id=...) returns a window seeded from inner.get()."""
        store = InMemorySessionStore()
        # Pre-seed inner directly
        sid = scoped_session_id(None, None, "user-2")   # 0.9.3: the store holds scoped ids
        store.append(sid, [0.1] * 384)
        store.append(sid, [0.2] * 384)
        c = Cache(
            backend="sqlite", db_path=str(tmp_path),
            context_window=4, session_store=store,
        )
        # Adapter rebuilds window with depth=2 from inner's two vectors
        window = c.get_context("user-2")
        assert window.depth == 2

    def test_clear_context_clears_inner(self, tmp_path):
        """cache.clear_context(sid) propagates to inner.clear(sid)."""
        store = InMemorySessionStore()
        sid = scoped_session_id(None, None, "user-3")   # 0.9.3: the store holds scoped ids
        store.append(sid, [0.1] * 384)
        c = Cache(
            backend="sqlite", db_path=str(tmp_path),
            context_window=4, session_store=store,
        )
        c.clear_context("user-3")
        assert store.get(sid) == []


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



# ═══════════════════════════════════════════════════════════════════════
# 0.10.0: a failing injected store — writes raise, reads warn
# ═══════════════════════════════════════════════════════════════════════

class _FailingStore(InMemorySessionStore):
    """An InMemorySessionStore whose named operations raise ConnectionError."""
    def __init__(self, *failing):
        super().__init__()
        self.failing = set(failing)

    def _maybe_fail(self, op):
        if op in self.failing:
            raise ConnectionError(f"session store down ({op})")

    def append(self, *a, **k):
        self._maybe_fail("append"); return super().append(*a, **k)

    reads = 0

    def get(self, *a, **k):
        self.reads += 1
        self._maybe_fail("get"); return super().get(*a, **k)

    def clear(self, *a, **k):
        self._maybe_fail("clear"); return super().clear(*a, **k)

    def summary(self, *a, **k):
        self._maybe_fail("summary"); return super().summary(*a, **k)


class TestFailingSessionStore:
    """Measured on 0.9.3: against a store whose every call raised, set, clear_context and get each returned
    normally and nothing was logged — a store that was down looked like a conversation with no history."""

    Q, R = "how do I reset the router", "hold the reset button for ten seconds"

    def _cache(self, tmp_path, fake_embedder, store, **kw):
        return Cache(backend="sqlite", db_path=str(tmp_path / "db"), embedding_model=fake_embedder,
                     context_window=4, context_threshold=0.85, session_store=store, telemetry=False,
                     **kw)

    def test_set_raises_and_the_entry_is_stored(self, tmp_path, fake_embedder):
        import pytest
        from sulci.sessions import SulciSessionError
        c = self._cache(tmp_path, fake_embedder, _FailingStore("append", "get", "clear"))
        with pytest.raises(SulciSessionError, match="WAS stored") as info:
            c.set(self.Q, self.R, session_id="s1")
        assert isinstance(info.value.__cause__, ConnectionError)
        assert c.get(self.Q)[0] == self.R

    def test_set_emits_its_event_before_raising(self, tmp_path, fake_embedder):
        import pytest
        from sulci.sessions import SulciSessionError
        sink = MagicMock()
        c = self._cache(tmp_path, fake_embedder, _FailingStore("append"), event_sink=sink)
        with pytest.raises(SulciSessionError):
            c.set(self.Q, self.R, session_id="s1")
        assert [call.args[0].event_type for call in sink.emit.call_args_list] == ["set"]

    def test_set_emits_telemetry_before_raising(self, tmp_path, fake_embedder, monkeypatch):
        import pytest
        import sulci
        from sulci.sessions import SulciSessionError
        emitted = []
        monkeypatch.setattr(sulci, "_emit", lambda name, payload: emitted.append(name), raising=False)
        c = self._cache(tmp_path, fake_embedder, _FailingStore("append"))
        c._telemetry = True
        with pytest.raises(SulciSessionError):
            c.set(self.Q, self.R, session_id="s1")
        assert emitted == ["cache.set"]

    def test_clear_raises_after_the_entries_are_cleared(self, tmp_path, fake_embedder):
        import pytest
        from sulci.sessions import SulciSessionError
        store = _FailingStore()
        c = self._cache(tmp_path, fake_embedder, store)
        c.set(self.Q, self.R, session_id="s1")
        c.get(self.Q)
        def down():
            raise ConnectionError("session store down (clear_all)")
        store.clear_all = down
        sink = MagicMock()
        c._event_sink = sink
        with pytest.raises(SulciSessionError, match="WERE cleared.*NOT cleared") as info:
            c.clear()
        assert [call.args[0].event_type for call in sink.emit.call_args_list] == ["clear"]
        assert isinstance(info.value.__cause__, ConnectionError)
        assert c.stats()["hits"] == 1                               # stats as they were
        assert c.get(self.Q)[0] is None                             # the entries are gone

    def test_a_read_that_fails_part_way_uses_no_history(self, tmp_path, fake_embedder, caplog):
        import logging
        store = _FailingStore()
        c = self._cache(tmp_path, fake_embedder, store)
        def partial(*a, **k):
            yield fake_embedder.embed("first")
            raise ConnectionError("connection reset")
        store.get = partial
        with caplog.at_level(logging.WARNING, logger="sulci"):
            assert c.get_context("s1").depth == 0
        assert any("failed to read conversation history" in r.getMessage() for r in caplog.records)

    def test_set_does_not_read_the_history_to_record_a_turn(self, tmp_path, fake_embedder, caplog):
        import logging
        store = _FailingStore("get")
        c = self._cache(tmp_path, fake_embedder, store)
        with caplog.at_level(logging.WARNING, logger="sulci"):
            c.set(self.Q, self.R, session_id="s1")
        assert caplog.records == [] and store.reads == 0
        store.failing.clear()
        assert len(store.get(scoped_session_id(None, None, "s1"))) == 1

    def test_clear_context_raises_and_says_nothing_was_cleared(self, tmp_path, fake_embedder):
        import pytest
        from sulci.sessions import SulciSessionError
        store = _FailingStore("clear")
        c = self._cache(tmp_path, fake_embedder, store)
        c.set(self.Q, self.R, session_id="s1")
        with pytest.raises(SulciSessionError, match="NOT cleared") as info:
            c.clear_context("s1")
        assert isinstance(info.value.__cause__, ConnectionError)
        assert len(store.get(scoped_session_id(None, None, "s1"))) == 1

    def test_get_warns_and_answers_without_history(self, tmp_path, fake_embedder, caplog):
        import logging
        c = self._cache(tmp_path, fake_embedder, _FailingStore("get"))
        c.set(self.Q, self.R)
        with caplog.at_level(logging.WARNING, logger="sulci"):
            response, _, depth = c.get(self.Q, session_id="s1")
        assert (response, depth) == (self.R, 0)
        assert any("failed to read conversation history" in r.getMessage() for r in caplog.records)

    def test_get_context_and_context_summary_warn(self, tmp_path, fake_embedder, caplog):
        import logging
        c = self._cache(tmp_path, fake_embedder, _FailingStore("get", "summary"))
        with caplog.at_level(logging.WARNING, logger="sulci"):
            assert c.get_context("s1").depth == 0
            assert c.context_summary("s1")["session_id"] == "s1"
            assert c.context_summary()["active_sessions"] == 0
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 3

    def test_a_window_from_get_context_raises_on_a_failed_write(self, tmp_path, fake_embedder):
        import pytest
        from sulci.sessions import SulciSessionError
        c = self._cache(tmp_path, fake_embedder, _FailingStore("append"))
        window = c.get_context("s1")
        with pytest.raises(SulciSessionError):
            window.add_turn("hello", role="user", embedding=fake_embedder.embed("hello"))
        window.add_turn("hi", role="assistant")                    # window-local: never reaches the store

    def test_cached_call_returns_the_llm_response_and_warns(self, tmp_path, fake_embedder, caplog):
        import logging
        calls = []
        c = self._cache(tmp_path, fake_embedder, _FailingStore("append"))
        with caplog.at_level(logging.WARNING, logger="sulci"):
            out = c.cached_call(self.Q, lambda q: calls.append(q) or self.R, session_id="s1")
        assert (out["response"], out["source"], calls) == (self.R, "llm", [self.Q])
        assert any("cached_call returned the LLM response" in r.getMessage() for r in caplog.records)
        assert c.get(self.Q)[0] == self.R

    def test_cached_call_returns_a_hit_and_warns(self, tmp_path, fake_embedder, caplog):
        import logging
        c = self._cache(tmp_path, fake_embedder, _FailingStore("append"))
        c.set(self.Q, self.R)
        with caplog.at_level(logging.WARNING, logger="sulci"):
            out = c.cached_call(self.Q, lambda q: "never called", session_id="s1")
        assert (out["response"], out["source"]) == (self.R, "cache")
        assert any("cached_call returned the cached response" in r.getMessage() for r in caplog.records)

    def test_async_cache_aset_and_aclear_context_raise(self, tmp_path, fake_embedder):
        import asyncio
        import pytest
        from sulci import AsyncCache
        from sulci.sessions import SulciSessionError
        c = AsyncCache(backend="sqlite", db_path=str(tmp_path / "db"), embedding_model=fake_embedder,
                       context_window=4, session_store=_FailingStore("append", "clear"), telemetry=False)
        with pytest.raises(SulciSessionError, match="WAS stored"):
            asyncio.run(c.aset(self.Q, self.R, session_id="s1"))
        with pytest.raises(SulciSessionError, match="NOT cleared"):
            asyncio.run(c.aclear_context("s1"))

    def test_a_working_store_logs_nothing(self, tmp_path, fake_embedder, caplog):
        import logging
        c = self._cache(tmp_path, fake_embedder, _FailingStore())
        with caplog.at_level(logging.WARNING, logger="sulci"):
            c.set(self.Q, self.R, session_id="s1")
            c.get(self.Q, session_id="s1")
            c.cached_call(self.Q, lambda q: "never called", session_id="s1")
            c.clear_context("s1")
        assert caplog.records == []
