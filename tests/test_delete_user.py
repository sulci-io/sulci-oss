# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_delete_user.py — ``Cache.delete_user`` / ``AsyncCache.adelete_user`` (0.10.0).

One end user of one tenant is erased: the backend's entries (``Backend.delete_user``) and the session store's
conversation history (``clear_scope(user_scope(tenant, user))``), returned as ``{"entries": N, "sessions": M}``.
Measured on 0.9.3 (``measure_oss16.py``): ``Cache`` had no ``delete_user`` (``AttributeError``); on SQLite a user's
entries — an expired one included — stayed until ``clear()``, and so did their sessions.

What each backend's own ``delete_user`` must do is the conformance suite's ``TestUserErasure``; this file is the
``Cache`` layer on SQLite (the backend that has it in this patch), a stand-in backend for failures, and the
managed backend over httpx's ``MockTransport``.
"""
from __future__ import annotations

import asyncio
import pickle
import socket
import sqlite3
import os
import time
import uuid
from functools import lru_cache

import httpx
import pytest

from sulci import AsyncCache
from sulci.core import Cache
from sulci.errors import SulciErasureError
from sulci.sessions import InMemorySessionStore
from sulci.sessions.scope import user_scope
from sulci.sinks import CacheEvent

from _fake_embedder import FakeEmbedder


def _cache(tmp_path, **kw):
    kw.setdefault("personalized", True)
    kw.setdefault("context_window", 4)
    return Cache(backend="sqlite", db_path=str(tmp_path), embedding_model=FakeEmbedder(), telemetry=False, **kw)


def _rows(tmp_path):
    c = sqlite3.connect(os.path.join(str(tmp_path), "sulci.db"))
    try:
        return c.execute("SELECT tenant_id, user_id, COUNT(*) FROM cache GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
    finally:
        c.close()


def _seed(cache):
    """u-42 has two entries and one session in t1; the same id in t2, another user and the shared user of t1
    have one each."""
    for t, u, q in [("t1", "u-42", "alpha"), ("t1", "u-42", "beta"), ("t2", "u-42", "alpha"),
                    ("t1", "u-7", "alpha"), ("t1", None, "alpha")]:
        cache.set(f"question {q}", f"{q} for {t}/{u}", tenant_id=t, user_id=u, session_id="s1")


SEEDED = [("t1", "global", 1), ("t1", "u-42", 2), ("t1", "u-7", 1), ("t2", "u-42", 1)]
AFTER = [("t1", "global", 1), ("t1", "u-7", 1), ("t2", "u-42", 1)]


class _Sink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)

    def flush(self):
        pass


# ── what is erased ────────────────────────────────────────────────────────────────────────────────────────────

class TestErases:
    def test_entries_and_sessions_of_one_user_in_one_tenant(self, tmp_path):
        c = _cache(tmp_path)
        _seed(c)
        assert _rows(tmp_path) == SEEDED
        assert c.delete_user("u-42", tenant_id="t1") == {"entries": 2, "sessions": 1}
        assert _rows(tmp_path) == AFTER
        assert sorted(c._sessions._windows) == sorted(
            k for k in c._sessions._windows if not k.startswith(user_scope("t1", "u-42")))
        assert len(c._sessions._windows) == 3

    def test_lookups_for_the_user_miss_and_everyone_else_still_hits(self, tmp_path):
        c = _cache(tmp_path, context_window=0)
        _seed(c)
        c.delete_user("u-42", tenant_id="t1")
        assert c.get("question alpha", tenant_id="t1", user_id="u-42")[0] is None
        assert c.get("question beta", tenant_id="t1", user_id="u-42")[0] is None
        assert c.get("question alpha", tenant_id="t2", user_id="u-42")[0] == "alpha for t2/u-42"
        assert c.get("question alpha", tenant_id="t1", user_id="u-7")[0] == "alpha for t1/u-7"
        assert c.get("question alpha", tenant_id="t1")[0] == "alpha for t1/None"

    def test_an_expired_entry_is_erased_and_counted(self, tmp_path):
        c = _cache(tmp_path)
        c._backend.store("k", "old", "old answer", FakeEmbedder().embed("old"), tenant_id="t1", user_id="u-42",
                         expires=time.time() - 60)
        assert c.delete_user("u-42", tenant_id="t1") == {"entries": 1, "sessions": 0}
        assert _rows(tmp_path) == []

    def test_a_second_call_returns_zero(self, tmp_path):
        c = _cache(tmp_path)
        _seed(c)
        c.delete_user("u-42", tenant_id="t1")
        assert c.delete_user("u-42", tenant_id="t1") == {"entries": 0, "sessions": 0}
        assert _rows(tmp_path) == AFTER

    def test_no_tenant_is_the_cache_without_a_tenant(self, tmp_path):
        c = _cache(tmp_path)
        c.set("q", "a", user_id="u-42", session_id="s1")
        c.set("q", "a", tenant_id="t1", user_id="u-42", session_id="s1")
        assert c.delete_user("u-42") == {"entries": 1, "sessions": 1}
        assert _rows(tmp_path) == [("t1", "u-42", 1)]

    def test_without_a_session_store_sessions_are_zero(self, tmp_path):
        c = _cache(tmp_path, context_window=0)
        assert c._sessions is None
        _seed(c)
        assert c.delete_user("u-42", tenant_id="t1") == {"entries": 2, "sessions": 0}

    def test_an_injected_store_is_erased_by_the_users_scope(self, tmp_path):
        store = InMemorySessionStore()
        c = _cache(tmp_path, session_store=store)
        _seed(c)
        c.set("question gamma", "g", tenant_id="t1", user_id="u-42", session_id="s2")
        assert len(store._data) == 5
        assert c.delete_user("u-42", tenant_id="t1") == {"entries": 3, "sessions": 2}
        assert len(store._data) == 3
        assert not [k for k in store._data if k.startswith(user_scope("t1", "u-42"))]


@lru_cache(maxsize=1)
def _redis_available() -> bool:
    for family, addr in ((socket.AF_INET, ("127.0.0.1", 6379)), (socket.AF_INET6, ("::1", 6379))):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.25)
                if sock.connect_ex(addr) == 0:
                    return True
        except OSError:
            continue
    return False


class TestRedisSessionStore:
    def test_the_users_sessions_and_index_are_erased(self, tmp_path):
        if not _redis_available():
            pytest.skip("redis-server not running on localhost:6379")
        import redis
        from sulci.sessions import RedisSessionStore
        client = redis.Redis(host="localhost", port=6379, db=15, socket_connect_timeout=0.25, socket_timeout=0.25)
        prefix = f"sulci:test:du:{uuid.uuid4().hex[:8]}:"
        store = RedisSessionStore(client, key_prefix=prefix)
        try:
            c = _cache(tmp_path, session_store=store)
            _seed(c)
            assert c.delete_user("u-42", tenant_id="t1") == {"entries": 2, "sessions": 1}
            assert store.clear_scope(user_scope("t1", "u-42")) == 0
            assert store.clear_scope(user_scope("t2", "u-42")) == 1
        finally:
            store.clear_all()
            for k in client.scan_iter(match=prefix + "*"):
                client.delete(k)


# ── refusals: nothing is touched ──────────────────────────────────────────────────────────────────────────────

class TestRefusals:
    @pytest.mark.parametrize("uid", [None, "", "global", 0, False, b"", "\ud800"], ids=repr)
    def test_a_reserved_or_invalid_user_id(self, tmp_path, uid):
        c = _cache(tmp_path)
        _seed(c)
        with pytest.raises(ValueError):
            c.delete_user(uid, tenant_id="t1")
        assert _rows(tmp_path) == SEEDED and len(c._sessions._windows) == 4

    @pytest.mark.parametrize("tid", ["", "global", 0, "\ud800"], ids=repr)
    def test_a_reserved_or_invalid_tenant_id(self, tmp_path, tid):
        c = _cache(tmp_path)
        _seed(c)
        with pytest.raises(ValueError):
            c.delete_user("u-42", tenant_id=tid)
        assert _rows(tmp_path) == SEEDED and len(c._sessions._windows) == 4

    def test_a_cache_that_is_not_personalized(self, tmp_path):
        c = _cache(tmp_path, personalized=False)
        c.set("q", "a", tenant_id="t1", user_id="u-42", session_id="s1")   # filed under no user
        with pytest.raises(ValueError, match="personalized=True") as e:
            c.delete_user("u-42", tenant_id="t1")
        assert "u-42" not in str(e.value)
        assert _rows(tmp_path) == [("t1", "global", 1)] and len(c._sessions._windows) == 1

    def test_a_backend_without_delete_user(self, tmp_path):
        class NoErase:
            def store(self, *a, **k): pass
            def search(self, *a, **k): return None, 0.0
            def clear(self): pass
        c = Cache(backend=NoErase(), embedding_model=FakeEmbedder(), personalized=True, context_window=4,
                  telemetry=False)
        c.set("q", "a", tenant_id="t1", user_id="u-42", session_id="s1")
        with pytest.raises(NotImplementedError, match="NoErase"):
            c.delete_user("u-42", tenant_id="t1")
        assert len(c._sessions._windows) == 1

    def test_the_id_is_positional_and_the_tenant_keyword_only(self, tmp_path):
        c = _cache(tmp_path)
        with pytest.raises(TypeError):
            c.delete_user("u-42", "t1")


# ── failures: what was erased is said ─────────────────────────────────────────────────────────────────────────

class _Backend:
    """A backend whose delete_user does what it is told. Holds nothing."""
    def __init__(self, result=0):
        self.result, self.calls = result, []

    def store(self, *a, **k): pass
    def search(self, *a, **k): return None, 0.0
    def clear(self): pass

    def delete_user(self, user_id, *, tenant_id=None):
        self.calls.append((user_id, tenant_id))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _Store(InMemorySessionStore):
    def __init__(self, fail=None, result=None):
        super().__init__()
        self.fail, self.result, self.scopes = fail, result, []

    def clear_scope(self, scope):
        self.scopes.append(scope)
        if self.fail:
            raise self.fail
        return super().clear_scope(scope) if self.result is None else self.result


def _with(backend, store=None, **kw):
    return Cache(backend=backend, embedding_model=FakeEmbedder(), personalized=True, context_window=4,
                 telemetry=False, session_store=store, **kw)


class TestFailures:
    def test_the_backend_fails_and_sessions_are_not_attempted(self):
        store = _Store()
        b = _Backend(RuntimeError("disk I/O error near u-42"))
        c = _with(b, store)
        with pytest.raises(SulciErasureError) as e:
            c.delete_user("u-42", tenant_id="t1")
        assert (e.value.entries, e.value.sessions, e.value.operation, e.value.error) == \
            (None, None, "delete_user", "RuntimeError")
        assert isinstance(e.value.__cause__, RuntimeError)
        assert "u-42" not in str(e.value) and "not touched" in str(e.value)
        assert store.scopes == [] and b.calls == [("u-42", "t1")]

    @pytest.mark.parametrize("bad", [None, -1, True, 1.0, "3"], ids=repr)
    def test_the_backend_returns_no_count(self, bad):
        store = _Store()
        with pytest.raises(SulciErasureError) as e:
            _with(_Backend(bad), store).delete_user("u-42", tenant_id="t1")
        assert e.value.error == "not_a_count" and e.value.entries is None
        assert store.scopes == []

    def test_the_session_store_fails_after_the_entries(self):
        store = _Store(fail=ConnectionError("redis down"))
        c = _with(_Backend(3), store)
        with pytest.raises(SulciErasureError) as e:
            c.delete_user("u-42", tenant_id="t1")
        assert (e.value.entries, e.value.sessions, e.value.error) == (3, None, "ConnectionError")
        assert "3 cache entries WERE erased" in str(e.value) and "NOT erased" in str(e.value)
        assert isinstance(e.value.__cause__, ConnectionError)
        assert store.scopes == [user_scope("t1", "u-42")]

    def test_a_retry_completes_the_erasure(self, tmp_path):
        store = _Store(fail=ConnectionError("redis down"))
        c = _cache(tmp_path, session_store=store)
        _seed(c)
        with pytest.raises(SulciErasureError) as e:
            c.delete_user("u-42", tenant_id="t1")
        assert e.value.entries == 2 and _rows(tmp_path) == AFTER
        store.fail = None
        assert c.delete_user("u-42", tenant_id="t1") == {"entries": 0, "sessions": 1}

    def test_the_session_store_returns_no_count(self):
        with pytest.raises(SulciErasureError) as e:
            _with(_Backend(2), _Store(result="2")).delete_user("u-42", tenant_id="t1")
        assert (e.value.entries, e.value.sessions, e.value.error) == (2, None, "not_a_count")

    def test_the_error_pickles_with_its_counts(self):
        e = SulciErasureError("delete_user", None, "ConnectionError", entries=3, detail="the user's 3 …")
        f = pickle.loads(pickle.dumps(e))
        assert (f.operation, f.status, f.error, f.entries, f.sessions, str(f)) == \
            ("delete_user", None, "ConnectionError", 3, None, str(e))

    def test_one_error_class_for_both_paths(self):
        from sulci.backends import cloud
        assert cloud.SulciErasureError is SulciErasureError


class TestSQLiteThreads:
    def test_every_use_of_the_shared_connection_holds_the_lock(self, tmp_path):
        """AsyncCache runs each call in a worker thread on one SQLiteBackend, so one sqlite3 connection. Measured
        before the lock, with one writer thread beside another (measure_sqlite_race.py): about one call in five
        raised, and delete_user returned 0 with the user's rows still there (8 in 300). Pinned here
        deterministically: the connection refuses any use from a thread that does not hold the backend's lock."""
        from sulci.backends.sqlite import SQLiteBackend
        b = SQLiteBackend(str(tmp_path))

        class Guarded:
            def __init__(self, conn):
                self._c = conn

            def __getattr__(self, name):
                attr = getattr(self._c, name)
                if name in ("execute", "commit", "executemany", "rollback"):
                    def checked(*a, **k):
                        assert b._lock._is_owned(), f"{name} without the lock"
                        return attr(*a, **k)
                    return checked
                return attr
        b._conn = Guarded(b._conn)
        v = FakeEmbedder().embed("q")
        b.store("k", "q", "a", v, tenant_id="t1", user_id="u-42")
        assert b.search(v, 0.5, tenant_id="t1", user_id="u-42")[0] == "a"
        assert b.delete_user("u-42", tenant_id="t1") == 1
        b.clear()


# ── the event ─────────────────────────────────────────────────────────────────────────────────────────────────

class TestEvent:
    def test_one_event_without_the_user_id(self, tmp_path):
        sink = _Sink()
        c = _cache(tmp_path, event_sink=sink)
        _seed(c)
        sink.events.clear()
        c.delete_user("u-42", tenant_id="t1")
        assert len(sink.events) == 1
        ev = sink.events[0]
        assert isinstance(ev, CacheEvent) and ev.event_type == "delete_user"
        assert (ev.tenant_id, ev.user_id, ev.session_id, ev.backend_id) == ("t1", None, None, "sqlite")
        assert "u-42" not in repr(ev)

    def test_no_event_when_the_erasure_failed(self):
        sink = _Sink()
        c = _with(_Backend(1), _Store(fail=ConnectionError("x")), event_sink=sink)
        with pytest.raises(SulciErasureError):
            c.delete_user("u-42", tenant_id="t1")
        assert sink.events == []

    def test_a_failing_sink_does_not_fail_the_erasure(self, tmp_path):
        class Boom(_Sink):
            def emit(self, event):
                raise RuntimeError("sink down")
        c = _cache(tmp_path, event_sink=Boom())
        _seed(c)
        assert c.delete_user("u-42", tenant_id="t1") == {"entries": 2, "sessions": 1}


# ── the managed backend ───────────────────────────────────────────────────────────────────────────────────────

def _managed(status=200, body=None, personalized=True):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json=body)
    c = Cache(backend="sulci", api_key="sk-sulci-test0000000000", personalized=personalized, telemetry=False)
    c._backend._client.close()
    c._backend._client = httpx.Client(base_url="https://api.sulci.io", transport=httpx.MockTransport(handler))
    return c, seen


OK = {"status": "ok", "deleted": 4, "tenant_id": 7, "user_id": "u-42", "sessions_deleted": 2,
      "entries_status": "ok", "sessions_status": "ok"}


class TestManaged:
    def test_the_gateways_counts_in_the_same_shape(self):
        c, seen = _managed(200, OK)
        assert c.delete_user("u-42") == {"entries": 4, "sessions": 2}
        assert [(r.method, r.url.raw_path) for r in seen] == [("DELETE", b"/v1/cache/user/u-42")]

    def test_a_tenant_id_is_refused_and_nothing_is_sent(self):
        c, seen = _managed(200, OK)
        with pytest.raises(TypeError, match="API key decides the tenant"):
            c.delete_user("u-42", tenant_id="t1")
        assert seen == []

    def test_not_personalized_is_refused_and_nothing_is_sent(self):
        c, seen = _managed(200, OK, personalized=False)
        with pytest.raises(ValueError, match="personalized=True"):
            c.delete_user("u-42")
        assert seen == []

    def test_a_reserved_id_is_refused_and_nothing_is_sent(self):
        c, seen = _managed(200, OK)
        with pytest.raises(ValueError):
            c.delete_user("global")
        assert seen == []

    @pytest.mark.parametrize("entries_status,sessions_status,want,said", [
        ("ok", "failed", (4, None), "the user's 4 cache entries WERE erased; the conversation history was NOT all erased"),
        ("failed", "ok", (None, 2), "the cache entries were NOT all erased; 2 sessions WERE erased"),
    ])
    def test_an_incomplete_erasure_sets_the_count_of_the_half_that_completed(
            self, entries_status, sessions_status, want, said):
        # The gateway's flattened 503 (public_api.py). A failed half is None, as on the self-hosted path; its partial
        # count stays in the body.
        body = {"error": "erasure_incomplete", "message": "m", "status": 503, "retry_after": 5, "deleted": 4,
                "sessions_deleted": 2, "entries_status": entries_status, "sessions_status": sessions_status,
                "tenant_id": 7, "user_id": "u-42"}
        c, _ = _managed(503, body)
        with pytest.raises(SulciErasureError) as e:
            c.delete_user("u-42")
        assert (e.value.status, e.value.error, (e.value.entries, e.value.sessions), e.value.body) == \
            (503, "erasure_incomplete", want, body)
        assert str(e.value) == f"sulci delete_user failed: HTTP 503: 'erasure_incomplete' — {said}. Retry the call."
        assert isinstance(e.value.__cause__, SulciErasureError)

    @pytest.mark.parametrize("extra", [{"entries_status": "failed", "sessions_status": "failed"}, {}], ids=["both", "no-status"])
    def test_an_incomplete_erasure_with_no_half_complete_is_the_gateways_error(self, extra):
        body = {"error": "erasure_incomplete", "message": "m", "status": 503, "deleted": 1, "sessions_deleted": 0, **extra}
        c, _ = _managed(503, body)
        with pytest.raises(SulciErasureError) as e:
            c.delete_user("u-42")
        assert (e.value.entries, e.value.sessions, e.value.__cause__) == (None, None, None)
        assert str(e.value) == "sulci delete_user failed: HTTP 503: 'erasure_incomplete'"

    def test_a_long_error_code_is_truncated_in_the_message(self):
        body = {"error": "x" * 5000, "deleted": 4, "sessions_deleted": 0, "entries_status": "ok", "sessions_status": "failed"}
        c, _ = _managed(503, body)
        with pytest.raises(SulciErasureError) as e:
            c.delete_user("u-42")
        assert len(str(e.value)) < 600 and e.value.error == "x" * 5000

    def test_a_failure_with_no_counts(self):
        c, _ = _managed(422, {"error": "validation_error", "message": "m"})
        with pytest.raises(SulciErasureError) as e:
            c.delete_user("u-42")
        assert (e.value.status, e.value.entries, e.value.sessions) == (422, None, None)
        assert str(e.value) == "sulci delete_user failed: HTTP 422: 'validation_error'"   # the cloud error, as raised
        assert e.value.__cause__ is None

    def test_an_ok_answer_without_both_counts_is_not_confirmation(self):
        c, _ = _managed(200, {"status": "ok", "deleted": 4, "tenant_id": 7})
        with pytest.raises(SulciErasureError) as e:
            c.delete_user("u-42")
        assert e.value.error == "unexpected_response"

    def test_the_event_has_no_user_id(self):
        sink = _Sink()
        c, _ = _managed(200, OK)
        c._event_sink = sink
        c.delete_user("u-42")
        assert [(e.event_type, e.tenant_id, e.user_id) for e in sink.events] == [("delete_user", None, None)]


# ── AsyncCache ────────────────────────────────────────────────────────────────────────────────────────────────

class TestAsyncCache:
    def test_adelete_user_and_the_passthrough(self, tmp_path):
        ac = AsyncCache(backend="sqlite", db_path=str(tmp_path), embedding_model=FakeEmbedder(), personalized=True,
                        context_window=4, telemetry=False)
        _seed(ac._cache)
        assert asyncio.run(ac.adelete_user("u-42", tenant_id="t1")) == {"entries": 2, "sessions": 1}
        assert ac.delete_user("u-7", tenant_id="t1") == {"entries": 1, "sessions": 1}
        assert _rows(tmp_path) == [("t1", "global", 1), ("t2", "u-42", 1)]

    def test_adelete_user_raises_as_cache_does(self, tmp_path):
        ac = AsyncCache(backend="sqlite", db_path=str(tmp_path), embedding_model=FakeEmbedder(), telemetry=False)
        with pytest.raises(ValueError, match="personalized=True"):
            asyncio.run(ac.adelete_user("u-42"))


# ── the built-in session store ────────────────────────────────────────────────────────────────────────────────

class TestBuiltinClearScope:
    def test_removes_the_prefix_only_and_counts_idle_sessions_too(self):
        from sulci.context import SessionStore
        s = SessionStore(max_turns=4, ttl_seconds=3600)
        a, b = user_scope("t1", "u-42"), user_scope("t1", "u-7")
        for sid in (a + "s1", a + "s2", b + "s1"):
            s.get(sid)
        s._last_active[a + "s2"] = time.time() - 7200          # idle past its ttl, not yet evicted
        assert s.clear_scope(a) == 2
        assert list(s._windows) == [b + "s1"] and list(s._last_active) == [b + "s1"]
        assert s.clear_scope(a) == 0
