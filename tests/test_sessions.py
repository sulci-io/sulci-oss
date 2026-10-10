# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""tests/test_sessions.py — v0.5.0 session store tests."""
import pytest
import socket
from functools import lru_cache
from sulci.sessions import InMemorySessionStore, RedisSessionStore, SessionStore


@pytest.fixture
def sample_vec():
    return [0.1, 0.2, 0.3, 0.4]


class TestInMemorySessionStoreBasics:
    def test_empty_session_returns_empty_list(self):
        store = InMemorySessionStore()
        assert store.get("nonexistent") == []

    def test_append_and_get(self, sample_vec):
        store = InMemorySessionStore()
        store.append("s1", sample_vec)
        assert store.get("s1") == [sample_vec]

    def test_multiple_appends_preserve_order(self):
        store = InMemorySessionStore()
        store.append("s1", [1.0])
        store.append("s1", [2.0])
        store.append("s1", [3.0])
        assert store.get("s1") == [[1.0], [2.0], [3.0]]

    def test_max_turns_trims_oldest(self):
        store = InMemorySessionStore()
        for i in range(10):
            store.append("s1", [float(i)], max_turns=3)
        history = store.get("s1")
        assert len(history) == 3
        assert history == [[7.0], [8.0], [9.0]]

    def test_clear_removes_session(self):
        store = InMemorySessionStore()
        store.append("s1", [1.0])
        store.clear("s1")
        assert store.get("s1") == []

    def test_clear_nonexistent_is_idempotent(self):
        store = InMemorySessionStore()
        store.clear("nonexistent")   # Should not raise

    def test_session_isolation(self):
        store = InMemorySessionStore()
        store.append("s1", [1.0])
        store.append("s2", [2.0])
        assert store.get("s1") == [[1.0]]
        assert store.get("s2") == [[2.0]]


class TestInMemorySummary:
    def test_global_summary_empty(self):
        store = InMemorySessionStore()
        s = store.summary()
        assert s["sessions"] == 0
        assert s["total_turns"] == 0
        assert s["avg_turns"] == 0.0

    def test_global_summary_with_data(self):
        store = InMemorySessionStore()
        store.append("s1", [1.0])
        store.append("s1", [2.0])
        store.append("s2", [3.0])
        s = store.summary()
        assert s["sessions"] == 2
        assert s["total_turns"] == 3
        assert s["avg_turns"] == 1.5

    def test_scoped_summary(self):
        store = InMemorySessionStore()
        store.append("s1", [1.0])
        store.append("s1", [2.0])
        s = store.summary(session_id="s1")
        assert s["sessions"] == 1
        assert s["total_turns"] == 2


class TestMaxTotalSessionsEviction:
    def test_evicts_oldest_session_when_limit_exceeded(self):
        store = InMemorySessionStore(max_total_sessions=2)
        store.append("s1", [1.0])
        store.append("s2", [2.0])
        store.append("s3", [3.0])   # Should evict s1
        assert store.get("s1") == []
        assert store.get("s2") == [[2.0]]
        assert store.get("s3") == [[3.0]]


class TestProtocolConformance:
    def test_in_memory_conforms_to_protocol(self):
        assert isinstance(InMemorySessionStore(), SessionStore)


class TestBackwardCompatShim:
    def test_sulci_context_SessionStore_still_importable(self):
        """v0.3.x code: `from sulci.context import SessionStore` must still work."""
        from sulci.context import SessionStore
        # Import path preserved; class identity is intentionally NOT asserted
        # because the new sulci.sessions module is additive, not a replacement.
        store = SessionStore()
        # The old class API is preserved
        assert hasattr(store, "get")
        assert hasattr(store, "delete")


# ═══════════════════════════════════════════════════════════════════════
# RedisSessionStore tests — require Redis; skipped if unavailable
# ═══════════════════════════════════════════════════════════════════════

# ⛔ MEASURED 2026-09-23 (windows-latest/3.12, Tests #272): every test here
# paid ~48s in SETUP — 384s for 16 tests — because the probe below had no
# connect timeout and Windows waits out ~21s per address for BOTH ::1 and
# 127.0.0.1, with nothing caching the answer. The same suite on ubuntu, where
# `Install Redis (Linux only)` provides a server, takes 0.16s. See
# sulci/tests/compat/conftest.py for the full measurement.
@lru_cache(maxsize=1)
def _redis_available() -> bool:
    for family, addr in ((socket.AF_INET, ("127.0.0.1", 6379)),
                         (socket.AF_INET6, ("::1", 6379))):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.25)
                if sock.connect_ex(addr) == 0:
                    return True
        except OSError:
            continue
    return False


@pytest.fixture
def redis_client():
    if not _redis_available():      # ~0.25s once, not ~48s per test
        pytest.skip("redis-server not running on localhost:6379")
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True,
                             socket_connect_timeout=0.25, socket_timeout=0.25)
        client.ping()
        client.flushdb()
        yield client
        client.flushdb()
    except (ImportError, Exception):
        pytest.skip("Redis not available at localhost:6379 db=15")


class TestRedisSessionStore:
    def test_empty_session_returns_empty_list(self, redis_client):
        store = RedisSessionStore(redis_client)
        assert store.get("nonexistent") == []

    def test_append_and_get(self, redis_client, sample_vec):
        store = RedisSessionStore(redis_client)
        store.append("s1", sample_vec)
        result = store.get("s1")
        assert len(result) == 1
        assert result[0] == sample_vec

    def test_order_preserved_across_appends(self, redis_client):
        store = RedisSessionStore(redis_client)
        store.append("s1", [1.0])
        store.append("s1", [2.0])
        store.append("s1", [3.0])
        assert store.get("s1") == [[1.0], [2.0], [3.0]]

    def test_max_turns_enforced(self, redis_client):
        store = RedisSessionStore(redis_client)
        for i in range(10):
            store.append("s1", [float(i)], max_turns=3)
        history = store.get("s1")
        assert len(history) == 3
        assert history == [[7.0], [8.0], [9.0]]

    def test_clear_removes_key(self, redis_client):
        store = RedisSessionStore(redis_client)
        store.append("s1", [1.0])
        store.clear("s1")
        assert store.get("s1") == []

    def test_ttl_applied_when_configured(self, redis_client):
        store = RedisSessionStore(redis_client, ttl_seconds=60)
        store.append("s1", [1.0])
        ttl = redis_client.ttl(store._key("s1"))
        assert 0 < ttl <= 60

    def test_protocol_conformance(self, redis_client):
        store = RedisSessionStore(redis_client)
        assert isinstance(store, SessionStore)

    def test_cross_instance_visibility(self, redis_client):
        """Core multi-replica guarantee: two RedisSessionStore instances
        pointing at the same Redis see each other's data."""
        store_a = RedisSessionStore(redis_client)
        store_b = RedisSessionStore(redis_client)
        store_a.append("s1", [1.0])
        assert store_b.get("s1") == [[1.0]]

class TestScopes:
    """0.9.3 (OSS-3): the stores take no tenant — the Cache scopes every id (sulci.sessions.scope) — and remove a
    tenant's, a user's or every session."""

    @staticmethod
    def _fill(store):
        from sulci.sessions import scoped_session_id
        ids = {(t, u): scoped_session_id(t, u, "s1") for t in ("tA", "tB") for u in ("alice", None)}
        for sid in ids.values():
            store.append(sid, [1.0])
        return ids

    def test_in_memory_clear_scope_removes_one_tenant(self):
        from sulci.sessions import tenant_scope
        store = InMemorySessionStore()
        ids = self._fill(store)
        assert store.clear_scope(tenant_scope("tA")) == 2
        assert [k for k, sid in ids.items() if store.get(sid)] == [("tB", "alice"), ("tB", None)]

    def test_in_memory_clear_scope_removes_one_user(self):
        from sulci.sessions import user_scope
        store = InMemorySessionStore()
        ids = self._fill(store)
        assert store.clear_scope(user_scope("tA", "alice")) == 1
        assert [k for k, sid in ids.items() if not store.get(sid)] == [("tA", "alice")]

    def test_in_memory_takes_no_tenant(self):
        with pytest.raises(TypeError):
            InMemorySessionStore(tenant_id="tenant-A")


class TestRedisIndex:
    """0.9.3 (OSS-3): RedisSessionStore indexes every id it writes, and erases from the index, never a SCAN."""

    def test_clear_all_and_clear_scope_never_scan_the_keyspace(self, redis_client, monkeypatch):
        from sulci.sessions import scoped_session_id, tenant_scope
        store = RedisSessionStore(redis_client)
        for t in ("tA", "tB"):
            store.append(scoped_session_id(t, "alice", "s1"), [1.0])
        def no_scan(*a, **k):
            raise AssertionError("keyspace SCAN")
        monkeypatch.setattr(redis_client, "scan", no_scan)
        monkeypatch.setattr(redis_client, "scan_iter", no_scan)
        assert store.clear_scope(tenant_scope("tA")) == 1
        assert store.summary()["sessions"] == 1
        assert store.clear_all() == 1
        assert redis_client.keys("*") == []                 # sessions and indexes alike are gone

    def test_an_id_cannot_reach_the_index_namespace(self, redis_client):
        store = RedisSessionStore(redis_client)
        store.append("i:", [1.0])                            # looks like the store's own index key
        store.append("s1", [2.0])
        assert store.get("i:") == [[1.0]] and store.summary()["sessions"] == 2
        assert store.clear_all() == 2

    def test_index_entries_outlive_their_session_by_the_grace_period(self, redis_client):
        import time as _time
        from sulci.sessions import scoped_session_id
        from sulci.sessions.redis import _INDEX_GRACE_SECONDS
        store = RedisSessionStore(redis_client, ttl_seconds=60)
        sid = scoped_session_id("tA", "alice", "s1")
        store.append(sid, [1.0])
        idx = [k for k in redis_client.keys("*") if ":i:" in k]
        assert len(idx) == 3
        for k in idx:
            assert abs(redis_client.zscore(k, sid) - (_time.time() + 60 + _INDEX_GRACE_SECONDS)) < 5

    def test_redis_takes_no_tenant(self, redis_client):
        with pytest.raises(TypeError):
            RedisSessionStore(redis_client, tenant_id="tenant-A")

    def test_one_write_is_one_transaction(self, redis_client, monkeypatch):
        from sulci.sessions import scoped_session_id
        store = RedisSessionStore(redis_client, ttl_seconds=60)
        made, real = [], redis_client.pipeline

        def recording(*a, **k):
            p = real(*a, **k)
            made.append((k.get("transaction", a[0] if a else True), p))
            return p
        monkeypatch.setattr(redis_client, "pipeline", recording)
        store.append(scoped_session_id("tA", "alice", "s1"), [1.0])
        assert len(made) == 1 and made[0][0] is True        # the session and its three index entries together

    def test_an_index_lists_live_sessions_not_every_id_ever_written(self, redis_client, monkeypatch):
        import time as _time
        from sulci.sessions import redis as rmod
        store = RedisSessionStore(redis_client, ttl_seconds=60)
        store.append("old", [1.0])
        later = _time.time() + 60 + rmod._INDEX_GRACE_SECONDS + 1
        monkeypatch.setattr(rmod.time, "time", lambda: later)
        store.append("new", [2.0])
        assert [m for m, _ in redis_client.zscan_iter(store._index())] == ["new"]

    def test_a_writer_with_a_slow_clock_cannot_shorten_an_entry(self, redis_client, monkeypatch):
        import time as _time
        from sulci.sessions import redis as rmod
        store = RedisSessionStore(redis_client, ttl_seconds=60)
        store.append("s1", [1.0])
        first = redis_client.zscore(store._index(), "s1")
        real = _time.time
        monkeypatch.setattr(rmod.time, "time", lambda: real() - 1000)
        store.append("s1", [2.0])
        assert redis_client.zscore(store._index(), "s1") == first

    def test_stores_with_different_ttls_can_share_a_prefix(self, redis_client):
        RedisSessionStore(redis_client).append("forever", [1.0])
        RedisSessionStore(redis_client, ttl_seconds=60).append("hour", [1.0])
        assert redis_client.ttl("sulci:sessions:i:") == -1          # the index never expires under "forever"
        assert RedisSessionStore(redis_client).clear_all() == 2

    def test_the_default_prefix_is_new(self, redis_client):
        assert RedisSessionStore(redis_client)._key("x") == "sulci:sessions:s:x"


class TestInMemoryThreads:
    def test_appends_and_clears_from_many_threads(self):
        import sys, threading, time
        from sulci.sessions import scoped_session_id, tenant_scope
        store, errors = InMemorySessionStore(max_total_sessions=50), []
        stop = time.monotonic() + 1.0
        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)                     # switch threads constantly, so the operations overlap

        def writer(n):
            i = 0
            try:
                while time.monotonic() < stop:
                    store.append(scoped_session_id(f"t{n}", "u", f"s{i % 80}"), [1.0]); i += 1
            except Exception as e:      # noqa: BLE001
                errors.append(e)

        def clearer():
            i = 0
            try:
                while time.monotonic() < stop:
                    store.clear_scope(tenant_scope(f"t{i % 4}")); i += 1
            except Exception as e:      # noqa: BLE001
                errors.append(e)
        ts = [threading.Thread(target=writer, args=(n,)) for n in range(4)] + [threading.Thread(target=clearer)]
        try:
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        finally:
            sys.setswitchinterval(old)
        assert errors == []
