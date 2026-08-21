# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""tests/test_sessions.py — v0.5.0 session store tests."""
import pytest
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

@pytest.fixture
def redis_client():
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
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

class TestTenantIsolation:
    def test_in_memory_tenants_do_not_share_history(self):
        a = InMemorySessionStore(tenant_id="tenant-A")
        b = InMemorySessionStore(tenant_id="tenant-B")
        a.append("user_42", [1.0, 2.0])
        b.append("user_42", [9.0, 8.0])
        assert a.get("user_42") == [[1.0, 2.0]]
        assert b.get("user_42") == [[9.0, 8.0]]

    def test_in_memory_default_tenant_none_unchanged(self):
        store = InMemorySessionStore()
        store.append("s1", [1.0])
        assert store.get("s1") == [[1.0]]

    def test_in_memory_summary_scoped_to_tenant(self):
        a = InMemorySessionStore(tenant_id="tenant-A")
        b = InMemorySessionStore(tenant_id="tenant-B")
        a.append("s1", [1.0])
        a.append("s2", [2.0])
        b.append("s1", [9.0])
        assert a.summary()["sessions"] == 2
        assert b.summary()["sessions"] == 1
        