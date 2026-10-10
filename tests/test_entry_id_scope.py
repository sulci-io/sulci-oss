# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_entry_id_scope.py
────────────────────────────
Storing a question in one scope never replaces another scope's entry (0.9.3).

Before: ``Cache.set`` derives an entry's key from the query text alone, and the
Chroma backend upserted by that key while the Redis backend used it as the hash
name. When tenant t2 (or another user) stored the same question, its entry
REPLACED tenant t1's: t1's next lookup missed. Now both backends store under
``scoped_id(key, tenant_id, user_id)``; the same scope storing again still
replaces its own entry.

Chroma runs through a recording stand-in (CI does not install chromadb). Redis
runs through a stand-in everywhere and for real wherever a server answers on
localhost (CI's Linux legs; the release verify requires one).
"""
from __future__ import annotations

import importlib.util
import uuid

import pytest

from sulci import Cache
from sulci.backends._scope import scoped_id

VEC = [1.0] + [0.0] * 63
Q = "how do i reset my password"


def _redis_up() -> bool:
    if importlib.util.find_spec("redis") is None:
        return False
    try:
        import redis
        return bool(redis.Redis.from_url("redis://localhost:6379", socket_connect_timeout=0.5).ping())
    except Exception:
        return False


REDIS_UP = _redis_up()


# ── the id ──────────────────────────────────────────────────────────────────────
def test_same_scope_same_id():
    assert scoped_id("k", "t1", "alice") == scoped_id("k", "t1", "alice")


@pytest.mark.parametrize("other", [("k", "t2", "alice"), ("k", "t1", "bob"), ("k2", "t1", "alice")])
def test_any_difference_gives_a_different_id(other):
    assert scoped_id("k", "t1", "alice") != scoped_id(*other)


def test_separators_cannot_make_two_scopes_collide():
    assert scoped_id("k", "a:b", "c") != scoped_id("k", "a", "b:c")
    assert scoped_id("k", "a", "b\x00c") != scoped_id("k", "a\x00b", "c")
    assert scoped_id("x", '"],"', "y") != scoped_id("x", "", '"],"y')


def test_none_is_the_global_scope_as_the_stored_fields_are():
    assert scoped_id("k") == scoped_id("k", "global", "global") == scoped_id("k", None, None)


def test_any_string_encodes_lone_surrogates_included():
    a, b = scoped_id("k", "t\ud800", "u"), scoped_id("k", "t\udc00", "u")
    assert a != b and len(a) == 32


def test_id_shape():
    i = scoped_id("k", "t1", "alice")
    assert len(i) == 32 and all(ch in "0123456789abcdef" for ch in i)
    assert scoped_id("k", "té", "用户") != scoped_id("k", "te", "user")


# ── Chroma: the upsert id ──────────────────────────────────────────────────────
class RecordingCollection:
    def __init__(self):
        self.ids = []

    def upsert(self, *, ids, **kw):
        self.ids.extend(ids)


def _chroma():
    from sulci.backends.chroma import ChromaBackend
    b = ChromaBackend.__new__(ChromaBackend)
    b._col = RecordingCollection()
    return b


def test_chroma_upserts_under_the_scoped_id():
    b = _chroma()
    b.store("k1", Q, "ALICE", VEC, tenant_id="t1", user_id="alice")
    b.store("k1", Q, "BOB", VEC, tenant_id="t2", user_id="bob")
    b.store("k1", Q, "CAROL", VEC, tenant_id="t1", user_id="carol")
    b.store("k1", Q, "ALICE2", VEC, tenant_id="t1", user_id="alice")
    assert b._col.ids == [scoped_id("k1", "t1", "alice"), scoped_id("k1", "t2", "bob"),
                          scoped_id("k1", "t1", "carol"), scoped_id("k1", "t1", "alice")]
    assert len(set(b._col.ids)) == 3          # the same scope replaces; other scopes do not


# ── Redis: the hash name ───────────────────────────────────────────────────────
class RecordingRedis:
    def __init__(self):
        self.hashes = {}

    def hset(self, name, mapping):
        self.hashes.setdefault(name, {}).update(mapping)

    def expireat(self, name, when):
        pass


def _redis_standin():
    from sulci.backends.redis import RedisBackend
    b = RedisBackend.__new__(RedisBackend)
    b._prefix, b._redis = "sulci:", RecordingRedis()
    return b


def test_redis_hashes_are_named_by_the_scoped_id():
    b = _redis_standin()
    b.store("k1", Q, "ALICE", VEC, tenant_id="t1", user_id="alice")
    b.store("k1", Q, "BOB", VEC, tenant_id="t2", user_id="bob")
    b.store("k1", Q, "ALICE2", VEC, tenant_id="t1", user_id="alice")
    assert set(b._redis.hashes) == {"sulci:" + scoped_id("k1", "t1", "alice"), "sulci:" + scoped_id("k1", "t2", "bob")}
    assert b._redis.hashes["sulci:" + scoped_id("k1", "t1", "alice")]["response"] == b"ALICE2"
    assert b._redis.hashes["sulci:" + scoped_id("k1", "t2", "bob")]["response"] == b"BOB"


def test_redis_keeps_its_prefix_so_clear_and_search_still_see_every_entry():
    b = _redis_standin()
    b._prefix = "tenantA:"
    b.store("k1", Q, "ALICE", VEC, tenant_id="t1", user_id="alice")
    assert all(name.startswith("tenantA:") for name in b._redis.hashes)


# ── end to end through Cache, on a real Redis ──────────────────────────────────
def _redis_cache(fake_embedder):
    from sulci.backends.redis import RedisBackend
    b = RedisBackend(key_prefix=f"sulci-test-{uuid.uuid4().hex[:12]}:")
    return Cache(backend=b, embedding_model=fake_embedder, telemetry=False, personalized=True)


@pytest.mark.skipif(not REDIS_UP, reason="no Redis server on localhost:6379")
def test_redis_another_tenant_and_user_storing_the_same_question_keeps_ours(fake_embedder):
    c = _redis_cache(fake_embedder)
    try:
        c.set(Q, "ALICE", tenant_id="t1", user_id="alice")
        c.set(Q, "BOB", tenant_id="t2", user_id="bob")
        assert c.get(Q, tenant_id="t1", user_id="alice")[0] == "ALICE"
        assert c.get(Q, tenant_id="t2", user_id="bob")[0] == "BOB"
    finally:
        c.clear()


@pytest.mark.skipif(not REDIS_UP, reason="no Redis server on localhost:6379")
def test_redis_the_same_user_in_another_tenant_keeps_ours_stored(fake_embedder):
    # Read in STORAGE: Redis lookups do not filter by tenant (a separate fix), so a read cannot tell the two apart.
    c = _redis_cache(fake_embedder)
    b = c._backend
    try:
        c.set(Q, "IN-T1", tenant_id="t1", user_id="alice")
        c.set(Q, "IN-T2", tenant_id="t2", user_id="alice")
        names = sorted(b._redis.keys(b._prefix + "*"))
        assert len(names) == 2
        assert sorted(b._redis.hget(n, "response") for n in names) == [b"IN-T1", b"IN-T2"]
    finally:
        c.clear()


@pytest.mark.skipif(not REDIS_UP, reason="no Redis server on localhost:6379")
def test_redis_another_user_storing_the_same_question_keeps_ours(fake_embedder):
    c = _redis_cache(fake_embedder)
    try:
        c.set(Q, "ALICE", tenant_id="t1", user_id="alice")
        c.set(Q, "CAROL", tenant_id="t1", user_id="carol")
        assert c.get(Q, tenant_id="t1", user_id="alice")[0] == "ALICE"
        assert c.get(Q, tenant_id="t1", user_id="carol")[0] == "CAROL"
    finally:
        c.clear()


@pytest.mark.skipif(not REDIS_UP, reason="no Redis server on localhost:6379")
def test_redis_the_same_scope_storing_again_replaces_its_entry(fake_embedder):
    c = _redis_cache(fake_embedder)
    try:
        c.set(Q, "OLD", tenant_id="t1", user_id="alice")
        c.set(Q, "NEW", tenant_id="t1", user_id="alice")
        assert c.get(Q, tenant_id="t1", user_id="alice")[0] == "NEW"
    finally:
        c.clear()
