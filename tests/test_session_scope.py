# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_session_scope.py
───────────────────────────
Conversation history belongs to one tenant and one user, and clear() removes it from any store (0.9.3, OSS-3).

Before: the Cache handed the session store the caller's ``session_id`` unchanged, so one store serving several
tenants (or users, with ``personalized=True``) blended one tenant's turns into another's lookups whenever they used
the same id; and with an injected store ``Cache.clear()`` left every session in place (the adapter's ``clear_all``
was a no-op). Measured at oss-0007 (``measure_oss3.py``): another tenant's and another user's lookup with the same
session id blended the owner's turns on the built-in, in-memory and Redis stores alike (depth 2, 1, 1), and after
``clear()`` the owner's injected sessions were still there (depth 1).

Now the Cache scopes every session id by tenant and, when personalized, user (``sulci.sessions.scope``); the shipped
stores index scoped ids by tenant and user and gain ``clear_scope`` / ``clear_all``; ``Cache.clear()`` calls
``clear_all``; a store without them is refused at construction. Redis runs where a server answers on localhost.
"""
from __future__ import annotations

import asyncio
import importlib.util
import uuid

import httpx
import pytest

from sulci import Cache
from sulci.sessions import InMemorySessionStore, RedisSessionStore, scoped_session_id, tenant_scope, user_scope
from sulci.sessions.scope import scopes_of

S = "conv-1"


def _redis():
    if importlib.util.find_spec("redis") is None:
        return None
    try:
        import redis
        r = redis.Redis.from_url("redis://localhost:6379", socket_connect_timeout=0.5)
        r.ping()
        return r
    except Exception:
        return None


REDIS = _redis()
KINDS = ["builtin", "memory", pytest.param("redis", marks=pytest.mark.skipif(REDIS is None, reason="no Redis here"))]


def _store(kind):
    if kind == "memory":
        return InMemorySessionStore()
    if kind == "redis":
        return RedisSessionStore(REDIS, key_prefix=f"sulci-test-{uuid.uuid4().hex[:12]}:")
    return None


def _cache(kind, tmp_path, emb, personalized=True):
    return Cache(backend="sqlite", db_path=str(tmp_path / "c"), embedding_model=emb, telemetry=False,
                 personalized=personalized, context_window=4, session_store=_store(kind))


def _depth(c, **scope):
    return c.get("how do i fix it", session_id=S, **scope)[2]


def _seed(c, **scope):
    c.set("my laptop will not boot", "Hold the power button.", session_id=S, **scope)


# ── the scope ───────────────────────────────────────────────────────────────────
def test_the_scoped_id_is_tenant_then_user_then_the_callers_id():
    sid = scoped_session_id("t1", "alice", "a:b*c")
    assert sid.endswith(":a:b*c")
    tpart, upart = sid.split(":")[:2]
    assert tpart[0] == "t" and len(tpart) == 33 and upart[0] == "u" and len(upart) == 33
    assert scoped_session_id(None, None, "x") == "t-:u-:x"
    assert sid.startswith(user_scope("t1", "alice")) and sid.startswith(tenant_scope("t1"))
    assert scopes_of(sid) == (tenant_scope("t1"), user_scope("t1", "alice"))
    assert scopes_of("plain-id") == ()


@pytest.mark.parametrize("a, b", [(("t1", "u"), ("t2", "u")), (("t1", "u1"), ("t1", "u2")), (("t:1", "u"), ("t", "1:u")),
                                  (("t1", None), ("t1", "-")), ((None, None), ("-", "-"))])
def test_different_scopes_never_share_a_prefix(a, b):
    assert not scoped_session_id(*a, S).startswith(user_scope(*b))
    assert scoped_session_id(*a, S) != scoped_session_id(*b, S)


# ── isolation through the Cache ─────────────────────────────────────────────────
@pytest.mark.parametrize("kind", KINDS)
def test_a_session_belongs_to_its_tenant_and_user(kind, tmp_path, fake_embedder):
    c = _cache(kind, tmp_path, fake_embedder)
    _seed(c, tenant_id="t1", user_id="alice")
    assert _depth(c, tenant_id="t1", user_id="alice") > 0
    assert _depth(c, tenant_id="t2", user_id="alice") == 0
    assert _depth(c, tenant_id="t1", user_id="bob") == 0
    assert _depth(c, tenant_id="t1") == 0
    assert _depth(c) == 0
    c.clear()


@pytest.mark.parametrize("kind", KINDS)
def test_not_personalized_shares_a_tenants_session_across_users_only(kind, tmp_path, fake_embedder):
    c = _cache(kind, tmp_path, fake_embedder, personalized=False)
    _seed(c, tenant_id="t1", user_id="alice")
    assert _depth(c, tenant_id="t1", user_id="bob") > 0
    assert _depth(c, tenant_id="t2", user_id="alice") == 0
    c.clear()


@pytest.mark.parametrize("kind", KINDS)
def test_clear_removes_every_session_from_any_store(kind, tmp_path, fake_embedder):
    c = _cache(kind, tmp_path, fake_embedder)
    for t, u in (("t1", "alice"), ("t2", "bob"), (None, None)):
        _seed(c, tenant_id=t, user_id=u)
    c.clear()
    for t, u in (("t1", "alice"), ("t2", "bob"), (None, None)):
        assert _depth(c, tenant_id=t, user_id=u) == 0


@pytest.mark.parametrize("kind", KINDS)
def test_the_context_api_names_the_scope(kind, tmp_path, fake_embedder):
    c = _cache(kind, tmp_path, fake_embedder)
    _seed(c, tenant_id="t1", user_id="alice")
    _seed(c, tenant_id="t2", user_id="alice")
    assert c.get_context(S, tenant_id="t1", user_id="alice").depth > 0
    assert c.get_context(S).depth == 0
    assert c.context_summary(S, tenant_id="t1", user_id="alice")["session_id"] == S
    assert c.context_summary(S, tenant_id="t1", user_id="alice")["depth"] > 0
    assert c.context_summary(S, tenant_id="t3", user_id="alice")["depth"] == 0
    with pytest.raises(TypeError, match="need a session_id"):
        c.context_summary(tenant_id="t1")
    c.clear_context(S, tenant_id="t1", user_id="alice")
    assert _depth(c, tenant_id="t1", user_id="alice") == 0
    assert _depth(c, tenant_id="t2", user_id="alice") > 0          # only the named scope was cleared
    with pytest.raises(ValueError):
        c.clear_context(S, tenant_id="global")
    c.clear()


@pytest.mark.parametrize("kind", ["memory", pytest.param("redis", marks=pytest.mark.skipif(REDIS is None, reason="no Redis here"))])
def test_the_store_holds_only_scoped_ids(kind, tmp_path, fake_embedder):
    store = _store(kind)
    c = Cache(backend="sqlite", db_path=str(tmp_path / "c"), embedding_model=fake_embedder, telemetry=False,
              personalized=True, context_window=4, session_store=store)
    _seed(c, tenant_id="t1", user_id="alice")
    assert store.get(S) == []
    assert len(store.get(scoped_session_id("t1", "alice", S))) == 1
    assert store.clear_scope(user_scope("t1", "bob")) == 0
    assert store.clear_scope(user_scope("t1", "alice")) == 1
    assert store.get(scoped_session_id("t1", "alice", S)) == []


def test_cached_call_records_a_hit_under_the_scope(tmp_path, fake_embedder):
    c = _cache("memory", tmp_path, fake_embedder)
    c.set("my laptop will not boot", "Hold the power button.", tenant_id="t1", user_id="alice")
    c.cached_call("my laptop will not boot", lambda q: "LLM", session_id=S, tenant_id="t1", user_id="alice")
    assert _depth(c, tenant_id="t1", user_id="alice") > 0
    assert _depth(c, tenant_id="t2", user_id="alice") == 0


def test_async_cache_passes_the_scope(tmp_path, fake_embedder):
    from sulci import AsyncCache
    ac = AsyncCache(backend="sqlite", db_path=str(tmp_path / "a"), embedding_model=fake_embedder, telemetry=False,
                    personalized=True, context_window=4)

    async def run():
        await ac.aset("my laptop will not boot", "Hold the power button.", session_id=S, tenant_id="t1", user_id="alice")
        assert (await ac.aget_context(S, tenant_id="t1", user_id="alice")).depth > 0
        assert (await ac.aget_context(S, tenant_id="t2", user_id="alice")).depth == 0
        await ac.aclear_context(S, tenant_id="t1", user_id="alice")
        assert (await ac.aget_context(S, tenant_id="t1", user_id="alice")).depth == 0
    asyncio.run(run())


# ── a store clear() could not reach is refused ──────────────────────────────────
class _OldStore:
    def get(self, session_id): return []
    def append(self, session_id, vector, max_turns=8): pass
    def clear(self, session_id): pass
    def summary(self, session_id=None): return {"sessions": 0, "total_turns": 0, "avg_turns": 0.0}


def test_a_store_without_clear_all_is_refused_at_construction(tmp_path, fake_embedder):
    with pytest.raises(TypeError, match="clear_scope or clear_all"):
        Cache(backend="sqlite", db_path=str(tmp_path / "c"), embedding_model=fake_embedder, context_window=4,
              session_store=_OldStore(), telemetry=False)


def test_a_failing_clear_all_is_not_swallowed(tmp_path, fake_embedder):
    store = InMemorySessionStore()
    def boom():
        raise RuntimeError("redis down")
    store.clear_all = boom
    c = Cache(backend="sqlite", db_path=str(tmp_path / "c"), embedding_model=fake_embedder, telemetry=False,
              context_window=4, session_store=store)
    with pytest.raises(RuntimeError, match="redis down"):
        c.clear()


# ── the managed backend: the gateway scopes; the SDK sends the caller's id ──────
def test_the_managed_backend_sends_the_callers_session_id():
    seen = []

    def handler(request):
        import json
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"response": None, "similarity": 0.0, "context_depth": 0})
    c = Cache(backend="sulci", api_key="sk-sulci-test0000000000", telemetry=False, personalized=True)
    c._backend._client.close()
    c._backend._client = httpx.Client(base_url="https://api.sulci.io", transport=httpx.MockTransport(handler))
    c.get("q", session_id=S, user_id="alice")
    assert seen and seen[0]["session_id"] == S
