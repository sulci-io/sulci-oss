# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_reserved_scope_ids.py
────────────────────────────────
``"global"`` and ``""`` are refused as a tenant or user id (0.9.3).

Before: every backend files entries without a tenant or user under ``"global"``,
the shared scope, so ``user_id="global"`` / ``tenant_id="global"`` read and wrote
the shared scope as if it were the caller's own, and ``""`` was stored there too
(on Qdrant a lookup with ``""`` could not read back what it had stored).

Now ``Cache.get`` and ``Cache.set`` raise ``ValueError`` first — before the
backend, the cloud gateway or the query counter is touched: ``tenant_id``
always, ``user_id`` only when ``personalized=True`` (otherwise it is ignored).
"""
from __future__ import annotations

import asyncio

import pytest

from sulci import Cache
from sulci._scope_ids import RESERVED_SCOPE_IDS, refuse_reserved_scope

RESERVED = ["global", ""]
FALSY = [0, False, b""]                      # every backend files `id or "global"`, so these landed there too


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def store(self, key, query, response, embedding, *, tenant_id=None, user_id=None, expires=None, metadata=None):
        self.calls.append(("store", tenant_id, user_id))

    def search(self, embedding, threshold, *, tenant_id=None, user_id=None, now=None):
        self.calls.append(("search", tenant_id, user_id))
        return None, 0.0

    def clear(self):
        pass


def _cache(emb, personalized=True):
    b = RecordingBackend()
    return Cache(backend=b, embedding_model=emb, telemetry=False, personalized=personalized), b


# ── the rule ────────────────────────────────────────────────────────────────────
def test_the_reserved_values():
    assert RESERVED_SCOPE_IDS == frozenset({"global", ""})


@pytest.mark.parametrize("value", RESERVED)
def test_reserved_tenant_is_refused_with_a_message_naming_it(value):
    with pytest.raises(ValueError, match=r"tenant_id=.* is reserved"):
        refuse_reserved_scope(value, None, personalized=False)


@pytest.mark.parametrize("value", RESERVED)
def test_reserved_user_is_refused_when_personalized(value):
    with pytest.raises(ValueError, match=r"user_id=.* is reserved"):
        refuse_reserved_scope(None, value, personalized=True)


@pytest.mark.parametrize("value", RESERVED)
def test_reserved_user_is_allowed_when_not_personalized(value):
    refuse_reserved_scope(None, value, personalized=False)      # Cache ignores user_id then


@pytest.mark.parametrize("value", FALSY)
def test_falsy_non_strings_are_refused_too(value):
    with pytest.raises(ValueError, match=r"tenant_id=.* is reserved"):
        refuse_reserved_scope(value, None, personalized=False)
    with pytest.raises(ValueError, match=r"user_id=.* is reserved"):
        refuse_reserved_scope(None, value, personalized=True)


@pytest.mark.parametrize("tenant, user", [(None, None), ("t1", "alice"), ("Global", "GLOBAL"), (" global", "global "),
                                          ("t1", None), (None, "alice"), (5, 7), (b"t1", b"alice")])
def test_everything_else_is_allowed(tenant, user):
    refuse_reserved_scope(tenant, user, personalized=True)


# ── Cache.get / Cache.set refuse before touching anything ──────────────────────
@pytest.mark.parametrize("value", RESERVED)
@pytest.mark.parametrize("which", ["tenant_id", "user_id"])
def test_cache_get_refuses_before_the_backend(fake_embedder, which, value):
    c, b = _cache(fake_embedder)
    before = c._query_count
    with pytest.raises(ValueError, match=f"{which}=.* is reserved"):
        c.get("q", **{which: value})
    assert b.calls == [] and c._query_count == before


@pytest.mark.parametrize("value", RESERVED)
@pytest.mark.parametrize("which", ["tenant_id", "user_id"])
def test_cache_set_refuses_before_the_backend(fake_embedder, which, value):
    c, b = _cache(fake_embedder)
    with pytest.raises(ValueError, match=f"{which}=.* is reserved"):
        c.set("q", "a", **{which: value})
    assert b.calls == []


def test_not_personalized_ignores_a_reserved_user_id_as_before(fake_embedder):
    c, b = _cache(fake_embedder, personalized=False)
    c.set("q", "a", user_id="global")
    c.get("q", user_id="")
    assert b.calls == [("store", None, None), ("search", None, None)]


def test_not_personalized_still_refuses_a_reserved_tenant(fake_embedder):
    c, b = _cache(fake_embedder, personalized=False)
    with pytest.raises(ValueError):
        c.set("q", "a", tenant_id="global")
    assert b.calls == []


def test_ordinary_ids_still_work(fake_embedder):
    c, b = _cache(fake_embedder)
    c.set("q", "a", tenant_id="t1", user_id="alice")
    c.get("q", tenant_id="t1", user_id="alice")
    assert b.calls == [("store", "t1", "alice"), ("search", "t1", "alice")]


def test_a_refused_call_emits_no_event(fake_embedder):
    events = []

    class Sink:
        def emit(self, event):
            events.append(event)

    c = Cache(backend=RecordingBackend(), embedding_model=fake_embedder, telemetry=False, personalized=True,
              event_sink=Sink())
    for call in (lambda: c.get("q", user_id="global"), lambda: c.set("q", "a", tenant_id="")):
        with pytest.raises(ValueError):
            call()
    assert events == []
    c.get("q", user_id="alice")
    assert len(events) == 1                   # the sink is wired: an ordinary call emits


def test_cached_call_refuses_and_never_calls_the_llm(fake_embedder):
    c, b = _cache(fake_embedder)
    called = []
    with pytest.raises(ValueError):
        c.cached_call("q", lambda q: called.append(q) or "a", user_id="global")
    assert called == [] and b.calls == []


def test_async_cache_refuses(fake_embedder):
    from sulci import AsyncCache
    b = RecordingBackend()
    ac = AsyncCache(backend=b, embedding_model=fake_embedder, telemetry=False, personalized=True)
    with pytest.raises(ValueError):
        asyncio.run(ac.aget("q", tenant_id=""))
    with pytest.raises(ValueError):
        asyncio.run(ac.aset("q", "a", user_id="global"))
    assert b.calls == []


# ── the cloud transport: refused before anything is sent ───────────────────────
class RecordingCloud:
    def __init__(self):
        self.sent = []

    def remote_get(self, *a, **kw):
        self.sent.append("get")
        return None, 0.0, 0

    def remote_set(self, *a, **kw):
        self.sent.append("set")


def test_cloud_transport_refuses_before_sending(fake_embedder):
    c, _ = _cache(fake_embedder)
    cloud = RecordingCloud()
    c._backend, c._is_remote_transport = cloud, True
    for call in (lambda: c.get("q", user_id="global"), lambda: c.set("q", "a", user_id=""),
                 lambda: c.get("q", tenant_id="global")):
        with pytest.raises(ValueError):
            call()
    assert cloud.sent == []
