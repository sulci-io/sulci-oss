# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_backend_isolation.py
───────────────────────────────
Every backend reads only the caller's own (tenant, user) scope (0.9.3).

Before: only Qdrant filtered a lookup by ``tenant_id``; Chroma, FAISS, SQLite,
Redis and Milvus accepted it and ignored it, and a lookup without ``user_id``
read every user's entries. Measured through ``Cache``: 14 of 32 cross-scope
reads served another tenant's or user's answer on each of those five.

Now every backend filters on both, with ``None`` meaning the shared scope
``"global"`` and nothing else, and the scope is part of the search itself — a
reader's own entry cannot be crowded out of the candidates by other scopes'
copies of the same question.

The full matrix runs for real on SQLite and Qdrant everywhere, on Redis wherever
a server answers on localhost, and on Chroma, FAISS and Milvus wherever those
packages are installed. Chroma and Milvus are also checked through recording
stand-ins, so the filter they are given is tested on every machine.
"""
from __future__ import annotations

import importlib.util
import os
import uuid

import pytest

from sulci import Cache

Q = "how do i export my invoices"
VEC = [1.0] + [0.0] * 63


def _has(mod):
    return importlib.util.find_spec(mod) is not None


def _redis_up():
    if not _has("redis"):
        return False
    try:
        import redis
        return bool(redis.Redis.from_url("redis://localhost:6379", socket_connect_timeout=0.5).ping())
    except Exception:
        return False


AVAILABLE = {
    "sqlite": True,
    "qdrant": _has("qdrant_client"),
    "redis": _redis_up(),
    "chroma": _has("chromadb"),
    "faiss": _has("faiss") and _has("numpy"),
    "milvus": _has("pymilvus") and _has("milvus_lite"),
}
BACKENDS = [pytest.param(n, marks=pytest.mark.skipif(not ok, reason=f"{n} not available here"))
            for n, ok in AVAILABLE.items()]


def _cache(name, tmp_path, emb):
    d = str(tmp_path / name)
    if name == "sqlite":
        return Cache(backend="sqlite", db_path=d, embedding_model=emb, telemetry=False, personalized=True)
    if name == "qdrant":
        from sulci.backends.qdrant import QdrantBackend; b = QdrantBackend(db_path=d, dimension=64)
    elif name == "redis":
        from sulci.backends.redis import RedisBackend; b = RedisBackend(key_prefix=f"sulci-test-{uuid.uuid4().hex[:12]}:")
    elif name == "chroma":
        from sulci.backends.chroma import ChromaBackend; b = ChromaBackend(db_path=d)
    elif name == "faiss":
        from sulci.backends.faiss import FAISSBackend; b = FAISSBackend(db_path=d)
    elif name == "milvus":
        from sulci.backends.milvus import MilvusBackend
        os.makedirs(d, exist_ok=True); b = MilvusBackend(db_path=os.path.join(d, "m.db"))
    return Cache(backend=b, embedding_model=emb, telemetry=False, personalized=True)


def _scope(t, u):
    return {k: v for k, v in (("tenant_id", t), ("user_id", u)) if v is not None}


WRITERS = [(None, None), ("t1", None), ("t1", "alice"), (None, "alice")]
READERS = [(t, u) for t in (None, "t1", "t2") for u in (None, "alice", "bob")]


# ── the matrix: a reader sees an entry exactly when its scope is the writer's ───
@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("writer", WRITERS, ids=lambda w: f"{w[0] or '-'}/{w[1] or '-'}")
def test_a_lookup_reads_only_its_own_scope(name, writer, tmp_path, fake_embedder):
    c = _cache(name, tmp_path, fake_embedder)
    try:
        c.set(Q, "ANS", **_scope(*writer))
        seen = {r for r in READERS if c.get(Q, **_scope(*r))[0] is not None}
        assert seen == {writer}
    finally:
        c.clear()


@pytest.mark.parametrize("name", BACKENDS)
def test_other_scopes_cannot_crowd_out_a_readers_own_entry(name, tmp_path, fake_embedder):
    c = _cache(name, tmp_path, fake_embedder)
    try:
        for i in range(8):
            c.set(Q, f"OTHER{i}", tenant_id=f"x{i}", user_id=f"y{i}")
        c.set(Q, "MINE", tenant_id="t1", user_id="alice")
        assert c.get(Q, tenant_id="t1", user_id="alice")[0] == "MINE"
    finally:
        c.clear()


@pytest.mark.parametrize("name", BACKENDS)
def test_near_duplicates_in_other_scopes_cannot_crowd_out_a_readers_own_entry(name, tmp_path, fake_embedder):
    # Identical copies cannot show crowding (the review of this patch found an approximate search that lost the
    # reader's own paraphrased entry behind ~1000 near-duplicates in other scopes); near-duplicates can.
    import random
    rnd = random.Random(7)
    base = "how do i export my invoices to a csv file from the billing page"
    c = _cache(name, tmp_path, fake_embedder)
    c.threshold = 0.5
    try:
        for i in range(300):
            c.set(f"{base} {rnd.choice(['please', 'now', 'quickly', 'today', 'again'])} {i}", f"OTHER{i}",
                  tenant_id=f"x{i % 30}", user_id="u")
        c.set(base + " for march", "MINE", tenant_id="me", user_id="u")
        assert c.get(base + " for april", tenant_id="me", user_id="u")[0] == "MINE"
    finally:
        c.clear()


@pytest.mark.parametrize("which", ["tenant_id", "user_id"])
def test_an_id_that_is_not_valid_text_is_refused(which, tmp_path, fake_embedder):
    c = _cache("sqlite", tmp_path, fake_embedder)
    for call in (lambda: c.set(Q, "A", **{which: "t\ud800"}), lambda: c.get(Q, **{which: "t\ud800"})):
        with pytest.raises(ValueError, match="not valid Unicode text"):
            call()


@pytest.mark.parametrize("name", BACKENDS)
def test_not_personalized_keeps_tenants_apart(name, tmp_path, fake_embedder):
    c = _cache(name, tmp_path, fake_embedder)
    c.personalized = False
    try:
        c.set(Q, "T1", tenant_id="t1", user_id="alice")
        assert c.get(Q, tenant_id="t1", user_id="bob")[0] == "T1"      # user ignored when not personalized
        assert c.get(Q, tenant_id="t2", user_id="alice")[0] is None
        assert c.get(Q)[0] is None
    finally:
        c.clear()


# ── the filter each backend is given, on every machine ─────────────────────────
class _Collection:
    def __init__(self):
        self.where = []

    def query(self, **kw):
        self.where.append(kw.get("where"))
        return {"documents": [[]], "distances": [[]], "metadatas": [[]]}


@pytest.mark.parametrize("tenant, user, want", [
    (None, None, ("global", "global")), ("t1", None, ("t1", "global")),
    (None, "alice", ("global", "alice")), ("t1", "alice", ("t1", "alice"))])
def test_chroma_filters_on_tenant_and_user(tenant, user, want):
    from sulci.backends.chroma import ChromaBackend
    b = ChromaBackend.__new__(ChromaBackend)
    b._col = _Collection()
    b.search(VEC, 0.5, tenant_id=tenant, user_id=user)
    assert b._col.where == [{"$and": [{"tenant_id": want[0]}, {"user_id": want[1]}]}]


class _MilvusClient:
    def __init__(self):
        self.filters, self.rows = [], []

    def insert(self, collection, rows):
        self.rows.extend(rows)

    def search(self, **kw):
        self.filters.append(kw["filter"])
        return [[]]


def _milvus():
    from sulci.backends.milvus import MilvusBackend
    b = MilvusBackend.__new__(MilvusBackend)
    b._client, b._ready = _MilvusClient(), True
    b._ensure = lambda dim: None
    return b


def test_milvus_files_and_filters_on_the_scope_key():
    """0.10.0: each entry carries the key of its (tenant, user); a lookup filters on its own key alone."""
    from sulci.sessions.scope import user_scope
    b = _milvus()
    b.store("k", Q, "A", VEC, tenant_id="t1", user_id="alice")
    b.store("k", Q, "B", VEC)
    assert [r["scope"] for r in b._client.rows] == [user_scope("t1", "alice"), user_scope(None, None)]
    assert not any("tenant_id" in r or "user_id" in r for r in b._client.rows)     # no raw ids stored
    b.search(VEC, 0.5, tenant_id="t1", user_id="alice")
    b.search(VEC, 0.5)
    assert b._client.filters == [f'scope == "{user_scope("t1", "alice")}"', f'scope == "{user_scope(None, None)}"']


def test_milvus_filter_holds_no_caller_text():
    b = _milvus()
    b.search(VEC, 0.5, tenant_id='t" or "1"=="1', user_id="a\\b\nc")
    (f,) = b._client.filters
    assert '"1"' not in f and "\\" not in f and "\n" not in f
    assert f.count('"') == 2


# ── every shipped backend now declares that it isolates ────────────────────────
@pytest.mark.parametrize("module, cls", [("sqlite", "SQLiteBackend"), ("chroma", "ChromaBackend"),
                                         ("faiss", "FAISSBackend"), ("redis", "RedisBackend"),
                                         ("milvus", "MilvusBackend"), ("qdrant", "QdrantBackend")])
def test_every_local_backend_declares_isolation(module, cls):
    mod = importlib.import_module(f"sulci.backends.{module}")
    assert getattr(mod, cls).ENFORCES_TENANT_ISOLATION is True
