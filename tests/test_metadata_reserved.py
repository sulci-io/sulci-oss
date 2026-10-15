# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_metadata_reserved.py
───────────────────────────────
A caller's ``metadata`` cannot use the names of the cache entry's own fields (0.9.3).

Before: the Qdrant, Chroma and FAISS backends spread ``metadata`` into the entry
AFTER their own fields, so ``{"user_id": "bob"}`` filed alice's entry under bob,
``{"tenant_id": "t2"}`` planted it in tenant t2 (Qdrant), ``{"expires": 0}`` made
it permanent and ``{"response": ...}`` replaced the served answer (Qdrant, FAISS).

Now such a ``metadata`` is refused with ``ValueError`` (the gateway answers 422),
naming the keys and never their values; a ``metadata`` that is not a mapping is
refused with ``TypeError``. Measured before this rule (``measure_d2.py``, at
oss-0006): every backend stored the entry with the reserved keys dropped and one
WARNING; the managed backend dropped the whole ``metadata`` silently; a list was
stored by SQLite, Redis, Milvus and the managed backend and failed in the other
three with an unrelated ``TypeError``.

Three layers are tested separately, so each one is known to be load-bearing:
  * ``Cache.set`` refuses before anything is embedded, written or sent;
  * each backend's ``store`` that keeps ``metadata`` refuses too, so a direct
    ``backend.store(...)`` call is covered as well;
  * those backends write ``metadata`` FIRST, so the entry's own fields win even
    with the refusal switched off.

Runs everywhere: Chroma's collection and FAISS's index are replaced by recording
stand-ins, so neither package is needed. Qdrant runs for real in local mode
(qdrant-client is installed in CI); the real Chroma and FAISS checks at the end
run only where those packages are installed.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json

import httpx
import pytest

from sulci import Cache
from sulci._metadata import RESERVED_METADATA_KEYS, checked_metadata

HAS_QDRANT = importlib.util.find_spec("qdrant_client") is not None
HAS_CHROMA = importlib.util.find_spec("chromadb") is not None
HAS_FAISS = importlib.util.find_spec("faiss") is not None and importlib.util.find_spec("numpy") is not None

SEVEN = {"key", "query", "response", "tenant_id", "user_id", "expires", "created"}
SECRET = "SECRET-VALUE-7c1e"          # must never appear in an error
PLANT = {
    "key": "planted-key", "query": "planted query", "response": "PLANTED",
    "tenant_id": "t2", "user_id": "bob", "expires": 0, "created": 1.0,
}
VEC = [1.0] + [0.0] * 63


def _refused(excinfo):
    """The keys the ValueError names, read from the start of its message (the message also lists every reserved
    name, so searching its text would pass whatever was refused)."""
    msg = str(excinfo.value)
    assert msg.startswith("metadata keys ") and " name fields of the cache entry itself" in msg, msg
    return msg[len("metadata keys "):msg.index(" name fields")].split(", ")


# ── the list ────────────────────────────────────────────────────────────────────
def test_reserved_list_is_the_seven_entry_fields():
    assert RESERVED_METADATA_KEYS == SEVEN
    assert isinstance(RESERVED_METADATA_KEYS, frozenset)


# ── the helper ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(SEVEN))
def test_each_reserved_key_is_refused(name):
    with pytest.raises(ValueError) as e:
        checked_metadata({name: SECRET, "source": "docs"})
    assert _refused(e) == [name] and SECRET not in str(e.value)


def test_ordinary_keys_are_kept_exactly():
    md = {"source": "docs", "n": 3, "nested": {"user_id": "x"}, "Query": "case differs", "query_text": "kept"}
    out = checked_metadata(md)
    assert out == md and out is not md


@pytest.mark.parametrize("empty", [None, {}])
def test_empty_gives_empty_dict(empty):
    assert checked_metadata(empty) == {}


def test_callers_dict_is_not_modified_either_way():
    md = dict(PLANT, source="docs")
    before = dict(md)
    with pytest.raises(ValueError):
        checked_metadata(md)
    assert md == before


def test_one_error_names_every_reserved_key_and_no_value():
    with pytest.raises(ValueError) as e:
        checked_metadata({k: SECRET for k in SEVEN} | {"source": SECRET})
    assert _refused(e) == sorted(SEVEN)
    assert SECRET not in str(e.value)


def test_a_mapping_that_is_not_a_dict_is_handled():
    from types import MappingProxyType
    assert checked_metadata(MappingProxyType({"source": "docs"})) == {"source": "docs"}
    with pytest.raises(ValueError):
        checked_metadata(MappingProxyType({"user_id": "bob", "source": "docs"}))


def test_an_unpackable_object_that_is_not_a_mapping_is_checked():
    class KeysOnly:                       # ** can unpack it; it is not a registered Mapping
        def __init__(self, d):
            self._d = d
        def keys(self):
            return self._d.keys()
        def __getitem__(self, k):
            return self._d[k]
        def __len__(self):
            return len(self._d)
    assert checked_metadata(KeysOnly({"source": "docs"})) == {"source": "docs"}
    obj = KeysOnly({"user_id": "bob", "response": "PLANTED", "source": "docs"})
    with pytest.raises(ValueError) as e:
        checked_metadata(obj)
    assert _refused(e) == ["response", "user_id"]
    assert obj._d == {"user_id": "bob", "response": "PLANTED", "source": "docs"}


@pytest.mark.parametrize("odd", [["not", "a", "mapping"], "user_id", 3])
def test_a_non_mapping_is_refused(odd):
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        checked_metadata(odd)


def test_non_string_keys_are_kept():
    assert checked_metadata({1: "a", ("user_id",): "b"}) == {1: "a", ("user_id",): "b"}


def test_a_str_subclass_cannot_hide_a_reserved_name():
    class Sly(str):                       # its own hash keeps it out of a set lookup by default
        def __hash__(self):
            return 0
    with pytest.raises(ValueError) as e:
        checked_metadata({Sly("expires"): 0, "source": "docs"})
    assert _refused(e) == ["expires"]


# ── layer 1: Cache.set ──────────────────────────────────────────────────────────
class RecordingBackend:
    """Stores nothing; remembers exactly what Cache.set handed to store()."""
    def __init__(self):
        self.calls = []

    def store(self, key, query, response, embedding, *, tenant_id=None, user_id=None, expires=None, metadata=None):
        self.calls.append(dict(key=key, query=query, response=response, tenant_id=tenant_id,
                               user_id=user_id, expires=expires, metadata=metadata))

    def search(self, embedding, threshold, *, tenant_id=None, user_id=None, now=None):
        return None, 0.0

    def clear(self):
        self.calls.clear()


class CountingEmbedder:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def embed(self, text):
        self.calls += 1
        return self.inner.embed(text)


def _cache(backend, emb, **kw):
    return Cache(backend=backend, embedding_model=emb, telemetry=False, personalized=True, **kw)


def test_cache_set_refuses_before_anything_is_embedded_or_stored(fake_embedder):
    b, emb = RecordingBackend(), CountingEmbedder(fake_embedder)
    c = _cache(b, emb)
    with pytest.raises(ValueError) as e:
        c.set("how do i rotate the key", "answer", tenant_id="t1", user_id="alice", metadata=dict(PLANT, source="docs"))
    assert _refused(e) == sorted(SEVEN) and "PLANTED" not in str(e.value)
    assert b.calls == [] and emb.calls == 0


def test_cache_set_refuses_a_non_mapping(fake_embedder):
    b = RecordingBackend()
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        _cache(b, fake_embedder).set("q", "a", user_id="alice", metadata=["not", "a", "mapping"])
    assert b.calls == []


def test_cache_set_with_ordinary_metadata_stores_it(fake_embedder):
    b = RecordingBackend()
    _cache(b, fake_embedder).set("q", "a", user_id="alice", metadata={"source": "docs"})
    _cache(b, fake_embedder).set("q2", "a", user_id="alice")
    assert [c["metadata"] for c in b.calls] == [{"source": "docs"}, {}]


def test_async_cache_aset_goes_through_the_same_rule(fake_embedder):
    from sulci import AsyncCache
    b = RecordingBackend()
    ac = AsyncCache(backend=b, embedding_model=fake_embedder, telemetry=False, personalized=True)
    with pytest.raises(ValueError):
        asyncio.run(ac.aset("q", "a", user_id="alice", metadata={"user_id": "bob", "source": "docs"}))
    assert b.calls == []


def test_the_managed_backend_refuses_before_anything_is_sent():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"status": "ok"})
    c = Cache(backend="sulci", api_key="sk-sulci-test0000000000", telemetry=False)
    c._backend._client.close()
    c._backend._client = httpx.Client(base_url="https://api.sulci.io", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        c.set("q", "a", user_id="alice", metadata={"user_id": "bob"})
    assert seen == []


def test_sqlite_refuses_and_stores_nothing(tmp_path, fake_embedder):
    c = Cache(backend="sqlite", db_path=str(tmp_path / "s"), embedding_model=fake_embedder, telemetry=False,
              personalized=True)
    with pytest.raises(ValueError):
        c.set("q", "a", user_id="alice", metadata={"expires": 0})
    assert c._backend._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0
    c.set("q", "a", user_id="alice", metadata={"source": "docs"})
    (md,) = c._backend._conn.execute("SELECT metadata FROM cache").fetchone()
    assert json.loads(md) == {"source": "docs"}


# ── layer 2: each backend's store ───────────────────────────────────────────────
class RecordingCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, **kw):
        self.upserts.append(kw)


def _chroma_standin():
    from sulci.backends.chroma import ChromaBackend
    b = ChromaBackend.__new__(ChromaBackend)
    b._col = RecordingCollection()
    return b


class _FakeNp:
    @staticmethod
    def array(x, dtype=None):
        return x


class _RecordingIndex:
    def __init__(self):
        self.added = []

    def add(self, vec):
        self.added.append(vec)


def _faiss_standin():
    from sulci.backends.faiss import FAISSBackend
    b = FAISSBackend.__new__(FAISSBackend)
    b._index, b._meta, b._np = _RecordingIndex(), [], _FakeNp()
    b._scopes = {}                                  # 0.9.3 (oss-0005): FAISS keeps a scope map
    b._lock = __import__("threading").RLock()       # 0.10.0 (oss-0022): store() holds the backend's lock
    b._ensure = lambda dim: None
    b._save = lambda: None
    return b


def _store(b, metadata):
    b.store("k1", "real question", "real answer", VEC, tenant_id="t1", user_id="alice",
            expires=4102444800.0, metadata=metadata)


def _chroma_written(metadata):
    b = _chroma_standin()
    _store(b, metadata)
    up = b._col.upserts[0]
    return up["metadatas"][0], up["documents"][0], up["ids"][0]


def _faiss_written(metadata):
    b = _faiss_standin()
    _store(b, metadata)
    return b._meta[0]


def _qdrant_backend(tmp_path):
    from sulci.backends.qdrant import QdrantBackend
    return QdrantBackend(db_path=str(tmp_path / "q"), dimension=64)


def _qdrant_points(b):
    points, _ = b._client.scroll(b.COLLECTION, limit=10, with_payload=True)
    return points


def _qdrant_written(tmp_path, metadata):
    b = _qdrant_backend(tmp_path)
    _store(b, metadata)
    points = _qdrant_points(b)
    assert len(points) == 1
    return b, points[0].payload


def test_chroma_store_refuses_and_writes_nothing():
    b = _chroma_standin()
    with pytest.raises(ValueError) as e:
        _store(b, dict(PLANT, source="docs"))
    assert _refused(e) == sorted(SEVEN) and b._col.upserts == []


def test_chroma_store_writes_ordinary_metadata():
    meta, doc, key = _chroma_written({"source": "docs"})
    assert meta == {"query": "real question", "expires": 4102444800.0, "user_id": "alice",
                    "tenant_id": "t1", "source": "docs"}
    from sulci.backends._scope import scoped_id     # 0.9.3 (oss-0003): Chroma ids are scoped to tenant and user
    assert (doc, key) == ("real answer", scoped_id("k1", "t1", "alice"))


def test_faiss_store_refuses_and_writes_nothing():
    b = _faiss_standin()
    ensured = []
    b._ensure = ensured.append        # a fresh FAISS backend creates its index here, sized to the vector
    with pytest.raises(ValueError) as e:
        _store(b, dict(PLANT, source="docs"))
    assert _refused(e) == sorted(SEVEN) and b._meta == [] and b._index.added == [] and ensured == []


def test_faiss_store_writes_ordinary_metadata():
    meta = _faiss_written({"source": "docs"})
    assert meta == {"key": "k1", "query": "real question", "response": "real answer", "tenant_id": "t1",
                    "user_id": "alice", "expires": 4102444800.0, "source": "docs"}     # tenant_id: 0.9.3 (oss-0005)


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
def test_qdrant_store_refuses_and_writes_nothing(tmp_path):
    b = _qdrant_backend(tmp_path)
    with pytest.raises(ValueError) as e:
        _store(b, dict(PLANT, source="docs"))
    assert _refused(e) == sorted(SEVEN) and _qdrant_points(b) == []


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
def test_qdrant_store_writes_ordinary_metadata(tmp_path):
    _, p = _qdrant_written(tmp_path, {"source": "docs"})
    assert {k: p[k] for k in ("key", "query", "response", "tenant_id", "user_id", "expires")} == {
        "key": "k1", "query": "real question", "response": "real answer",
        "tenant_id": "t1", "user_id": "alice", "expires": 4102444800.0}
    assert p["created"] > 1.0 and p["source"] == "docs"


def test_sqlite_store_refuses_and_writes_nothing(tmp_path):
    from sulci.backends.sqlite import SQLiteBackend
    b = SQLiteBackend(db_path=str(tmp_path / "s"))
    with pytest.raises(ValueError) as e:
        _store(b, {"expires": 0, "source": "docs"})
    assert _refused(e) == ["expires"]
    assert b._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0


@pytest.mark.parametrize("which", ["qdrant", "chroma", "faiss", "sqlite"])
def test_every_store_that_keeps_metadata_refuses_a_non_mapping(which, tmp_path):
    if which == "qdrant":
        if not HAS_QDRANT:
            pytest.skip("qdrant-client not installed")
        b = _qdrant_backend(tmp_path)
    elif which == "sqlite":
        from sulci.backends.sqlite import SQLiteBackend
        b = SQLiteBackend(db_path=str(tmp_path / "s"))
    else:
        b = _chroma_standin() if which == "chroma" else _faiss_standin()
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        _store(b, ["not", "a", "mapping"])


# ── layer 3: the ORDER alone — with the refusal switched off, the entry's own fields still win ────────────
def _no_check(monkeypatch, module):
    monkeypatch.setattr(module, "checked_metadata", lambda md: dict(md or {}))


def test_chroma_order_alone_keeps_its_own_fields(monkeypatch):
    from sulci.backends import chroma
    _no_check(monkeypatch, chroma)
    meta, _, _ = _chroma_written(dict(PLANT))
    assert (meta["query"], meta["user_id"], meta["tenant_id"], meta["expires"]) == ("real question", "alice", "t1", 4102444800.0)


def test_faiss_order_alone_keeps_its_own_fields(monkeypatch):
    from sulci.backends import faiss
    _no_check(monkeypatch, faiss)
    meta = _faiss_written(dict(PLANT))
    assert (meta["key"], meta["query"], meta["response"], meta["user_id"], meta["expires"]) == (
        "k1", "real question", "real answer", "alice", 4102444800.0)


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
def test_qdrant_order_alone_keeps_its_own_fields(tmp_path, monkeypatch):
    from sulci.backends import qdrant
    _no_check(monkeypatch, qdrant)
    _, p = _qdrant_written(tmp_path, dict(PLANT))
    assert {k: p[k] for k in ("key", "query", "response", "tenant_id", "user_id", "expires")} == {
        "key": "k1", "query": "real question", "response": "real answer",
        "tenant_id": "t1", "user_id": "alice", "expires": 4102444800.0}
    assert p["created"] > 1.0


# ── the pin: the list is exactly what the backends write ────────────────────────
def test_pin_reserved_list_equals_what_the_backends_write(tmp_path):
    chroma, _, _ = _chroma_written({})
    written = {"chroma": set(chroma), "faiss": set(_faiss_written({}))}
    if HAS_QDRANT:
        written["qdrant"] = set(_qdrant_written(tmp_path, {})[1])
    else:   # read from the source when the client is absent: every key of the payload literal
        import inspect, re
        from sulci.backends import qdrant
        src = inspect.getsource(qdrant.QdrantBackend.store)
        written["qdrant"] = set(re.findall(r'"([a-z_]+)":', src))
    for name, keys in written.items():
        assert keys <= RESERVED_METADATA_KEYS, f"{name} writes {sorted(keys - RESERVED_METADATA_KEYS)}, not reserved"
    assert set().union(*written.values()) == RESERVED_METADATA_KEYS


# ── end to end through Cache, on every backend this machine has ────────────────
def _backends(tmp_path):
    out = []
    if HAS_QDRANT:
        from sulci.backends.qdrant import QdrantBackend
        out.append(("qdrant", lambda: QdrantBackend(db_path=str(tmp_path / "qe"), dimension=64)))
    if HAS_CHROMA:
        from sulci.backends.chroma import ChromaBackend
        out.append(("chroma", lambda: ChromaBackend(db_path=str(tmp_path / "ce"))))
    if HAS_FAISS:
        from sulci.backends.faiss import FAISSBackend
        out.append(("faiss", lambda: FAISSBackend(db_path=str(tmp_path / "fe"))))
    return out


Q, R = "how do i rotate the api key", "Rotate it in Settings > Keys."


@pytest.mark.skipif(not (HAS_QDRANT or HAS_CHROMA or HAS_FAISS), reason="no vector backend installed")
@pytest.mark.parametrize("plant", [{"user_id": "bob"}, {"user_id": "global"}, {"response": "PLANTED"},
                                   {"expires": 1.0}, {"tenant_id": "t2"}])
def test_end_to_end_a_plant_is_refused_and_nothing_is_stored(tmp_path, plant, fake_embedder):
    for name, make in _backends(tmp_path):
        c = Cache(backend=make(), embedding_model=fake_embedder, telemetry=False, personalized=True)
        with pytest.raises(ValueError):
            c.set(Q, R, tenant_id="t1", user_id="alice", metadata=dict(plant))
        for t, u in (("t1", "alice"), ("t1", "bob"), ("t2", "alice"), ("t1", None), (None, None)):
            assert c.get(Q, tenant_id=t, user_id=u)[0] is None, (name, t, u)
        c.set(Q, R, tenant_id="t1", user_id="alice", metadata={"source": "docs"})
        assert c.get(Q, tenant_id="t1", user_id="alice")[0] == R, name
        c.clear()
