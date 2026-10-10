# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_metadata_reserved.py
───────────────────────────────
A caller's ``metadata`` cannot replace the fields of the cache entry itself (0.9.3).

Before: the Qdrant, Chroma and FAISS backends spread ``metadata`` into the entry
AFTER their own fields, so ``{"user_id": "bob"}`` filed alice's entry under bob,
``{"tenant_id": "t2"}`` planted it in tenant t2 (Qdrant), ``{"expires": 0}`` made
it permanent and ``{"response": ...}`` replaced the served answer (Qdrant, FAISS).

Two layers are tested separately, so each one is known to be load-bearing:
  * ``Cache.set`` drops the reserved keys (one WARNING naming them, never values);
  * each backend's ``store`` drops them too and writes its own fields last, so a
    direct ``backend.store(...)`` call is covered as well.

Runs everywhere: Chroma's collection and FAISS's index are replaced by recording
stand-ins, so neither package is needed. Qdrant runs for real in local mode
(qdrant-client is installed in CI); the real Chroma and FAISS checks at the end
run only where those packages are installed.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import time

import pytest

from sulci import Cache
from sulci._metadata import RESERVED_METADATA_KEYS, without_reserved

HAS_QDRANT = importlib.util.find_spec("qdrant_client") is not None
HAS_CHROMA = importlib.util.find_spec("chromadb") is not None
HAS_FAISS = importlib.util.find_spec("faiss") is not None and importlib.util.find_spec("numpy") is not None

SEVEN = {"key", "query", "response", "tenant_id", "user_id", "expires", "created"}
SECRET = "SECRET-VALUE-7c1e"          # must never appear in a log line
PLANT = {
    "key": "planted-key", "query": "planted query", "response": "PLANTED",
    "tenant_id": "t2", "user_id": "bob", "expires": 0, "created": 1.0,
}
VEC = [1.0] + [0.0] * 63


def _dropped(caplog):
    """The keys named as dropped, read from the WARNING's own argument (the message also lists every reserved
    name, so searching its text would pass whatever was dropped). Exactly one such WARNING is required."""
    recs = [r for r in caplog.records if r.name == "sulci._metadata" and r.levelno == logging.WARNING]
    assert len(recs) == 1, f"{len(recs)} warnings"
    return recs[0].args[0].split(", ")


# ── the list ────────────────────────────────────────────────────────────────────
def test_reserved_list_is_the_seven_entry_fields():
    assert RESERVED_METADATA_KEYS == SEVEN
    assert isinstance(RESERVED_METADATA_KEYS, frozenset)


# ── the helper ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(SEVEN))
def test_each_reserved_key_is_dropped(name, caplog):
    with caplog.at_level(logging.WARNING, logger="sulci"):
        out = without_reserved({name: SECRET, "source": "docs"})
    assert out == {"source": "docs"}
    assert _dropped(caplog) == [name] and SECRET not in caplog.text


def test_ordinary_keys_are_kept_exactly(caplog):
    md = {"source": "docs", "n": 3, "nested": {"user_id": "x"}, "Query": "case differs", "query_text": "kept"}
    with caplog.at_level(logging.WARNING, logger="sulci"):
        assert without_reserved(md) == md
    assert caplog.records == []


@pytest.mark.parametrize("empty", [None, {}])
def test_empty_gives_empty_dict_and_no_log(empty, caplog):
    with caplog.at_level(logging.WARNING, logger="sulci"):
        assert without_reserved(empty) == {}
    assert caplog.records == []


def test_callers_dict_is_not_modified():
    md = dict(PLANT, source="docs")
    before = dict(md)
    out = without_reserved(md)
    assert md == before and out is not md


def test_one_warning_names_every_dropped_key_and_no_value(caplog):
    with caplog.at_level(logging.WARNING, logger="sulci"):
        without_reserved({k: SECRET for k in SEVEN} | {"source": SECRET})
    assert _dropped(caplog) == sorted(SEVEN)
    assert SECRET not in caplog.text


def test_a_mapping_that_is_not_a_dict_is_handled():
    from types import MappingProxyType
    assert without_reserved(MappingProxyType({"user_id": "bob", "source": "docs"})) == {"source": "docs"}


def test_an_unpackable_object_that_is_not_a_mapping_is_filtered(caplog):
    class KeysOnly:                       # ** can unpack it; it is not a registered Mapping
        def __init__(self, d):
            self._d = d
        def keys(self):
            return self._d.keys()
        def __getitem__(self, k):
            return self._d[k]
        def __len__(self):
            return len(self._d)
    obj = KeysOnly({"user_id": "bob", "response": "PLANTED", "source": "docs"})
    with caplog.at_level(logging.WARNING, logger="sulci"):
        assert without_reserved(obj) == {"source": "docs"}
    assert _dropped(caplog) == ["response", "user_id"]
    assert obj._d == {"user_id": "bob", "response": "PLANTED", "source": "docs"}


def test_a_non_mapping_is_returned_unchanged():
    odd = ["not", "a", "mapping"]
    assert without_reserved(odd) is odd


def test_non_string_keys_are_kept():
    assert without_reserved({1: "a", ("user_id",): "b"}) == {1: "a", ("user_id",): "b"}


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


def _cache(backend, emb, **kw):
    return Cache(backend=backend, embedding_model=emb, telemetry=False, personalized=True, **kw)


def test_cache_set_hands_the_backend_no_reserved_key(caplog, fake_embedder):
    b = RecordingBackend()
    with caplog.at_level(logging.WARNING, logger="sulci"):
        _cache(b, fake_embedder).set("how do i rotate the key", "answer", tenant_id="t1", user_id="alice",
                      metadata=dict(PLANT, source="docs"))
    (call,) = b.calls
    assert call["metadata"] == {"source": "docs"}
    assert (call["tenant_id"], call["user_id"], call["response"]) == ("t1", "alice", "answer")
    assert call["expires"] and call["expires"] > time.time()
    assert _dropped(caplog) == sorted(SEVEN) and "PLANTED" not in caplog.text


def test_cache_set_without_reserved_keys_logs_nothing(caplog, fake_embedder):
    b = RecordingBackend()
    with caplog.at_level(logging.WARNING, logger="sulci"):
        _cache(b, fake_embedder).set("q", "a", user_id="alice", metadata={"source": "docs"})
        _cache(b, fake_embedder).set("q2", "a", user_id="alice")
    assert [c["metadata"] for c in b.calls] == [{"source": "docs"}, {}]
    assert not [r for r in caplog.records if r.name.startswith("sulci._metadata")]


def test_async_cache_aset_goes_through_the_same_rule(fake_embedder):
    from sulci import AsyncCache
    b = RecordingBackend()
    ac = AsyncCache(backend=b, embedding_model=fake_embedder, telemetry=False, personalized=True)
    asyncio.run(ac.aset("q", "a", user_id="alice", metadata={"user_id": "bob", "source": "docs"}))
    assert b.calls[0]["metadata"] == {"source": "docs"} and b.calls[0]["user_id"] == "alice"


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
    b._ensure = lambda dim: None
    b._save = lambda: None
    return b


def _chroma_written(metadata):
    b = _chroma_standin()
    b.store("k1", "real question", "real answer", VEC, tenant_id="t1", user_id="alice",
            expires=4102444800.0, metadata=metadata)
    up = b._col.upserts[0]
    return up["metadatas"][0], up["documents"][0], up["ids"][0]


def _faiss_written(metadata):
    b = _faiss_standin()
    b.store("k1", "real question", "real answer", VEC, tenant_id="t1", user_id="alice",
            expires=4102444800.0, metadata=metadata)
    return b._meta[0]


def _qdrant_written(tmp_path, metadata):
    from sulci.backends.qdrant import QdrantBackend
    b = QdrantBackend(db_path=str(tmp_path / "q"), dimension=64)
    b.store("k1", "real question", "real answer", VEC, tenant_id="t1", user_id="alice",
            expires=4102444800.0, metadata=metadata)
    points, _ = b._client.scroll(b.COLLECTION, limit=10, with_payload=True)
    assert len(points) == 1
    return b, points[0].payload


def test_chroma_store_keeps_its_own_fields(caplog):
    with caplog.at_level(logging.WARNING, logger="sulci"):
        meta, doc, key = _chroma_written(dict(PLANT, source="docs"))
    assert meta == {"query": "real question", "expires": 4102444800.0, "user_id": "alice",
                    "tenant_id": "t1", "source": "docs"}
    assert (doc, key) == ("real answer", "k1")
    assert _dropped(caplog) == sorted(SEVEN)


def test_faiss_store_keeps_its_own_fields(caplog):
    with caplog.at_level(logging.WARNING, logger="sulci"):
        meta = _faiss_written(dict(PLANT, source="docs"))
    assert meta == {"key": "k1", "query": "real question", "response": "real answer",
                    "user_id": "alice", "expires": 4102444800.0, "source": "docs"}
    assert _dropped(caplog) == sorted(SEVEN)


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
def test_qdrant_store_keeps_its_own_fields(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="sulci"):
        _, p = _qdrant_written(tmp_path, dict(PLANT, source="docs"))
    assert {k: p[k] for k in ("key", "query", "response", "tenant_id", "user_id", "expires")} == {
        "key": "k1", "query": "real question", "response": "real answer",
        "tenant_id": "t1", "user_id": "alice", "expires": 4102444800.0}
    assert p["created"] > 1.0 and p["source"] == "docs"
    assert _dropped(caplog) == sorted(SEVEN)


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
def test_qdrant_planted_entry_is_still_alices_and_still_t1s(tmp_path):
    b, _ = _qdrant_written(tmp_path, dict(PLANT))
    assert b.search_match(VEC, 0.5, tenant_id="t1", user_id="alice") == ("real answer", pytest.approx(1.0), "real question")
    assert b.search(VEC, 0.5, tenant_id="t1", user_id="bob")[0] is None
    assert b.search(VEC, 0.5, tenant_id="t2", user_id="alice")[0] is None
    assert b.search(VEC, 0.5, tenant_id="t1")[0] is None          # not moved to the shared scope


# ── layer 2b: the ORDER alone — with the filter switched off, the entry's own fields still win ─────────────
def _no_filter(monkeypatch, module):
    monkeypatch.setattr(module, "without_reserved", lambda md: dict(md or {}))


def test_chroma_order_alone_keeps_its_own_fields(monkeypatch):
    from sulci.backends import chroma
    _no_filter(monkeypatch, chroma)
    meta, _, _ = _chroma_written(dict(PLANT))
    assert (meta["query"], meta["user_id"], meta["tenant_id"], meta["expires"]) == ("real question", "alice", "t1", 4102444800.0)


def test_faiss_order_alone_keeps_its_own_fields(monkeypatch):
    from sulci.backends import faiss
    _no_filter(monkeypatch, faiss)
    meta = _faiss_written(dict(PLANT))
    assert (meta["key"], meta["query"], meta["response"], meta["user_id"], meta["expires"]) == (
        "k1", "real question", "real answer", "alice", 4102444800.0)


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
def test_qdrant_order_alone_keeps_its_own_fields(tmp_path, monkeypatch):
    from sulci.backends import qdrant
    _no_filter(monkeypatch, qdrant)
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
def test_end_to_end_a_plant_changes_nothing(tmp_path, plant, fake_embedder):
    for name, make in _backends(tmp_path):
        c = Cache(backend=make(), embedding_model=fake_embedder, telemetry=False, personalized=True)
        c.set(Q, R, tenant_id="t1", user_id="alice", metadata=dict(plant))
        assert c.get(Q, tenant_id="t1", user_id="alice")[0] == R, name
        assert c.get(Q, tenant_id="t1", user_id="bob")[0] is None, name
        if name == "qdrant":              # the only backend that reads tenant_id
            assert c.get(Q, tenant_id="t2", user_id="alice")[0] is None, name
        c.clear()
