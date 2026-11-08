# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_variant_accept.py — oss-0064: ``variant=`` and ``accept=``.

``variant`` keeps an entry written for one kind of request (one system prompt, model and parameters: the hosted
chat endpoint's fingerprint) away from every other kind, inside the search itself. ``accept`` lets a caller turn a
candidate hit away (an identifier check) BEFORE the hit is counted, recorded or emitted, so a hit nobody was served
is never billed as one.

Runs offline: embedded Qdrant and the deterministic FakeEmbedder (plumbing, not retrieval quality).
"""
from __future__ import annotations

import asyncio
import logging

import pytest

pytest.importorskip("qdrant_client")

from sulci import AsyncCache, Cache
from sulci.backends.qdrant import QdrantBackend
from sulci.backends.sqlite import SQLiteBackend

from tests._fake_embedder import FakeEmbedder

Q = "how do I reset my password"
A = "Open Settings, choose Security, then Reset password."


# Every embedded Qdrant these tests open is closed after the test: each holds files open (two, measured), and the
# whole suite on macOS's default of 256 open files ran out (oss-0064 kit v2, the owner's laptop, 2026-11-08).
_OPEN: list = []


def _qb(path, cls=QdrantBackend):
    b = cls(db_path=str(path), dimension=FakeEmbedder().dimension)
    _OPEN.append(b)
    return b


@pytest.fixture(autouse=True)
def _close_backends():
    yield
    while _OPEN:
        try:
            _OPEN.pop()._client.close()
        except Exception:
            pass


class ListSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def _cache(tmp_path, **kw):
    backend = _qb(tmp_path / "qdrant")
    return Cache(backend=backend, embedding_model=FakeEmbedder(), telemetry=False, threshold=0.9, **kw)


# ── variant ────────────────────────────────────────────────────────────────────────────────────────────────────────

class TestVariant:
    def test_same_variant_hits(self, tmp_path):
        c = _cache(tmp_path)
        c.set(Q, A, variant="fp-a")
        resp, sim, _ = c.get(Q, variant="fp-a")
        assert resp == A and sim >= 0.99

    def test_other_variant_misses(self, tmp_path):
        c = _cache(tmp_path)
        c.set(Q, A, variant="fp-a")
        assert c.get(Q, variant="fp-b")[0] is None

    def test_lookup_without_variant_never_sees_a_variant_entry(self, tmp_path):
        # The leak the hosted endpoint's review found: a plain /v1/cache/get in the shared scope served an answer
        # written under some other app's system prompt.
        c = _cache(tmp_path)
        c.set(Q, A, variant="fp-a")
        assert c.get(Q)[0] is None

    def test_variant_lookup_never_sees_a_plain_entry(self, tmp_path):
        c = _cache(tmp_path)
        c.set(Q, A)
        assert c.get(Q, variant="fp-a")[0] is None
        assert c.get(Q)[0] == A            # plain entries work exactly as before

    def test_variant_and_scope_both_apply(self, tmp_path):
        c = _cache(tmp_path, personalized=True)
        c.set(Q, A, user_id="acme", variant="fp-a")
        assert c.get(Q, user_id="globex", variant="fp-a")[0] is None
        assert c.get(Q, user_id="acme", variant="fp-a")[0] == A

    def test_metadata_cannot_plant_a_variant(self, tmp_path):
        c = _cache(tmp_path)
        with pytest.raises(ValueError):
            c.set(Q, A, metadata={"sulci_variant": "fp-a"})
        assert c.get(Q, variant="fp-a")[0] is None

    def test_users_own_variant_metadata_key_is_untouched(self, tmp_path):
        c = _cache(tmp_path)
        c.set(Q, A, metadata={"variant": "blue"})
        assert c.get(Q)[0] == A

    @pytest.mark.parametrize("bad, exc", [("", ValueError), ("x" * 257, ValueError), (123, TypeError),
                                          (b"fp", TypeError)])
    def test_bad_variant_refused_before_anything_is_written(self, tmp_path, bad, exc):
        c = _cache(tmp_path)
        with pytest.raises(exc):
            c.set(Q, A, variant=bad)
        with pytest.raises(exc):
            c.get(Q, variant=bad)
        assert c.get(Q)[0] is None

    def test_backend_without_support_refuses_rather_than_ignores(self, tmp_path):
        c = Cache(backend=SQLiteBackend(db_path=str(tmp_path / "s")), embedding_model=FakeEmbedder(),
                  telemetry=False, threshold=0.9)
        with pytest.raises(TypeError, match="SUPPORTS_VARIANT"):
            c.set(Q, A, variant="fp-a")
        with pytest.raises(TypeError, match="SUPPORTS_VARIANT"):
            c.get(Q, variant="fp-a")
        assert c.get(Q)[0] is None          # nothing was stored
        c.set(Q, A)                         # and the backend works as before without it
        assert c.get(Q)[0] == A

    def test_managed_backend_refuses(self, tmp_path):
        c = _cache(tmp_path)
        c._is_remote_transport = True       # the check runs before the transport is touched
        with pytest.raises(TypeError, match="managed"):
            c.get(Q, variant="fp-a")
        with pytest.raises(TypeError, match="managed"):
            c.set(Q, A, variant="fp-a")


# ── accept ─────────────────────────────────────────────────────────────────────────────────────────────────────────

class TestAccept:
    def test_accepted_hit_is_a_hit(self, tmp_path):
        sink = ListSink()
        c = _cache(tmp_path, event_sink=sink)
        c.set(Q, A)
        seen = []
        resp, sim, _ = c.get(Q, accept=lambda r, s: seen.append((r, s)) or True)
        assert resp == A
        assert seen == [(A, pytest.approx(sim))]
        assert sink.events[-1].event_type == "hit"
        assert sink.events[-1].metadata == {}

    def test_refused_hit_is_a_miss_everywhere(self, tmp_path):
        sink = ListSink()
        c = _cache(tmp_path, event_sink=sink)
        c.set(Q, A)
        before = c.stats()
        resp, sim, depth = c.get(Q, accept=lambda r, s: False)
        assert resp is None
        assert sim >= 0.99                  # the caller can still say how close the turned-away entry was
        assert depth == 0
        ev = sink.events[-1]
        assert ev.event_type == "miss"      # billed as a miss
        assert ev.metadata == {"refused": True}
        assert ev.matched_query_hash is None
        after = c.stats()
        assert after["misses"] == before["misses"] + 1
        assert after["hits"] == before["hits"]

    def test_truthy_but_not_true_is_a_refusal(self, tmp_path):
        c = _cache(tmp_path)
        c.set(Q, A)
        assert c.get(Q, accept=lambda r, s: "yes")[0] is None

    def test_exception_is_a_refusal_and_is_logged(self, tmp_path, caplog):
        c = _cache(tmp_path)
        c.set(Q, A)

        def boom(r, s):
            raise RuntimeError("guard broke")

        with caplog.at_level(logging.WARNING):
            assert c.get(Q, accept=boom)[0] is None
        assert "RuntimeError" in caplog.text and A not in caplog.text

    def test_not_called_on_a_miss(self, tmp_path):
        c = _cache(tmp_path)
        calls = []
        assert c.get(Q, accept=lambda r, s: calls.append(1) or True)[0] is None
        assert calls == []

    def test_not_callable_refused(self, tmp_path):
        c = _cache(tmp_path)
        with pytest.raises(TypeError, match="callable"):
            c.get(Q, accept=True)

    def test_managed_backend_refuses(self, tmp_path):
        c = _cache(tmp_path)
        c._is_remote_transport = True
        with pytest.raises(TypeError, match="managed"):
            c.get(Q, accept=lambda r, s: True)

    def test_refused_hit_records_no_turn(self, tmp_path):
        # A hit records a turn (oss-0049), and a session with a turn stores no follow-up. A refused hit was never
        # served, so it must not make the next set in that session look like a follow-up.
        c = _cache(tmp_path, context_window=4)
        c.set(Q, A)
        assert c.get(Q, session_id="s1", accept=lambda r, s: False)[0] is None
        other = "what is the refund policy"
        c.set(other, "Refunds within 30 days.", session_id="s1")
        assert c.get(other)[0] == "Refunds within 30 days."


class TestAsyncPassthrough:
    def test_async_variant_and_accept(self, tmp_path):
        ac = AsyncCache(backend=_qb(tmp_path / "qdrant"),
                        embedding_model=FakeEmbedder(), telemetry=False, threshold=0.9)

        async def run():
            await ac.aset(Q, A, variant="fp-a")
            hit = await ac.aget(Q, variant="fp-a")
            refused = await ac.aget(Q, variant="fp-a", accept=lambda r, s: False)
            plain = await ac.aget(Q)
            return hit, refused, plain

        hit, refused, plain = asyncio.run(run())
        assert hit[0] == A and refused[0] is None and plain[0] is None
        ac.set("q two", "a two", variant="fp-b")
        assert ac.get("q two", variant="fp-b")[0] == "a two"
        assert ac.get("q two", variant="fp-b", accept=lambda r, s: False)[0] is None


# ── review breaks (oss-0064 review, 2026-11-08): each reproduced break is a test ───────────────────────────────────

class _CopiedLookup(QdrantBackend):
    """Like sulci-platform's ClockedQdrantBackend: inherits SUPPORTS_VARIANT, overrides search_match with its own
    copy of the lookup (no variant condition), and swallows unknown keywords."""

    def search_match(self, embedding, threshold, *, tenant_id=None, user_id=None, now=None, **kwargs):
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        f = Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id or "global")),
                         FieldCondition(key="user_id", match=MatchValue(value=user_id or "global"))])
        res = self._client.query_points(collection_name=self.COLLECTION, query=embedding, query_filter=f, limit=5,
                                        with_payload=True)
        for r in res.points:
            if r.score >= threshold:
                return r.payload.get("response"), r.score, r.payload.get("query")
        return None, 0.0, None


class _OldStore(QdrantBackend):
    def store(self, key, query, response, embedding, *, tenant_id=None, user_id=None, expires=None, metadata=None):
        return super().store(key, query, response, embedding, tenant_id=tenant_id, user_id=user_id, expires=expires,
                             metadata=metadata)


class _Declared(_CopiedLookup):
    """A subclass that has made its copy honour the variant, and says so in its own body."""
    SUPPORTS_VARIANT = True

    def search_match(self, embedding, threshold, **kw):
        return QdrantBackend.search_match(self, embedding, threshold, **kw)


class TestReviewBreaks:
    def _c(self, tmp_path, cls):
        return Cache(backend=_qb(tmp_path / "q", cls),
                     embedding_model=FakeEmbedder(), telemetry=False, threshold=0.9)

    def test_subclass_with_copied_lookup_cannot_store_a_variant(self, tmp_path):
        # Review finding 1: the inherited flag was trusted, so set(variant=) wrote an entry that this subclass's
        # plain lookup then served. Now the variant is refused, so no such entry can exist.
        c = self._c(tmp_path, _CopiedLookup)
        with pytest.raises(TypeError, match="SUPPORTS_VARIANT"):
            c.set(Q, "ANSWER-UNDER-APP-A", variant="fp-a")
        assert c.get(Q)[0] is None
        c.set(Q, A)                         # and it works as before without one
        assert c.get(Q)[0] == A

    def test_subclass_with_old_store_refused_before_embedding(self, tmp_path):
        # Review finding 2: failed late, inside the backend, after the embedding.
        c = self._c(tmp_path, _OldStore)
        with pytest.raises(TypeError, match="SUPPORTS_VARIANT"):
            c.set(Q, A, variant="fp-a")

    def test_subclass_that_declares_it_works(self, tmp_path):
        c = self._c(tmp_path, _Declared)
        c.set(Q, A, variant="fp-a")
        assert c.get(Q, variant="fp-a")[0] == A
        assert c.get(Q)[0] is None
        assert c.get(Q, variant="fp-b")[0] is None

    def test_async_accept_refused(self, tmp_path):
        # Review finding 4: a coroutine is never True, so every hit was silently turned away.
        c = _cache(tmp_path)

        async def guard(r, s):
            return True

        class AsyncGuard:
            async def __call__(self, r, s):
                return True

        for g in (guard, AsyncGuard()):
            with pytest.raises(TypeError, match="async"):
                c.get(Q, accept=g)

    def test_managed_backend_message_wins_over_value_checks(self, tmp_path):
        # Review finding 9: variant="" on the managed backend raised ValueError instead of the documented TypeError.
        c = _cache(tmp_path)
        c._is_remote_transport = True
        with pytest.raises(TypeError, match="managed"):
            c.get(Q, variant="")

    @pytest.mark.parametrize("bad", ["", ["fp-a"], 0])
    def test_backend_store_validates_a_direct_caller(self, tmp_path, bad):
        # Review finding 8: store(variant="") called directly bypassed Cache's check; Qdrant's empty-check then hid
        # it from plain lookups, and a list matched any element.
        b = _qb(tmp_path / "q")
        with pytest.raises(ValueError):
            b.store("k", Q, A, FakeEmbedder().embed(Q), variant=bad)
