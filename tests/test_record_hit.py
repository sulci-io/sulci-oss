# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_record_hit.py — oss-0065: ``Cache.get(..., record_hit=False)``, public.

Since 1.1.1 (oss-0049) a hit in a session is recorded as that conversation's newest turn. A caller that records turns
itself needs to look up without that: sulci-platform's gateway keeps up to 20 turns per session, while the session
adapter trims to the looking-up Cache's ``context_window`` whenever it appends, so the library's record of a hit cut
the gateway's history to the lookup's window (and recorded the turn a second time). ``record_hit`` existed privately
(``_get``, for the MCP tool); oss-0065 makes it part of ``get`` and ``AsyncCache``.

Runs offline: embedded Qdrant (closed after each test) and the deterministic FakeEmbedder.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("qdrant_client")

from sulci import AsyncCache, Cache
from sulci.backends.qdrant import QdrantBackend
from sulci.sessions import InMemorySessionStore

from tests._fake_embedder import FakeEmbedder

Q = "how do I reset my password"
A = "Open Settings, choose Security, then Reset password."

_OPEN: list = []


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


def _cache(tmp_path, *, window=2, **kw):
    b = QdrantBackend(db_path=str(tmp_path / "qdrant"), dimension=FakeEmbedder().dimension)
    _OPEN.append(b)
    store = InMemorySessionStore()
    c = Cache(backend=b, embedding_model=FakeEmbedder(), telemetry=False, threshold=0.9, context_window=window,
              session_store=store, **kw)
    c.set(Q, A)                         # stored with no session: a first question
    return c, store


def _turns(c, store, session="s1"):
    return len(store.get(c._session_key(session, None, None)))


class TestRecordHit:
    def test_by_default_a_hit_is_a_turn(self, tmp_path):
        c, store = _cache(tmp_path)
        resp, _, _ = c.get(Q, session_id="s1")
        assert resp == A and _turns(c, store) == 1

    def test_false_records_nothing_and_still_hits(self, tmp_path):
        sink = ListSink()
        c, store = _cache(tmp_path, event_sink=sink)
        sink.events.clear()
        resp, sim, _ = c.get(Q, session_id="s1", record_hit=False)
        assert resp == A and sim >= 0.9
        assert _turns(c, store) == 0
        assert [e.event_type for e in sink.events] == ["hit"], "the hit is still emitted"
        assert c.stats()["hits"] == 1, "and still counted"

    def test_the_reason_the_gateway_needs_it_the_adapter_trims_to_the_window(self, tmp_path):
        """A session with 5 turns kept by a caller that keeps more than this Cache's window (2)."""
        c, store = _cache(tmp_path, window=2)
        sid = c._session_key("s1", None, None)
        for i in range(5):
            store.append(sid, FakeEmbedder().embed(f"turn {i}"), max_turns=20)
        # context_threshold: the 5 turns blend into the lookup (and a blended lookup without one warns)
        assert c.get(Q, session_id="s1", record_hit=False, context_threshold=0.5)[0] == A
        assert _turns(c, store) == 5, "record_hit=False leaves the caller's history alone"
        assert c.get(Q, session_id="s1", context_threshold=0.5)[0] == A
        assert _turns(c, store) == 2, "the default records the hit and trims to the lookup's window"

    def test_a_miss_records_nothing_either_way(self, tmp_path):
        c, store = _cache(tmp_path)
        for rh in (True, False):
            resp, _, _ = c.get("what are your opening hours", session_id="s1", record_hit=rh)
            assert resp is None
        assert _turns(c, store) == 0

    @pytest.mark.parametrize("bad", [0, 1, None, "no", 0.0])
    def test_only_a_bool(self, tmp_path, bad):
        c, store = _cache(tmp_path)
        with pytest.raises(TypeError, match="record_hit"):
            c.get(Q, session_id="s1", record_hit=bad)
        assert _turns(c, store) == 0

    def test_with_variant_and_accept(self, tmp_path):
        c, store = _cache(tmp_path)
        c.set(Q, A, variant="fp1")
        resp, _, _ = c.get(Q, session_id="s1", variant="fp1", accept=lambda r, s: True, record_hit=False)
        assert resp == A and _turns(c, store) == 0


class TestAsyncCache:
    def test_aget_forwards_it(self, tmp_path):
        c, store = _cache(tmp_path)
        ac = AsyncCache.__new__(AsyncCache)
        ac._cache = c
        resp, _, _ = asyncio.run(ac.aget(Q, session_id="s1", record_hit=False))
        assert resp == A and _turns(c, store) == 0
        asyncio.run(ac.aget(Q, session_id="s1"))
        assert _turns(c, store) == 1

    def test_the_sync_passthrough_forwards_it(self, tmp_path):
        c, store = _cache(tmp_path)
        ac = AsyncCache.__new__(AsyncCache)
        ac._cache = c
        assert ac.get(Q, session_id="s1", record_hit=False)[0] == A
        assert _turns(c, store) == 0


class TestReviewFindings:
    """oss-0065's independent review."""

    def test_a_refused_hit_records_nothing_even_by_default(self, tmp_path):
        c, store = _cache(tmp_path)
        resp, _, _ = c.get(Q, session_id="s1", accept=lambda r, s: False)
        assert resp is None and _turns(c, store) == 0

    def test_false_is_refused_on_the_managed_backend(self):
        c = Cache(backend="sulci", api_key="sk-sulci-" + "0" * 32, telemetry=False)
        with pytest.raises(TypeError, match="managed backend"):
            c.get(Q, session_id="s1", record_hit=False)

    def test_numpy_bools_get_a_clear_message(self, tmp_path):
        np = pytest.importorskip("numpy")
        c, _ = _cache(tmp_path)
        with pytest.raises(TypeError, match="numpy"):
            c.get(Q, record_hit=np.bool_(False))

    def test_async_cache_does_not_send_the_default(self, tmp_path):
        """A wrapped Cache whose get predates record_hit keeps working through AsyncCache."""
        seen = []

        class Old:
            def get(self, query, threshold=None, tenant_id=None, user_id=None, session_id=None, plan=None,
                    context_threshold=None, variant=None, accept=None):
                seen.append(query)
                return None, 0.0, 0
        ac = AsyncCache.__new__(AsyncCache)
        ac._cache = Old()
        assert asyncio.run(ac.aget(Q)) == (None, 0.0, 0) and ac.get(Q) == (None, 0.0, 0)
        assert seen == [Q, Q]

    def test_the_mcp_lookup_over_an_async_cache_records_no_turn(self, tmp_path):
        pytest.importorskip("mcp")
        import json as _json
        from sulci.integrations.mcp_server import build_server
        c, store = _cache(tmp_path)
        ac = AsyncCache.__new__(AsyncCache)
        ac._cache = c
        server = build_server(ac)
        out = asyncio.run(server.call_tool("cache_lookup", {"query": Q, "session_id": "s1"}))
        text = "".join(getattr(b, "text", "") for b in getattr(out, "content", out))
        assert _json.loads(text)["cache_hit"] is True
        assert _turns(c, store) == 0
