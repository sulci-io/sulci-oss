# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""oss-0049: on a self-hosted backend a follow-up is recorded but NOT stored — the managed gateway's rule since 0.10.2.

Found 2026-10-31 on sulci.io/demo, which mirrors ``Cache.set``: "How do I speed it up?" asked twice in a PostgreSQL
conversation missed the second time (its own turn diluted the topic), and the PostgreSQL answer was stored under the
bare follow-up words. Every later conversation that asked them — Python included — then hit that entry. Measured on
all-MiniLM-L6-v2 with the demo's blend (α=0.70, decay=0.50): the stored follow-up scored 0.961, the right answer 0.767.

A follow-up = a set whose ``session_id`` window already holds a turn. ``standalone=True`` stores it anyway (a
self-contained rewrite, or an agent prompt that carries its own state), as on managed. The first question of a
session, a set without ``session_id``, and a Cache with no context window store as before.

The embedder is tests/_fake_embedder.py: these prove the store/no-store plumbing, not retrieval quality.
"""

import asyncio
import logging
import sqlite3

import pytest

from sulci import AsyncCache, Cache
from sulci.sessions import InMemorySessionStore

PY = "How can I make my Python script run faster?"
PG = "How do I speed up slow PostgreSQL queries?"
FU = "How do I speed it up?"


def _cache(tmp_path, fake_embedder, **kw):
    kw.setdefault("context_window", 4)
    return Cache(backend="sqlite", db_path=str(tmp_path / "db"), embedding_model=fake_embedder,
                 telemetry=False, **kw)


def _stored(c):
    """The queries actually written to the SQLite table, oldest first."""
    con = sqlite3.connect(c._backend._db_file)
    try:
        return [r[0] for r in con.execute("SELECT query FROM cache ORDER BY rowid")]
    finally:
        con.close()


def _depth(c, session_id):
    return c.get_context(session_id).depth


def _user_turns(c, session_id):
    return len([t for t in c.get_context(session_id).turns if t.role == "user"])


# ── the rule ─────────────────────────────────────────────────────────────────

def test_first_question_of_a_session_is_stored_and_the_follow_up_is_not(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder)
    c.set(PG, "postgres answer", session_id="s1")
    c.set(FU, "postgres answer to the follow-up", session_id="s1")
    assert _stored(c) == [PG]


def test_the_follow_up_turn_is_still_recorded(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder)
    c.set(PG, "a", session_id="s1")
    before = _depth(c, "s1")
    c.set(FU, "b", session_id="s1")
    assert _depth(c, "s1") > before          # the conversation moved on; only the entry was skipped


def test_standalone_stores_a_follow_up(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder)
    c.set(PG, "a", session_id="s1")
    c.set("How do I add an index to a PostgreSQL table?", "b", session_id="s1", standalone=True)
    assert _stored(c) == [PG, "How do I add an index to a PostgreSQL table?"]


def test_without_a_session_every_set_is_stored(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder)
    c.set(PG, "a")
    c.set(FU, "b")
    assert _stored(c) == [PG, FU]


def test_a_cache_without_a_context_window_stores_every_set(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder, context_window=0)
    c.set(PG, "a", session_id="s1")
    c.set(FU, "b", session_id="s1")
    assert _stored(c) == [PG, FU]


def test_each_session_stores_its_own_first_question(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder)
    c.set(PY, "py", session_id="s1")
    c.set(PG, "pg", session_id="s2")
    c.set(FU, "fu", session_id="s2")
    assert _stored(c) == [PY, PG]


def test_after_clear_context_the_next_set_is_a_first_question(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder)
    c.set(PG, "a", session_id="s1")
    c.clear_context("s1")
    c.set(PY, "b", session_id="s1")
    assert _stored(c) == [PG, PY]


def test_a_pre_seeded_window_makes_the_first_set_a_follow_up(tmp_path, fake_embedder):
    # get_context() lets a caller inject history; a question asked against it is context-dependent too.
    c = _cache(tmp_path, fake_embedder)
    c.get_context("s1").add_turn("We are tuning a PostgreSQL database", role="user")
    c.set(FU, "a", session_id="s1")
    assert _stored(c) == []


def test_session_scope_is_per_tenant(tmp_path, fake_embedder):
    # A turn in tenant A's "s1" is not a turn in tenant B's "s1" (0.9.3 session scoping).
    c = _cache(tmp_path, fake_embedder)
    c.set(PG, "a", session_id="s1", tenant_id="acme")
    c.set(PY, "b", session_id="s1", tenant_id="globex")
    assert _stored(c) == [PG, PY]


# ── cached_call ─────────────────────────────────────────────────────────────

def test_cached_call_stores_the_first_miss_and_not_the_follow_up_miss(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder, threshold=0.999, context_threshold=0.999)        # every call below misses
    r1 = c.cached_call(PG, lambda q: "pg", session_id="s1")
    r2 = c.cached_call(FU, lambda q: "pg follow-up", session_id="s1")
    assert (r1["source"], r2["source"]) == ("llm", "llm")
    assert r2["response"] == "pg follow-up"                      # the caller still gets the answer
    assert _stored(c) == [PG]


def test_cached_call_standalone_stores_the_follow_up(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder, threshold=0.999, context_threshold=0.999)
    c.cached_call(PG, lambda q: "pg", session_id="s1")
    c.cached_call(FU, lambda q: "pg follow-up", session_id="s1", standalone=True)
    assert _stored(c) == [PG, FU]


def test_the_demo_sequence_no_longer_serves_one_conversation_s_answer_in_another(tmp_path, fake_embedder):
    """The 10-31 demo run, as library calls: PostgreSQL, follow-up twice; then Python, follow-up. Whatever the
    embedder scores, the bare follow-up words can no longer carry the PostgreSQL answer into the Python session,
    because no entry is ever stored under them."""
    c = _cache(tmp_path, fake_embedder, threshold=0.999, context_threshold=0.999)
    c.cached_call(PG, lambda q: "POSTGRES", session_id="pg")
    c.cached_call(FU, lambda q: "POSTGRES follow-up", session_id="pg")
    c.cached_call(FU, lambda q: "POSTGRES follow-up again", session_id="pg")
    c.cached_call(PY, lambda q: "PYTHON", session_id="py")
    assert _stored(c) == [PG, PY]
    assert c.get(FU)[0] is None                                    # nothing answers the bare words


def test_a_hit_counts_as_the_session_s_turn(tmp_path, fake_embedder):
    # cached_call records a turn on a hit, so a miss after it is a follow-up.
    c = _cache(tmp_path, fake_embedder)
    c.set(PG, "pg")                                               # stored, no session
    assert c.cached_call(PG, lambda q: "unused", session_id="s1")["source"] == "cache"
    c.cached_call("Something unrelated entirely 42", lambda q: "x", session_id="s1", threshold=0.999, context_threshold=0.999)
    assert _stored(c) == [PG]


# ── injected session stores ─────────────────────────────────────────────────

def test_an_injected_session_store_applies_the_same_rule(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder, session_store=InMemorySessionStore())
    c.set(PG, "a", session_id="s1")
    c.set(FU, "b", session_id="s1")
    c.set(FU, "c", session_id="s1", standalone=True)
    assert _stored(c) == [PG, FU]


class _UnreadableStore(InMemorySessionStore):
    def get(self, *a, **k):
        raise ConnectionError("session store down")


def test_an_unreadable_session_store_stores_nothing(tmp_path, fake_embedder, caplog):
    # The gateway's "session_unavailable": a set that cannot tell whether it is a follow-up stores nothing.
    c = _cache(tmp_path, fake_embedder, session_store=_UnreadableStore())
    with caplog.at_level(logging.WARNING, logger="sulci"):
        c.set(PG, "a", session_id="s1")
    assert _stored(c) == []
    assert "stores nothing and records no turn" in caplog.text
    c.set(PG, "a")                                                # no session: stored, nothing to read
    assert _stored(c) == [PG]


# ── get() + set(): the integrations' pattern (proxy, LiteLLM, LlamaIndex, MCP, the README's FastAPI) ─────────

def test_a_hit_in_get_is_a_turn_so_the_next_set_is_a_follow_up(tmp_path, fake_embedder):
    """The independent review's repro: the first question HITS via get(); before, get() recorded no turn, the
    session stayed empty, and the follow-up's set looked like a first question and was stored."""
    c = _cache(tmp_path, fake_embedder)
    c.set(PG, "pg")                                               # warm, no session
    assert c.get(PG, session_id="s1")[0] == "pg"                  # hit
    assert _user_turns(c, "s1") == 1                              # ...recorded, as the gateway's /cache/get does
    c.set(FU, "pg follow-up", session_id="s1")
    assert _stored(c) == [PG]


def test_a_miss_in_get_records_nothing(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder)
    assert c.get(PG, session_id="s1")[0] is None
    assert _depth(c, "s1") == 0                                   # the set after the LLM call records it
    c.set(PG, "pg", session_id="s1")
    assert _stored(c) == [PG] and _depth(c, "s1") > 0


def test_cached_call_records_a_hit_once(tmp_path, fake_embedder):
    # The hit's turn moved from cached_call into get(); it must not be recorded twice.
    c = _cache(tmp_path, fake_embedder, context_window=6)
    c.set(PG, "pg")
    c.cached_call(PG, lambda q: "unused", session_id="s1")
    assert _user_turns(c, "s1") == 1


# ── what the caller is told ─────────────────────────────────────────────────

def test_a_failed_turn_write_on_a_follow_up_says_no_entry_was_stored(tmp_path, fake_embedder):
    from sulci.sessions import SulciSessionError

    class _AppendFailsAfterOne(InMemorySessionStore):
        n = 0
        def append(self, *a, **k):
            self.n += 1
            if self.n > 1:
                raise ConnectionError("down")
            return super().append(*a, **k)

    c = _cache(tmp_path, fake_embedder, session_store=_AppendFailsAfterOne())
    c.set(PG, "a", session_id="s1")
    with pytest.raises(SulciSessionError, match="no entry was stored") as info:
        c.set(FU, "b", session_id="s1")
    assert info.value.stored is False
    assert _stored(c) == [PG]


def test_the_mcp_tool_reports_a_follow_up_as_not_stored(tmp_path, fake_embedder):
    pytest.importorskip("mcp")
    import json
    from sulci.integrations.mcp_server import build_server
    server = build_server(_cache(tmp_path, fake_embedder))

    def call(**args):   # as tests/test_integrations_mcp.py: mcp 2.0.0's CallToolResult.content
        result = asyncio.run(server.call_tool("cache_store", args))
        return json.loads("".join(getattr(b, "text", "") for b in getattr(result, "content", result)))

    assert call(query=PG, response="a", session_id="s1") == {"stored": True}
    assert call(query=FU, response="b", session_id="s1") == {"stored": False}


def _mcp_call(server, name, **args):
    import json
    result = asyncio.run(server.call_tool(name, args))
    return json.loads("".join(getattr(b, "text", "") for b in getattr(result, "content", result)))


def test_the_mcp_lookup_tool_stays_read_only(tmp_path, fake_embedder):
    # cache_lookup is annotated readOnlyHint=True ("Does not modify anything"): a hit records no turn there.
    pytest.importorskip("mcp")
    from sulci.integrations.mcp_server import build_server
    c = _cache(tmp_path, fake_embedder)
    c.set(PG, "pg")
    out = _mcp_call(build_server(c), "cache_lookup", query=PG, session_id="s1")
    assert out["cache_hit"] is True
    assert _user_turns(c, "s1") == 0


def test_the_mcp_store_tool_runs_a_subclass_s_set(tmp_path, fake_embedder):
    pytest.importorskip("mcp")
    from sulci.integrations.mcp_server import build_server
    calls = []

    class Audited(Cache):
        def set(self, query, response, **kw):
            calls.append(query)
            return super().set(query, response, **kw)

    c = Audited(backend="sqlite", db_path=str(tmp_path / "db"), embedding_model=fake_embedder, context_window=4,
                telemetry=False)
    assert _mcp_call(build_server(c), "cache_store", query=PG, response="a", session_id="s1") == {"stored": True}
    assert calls == [PG]


def test_the_mcp_store_tool_over_an_async_cache_reports_a_follow_up(tmp_path, fake_embedder):
    pytest.importorskip("mcp")
    from sulci.integrations.mcp_server import build_server
    a = AsyncCache(backend="sqlite", db_path=str(tmp_path / "db"), embedding_model=fake_embedder, context_window=4,
                   telemetry=False)
    server = build_server(a)
    assert _mcp_call(server, "cache_store", query=PG, response="a", session_id="s1") == {"stored": True}
    assert _mcp_call(server, "cache_store", query=FU, response="b", session_id="s1") == {"stored": False}
    assert _stored(a._cache) == [PG]


def test_personalized_sessions_are_per_user(tmp_path, fake_embedder):
    c = _cache(tmp_path, fake_embedder, personalized=True)
    c.set(PG, "a", session_id="s1", user_id="u1")
    c.set(PY, "b", session_id="s1", user_id="u2")                # u2's s1 is a different conversation
    c.set(FU, "c", session_id="s1", user_id="u1")                # u1's follow-up
    assert _stored(c) == [PG, PY]


# ── AsyncCache ───────────────────────────────────────────────────────────────

def test_async_cache_mirrors_the_rule(tmp_path, fake_embedder):
    a = AsyncCache(backend="sqlite", db_path=str(tmp_path / "db"), embedding_model=fake_embedder,
                   context_window=4, telemetry=False)
    asyncio.run(a.aset(PG, "a", session_id="s1"))
    asyncio.run(a.aset(FU, "b", session_id="s1"))
    asyncio.run(a.aset(FU, "c", session_id="s1", standalone=True))
    assert _stored(a._cache) == [PG, FU]


@pytest.mark.parametrize("kw", [{}, {"standalone": False}])
def test_standalone_false_is_the_default(tmp_path, fake_embedder, kw):
    c = _cache(tmp_path, fake_embedder)
    c.set(PG, "a", session_id="s1")
    c.set(FU, "b", session_id="s1", **kw)
    assert _stored(c) == [PG]
