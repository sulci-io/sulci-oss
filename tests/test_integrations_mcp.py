# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
tests/test_integrations_mcp.py
───────────────────────────────
Covers sulci.integrations.mcp_server.

Runs offline: the cache is built with tests/_fake_embedder.FakeEmbedder, so
no model weights are fetched. See that module for why that matters.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from sulci import Cache

mcp_server = pytest.importorskip(
    "sulci.integrations.mcp_server",
    reason='mcp>=2.0.0 not installed; pip install "sulci[mcp]"',
)
build_server = mcp_server.build_server


# ── fixtures ─────────────────────────────────────────────────────────────
@pytest.fixture
def cache(tmp_path, fake_embedder):
    return Cache(
        backend="sqlite",
        db_path=str(tmp_path / "mcp"),
        embedding_model=fake_embedder,
        threshold=0.85,
        telemetry=False,
    )


@pytest.fixture
def server(cache):
    return build_server(cache)


def _tools(server):
    return {t.name: t for t in asyncio.run(server.list_tools())}


def _call(server, name, **args):
    # mcp 2.0.0: call_tool returns a CallToolResult whose .content is a list
    # of TextContent. Measured 2026-08-11 — earlier drafts assumed a bare
    # list and a (blocks, structured) tuple; neither is what ships.
    result = asyncio.run(server.call_tool(name, args))
    blocks = getattr(result, "content", result)
    text = "".join(getattr(b, "text", "") for b in blocks)
    return json.loads(text)


QUERY = "What is semantic caching?"
RESP = "Semantic caching stores LLM responses indexed by meaning."


# ── tool registration ────────────────────────────────────────────────────
def test_registers_three_tools(server):
    assert set(_tools(server)) == {"cache_lookup", "cache_store", "cache_stats"}


def test_read_only_mode_omits_the_write_tool(cache):
    names = set(_tools(build_server(cache, read_only=True)))
    assert names == {"cache_lookup", "cache_stats"}
    assert "cache_store" not in names


def test_read_only_honours_env(cache, monkeypatch):
    monkeypatch.setenv("SULCI_MCP_READ_ONLY", "1")
    assert "cache_store" not in _tools(build_server(cache))


def test_lookup_and_stats_are_annotated_read_only(server):
    tools = _tools(server)
    assert tools["cache_lookup"].annotations.read_only_hint is True
    assert tools["cache_stats"].annotations.read_only_hint is True


def test_store_is_annotated_as_a_write(server):
    # gh-aw guidance is that custom MCP servers should be read-only. This
    # tool is a write and must say so, so a reviewer can make that call.
    assert _tools(server)["cache_store"].annotations.read_only_hint is False


def test_server_advertises_the_sulci_version(cache):
    import sulci

    assert build_server(cache).version == sulci.__version__


def test_instructions_warn_about_state_dependent_answers(cache):
    text = build_server(cache).instructions.lower()
    assert "commit sha" in text or "tree hash" in text


# ── behaviour ────────────────────────────────────────────────────────────
def test_miss_returns_cache_hit_false(server):
    out = _call(server, "cache_lookup", query=QUERY)
    assert out["cache_hit"] is False
    assert out["response"] is None


def test_store_then_lookup_hits(server):
    assert _call(server, "cache_store", query=QUERY, response=RESP)["stored"] is True
    out = _call(server, "cache_lookup", query=QUERY)
    assert out["cache_hit"] is True
    assert out["response"] == RESP


def test_lookup_reports_similarity_and_depth(server):
    _call(server, "cache_store", query=QUERY, response=RESP)
    out = _call(server, "cache_lookup", query=QUERY)
    # Cache.get returns a 3-tuple; a scalar unpack would have thrown here.
    assert "similarity" in out and "context_depth" in out
    assert out["similarity"] >= 0.85


def test_unrelated_query_does_not_hit(server):
    _call(server, "cache_store", query=QUERY, response=RESP)
    out = _call(server, "cache_lookup", query="zebra migration patterns")
    assert out["cache_hit"] is False


def test_tenant_id_is_accepted(server):
    # NOTE the assertion. On sqlite, tenant_id is accepted and IGNORED —
    # only qdrant sets ENFORCES_TENANT_ISOLATION = True. This test asserts
    # the behaviour the code actually has. If a future sqlite backend starts
    # enforcing, this test SHOULD fail and be rewritten; that is the point.
    _call(server, "cache_store", query=QUERY, response=RESP, tenant_id="repo-a")
    assert _call(server, "cache_lookup", query=QUERY, tenant_id="repo-a")["cache_hit"] is True
    assert _call(server, "cache_lookup", query=QUERY, tenant_id="repo-b")["cache_hit"] is True


def test_unenforced_scope_warns_loudly(cache):
    from sulci.integrations._scope import ScopeNotEnforcedWarning

    with pytest.warns(ScopeNotEnforcedWarning, match="ACCEPTED AND IGNORED"):
        build_server(cache, default_tenant_id="owner/repo")


def test_no_warning_when_no_scope_is_requested(cache, recwarn):
    from sulci.integrations._scope import ScopeNotEnforcedWarning

    build_server(cache)
    assert not [w for w in recwarn if w.category is ScopeNotEnforcedWarning]


def test_stats_round_trips(server):
    _call(server, "cache_store", query=QUERY, response=RESP)
    _call(server, "cache_lookup", query=QUERY)
    stats = _call(server, "cache_stats")
    assert stats["hits"] == 1
    assert "hit_rate" in stats and "saved_cost" in stats


def test_threshold_argument_is_honoured(server):
    _call(server, "cache_store", query=QUERY, response=RESP)
    strict = _call(server, "cache_lookup", query="What is caching?", threshold=0.999)
    assert strict["cache_hit"] is False


# ── failure isolation ────────────────────────────────────────────────────
class _Exploding:
    def get(self, *a, **k):
        raise RuntimeError("backend down")

    def set(self, *a, **k):
        raise RuntimeError("backend down")

    def stats(self):
        raise RuntimeError("backend down")


def test_lookup_failure_is_reported_not_raised():
    s = build_server(_Exploding())
    out = _call(s, "cache_lookup", query=QUERY)
    assert out["cache_hit"] is False and "backend down" in out["error"]


def test_store_failure_is_reported_not_raised():
    s = build_server(_Exploding())
    out = _call(s, "cache_store", query=QUERY, response=RESP)
    assert out["stored"] is False and "error" in out


def test_stats_failure_is_reported_not_raised():
    assert "error" in _call(build_server(_Exploding()), "cache_stats")


# ── constructor guards ───────────────────────────────────────────────────
def test_cache_and_kwargs_are_mutually_exclusive(cache):
    with pytest.raises(TypeError):
        build_server(cache, backend="sqlite")


# ── CLI ──────────────────────────────────────────────────────────────────
def test_cli_defaults_to_sqlite_not_chroma():
    # sulci.Cache defaults to chroma, which needs a server. Inside a CI
    # runner that is the wrong default; the CLI must override it.
    args = mcp_server._build_arg_parser().parse_args([])
    assert args.backend == "sqlite"
    assert args.transport == "stdio"


def test_cli_accepts_the_documented_transports():
    for t in ("stdio", "sse", "streamable-http"):
        assert mcp_server._build_arg_parser().parse_args(["--transport", t]).transport == t
