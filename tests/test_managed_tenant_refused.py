"""1.0.1 (G33, 2026-10-21): tenant_id on the managed backend is refused, not silently dropped.

The SDK never sent tenant_id to the gateway — the API key is the managed backend's only partition — so before 1.0.1 a
caller's tenant scoping was ignored and one cache was shared across tenants. delete_user() refused it since 0.10.0;
get(), set() and cached_call() now do too, before anything is sent, and the integrations adjust.
"""
import asyncio
import json
import warnings

import pytest

from sulci import AsyncCache, Cache

KEY = "sk-sulci-" + "t" * 32


class _Resp:
    status_code = 200

    def __init__(self, body=None):
        self._body = body or {"response": None, "similarity": 0.0, "context_depth": 0}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _managed(**kw):
    return Cache(backend="sulci", api_key=KEY, telemetry=False, **kw)


def _capture(cache):
    sent = []
    cache._backend._client.post = lambda path, json=None, **k: (sent.append((path, json)), _Resp())[1]
    return sent


# ── core ─────────────────────────────────────────────────────────────────────
def test_get_refuses_a_tenant_id_and_sends_nothing():
    cache = _managed()
    sent = _capture(cache)
    with pytest.raises(TypeError, match="managed backend"):
        cache.get("how do refunds work?", tenant_id="acme")
    assert sent == []


def test_set_refuses_a_tenant_id_and_sends_nothing():
    cache = _managed()
    sent = _capture(cache)
    with pytest.raises(TypeError, match="managed backend"):
        cache.set("how do refunds work?", "within 5 days", tenant_id="acme")
    assert sent == []


def test_cached_call_refuses_before_calling_the_llm():
    cache = _managed()
    sent = _capture(cache)
    called = []
    with pytest.raises(TypeError, match="managed backend"):
        cache.cached_call("how do refunds work?", lambda q: called.append(q) or "x", tenant_id="acme")
    assert called == [] and sent == []


def test_async_cache_refuses_too():
    cache = AsyncCache(backend="sulci", api_key=KEY, telemetry=False)
    sent = _capture(cache._cache)
    with pytest.raises(TypeError, match="managed backend"):
        asyncio.run(cache.aget("how do refunds work?", tenant_id="acme"))
    assert sent == []


def test_an_explicit_none_is_no_tenant():
    cache = _managed()
    sent = _capture(cache)
    cache.get("how do refunds work?", tenant_id=None)
    cache.set("how do refunds work?", "within 5 days", tenant_id=None)
    assert [p for p, _ in sent] == ["/v1/cache/get", "/v1/cache/set"]


def test_async_set_and_cached_call_refuse_too():
    cache = AsyncCache(backend="sulci", api_key=KEY, telemetry=False)
    sent = _capture(cache._cache)
    with pytest.raises(TypeError, match="managed backend"):
        asyncio.run(cache.aset("q", "a", tenant_id="acme"))
    with pytest.raises(TypeError, match="managed backend"):
        asyncio.run(cache.acached_call("q", lambda q: "a", tenant_id="acme"))
    assert sent == []


def test_no_tenant_id_is_unchanged():
    cache = _managed()
    sent = _capture(cache)
    assert cache.get("how do refunds work?")[0] is None
    cache.set("how do refunds work?", "within 5 days")
    assert [p for p, _ in sent] == ["/v1/cache/get", "/v1/cache/set"]
    assert all("tenant_id" not in (b or {}) for _, b in sent)


def test_self_hosted_backends_still_take_a_tenant_id(tmp_path, fake_embedder):
    cache = Cache(backend="sqlite", db_path=str(tmp_path / "t"), embedding_model=fake_embedder, telemetry=False)
    cache.set("how do refunds work?", "within 5 days", tenant_id="acme")
    assert cache.get("how do refunds work?", tenant_id="acme")[0] == "within 5 days"
    assert cache.get("how do refunds work?", tenant_id="other")[0] is None


# ── LiteLLM ──────────────────────────────────────────────────────────────────
def test_litellm_namespace_by_model_warns_and_shares_on_managed(caplog):
    pytest.importorskip("litellm")
    from sulci.integrations.litellm import SulciLiteLLMCache
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # a log warning, not warnings.warn: -W error must not raise
        a = SulciLiteLLMCache(_managed(), namespace_by_model=True)
    assert "namespace_by_model=True cannot isolate models" in caplog.text
    assert a.namespace_by_model is False
    assert a._tenant(model="gpt-4o") is None


def test_litellm_namespace_false_on_managed_is_silent():
    pytest.importorskip("litellm")
    from sulci.integrations.litellm import SulciLiteLLMCache
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SulciLiteLLMCache(_managed(), namespace_by_model=False)


# ── proxy ────────────────────────────────────────────────────────────────────
def _proxy(cache, **kw):
    pytest.importorskip("fastapi")
    import httpx
    from fastapi.testclient import TestClient

    from sulci.proxy.app import build_app
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "model": "gpt-4o",
                                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                      "finish_reason": "stop"}]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TestClient(build_app(cache, client=client, **kw)), calls


BODY = {"model": "gpt-4o", "messages": [{"role": "user", "content": "how do refunds work?"}]}


def test_proxy_refuses_a_tenant_header_on_managed_and_forwards_nothing():
    cache = _managed()
    sent = _capture(cache)
    c, upstream = _proxy(cache, share_across_models=True)
    r = c.post("/v1/chat/completions", json=BODY, headers={"x-sulci-tenant-id": "acme"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "sulci_tenant_unsupported"
    assert upstream == [] and sent == []


ANTHROPIC = {"model": "claude-sonnet-4-5", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}


def test_proxy_refuses_on_messages_with_anthropic_shape():
    cache = _managed()
    c, upstream = _proxy(cache, share_across_models=True)
    r = c.post("/v1/messages", json=ANTHROPIC, headers={"x-sulci-tenant-id": "acme"})
    assert r.status_code == 400
    assert r.json()["type"] == "error" and r.json()["error"]["type"] == "invalid_request_error"
    assert upstream == []


def test_proxy_refuses_a_streaming_request_with_a_tenant_too():
    cache = _managed()
    c, upstream = _proxy(cache, share_across_models=True)
    r = c.post("/v1/chat/completions", json=dict(BODY, stream=True), headers={"x-sulci-tenant-id": "acme"})
    assert r.status_code == 400 and upstream == []


def test_proxy_an_empty_tenant_header_is_no_tenant():
    cache = _managed()
    sent = _capture(cache)
    c, _upstream = _proxy(cache, share_across_models=True)
    r = c.post("/v1/chat/completions", json=BODY, headers={"x-sulci-tenant-id": ""})
    assert r.status_code == 200 and [p for p, _ in sent][:1] == ["/v1/cache/get"]


def test_proxy_on_managed_shares_across_models_instead_of_failing_every_lookup(caplog):
    cache = _managed()
    sent = _capture(cache)
    c, _upstream = _proxy(cache, share_across_models=False)
    assert "cannot scope by model" in caplog.text
    r = c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200
    assert [p for p, _ in sent][:1] == ["/v1/cache/get"], "the lookup reached the gateway (no TypeError)"


# ── MCP ──────────────────────────────────────────────────────────────────────
def test_mcp_default_tenant_on_managed_refuses_at_build():
    pytest.importorskip("mcp")
    from sulci.integrations.mcp_server import build_server
    with pytest.raises(ValueError, match="managed backend"):
        build_server(_managed(), default_tenant_id="acme")


def test_mcp_store_with_a_tenant_on_managed_returns_the_error():
    pytest.importorskip("mcp")
    from sulci.integrations.mcp_server import build_server
    cache = _managed()
    sent = _capture(cache)
    server = build_server(cache)
    result = asyncio.run(server.call_tool("cache_store", {"query": "q", "response": "a", "tenant_id": "acme"}))
    text = "".join(getattr(b, "text", "") for b in getattr(result, "content", result))
    assert "managed backend" in text
    assert sent == []


def test_mcp_instructions_on_managed_do_not_ask_for_a_tenant():
    pytest.importorskip("mcp")
    from sulci.integrations.mcp_server import _DEFAULT_INSTRUCTIONS, build_server
    managed = build_server(_managed())
    assert "Do not pass `tenant_id`" in managed.instructions
    assert "Pass a `tenant_id`" in _DEFAULT_INSTRUCTIONS


def test_mcp_tool_tenant_on_managed_returns_the_error():
    pytest.importorskip("mcp")
    from sulci.integrations.mcp_server import build_server
    cache = _managed()
    sent = _capture(cache)
    server = build_server(cache)
    result = asyncio.run(server.call_tool("cache_lookup", {"query": "how do refunds work?", "tenant_id": "acme"}))
    out = json.loads("".join(getattr(b, "text", "") for b in getattr(result, "content", result)))
    assert out["cache_hit"] is False and "managed backend" in out["error"]
    assert sent == []
