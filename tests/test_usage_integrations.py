# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""oss-0053 (sulci 1.3.0): sulci-proxy and SulciLiteLLMCache send token counts on the managed backend.

The managed backend's HTTP client is replaced by a recorder, so each test reads the exact /v1/cache/set body. Local
backends are checked to receive no counts (decision 3), streams and tool calls none (decisions 1 and 2), and every
reply is stored whether or not its counts could be read.
"""
from __future__ import annotations

import json
import logging

import pytest

from sulci import Cache

KEY = "sk-sulci-" + "t" * 32


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _managed():
    cache = Cache(backend="sulci", api_key=KEY, telemetry=False)
    sent = []

    def post(path, json=None, **k):
        sent.append((path, json))
        return _Resp({"response": None, "similarity": 0.0, "context_depth": 0} if path.endswith("/get")
                     else {"status": "ok", "stored": True})
    cache._backend._client.post = post
    return cache, sent


def _sets(sent):
    return [body for path, body in sent if path == "/v1/cache/set"]


OPENAI_REPLY = {"id": "c1", "object": "chat.completion", "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "It caches by meaning."},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 90, "completion_tokens": 30, "total_tokens": 120,
                          "completion_tokens_details": {"reasoning_tokens": 12}}}
ANTHROPIC_REPLY = {"id": "m1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
                   "content": [{"type": "text", "text": "It caches by meaning."}], "stop_reason": "end_turn",
                   "usage": {"input_tokens": 50, "output_tokens": 20, "cache_creation_input_tokens": 0,
                             "cache_read_input_tokens": 400}}
OPENAI_BODY = {"model": "gpt-4o", "messages": [{"role": "user", "content": "What is semantic caching?"}]}
ANTHROPIC_BODY = {"model": "claude-sonnet-4-5", "max_tokens": 64,
                  "messages": [{"role": "user", "content": "What is semantic caching?"}]}


# ── sulci-proxy ──────────────────────────────────────────────────────────────
def _proxy(cache, reply, **kw):
    pytest.importorskip("fastapi", reason='pip install "sulci[proxy]"')
    import httpx
    from fastapi.testclient import TestClient
    from sulci.proxy.app import build_app
    upstream = []

    def handler(request):
        upstream.append(request)
        return httpx.Response(200, json=reply, headers={"content-type": "application/json"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TestClient(build_app(cache, client=client, share_across_models=True, **kw)), upstream


def test_proxy_openai_reply_carries_its_counts_reasoning_included():
    cache, sent = _managed()
    c, _ = _proxy(cache, OPENAI_REPLY)
    r = c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert r.status_code == 200 and r.json() == OPENAI_REPLY   # the client gets upstream's reply, untouched
    assert _sets(sent)[0]["usage"] == {"input_tokens": 90, "output_tokens": 30}


def test_proxy_anthropic_reply_counts_prompt_cache_reads_as_input():
    cache, sent = _managed()
    c, _ = _proxy(cache, ANTHROPIC_REPLY)
    c.post("/v1/messages", json=ANTHROPIC_BODY)
    assert _sets(sent)[0]["usage"] == {"input_tokens": 450, "output_tokens": 20}


def test_proxy_reply_without_usage_is_stored_without_counts():
    cache, sent = _managed()
    reply = {k: v for k, v in OPENAI_REPLY.items() if k != "usage"}
    c, _ = _proxy(cache, reply)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    body = _sets(sent)[0]
    assert body["response"] == "It caches by meaning." and "usage" not in body


def test_proxy_unreadable_usage_is_stored_without_counts_and_logged(caplog):
    cache, sent = _managed()
    reply = dict(OPENAI_REPLY, usage={"prompt_tokens": -1, "completion_tokens": 3})
    c, _ = _proxy(cache, reply)
    with caplog.at_level(logging.WARNING):
        c.post("/v1/chat/completions", json=OPENAI_BODY)
    body = _sets(sent)[0]
    assert body["response"] == "It caches by meaning." and "usage" not in body
    assert "token counts were not recorded" in caplog.text


def test_proxy_stream_passes_through_and_stores_nothing():
    pytest.importorskip("fastapi")
    import httpx
    from fastapi.testclient import TestClient
    from sulci.proxy.app import build_app
    cache, sent = _managed()

    class _Stream(httpx.AsyncByteStream):   # a real byte stream, as tests/test_proxy.py builds it
        async def __aiter__(self):
            yield b'data: {"choices":[]}\n\n'
            yield b"data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(200, stream=_Stream(), headers={"content-type": "text/event-stream"})
    c = TestClient(build_app(cache, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                             share_across_models=True))
    r = c.post("/v1/chat/completions", json=dict(OPENAI_BODY, stream=True))
    assert r.headers["x-sulci-cache"] == "bypass" and _sets(sent) == []


def test_proxy_tool_call_reply_is_not_stored_so_carries_no_counts():
    cache, sent = _managed()
    reply = dict(OPENAI_REPLY, choices=[{"index": 0, "finish_reason": "tool_calls",
                                         "message": {"role": "assistant", "content": None,
                                                     "tool_calls": [{"id": "t", "type": "function"}]}}])
    c, _ = _proxy(cache, reply)
    r = c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert r.headers["x-sulci-cache"] == "miss-uncacheable" and _sets(sent) == []


def test_proxy_on_a_local_backend_sends_no_counts(tmp_path, fake_embedder, monkeypatch):
    cache = Cache(backend="sqlite", db_path=str(tmp_path / "p"), embedding_model=fake_embedder, telemetry=False)
    seen = []
    real = Cache._set

    def spy(self, *a, **kw):
        seen.append(kw)
        return real(self, *a, **kw)
    monkeypatch.setattr(Cache, "_set", spy)
    c, _ = _proxy(cache, OPENAI_REPLY)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    # no counts: either no usage keyword (the direct _set path) or usage=None (via Cache.set, which always passes it)
    assert seen and all(kw.get("usage") is None for kw in seen)


def test_proxy_subclass_set_without_usage_still_stores(tmp_path, fake_embedder):
    stored = []

    class Old(Cache):   # an override written before 1.1.0
        def set(self, query, response, tenant_id=None, session_id=None):
            stored.append(response)
            return super().set(query, response, tenant_id=tenant_id, session_id=session_id)
    cache, sent = _managed()
    cache.__class__ = Old
    c, _ = _proxy(cache, OPENAI_REPLY)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert stored == ["It caches by meaning."] and "usage" not in _sets(sent)[0]


def test_proxy_subclass_set_with_kwargs_gets_the_counts():
    class New(Cache):
        def set(self, query, response, **kw):
            return super().set(query, response, **kw)
    cache, sent = _managed()
    cache.__class__ = New
    c, _ = _proxy(cache, OPENAI_REPLY)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert _sets(sent)[0]["usage"] == {"input_tokens": 90, "output_tokens": 30}


def test_proxy_hit_still_reports_zero_upstream_tokens():
    """The hit envelope's usage stays 0: no tokens were consumed upstream (billing reconciliation)."""
    cache, sent = _managed()
    cache._backend._client.post = lambda path, json=None, **k: _Resp(
        {"response": "cached", "similarity": 0.97, "context_depth": 0})
    c, upstream = _proxy(cache, OPENAI_REPLY)
    r = c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert r.headers["x-sulci-cache"] == "hit" and upstream == []
    assert r.json()["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ── SulciLiteLLMCache ────────────────────────────────────────────────────────
def _litellm(cache):
    pytest.importorskip("litellm", reason='pip install "sulci[litellm]"')
    from sulci.integrations.litellm import SulciLiteLLMCache
    return SulciLiteLLMCache(cache, namespace_by_model=False)


MSG = {"model": "gpt-4o", "messages": [{"role": "user", "content": "What is semantic caching?"}]}


def _litellm_value(usage):
    """What LiteLLM's Cache._add_cache_logic hands set_cache: the response as model_dump_json()."""
    resp = {"id": "c1", "object": "chat.completion", "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "It caches by meaning."}}]}
    if usage is not None:
        resp["usage"] = usage
    return {"timestamp": 1.0, "response": json.dumps(resp)}


def test_litellm_sends_the_response_s_counts_on_managed():
    cache, sent = _managed()
    adapter = _litellm(cache)
    adapter.set_cache("k", _litellm_value({"prompt_tokens": 70, "completion_tokens": 10}), **MSG)
    assert _sets(sent)[0]["usage"] == {"input_tokens": 70, "output_tokens": 10}


def test_litellm_real_model_response_anthropic_cache_tokens_once():
    """A real litellm ModelResponse (Anthropic, normalised): prompt_tokens already includes the cache tokens."""
    litellm = pytest.importorskip("litellm")
    cache, sent = _managed()
    adapter = _litellm(cache)
    usage = litellm.Usage(prompt_tokens=150, completion_tokens=30, total_tokens=180,
                          cache_read_input_tokens=40, cache_creation_input_tokens=10)
    resp = litellm.ModelResponse(usage=usage, choices=[{"message": {"role": "assistant", "content": "x"}}])
    adapter.set_cache("k", {"timestamp": 1.0, "response": resp.model_dump_json()}, **MSG)
    assert _sets(sent)[0]["usage"] == {"input_tokens": 150, "output_tokens": 30}


def test_litellm_unreadable_usage_is_stored_without_counts_warned_once(caplog):
    cache, sent = _managed()
    adapter = _litellm(cache)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            adapter.set_cache("k", _litellm_value({"prompt_tokens": 7, "total_tokens": 7}), **MSG)   # no output
    assert len(_sets(sent)) == 3 and all("usage" not in b for b in _sets(sent))
    assert caplog.text.count("token counts not recorded") == 1


def test_litellm_embedding_entry_carries_no_counts_silently(caplog):
    """LiteLLM stores embeddings as CachedEmbedding — a top-level prompt_tokens, no usage key."""
    cache, sent = _managed()
    value = {"timestamp": 1.0, "response": {"embedding": [0.1, 0.2], "index": 0, "object": "embedding",
                                             "model": "text-embedding-3-small", "prompt_tokens": 7}}
    with caplog.at_level(logging.WARNING):
        _litellm(cache).set_cache("k", value, input="What is semantic caching?")
    assert _sets(sent) and "usage" not in _sets(sent)[0] and "token counts" not in caplog.text


def test_litellm_a_streamed_call_carries_no_counts():
    """LiteLLM caches an assembled stream; its counts may be LiteLLM's own estimate — never sent."""
    cache, sent = _managed()
    _litellm(cache).set_cache("k", _litellm_value({"prompt_tokens": 70, "completion_tokens": 10}), stream=True, **MSG)
    assert _sets(sent) and "usage" not in _sets(sent)[0]


def test_litellm_more_than_one_choice_carries_no_counts():
    cache, sent = _managed()
    _litellm(cache).set_cache("k", _litellm_value({"prompt_tokens": 70, "completion_tokens": 40}), n=4, **MSG)
    assert _sets(sent) and "usage" not in _sets(sent)[0]


def test_proxy_more_than_one_choice_is_stored_without_counts():
    cache, sent = _managed()
    two = dict(OPENAI_REPLY, choices=OPENAI_REPLY["choices"] + [dict(OPENAI_REPLY["choices"][0], index=1)])
    c, _ = _proxy(cache, two)
    c.post("/v1/chat/completions", json=dict(OPENAI_BODY, n=2))
    body = _sets(sent)[0]
    assert body["response"] == "It caches by meaning." and "usage" not in body


def test_proxy_wraps_decorated_set_without_usage_still_stores():
    """A @functools.wraps(Cache.set) wrapper with a narrower signature must not be handed usage=."""
    import functools
    cache, sent = _managed()
    stored = []
    real = cache.set

    @functools.wraps(Cache.set)
    def audited(query, response, tenant_id=None, session_id=None):
        stored.append(response)
        return real(query, response, tenant_id=tenant_id, session_id=session_id)
    cache.set = audited
    c, _ = _proxy(cache, OPENAI_REPLY)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert stored == ["It caches by meaning."] and "usage" not in _sets(sent)[0]


def test_proxy_anthropic_iterations_are_summed():
    cache, sent = _managed()
    reply = dict(ANTHROPIC_REPLY, usage={"input_tokens": 50, "output_tokens": 20, "iterations": [
        {"input_tokens": 30, "output_tokens": 15, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0},
        {"input_tokens": 50, "output_tokens": 20, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 5}]})
    c, _ = _proxy(cache, reply)
    c.post("/v1/messages", json=ANTHROPIC_BODY)
    assert _sets(sent)[0]["usage"] == {"input_tokens": 185, "output_tokens": 35}


def test_litellm_on_a_local_backend_sends_no_counts(tmp_path, fake_embedder, monkeypatch):
    cache = Cache(backend="sqlite", db_path=str(tmp_path / "l"), embedding_model=fake_embedder, telemetry=False)
    seen = []
    real = Cache.set

    def spy(self, *a, **kw):
        seen.append(kw)
        return real(self, *a, **kw)
    monkeypatch.setattr(Cache, "set", spy)
    adapter = _litellm(cache)
    adapter.set_cache("k", _litellm_value({"prompt_tokens": 70, "completion_tokens": 10}), **MSG)
    assert seen and all("usage" not in kw for kw in seen)


def test_litellm_value_without_usage_is_stored_without_counts():
    cache, sent = _managed()
    _litellm(cache).set_cache("k", _litellm_value(None), **MSG)
    assert _sets(sent) and "usage" not in _sets(sent)[0]


@pytest.mark.parametrize("value", ["not json", b"\xff\xfe", 42, None])
def test_litellm_odd_values_never_raise(value):
    pytest.importorskip("litellm", reason='pip install "sulci[litellm]"')   # before the import: 3.9 has no litellm
    from sulci.integrations.litellm import _usage_from_value
    assert _usage_from_value(value) is None
