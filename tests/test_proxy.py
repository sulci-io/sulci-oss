# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
tests/test_proxy.py
────────────────────
Covers sulci.proxy — the OpenAI/Anthropic-compatible caching shim.

Upstream is a stub httpx transport, so nothing leaves the machine and the
call count is directly observable: the whole point of the proxy is that the
second identical request does not reach upstream.
"""

from __future__ import annotations

import json

import pytest

from sulci import Cache

pytest.importorskip("fastapi", reason='pip install "sulci[proxy]"')
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from sulci.proxy.app import (  # noqa: E402
    build_app,
    extract_prompt,
    is_cacheable_response,
)

OPENAI_BODY = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What is semantic caching?"}],
}
ANTHROPIC_BODY = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "What is semantic caching?"}],
}
OPENAI_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "It caches by meaning."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
}
ANTHROPIC_REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "It caches by meaning."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 9, "output_tokens": 5},
}


class Upstream:
    """Counting stub transport."""

    def __init__(self, reply=None, status=200, sse=False):
        self.calls = []
        self.reply = reply if reply is not None else OPENAI_REPLY
        self.status = status
        self.sse = sse

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.sse:
            # A real byte stream, so the pass-through path exercises
            # aiter_raw() the way it does against a live upstream. A
            # json= response is pre-loaded and raises StreamConsumed.
            async def body():
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            return httpx.Response(
                self.status,
                stream=httpx.AsyncByteStream() if False else _AsyncStream(body()),
                headers={"content-type": "text/event-stream", "x-request-id": "up-1"},
            )
        return httpx.Response(
            self.status,
            json=self.reply,
            headers={"content-type": "application/json", "x-request-id": "up-1"},
        )


class _AsyncStream(httpx.AsyncByteStream):
    def __init__(self, it):
        self._it = it

    async def __aiter__(self):
        async for chunk in self._it:
            yield chunk


@pytest.fixture
def cache(tmp_path, fake_embedder):
    return Cache(
        backend="sqlite",
        db_path=str(tmp_path / "px"),
        embedding_model=fake_embedder,
        threshold=0.85,
        telemetry=False,
    )


def make(cache, upstream: Upstream, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    app = build_app(cache, client=client, share_across_models=True, **kw)
    return TestClient(app)


# ── pure helpers ─────────────────────────────────────────────────────────
def test_extract_prompt_takes_the_last_user_turn():
    body = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
    }
    assert extract_prompt(body) == "second"


def test_extract_prompt_flattens_content_blocks():
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    assert extract_prompt(body) == "hi"


def test_extract_prompt_ignores_system_only_requests():
    assert extract_prompt({"messages": [{"role": "system", "content": "be nice"}]}) is None


def test_extract_prompt_handles_a_malformed_body():
    assert extract_prompt({}) is None
    assert extract_prompt({"messages": "nope"}) is None


def test_openai_tool_calls_are_not_cacheable():
    payload = {
        "choices": [
            {"message": {"tool_calls": [{"id": "1"}]}, "finish_reason": "tool_calls"}
        ]
    }
    assert is_cacheable_response(payload) is False


def test_anthropic_tool_use_is_not_cacheable():
    assert is_cacheable_response({"stop_reason": "tool_use", "content": []}) is False
    assert is_cacheable_response({"content": [{"type": "tool_use"}]}) is False


def test_plain_replies_are_cacheable():
    assert is_cacheable_response(OPENAI_REPLY) is True
    assert is_cacheable_response(ANTHROPIC_REPLY) is True


# ── OpenAI route ─────────────────────────────────────────────────────────
def test_first_call_is_a_miss_and_reaches_upstream(cache):
    up = Upstream()
    r = make(cache, up).post("/v1/chat/completions", json=OPENAI_BODY)
    assert r.status_code == 200
    assert r.headers["x-sulci-cache"] == "miss"
    assert len(up.calls) == 1


def test_second_identical_call_never_reaches_upstream(cache):
    up = Upstream()
    c = make(cache, up)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    r = c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert r.headers["x-sulci-cache"] == "hit"
    assert len(up.calls) == 1  # the entire point of the proxy


def test_cached_reply_is_valid_openai_shape(cache):
    up = Upstream()
    c = make(cache, up)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    body = c.post("/v1/chat/completions", json=OPENAI_BODY).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "It caches by meaning."
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["model"] == "gpt-4o"


def test_cached_reply_reports_zero_tokens_not_fabricated_ones(cache):
    up = Upstream()
    c = make(cache, up)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    usage = c.post("/v1/chat/completions", json=OPENAI_BODY).json()["usage"]
    # No tokens were consumed upstream. Echoing the original counts would
    # corrupt any billing reconciliation built on this proxy.
    assert usage["total_tokens"] == 0


def test_hit_reports_similarity(cache):
    up = Upstream()
    c = make(cache, up)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    r = c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert float(r.headers["x-sulci-similarity"]) >= 0.85


def test_different_prompt_is_a_second_miss(cache):
    up = Upstream()
    c = make(cache, up)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    other = dict(OPENAI_BODY, messages=[{"role": "user", "content": "zebra migration"}])
    r = c.post("/v1/chat/completions", json=other)
    assert r.headers["x-sulci-cache"] == "miss"
    assert len(up.calls) == 2


# ── Anthropic route ──────────────────────────────────────────────────────
def test_anthropic_round_trip(cache):
    up = Upstream(reply=ANTHROPIC_REPLY)
    c = make(cache, up)
    assert c.post("/v1/messages", json=ANTHROPIC_BODY).headers["x-sulci-cache"] == "miss"
    r = c.post("/v1/messages", json=ANTHROPIC_BODY)
    assert r.headers["x-sulci-cache"] == "hit"
    assert len(up.calls) == 1


def test_cached_anthropic_reply_is_valid_shape(cache):
    up = Upstream(reply=ANTHROPIC_REPLY)
    c = make(cache, up)
    c.post("/v1/messages", json=ANTHROPIC_BODY)
    body = c.post("/v1/messages", json=ANTHROPIC_BODY).json()
    assert body["type"] == "message"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "It caches by meaning."
    assert body["stop_reason"] == "end_turn"


# ── things that must NOT be cached ───────────────────────────────────────
def test_streaming_bypasses_the_cache_entirely(cache):
    up = Upstream(sse=True)
    c = make(cache, up)
    streaming = dict(OPENAI_BODY, stream=True)
    r1 = c.post("/v1/chat/completions", json=streaming)
    assert r1.headers["x-sulci-cache"] == "bypass"
    c.post("/v1/chat/completions", json=streaming)
    assert len(up.calls) == 2  # never served from cache


def test_tool_call_responses_are_not_stored(cache):
    tool_reply = dict(
        OPENAI_REPLY,
        choices=[
            {"message": {"tool_calls": [{"id": "1"}]}, "finish_reason": "tool_calls"}
        ],
    )
    up = Upstream(reply=tool_reply)
    c = make(cache, up)
    r = c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert r.headers["x-sulci-cache"] == "miss-uncacheable"
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert len(up.calls) == 2


def test_upstream_errors_are_not_cached(cache):
    up = Upstream(status=500)
    c = make(cache, up)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    assert len(up.calls) == 2


def test_error_status_is_passed_through(cache):
    up = Upstream(status=429)
    r = make(cache, up).post("/v1/chat/completions", json=OPENAI_BODY)
    assert r.status_code == 429


# ── headers & scoping ────────────────────────────────────────────────────
def test_upstream_headers_survive(cache):
    up = Upstream()
    r = make(cache, up).post("/v1/chat/completions", json=OPENAI_BODY)
    assert r.headers["x-request-id"] == "up-1"


def test_auth_header_is_forwarded(cache):
    up = Upstream()
    make(cache, up).post(
        "/v1/chat/completions", json=OPENAI_BODY, headers={"authorization": "Bearer x"}
    )
    assert up.calls[0].headers["authorization"] == "Bearer x"


def test_tenant_header_scopes_the_lookup(cache):
    up = Upstream()
    c = make(cache, up)
    c.post("/v1/chat/completions", json=OPENAI_BODY, headers={"x-sulci-tenant-id": "sha-a"})
    r = c.post("/v1/chat/completions", json=OPENAI_BODY, headers={"x-sulci-tenant-id": "sha-b"})
    # ⚠️ sqlite does not enforce tenant isolation, so this is a HIT. The
    # assertion records real behaviour; see _scope.py for why the adapter
    # warns rather than pretending otherwise.
    assert r.headers["x-sulci-cache"] == "hit"


def test_per_model_scoping_warns_when_unenforced(cache):
    from sulci.integrations._scope import ScopeNotEnforcedWarning

    up = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(up.handler))
    with pytest.warns(ScopeNotEnforcedWarning):
        build_app(cache, client=client, share_across_models=False)


# ── ops endpoints ────────────────────────────────────────────────────────
def test_healthz(cache):
    assert make(cache, Upstream()).get("/healthz").json() == {"status": "ok"}


def test_stats_endpoint_reports_hits(cache):
    up = Upstream()
    c = make(cache, up)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    c.post("/v1/chat/completions", json=OPENAI_BODY)
    stats = c.get("/stats").json()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_build_app_rejects_cache_and_kwargs_together(cache):
    with pytest.raises(TypeError):
        build_app(cache, backend="sqlite")
