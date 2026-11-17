# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_proxy_observe.py — ``sulci-proxy --observe`` (oss-0070, 1.5.0).

An observed proxy forwards EVERY call upstream and returns the upstream answer; ``x-sulci-observe`` says what the
cache would have done; the upstream's token counts are recorded (what a hit would have saved); a stream, which is
never looked up, is counted as skipped. Refused with the managed backend, as the Cache is.
"""
from __future__ import annotations

import sqlite3

import pytest

from sulci import Cache

pytest.importorskip("fastapi", reason='pip install "sulci[proxy]"')
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from sulci.proxy.app import build_app  # noqa: E402

KEY = "sk-sulci-" + "t" * 32
BODY = {"model": "gpt-4o", "messages": [{"role": "user", "content": "What is semantic caching?"}]}
REPLY = {"id": "c1", "object": "chat.completion", "model": "gpt-4o",
         "choices": [{"index": 0, "message": {"role": "assistant", "content": "It caches by meaning."},
                      "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14}}


class _AsyncStream(httpx.AsyncByteStream):
    def __init__(self, it):
        self._it = it

    async def __aiter__(self):
        async for chunk in self._it:
            yield chunk


class Upstream:
    def __init__(self):
        self.calls = 0

    def handler(self, request):
        self.calls += 1
        if b'"stream": true' in request.content or b'"stream":true' in request.content:
            async def body():
                yield b"data: [DONE]\n\n"
            return httpx.Response(200, stream=_AsyncStream(body()), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=REPLY, headers={"content-type": "application/json"})


@pytest.fixture
def setup(tmp_path, fake_embedder):
    cache = Cache(backend="sqlite", db_path=str(tmp_path / "px"), embedding_model=fake_embedder, threshold=0.85,
                  telemetry=False, observe=True, observe_path=str(tmp_path / "obs.db"))
    up = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(up.handler))
    return TestClient(build_app(cache, client=client, share_across_models=True)), up, str(tmp_path / "obs.db")


def _rows(path, sql):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_every_call_goes_upstream_and_the_header_says_what_would_have_happened(setup):
    client, up, path = setup
    first = client.post("/v1/chat/completions", json=BODY)
    second = client.post("/v1/chat/completions", json=BODY)
    assert up.calls == 2
    assert first.json()["choices"][0]["message"]["content"] == "It caches by meaning."
    assert second.headers["x-sulci-cache"] == "miss"
    assert first.headers["x-sulci-observe"] == "would-miss"
    assert second.headers["x-sulci-observe"] == "would-hit"


def test_the_upstream_token_counts_are_recorded(setup):
    client, _, path = setup
    client.post("/v1/chat/completions", json=BODY)
    client.post("/v1/chat/completions", json=BODY)
    assert _rows(path, "SELECT label, input_tokens, output_tokens FROM observe_events WHERE kind='get' ORDER BY id") \
        == [("would_miss", 9, 5), ("would_hit", 9, 5)]
    assert _rows(path, "SELECT count(*) FROM observe_pairs")[0][0] == 1


def test_a_stream_is_counted_as_skipped(setup):
    client, up, path = setup
    r = client.post("/v1/chat/completions", json=dict(BODY, stream=True))
    assert r.status_code == 200 and up.calls == 1
    assert _rows(path, "SELECT label, reason FROM observe_events") == [("skipped", "stream")]


def test_refused_with_the_managed_backend():
    with pytest.raises(TypeError, match="Observe mode"):
        build_app(backend="sulci", api_key=KEY, telemetry=False, observe=True)


def test_the_cli_takes_observe_and_refuses_a_path_without_it(monkeypatch):
    pytest.importorskip("uvicorn")
    from sulci.proxy import __main__ as cli
    got = {}
    monkeypatch.setattr("sulci.proxy.app.build_app", lambda **kw: got.update(kw) or object())
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    cli.main(["--observe", "--observe-path", "/tmp/x.db"])
    assert got["observe"] is True and got["observe_path"] == "/tmp/x.db"
    with pytest.raises(SystemExit):
        cli.main(["--observe-path", "/tmp/x.db"])
