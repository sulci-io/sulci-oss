"""
flow_2_e2e.py — verify Flow 2 · full round-trip with mocked gateway
====================================================================

What this script verifies (that ``flow_2.py`` doesn't)
------------------------------------------------------
``flow_2.py`` validates the SulciCloudBackend's *construction* — headers,
key resolution, public method names. This script validates the actual
*wire behavior*: a full ``remote_set → remote_get`` round-trip with a
mock gateway, asserting:

  1. ``remote_set(query, response)`` POSTs to ``/v1/cache/set`` with the
     v0.6.0 CacheSetRequest shape: ``{query, response, user_id,
     session_id, ttl_seconds}``. No embedding, no tenant_id (server-side
     auth context handles tenant isolation).

  2. ``remote_get(query, threshold)`` POSTs to ``/v1/cache/get`` with the
     v0.6.0 CacheGetRequest shape: ``{query, threshold, user_id,
     session_id}``. Returns ``(response, similarity, context_depth)``.

  3. ``remote_get`` returns ``(None, 0.0, 0)`` — a silent miss — on every
     documented failure mode: 404, 5xx, timeout, connection error. The
     SDK contract is "cache never crashes the user's app", and this
     is the test that holds that contract honest.

  4. ``remote_set`` returns ``None`` and does not raise on any failure
     mode. Same contract, asymmetric: set is fire-and-forget.

  5. The X-Sulci-Key header is on every request — no public API call
     forgets the auth header.

Pre-v0.6.0 architecture
-----------------------
Pre-v0.6.0 the SDK embedded queries locally and sent ``{embedding,
tenant_id, ...}`` to the gateway. The gateway expected ``{query, ...}``
and 422-rejected every request — bug #62, masked for 14 months by the
swallow-all except clause. v0.6.0 closed the bug by making the SDK a
pure transport: it sends raw queries, the gateway does the embedding,
search, and event emit. This is the "library-engine" architecture.

This script is what you run when you suspect the wire format has drifted
again. If a future PR re-introduces ``embedding`` to the request body, or
re-introduces ``tenant_id`` (which is now derived from the api_key on the
gateway side), this script's payload-shape assertions will catch it.

How it runs offline
-------------------
Patches ``httpx.Client.post`` for the SulciCloudBackend's internal client.
The mock records each request (URL, body, headers) and returns
configurable stub responses. No real network, no real gateway.
"""
from __future__ import annotations

# ── Windows stdout encoding fix ─────────────────────────────────────────────
# Windows Python defaults sys.stdout.encoding to cp1252, which cannot encode
# the ✓ / ✗ characters used in the PASS/FAIL banners below. Reconfigure
# stdout/stderr to UTF-8 at startup so the script runs cleanly on every
# OS in the CI matrix without relying on PYTHONIOENCODING propagating
# through the subprocess wrapper.
import sys as _sys_for_encoding
if hasattr(_sys_for_encoding.stdout, "reconfigure"):
    try:
        _sys_for_encoding.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys_for_encoding.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
# ────────────────────────────────────────────────────────────────────────────

import os
import sys
from typing import Any
from unittest.mock import patch, MagicMock

import httpx


# ── Mock gateway ────────────────────────────────────────────────────────────
class MockGateway:
    """Stand-in for the live gateway. Records every request and
    returns scripted responses based on URL."""

    def __init__(self):
        self.calls: list[dict] = []
        # Scripted responses per URL — set per test.
        self.scripts: dict[str, Any] = {}

    def __call__(self, url: str, json=None, **kwargs):
        self.calls.append({
            "url":     url,
            "body":    json,
            "headers": dict(self._client_headers()),
        })
        return self.scripts.get(url, self._default_response(url))

    @staticmethod
    def _client_headers():
        # Headers come from the httpx.Client construction; the post()
        # call passes empty default. We read them off the instance below.
        return {}

    def _default_response(self, url: str):
        r = MagicMock()
        r.status_code = 200
        r.json = MagicMock(return_value={
            "response": None, "similarity": 0.0, "context_depth": 0,
        })
        r.raise_for_status = MagicMock()
        return r

    @staticmethod
    def ok_get(response: str, similarity: float, depth: int = 0):
        """Build a 200 hit response."""
        r = MagicMock()
        r.status_code = 200
        r.json = MagicMock(return_value={
            "response": response, "similarity": similarity,
            "context_depth": depth,
        })
        r.raise_for_status = MagicMock()
        return r

    @staticmethod
    def status(code: int, body: dict | None = None):
        r = MagicMock()
        r.status_code = code
        r.json = MagicMock(return_value=body or {})
        def _raise():
            raise httpx.HTTPStatusError(
                f"{code}", request=MagicMock(), response=r)
        r.raise_for_status = _raise
        return r

    @staticmethod
    def timeout():
        return httpx.TimeoutException("simulated timeout")

    @staticmethod
    def connerror():
        return httpx.ConnectError("simulated connection error")


def _install_mock(backend, mock: MockGateway):
    """Replace the backend's httpx.Client.post with the mock, preserving
    the client's headers for inspection."""
    orig_post = backend._client.post

    def fake_post(url, json=None, **kwargs):
        # Re-attach the client's headers to the call record so we can
        # assert X-Sulci-Key without rummaging through internals.
        # We do this by recording the client's headers before the mock
        # returns.
        result = mock(url, json=json, **kwargs)
        mock.calls[-1]["headers"] = dict(backend._client.headers)
        if isinstance(result, Exception):
            raise result
        return result

    backend._client.post = fake_post  # type: ignore[method-assign]


# ── Tests ───────────────────────────────────────────────────────────────────
def run() -> int:
    failures: list[str] = []
    os.environ.pop("SULCI_API_KEY", None)

    from sulci.backends.cloud import SulciCloudBackend

    def expect(cond, msg):
        if not cond:
            failures.append(msg)

    # ── 1. remote_set happy path: payload shape + headers ───────────────────
    be = SulciCloudBackend(api_key="sk-sulci-flow-2-e2e")
    mock = MockGateway()
    _install_mock(be, mock)

    be.remote_set(
        query        = "How do I deploy to AWS?",
        response     = "Use 'aws ecs update-service' or CDK.",
        user_id      = "u_alice",
        session_id   = "s_xyz",
        ttl_seconds  = 3600,
    )

    expect(len(mock.calls) == 1, f"remote_set should POST once; got {len(mock.calls)}")
    if mock.calls:
        call = mock.calls[0]
        expect(call["url"] == "/v1/cache/set",
               f"remote_set URL must be /v1/cache/set; got {call['url']!r}")
        body = call["body"] or {}
        expected_keys = {"query", "response", "user_id", "session_id", "ttl_seconds"}
        actual_keys   = set(body.keys())
        expect(actual_keys == expected_keys,
               f"CacheSetRequest shape drift; "
               f"extra={actual_keys - expected_keys}, "
               f"missing={expected_keys - actual_keys}")
        # Privacy: no embedding, no tenant_id, no raw vector
        for forbidden in ("embedding", "tenant_id", "vector"):
            expect(forbidden not in body,
                   f"set payload must not carry {forbidden!r} "
                   f"(tenant_id is server-side from api_key auth context; "
                   f"embedding moved server-side in v0.6.0)")
        expect(body.get("query")    == "How do I deploy to AWS?",
               "query field round-trips into the request body")
        expect(body.get("response") == "Use 'aws ecs update-service' or CDK.",
               "response field round-trips")
        expect(body.get("ttl_seconds") == 3600,
               "ttl_seconds field round-trips")
        # Auth header present (httpx normalizes header names to lowercase)
        hdrs_lower = {k.lower(): v for k, v in call["headers"].items()}
        expect(hdrs_lower.get("x-sulci-key") == "sk-sulci-flow-2-e2e",
               f"X-Sulci-Key must be on every request; got {hdrs_lower!r}")

    # ── 2. remote_get happy path: hit returns (response, sim, depth) ────────
    be = SulciCloudBackend(api_key="sk-sulci-flow-2-e2e")
    mock = MockGateway()
    mock.scripts["/v1/cache/get"] = mock.ok_get(
        response   = "Use 'aws ecs update-service' or CDK.",
        similarity = 0.93,
        depth      = 2,
    )
    _install_mock(be, mock)

    response, similarity, depth = be.remote_get(
        query     = "How do I push updates to ECS?",
        threshold = 0.85,
        user_id   = "u_alice",
        session_id = "s_xyz",
    )

    expect(len(mock.calls) == 1, "remote_get should POST exactly once")
    if mock.calls:
        call = mock.calls[0]
        expect(call["url"] == "/v1/cache/get",
               f"remote_get URL must be /v1/cache/get; got {call['url']!r}")
        body = call["body"] or {}
        expected_keys = {"query", "threshold", "user_id", "session_id"}
        actual_keys   = set(body.keys())
        expect(actual_keys == expected_keys,
               f"CacheGetRequest shape drift; "
               f"extra={actual_keys - expected_keys}, "
               f"missing={expected_keys - actual_keys}")
        for forbidden in ("embedding", "tenant_id", "vector"):
            expect(forbidden not in body,
                   f"get payload must not carry {forbidden!r}")
        expect(body.get("query") == "How do I push updates to ECS?",
               "query field round-trips")
        expect(body.get("threshold") == 0.85,
               "threshold field round-trips")

    expect(response   == "Use 'aws ecs update-service' or CDK.",
           f"hit response should round-trip; got {response!r}")
    expect(similarity == 0.93, f"similarity should round-trip; got {similarity!r}")
    expect(depth      == 2,    f"context_depth should round-trip; got {depth!r}")

    # ── 3. remote_get miss: 200 with response=null returns (None, 0.0, 0) ──
    be = SulciCloudBackend(api_key="sk-sulci-flow-2-e2e")
    mock = MockGateway()
    mock.scripts["/v1/cache/get"] = mock.ok_get(
        response=None, similarity=0.0, depth=0,
    )
    _install_mock(be, mock)
    r, s, d = be.remote_get(query="anything", threshold=0.85)
    expect((r, s, d) == (None, 0.0, 0),
           f"miss should return (None, 0.0, 0); got ({r!r}, {s!r}, {d!r})")

    # ── 4. remote_get error modes ALL silent-miss ──────────────────────────
    for scenario, response in [
        ("404",            MockGateway.status(404)),
        ("500",            MockGateway.status(500)),
        ("422",            MockGateway.status(422, {"detail": "schema mismatch"})),
        ("timeout",        MockGateway.timeout()),
        ("connection",     MockGateway.connerror()),
    ]:
        be = SulciCloudBackend(api_key="sk-sulci-flow-2-e2e")
        mock = MockGateway()
        mock.scripts["/v1/cache/get"] = response
        _install_mock(be, mock)
        try:
            r, s, d = be.remote_get(query="x", threshold=0.85)
            expect((r, s, d) == (None, 0.0, 0),
                   f"{scenario}: remote_get must return silent miss; "
                   f"got ({r!r}, {s!r}, {d!r})")
        except Exception as e:
            failures.append(
                f"{scenario}: remote_get raised {type(e).__name__}: {e}; "
                f"contract is 'never raise'")

    # ── 5. remote_set error modes ALL fire-and-forget (no raise) ───────────
    for scenario, response in [
        ("404",        MockGateway.status(404)),
        ("500",        MockGateway.status(500)),
        ("timeout",    MockGateway.timeout()),
        ("connection", MockGateway.connerror()),
    ]:
        be = SulciCloudBackend(api_key="sk-sulci-flow-2-e2e")
        mock = MockGateway()
        mock.scripts["/v1/cache/set"] = response
        _install_mock(be, mock)
        try:
            result = be.remote_set(query="x", response="y")
            expect(result is None,
                   f"{scenario}: remote_set must return None; got {result!r}")
        except Exception as e:
            failures.append(
                f"{scenario}: remote_set raised {type(e).__name__}: {e}; "
                f"contract is 'never raise (fire-and-forget)'")

    # ── 6. delete_user + clear hit canonical GDPR paths (v0.6.2 / #103) ────
    be = SulciCloudBackend(api_key="sk-sulci-flow-2-e2e")
    delete_calls: list[str] = []
    def fake_delete(url, **kwargs):
        delete_calls.append(url)
        r = MagicMock()
        r.status_code = 200   # 0.9.3: only {"status": "ok"} confirms an erasure; anything else raises
        r.json.return_value = {"status": "ok", "deleted": 0, "tenant_id": 1}
        return r
    be._client.delete = fake_delete  # type: ignore[method-assign]

    be.delete_user("u_alice")
    be.clear()

    expect(delete_calls == ["/v1/cache/user/u_alice", "/v1/cache/clear"],
           f"delete_user + clear should hit /v1/cache/user/<id> + /v1/cache/clear "
           f"(GDPR canonical routes from sulci-platform #103); got {delete_calls}")

    if failures:
        print("FAIL — Flow 2 e2e")
        for f in failures:
            print(f"  ✗ {f}")
        return 1

    print("PASS — Flow 2 e2e · full round-trip with mocked gateway")
    print("  ✓ remote_set POSTs /v1/cache/set with v0.6.0 CacheSetRequest shape")
    print("  ✓ remote_get POSTs /v1/cache/get with v0.6.0 CacheGetRequest shape")
    print("  ✓ no embedding/tenant_id/vector fields leak on the wire")
    print("  ✓ X-Sulci-Key present on every request")
    print("  ✓ hit round-trips (response, similarity, context_depth)")
    print("  ✓ miss returns (None, 0.0, 0)")
    print("  ✓ 404/500/422/timeout/connection-error all silent-miss on get")
    print("  ✓ 404/500/timeout/connection-error all fire-and-forget on set")
    print("  ✓ delete_user/clear hit /v1/cache/user/<id> + /v1/cache/clear (GDPR)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
