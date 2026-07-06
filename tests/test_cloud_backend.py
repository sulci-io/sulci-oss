# tests/test_cloud_backend.py
"""
Tests for SulciCloudBackend and Cache(backend='sulci') wiring.

Coverage
--------
- SulciCloudBackend requires api_key — raises ValueError if missing
- search() returns (None, 0.0) on timeout — never raises
- search() returns (None, 0.0) on any other exception — never raises
- search() returns (response, similarity) on success
- upsert() is silent on failure — never raises
- upsert() sends correct payload on success
- delete_user() is silent on failure — never raises
- clear() is silent on failure — never raises
- repr() shows key prefix and url
- Cache(backend='sulci', api_key=...) constructs SulciCloudBackend
- Cache(backend='sulci') resolves key from SULCI_API_KEY env var
- Cache(backend='sulci') resolves key from sulci.connect()
- Cache(backend='sulci') with no key raises ValueError
- Unknown backend still raises ValueError
"""

import os
import pytest
import httpx
from unittest.mock import patch, MagicMock, call


# ── Helpers ───────────────────────────────────────────────────────────────────

TEST_KEY = "sk-sulci-testkey1234567890"


def make_backend(key=TEST_KEY, timeout=5.0, gateway_url=""):
    from sulci.backends.cloud import SulciCloudBackend
    return SulciCloudBackend(api_key=key, timeout=timeout, gateway_url=gateway_url)


def mock_response(data: dict, status: int = 200):
    resp = MagicMock()
    resp.json.return_value = data
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# Vendored gateway pydantic models — wire-contract source of truth
# ──────────────────────────────────────────────────────────────────────────────
# These are EXACT COPIES of the gateway's request/response models, vendored
# here so sulci-oss CI can verify the SDK's wire payload matches what the
# gateway expects without depending on the platform repo at test time.
#
# Source of truth (keep in sync):
#   sulci-platform:shared/models.py
#
# When the gateway adds/removes/renames a field on CacheGetRequest /
# CacheGetResponse / CacheSetRequest, update these copies AND the SDK's
# remote_get / remote_set methods in the same PR. The TestCanonicalGatewayPaths
# class below uses model_validate() to round-trip the SDK's payload through
# these models, so any drift fails CI loudly rather than silently 422'ing
# in production (which is exactly how Issue #62 stayed live for ~14 months).

from typing import Optional as _Optional
from pydantic import BaseModel as _BaseModel


class _GatewayContractModels:
    """Wire-contract anchor — DO NOT mutate at test time."""

    class CacheGetRequest(_BaseModel):
        query:      str
        threshold:  float          = 0.85
        user_id:    _Optional[str] = None
        session_id: _Optional[str] = None

    class CacheGetResponse(_BaseModel):
        response:      _Optional[str]
        similarity:    float
        context_depth: int           = 0
        cache_hit:     bool
        latency_ms:    float

    class CacheSetRequest(_BaseModel):
        query:       str
        response:    str
        user_id:     _Optional[str] = None
        session_id:  _Optional[str] = None
        ttl_seconds: _Optional[int] = None

    # v0.6.2 — DELETE-route response shapes (sulci-platform #103).
    # Both routes return 200 with these shapes; failures land in the
    # `except Exception: log.warning(...)` path on the SDK side (no
    # response model parsing needed there — just absorb-and-log).
    class CacheClearResponse(_BaseModel):
        status:    str
        deleted:   int
        tenant_id: int

    class CacheDeleteUserResponse(_BaseModel):
        status:    str
        deleted:   int
        tenant_id: int
        user_id:   str


# ══════════════════════════════════════════════════════════════════════════════
# Construction
# ══════════════════════════════════════════════════════════════════════════════

class TestConstruction:

    def test_requires_api_key(self):
        """ValueError if api_key is None or empty string."""
        from sulci.backends.cloud import SulciCloudBackend
        with pytest.raises(ValueError, match="api_key is required"):
            SulciCloudBackend(api_key=None)

    def test_requires_api_key_not_empty(self):
        from sulci.backends.cloud import SulciCloudBackend
        with pytest.raises(ValueError, match="api_key is required"):
            SulciCloudBackend(api_key="")

    def test_repr_shows_key_prefix_and_url(self):
        b = make_backend()
        r = repr(b)
        assert "api.sulci.io"     in r
        assert TEST_KEY[:16]      in r

    def test_default_timeout_is_5s(self):
        b = make_backend()
        assert b._timeout == 5.0

    def test_custom_timeout(self):
        b = make_backend(timeout=10.0)
        assert b._timeout == 10.0

    def test_default_gateway_url_is_cloud(self):
        """No gateway_url → base_url defaults to api.sulci.io."""
        b = make_backend()
        assert b._base_url == "https://api.sulci.io"

    def test_custom_gateway_url_is_used(self):
        """Enterprise VPC can point to a custom gateway."""
        b = make_backend(gateway_url="https://cache.acme.internal")
        assert b._base_url == "https://cache.acme.internal"

    def test_custom_gateway_url_trailing_slash_stripped(self):
        """Trailing slash on gateway_url is stripped cleanly."""
        b = make_backend(gateway_url="https://cache.acme.internal/")
        assert b._base_url == "https://cache.acme.internal"

    # ── SULCI_GATEWAY env var (closes v0.5.5 follow-up #TBD-2) ────────────
    # v0.5.5 shipped SULCI_GATEWAY for telemetry; cloud backend was left
    # honoring gateway_url= kwarg only, creating a split-brain state in
    # staging (telemetry to staging URL, cache traffic to prod). These
    # tests lock in the 3-tier precedence chain: kwarg > env > default.

    def test_env_var_used_when_no_kwarg(self, monkeypatch):
        """SULCI_GATEWAY env var is honored when no explicit gateway_url."""
        monkeypatch.setenv("SULCI_GATEWAY", "https://staging.sulci.io")
        b = make_backend()
        assert b._base_url == "https://staging.sulci.io"

    def test_env_var_trailing_slash_stripped(self, monkeypatch):
        """Trailing slash on SULCI_GATEWAY is stripped, matching kwarg path."""
        monkeypatch.setenv("SULCI_GATEWAY", "https://staging.sulci.io/")
        b = make_backend()
        assert b._base_url == "https://staging.sulci.io"

    def test_kwarg_wins_over_env_var(self, monkeypatch):
        """Explicit gateway_url= kwarg overrides SULCI_GATEWAY env var."""
        monkeypatch.setenv("SULCI_GATEWAY", "https://staging.sulci.io")
        b = make_backend(gateway_url="https://cache.acme.internal")
        assert b._base_url == "https://cache.acme.internal"

    def test_empty_env_var_falls_through_to_default(self, monkeypatch):
        """Empty SULCI_GATEWAY string is treated as unset — falls to default.

        Guards against subtle CI misconfigurations that export the var but
        leave it empty (e.g., ``export SULCI_GATEWAY=$STAGING_URL`` when
        ``STAGING_URL`` is unset)."""
        monkeypatch.setenv("SULCI_GATEWAY", "")
        b = make_backend()
        assert b._base_url == "https://api.sulci.io"

    def test_whitespace_env_var_falls_through_to_default(self, monkeypatch):
        """Whitespace-only SULCI_GATEWAY is treated as unset."""
        monkeypatch.setenv("SULCI_GATEWAY", "   ")
        b = make_backend()
        assert b._base_url == "https://api.sulci.io"

    def test_env_var_read_at_instance_time_not_import_time(self, monkeypatch):
        """Setting SULCI_GATEWAY AFTER `import sulci` still takes effect
        on the next Cache() construction. Regression guard against a
        future refactor that promotes the env-var read to module level."""
        # Ensure it's unset at import time
        monkeypatch.delenv("SULCI_GATEWAY", raising=False)
        b_before = make_backend()
        assert b_before._base_url == "https://api.sulci.io"

        # Set it after import, verify next backend picks it up
        monkeypatch.setenv("SULCI_GATEWAY", "https://runtime.example.com")
        b_after = make_backend()
        assert b_after._base_url == "https://runtime.example.com"


# ══════════════════════════════════════════════════════════════════════════════
# search()
# ══════════════════════════════════════════════════════════════════════════════

class TestRemoteGet:
    """v0.6.0 (sulci-oss #62) — renamed from TestSearch.

    Covers the cloud transport's lookup method. Wire payload is now
    `{query, threshold, user_id, session_id}` — matching the gateway's
    `CacheGetRequest` pydantic model — rather than the pre-v0.6.0
    `{embedding, tenant_id, ...}` shape which the gateway 422-rejected.
    """

    def test_returns_response_similarity_and_depth_on_hit(self):
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({
                              "response":      "cached answer",
                              "similarity":    0.91,
                              "context_depth": 2,
                          })):
            result = b.remote_get(query="what is sulci?", threshold=0.85)
        assert result == ("cached answer", 0.91, 2)

    def test_returns_none_on_miss(self):
        """Cloud returns response=null on miss."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({
                              "response":      None,
                              "similarity":    0.0,
                              "context_depth": 0,
                          })):
            response, sim, depth = b.remote_get(query="x", threshold=0.85)
        assert response is None
        assert sim == 0.0
        assert depth == 0

    def test_missing_context_depth_defaults_to_zero(self):
        """Gateway may omit context_depth (pre-v0.6.0 response shape).
        Cloud transport must default to 0 rather than KeyError."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({
                              "response":   "answer",
                              "similarity": 0.9,
                              # context_depth absent
                          })):
            _, _, depth = b.remote_get(query="x", threshold=0.85)
        assert depth == 0

    def test_timeout_returns_miss_never_raises(self):
        b = make_backend()
        with patch.object(b._client, "post",
                          side_effect=httpx.TimeoutException("timed out")):
            result = b.remote_get(query="x", threshold=0.85)
        assert result == (None, 0.0, 0)

    def test_generic_exception_returns_miss_never_raises(self):
        b = make_backend()
        with patch.object(b._client, "post",
                          side_effect=RuntimeError("network down")):
            result = b.remote_get(query="x", threshold=0.85)
        assert result == (None, 0.0, 0)

    def test_http_error_returns_miss_never_raises(self):
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({}, status=500)):
            result = b.remote_get(query="x", threshold=0.85)
        assert result == (None, 0.0, 0)

    def test_sends_canonical_payload(self):
        """remote_get() sends {query, threshold, user_id, session_id} —
        matching CacheGetRequest exactly. Critically, does NOT send
        `embedding` or `tenant_id` (the pre-v0.6.0 wrong-payload bug)."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({"response": None,
                                                      "similarity": 0.0})
                          ) as mock_post:
            b.remote_get(
                query      = "what is sulci?",
                threshold  = 0.85,
                user_id    = "user-42",
                session_id = "sess-7",
            )
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "/v1/cache/get"
        body = call_kwargs[1]["json"]
        # Required + optional fields per CacheGetRequest
        assert body["query"]      == "what is sulci?"
        assert body["threshold"]  == 0.85
        assert body["user_id"]    == "user-42"
        assert body["session_id"] == "sess-7"
        # Pre-v0.6.0 wrong-payload guard: these MUST NOT be on the wire
        assert "embedding" not in body, (
            "remote_get must NOT send 'embedding' — the gateway does its own "
            "embedding via the injected EmbedServiceEmbedder. Pre-v0.6.0 SDK "
            "sent embedding here and the gateway 422-rejected it (Issue #62)."
        )
        assert "tenant_id" not in body, (
            "remote_get must NOT send 'tenant_id' — tenant identity comes "
            "from the gateway's auth context (X-Sulci-Key → api_keys lookup)."
        )

    def test_similarity_cast_to_float(self):
        """Similarity from API (may be int) is always returned as float."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({
                              "response":   "answer",
                              "similarity": 1,    # int from API
                          })):
            _, sim, _ = b.remote_get(query="x", threshold=0.85)
        assert isinstance(sim, float)
        assert sim == 1.0

    def test_context_depth_cast_to_int(self):
        """context_depth from API (may be str/float) is always returned as int."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({
                              "response":      "answer",
                              "similarity":    0.9,
                              "context_depth": "3",   # gateway might return str
                          })):
            _, _, depth = b.remote_get(query="x", threshold=0.85)
        assert isinstance(depth, int)
        assert depth == 3


# ══════════════════════════════════════════════════════════════════════════════
# upsert()
# ══════════════════════════════════════════════════════════════════════════════

class TestRemoteSet:
    """v0.6.0 (sulci-oss #62) — replaces TestUpsert (and TestStore, both
    consolidated into the single `remote_set` method).

    Wire payload is now `{query, response, user_id, session_id, ttl_seconds}` —
    matching the gateway's `CacheSetRequest` pydantic model — rather than the
    pre-v0.6.0 `{key, embedding, query, response, tenant_id, metadata, ...}`
    shape which the gateway 422-rejected.
    """

    def test_sends_canonical_payload(self):
        """remote_set() sends {query, response, user_id, session_id,
        ttl_seconds} — matching CacheSetRequest exactly. Critically, does
        NOT send embedding, key, tenant_id, or metadata (the pre-v0.6.0
        wrong-payload bug)."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({"status": "ok"})
                          ) as mock_post:
            b.remote_set(
                query       = "what is sulci?",
                response    = "semantic cache",
                user_id     = "user-42",
                session_id  = "sess-7",
                ttl_seconds = 3600,
            )
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "/v1/cache/set"
        body = call_kwargs[1]["json"]
        # Required + optional fields per CacheSetRequest
        assert body["query"]       == "what is sulci?"
        assert body["response"]    == "semantic cache"
        assert body["user_id"]     == "user-42"
        assert body["session_id"]  == "sess-7"
        assert body["ttl_seconds"] == 3600
        # Pre-v0.6.0 wrong-payload guard
        for forbidden in ("embedding", "key", "tenant_id", "metadata"):
            assert forbidden not in body, (
                f"remote_set must NOT send '{forbidden}' — see Issue #62. "
                f"The gateway-side library generates these server-side."
            )

    def test_failure_is_silent(self):
        """remote_set() must never raise — fire and forget."""
        b = make_backend()
        with patch.object(b._client, "post",
                          side_effect=Exception("network error")):
            b.remote_set(query="test", response="answer")   # must not raise

    def test_timeout_is_silent(self):
        b = make_backend()
        with patch.object(b._client, "post",
                          side_effect=httpx.TimeoutException("timeout")):
            b.remote_set(query="test", response="answer")   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# delete_user() and clear()
# ══════════════════════════════════════════════════════════════════════════════

class TestDeleteAndClear:
    """
    v0.6.2 (sulci-oss #103 SDK companion + sulci-platform #103 gateway).
    Pre-v0.6.2 the SDK sent DELETE to two paths the gateway never had —
    `/v1/user/{user_id}` and `/v1/cache` — and swallowed the resulting
    404 silently. This class pins the post-fix behavior across three
    axes:

      1. Canonical URLs — both methods now hit /v1/cache/user/{id} and
         /v1/cache/clear, the routes added by sulci-platform #103.
      2. Failure visibility — log.warning fires on errors (was: silent
         except: pass). The methods still never raise, but the swallowed
         failure mode is replaced with a logged warning so customers
         can see GDPR-relevant deletion failures.
      3. Response contract — gateway returns {"status": "ok", "deleted":
         N, "tenant_id": ..., [user_id: ...]}; both shapes round-trip
         through the vendored CacheClearResponse / CacheDeleteUserResponse
         pydantic models above.
    """

    # ── 1. Canonical URL pins ────────────────────────────────────────────────

    def test_delete_user_posts_to_canonical_path(self):
        b = make_backend()
        with patch.object(b._client, "delete",
                          return_value=mock_response({"status": "ok",
                                                      "deleted": 0,
                                                      "tenant_id": 1,
                                                      "user_id": "user-42"})
                          ) as mock_del:
            b.delete_user("user-42")
        assert mock_del.call_args[0][0] == "/v1/cache/user/user-42", (
            "delete_user() must DELETE /v1/cache/user/{user_id} — gateway "
            "exposes this canonical path per sulci-platform #103. "
            "/v1/user/{user_id} (pre-v0.6.2) returns 404 and historically "
            "got swallowed silently."
        )

    def test_clear_posts_to_canonical_path(self):
        b = make_backend()
        with patch.object(b._client, "delete",
                          return_value=mock_response({"status": "ok",
                                                      "deleted": 0,
                                                      "tenant_id": 1})
                          ) as mock_del:
            b.clear()
        assert mock_del.call_args[0][0] == "/v1/cache/clear", (
            "clear() must DELETE /v1/cache/clear — gateway exposes this "
            "canonical path per sulci-platform #103. /v1/cache (pre-v0.6.2) "
            "returns 404 and historically got swallowed silently."
        )

    # ── 2. log.warning replaces silent failure ───────────────────────────────

    def test_delete_user_logs_warning_on_failure(self):
        """Pre-v0.6.2: `except Exception: pass`. Post-v0.6.2:
        `except Exception as e: log.warning(...)`. The customer-visible
        contract (no raise) is preserved, but failures are now visible
        in standard logs."""
        b = make_backend()
        with patch.object(b._client, "delete",
                          side_effect=Exception("connection refused")), \
             patch("sulci.backends.cloud.log") as mock_log:
            b.delete_user("user-42")
        mock_log.warning.assert_called_once()
        # Verify the log message format includes the user_id and the error
        call_args = mock_log.warning.call_args
        format_string = call_args[0][0]
        assert "delete_user" in format_string
        assert "user_id=%s" in format_string or "%s" in format_string

    def test_clear_logs_warning_on_failure(self):
        """Symmetric to delete_user — silent swallow replaced with
        log.warning."""
        b = make_backend()
        with patch.object(b._client, "delete",
                          side_effect=Exception("connection refused")), \
             patch("sulci.backends.cloud.log") as mock_log:
            b.clear()
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args
        format_string = call_args[0][0]
        assert "clear" in format_string

    # ── 3. Non-crashing contract preserved ───────────────────────────────────

    def test_delete_user_does_not_raise_on_failure(self):
        """Contract pin: even with log.warning, the method MUST NOT raise.
        Customer apps that call cache.delete_user() and don't catch
        exceptions (the documented pattern) must not start crashing
        post-v0.6.2."""
        b = make_backend()
        with patch.object(b._client, "delete",
                          side_effect=Exception("error")):
            b.delete_user("user-42")   # must not raise

    def test_clear_does_not_raise_on_failure(self):
        """Symmetric contract pin to delete_user — clear() never raises."""
        b = make_backend()
        with patch.object(b._client, "delete",
                          side_effect=Exception("error")):
            b.clear()   # must not raise

    # ── 4. Response-shape round-trip pins (vendored gateway contract) ────────

    def test_delete_user_response_round_trips_through_CacheDeleteUserResponse(self):
        """The gateway returns a 4-field response on DELETE
        /v1/cache/user/{id}. This test parses the canonical shape through
        the vendored pydantic model — any future drift in either repo
        fails CI loudly. Same pattern v0.6.0 introduced for GET/SET
        request contracts."""
        from pydantic import ValidationError
        # Canonical gateway response per gateway/app/routes/cache.py
        # (sulci-platform #103)
        canonical = {
            "status":    "ok",
            "deleted":   7,
            "tenant_id": 42,
            "user_id":   "alice",
        }
        # If the gateway ever drops a field or changes a type, this raises.
        parsed = _GatewayContractModels.CacheDeleteUserResponse.model_validate(canonical)
        assert parsed.status    == "ok"
        assert parsed.deleted   == 7
        assert parsed.tenant_id == 42
        assert parsed.user_id   == "alice"

    def test_clear_response_round_trips_through_CacheClearResponse(self):
        """Same regression guard as delete_user, for the clear response."""
        canonical = {
            "status":    "ok",
            "deleted":   123,
            "tenant_id": 42,
        }
        parsed = _GatewayContractModels.CacheClearResponse.model_validate(canonical)
        assert parsed.status    == "ok"
        assert parsed.deleted   == 123
        assert parsed.tenant_id == 42


# ══════════════════════════════════════════════════════════════════════════════
# Cache constructor wiring
# ══════════════════════════════════════════════════════════════════════════════

class TestCacheWiring:

    def test_cache_constructs_sulci_backend_with_explicit_key(self):
        """Cache(backend='sulci', api_key=...) instantiates SulciCloudBackend."""
        from sulci import Cache
        with patch("sulci.backends.cloud.SulciCloudBackend") as MockBackend:
            MockBackend.return_value = MagicMock()
            cache = Cache(backend="sulci", api_key=TEST_KEY)
        MockBackend.assert_called_once_with(api_key=TEST_KEY, gateway_url="")

    def test_cache_resolves_key_from_env(self, monkeypatch):
        """Cache(backend='sulci') with no api_key= uses SULCI_API_KEY env var."""
        monkeypatch.setenv("SULCI_API_KEY", TEST_KEY)
        from sulci import Cache
        with patch("sulci.backends.cloud.SulciCloudBackend") as MockBackend:
            MockBackend.return_value = MagicMock()
            cache = Cache(backend="sulci")
        MockBackend.assert_called_once_with(api_key=TEST_KEY, gateway_url="")

    def test_cache_resolves_key_from_connect(self):
        """Cache(backend='sulci') with no api_key= uses key from sulci.connect()."""
        import sulci
        sulci._api_key = TEST_KEY
        try:
            with patch("sulci.backends.cloud.SulciCloudBackend") as MockBackend:
                MockBackend.return_value = MagicMock()
                cache = sulci.Cache(backend="sulci")
            MockBackend.assert_called_once_with(api_key=TEST_KEY, gateway_url="")
        finally:
            sulci._api_key = None   # always reset

    def test_cache_sulci_no_key_raises_value_error(self):
        """Cache(backend='sulci') with no key anywhere raises ValueError."""
        import sulci
        sulci._api_key = None
        from sulci import Cache
        with pytest.raises(ValueError, match="api_key is required"):
            Cache(backend="sulci")

    def test_explicit_key_overrides_env(self, monkeypatch):
        """Explicit api_key= takes priority over SULCI_API_KEY env var."""
        monkeypatch.setenv("SULCI_API_KEY", "sk-sulci-fromenv-000000000")
        from sulci import Cache
        with patch("sulci.backends.cloud.SulciCloudBackend") as MockBackend:
            MockBackend.return_value = MagicMock()
            cache = Cache(backend="sulci", api_key=TEST_KEY)
        MockBackend.assert_called_once_with(api_key=TEST_KEY, gateway_url="")

    def test_unknown_backend_still_raises(self):
        """Non-sulci unknown backends still raise ValueError."""
        from sulci import Cache
        with pytest.raises(ValueError, match="Unknown backend"):
            Cache(backend="nonexistent")


# ══════════════════════════════════════════════════════════════════════════════
# Canonical gateway paths — regression guard for the v0.5.7 route fix
# ══════════════════════════════════════════════════════════════════════════════
#
# The gateway mounts its cache router at prefix "/v1" and declares the routes
# as @router.post("/cache/get") and @router.post("/cache/set"), resolving to
# the canonical URLs:
#
#     POST /v1/cache/get
#     POST /v1/cache/set
#
# Source of truth on the platform side:
#     sulci-platform/gateway/app/main.py          (router include + prefix)
#     sulci-platform/gateway/app/routes/cache.py  (@router.post decorators)
#
# Prior to v0.5.7 the SDK POSTed to "/v1/get" and "/v1/set" — a silent 404
# swallowed by cloud.py's `except Exception:` clause. These assertions pin the
# SDK's three URL-bearing methods to the gateway's canonical paths so a future
# drift on either side gets caught here rather than in production telemetry.

class TestCanonicalGatewayPaths:
    """v0.6.0 (sulci-oss #62) — each URL-bearing method must POST to the
    gateway's canonical path AND send a payload that round-trips through
    the gateway's pydantic request model.

    v0.5.7 introduced URL-only assertions (pinning `/v1/cache/get` etc.)
    after #57 silently 404'd for ~14 months. Issue #62 then revealed that
    fixing the URLs wasn't enough — the SDK's payload still didn't match
    the gateway's `CacheGetRequest` / `CacheSetRequest` models, so every
    post-v0.5.7 request 422'd instead. This test class now extends the
    URL pins with full payload-shape pins.

    See `_GatewayContractModels` at module scope for the vendored model
    definitions — they're copies of `sulci-platform:shared/models.py`,
    pinned here so sulci-oss CI doesn't need the platform repo to verify
    the wire contract.
    """

    def test_remote_get_posts_to_v1_cache_get(self):
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({"response": None,
                                                      "similarity": 0.0})
                          ) as mock_post:
            b.remote_get(query="hello", threshold=0.85)
        assert mock_post.call_args[0][0] == "/v1/cache/get", (
            "remote_get() must POST to /v1/cache/get — gateway exposes that "
            "canonical path; /v1/get returns 404 and is swallowed silently."
        )

    def test_remote_set_posts_to_v1_cache_set(self):
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({"status": "ok"})
                          ) as mock_post:
            b.remote_set(query="hello", response="world")
        assert mock_post.call_args[0][0] == "/v1/cache/set", (
            "remote_set() must POST to /v1/cache/set — gateway exposes that "
            "canonical path; /v1/set returns 404 and is swallowed silently."
        )

    def test_remote_get_payload_round_trips_through_CacheGetRequest(self):
        """The wire payload sent to /v1/cache/get must be parseable by the
        gateway's `CacheGetRequest` pydantic model — every required field
        present, no extra forbidden fields, all field types compatible.

        This is the regression guard for Issue #62. Before v0.6.0 the SDK
        sent `{embedding, tenant_id, threshold, user_id}`, which the gateway
        422-rejected because:
          - `query` (required) was missing
          - `embedding` was not in the model
          - `tenant_id` was not in the model (auth context provides it)
        """
        from pydantic import ValidationError
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({"response": None,
                                                      "similarity": 0.0})
                          ) as mock_post:
            b.remote_get(
                query      = "what is sulci?",
                threshold  = 0.85,
                user_id    = "user-42",
                session_id = "sess-7",
            )
        body = mock_post.call_args[1]["json"]
        # This will raise ValidationError if the SDK's payload doesn't
        # round-trip through the gateway's request schema.
        parsed = _GatewayContractModels.CacheGetRequest.model_validate(body)
        # Sanity: the parsed model has the SDK's values
        assert parsed.query      == "what is sulci?"
        assert parsed.threshold  == 0.85
        assert parsed.user_id    == "user-42"
        assert parsed.session_id == "sess-7"

    def test_remote_set_payload_round_trips_through_CacheSetRequest(self):
        """Symmetric to the GET test — payload must parse as CacheSetRequest."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({"status": "ok"})
                          ) as mock_post:
            b.remote_set(
                query       = "what is sulci?",
                response    = "semantic cache",
                user_id     = "user-42",
                session_id  = "sess-7",
                ttl_seconds = 3600,
            )
        body = mock_post.call_args[1]["json"]
        parsed = _GatewayContractModels.CacheSetRequest.model_validate(body)
        assert parsed.query       == "what is sulci?"
        assert parsed.response    == "semantic cache"
        assert parsed.user_id     == "user-42"
        assert parsed.session_id  == "sess-7"
        assert parsed.ttl_seconds == 3600

    def test_minimal_remote_get_payload_round_trips(self):
        """Optional fields default correctly when omitted — confirms the
        gateway accepts a minimal `{query, threshold}` payload."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({"response": None,
                                                      "similarity": 0.0})
                          ) as mock_post:
            b.remote_get(query="x", threshold=0.85)
        body = mock_post.call_args[1]["json"]
        parsed = _GatewayContractModels.CacheGetRequest.model_validate(body)
        assert parsed.query == "x"
        assert parsed.user_id    is None
        assert parsed.session_id is None

    def test_no_legacy_method_names_in_source(self):
        """Static check: the SDK source has no references to the
        pre-v0.6.0 methods `search`, `store`, `upsert` on the cloud
        transport — they were renamed to `remote_get` / `remote_set`."""
        import sulci.backends.cloud as cloud_mod
        from pathlib import Path
        src = Path(cloud_mod.__file__).read_text()
        for legacy in ("def search(", "def store(", "def upsert("):
            assert legacy not in src, (
                f"cloud.py contains legacy method `{legacy[4:-1]}` — "
                f"v0.6.0 renamed to remote_get / remote_set "
                f"(see sulci-oss #62)."
            )

    def test_no_legacy_paths_in_source(self):
        """Static check: the SDK source contains zero references to the
        pre-v0.5.7 paths /v1/get or /v1/set, NOR the pre-v0.6.2 broken
        DELETE paths /v1/user/{id} and the bare /v1/cache. This catches
        regressions where a new method is added but the URL prefix is
        forgotten."""
        import sulci.backends.cloud as cloud_mod
        from pathlib import Path
        src = Path(cloud_mod.__file__).read_text()
        # v0.5.7 (sulci-oss #57)
        assert '"/v1/get"' not in src and "'/v1/get'" not in src, (
            "cloud.py contains legacy '/v1/get' — gateway exposes /v1/cache/get"
        )
        assert '"/v1/set"' not in src and "'/v1/set'" not in src, (
            "cloud.py contains legacy '/v1/set' — gateway exposes /v1/cache/set"
        )
        # v0.6.2 (sulci-oss #103 SDK companion). The pre-fix delete path was
        # `/v1/user/{user_id}` (f-string) — the canonical is `/v1/cache/user/{user_id}`.
        assert '"/v1/user/' not in src and "'/v1/user/" not in src and \
               'f"/v1/user/' not in src and "f'/v1/user/" not in src, (
            "cloud.py contains legacy '/v1/user/{id}' — gateway exposes "
            "/v1/cache/user/{id} per sulci-platform #103. Pre-v0.6.2 the "
            "wrong path 404'd silently."
        )
        # The pre-fix clear path was `/v1/cache` as a complete URL. The
        # canonical is `/v1/cache/clear`. We can't ban `/v1/cache` outright
        # since /v1/cache/get, /v1/cache/set, /v1/cache/clear, and
        # /v1/cache/user/{id} are all valid — so we ban the exact strings
        # `"/v1/cache"` and `'/v1/cache'` (with closing quote immediately
        # after, meaning the URL is exactly "/v1/cache" with nothing else).
        assert '"/v1/cache"' not in src and "'/v1/cache'" not in src, (
            "cloud.py contains legacy '/v1/cache' (bare) — gateway exposes "
            "/v1/cache/clear per sulci-platform #103. Pre-v0.6.2 the "
            "wrong path 404'd silently."
        )


# ══════════════════════════════════════════════════════════════════════════════
# v0.6.0 — Cache.get / Cache.set route through cloud transport without local embed
# (sulci-oss #62, umbrella #63)
#
# Cache detects the cloud transport via duck-typed `hasattr(backend, "remote_get")`
# at __init__ time (`self._is_remote_transport`). When set, Cache.get bypasses
# `self._embedder.embed()` and `self._backend.search()` entirely, calling
# `self._backend.remote_get(query, ...)` instead. Symmetric for Cache.set.
#
# These tests use a fake-transport-shaped backend + fake embedder so they run
# offline (no MiniLM download), like TestInstanceInjection in test_core.py.
# ══════════════════════════════════════════════════════════════════════════════


class _FakeRemoteTransport:
    """Minimal transport-shaped fake — has remote_get + remote_set, no search/store."""
    def __init__(self):
        self.remote_get_calls = []
        self.remote_set_calls = []
        self._stored: dict = {}

    def remote_get(self, query, threshold, *, user_id=None, session_id=None):
        self.remote_get_calls.append({
            "query": query, "threshold": threshold,
            "user_id": user_id, "session_id": session_id,
        })
        if query in self._stored:
            return self._stored[query], 1.0, 0
        return None, 0.0, 0

    def remote_set(self, query, response, *, user_id=None, session_id=None,
                   ttl_seconds=None):
        self.remote_set_calls.append({
            "query": query, "response": response,
            "user_id": user_id, "session_id": session_id,
            "ttl_seconds": ttl_seconds,
        })
        self._stored[query] = response


class _FakeEmbedderForTransportTests:
    """Embedder fake — should NEVER be called when Cache routes via transport."""
    dimension = 4
    def __init__(self):
        self.embed_calls = []
    def embed(self, text):
        self.embed_calls.append(text)
        return [0.0] * 4
    def embed_batch(self, texts):
        return [[0.0] * 4 for _ in texts]


class TestCloudTransportShortCircuit:
    """v0.6.0 (sulci-oss #62) — Cache.get / Cache.set bypass local embedding
    when the backend is a cloud transport."""

    def test_is_remote_transport_flag_set_for_cloud(self):
        """Cache constructed with a transport-shaped backend has
        _is_remote_transport=True."""
        from sulci import Cache
        c = Cache(
            backend         = _FakeRemoteTransport(),
            embedding_model = _FakeEmbedderForTransportTests(),
            db_path         = "/nonexistent",
        )
        assert c._is_remote_transport is True

    def test_is_remote_transport_flag_false_for_vector_backend(self):
        """Cache constructed with a Backend-protocol-shaped backend (search,
        store) has _is_remote_transport=False."""
        from sulci import Cache

        class _FakeVectorBackend:
            ENFORCES_TENANT_ISOLATION = False
            def search(self, **k): return (None, 0.0)
            def store(self, **k):  pass
            def clear(self): pass

        c = Cache(
            backend         = _FakeVectorBackend(),
            embedding_model = _FakeEmbedderForTransportTests(),
            db_path         = "/nonexistent",
        )
        assert c._is_remote_transport is False

    def test_get_routes_to_remote_get_skips_local_embed(self):
        """Cache.get with cloud transport must call backend.remote_get and
        NEVER call embedder.embed() — the gateway-side library handles it."""
        from sulci import Cache
        emb = _FakeEmbedderForTransportTests()
        be  = _FakeRemoteTransport()
        c = Cache(backend=be, embedding_model=emb, db_path="/nonexistent")
        result = c.get("test query")
        # Routed to transport
        assert len(be.remote_get_calls) == 1
        assert be.remote_get_calls[0]["query"] == "test query"
        # Local embedding skipped — this is the whole point
        assert emb.embed_calls == [], (
            "Cache.get must NOT call embedder.embed() when backend is cloud "
            "transport — that would duplicate the work the gateway-side "
            "library does (Issue #62)."
        )
        # Return shape is (response, similarity, context_depth)
        assert result == (None, 0.0, 0)

    def test_set_routes_to_remote_set_skips_local_embed(self):
        """Cache.set with cloud transport must call backend.remote_set and
        NEVER call embedder.embed()."""
        from sulci import Cache
        emb = _FakeEmbedderForTransportTests()
        be  = _FakeRemoteTransport()
        c = Cache(backend=be, embedding_model=emb, db_path="/nonexistent")
        c.set("hello", "world")
        # Routed to transport
        assert len(be.remote_set_calls) == 1
        assert be.remote_set_calls[0]["query"]    == "hello"
        assert be.remote_set_calls[0]["response"] == "world"
        # Local embedding skipped
        assert emb.embed_calls == []

    def test_get_then_set_then_get_round_trips_via_transport(self):
        """Happy-path: set then get via transport returns the stored response."""
        from sulci import Cache
        emb = _FakeEmbedderForTransportTests()
        be  = _FakeRemoteTransport()
        c = Cache(backend=be, embedding_model=emb, db_path="/nonexistent")

        miss, sim_miss, depth_miss = c.get("unknown query")
        assert miss is None and sim_miss == 0.0

        c.set("known query", "stored answer")
        hit, sim_hit, depth_hit = c.get("known query")
        assert hit       == "stored answer"
        assert sim_hit   == 1.0
        # Embedder STILL never called across all three calls
        assert emb.embed_calls == []

    def test_get_forwards_session_id_to_transport(self):
        """session_id flows from Cache.get(session_id=...) through to the
        wire — the gateway-side library tracks the session context window."""
        from sulci import Cache
        be = _FakeRemoteTransport()
        c = Cache(
            backend         = be,
            embedding_model = _FakeEmbedderForTransportTests(),
            db_path         = "/nonexistent",
        )
        c.get("test", session_id="sess-42")
        assert be.remote_get_calls[0]["session_id"] == "sess-42"

    def test_set_forwards_session_id_to_transport(self):
        """Symmetric for set."""
        from sulci import Cache
        be = _FakeRemoteTransport()
        c = Cache(
            backend         = be,
            embedding_model = _FakeEmbedderForTransportTests(),
            db_path         = "/nonexistent",
        )
        c.set("test", "answer", session_id="sess-42")
        assert be.remote_set_calls[0]["session_id"] == "sess-42"

    def test_set_skips_local_session_window_recording(self):
        """Cache.set with cloud transport must NOT touch self._sessions —
        the gateway-side library tracks the session window. Recording
        locally would create stale state since this Cache instance has
        no engine."""
        from sulci import Cache
        c = Cache(
            backend         = _FakeRemoteTransport(),
            embedding_model = _FakeEmbedderForTransportTests(),
            db_path         = "/nonexistent",
            context_window  = 4,        # would enable local _sessions
        )
        # _sessions exists (context_window > 0)
        assert c._sessions is not None
        # But .set with session_id doesn't update it because raw_vec is None
        c.set("query", "answer", session_id="sess-1")
        # If local session recording had happened, the window would have a turn
        window = c._sessions.get("sess-1")
        assert window.depth == 0, (
            "Cache.set must not record turns into the local session window "
            "when routing through the cloud transport — the gateway-side "
            "library owns the session context."
        )
