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


def make_backend(key=TEST_KEY, timeout=5.0):
    from sulci.backends.cloud import SulciCloudBackend
    return SulciCloudBackend(api_key=key, timeout=timeout)


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


# ══════════════════════════════════════════════════════════════════════════════
# search()
# ══════════════════════════════════════════════════════════════════════════════

class TestSearch:

    def test_returns_response_and_similarity_on_hit(self):
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({
                              "response":   "cached answer",
                              "similarity": 0.91,
                          })):
            result = b.search(
                embedding=[0.1] * 384,
                threshold=0.85,
            )
        assert result == ("cached answer", 0.91)

    def test_returns_none_on_miss(self):
        """Cloud returns response=null on miss."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({
                              "response":   None,
                              "similarity": 0.0,
                          })):
            response, sim = b.search(
                embedding=[0.1] * 384,
                threshold=0.85,
            )
        assert response is None
        assert sim == 0.0

    def test_timeout_returns_miss_never_raises(self):
        """TimeoutException must be swallowed — treated as cache miss."""
        b = make_backend()
        with patch.object(b._client, "post",
                          side_effect=httpx.TimeoutException("timed out")):
            response, sim = b.search(
                embedding=[0.1] * 384,
                threshold=0.85,
            )
        assert response is None
        assert sim == 0.0

    def test_generic_exception_returns_miss_never_raises(self):
        """Any unexpected error must be swallowed."""
        b = make_backend()
        with patch.object(b._client, "post",
                          side_effect=RuntimeError("network down")):
            response, sim = b.search(
                embedding=[0.1] * 384,
                threshold=0.85,
            )
        assert response is None
        assert sim == 0.0

    def test_http_error_returns_miss_never_raises(self):
        """HTTP 5xx must be swallowed."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({}, status=500)):
            response, sim = b.search(
                embedding=[0.1] * 384,
                threshold=0.85,
            )
        assert response is None
        assert sim == 0.0

    def test_sends_correct_payload(self):
        """search() sends embedding, threshold, user_id to /v1/get."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({"response": None,
                                                      "similarity": 0.0})
                          ) as mock_post:
            b.search(
                embedding=[0.1] * 384,
                threshold=0.85,
                user_id="user-42",
            )
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "/v1/get"
        body = call_kwargs[1]["json"]
        assert body["threshold"] == 0.85
        assert body["user_id"]   == "user-42"
        assert len(body["embedding"]) == 384

    def test_similarity_cast_to_float(self):
        """Similarity from API (may be int) is always returned as float."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({
                              "response":   "answer",
                              "similarity": 1,    # int from API
                          })):
            _, sim = b.search(embedding=[0.1] * 384, threshold=0.85)
        assert isinstance(sim, float)
        assert sim == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# upsert()
# ══════════════════════════════════════════════════════════════════════════════

class TestUpsert:

    def test_sends_correct_payload(self):
        """upsert() sends embedding, query, response, user_id to /v1/set."""
        b = make_backend()
        with patch.object(b._client, "post",
                          return_value=mock_response({"status": "ok"})
                          ) as mock_post:
            b.upsert(
                embedding   = [0.1] * 384,
                query       = "what is sulci?",
                response    = "semantic cache",
                user_id     = "user-42",
                ttl_seconds = 3600,
            )
        body = mock_post.call_args[1]["json"]
        assert body["query"]       == "what is sulci?"
        assert body["response"]    == "semantic cache"
        assert body["user_id"]     == "user-42"
        assert body["ttl_seconds"] == 3600

    def test_failure_is_silent(self):
        """upsert() must never raise — fire and forget."""
        b = make_backend()
        with patch.object(b._client, "post",
                          side_effect=Exception("network error")):
            b.upsert(
                embedding=[0.1] * 384,
                query="test",
                response="answer",
            )   # must not raise

    def test_timeout_is_silent(self):
        b = make_backend()
        with patch.object(b._client, "post",
                          side_effect=httpx.TimeoutException("timeout")):
            b.upsert(
                embedding=[0.1] * 384,
                query="test",
                response="answer",
            )   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# delete_user() and clear()
# ══════════════════════════════════════════════════════════════════════════════

class TestDeleteAndClear:

    def test_delete_user_is_silent_on_failure(self):
        b = make_backend()
        with patch.object(b._client, "delete",
                          side_effect=Exception("error")):
            b.delete_user("user-42")   # must not raise

    def test_clear_is_silent_on_failure(self):
        b = make_backend()
        with patch.object(b._client, "delete",
                          side_effect=Exception("error")):
            b.clear()   # must not raise

    def test_delete_user_sends_correct_url(self):
        b = make_backend()
        with patch.object(b._client, "delete",
                          return_value=mock_response({"status": "ok"})
                          ) as mock_del:
            b.delete_user("user-42")
        assert mock_del.call_args[0][0] == "/v1/user/user-42"

    def test_clear_sends_correct_url(self):
        b = make_backend()
        with patch.object(b._client, "delete",
                          return_value=mock_response({"status": "ok"})
                          ) as mock_del:
            b.clear()
        assert mock_del.call_args[0][0] == "/v1/cache"


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
        MockBackend.assert_called_once_with(api_key=TEST_KEY)

    def test_cache_resolves_key_from_env(self, monkeypatch):
        """Cache(backend='sulci') with no api_key= uses SULCI_API_KEY env var."""
        monkeypatch.setenv("SULCI_API_KEY", TEST_KEY)
        from sulci import Cache
        with patch("sulci.backends.cloud.SulciCloudBackend") as MockBackend:
            MockBackend.return_value = MagicMock()
            cache = Cache(backend="sulci")
        MockBackend.assert_called_once_with(api_key=TEST_KEY)

    def test_cache_resolves_key_from_connect(self):
        """Cache(backend='sulci') with no api_key= uses key from sulci.connect()."""
        import sulci
        sulci._api_key = TEST_KEY
        try:
            with patch("sulci.backends.cloud.SulciCloudBackend") as MockBackend:
                MockBackend.return_value = MagicMock()
                cache = sulci.Cache(backend="sulci")
            MockBackend.assert_called_once_with(api_key=TEST_KEY)
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
        MockBackend.assert_called_once_with(api_key=TEST_KEY)

    def test_unknown_backend_still_raises(self):
        """Non-sulci unknown backends still raise ValueError."""
        from sulci import Cache
        with pytest.raises(ValueError, match="Unknown backend"):
            Cache(backend="nonexistent")
