# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
tests/test_oss_connect.py
=========================
Unit tests for the OSS-Connect device-code flow client (D12 / v0.6.0).

The module under test (``sulci.oss_connect``) is a thin RFC 8628 client
that talks to the gateway's ``/v1/oss-connect/{device-code,token}``
endpoints. These tests mock httpx so they're deterministic and fast,
and so they can simulate the various terminal states the gateway can
return without requiring a live gateway.

Coverage
--------
- Happy path: pending → authorized → returns api_key
- Slow_down: gateway 400 increments the polling interval by 5s
- Denied: 403 access_denied → RuntimeError with the gateway's error code
- Expired/not-found: 404 → RuntimeError
- Initial /device-code request fails → RuntimeError before any polling
- Transient network error during polling → keeps polling, no exception
- Deadline elapsed without authorization → RuntimeError("timed out")
- The error envelope is treated as RFC 8628 unwrapped form
  (``response.json().get("error")``) — matches gateway's D5 contract
- _safe_error_field tolerates malformed bodies / non-dict bodies
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# A "device-code" gateway response shape we'll reuse across tests. Mirrors
# what gateway/app/routes/oss_connect.py:post_device_code returns.
_DEVICE_CODE_BODY = {
    "device_code":               "fake-device-code-43-chars-padded-pad-pad-padd",
    "user_code":                 "WXYZ-2345",
    "verification_uri":          "https://dashboard.sulci.io/oss-connect",
    "verification_uri_complete": "https://dashboard.sulci.io/oss-connect?code=WXYZ-2345",
    "expires_in":                900,
    "interval":                  5,
}


def _ok(status: int, body: dict | None = None) -> MagicMock:
    """Build a fake httpx.Response with the given status + JSON body."""
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=body or {})
    # raise_for_status is a no-op on success; raises on >=400.
    if status >= 400:
        r.raise_for_status = MagicMock(side_effect=Exception(f"{status} response"))
    else:
        r.raise_for_status = MagicMock(return_value=None)
    return r


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """time.sleep would slow tests down + make them flaky. Replace with
    a no-op so the polling loop iterates as fast as possible while still
    respecting the deadline check (which uses time.time, not sleep).
    """
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_: None)


# ── Happy path ───────────────────────────────────────────────────────────────

class TestHappyPath:

    def test_pending_then_authorized_returns_api_key(self):
        """Most common shape of a successful flow: one or more 425s
        (user hasn't clicked yet) followed by a 200 with the api_key."""
        from sulci import oss_connect

        device_code_resp = _ok(200, _DEVICE_CODE_BODY)
        pending_resp     = _ok(425, {"error": "authorization_pending",
                                     "message": "User has not authorized yet"})
        authorized_resp  = _ok(200, {
            "api_key":   "sk-sulci-flow-success-abcdef",
            "tenant_id": 42,
            "plan":      "oss_connect",
            "email":     "kathir@example.com",
        })

        with patch("sulci.oss_connect.httpx.post",
                   side_effect=[device_code_resp, pending_resp, authorized_resp]):
            result = oss_connect.run_device_code_flow(
                gateway_base="https://api.sulci.io",
                sdk_version="0.6.0",
            )

        assert result == "sk-sulci-flow-success-abcdef"

    def test_first_poll_authorized_short_path(self):
        """If the user authorizes faster than the SDK's first sleep
        finishes, the very first poll returns 200. No 425 in between."""
        from sulci import oss_connect

        with patch("sulci.oss_connect.httpx.post",
                   side_effect=[
                       _ok(200, _DEVICE_CODE_BODY),
                       _ok(200, {"api_key": "sk-sulci-fast", "tenant_id": 1,
                                 "plan": "oss_connect", "email": "a@b.c"}),
                   ]):
            result = oss_connect.run_device_code_flow(
                gateway_base="https://api.sulci.io",
                sdk_version="0.6.0",
            )
        assert result == "sk-sulci-fast"


# ── Initial device-code request failures ─────────────────────────────────────

class TestInitialRequestFailure:

    def test_device_code_request_network_error_raises(self):
        """If we can't even get a device_code, fail fast with a clear
        error — there's nothing to poll for, so retrying inside this
        function is pointless."""
        from sulci import oss_connect

        with patch("sulci.oss_connect.httpx.post",
                   side_effect=ConnectionError("DNS lookup failed")):
            with pytest.raises(RuntimeError, match="could not request device code"):
                oss_connect.run_device_code_flow(
                    gateway_base="https://api.sulci.io",
                    sdk_version="0.6.0",
                )

    def test_device_code_request_5xx_raises(self):
        """Same fail-fast behavior for gateway 5xx — the rate-limit /
        Redis-down paths the gateway documents both surface as 500/429.
        No partial state to clean up."""
        from sulci import oss_connect

        bad = _ok(500, {"detail": {"error": "server_error"}})
        with patch("sulci.oss_connect.httpx.post", side_effect=[bad]):
            with pytest.raises(RuntimeError, match="could not request device code"):
                oss_connect.run_device_code_flow(
                    gateway_base="https://api.sulci.io",
                    sdk_version="0.6.0",
                )


# ── Slow_down handling ──────────────────────────────────────────────────────

class TestSlowDown:

    def test_slow_down_increments_interval(self):
        """RFC 8628 §3.5 — when the gateway returns 400 slow_down, the
        SDK SHOULD increase its polling interval by 5 seconds. We
        verify by inspecting the recorded sleep durations."""
        from sulci import oss_connect

        device_code_resp = _ok(200, _DEVICE_CODE_BODY)         # interval=5
        slow_down_resp   = _ok(400, {"error": "slow_down",
                                     "message": "Polling too aggressively"})
        authorized_resp  = _ok(200, {"api_key": "sk-sulci-slowed",
                                     "tenant_id": 1, "plan": "oss_connect",
                                     "email": "a@b.c"})

        sleep_durations = []

        def _record_sleep(d):
            sleep_durations.append(d)

        with patch("sulci.oss_connect.httpx.post",
                   side_effect=[device_code_resp, slow_down_resp, authorized_resp]), \
             patch("sulci.oss_connect.time.sleep", side_effect=_record_sleep):
            result = oss_connect.run_device_code_flow(
                gateway_base="https://api.sulci.io",
                sdk_version="0.6.0",
            )

        assert result == "sk-sulci-slowed"
        # Two sleeps total (one before each poll). Second sleep is the
        # incremented interval (5 → 10 after slow_down).
        assert sleep_durations == [5, 10]

    def test_other_400_does_not_increment_interval(self):
        """A 400 with a non-slow_down error code falls through to the
        terminal error block. Tests that the 400 short-circuit is gated
        on error == 'slow_down'."""
        from sulci import oss_connect

        # Construct a 400 with a different error — say, the gateway
        # rejected the body for some reason (this would normally be 422
        # but defense-in-depth).
        weird_400 = _ok(400, {"error": "weird_thing",
                              "message": "no idea"})

        with patch("sulci.oss_connect.httpx.post",
                   side_effect=[_ok(200, _DEVICE_CODE_BODY), weird_400]):
            # 400 with non-slow_down error doesn't match any terminal
            # branch, so the loop continues. Eventually the deadline
            # expires (we mock that immediately by patching time.time).
            with patch("sulci.oss_connect.time.time",
                       side_effect=[0, 1, 99999]):  # 99999 > deadline
                with pytest.raises(RuntimeError, match="timed out"):
                    oss_connect.run_device_code_flow(
                        gateway_base="https://api.sulci.io",
                        sdk_version="0.6.0",
                    )


# ── Terminal error states ───────────────────────────────────────────────────

class TestTerminalErrors:

    @pytest.mark.parametrize("status,error", [
        (403, "access_denied"),
        (404, "invalid_token"),
        (410, "expired_token"),
    ])
    def test_terminal_status_raises_runtime_error(self, status, error):
        """403/404/410 are all terminal — surface as RuntimeError with
        the gateway's error code in the message."""
        from sulci import oss_connect

        terminal_resp = _ok(status, {"error": error, "message": "..."})

        with patch("sulci.oss_connect.httpx.post",
                   side_effect=[_ok(200, _DEVICE_CODE_BODY), terminal_resp]):
            with pytest.raises(RuntimeError, match=error):
                oss_connect.run_device_code_flow(
                    gateway_base="https://api.sulci.io",
                    sdk_version="0.6.0",
                )

    def test_terminal_with_unparseable_body_uses_unknown(self):
        """If the gateway returns a terminal status but the body isn't
        parseable JSON, the error message should contain 'unknown'
        rather than crashing."""
        from sulci import oss_connect

        # Build a response where .json() raises.
        bad = MagicMock()
        bad.status_code = 403
        bad.json = MagicMock(side_effect=ValueError("not JSON"))
        bad.raise_for_status = MagicMock(side_effect=Exception("403"))

        with patch("sulci.oss_connect.httpx.post",
                   side_effect=[_ok(200, _DEVICE_CODE_BODY), bad]):
            with pytest.raises(RuntimeError, match="unknown"):
                oss_connect.run_device_code_flow(
                    gateway_base="https://api.sulci.io",
                    sdk_version="0.6.0",
                )


# ── Polling-loop resilience ─────────────────────────────────────────────────

class TestPollingResilience:

    def test_transient_poll_network_error_continues(self):
        """A network error during polling is transient — the loop
        keeps going. Only the *initial* device-code request raises on
        network error; once we have a device_code, the right thing is
        to keep trying until the 15-minute deadline."""
        from sulci import oss_connect

        # Sequence: device_code → network error → 425 → authorized
        device_code_resp = _ok(200, _DEVICE_CODE_BODY)
        authorized_resp  = _ok(200, {"api_key": "sk-sulci-resilient",
                                     "tenant_id": 1, "plan": "oss_connect",
                                     "email": "a@b.c"})

        post_calls = [
            device_code_resp,                      # 1st: initial request
            ConnectionError("temporary glitch"),   # 2nd: poll fails
            _ok(425, {"error": "authorization_pending"}),  # 3rd: pending
            authorized_resp,                       # 4th: success
        ]

        with patch("sulci.oss_connect.httpx.post", side_effect=post_calls):
            result = oss_connect.run_device_code_flow(
                gateway_base="https://api.sulci.io",
                sdk_version="0.6.0",
            )
        assert result == "sk-sulci-resilient"


# ── Deadline / timeout ──────────────────────────────────────────────────────

class TestDeadline:

    def test_deadline_elapsed_raises(self):
        """If the user never authorizes, the loop exits when the
        15-minute deadline (expires_in) elapses. We simulate by mocking
        time.time so the deadline check fails immediately after the
        first poll."""
        from sulci import oss_connect

        with patch("sulci.oss_connect.httpx.post",
                   return_value=_ok(200, _DEVICE_CODE_BODY)) as mocked_post, \
             patch("sulci.oss_connect.time.time",
                   side_effect=[0, 99999]):  # 0 = now, 99999 = past deadline
            with pytest.raises(RuntimeError, match="timed out"):
                oss_connect.run_device_code_flow(
                    gateway_base="https://api.sulci.io",
                    sdk_version="0.6.0",
                )
        # Initial request was made; loop exited before any token poll.
        assert mocked_post.call_count == 1


# ── _safe_error_field helper ────────────────────────────────────────────────

class TestSafeErrorField:
    """The helper that extracts ``error`` from a response body. Defensive
    against hostile / malformed responses so a bad gateway response never
    crashes the user's interactive ``sulci.connect()`` call."""

    def test_returns_error_string_when_present(self):
        from sulci import oss_connect
        r = MagicMock()
        r.json = MagicMock(return_value={"error": "access_denied"})
        assert oss_connect._safe_error_field(r) == "access_denied"

    def test_returns_none_when_body_not_json(self):
        from sulci import oss_connect
        r = MagicMock()
        r.json = MagicMock(side_effect=ValueError("not JSON"))
        assert oss_connect._safe_error_field(r) is None

    def test_returns_none_when_body_is_list(self):
        """Hostile body that's valid JSON but not a dict."""
        from sulci import oss_connect
        r = MagicMock()
        r.json = MagicMock(return_value=["not", "a", "dict"])
        assert oss_connect._safe_error_field(r) is None

    def test_returns_none_when_no_error_field(self):
        from sulci import oss_connect
        r = MagicMock()
        r.json = MagicMock(return_value={"message": "something else"})
        assert oss_connect._safe_error_field(r) is None

    def test_returns_none_when_error_is_not_string(self):
        from sulci import oss_connect
        r = MagicMock()
        r.json = MagicMock(return_value={"error": 42})
        assert oss_connect._safe_error_field(r) is None


# ── URLs and request shape ──────────────────────────────────────────────────

class TestRequestShape:
    """Verify the SDK constructs requests the gateway expects."""

    def test_device_code_request_uses_correct_url_and_body(self):
        """Initial request hits /v1/oss-connect/device-code with
        sdk_version + client_name."""
        from sulci import oss_connect

        with patch("sulci.oss_connect.httpx.post",
                   side_effect=[
                       _ok(200, _DEVICE_CODE_BODY),
                       _ok(200, {"api_key": "k", "tenant_id": 1,
                                 "plan": "oss_connect", "email": "a@b.c"}),
                   ]) as mock_post:
            oss_connect.run_device_code_flow(
                gateway_base="https://staging.example.com",
                sdk_version="9.9.9",
            )

        # First call: device-code request
        first_call = mock_post.call_args_list[0]
        assert first_call.args[0] == "https://staging.example.com/v1/oss-connect/device-code"
        assert first_call.kwargs["json"] == {
            "sdk_version": "9.9.9",
            "client_name": "sulci-python",
        }

    def test_token_poll_uses_correct_url_and_body(self):
        from sulci import oss_connect

        with patch("sulci.oss_connect.httpx.post",
                   side_effect=[
                       _ok(200, _DEVICE_CODE_BODY),
                       _ok(200, {"api_key": "k", "tenant_id": 1,
                                 "plan": "oss_connect", "email": "a@b.c"}),
                   ]) as mock_post:
            oss_connect.run_device_code_flow(
                gateway_base="https://staging.example.com",
                sdk_version="0.6.0",
            )

        # Second call: token poll
        poll_call = mock_post.call_args_list[1]
        assert poll_call.args[0] == "https://staging.example.com/v1/oss-connect/token"
        assert poll_call.kwargs["json"] == {
            "device_code": _DEVICE_CODE_BODY["device_code"],
            "grant_type":  oss_connect.GRANT_TYPE,
        }
        # The grant_type literal must match exactly what the gateway's
        # Pydantic Literal expects.
        assert oss_connect.GRANT_TYPE == "urn:ietf:params:oauth:grant-type:device_code"
