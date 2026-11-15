"""
tests/test_telemetry_gateway_override.py
========================================
Regression tests for the v0.5.5 fix that makes ``SULCI_GATEWAY`` actually
redirect telemetry POSTs.

Background
----------
In v0.5.4 and earlier, ``sulci/__init__.py`` defined::

    _TELEMETRY_URL = "https://api.sulci.io/v1/telemetry"   # hardcoded literal
    ...
    _GATEWAY_BASE  = os.environ.get("SULCI_GATEWAY", "https://api.sulci.io")...

The comment above ``_GATEWAY_BASE`` claimed a test/local-dev gateway override
via ``SULCI_GATEWAY`` — but ``_TELEMETRY_URL`` (which ``_post()`` actually
uses) was a separate literal that ignored the env var entirely. This
broke smoke tests where the SDK is supposed to send telemetry to a
non-prod gateway (e.g. a local or self-hosted one).

Fix
---
v0.5.5 derives ``_TELEMETRY_URL = f"{_GATEWAY_BASE}/v1/telemetry"`` so a
single env var redirects both the device-code flow and the telemetry
pipeline.

Test strategy
-------------
``_TELEMETRY_URL`` and ``_GATEWAY_BASE`` are computed at module import
time, so changing the env var after ``import sulci`` has no effect on
already-resolved values. Each test purges the ``sulci`` module from
``sys.modules`` and re-imports it after setting (or clearing) the env
var, so we observe the values as a freshly-started process would.
"""
from __future__ import annotations

import importlib
import sys

import pytest
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reimport_sulci(monkeypatch, gateway_value):
    """Re-import ``sulci`` after applying ``SULCI_GATEWAY`` override.

    Parameters
    ----------
    monkeypatch
        pytest's monkeypatch fixture (auto-undoes env edits at teardown).
    gateway_value
        Value to set ``SULCI_GATEWAY`` to. Pass ``None`` to delete it
        and exercise the default-URL path.
    """
    if gateway_value is None:
        monkeypatch.delenv("SULCI_GATEWAY", raising=False)
    else:
        monkeypatch.setenv("SULCI_GATEWAY", gateway_value)

    # Purge sulci and any submodule that may have captured the old
    # module-level constants. Without this, a prior test's import
    # leaks the cached _GATEWAY_BASE into this one.
    for name in list(sys.modules):
        if name == "sulci" or name.startswith("sulci."):
            del sys.modules[name]

    return importlib.import_module("sulci")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGatewayBaseAndTelemetryUrl:
    """Module-level constant resolution from SULCI_GATEWAY."""

    def test_default_when_env_unset(self, monkeypatch):
        sulci = _reimport_sulci(monkeypatch, gateway_value=None)
        assert sulci._GATEWAY_BASE  == "https://api.sulci.io"
        assert sulci._TELEMETRY_URL == "https://api.sulci.io/v1/telemetry"

    def test_env_redirects_both_constants(self, monkeypatch):
        """The whole point of v0.5.5: one env var, both URLs follow."""
        sulci = _reimport_sulci(
            monkeypatch,
            gateway_value="https://gateway-production-de5c.up.railway.app",
        )
        assert sulci._GATEWAY_BASE == (
            "https://gateway-production-de5c.up.railway.app"
        )
        assert sulci._TELEMETRY_URL == (
            "https://gateway-production-de5c.up.railway.app/v1/telemetry"
        )

    def test_trailing_slash_normalized(self, monkeypatch):
        """``rstrip('/')`` on _GATEWAY_BASE prevents double-slash in URL."""
        sulci = _reimport_sulci(monkeypatch, gateway_value="https://gateway.example.com/")
        assert sulci._GATEWAY_BASE  == "https://gateway.example.com"
        assert sulci._TELEMETRY_URL == "https://gateway.example.com/v1/telemetry"

    def test_localhost_for_local_dev(self, monkeypatch):
        sulci = _reimport_sulci(monkeypatch, gateway_value="http://localhost:8080")
        assert sulci._TELEMETRY_URL == "http://localhost:8080/v1/telemetry"


class TestPostHonorsResolvedUrl:
    """End-to-end on the function that actually moves bytes."""

    def test_post_sends_to_overridden_url(self, monkeypatch):
        sulci = _reimport_sulci(
            monkeypatch,
            gateway_value="https://gateway.example.com",
        )
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True

        # _post() does `import httpx` lazily inside the function body,
        # so patching `httpx.post` in the global namespace is what
        # _post() will see.
        with patch("httpx.post") as mock_post:
            sulci._post({
                "event":          "cache.get",
                "ts":             0.0,
                "fingerprint":    "f" * 24,
                "sdk_version":    sulci.__version__,
                "backend":        "sqlite",
                "hits":           1,
                "misses":         0,
                "avg_latency_ms": 0.74,
            })

        mock_post.assert_called_once()
        called_url = mock_post.call_args.args[0]
        assert called_url == "https://gateway.example.com/v1/telemetry"

    def test_post_uses_default_when_env_unset(self, monkeypatch):
        sulci = _reimport_sulci(monkeypatch, gateway_value=None)
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True

        with patch("httpx.post") as mock_post:
            sulci._post({
                "event":       "startup",
                "ts":          0.0,
                "fingerprint": "f" * 24,
                "sdk_version": sulci.__version__,
                "backend":     "",
            })

        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == "https://api.sulci.io/v1/telemetry"
