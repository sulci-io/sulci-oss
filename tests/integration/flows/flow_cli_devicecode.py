"""
flow_cli_devicecode.py — INTERNAL · device-code SDK contract verification
==========================================================================

STATUS: Internal reference. Not part of the three user-facing flows.

This script verifies the SDK side of the RFC 8628 device-code grant
that lives in ``sulci.oss_connect``. The mechanism is preserved in the
SDK because a future ``sulci`` CLI (e.g. ``sulci auth login``,
``sulci deployments list``) will need exactly this kind of "auth this
terminal session" handshake — the same primitive ``gh auth login``,
``gcloud auth login``, and ``aws sso login`` rely on.

Today, no user-facing surface points at this code. The /signup form
issues keys by email; users follow Flow 1 / Flow 2 / Flow 3, none of
which call ``prompt=True``. This script is the institutional knowledge
that says: "this SDK module is intentional infrastructure, not dead
code; keep it green so a future CLI ship doesn't need to rebuild it."

How it runs offline
-------------------
Patches ``sulci.oss_connect.httpx.post`` with a stub that simulates the
gateway's two endpoints (``/device-code`` and ``/token``). First poll
returns 425 (pending), second poll returns 200 with the api_key. No
real network, Clerk, or dashboard touched.

Known platform-side gap (P2, was P0 before flow demotion)
---------------------------------------------------------
The platform-side dashboard route (``dashboard.sulci.io/oss-connect``) still
redirects to the paid-tier key-paste page after Clerk sign-up. This
script does NOT exercise the dashboard surface, so it PASSes even while
the dashboard is broken. When the CLI ships, the dashboard route bug
becomes user-facing again and must be fixed first.
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
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Fixture state ────────────────────────────────────────────────────────────
# Mutable counter so the stubbed httpx.post can return different responses
# on each call (one /device-code, then poll responses for /token).
_calls: list[dict] = []


def _stub_httpx_post(url: str, json=None, timeout=None, **kwargs):
    """Stand-in for the gateway. Returns a MagicMock shaped like an
    httpx.Response so the SDK code that calls ``.raise_for_status()``,
    ``.json()``, and reads ``.status_code`` keeps working."""
    _calls.append({"url": url, "body": json})

    resp = MagicMock()
    resp.raise_for_status = MagicMock()

    if url.endswith("/v1/oss-connect/device-code"):
        resp.status_code = 200
        resp.json = MagicMock(return_value={
            "device_code":              "x" * 43,
            "user_code":                "WXYZ-1234",
            "verification_uri":         "https://dashboard.sulci.io/oss-connect",
            "verification_uri_complete":"https://dashboard.sulci.io/oss-connect?code=WXYZ-1234",
            # interval=0 so the SDK's time.sleep is a no-op and the test
            # runs in milliseconds rather than waiting 5s between polls.
            "expires_in":               900,
            "interval":                 0,
        })
        return resp

    if url.endswith("/v1/oss-connect/token"):
        token_polls = [c for c in _calls if c["url"].endswith("/token")]
        if len(token_polls) == 1:
            # First poll — user hasn't authorized yet. 425 Too Early.
            resp.status_code = 425
            resp.json = MagicMock(return_value={"error": "authorization_pending"})
            return resp
        # Second poll — user has authorized in browser. 200 + api_key.
        resp.status_code = 200
        resp.json = MagicMock(return_value={
            "api_key": "sk-sulci-flow-a-test",
            "email":   "test@example.com",
            "plan":    "oss_connect",
        })
        return resp

    raise AssertionError(f"unexpected URL: {url}")


# ── Test harness ─────────────────────────────────────────────────────────────
def run() -> int:
    """Returns 0 on PASS, 1 on FAIL. Prints a structured report either way."""

    # 1. Ensure key resolution chain is empty.
    os.environ.pop("SULCI_API_KEY", None)
    tmp_home = tempfile.mkdtemp(prefix="flow_a_home_")
    os.environ["HOME"] = tmp_home
    os.environ["USERPROFILE"] = tmp_home  # Windows: Path.home() reads this, not HOME
    config_path = Path(tmp_home) / ".sulci" / "config"
    assert not config_path.exists(), "fresh temp HOME should have no config"

    # 2. Reset sulci module state so prior runs don't leak.
    import sulci
    sulci._api_key = None
    sulci._telemetry_enabled = False

    # 3. Patch httpx in BOTH oss_connect (device-code) and the
    #    telemetry-flush path (the startup event emit) so we don't
    #    accidentally touch the network on a CI runner.
    failures: list[str] = []

    with patch("sulci.oss_connect.httpx.post", side_effect=_stub_httpx_post), \
         patch("sulci._emit", lambda *_args, **_kw: None):
        try:
            sulci.connect(prompt=True)
        except Exception as e:
            failures.append(f"connect() raised unexpectedly: {e!r}")

    # 4. Assertions
    def expect(cond: bool, msg: str):
        if not cond:
            failures.append(msg)

    # (a) Two calls to /token (pending, then authorized) preceded by one
    #     /device-code request — total 3 gateway hits.
    expect(len(_calls) == 3,
           f"expected 3 gateway calls, got {len(_calls)}: "
           f"{[c['url'].rsplit('/', 1)[-1] for c in _calls]}")
    expect(_calls[0]["url"].endswith("/device-code"),
           "first call must be /device-code")
    expect(all(c["url"].endswith("/token") for c in _calls[1:]),
           "subsequent calls must be /token")

    # (b) Device-code request body matches the spec.
    body = _calls[0]["body"] or {}
    expect("sdk_version"  in body, "device-code body missing sdk_version")
    expect(body.get("client_name") == "sulci-python",
           f"client_name should be 'sulci-python', got {body.get('client_name')!r}")

    # (c) Token poll body uses the RFC 8628 grant_type and the device_code
    #     received in step 1.
    poll_body = _calls[1]["body"] or {}
    expect(poll_body.get("grant_type") == "urn:ietf:params:oauth:grant-type:device_code",
           "grant_type must be RFC 8628 device_code URI")
    expect(poll_body.get("device_code") == "x" * 43,
           "device_code from response must be echoed in poll body")

    # (d) The api_key landed in module state.
    expect(sulci._api_key == "sk-sulci-flow-a-test",
           f"sulci._api_key should be sk-sulci-flow-a-test, "
           f"got {sulci._api_key!r}")

    # (e) The api_key was persisted to ~/.sulci/config — this is the
    #     promise that "subsequent runs short-circuit on config" rests on.
    expect(config_path.exists(),
           f"~/.sulci/config should have been created at {config_path}")
    if config_path.exists():
        content = config_path.read_text()
        expect("sk-sulci-flow-a-test" in content,
               "config file should contain the api_key")

    # ── Report ──────────────────────────────────────────────────────────────
    if failures:
        print("FAIL — Flow A")
        for f in failures:
            print(f"  ✗ {f}")
        return 1

    print("PASS — flow_cli_devicecode · SDK device-code contract (internal)")
    print(f"  ✓ {len(_calls)} gateway calls in correct order")
    print("  ✓ api_key resolved into sulci._api_key")
    print(f"  ✓ api_key persisted to {config_path}")
    print()
    print("  (this verifies the SDK module preserved for a future CLI; the")
    print("   platform-side dashboard route remains broken — P2, see flows.md)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
