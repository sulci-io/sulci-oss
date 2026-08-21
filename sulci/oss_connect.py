# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/oss_connect.py — OSS-Connect device-code flow client (D12 / v0.6.0)
=========================================================================

Implements the SDK side of the OAuth 2.0 Device Authorization Grant
(RFC 8628) so that ``sulci.connect()`` with no ``api_key`` can obtain a
key via a browser handshake instead of forcing the user to copy-paste
from a welcome email. This is the **first-run UX path** for users on
the OSS-Connect tier.

The gateway side of this flow is owned by sulci-platform; full API
contract is at:

    sulci-platform/docs/architecture/specs/oss-connect-device-code-flow.md

Naming
------
This module is named ``oss_connect``, NOT ``connect``, because
``sulci.connect()`` is already a public function defined in
``sulci/__init__.py``. A ``sulci/connect.py`` submodule would shadow
the function on any ``import sulci.connect`` (Python's submodule-
import semantics replace the attribute on the package). See ADR 0014
§"Naming" for the full chronology.

OSS-Connect-only by enforcement
-------------------------------
The gateway returns ``409 wrong_plan`` if the resolving Clerk user is
on ``free``/``pro``/``business``/``enterprise``. Paid tiers continue to
use the email + paste flow shipped pre-Wave-2. The SDK surfaces that
gateway error as a ``RuntimeError`` to the caller; users on paid tiers
who land here in error will see a message pointing them at their
welcome-email API key.
"""
from __future__ import annotations

import sys
import time
from typing import Tuple

# Module-scope import. This module is itself lazy-imported from
# sulci/__init__.py (only loaded when run_device_code_flow runs), so
# putting httpx here doesn't affect ``import sulci`` startup time —
# and it makes the module trivially mockable in tests
# (``patch("sulci.oss_connect.httpx.post", ...)``).
import httpx


# ── Constants ─────────────────────────────────────────────────────────────────

# Standard RFC 8628 §3.4 grant_type. The gateway validates this as a
# Pydantic Literal so any drift here produces a 422 immediately.
GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

# Default identifier this SDK presents to the gateway. Future non-Python
# clients (e.g. a JS SDK) would pass their own value.
CLIENT_NAME = "sulci-python"


# ── Public entry point ────────────────────────────────────────────────────────

def run_device_code_flow(
    gateway_base: str,
    sdk_version:  str,
) -> str:
    """Run the full device-code flow. Blocks until terminal state.

    Parameters
    ----------
    gateway_base : str
        Base URL of the Sulci gateway, e.g. ``https://api.sulci.io``.
        No trailing slash. Caller is responsible for resolving env-var
        overrides (``SULCI_GATEWAY``) before calling.
    sdk_version : str
        The SDK's own version string; sent to the gateway for analytics
        + future client-version-specific behavior.

    Returns
    -------
    str
        The raw ``sk-sulci-...`` API key on success. Caller is
        responsible for persistence (e.g. ``sulci.config.update(...)``)
        and for activating it for telemetry.

    Raises
    ------
    RuntimeError
        On any user-visible terminal failure: denied, expired,
        not-found, network failure during initial request, or 15-minute
        deadline elapsed without authorization. The exception message
        is suitable for showing to the user — it's already prefixed
        ``sulci.connect() failed: ...``.

    Notes
    -----
    - This function blocks. The 15-minute deadline is the device_code's
      TTL on the gateway; we don't extend it client-side.
    - ``time.sleep(interval)`` runs *before* the first poll. RFC 8628
      §3.5 specifies the polling interval is the minimum gap between
      requests; the spec example calls sleep first too.
    - Transient network errors during polling are swallowed and the
      poll continues. Only the *initial* device-code request raises on
      network error — without a device_code we have nothing to retry.
    """
    # 1. Request a fresh device_code + user_code pair.
    try:
        resp = httpx.post(
            f"{gateway_base}/v1/oss-connect/device-code",
            json={"sdk_version": sdk_version, "client_name": CLIENT_NAME},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(
            f"sulci.connect() failed: could not request device code ({e})"
        ) from e

    # 2. Tell the user what to do. Two lines: the action, then the
    #    waiting state. Flush so the prompt appears immediately even
    #    when stdout is line-buffered behind a pipe.
    print(
        f"[sulci] Visit {data['verification_uri']} "
        f"and enter code: {data['user_code']}"
    )
    print("[sulci] Waiting for authorization (Ctrl+C to cancel)...")
    sys.stdout.flush()

    # 3. Poll /v1/oss-connect/token until terminal state.
    interval = int(data["interval"])
    deadline = time.time() + int(data["expires_in"])

    while time.time() < deadline:
        time.sleep(interval)

        try:
            poll = httpx.post(
                f"{gateway_base}/v1/oss-connect/token",
                json={
                    "device_code": data["device_code"],
                    "grant_type":  GRANT_TYPE,
                },
                timeout=10,
            )
        except Exception:
            # Transient network error — keep polling. The 15-minute
            # deadline still applies, so a sustained outage will fail
            # at the loop boundary with a clear timeout message.
            continue

        # Branch on RFC 8628 status codes. The gateway emits the bare
        # OAuth error envelope `{"error": ..., "message": ...}`, NOT
        # FastAPI's wrapped form — see the spec §"D5 error envelope
        # shape" and gateway/app/routes/oss_connect.py module comment.
        sc = poll.status_code

        if sc == 200:
            body = poll.json()
            print(f"[sulci] ✓ Connected as {body['email']}")
            return body["api_key"]

        if sc == 425:
            # User hasn't authorized yet. Keep polling at the agreed
            # interval. The gateway's slow_down throttle will catch us
            # if we ever hit this faster than `interval`, so this is
            # the safe path.
            continue

        if sc == 400:
            err = _safe_error_field(poll)
            if err == "slow_down":
                # RFC 8628 §3.5 — the SDK SHOULD increase its interval
                # by 5 seconds whenever the AS asks it to slow down.
                interval += 5
                continue
            # Any other 400 falls through to the terminal block below.

        if sc in (403, 404, 410):
            err = _safe_error_field(poll) or "unknown"
            raise RuntimeError(f"sulci.connect() failed: {err}")

        # Any other status is unexpected; we keep polling rather than
        # failing fast — this gives the user a chance to recover from
        # a transient gateway 5xx without blowing up their session.
        # If it persists, the deadline will eventually fire.

    raise RuntimeError("sulci.connect() timed out — please try again")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_error_field(response) -> str | None:
    """Extract the ``error`` field from an httpx response body. Returns
    None if the body isn't valid JSON or doesn't have an ``error`` field.

    Defensive because a hostile network intermediary could substitute a
    junk body, and we'd rather print "unknown" than crash the user's
    interactive ``sulci.connect()`` call.
    """
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    val = body.get("error")
    return val if isinstance(val, str) else None
