"""
flow_1.py — verify Flow 1 · OSS-Connect via /signup
=====================================================

What this script verifies
-------------------------
The default selection at /signup, end-to-end. User picks "OSS-Connect" on
the signup dialog, receives an `sk-sulci-…` key by email, and uses it
locally:

    sulci.connect(api_key="sk-sulci-…")          # one-time, per machine
    cache = Cache(backend="sqlite", ...)           # any local backend
    cache.get(...)                                 # operations stay LOCAL
    # background thread POSTs 8-WIRE-FIELD telemetry to dashboard.sulci.io

This script asserts the SDK side of that contract:
- explicit-arg api_key resolves through precedence rung 1
- ~/.sulci/config is NOT written (explicit-arg path is filesystem-clean)
- telemetry is enabled by default
- telemetry=False variant correctly suppresses egress

How it runs offline
-------------------
Telemetry emit is mocked so no network is touched. The device-code
module is patched to raise on touch — it has no business being invoked
in the explicit-arg path.

Current-state caveat
--------------------
Flow 1's telemetry POST depends on `httpx`, which is in the [cloud]
extra (bug B1, see flows.md). On a bare `pip install sulci`, the
telemetry thread silently fails — user sees no error but no deployment
appears on dashboard.sulci.io. This script doesn't exercise that failure mode
(we test the contract, not the missing-dep failure).
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
from unittest.mock import patch


def _explode(*_a, **_kw):
    raise AssertionError(
        "Flow 1 must not invoke device-code when an explicit api_key is passed.")


def run() -> int:
    failures: list[str] = []

    tmp_home = tempfile.mkdtemp(prefix="flow_1_home_")
    os.environ["HOME"] = tmp_home
    os.environ["USERPROFILE"] = tmp_home  # Windows: Path.home() reads this, not HOME
    os.environ.pop("SULCI_API_KEY", None)
    config_path = Path(tmp_home) / ".sulci" / "config"

    import sulci
    sulci._api_key = None
    sulci._telemetry_enabled = False

    # ── Path A: default telemetry=True ──────────────────────────────────────
    with patch("sulci.oss_connect.httpx.post", side_effect=_explode), \
         patch("sulci._emit", lambda *_a, **_kw: None):
        sulci.connect(api_key="sk-sulci-flow-1-key")

    def expect(cond, msg):
        if not cond:
            failures.append(msg)

    expect(sulci._api_key == "sk-sulci-flow-1-key",
           f"explicit arg should be resolved; got {sulci._api_key!r}")
    expect(sulci._telemetry_enabled is True,
           "telemetry default-true should kick in")
    expect(not config_path.exists(),
           f"explicit-arg path must NOT write config; "
           f"unexpected file at {config_path}")

    # ── Path B: telemetry=False variant ─────────────────────────────────────
    sulci._api_key = None
    sulci._telemetry_enabled = False
    with patch("sulci.oss_connect.httpx.post", side_effect=_explode), \
         patch("sulci._emit", lambda *_a, **_kw: None):
        sulci.connect(api_key="sk-sulci-flow-1-key", telemetry=False)
    expect(sulci._api_key == "sk-sulci-flow-1-key",
           "telemetry=False: key still registered")
    expect(sulci._telemetry_enabled is False,
           "telemetry=False: flush thread suppressed")

    if failures:
        print("FAIL — Flow 1")
        for f in failures:
            print(f"  ✗ {f}")
        return 1

    print("PASS — Flow 1 · OSS-Connect via /signup")
    print("  ✓ explicit arg resolves (precedence rung 1)")
    print("  ✓ ~/.sulci/config not written — explicit-arg is filesystem-clean")
    print("  ✓ device-code never invoked")
    print("  ✓ telemetry=False variant suppresses egress while keeping key in-memory")
    return 0


if __name__ == "__main__":
    sys.exit(run())
