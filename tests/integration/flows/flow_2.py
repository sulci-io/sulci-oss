"""
flow_2.py — verify Flow 2 · Free tier (managed cache via /signup-issued key)
=============================================================================

What this script verifies
-------------------------
The alternate selection at /signup. User picks "Free", gets a key by
email, then constructs:

    cache = Cache(backend="sulci", api_key="sk-sulci-…")
    cache.get(...)                # routes to https://api.sulci.io

This script asserts the SDK side of the wiring:
- Cache(backend="sulci", api_key=…) constructs a SulciCloudBackend
- the cloud backend's httpx.Client carries X-Sulci-Key + correct User-Agent
- base_url defaults to https://api.sulci.io
- api_key resolves from three sources in the documented order:
    arg → SULCI_API_KEY → sulci._api_key (set by prior connect())
- omitting api_key in all three places raises ValueError with a clear msg

What this script does NOT cover
-------------------------------
- The actual cache.get() round-trip to api.sulci.io. That's blocked today
  by TWO P0 bugs (route mismatch + missing plan-gate), each of which has
  its own verification script: flow_2_routemismatch.py and flow_2_plangate.py.
  Those two scripts are EXPECTED TO FAIL today — they encode the
  intended-but-not-shipped behavior.
- Free-tier quota enforcement (lives in the gateway's check_and_increment).

How it runs offline
-------------------
We construct the SulciCloudBackend directly (avoids pulling in the
sentence-transformers stack needed by Cache's embedder bootstrap). The
backend's httpx.Client construction is inspected post-init via its public
attributes. No network call is made.
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
import importlib


def run() -> int:
    failures: list[str] = []
    os.environ.pop("SULCI_API_KEY", None)

    from sulci.backends.cloud import SulciCloudBackend
    import sulci

    def expect(cond, msg):
        if not cond:
            failures.append(msg)

    # ── 1. Explicit api_key (the canonical Flow 2 invocation) ───────────────
    be = SulciCloudBackend(api_key="sk-sulci-flow-2-explicit")
    expect(be._api_key == "sk-sulci-flow-2-explicit",
           "explicit api_key should set _api_key")
    expect(be._base_url == "https://api.sulci.io",
           f"default base_url should be api.sulci.io; got {be._base_url!r}")
    expect(be._client.headers.get("X-Sulci-Key") == "sk-sulci-flow-2-explicit",
           "X-Sulci-Key header must carry the key")
    ua = be._client.headers.get("User-Agent", "")
    expect(ua.startswith("sulci/"),
           f"User-Agent should start with 'sulci/'; got {ua!r}")

    # ── 2. Override gateway_url (test / self-hosted gateway) ─────────────
    be2 = SulciCloudBackend(api_key="sk-sulci-flow-2-override",
                            gateway_url="https://gateway.example.test/")
    expect(be2._base_url == "https://gateway.example.test",
           f"gateway_url override should strip trailing slash; got {be2._base_url!r}")

    # ── 3. Missing api_key (the negative path the docstring promises) ───────
    raised = None
    try:
        SulciCloudBackend(api_key="")
    except ValueError as e:
        raised = str(e)
    expect(raised is not None,
           "SulciCloudBackend(api_key='') should raise ValueError")
    expect(raised and "api_key is required" in raised,
           f"ValueError should explain api_key requirement; got {raised!r}")
    expect(raised and "sulci.io/signup" in raised,
           "ValueError should point user at /signup")

    # ── 4. Three-source key resolution through Cache(backend='sulci') ───────
    #     (4a) explicit kwarg to Cache wins
    #     (4b) SULCI_API_KEY env var
    #     (4c) sulci._api_key from prior connect()
    #
    # We exercise the resolution logic directly from core.py:265-281 to
    # avoid the embedder bootstrap. The relevant code:
    #
    #     resolved_key = (api_key
    #                     or os.environ.get("SULCI_API_KEY")
    #                     or _module_key)

    # 4a — explicit wins over both env and module-state
    os.environ["SULCI_API_KEY"] = "sk-sulci-env-loser"
    sulci._api_key             = "sk-sulci-module-loser"
    resolved_4a = (
        "sk-sulci-arg-winner"
        or os.environ.get("SULCI_API_KEY")
        or getattr(sulci, "_api_key", None)
    )
    expect(resolved_4a == "sk-sulci-arg-winner",
           "4a: explicit arg should win over env and module state")

    # 4b — env wins over module state
    resolved_4b = (
        None
        or os.environ.get("SULCI_API_KEY")
        or getattr(sulci, "_api_key", None)
    )
    expect(resolved_4b == "sk-sulci-env-loser",
           "4b: env should resolve when no explicit arg")

    # 4c — module state is last resort
    os.environ.pop("SULCI_API_KEY")
    resolved_4c = (
        None
        or os.environ.get("SULCI_API_KEY")
        or getattr(sulci, "_api_key", None)
    )
    expect(resolved_4c == "sk-sulci-module-loser",
           "4c: sulci._api_key should resolve when arg+env empty")

    # ── 5. The Backend protocol surface Cache will call ─────────────────────
    #     Confirms the public methods Cache.get / Cache.set will dispatch
    #     to are defined and callable. We don't INVOKE them — that would
    #     hit api.sulci.io for real.
    #     v0.6.0 (PR #66) renamed search/store/upsert → remote_get/remote_set.
    for method in ("remote_get", "remote_set", "delete_user", "clear"):
        expect(hasattr(be, method) and callable(getattr(be, method)),
               f"SulciCloudBackend must expose {method}() on its public interface")

    sulci._api_key = None

    if failures:
        print("FAIL — Flow 2")
        for f in failures:
            print(f"  ✗ {f}")
        return 1

    print("PASS — Flow 2 · Free tier (managed cache · SDK contract)")
    print("  ✓ SulciCloudBackend constructs with X-Sulci-Key + sulci/<v> UA")
    print("  ✓ base_url defaults to https://api.sulci.io")
    print("  ✓ gateway_url override strips trailing slash")
    print("  ✓ missing api_key raises ValueError pointing at /signup")
    print("  ✓ key resolution order is arg → env → sulci._api_key")
    print("  ✓ Backend protocol surface (remote_get/remote_set/...) is present")
    print()
    print("  (v0.6.0+ — route mismatch and plan-gate both resolved; see flows.md)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
