# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/__init__.py
================
Public API surface for the sulci semantic caching library.

Exports
-------
Cache           — main cache engine (context-aware, v0.2+)
ContextWindow   — per-session conversation window
SessionStore    — multi-session manager
connect()       — opt-in telemetry + cloud key registration (v0.3+)

Telemetry
---------
Nothing phones home by default.  Telemetry is strictly opt-in:

    import sulci
    sulci.connect(api_key="sk-sulci-...")   # enables telemetry

Or per-instance:

    cache = Cache(backend="sulci", api_key="sk-sulci-...")

What is sent (aggregate counts only — no query content, no user data):
    {event, backend, hits, misses, avg_latency_ms, sdk_version}

Data never sent:
    query text, response text, embeddings, user_id, session_id, IP address
"""

import os
import threading
import time
from typing import Optional

# ── Module-level telemetry state ─────────────────────────────────────────────
# Both are False/None by default — connect() is the only way to change them.

_api_key:           Optional[str] = None
_telemetry_enabled: bool          = False

_TELEMETRY_URL = "https://api.sulci.io/v1/telemetry"
_SDK_VERSION   = "0.3.0"
_FLUSH_INTERVAL_SECONDS = 30

_event_buffer: list  = []
_buffer_lock          = threading.Lock()
_flush_thread_started = False


# ── Public API ────────────────────────────────────────────────────────────────

def connect(
    api_key:   Optional[str] = None,
    telemetry: bool          = True,
) -> None:
    """
    Opt-in gate for Sulci Cloud telemetry and key registration.

    Parameters
    ----------
    api_key : str, optional
        Your Sulci Cloud API key (sk-sulci-...).
        Falls back to SULCI_API_KEY environment variable if not provided.
        Required for telemetry to be enabled.
    telemetry : bool, default True
        Set to False to register your key without enabling telemetry.
        Useful for the sulci backend driver without usage reporting.

    Examples
    --------
    # Typical usage — enables telemetry
    import sulci
    sulci.connect(api_key="sk-sulci-...")

    # Key from environment variable
    # export SULCI_API_KEY="sk-sulci-..."
    sulci.connect()

    # Register key but disable telemetry
    sulci.connect(api_key="sk-sulci-...", telemetry=False)

    # No-op — nothing sent, no key set
    # (just don't call connect() at all)
    """
    global _api_key, _telemetry_enabled

    _api_key = api_key or os.environ.get("SULCI_API_KEY")

    # Telemetry is only active when BOTH conditions are true:
    #   1. the caller explicitly passed telemetry=True (the default)
    #   2. an api_key was resolved
    _telemetry_enabled = telemetry and (_api_key is not None)

    if _telemetry_enabled:
        _start_flush_thread()
        _emit("startup", {})


# ── Internal telemetry helpers ────────────────────────────────────────────────
# All functions below are no-ops when _telemetry_enabled is False.
# All exceptions are swallowed — telemetry must never affect the user's app.

def _emit(event: str, data: dict) -> None:
    """
    Buffer a telemetry event.  O(1) — safe to call from the Cache hot path.

    No-op when telemetry is disabled (the default).
    """
    if not _telemetry_enabled or not _api_key:
        return
    with _buffer_lock:
        _event_buffer.append({
            "event": event,
            "ts":    time.time(),
            **data,
        })


def _flush() -> None:
    """
    Drain the event buffer and POST a single aggregated batch to api.sulci.io.

    Aggregation: sum hits/misses across all buffered cache.get events,
    average the latency.  One HTTP call per flush interval regardless of
    how many individual events were buffered.

    Never raises — all exceptions are swallowed silently.
    """
    global _event_buffer

    with _buffer_lock:
        if not _event_buffer:
            return
        batch          = _event_buffer[:]
        _event_buffer  = []

    # Aggregate cache.get events
    get_events  = [e for e in batch if e.get("event") == "cache.get"]
    hits        = sum(e.get("hits",   0) for e in get_events)
    misses      = sum(e.get("misses", 0) for e in get_events)
    latencies   = [e.get("latency_ms", 0) for e in get_events if e.get("latency_ms")]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    backend     = get_events[0].get("backend", "") if get_events else \
                  batch[0].get("backend", "")

    payload = {
        "event":          "cache.get",
        "backend":        backend,
        "hits":           hits,
        "misses":         misses,
        "avg_latency_ms": avg_latency,
        "sdk_version":    _SDK_VERSION,
        "python_version": _python_version(),
    }

    try:
        import httpx
        httpx.post(
            _TELEMETRY_URL,
            json    = payload,
            headers = {"X-Sulci-Key": _api_key},
            timeout = 3.0,
        )
    except Exception:
        # Never let a telemetry failure surface to the user's app.
        pass


def _flush_loop() -> None:
    """Background thread target: flush every FLUSH_INTERVAL_SECONDS."""
    while True:
        time.sleep(_FLUSH_INTERVAL_SECONDS)
        if not _telemetry_enabled:
            # Telemetry was disabled after the thread started — stop quietly.
            return
        _flush()


def _start_flush_thread() -> None:
    """
    Start the background flush thread exactly once.

    Uses a module-level flag rather than checking thread.is_alive() to
    avoid the overhead of thread object lookup on every connect() call.
    """
    global _flush_thread_started
    if _flush_thread_started:
        return
    _flush_thread_started = True
    t = threading.Thread(target=_flush_loop, daemon=True, name="sulci-telemetry-flush")
    t.start()


# ── Core library imports (lazy) ───────────────────────────────────────────────
# Imported here rather than at the top so:
#   1. The telemetry module is independently testable without the full
#      sulci package installed (test_connect.py has no dependency on Cache).
#   2. Circular import risk between __init__ -> core -> __init__ is avoided.
#
# In normal usage (pip install sulci) these always resolve.
# In test-only environments (just __init__.py present) they gracefully
# return None and the telemetry tests still pass.

try:
    from sulci.core import Cache
    from sulci.context import ContextWindow, SessionStore
except ImportError:
    Cache = None          # type: ignore[assignment]
    ContextWindow = None  # type: ignore[assignment]
    SessionStore = None   # type: ignore[assignment]

def _python_version() -> str:
    import sys
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"

# ── Public exports ────────────────────────────────────────────────────────────

__all__ = [
    "Cache",
    "ContextWindow",
    "SessionStore",
    "connect",
]
