# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/__init__.py
================
Public API surface for the sulci semantic caching library.

Exports
-------
Cache           — main cache engine (context-aware, v0.2+)
AsyncCache      — non-blocking async wrapper around Cache (v0.3.7+)
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
    {event, backend, hits, misses, avg_latency_ms, sdk_version,
     python_version, fingerprint}

The 9-field shape is locked by the gateway's TelemetryEvent schema
(``extra='forbid'`` — anything else is rejected with HTTP 422 and the
batch is silently dropped). See ``sulci.telemetry.WIRE_FIELDS``.

Data never sent:
    query text, response text, embeddings, user_id, session_id, IP address

The ``fingerprint`` field is a stable 24-char per-deployment hash —
``blake2b(machine_id || backend || embedding_model || threshold ||
context_window, digest_size=12)``. The ``machine_id`` is a
locally-generated ``uuid4`` persisted at ``~/.sulci/config``; it never
leaves the local machine. See :func:`sulci.telemetry.build_fingerprint`.

AsyncCache
----------
Drop-in non-blocking wrapper for FastAPI, LangChain async chains,
LlamaIndex async agents, and any asyncio-based application:

    from sulci import AsyncCache

    cache = AsyncCache(backend="sqlite", threshold=0.85, context_window=4)

    @app.post("/chat")
    async def chat(query: str, session_id: str):
        response, sim, depth = await cache.aget(query, session_id=session_id)
        if response:
            return {"response": response, "source": "cache"}
        response = await call_llm(query)
        await cache.aset(query, response, session_id=session_id)
        return {"response": response, "source": "llm"}

All constructor parameters are identical to Cache.
"""

import atexit
import logging
import os
import threading
import time
from typing import Optional
from importlib.metadata import version as _pkg_version, PackageNotFoundError

# ── Package version ──────────────────────────────────────────────────────────
# Single source of truth: pyproject.toml. We read it via importlib.metadata
# at import time. Editable installs and uninstalled trees fall back to a
# placeholder so the import doesn't crash in dev/test environments.
try:
    __version__ = _pkg_version("sulci")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# ── Module logger ────────────────────────────────────────────────────────────
# Lives under the 'sulci' namespace so callers can configure verbosity with
# `logging.getLogger("sulci").setLevel(logging.INFO)` or via dictConfig.
# Default Python logging level (WARNING) keeps INFO lines quiet for callers
# who haven't opted into verbose logging — no stdout spam by default.
log = logging.getLogger(__name__)

# ── Module-level telemetry state ─────────────────────────────────────────────
# Both are False/None by default — connect() is the only way to change them.

_api_key:           Optional[str] = None
_telemetry_enabled: bool          = False

# Gateway base URL — read once at import time. Production points at
# api.sulci.io; staging/local-dev override via SULCI_GATEWAY.
# Resolved here (not inside `connect()`) so that the v0.6.0 device-code
# flow AND the v0.5.x telemetry pipeline see the same value, and so
# tests that monkeypatch the env var before importing `sulci` still
# pick up the override.
#
# v0.5.5 fix (#51): _TELEMETRY_URL is now derived from _GATEWAY_BASE.
# Prior to v0.5.5, _TELEMETRY_URL was a hardcoded literal, so setting
# SULCI_GATEWAY redirected the device-code flow but silently did NOT
# redirect telemetry POSTs — contradicting the comment above and
# blocking staging-gateway smoke tests.
_GATEWAY_BASE  = os.environ.get("SULCI_GATEWAY", "https://api.sulci.io").rstrip("/")
_TELEMETRY_URL = f"{_GATEWAY_BASE}/v1/telemetry"
_SDK_VERSION   = __version__   # deprecated alias; new code should use sulci.__version__
_FLUSH_INTERVAL_SECONDS = 30

_event_buffer: list  = []
_buffer_lock          = threading.Lock()
_flush_thread_started = False


# ── Public API ────────────────────────────────────────────────────────────────

def connect(
    api_key:   Optional[str] = None,
    telemetry: bool          = True,
    prompt:    bool          = False,
) -> None:
    """
    Connect this process to Sulci Cloud.

    Resolution order for the api_key (first match wins):

      1. ``api_key`` argument
      2. ``SULCI_API_KEY`` environment variable
      3. ``~/.sulci/config`` (persisted from a prior successful connect)
      4. Browser-based device-code flow — only if ``prompt=True`` AND
         none of the above produced a key. Blocks until the user
         authorizes via the browser, denies, or the 15-minute device
         code expires. **OSS-Connect tier only** (the gateway returns
         409 wrong_plan for any other tier; paid-tier users should use
         the API key from their signup email).

    Each rung emits an INFO-level log line on the ``sulci`` logger
    indicating which source supplied the key (including the key prefix
    and, for the config rung, the file mtime). Default Python logging
    level (WARNING) keeps these quiet; opt in with
    ``logging.getLogger("sulci").setLevel(logging.INFO)`` when
    debugging "wrong key" or "telemetry not arriving" issues. Added
    2026-05-13 — see sulci-oss #79 for context.

    .. warning::
       In v0.5.3 the device-code flow ships **latent**: the SDK code
       is in place, but the gateway endpoints (sulci-platform
       ``/v1/oss-connect/*``) and the dashboard ``/oss-connect``
       page may not yet be deployed in your environment. Until
       both ship, calling ``connect(prompt=True)`` interactively
       on a missing key will print a "Visit ..." prompt with a URL
       that 404s and then block for 15 minutes before timing out.

       The default is ``prompt=False`` for that reason. **Setting
       ``prompt=True`` against an environment that hasn't announced
       OSS-Connect availability is user error** — wait for the
       Sulci team's release announcement that the full chain is
       live (gateway + dashboard) before flipping it on.

    .. note::
       As of 2026-07-06 the full OSS-Connect chain (SDK + gateway
       D4/D4.5/D5 + dashboard ``/oss-connect``) has been live
       end-to-end for two months (cutover 2026-05-08). The default
       nonetheless remains ``prompt=False``, and this is now a
       sustained decision rather than a stale promise:

         1. Non-interactive default is safe for library callers that
            have no tty / browser (LangChain / LlamaIndex agents,
            FastAPI request handlers, LangGraph nodes, CI runners).
            ``prompt=True`` at import time would block those callers
            on a 15-minute device-code timeout with no visible cause.
         2. v0.7.0 shipped ``Cache()`` auto-connect — passing an
            ``api_key=`` to the ``Cache`` constructor attaches
            telemetry automatically, which is the ergonomic
            "make it easy" path users actually reach for. That
            covers the goal without a blocking browser prompt as
            an import-time side effect.
         3. Explicit ``prompt=True`` remains a first-class supported
            call — users on interactive machines can still opt in
            per call with one keyword argument.

    Parameters
    ----------
    api_key : str, optional
        Your Sulci Cloud API key (sk-sulci-...). If omitted, falls
        through the resolution order above.
    telemetry : bool, default True
        Set to False to register your key without enabling telemetry.
        Useful for the sulci backend driver without usage reporting.
    prompt : bool, default False (sustained — see note above)
        When True, if no api_key is found through args/env/config,
        run the browser-based device-code flow to obtain one. Safe to
        use on interactive machines; not recommended in library-facing
        code paths (agents, request handlers, CI runners) where a
        15-minute blocking prompt with no tty is a footgun.

    Examples
    --------
    # Paid-tier user — paste the key from your welcome email
    sulci.connect(api_key="sk-sulci-...")

    # Or set SULCI_API_KEY env var, then:
    sulci.connect()

    # OSS-Connect user — no key in hand, follow the browser prompt
    sulci.connect()

    # Subsequent runs short-circuit on ~/.sulci/config — no browser
    sulci.connect()

    # CI / headless: don't try to prompt, just be a no-op if no key
    sulci.connect(prompt=False)

    # Register key but disable telemetry (key still cached for cache lookups)
    sulci.connect(api_key="sk-sulci-...", telemetry=False)

    Raises
    ------
    RuntimeError
        If the device-code flow runs and fails (denied, expired,
        timeout, network error). Pass ``prompt=False`` to skip the
        flow entirely if you'd rather have a silent no-op on failure.
    """
    global _api_key, _telemetry_enabled

    # Resolution chain — each rung emits an INFO-level log line on the
    # selected source so callers can diagnose "wrong key" issues without
    # exposing the key value. Closes sulci-oss #79 (silent fallback was
    # debug-hostile during the v0.6.x close-out session). Default Python
    # logging level is WARNING, so these lines stay quiet unless the
    # caller opts into INFO+ verbosity.
    resolved: Optional[str] = None

    # 1. Explicit argument
    if api_key:
        resolved = api_key
        log.info(
            "sulci.connect: using explicit api_key argument (prefix=%s)",
            _key_prefix(resolved),
        )

    # 2. SULCI_API_KEY env var
    if not resolved:
        env_key = os.environ.get("SULCI_API_KEY")
        if env_key:
            resolved = env_key
            log.info(
                "sulci.connect: using SULCI_API_KEY env var (prefix=%s)",
                _key_prefix(resolved),
            )

    # 3. Persisted config (~/.sulci/config)
    if not resolved:
        cfg_key = _read_key_from_config()
        if cfg_key:
            resolved = cfg_key
            # mtime helps diagnose stale-key issues — a config written
            # months ago is the most common source of "telemetry not
            # showing up" surprises. None if file missing or unreadable.
            log.info(
                "sulci.connect: using persisted ~/.sulci/config (prefix=%s, mtime=%s)",
                _key_prefix(resolved),
                _config_file_mtime() or "unknown",
            )

    # 4. Browser-based device-code flow (D12 — v0.6.0)
    if not resolved and prompt:
        # Lazy import: only reached on first-run for an OSS-Connect user.
        # Avoids paying httpx import cost (and module-level side effects)
        # for the common case where a key is already available.
        from sulci import oss_connect as _oss_connect
        resolved = _oss_connect.run_device_code_flow(
            gateway_base = _GATEWAY_BASE,
            sdk_version  = __version__,
        )
        if resolved:
            log.info(
                "sulci.connect: using device-code flow result (prefix=%s)",
                _key_prefix(resolved),
            )
        # Persist for next invocation. Failure to persist is non-fatal —
        # the user just gets prompted again next time, which is mildly
        # annoying but not broken.
        _persist_key_to_config(resolved)

    # DEBUG-level (not INFO) — this is fine behavior when prompt=False,
    # not a problem. Users who want to see it pass logging.DEBUG.
    if not resolved:
        log.debug(
            "sulci.connect: no api_key resolved through any rung "
            "(arg/env/config/device-code) — telemetry stays disabled"
        )

    _api_key = resolved

    # Telemetry is only active when BOTH conditions are true:
    #   1. the caller explicitly passed telemetry=True (the default)
    #   2. an api_key was resolved
    _telemetry_enabled = telemetry and (_api_key is not None)

    if _telemetry_enabled:
        _start_flush_thread()
        _emit("startup", {})


def _key_prefix(api_key: Optional[str]) -> str:
    """Return the first 16 characters of an api_key for logging.

    16 chars is enough to identify which key is in use (the dashboard
    shows the same prefix on the deployments view) but not enough to
    use as a credential — full keys are ~52 chars. None or empty input
    returns the empty string so log format strings stay safe.

    Added 2026-05-13 alongside connect() resolution-path logging
    (sulci-oss #79).
    """
    return (api_key or "")[:16]


def _config_file_mtime() -> Optional[str]:
    """Return ISO-format mtime of ~/.sulci/config, or None on failure.

    Used in the resolution-path INFO log when connect() falls through
    to the persisted-config rung. Knowing *when* the config was last
    written is the single most useful diagnostic for "stale key"
    issues — a config written months ago is the most common source
    of "my telemetry is going to the wrong account" surprises.

    Any failure mode (file missing, permission denied, OS oddity)
    returns None so the caller can substitute a placeholder without
    crashing on the log call. Added 2026-05-13 (sulci-oss #79).
    """
    try:
        from sulci import config
        import datetime
        path = config._config_path()
        if path.exists():
            return datetime.datetime.fromtimestamp(
                path.stat().st_mtime, tz=datetime.timezone.utc
            ).isoformat(timespec="seconds")
    except Exception:
        pass
    return None


# ── Config staleness threshold ───────────────────────────────────────────────
# Persisted ~/.sulci/config keys older than this are skipped during
# resolution (treated as stale). 90 days is generous enough that
# legitimate persisted keys from regular users do not hit it but tight
# enough that stale keys from old account-cycling sessions get caught.
# See sulci-oss #80 for context.
_CONFIG_MAX_AGE_DAYS = 90


def _read_key_from_config() -> Optional[str]:
    """Read api_key from ~/.sulci/config, with staleness guard.

    Returns ``None`` (skipping the config rung in resolution) when:

      - ``written_at`` field is missing (config predates v0.6.5 — treated
        as stale because we cannot verify the age)
      - ``written_at`` is older than ``_CONFIG_MAX_AGE_DAYS``
      - ``written_at`` is unparseable (corrupt or future-incompatible)
      - any other failure mode (file missing, malformed JSON, permission
        denied) — these are silent per the config module design rules

    On each stale-skip path, emits a WARNING log line telling the user
    how to refresh the key (re-run with ``prompt=True`` or pass
    ``api_key=`` explicitly).

    The config module is dependency-free + corruption-tolerant by its
    own design rules (see sulci/config.py module docstring), but we
    still wrap with try/except as defense-in-depth.

    Closes sulci-oss #80. Added 2026-05-13.
    """
    try:
        import datetime
        from sulci import config
        data = config.load()
        api_key = data.get("api_key")
        if not api_key:
            return None

        # Staleness guard.
        written_at_str = data.get("written_at")
        if not written_at_str:
            log.warning(
                "sulci.connect: ~/.sulci/config has no written_at timestamp "
                "(predates v0.6.5) — treating as stale and skipping. "
                "Re-run with sulci.connect(prompt=True) to refresh the "
                "persisted key, or pass api_key=... explicitly."
            )
            return None

        try:
            written_at = datetime.datetime.fromisoformat(written_at_str)
            # Defensive: a tz-naive value (shouldn't happen with our writer,
            # but a hand-edited config might) is assumed UTC so the
            # subtraction below is well-defined.
            if written_at.tzinfo is None:
                written_at = written_at.replace(tzinfo=datetime.timezone.utc)
        except (ValueError, TypeError):
            log.warning(
                "sulci.connect: ~/.sulci/config has unparseable written_at "
                "value (%r) — treating as stale and skipping. "
                "Re-run with sulci.connect(prompt=True) to refresh, or pass "
                "api_key=... explicitly.",
                written_at_str,
            )
            return None

        age = datetime.datetime.now(tz=datetime.timezone.utc) - written_at
        if age.days > _CONFIG_MAX_AGE_DAYS:
            log.warning(
                "sulci.connect: ~/.sulci/config is %d days old "
                "(written %s; threshold %d days) — treating as stale and "
                "skipping. Re-run with sulci.connect(prompt=True) to "
                "refresh, or pass api_key=... explicitly.",
                age.days, written_at_str, _CONFIG_MAX_AGE_DAYS,
            )
            return None

        return api_key
    except Exception:
        return None


def _persist_key_to_config(api_key: str) -> None:
    """Persist api_key to ~/.sulci/config. Failures are non-fatal;
    a config-write failure means the user gets prompted again on the
    next invocation, which is recoverable.
    """
    try:
        from sulci import config
        config.update(api_key=api_key)
    except Exception:
        pass


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
    Drain the event buffer and POST aggregated batches to the configured
    Sulci gateway (``_TELEMETRY_URL``, derived from ``SULCI_GATEWAY`` —
    defaults to ``https://api.sulci.io/v1/telemetry``).

    v0.5.2: aggregates cache.get and cache.set events separately, sending
    one HTTP call per event type that has events. cache.get carries
    hit/miss/latency aggregates; cache.set carries write count and
    average write latency (hits = count, misses = 0 by convention — see
    "cache.set semantics" note below).

    Each payload now includes a ``fingerprint`` field — a stable,
    anonymous, config-aware deployment identifier (see
    :func:`sulci.telemetry.build_fingerprint`). This lets the
    ``/v1/analytics/deployments`` dashboard tile group events by
    deployment.

    Startup events (emitted by :func:`connect`) are POSTed once per
    flush cycle that contains any startup event. Backend is sniffed
    from any non-startup event in the same batch so the row joins
    cleanly with later cache.get/cache.set rows on the dashboard; if
    no get/set has happened yet (the typical case for the first flush
    after :func:`connect`), the startup goes out with ``backend=""``.
    The gateway accepts an empty backend, and the fingerprint alone is
    enough to dedupe the deployment row once cache traffic begins.

    Never raises — all exceptions are swallowed silently.

    cache.set semantics
    -------------------
    The gateway's TelemetryEvent schema reuses ``hits`` / ``misses`` /
    ``avg_latency_ms`` for all event types. For ``event='cache.set'``
    the SDK convention is::

        hits           = number of set() calls aggregated
        misses         = 0
        avg_latency_ms = average set() latency

    This is documented here and on the gateway side; a future schema
    revision may rename these fields per-event-type.
    """
    global _event_buffer

    with _buffer_lock:
        if not _event_buffer:
            return
        batch          = _event_buffer[:]
        _event_buffer  = []

    # Build the fingerprint once per flush. We use the most recent
    # event's backend (events from one process should all share one
    # backend; if they don't, the dashboard will show them as separate
    # deployments which is the desired behavior).
    fingerprint = _build_fingerprint_for_batch(batch)

    # Aggregate cache.get events
    get_events  = [e for e in batch if e.get("event") == "cache.get"]
    if get_events:
        hits        = sum(e.get("hits",   0) for e in get_events)
        misses      = sum(e.get("misses", 0) for e in get_events)
        latencies   = [e.get("latency_ms", 0) for e in get_events if e.get("latency_ms")]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        backend     = get_events[0].get("backend", "")

        _post({
            "event":          "cache.get",
            "backend":        backend,
            "hits":           hits,
            "misses":         misses,
            "avg_latency_ms": avg_latency,
            "sdk_version":    _SDK_VERSION,
            "python_version": _python_version(),
            "fingerprint":    fingerprint,
        })

    # Aggregate cache.set events (additive, v0.5.2)
    set_events = [e for e in batch if e.get("event") == "cache.set"]
    if set_events:
        latencies   = [e.get("latency_ms", 0) for e in set_events if e.get("latency_ms")]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        backend     = set_events[0].get("backend", "")

        _post({
            "event":          "cache.set",
            "backend":        backend,
            "hits":           len(set_events),   # see "cache.set semantics" above
            "misses":         0,
            "avg_latency_ms": avg_latency,
            "sdk_version":    _SDK_VERSION,
            "python_version": _python_version(),
            "fingerprint":    fingerprint,
        })

    # Forward startup events (#41). One POST per flush cycle that contains
    # any startup event — multiple buffered startups in a single cycle
    # collapse to a single row on the dashboard, which is what we want
    # ("deployment alive" is a state, not a counter).
    #
    # Backend is unknown at startup time (Cache is typically instantiated
    # AFTER sulci.connect()), so we sniff it from any non-startup event
    # in the same batch. If no get/set has fired yet, backend goes out
    # as "" — gateway accepts empty backend, and the fingerprint dedupes
    # the deployment row once real traffic begins.
    if any(e.get("event") == "startup" for e in batch):
        sniffed_backend = next(
            (e["backend"] for e in batch
             if e.get("event") != "startup" and e.get("backend")),
            "",
        )
        _post({
            "event":          "startup",
            "backend":        sniffed_backend,
            "hits":           0,
            "misses":         0,
            "avg_latency_ms": 0.0,
            "sdk_version":    _SDK_VERSION,
            "python_version": _python_version(),
            "fingerprint":    fingerprint,
        })


def _build_fingerprint_for_batch(batch: list) -> Optional[str]:
    """Compute the per-deployment fingerprint for a batch of events.

    Returns ``None`` if the SDK's telemetry helpers can't be imported
    (e.g. test env with only ``__init__.py`` present) — the gateway
    schema accepts ``fingerprint=None``.
    """
    try:
        from sulci.config import get_machine_id
        from sulci.telemetry import build_fingerprint
        machine_id = get_machine_id()
        # Pull config bits off any event in the batch — they're all from
        # the same Cache instance in normal usage. We sniff backend from
        # the first event that carries one.
        backend = ""
        for e in batch:
            if e.get("backend"):
                backend = e["backend"]
                break
        return build_fingerprint(
            machine_id      = machine_id,
            backend         = backend,
            embedding_model = batch[0].get("embedding_model") if batch else None,
            threshold       = batch[0].get("threshold")       if batch else None,
            context_window  = batch[0].get("context_window")  if batch else None,
        )
    except Exception:
        return None


def _post(payload: dict) -> None:
    """POST one aggregated payload to ``/v1/telemetry``. Never raises.

    Strips any non-wire field via :func:`sulci.telemetry.coerce_to_wire`
    as a final guarantee against future flush() regressions accidentally
    leaking SDK-internal fields. The gateway uses ``extra='forbid'``;
    one stray field would HTTP-422 the entire batch.
    """
    try:
        try:
            from sulci.telemetry import coerce_to_wire
            payload = coerce_to_wire(payload)
        except Exception:
            # Helper unavailable (bare-init test env) — payload already
            # constructed with allowlisted keys upstream. Send as-is.
            pass
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

    Also registers an :mod:`atexit` hook to drain the buffer on process
    exit — the daemon flush thread dies the moment the main thread
    exits, so without this hook any events buffered since the last 30s
    tick would be silently lost. Affects every short-lived process:
    CLI commands, serverless invocations, test runs, demo scripts.
    """
    global _flush_thread_started
    if _flush_thread_started:
        return
    _flush_thread_started = True
    t = threading.Thread(target=_flush_loop, daemon=True, name="sulci-telemetry-flush")
    t.start()

    # v0.6.4: drain buffer on process exit. Wrapped in try/except so
    # it preserves the "telemetry never raises" contract; a stalled
    # gateway at exit delays termination by httpx's default timeout
    # but won't crash the process or break the user's code.
    atexit.register(_flush_on_exit)


def _flush_on_exit() -> None:
    """atexit handler — drain any remaining buffered events.

    Called automatically by the Python interpreter at normal process
    exit. No-op if telemetry was disabled (or never enabled) since
    last :func:`connect`."""
    try:
        if _telemetry_enabled:
            _flush()
    except Exception:
        # The telemetry contract is "never raise" — an exception here
        # would be reported by the atexit machinery as a warning. We
        # swallow silently to match the contract everywhere else.
        pass


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
    from sulci.async_cache import AsyncCache
    # v0.5.0 — new protocols and implementations (additive; ADR 0004 + ADR 0007)
    # Note: top-level `sulci.SessionStore` continues to be the legacy class
    # from sulci.context (backward compat). The new sulci.sessions.SessionStore
    # protocol is namespaced and accessed via `from sulci.sessions import SessionStore`.
    from sulci.sessions import InMemorySessionStore, RedisSessionStore
    from sulci.sinks    import EventSink, NullSink, RedisStreamSink, TelemetrySink, CacheEvent
    SyncCache = Cache   # naming symmetry with AsyncCache
except ImportError:
    Cache                = None  # type: ignore[assignment]
    ContextWindow        = None  # type: ignore[assignment]
    SessionStore         = None  # type: ignore[assignment]
    AsyncCache           = None  # type: ignore[assignment]
    SyncCache            = None  # type: ignore[assignment]
    InMemorySessionStore = None  # type: ignore[assignment]
    RedisSessionStore    = None  # type: ignore[assignment]
    EventSink            = None  # type: ignore[assignment]
    NullSink             = None  # type: ignore[assignment]
    RedisStreamSink      = None  # type: ignore[assignment]
    TelemetrySink        = None  # type: ignore[assignment]
    CacheEvent           = None  # type: ignore[assignment]


def _python_version() -> str:
    import sys
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


# ── Public exports ────────────────────────────────────────────────────────────

__all__ = [
    "Cache",
    "SyncCache",
    "AsyncCache",
    "ContextWindow",
    "SessionStore",            # legacy class (sulci.context)
    "InMemorySessionStore",    # new protocol impl (sulci.sessions)
    "RedisSessionStore",       # new protocol impl (sulci.sessions)
    "EventSink",               # new protocol (sulci.sinks)
    "NullSink",                # new protocol impl
    "RedisStreamSink",         # new protocol impl
    "TelemetrySink",           # new protocol impl
    "CacheEvent",              # event dataclass
    "connect",
]
