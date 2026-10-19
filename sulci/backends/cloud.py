# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

# sulci/backends/cloud.py
"""
SulciCloudBackend — routes cache operations to Sulci Cloud via HTTPS.

Zero infrastructure for the user. One parameter change from self-hosted:

    # Before
    cache = Cache(backend="sqlite", threshold=0.85)

    # After
    cache = Cache(backend="sulci", api_key="sk-sulci-...")   # 0.10.1: no threshold = the dashboard's

v0.6.0 transport contract (sulci-oss #62, umbrella #63)
-------------------------------------------------------
SulciCloudBackend is a **transport** for the entire `Cache.get/set` call,
not a `Backend` protocol implementer in the vector-search sense. The
gateway-side library (LibraryBackedCache) does ALL the engine work:
embedding via the embed-service container, ANN search in managed Qdrant,
event-stream emit for billing.

Public methods:
    remote_get(query, threshold, user_id, session_id)
        -> (response|None, similarity, context_depth)
    remote_set(query, response, user_id, session_id, ttl_seconds)
    delete_user(user_id) -> dict   # DELETE /v1/cache/user/{id}, id encoded; raises SulciErasureError (0.9.3)
    clear() -> dict                # DELETE /v1/cache/clear;                raises SulciErasureError (0.9.3)

Pre-v0.6.0 (search/store/upsert) embedded queries locally and sent
`{embedding, ...}` to the gateway — a parallel-implementation pattern that
ADR 0008 explicitly retired (archived at `archive/parallel-impl-final`,
tag `parallel-impl-final-v0.5.5`, commit a4e4ad0). Cache.get/set now
detects this transport via `hasattr(backend, "remote_get")` and bypasses
local embedding entirely.

v0.6.2 (sulci-oss #103 SDK companion, sulci-platform #103 gateway side)
----------------------------------------------------------------------
`delete_user(user_id)` and `clear()` now hit the canonical gateway
routes that landed with sulci-platform #103:
  - `DELETE /v1/cache/user/{user_id}`   (was `/v1/user/{user_id}` — silent 404)
  - `DELETE /v1/cache/clear`            (was `/v1/cache` — silent 404)

Both methods continue to never crash the customer's app, but failures
are no longer silent: they now produce a `log.warning(...)` with the
HTTP error so GDPR-relevant deletion failures are visible in customer
logs. The contract is still `-> None`; customers wanting strong
deletion confirmation should check the gateway access logs or run
`Cache.stats()` to verify the cache size dropped as expected.

0.9.3 replaces the paragraph above: the user id is percent-encoded into one path segment (`/` and `.`
included), and both methods return the gateway's answer when it confirms the erasure and raise
SulciErasureError — with the status and error code — whenever it does not. There is no silent path.
"""

import os
from urllib.parse import quote

import httpx
from typing import Optional

from sulci import __version__
# 0.10.0: SulciErasureError lives in sulci.errors — Cache.delete_user raises it on every backend, this module on
# the managed one. Imported here because this module raises it.
from sulci.errors import SulciErasureError


def _user_path(user_id) -> str:
    """``/v1/cache/user/<id>`` with the id as ONE path segment the gateway decodes once.

    ``quote(..., safe="")`` escapes ``/``, ``?``, ``#``, ``%``, spaces and non-ASCII (as UTF-8),
    but leaves ``.`` — and HTTP clients collapse the dot segments ``/user/.`` and ``/user/..``
    before sending (``..`` would become ``DELETE /v1/cache/``). Escaping every ``.`` as ``%2E``
    is harmless: the gateway decodes once.
    """
    return "/v1/cache/user/" + quote(str(user_id), safe="").replace(".", "%2E")


class SulciCloudBackend:
    """
    Thin HTTPS client wrapping the Sulci Cloud API.

    Preserves the exact same backend interface as all local backends
    so core.py needs zero changes to use it.

    Base URL resolution — precedence chain, first match wins:
        1. ``gateway_url=`` kwarg passed to ``__init__`` (VPC customers
           per v0.3.4)
        2. ``SULCI_GATEWAY`` environment variable (staging redirect —
           matches the same env var honored by ``sulci`` module-level
           telemetry pipeline since v0.5.5)
        3. ``self.CLOUD_URL`` default: ``https://api.sulci.io``

    Failure policy:
        - search() timeout or error  → returns (None, 0.0)  — treated as cache miss
        - upsert() failure           → silently ignored      — fire and forget
        - remote_get/remote_set never raise to the caller — a cache must never crash the app
        - delete_user() / clear() raise SulciErasureError unless the gateway confirms the
          erasure (0.9.3): an erasure that did not happen must not look like one that did
    """

    #: Tenant isolation is enforced server-side by the Sulci Cloud gateway,
    #: keyed off the api_key (1:1 with a tenant). Declared False here because
    #: the OSS conformance suite cannot reach the gateway to verify enforcement
    #: locally. A custom local-gateway test harness could flip this to True.
    ENFORCES_TENANT_ISOLATION: bool = False

    CLOUD_URL    = "https://api.sulci.io"
    USER_AGENT   = f"sulci/{__version__}"
    #: 0.10.1: remote_get(threshold=None) sends no threshold, so Cache hands None through when the caller set none.
    ACCEPTS_UNSET_THRESHOLD = True
    # 0.10.2: remote_get takes context_window= and remote_set takes standalone=; Cache passes them only to a
    # transport that says so (a custom transport written against 0.10.1 is not handed keywords it lacks).
    ACCEPTS_CONTEXT_WINDOW    = True
    ACCEPTS_STANDALONE        = True
    ACCEPTS_CONTEXT_THRESHOLD = True

    def __init__(
        self,
        api_key:     str,
        timeout:     float = 5.0,
        gateway_url: str   = "",
    ):
        if not api_key:
            raise ValueError(
                "api_key is required for backend='sulci'. "
                "Get your free key at https://sulci.io/signup"
            )

        self._api_key  = api_key
        self._timeout  = timeout

        # Base URL precedence — first match wins:
        #   1. explicit ``gateway_url=`` kwarg (VPC customers per v0.3.4)
        #   2. ``SULCI_GATEWAY`` env var — v0.5.5 shipped this for the
        #      telemetry path in ``sulci/__init__.py``; this closes the
        #      follow-up gap (``#TBD-2`` in the v0.5.5 changelog) where
        #      cache traffic wasn't reading the same env var, creating a
        #      split-brain state in staging (telemetry to staging URL,
        #      cache traffic to prod).
        #   3. ``self.CLOUD_URL`` default (production api.sulci.io)
        #
        # Env var is read at instance construction time, not import time —
        # so ``os.environ["SULCI_GATEWAY"] = "..."; Cache(...)`` works
        # without needing to set the var before ``import sulci``. Slightly
        # more flexible than the module-level pattern used by telemetry;
        # end-user visible effect is identical in the common case (env
        # vars are set before any Python code runs).
        if gateway_url:
            self._base_url = gateway_url.rstrip("/")
        else:
            env_gateway = os.environ.get("SULCI_GATEWAY", "").strip()
            self._base_url = env_gateway.rstrip("/") if env_gateway else self.CLOUD_URL

        self._client   = httpx.Client(
            base_url = self._base_url,
            headers  = {
                "X-Sulci-Key":   api_key,
                "Content-Type":  "application/json",
                "User-Agent":    self.USER_AGENT,
            },
            timeout  = httpx.Timeout(timeout),
        )

    # ── Public interface — matches local backend contract ─────────────────────

    def remote_get(
        self,
        query:      str,
        threshold:  Optional[float],
        *,
        user_id:    Optional[str] = None,
        session_id: Optional[str] = None,
        context_window: Optional[int] = None,
        context_threshold: Optional[float] = None,
    ) -> tuple:
        """
        Forward a cache lookup to the Sulci Cloud gateway as a TRANSPORT
        operation — the gateway-side library does the embedding, ANN
        search, and event emit.

        v0.6.0 (sulci-oss #62) — renamed from `search()`. Wire payload is
        now `{query, threshold, user_id, session_id}` matching the
        gateway's CacheGetRequest pydantic model exactly. Pre-v0.6.0 the
        SDK sent `{embedding, tenant_id, ...}` which the gateway rejected
        as 422 (silently swallowed by the outer except clause below).
        Tenant isolation is enforced server-side from the api_key auth
        context; tenant_id is no longer sent on the wire.

        0.10.1: ``threshold=None`` sends no ``threshold`` at all, so the gateway applies the tenant's dashboard
        setting; ``Cache`` passes ``None`` when the caller set no threshold anywhere. A number is sent as given.

        Returns:
            (response, similarity, context_depth)
            response is None on miss.
            Falls back to (None, 0.0, 0) on any network error — never raises.

        0.10.2: ``context_window=None`` sends no ``context_window``, so the gateway applies the tenant's dashboard
        setting; a number is sent as given (``Cache`` keeps it within the gateway's 0..20). ``context_threshold``
        likewise: None sends none; a number is the threshold for a lookup that blends. A gateway older than
        sulci-platform 0248 ignores the field.
        """
        body = {"query": query, "user_id": user_id, "session_id": session_id}
        if threshold is not None:
            body["threshold"] = threshold
        if context_window is not None:
            body["context_window"] = context_window
        if context_threshold is not None:
            body["context_threshold"] = context_threshold
        try:
            resp = self._client.post("/v1/cache/get", json=body)
            resp.raise_for_status()
            data = resp.json()
            return (
                data.get("response"),
                float(data.get("similarity", 0.0)),
                int(data.get("context_depth", 0)),
            )
        except httpx.TimeoutException:
            # Timeout — treat as cache miss, never crash
            return None, 0.0, 0
        except Exception:
            # Any other error — treat as cache miss, never crash
            return None, 0.0, 0

    def remote_set(
        self,
        query:       str,
        response:    str,
        *,
        user_id:     Optional[str] = None,
        session_id:  Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        standalone:  bool          = False,
    ) -> None:
        """
        Forward a cache write to the Sulci Cloud gateway as a TRANSPORT
        operation. The gateway-side library handles embedding, Qdrant
        upsert, and the `cache.set` billing event.

        v0.6.0 (sulci-oss #62) — replaces both `store()` and `upsert()`.
        Wire payload is `{query, response, user_id, session_id, ttl_seconds}`
        matching the gateway's CacheSetRequest pydantic model exactly.
        The pre-v0.6.0 split between `store(key, embedding, query, response,
        tenant_id, metadata, ...)` and `upsert(embedding, query, response, ...)`
        was an artifact of the Backend protocol's vector-search shape; with
        the cloud as a TRANSPORT (not a backend), a single canonical method
        suffices.

        Fire-and-forget — silently ignores all errors. The user's app must
        never crash on a failed cache write.

        0.10.2: ``standalone=True`` sends ``"standalone": true`` — store the entry even when the session already
        holds a turn (the gateway's follow-up rule; see ``Cache.set``). False sends nothing, as before.
        """
        body = {
            "query":       query,
            "response":    response,
            "user_id":     user_id,
            "session_id":  session_id,
            "ttl_seconds": ttl_seconds,
        }
        if standalone:
            body["standalone"] = True
        try:
            self._client.post(
                "/v1/cache/set",
                json=body,
            )
        except Exception:
            pass   # Never crash the user's app on a failed write

    def delete_user(self, user_id: str) -> dict:
        """Delete all cache entries — and conversation history — for one user within this tenant.

        Sends ``DELETE /v1/cache/user/{user_id}`` with the id percent-encoded: every character
        outside ``A-Z a-z 0-9 - _ ~`` is escaped, ``/`` and ``.`` included, so the id is one
        path segment the gateway decodes once. (Before 0.9.3 it was sent as-is: ``"../clear"``
        reached ``DELETE /v1/cache/clear``, and ``"a?b"`` / ``"a#b"`` erased the user ``"a"``.)

        Returns the gateway's answer — ``{"status": "ok", "deleted": N, "tenant_id": ...,
        "user_id": ..., ...}`` — only when it confirms the erasure. Otherwise raises
        :class:`SulciErasureError`: a transport error, any non-2xx answer such as ``422``
        (``validation_error``: e.g. a reserved id) or ``503`` (``erasure_incomplete``), a 2xx answer
        that is not that JSON, or an id that cannot be encoded (then nothing is sent).

        Before 0.9.3 a non-2xx answer was not checked at all: no exception, no log line.
        """
        return self._erase("delete_user", lambda: _user_path(user_id))

    def clear(self) -> dict:
        """Clear all cache entries — and conversation history — for this tenant.

        Sends ``DELETE /v1/cache/clear``. Returns the gateway's answer (``{"status": "ok",
        "deleted": N, "tenant_id": ..., ...}``) only when it confirms the erasure; otherwise raises
        :class:`SulciErasureError`, as :meth:`delete_user` does. ``Cache.clear()`` lets it propagate.

        Before 0.9.3 a non-2xx answer (e.g. ``503 erasure_incomplete``) was not checked.
        """
        return self._erase("clear", lambda: "/v1/cache/clear")

    def _erase(self, operation, make_path) -> dict:
        """One DELETE. Returns the gateway's answer when it confirms the erasure — a 2xx answer whose
        JSON body says {"status": "ok"} — and raises SulciErasureError otherwise. Anything else (an
        HTML page from a misconfigured gateway URL, an empty 204) is not proof that anything was
        erased. An id that cannot be encoded (a lone surrogate, a failing __str__) raises the same
        error, with the cause chained, before anything is sent."""
        try:
            path = make_path()
        except Exception as e:
            raise SulciErasureError(operation, None, f"unencodable user_id ({type(e).__name__})") from e
        try:
            resp = self._client.delete(path)
        except Exception as e:
            # The type only: a transport's text can carry the request URL, and so the user id; the text stays
            # on __cause__ for whoever handles the exception.
            raise SulciErasureError(operation, None, type(e).__name__) from e
        try:
            body = resp.json()
        except Exception:
            body = None
        status = resp.status_code
        if isinstance(status, int) and 200 <= status < 300:
            if isinstance(body, dict) and body.get("status") == "ok":
                return body
            raise SulciErasureError(operation, status, "unexpected_response", body)
        error = (body.get("error") if isinstance(body, dict) else None) or "error"
        raise SulciErasureError(operation, status, str(error), body)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying httpx client."""
        try:
            self._client.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        return (
            f"SulciCloudBackend("
            f"url={self._base_url!r}, "
            f"key_prefix={self._api_key[:16]!r}, "
            f"timeout={self._timeout})"
        )
