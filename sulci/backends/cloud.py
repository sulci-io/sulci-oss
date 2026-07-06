# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

# sulci/backends/cloud.py
"""
SulciCloudBackend — routes cache operations to Sulci Cloud via HTTPS.

Zero infrastructure for the user. One parameter change from self-hosted:

    # Before
    cache = Cache(backend="sqlite", threshold=0.85)

    # After
    cache = Cache(backend="sulci", api_key="sk-sulci-...", threshold=0.85)

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
    delete_user(user_id)   # v0.6.2: real deletion via DELETE /v1/cache/user/{id}
    clear()                # v0.6.2: real wipe via DELETE /v1/cache/clear

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
"""

import logging
import os

import httpx
from typing import Optional

from sulci import __version__

log = logging.getLogger(__name__)


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
        - Never raises to the caller — the user's app must never crash due to cache
    """

    #: Tenant isolation is enforced server-side by the Sulci Cloud gateway,
    #: keyed off the api_key (1:1 with a tenant). Declared False here because
    #: the OSS conformance suite cannot reach the gateway to verify enforcement
    #: locally. A custom local-gateway test harness could flip this to True.
    ENFORCES_TENANT_ISOLATION: bool = False

    CLOUD_URL    = "https://api.sulci.io"
    USER_AGENT   = f"sulci/{__version__}"

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
        threshold:  float,
        *,
        user_id:    Optional[str] = None,
        session_id: Optional[str] = None,
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

        Returns:
            (response, similarity, context_depth)
            response is None on miss.
            Falls back to (None, 0.0, 0) on any network error — never raises.
        """
        try:
            resp = self._client.post(
                "/v1/cache/get",
                json={
                    "query":      query,
                    "threshold":  threshold,
                    "user_id":    user_id,
                    "session_id": session_id,
                },
            )
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
        """
        try:
            self._client.post(
                "/v1/cache/set",
                json={
                    "query":       query,
                    "response":    response,
                    "user_id":     user_id,
                    "session_id":  session_id,
                    "ttl_seconds": ttl_seconds,
                },
            )
        except Exception:
            pass   # Never crash the user's app on a failed write

    def delete_user(self, user_id: str) -> None:
        """Delete all cache entries for a given user_id within this tenant.

        Sends `DELETE /v1/cache/user/{user_id}`. The gateway resolves the
        tenant from the `X-Sulci-Key` auth header and deletes only points
        where BOTH tenant_id AND user_id match — strict GDPR isolation,
        no cross-tenant leakage possible (see sulci-platform #103 for
        the gateway-side filter).

        Returns None unconditionally. The method never raises — a failure
        in the deletion path will not crash the customer's app. However,
        unlike pre-v0.6.2, failures are no longer silent: they produce
        a `log.warning(...)` with the HTTP error so customers can see
        deletion problems in their application logs.

        Notes for GDPR auditing:
          - The gateway response includes the count of points actually
            deleted (`{"status": "ok", "deleted": N, "tenant_id": ...,
            "user_id": ...}`), but this SDK method returns None to
            preserve the v0.6.x contract. Inspect gateway access logs
            or run `Cache.stats()` post-call to verify deletion.
          - Pre-v0.6.2 the SDK sent `DELETE /v1/user/{user_id}` (route
            never existed on the gateway, silently 404'd, no deletion
            ever happened). If you were relying on that behavior for
            14+ months between v0.3.0 and v0.6.2, your customer data
            was likely retained longer than you intended. v0.6.2 closes
            this gap.
        """
        try:
            self._client.delete(f"/v1/cache/user/{user_id}")
        except Exception as e:
            # GDPR-relevant: surface failures via log.warning instead of
            # swallowing them silently. The contract is still non-crashing
            # (return None unconditionally), but customers and operators
            # can now see something went wrong in standard logs.
            log.warning(
                "sulci.cloud.delete_user failed for user_id=%s: %s",
                user_id, e,
            )

    def clear(self) -> None:
        """Clear all cache entries for this tenant.

        Sends `DELETE /v1/cache/clear`. Tenant-scoped — wipes only the
        authenticated tenant's data; other tenants on shared infrastructure
        are unaffected (see sulci-platform #103 gateway-side filter).

        Returns None unconditionally. Like `delete_user`, this method
        never raises, but failures now produce a `log.warning(...)`
        instead of being silently swallowed.

        Pre-v0.6.2 sent `DELETE /v1/cache` (route never existed; silent
        404). v0.6.2 closes this gap — the gateway-side route exists
        as of sulci-platform #103.
        """
        try:
            self._client.delete("/v1/cache/clear")
        except Exception as e:
            log.warning("sulci.cloud.clear failed: %s", e)

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
