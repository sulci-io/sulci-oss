# sulci/backends/cloud.py
"""
SulciCloudBackend — routes cache operations to Sulci Cloud via HTTPS.

Zero infrastructure for the user. One parameter change from self-hosted:

    # Before
    cache = Cache(backend="sqlite", threshold=0.85)

    # After
    cache = Cache(backend="sulci", api_key="sk-sulci-...", threshold=0.85)

All public methods match the contract expected by Cache in core.py:
    search(embedding, threshold, user_id, now) -> (response|None, similarity)
    upsert(embedding, query, response, user_id, ttl_seconds)
    delete_user(user_id)
    clear()
"""

import httpx
from typing import Optional


class SulciCloudBackend:
    """
    Thin HTTPS client wrapping the Sulci Cloud API.

    Preserves the exact same backend interface as all local backends
    so core.py needs zero changes to use it.

    Failure policy:
        - search() timeout or error  → returns (None, 0.0)  — treated as cache miss
        - upsert() failure           → silently ignored      — fire and forget
        - Never raises to the caller — the user's app must never crash due to cache
    """

    CLOUD_URL    = "https://api.sulci.io"
    USER_AGENT   = "sulci/0.3.0"

    def __init__(
        self,
        api_key:  str,
        timeout:  float = 5.0,
    ):
        if not api_key:
            raise ValueError(
                "api_key is required for backend='sulci'. "
                "Get your free key at https://sulci.io/signup"
            )

        self._api_key = api_key
        self._timeout = timeout
        self._client  = httpx.Client(
            base_url = self.CLOUD_URL,
            headers  = {
                "X-Sulci-Key":   api_key,
                "Content-Type":  "application/json",
                "User-Agent":    self.USER_AGENT,
            },
            timeout  = httpx.Timeout(timeout),
        )

    # ── Public interface — matches local backend contract ─────────────────────

    def search(
        self,
        embedding:  list,
        threshold:  float,
        user_id:    Optional[str] = None,
        now:        Optional[float] = None,
    ) -> tuple:
        """
        Semantic lookup via cloud API.

        Returns:
            (response, similarity) where response is None on miss.
            Falls back to (None, 0.0) on any network error — never raises.
        """
        try:
            resp = self._client.post(
                "/v1/get",
                json={
                    "embedding": embedding,
                    "threshold": threshold,
                    "user_id":   user_id,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return (
                data.get("response"),
                float(data.get("similarity", 0.0)),
            )
        except httpx.TimeoutException:
            # Timeout — treat as cache miss, never crash
            return None, 0.0
        except Exception:
            # Any other error — treat as cache miss, never crash
            return None, 0.0

    def upsert(
        self,
        embedding:   list,
        query:       str,
        response:    str,
        user_id:     Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Store a cache entry in the cloud.
        Fire-and-forget — silently ignores all errors.
        """
        try:
            self._client.post(
                "/v1/set",
                json={
                    "embedding":   embedding,
                    "query":       query,
                    "response":    response,
                    "user_id":     user_id,
                    "ttl_seconds": ttl_seconds,
                },
            )
        except Exception:
            pass   # Never crash the user's app on a failed write

    def delete_user(self, user_id: str) -> None:
        """Delete all cache entries for a given user_id."""
        try:
            self._client.delete(f"/v1/user/{user_id}")
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all cache entries for this tenant."""
        try:
            self._client.delete("/v1/cache")
        except Exception:
            pass

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
            f"url={self.CLOUD_URL!r}, "
            f"key_prefix={self._api_key[:16]!r}, "
            f"timeout={self._timeout})"
        )
