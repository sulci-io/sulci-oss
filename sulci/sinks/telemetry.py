# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sinks/telemetry.py — TelemetrySink (v0.5.0)

HTTPS POST sink with strict field allowlist.

Privacy guarantee: TelemetrySink NEVER sends query text, response text,
or embedding vectors externally. Only the allowlisted metadata fields
leave the process. This is enforced in code (not convention) via
_ALLOWED_FIELDS below.
"""
from __future__ import annotations
import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict
from typing import Optional

from sulci.sinks.protocol import CacheEvent, EventSink

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# PRIVACY-CRITICAL: the only fields ever shipped to external endpoints.
# Modifying this list requires explicit review; changes could leak data.
#
# Rule for adding fields: a candidate must satisfy ALL THREE criteria
# before it goes in this list. The rule was articulated when `plan` was
# added in v0.5.6 and is meant to keep this list from drifting toward
# "anything we find useful":
#
#   (a) LOW CARDINALITY — the field's domain is small and bounded
#       (e.g. an enum, a tier label, a country code). Free-text fields
#       must NEVER be added; they're indistinguishable from arbitrary
#       user content.
#
#   (b) ALREADY KNOWN TO THE RECIPIENT — the receiving service can
#       independently derive the value from the auth context it already
#       has. Adding the field doesn't expose anything the recipient
#       didn't already know; it just removes a join.
#
#   (c) EXPLICITLY BILLING- OR ROUTING-RELEVANT — the field exists
#       because a downstream consumer needs it for billing attribution,
#       routing, or rate-limiting. Fields that are merely "nice to
#       have for analytics" are not enough; they belong in the
#       customer's own metadata sink, not the privacy-firewalled one.
# ----------------------------------------------------------------------
_ALLOWED_FIELDS = frozenset([
    "event_type",
    "tenant_id",
    "user_id",          # may contain PII; caller's responsibility to anonymize
    "session_id",       # may contain PII; caller's responsibility to anonymize
    "backend_id",
    "embedding_model",
    "similarity",
    "latency_ms",
    "context_depth",
    "timestamp",
    # v0.5.6 — sulci-oss #36. Customer plan tier (free / pro / business /
    # enterprise / oss_connect). Satisfies all three rules above:
    # (a) closed enum of ~5 values; (b) recipient already knows the
    # plan from the API key auth lookup; (c) sulci-platform's billing
    # pipeline routes events by plan and was previously doing a
    # per-event Postgres join to recover this.
    "plan",
])


def _scrub(event: CacheEvent) -> dict:
    """
    Return a dict containing ONLY allowlisted fields.
    Never includes query, response, embedding, or metadata dict contents.

    This is the privacy firewall. Never modify without careful review.
    """
    raw = asdict(event)
    return {k: v for k, v in raw.items() if k in _ALLOWED_FIELDS}


class TelemetrySink(EventSink):
    """
    Batch events, POST to a telemetry endpoint over HTTPS.

    Args:
        endpoint_url:  HTTPS URL to POST batches to
        api_key:       Optional bearer token
        batch_size:    Events accumulated before flush (default 100)
        flush_interval: Seconds before auto-flush even if batch not full
        timeout_seconds: HTTP timeout per batch
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        flush_interval: float = 30.0,
        timeout_seconds: float = 5.0,
    ):
        if not endpoint_url.startswith("https://"):
            raise ValueError(
                "TelemetrySink requires HTTPS endpoint. "
                "Plain HTTP is not acceptable for telemetry."
            )
        try:
            import httpx    # noqa: F401
        except ImportError:
            raise ImportError(
                "httpx required for TelemetrySink. "
                'Install with: pip install "sulci[cloud]"'
            )
        self._endpoint = endpoint_url
        self._api_key = api_key
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._timeout = timeout_seconds

        self._buf: deque = deque()
        self._lock = threading.Lock()
        self._last_flush = time.time()

    def emit(self, event: CacheEvent) -> None:
        scrubbed = _scrub(event)   # privacy firewall
        with self._lock:
            self._buf.append(scrubbed)
            should_flush = (
                len(self._buf) >= self._batch_size
                or (time.time() - self._last_flush) > self._flush_interval
            )
        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buf:
                return
            batch = list(self._buf)
            self._buf.clear()
            self._last_flush = time.time()

        try:
            import httpx
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            response = httpx.post(
                self._endpoint,
                headers=headers,
                content=json.dumps({"events": batch}),
                timeout=self._timeout,
            )
            if response.status_code >= 400:
                log.warning(
                    "TelemetrySink received %d from endpoint", response.status_code
                )
        except Exception as e:  # noqa: BLE001 - must not raise
            # Drop events on network error. Degrade silently per protocol spec.
            log.debug("TelemetrySink flush failed: %s", e)
