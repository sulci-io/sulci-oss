# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sinks/redis_stream.py — RedisStreamSink (v0.5.0)

Writes cache events to a Redis Stream for downstream consumers
(billing pipeline, analytics, etc.). Used by sulci-platform's
billing infrastructure.

Uses the same allowlist as TelemetrySink — never emits query/response/vectors.
"""
from __future__ import annotations
import json
import logging
from dataclasses import asdict
from typing import Optional

from sulci.sinks.protocol import CacheEvent, EventSink
from sulci.sinks.telemetry import _ALLOWED_FIELDS, _scrub

log = logging.getLogger(__name__)


class RedisStreamSink(EventSink):
    """
    Write events to a Redis Stream.

    Args:
        redis_client: Configured redis.Redis instance
        stream_key:   Redis Stream name (default "sulci:events")
        max_length:   Trim stream to this length on each write (approximate)
    """

    def __init__(
        self,
        redis_client,
        stream_key: str = "sulci:events",
        max_length: Optional[int] = 1_000_000,
    ):
        try:
            import redis   # noqa: F401
        except ImportError:
            raise ImportError(
                "redis package not installed. "
                'Install with: pip install "sulci[redis]"'
            )
        self._redis = redis_client
        self._stream = stream_key
        self._max_length = max_length

    def emit(self, event: CacheEvent) -> None:
        scrubbed = _scrub(event)  # privacy firewall — same as TelemetrySink

        # Redis stream entries are flat string maps; serialize non-str values
        entry = {k: (json.dumps(v) if not isinstance(v, (str, int, float, bool, type(None))) else ("" if v is None else str(v)))
                 for k, v in scrubbed.items()}

        try:
            if self._max_length:
                self._redis.xadd(
                    self._stream, entry,
                    maxlen=self._max_length, approximate=True,
                )
            else:
                self._redis.xadd(self._stream, entry)
        except Exception as e:  # noqa: BLE001
            # Never raise from a sink; caller's cache call must not fail
            log.debug("RedisStreamSink emit failed: %s", e)

    def flush(self) -> None:
        # Redis XADD is synchronous — no buffering to flush
        pass
