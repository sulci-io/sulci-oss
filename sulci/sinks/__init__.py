# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
sulci.sinks — v0.5.0 event sink protocols and implementations.

Public API:
    CacheEvent          (dataclass)
    EventSink           (protocol)
    NullSink            (default no-op)
    TelemetrySink       (HTTPS POST with field allowlist)
    RedisStreamSink     (for platform billing pipeline)
"""
from sulci.sinks.protocol import CacheEvent, EventSink, query_hash
from sulci.sinks.null import NullSink
from sulci.sinks.telemetry import TelemetrySink
from sulci.sinks.redis_stream import RedisStreamSink

__all__ = [
    "CacheEvent", "EventSink", "query_hash",
    "NullSink", "TelemetrySink", "RedisStreamSink",
]
