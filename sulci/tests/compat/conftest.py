# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/tests/compat/conftest.py
==============================
Fixtures for the SessionStore and EventSink conformance suites.

To validate a custom SessionStore or EventSink implementation, add a
factory entry to the appropriate registry below and run:

    pytest sulci/tests/compat/

Each factory returns either:
  - A live instance ready to use, OR
  - None, signaling "not available locally" (missing dep, no server, etc.)
    in which case parametrized tests for that impl are skipped, not failed.

External implementers can override these fixtures from their own
conftest.py without modifying this file.
"""
from __future__ import annotations
from typing import Any, Optional, Callable, List, Tuple

import pytest


# -----------------------------------------------------------------------------
# SessionStore registry
# -----------------------------------------------------------------------------

def _make_in_memory_session() -> Optional[Any]:
    from sulci.sessions import InMemorySessionStore
    return InMemorySessionStore()


def _make_redis_session() -> Optional[Any]:
    try:
        import redis
    except ImportError:
        return None
    try:
        client = redis.Redis(
            host="localhost", port=6379, db=15, decode_responses=True
        )
        client.ping()
        client.flushdb()
    except Exception:
        return None
    from sulci.sessions import RedisSessionStore
    return RedisSessionStore(client, key_prefix="sulci:conformance:session:")


SESSION_STORE_FACTORIES: List[Tuple[str, Callable[[], Optional[Any]]]] = [
    ("InMemorySessionStore", _make_in_memory_session),
    ("RedisSessionStore",    _make_redis_session),
]


@pytest.fixture(params=SESSION_STORE_FACTORIES, ids=lambda f: f[0])
def session_store(request):
    """
    Yields each registered SessionStore implementation in turn.
    Skips parametrizations that can't be constructed locally.
    """
    name, factory = request.param
    inst = factory()
    if inst is None:
        pytest.skip(f"{name}: not available locally")
    yield inst
    # Best-effort cleanup for stateful stores
    try:
        if hasattr(inst, "_redis"):
            inst._redis.flushdb()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# EventSink registry
# -----------------------------------------------------------------------------

def _make_null_sink() -> Optional[Any]:
    from sulci.sinks import NullSink
    return NullSink()


def _make_telemetry_sink() -> Optional[Any]:
    try:
        import httpx  # noqa: F401
    except ImportError:
        return None
    from sulci.sinks import TelemetrySink
    # HTTPS-only dummy endpoint. The protocol spec requires emit/flush
    # to never raise on network failure, so unreachable URLs are fine —
    # the conformance suite explicitly tests this contract.
    return TelemetrySink(
        endpoint_url="https://127.0.0.1:1/sulci-conformance-test",
        batch_size=1,
        flush_interval=0.01,
        timeout_seconds=0.1,
    )


def _make_redis_stream_sink() -> Optional[Any]:
    try:
        import redis
    except ImportError:
        return None
    try:
        client = redis.Redis(host="localhost", port=6379, db=15)
        client.ping()
        # Defensive setup clean — drop the test stream if a prior run
        # left entries behind. Redis stream keys persist; without this,
        # tests asserting "fresh stream behaviour" can see leftover state.
        client.delete("sulci:conformance:events")
    except Exception:
        return None
    from sulci.sinks import RedisStreamSink
    return RedisStreamSink(
        client, stream_key="sulci:conformance:events", max_length=1000,
    )


EVENT_SINK_FACTORIES: List[Tuple[str, Callable[[], Optional[Any]]]] = [
    ("NullSink",        _make_null_sink),
    ("TelemetrySink",   _make_telemetry_sink),
    ("RedisStreamSink", _make_redis_stream_sink),
]


@pytest.fixture(params=EVENT_SINK_FACTORIES, ids=lambda f: f[0])
def event_sink(request):
    """
    Yields each registered EventSink implementation in turn.
    Skips parametrizations that can't be constructed locally.
    """
    name, factory = request.param
    inst = factory()
    if inst is None:
        pytest.skip(f"{name}: not available locally")
    yield inst
    # Best-effort cleanup for stateful sinks. NullSink and TelemetrySink
    # are in-memory; only Redis-backed sinks leak state across tests.
    try:
        if hasattr(inst, "_redis"):
            inst._redis.delete("sulci:conformance:events")
    except Exception:
        pass
