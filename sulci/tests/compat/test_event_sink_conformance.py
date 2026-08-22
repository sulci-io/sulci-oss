# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""Public conformance suite for EventSink implementations."""
import pytest
from sulci.sinks import CacheEvent, EventSink


@pytest.fixture
def sample_event():
    return CacheEvent(
        event_type="hit",
        backend_id="test",
        embedding_model="test",
        similarity=0.9,
        latency_ms=10,
        timestamp=1_700_000_000.0,
    )


class TestEventSinkProtocol:
    def test_conforms_to_protocol(self, event_sink):
        assert isinstance(event_sink, EventSink)


class TestEmit:
    def test_emit_accepts_cache_event(self, event_sink, sample_event):
        event_sink.emit(sample_event)   # Must not raise

    def test_emit_handles_minimal_event(self, event_sink):
        event_sink.emit(CacheEvent(event_type="miss"))

    def test_emit_handles_various_event_types(self, event_sink):
        for et in ("hit", "miss", "set", "clear"):
            event_sink.emit(CacheEvent(event_type=et))


class TestFlush:
    def test_flush_does_not_raise(self, event_sink):
        event_sink.flush()

    def test_flush_after_emit(self, event_sink, sample_event):
        event_sink.emit(sample_event)
        event_sink.flush()


class TestNeverRaises:
    """Critical invariant: cache operations must never fail due to sink errors."""

    def test_emit_does_not_raise_on_normal_input(self, event_sink, sample_event):
        try:
            event_sink.emit(sample_event)
        except Exception as e:
            pytest.fail(f"emit() raised {e!r}; sinks must never raise from emit()")
