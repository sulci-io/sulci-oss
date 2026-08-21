# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_sinks.py — v0.5.0 event sink tests.

Includes critical privacy tests: verifies TelemetrySink and RedisStreamSink
NEVER leak query text, response text, or embedding vectors.
"""
import pytest
from unittest.mock import MagicMock, patch
from sulci.sinks import CacheEvent, EventSink, NullSink, TelemetrySink, RedisStreamSink
from sulci.sinks.telemetry import _ALLOWED_FIELDS, _scrub


@pytest.fixture
def sample_event():
    return CacheEvent(
        event_type="hit",
        tenant_id=42,
        user_id="alice",
        session_id="sess-123",
        backend_id="qdrant",
        embedding_model="minilm",
        similarity=0.92,
        latency_ms=15,
        context_depth=2,
        timestamp=1_700_000_000.0,
        plan="pro",
        metadata={"custom": "data"},
    )


class TestAllowlist:
    """The privacy firewall. These tests MUST stay green forever."""

    def test_allowlist_contents_are_stable(self):
        """Explicit pin of allowlist contents. Any change is a privacy review."""
        assert _ALLOWED_FIELDS == frozenset([
            "event_type", "tenant_id", "user_id", "session_id",
            "backend_id", "embedding_model", "similarity", "latency_ms",
            "context_depth", "timestamp",
            # v0.5.6 — sulci-oss #36. See sinks/telemetry.py for the
            # three-criteria rule that gates additions to this list.
            "plan",
        ])

    def test_metadata_dict_is_not_in_allowlist(self):
        """The metadata dict has arbitrary user content; must NEVER be shipped."""
        assert "metadata" not in _ALLOWED_FIELDS

    def test_scrub_excludes_metadata(self, sample_event):
        scrubbed = _scrub(sample_event)
        assert "metadata" not in scrubbed

    def test_scrub_preserves_allowlist_fields(self, sample_event):
        scrubbed = _scrub(sample_event)
        for field_name in _ALLOWED_FIELDS:
            assert field_name in scrubbed

    def test_scrub_excludes_any_added_attrs(self):
        """If a caller sneaks extra attrs onto the event, they must NOT ship."""
        event = CacheEvent(event_type="hit")
        # Attempt to inject — this is not supported via dataclass but tests
        # that our scrubber only emits allowlisted fields regardless.
        scrubbed = _scrub(event)
        allowed = set(_ALLOWED_FIELDS)
        assert set(scrubbed.keys()) <= allowed

    # ── v0.5.6 (sulci-oss #36) — plan field flows through scrubbing ──

    def test_scrub_preserves_plan_when_set(self):
        """plan='pro' on the event must reach scrubbed output verbatim."""
        event = CacheEvent(event_type="hit", plan="pro")
        scrubbed = _scrub(event)
        assert scrubbed["plan"] == "pro"

    def test_scrub_preserves_plan_when_none(self):
        """plan=None (the default) must come through as None — backward-compat."""
        event = CacheEvent(event_type="hit")
        scrubbed = _scrub(event)
        assert "plan" in scrubbed
        assert scrubbed["plan"] is None


class TestNullSink:
    def test_emit_is_noop(self, sample_event):
        sink = NullSink()
        sink.emit(sample_event)   # Should not raise

    def test_flush_is_noop(self):
        NullSink().flush()

    def test_conforms_to_protocol(self):
        assert isinstance(NullSink(), EventSink)


class TestTelemetrySinkValidation:
    def test_https_required(self):
        with pytest.raises(ValueError, match="HTTPS"):
            TelemetrySink(endpoint_url="http://insecure.example.com/events")

    def test_https_accepted(self):
        sink = TelemetrySink(endpoint_url="https://secure.example.com/events")
        assert sink is not None


class TestTelemetrySinkBatching:
    def test_does_not_flush_below_batch_size(self, sample_event):
        with patch("httpx.post") as mock_post:
            sink = TelemetrySink(
                endpoint_url="https://example.com/events",
                batch_size=10,
                flush_interval=999_999.0,  # Don't time-based flush in test
            )
            for _ in range(5):
                sink.emit(sample_event)
            assert not mock_post.called

    def test_flushes_when_batch_full(self, sample_event):
        with patch("httpx.post") as mock_post:
            mock_post.return_value.status_code = 200
            sink = TelemetrySink(
                endpoint_url="https://example.com/events",
                batch_size=3,
                flush_interval=999_999.0,
            )
            for _ in range(3):
                sink.emit(sample_event)
            assert mock_post.called

    def test_flush_sends_scrubbed_events(self, sample_event):
        with patch("httpx.post") as mock_post:
            mock_post.return_value.status_code = 200
            sink = TelemetrySink(
                endpoint_url="https://example.com/events",
                batch_size=1, flush_interval=999_999.0,
            )
            sink.emit(sample_event)

            assert mock_post.called
            sent_body = mock_post.call_args.kwargs["content"]

            # The payload must NOT contain the metadata field
            assert '"metadata"' not in sent_body

            # The payload MUST contain core allowlisted fields
            assert '"event_type"' in sent_body
            assert '"similarity"' in sent_body

    def test_network_error_does_not_raise(self, sample_event):
        with patch("httpx.post", side_effect=Exception("network down")):
            sink = TelemetrySink(
                endpoint_url="https://example.com/events",
                batch_size=1, flush_interval=999_999.0,
            )
            sink.emit(sample_event)   # Must not raise
            sink.flush()               # Must not raise


class TestTelemetrySinkConformance:
    def test_conforms_to_protocol(self):
        sink = TelemetrySink(endpoint_url="https://example.com/events")
        assert isinstance(sink, EventSink)


class TestRedisStreamSink:
    def test_emits_to_stream(self, sample_event):
        mock_redis = MagicMock()
        sink = RedisStreamSink(mock_redis, stream_key="test:events")
        sink.emit(sample_event)
        assert mock_redis.xadd.called

        # Verify scrubbed contents
        call_args = mock_redis.xadd.call_args
        entry = call_args[0][1]
        assert "metadata" not in entry   # privacy allowlist applied
        assert entry["event_type"] == "hit"

    def test_network_error_does_not_raise(self, sample_event):
        mock_redis = MagicMock()
        mock_redis.xadd.side_effect = Exception("redis down")

        sink = RedisStreamSink(mock_redis)
        sink.emit(sample_event)   # Must not raise

    def test_conforms_to_protocol(self):
        mock_redis = MagicMock()
        sink = RedisStreamSink(mock_redis)
        assert isinstance(sink, EventSink)
