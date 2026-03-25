"""
tests/test_connect.py
=====================
Unit tests for sulci.connect() — Step 4 (Week 2).

Coverage
--------
- _telemetry_enabled is False by default (most important invariant)
- connect() enables telemetry only when api_key is present
- connect(telemetry=False) registers key without enabling telemetry
- connect() resolves key from SULCI_API_KEY env var
- connect() with no key and no env var does not enable telemetry
- connect() does not start flush thread without a key
- connect() called twice does not start a second flush thread
- connect() emits startup event on successful connect
- connect() does not emit startup when telemetry=False
- _emit() is a no-op when telemetry is disabled
- _emit() is a no-op when api_key is None even if flag is True
- _emit() buffers events when telemetry is enabled
- _emit() never raises — telemetry must never crash the caller
- _flush() drains the buffer and sends one aggregated HTTP call
- _flush() sends correct X-Sulci-Key auth header
- _flush() uses 3s timeout
- _flush() swallows httpx.TimeoutException — never raises
- _flush() swallows generic exceptions — never raises
- _flush() is a no-op on empty buffer
- 50 concurrent threads emitting — no events lost (thread safety)
- Cache(telemetry=False) stores the flag correctly
- Cache telemetry=True by default
- Cache.get() does not emit when telemetry=False
- Cache.get() emits when telemetry=True and connect() called
"""

import os
import threading
import time
import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_module():
    """Reset all module-level telemetry state between tests."""
    import sulci
    sulci._api_key             = None
    sulci._telemetry_enabled   = False
    sulci._event_buffer        = []
    sulci._flush_thread_started = False

# ══════════════════════════════════════════════════════════════════════════════
# Default state
# ══════════════════════════════════════════════════════════════════════════════

class TestDefaultState:
    def setup_method(self):
        _reset_module()

    def test_telemetry_disabled_by_default(self):
        """Most critical invariant — nothing phones home without connect()."""
        import sulci
        assert sulci._telemetry_enabled is False

    def test_api_key_none_by_default(self):
        import sulci
        assert sulci._api_key is None

    def test_event_buffer_empty_by_default(self):
        import sulci
        assert sulci._event_buffer == []

    def test_emit_is_noop_by_default(self):
        """_emit() must not modify buffer when telemetry is off."""
        import sulci
        sulci._emit("cache.get", {"hits": 1, "misses": 0})
        assert sulci._event_buffer == []


# ══════════════════════════════════════════════════════════════════════════════
# connect()
# ══════════════════════════════════════════════════════════════════════════════

class TestConnect:
    def setup_method(self):
        _reset_module()

    def test_connect_enables_telemetry_with_key(self):
        """connect() with a valid key sets api_key and enables telemetry."""
        import sulci
        with patch("sulci._start_flush_thread"), patch("sulci._emit"):
            sulci.connect(api_key="sk-sulci-test123")
        assert sulci._telemetry_enabled is True
        assert sulci._api_key == "sk-sulci-test123"

    def test_connect_without_key_does_not_enable_telemetry(self):
        """connect() with no key and no env var leaves telemetry disabled."""
        import sulci
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SULCI_API_KEY", None)
            sulci.connect()
        assert sulci._telemetry_enabled is False
        assert sulci._api_key is None

    def test_connect_telemetry_false_sets_key_but_not_telemetry(self):
        """
        connect(telemetry=False) stores the key but leaves telemetry disabled.
        Thread must not be started.
        """
        import sulci
        with patch("sulci._start_flush_thread") as mock_thread, \
             patch("sulci._emit"):
            sulci.connect(api_key="sk-sulci-test123", telemetry=False)
        assert sulci._api_key           == "sk-sulci-test123"
        assert sulci._telemetry_enabled is False
        mock_thread.assert_not_called()

    def test_connect_reads_key_from_env(self):
        """connect() with no explicit key falls back to SULCI_API_KEY env var."""
        import sulci
        with patch.dict(os.environ, {"SULCI_API_KEY": "sk-sulci-fromenv"}), \
             patch("sulci._start_flush_thread"), patch("sulci._emit"):
            sulci.connect()
        assert sulci._api_key           == "sk-sulci-fromenv"
        assert sulci._telemetry_enabled is True

    def test_connect_explicit_key_takes_precedence_over_env(self):
        """Explicit api_key argument overrides SULCI_API_KEY env var."""
        import sulci
        with patch.dict(os.environ, {"SULCI_API_KEY": "sk-sulci-fromenv"}), \
             patch("sulci._start_flush_thread"), patch("sulci._emit"):
            sulci.connect(api_key="sk-sulci-explicit")
        assert sulci._api_key == "sk-sulci-explicit"

    def test_connect_starts_flush_thread(self):
        """connect() with a valid key starts the background flush thread."""
        import sulci
        with patch("sulci._start_flush_thread") as mock_thread, \
             patch("sulci._emit"):
            sulci.connect(api_key="sk-sulci-test123")
        mock_thread.assert_called_once()

    def test_connect_does_not_start_thread_without_key(self):
        """connect() without a key must NOT start the flush thread."""
        import sulci
        with patch("sulci._start_flush_thread") as mock_thread:
            sulci.connect(api_key=None)
        mock_thread.assert_not_called()

    def test_connect_emits_startup_event(self):
        """connect() emits a startup telemetry event when key is present."""
        import sulci
        with patch("sulci._start_flush_thread"), \
             patch("sulci._emit") as mock_emit:
            sulci.connect(api_key="sk-sulci-test123")
        mock_emit.assert_called_once_with("startup", {})

    def test_connect_does_not_emit_startup_when_telemetry_false(self):
        """connect(telemetry=False) must not emit any events."""
        import sulci
        with patch("sulci._emit") as mock_emit:
            sulci.connect(api_key="sk-sulci-test123", telemetry=False)
        mock_emit.assert_not_called()

    def test_connect_twice_starts_flush_thread_only_once(self):
        """
        Calling connect() twice must not create two flush threads.
        _flush_thread_started flag prevents duplicates.
        """
        import sulci
        with patch("sulci._emit"), \
             patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            sulci.connect(api_key="sk-sulci-test123")
            sulci.connect(api_key="sk-sulci-test123")
        assert mock_thread_cls.call_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# _emit()
# ══════════════════════════════════════════════════════════════════════════════

class TestEmit:
    def setup_method(self):
        _reset_module()

    def test_emit_buffers_event_when_enabled(self):
        """_emit() appends an event to _event_buffer when telemetry is on."""
        import sulci
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True
        sulci._emit("cache.get", {"hits": 1, "misses": 0, "backend": "sqlite"})
        assert len(sulci._event_buffer) == 1
        assert sulci._event_buffer[0]["event"] == "cache.get"
        assert sulci._event_buffer[0]["hits"]  == 1

    def test_emit_noop_when_disabled(self):
        """_emit() is a no-op when _telemetry_enabled is False."""
        import sulci
        sulci._telemetry_enabled = False
        sulci._emit("cache.get", {"hits": 1})
        assert sulci._event_buffer == []

    def test_emit_noop_when_no_key(self):
        """
        _emit() is a no-op when _api_key is None even if flag is True.
        Guards against state where flag was set but key was later cleared.
        """
        import sulci
        sulci._telemetry_enabled = True
        sulci._api_key           = None
        sulci._emit("cache.get", {"hits": 1})
        assert sulci._event_buffer == []

    def test_emit_multiple_events_appends(self):
        """Multiple _emit() calls all land in the buffer."""
        import sulci
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True
        for i in range(5):
            sulci._emit("cache.get", {"hits": 1, "misses": 0})
        assert len(sulci._event_buffer) == 5

    def test_emit_includes_timestamp(self):
        """Every buffered event includes a ts unix timestamp."""
        import sulci
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True
        before = time.time()
        sulci._emit("cache.get", {"hits": 1})
        after  = time.time()
        ts = sulci._event_buffer[0]["ts"]
        assert before <= ts <= after

    def test_emit_never_raises_on_exception(self):
        """
        _emit() must silently swallow all exceptions.
        Telemetry must never crash the user's application.
        """
        import sulci
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True
        with patch.object(sulci, "_event_buffer", side_effect=Exception("buffer error")):
            try:
                sulci._emit("cache.get", {"hits": 1})
            except Exception:
                pytest.fail("_emit() raised — it must never raise")


# ══════════════════════════════════════════════════════════════════════════════
# _flush()
# ══════════════════════════════════════════════════════════════════════════════

class TestFlush:
    def setup_method(self):
        _reset_module()

    def test_flush_noop_on_empty_buffer(self):
        """_flush() on an empty buffer must not call httpx.post."""
        import sulci
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True
        with patch("httpx.post") as mock_post:
            sulci._flush()
        mock_post.assert_not_called()

    def test_flush_sends_aggregated_payload(self):
        """
        _flush() aggregates buffered events into a single HTTP POST.
        hits and misses are summed; latency is averaged.
        """
        import sulci
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True
        sulci._event_buffer = [
            {"event": "cache.get", "hits": 1, "misses": 0, "latency_ms": 10.0, "backend": "sqlite", "ts": time.time()},
            {"event": "cache.get", "hits": 1, "misses": 0, "latency_ms": 20.0, "backend": "sqlite", "ts": time.time()},
            {"event": "cache.get", "hits": 0, "misses": 1, "latency_ms":  5.0, "backend": "sqlite", "ts": time.time()},
        ]
        with patch("httpx.post") as mock_post:
            sulci._flush()
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        assert payload["hits"]            == 2
        assert payload["misses"]          == 1
        assert payload["avg_latency_ms"]  == round((10 + 20 + 5) / 3, 2)
        assert payload["backend"]         == "sqlite"
        assert payload["sdk_version"]     == sulci._SDK_VERSION
        assert "python_version"           in payload

    def test_flush_sends_correct_auth_header(self):
        """_flush() sends X-Sulci-Key header with the module api_key."""
        import sulci
        sulci._api_key           = "sk-sulci-mykey"
        sulci._telemetry_enabled = True
        sulci._event_buffer = [
            {"event": "cache.get", "hits": 1, "misses": 0,
             "backend": "sqlite", "ts": time.time()}
        ]
        with patch("httpx.post") as mock_post:
            sulci._flush()
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Sulci-Key"] == "sk-sulci-mykey"

    def test_flush_drains_buffer(self):
        """Buffer is empty after a successful flush."""
        import sulci
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True
        sulci._event_buffer = [
            {"event": "cache.get", "hits": 1, "misses": 0,
             "backend": "sqlite", "ts": time.time()}
        ]
        with patch("httpx.post"):
            sulci._flush()
        assert sulci._event_buffer == []

    def test_flush_swallows_http_exception(self):
        """httpx.TimeoutException during flush must not propagate."""
        import sulci, httpx
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True
        sulci._event_buffer = [
            {"event": "cache.get", "hits": 1, "misses": 0,
             "backend": "sqlite", "ts": time.time()}
        ]
        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            sulci._flush()   # must not raise

    def test_flush_swallows_generic_exception(self):
        """Any exception during flush must not propagate."""
        import sulci
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True
        sulci._event_buffer = [
            {"event": "cache.get", "hits": 1, "misses": 0,
             "backend": "sqlite", "ts": time.time()}
        ]
        with patch("httpx.post", side_effect=RuntimeError("network down")):
            sulci._flush()   # must not raise

    def test_flush_uses_3s_timeout(self):
        """_flush() passes timeout=3.0 to httpx.post — never blocks indefinitely."""
        import sulci
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True
        sulci._event_buffer = [
            {"event": "cache.get", "hits": 1, "misses": 0,
             "backend": "sqlite", "ts": time.time()}
        ]
        with patch("httpx.post") as mock_post:
            sulci._flush()
        assert mock_post.call_args.kwargs["timeout"] == 3.0


# ══════════════════════════════════════════════════════════════════════════════
# Cache integration - TBD uncomment after week 2 step 6 completed
# ══════════════════════════════════════════════════════════════════════════════

class TestCacheIntegration:
    def setup_method(self):
        _reset_module()

    def test_cache_constructor_accepts_telemetry_false(self):
        """Cache(telemetry=False) stores the flag without error."""
        import sulci
        cache = sulci.Cache(backend="sqlite", telemetry=False)
        assert cache._telemetry is False

    def test_cache_constructor_telemetry_true_by_default(self):
        """Cache telemetry defaults to True at the instance level."""
        import sulci
        cache = sulci.Cache(backend="sqlite")
        assert cache._telemetry is True

    def test_cache_get_does_not_emit_when_telemetry_false(self):
        """
        Cache(telemetry=False).get() must not call _emit()
        even if sulci.connect() has been called.
        """
        import sulci
        with patch("sulci._start_flush_thread"):
            sulci.connect(api_key="sk-sulci-test123")
        cache = sulci.Cache(backend="sqlite", telemetry=False)
        with patch.object(sulci, "_emit") as mock_emit:
            cache.get("test query")
        mock_emit.assert_not_called()

    def test_cache_get_emits_when_telemetry_enabled(self):
        """
        Cache.get() calls _emit() with cache.get event when
        telemetry is True and sulci.connect() has been called.
        """
        import sulci
        with patch("sulci._start_flush_thread"):
            sulci.connect(api_key="sk-sulci-test123")
        cache = sulci.Cache(backend="sqlite", telemetry=True)
        cache.set("what is semantic caching", "it caches by meaning")
        with patch.object(sulci, "_emit") as mock_emit:
            cache.get("what is semantic caching")
        mock_emit.assert_called_once()
        event, data = mock_emit.call_args[0]
        assert event       == "cache.get"
        assert "hits"      in data
        assert "misses"    in data
        assert "latency_ms" in data


# ══════════════════════════════════════════════════════════════════════════════
# Thread safety
# ══════════════════════════════════════════════════════════════════════════════

class TestThreadSafety:
    def setup_method(self):
        _reset_module()

    def test_concurrent_emits_do_not_lose_events(self):
        """
        50 threads each emit 10 events = 500 total.
        All must be in the buffer — no lost writes under contention.
        """
        import sulci
        sulci._api_key           = "sk-sulci-test"
        sulci._telemetry_enabled = True

        def emit_batch():
            for _ in range(10):
                sulci._emit("cache.get", {"hits": 1, "misses": 0, "backend": "sqlite"})

        threads = [threading.Thread(target=emit_batch) for _ in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(sulci._event_buffer) == 500
