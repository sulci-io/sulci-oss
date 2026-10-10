# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_cloud_erasure.py — SulciCloudBackend.delete_user / clear (0.9.3).

1. The user id is percent-encoded into ONE path segment. Measured on 0.9.2 with an httpx
   MockTransport: ``delete_user("../clear")`` sent ``DELETE /v1/cache/clear`` (the tenant's whole
   cache); ``"a?b=c"`` and ``"a#frag"`` erased the user ``"a"``; ``"org/alice"`` sent a two-segment
   path. ``quote(id, safe="")`` alone is not enough: it leaves ``.``, and httpx collapses the dot
   segments ``/user/.`` and ``/user/..`` — so ``.`` is escaped too.
2. An erasure the gateway does not confirm raises. 0.9.2 never looked at the status: a ``422`` or a
   ``503 erasure_incomplete`` produced no exception and no log line. Now ``SulciErasureError``, with
   the status and the gateway's error code, and nothing logged — the exception is the report — and
   ``Cache.clear()`` lets it propagate before resetting anything. Success returns the gateway's
   answer. Measured on oss-0005 (``measure_d1.py``): every failure returned ``None`` plus one WARNING,
   and ``Cache.clear()`` reset its stats as if the erasure had happened.

These run against httpx's MockTransport, so they see the bytes httpx actually sends.
"""
import logging
from urllib.parse import unquote

import httpx
import pytest

from sulci.backends.cloud import SulciCloudBackend, SulciErasureError

PREFIX = b"/v1/cache/user/"


def _backend(handler):
    b = SulciCloudBackend(api_key="sk-sulci-test0000000000")
    b._client.close()
    b._client = httpx.Client(base_url="https://api.sulci.io", transport=httpx.MockTransport(handler))
    return b


def _recorder(status=200, body=None):
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(status, json=body if body is not None else {"status": "ok", "deleted": 1, "tenant_id": 7})
    return seen, handler


IDS = ["alice", "user-42", "org/alice", "a/b/c", "/", "alice/", "../clear", "..", ".", "./x", "a?b=c",
       "a#frag", "café", "émoji 🙂", "a\nb", "alice\n", "a b", "100%", "%2F", "a+b", "global/", "x" * 300]


class TestTheIdIsOnePathSegment:
    @pytest.mark.parametrize("uid", IDS, ids=[repr(i)[:24] for i in IDS])
    def test_the_request_names_exactly_this_user(self, uid):
        seen, h = _recorder(body={"status": "ok", "deleted": 1, "tenant_id": 7, "user_id": uid})
        _backend(h).delete_user(uid)
        assert len(seen) == 1 and seen[0].method == "DELETE"
        raw = seen[0].url.raw_path
        assert raw.startswith(PREFIX), raw
        seg = raw[len(PREFIX):]
        assert b"/" not in seg and b"?" not in seg and b"#" not in seg, f"{uid!r} was sent as {raw!r}"
        assert seg not in (b"", b".", b".."), raw
        assert unquote(seg.decode("ascii")) == uid, f"{uid!r} decodes as {unquote(seg.decode())!r}"

    @pytest.mark.parametrize("uid", ["../clear", "..", "../../v1/cache/clear", "./../clear"])
    def test_no_id_reaches_the_tenant_wide_clear(self, uid):
        seen, h = _recorder()
        _backend(h).delete_user(uid)
        assert all(not r.url.raw_path.startswith(b"/v1/cache/clear") for r in seen), seen[0].url.raw_path
        assert all(r.url.raw_path.startswith(PREFIX) for r in seen)

    def test_a_query_or_fragment_character_stays_in_the_id(self):
        seen, h = _recorder()
        b = _backend(h)
        b.delete_user("a?b=c"); b.delete_user("a#frag")
        assert [r.url.query for r in seen] == [b"", b""]
        assert [r.url.raw_path for r in seen] == [b"/v1/cache/user/a%3Fb%3Dc", b"/v1/cache/user/a%23frag"]

    def test_a_non_string_id_is_sent_as_its_text(self):
        seen, h = _recorder()
        _backend(h).delete_user(42)
        assert seen[0].url.raw_path == b"/v1/cache/user/42"


class TestStatusIsChecked:
    def test_success_returns_the_gateways_answer(self):
        body = {"status": "ok", "deleted": 3, "tenant_id": 7, "user_id": "alice",
                "sessions_deleted": 1, "entries_status": "ok", "sessions_status": "ok"}
        _, h = _recorder(200, body)
        assert _backend(h).delete_user("alice") == body
        _, h = _recorder(200, {"status": "ok", "deleted": 9, "tenant_id": 7})
        assert _backend(h).clear() == {"status": "ok", "deleted": 9, "tenant_id": 7}

    @pytest.mark.parametrize("status,error", [(422, "validation_error"), (503, "erasure_incomplete"),
                                              (404, "not_found"), (401, "unauthorized"), (500, "internal")])
    @pytest.mark.parametrize("op", ["delete_user", "clear"])
    def test_a_failure_raises_with_the_status_and_code(self, op, status, error):
        body = {"error": error, "message": "m"}
        _, h = _recorder(status, body)
        b = _backend(h)
        with pytest.raises(SulciErasureError) as e:
            b.delete_user("alice") if op == "delete_user" else b.clear()
        assert (e.value.operation, e.value.status, e.value.error, e.value.body) == (op, status, error, body)
        assert str(e.value) == f"sulci {op} failed: HTTP {status}: {error!r}"

    @pytest.mark.parametrize("op", ["delete_user", "clear"])
    def test_a_transport_error_raises_with_no_status(self, op):
        def boom(request):
            raise httpx.ConnectError("connection refused", request=request)
        b = _backend(boom)
        with pytest.raises(SulciErasureError) as e:
            b.delete_user("alice") if op == "delete_user" else b.clear()
        assert e.value.status is None and isinstance(e.value.__cause__, httpx.ConnectError)
        assert str(e.value) == f"sulci {op} failed: no response: 'ConnectError'"
        assert "connection refused" in str(e.value.__cause__)

    def test_a_non_json_failure_still_reports_its_status(self):
        b = _backend(lambda request: httpx.Response(502, text="<html>bad gateway</html>"))
        with pytest.raises(SulciErasureError) as e:
            b.clear()
        assert (e.value.status, e.value.error, e.value.body) == (502, "error", None)

    @pytest.mark.parametrize("op,kind", [(op, k) for op in ("delete_user", "clear") for k in ("503", "transport", "html")]
                             + [("delete_user", "unencodable")])
    def test_a_failure_logs_nothing_the_exception_is_the_report(self, caplog, op, kind):
        def transport(request):
            raise httpx.ConnectError("connection refused", request=request)
        b = _backend({"503": lambda r: httpx.Response(503, json={"error": "erasure_incomplete"}),
                      "transport": transport, "unencodable": _recorder()[1],
                      "html": lambda r: httpx.Response(200, text="<html>sign in</html>")}[kind])
        uid = "\ud800" if kind == "unencodable" else "alice"
        with caplog.at_level(logging.DEBUG, logger="sulci"):
            with pytest.raises(SulciErasureError):
                b.delete_user(uid) if op == "delete_user" else b.clear()
        assert [r for r in caplog.records if r.name.startswith("sulci.backends.cloud")] == []

    def test_the_message_never_carries_the_user_id(self):
        """An exception ends up in tracebacks and error trackers; the user id may be personal data."""
        _, h = _recorder(503, {"error": "erasure_incomplete"})
        with pytest.raises(SulciErasureError) as e:
            _backend(h).delete_user("alice@example.com")
        assert "alice" not in str(e.value) and "alice" not in repr(e.value)

    def test_a_transport_error_naming_the_url_does_not_carry_the_id_into_the_message(self):
        def boom(request):
            raise httpx.ConnectError(f"failed for {request.url}", request=request)
        with pytest.raises(SulciErasureError) as e:
            _backend(boom).delete_user("alice@example.com")
        assert "alice" not in str(e.value) and "alice" not in repr(e.value)
        assert "alice" in str(e.value.__cause__)   # the detail is still there for the handler

    @pytest.mark.parametrize("op", ["delete_user", "clear"])
    def test_there_is_no_flag_to_turn_the_exception_off(self, op):
        import inspect
        method = getattr(SulciCloudBackend, op)
        assert "raise_on_error" not in inspect.signature(method).parameters
        _, h = _recorder()
        with pytest.raises(TypeError):
            getattr(_backend(h), op)(*(["alice"] if op == "delete_user" else []), raise_on_error=False)


class TestCacheClear:
    """Cache.clear() on the managed backend: a failed erasure propagates, and nothing is reset."""

    @staticmethod
    def _cache(status, body, cls=None):
        from sulci import Cache

        class Rec:
            events = []

            def emit(self, event):
                self.events.append(event.event_type)

            def flush(self):
                pass

            def close(self):
                pass

        sink = Rec()
        outer = (cls or Cache)(backend="sulci", api_key="sk-sulci-test0000000000", telemetry=False, event_sink=sink)
        cache = getattr(outer, "_cache", outer)
        _, h = _recorder(status, body)
        cache._backend._client.close()
        cache._backend._client = httpx.Client(base_url="https://api.sulci.io", transport=httpx.MockTransport(h))
        cache._stats["hits"] = 5
        from unittest.mock import MagicMock
        cache._sessions = MagicMock()
        return outer, cache, sink

    def test_a_failed_erasure_raises_and_resets_nothing(self):
        _, cache, sink = self._cache(503, {"error": "erasure_incomplete"})
        with pytest.raises(SulciErasureError) as e:
            cache.clear()
        assert (e.value.operation, e.value.status) == ("clear", 503)
        assert cache._stats["hits"] == 5 and "clear" not in sink.events
        assert not cache._sessions.clear_all.called

    def test_a_confirmed_erasure_resets_the_stats_and_reports_it(self):
        _, cache, sink = self._cache(200, {"status": "ok", "deleted": 4, "tenant_id": 7})
        cache.clear()
        assert cache._stats["hits"] == 0 and "clear" in sink.events
        assert cache._sessions.clear_all.called

    def test_async_aclear_raises_too(self):
        import asyncio
        from sulci import AsyncCache
        ac, cache, sink = self._cache(503, {"error": "erasure_incomplete"}, AsyncCache)
        with pytest.raises(SulciErasureError):
            asyncio.run(ac.aclear())
        assert "clear" not in sink.events
        assert cache._stats["hits"] == 5 and not cache._sessions.clear_all.called


class TestWhatTheReviewFound:
    """Independent review of the first cut (2026-10-09). Each was a real defect of that cut."""

    @pytest.mark.parametrize("bad", ["\ud800", "a\udcffb"])
    def test_an_unencodable_id_raises_and_sends_nothing(self, bad):
        seen, h = _recorder()
        with pytest.raises(SulciErasureError) as e:
            _backend(h).delete_user(bad)
        assert seen == []
        assert e.value.status is None and e.value.error.startswith("unencodable user_id")
        assert isinstance(e.value.__cause__, UnicodeEncodeError)

    def test_an_id_whose_str_raises_raises_and_sends_nothing(self):
        class Bad:
            def __str__(self):
                raise ValueError("no text")
        seen, h = _recorder()
        with pytest.raises(SulciErasureError) as e:
            _backend(h).delete_user(Bad())
        assert seen == [] and isinstance(e.value.__cause__, ValueError)

    @pytest.mark.parametrize("resp", [
        lambda r: httpx.Response(200, text="<html>sign in</html>"),
        lambda r: httpx.Response(204),
        lambda r: httpx.Response(200, json=[1]),
        lambda r: httpx.Response(200, json=None),
        lambda r: httpx.Response(200, json={"status": "error"}),
        lambda r: httpx.Response(200, json={}),
    ], ids=["html", "204", "list", "null", "status-error", "empty-dict"])
    @pytest.mark.parametrize("op", ["delete_user", "clear"])
    def test_a_2xx_that_is_not_an_ok_answer_is_not_a_confirmed_erasure(self, op, resp):
        b = _backend(resp)
        with pytest.raises(SulciErasureError) as e:
            b.delete_user("alice") if op == "delete_user" else b.clear()
        assert e.value.error == "unexpected_response" and 200 <= e.value.status < 300

    def test_an_error_code_with_control_characters_cannot_forge_a_line(self):
        _, h = _recorder(503, {"error": "x\nWARNING forged line"})
        with pytest.raises(SulciErasureError) as e:
            _backend(h).clear()
        assert "\n" not in str(e.value)

    def test_a_long_error_code_is_truncated_in_the_message(self):
        _, h = _recorder(503, {"error": "x" * 100_000})
        with pytest.raises(SulciErasureError) as e:
            _backend(h).clear()
        assert len(str(e.value)) < 1000 and len(e.value.error) == 100_000

    def test_the_error_pickles(self):
        import pickle
        e = SulciErasureError("delete_user", 503, "erasure_incomplete", {"error": "erasure_incomplete"})
        f = pickle.loads(pickle.dumps(e))
        assert (f.operation, f.status, f.error, f.body, str(f)) == (e.operation, e.status, e.error, e.body, str(e))

    @pytest.mark.parametrize("gateway_url", ["https://gw.example/prefix", "https://gw.example/prefix/", "https://gw.example"])
    def test_a_gateway_url_with_a_path_keeps_the_id_one_segment(self, gateway_url):
        b = SulciCloudBackend(api_key="sk-sulci-test0000000000", gateway_url=gateway_url)
        seen, h = _recorder()
        base = b._client.base_url
        b._client.close()
        b._client = httpx.Client(base_url=base, transport=httpx.MockTransport(h))
        b.delete_user("org/../alice")
        raw = seen[0].url.raw_path
        prefix = b"/prefix" if "prefix" in gateway_url else b""
        assert raw == prefix + b"/v1/cache/user/org%2F%2E%2E%2Falice", raw
