# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_cloud_erasure.py — SulciCloudBackend.delete_user / clear (0.9.3).

1. The user id is percent-encoded into ONE path segment. Measured on 0.9.2 with an httpx
   MockTransport: ``delete_user("../clear")`` sent ``DELETE /v1/cache/clear`` (the tenant's whole
   cache); ``"a?b=c"`` and ``"a#frag"`` erased the user ``"a"``; ``"org/alice"`` sent a two-segment
   path. ``quote(id, safe="")`` alone is not enough: it leaves ``.``, and httpx collapses the dot
   segments ``/user/.`` and ``/user/..`` — so ``.`` is escaped too.
2. A non-2xx answer is no longer silent. 0.9.2 never looked at the status: a ``422`` or a ``503
   erasure_incomplete`` produced no exception and no log line. Now: a WARNING with the status and
   the gateway's error code, ``None`` by default (the method still never raises), and
   ``SulciErasureError`` with ``raise_on_error=True``. Success returns the gateway's answer.

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
    def test_a_failure_logs_a_warning_and_returns_none_by_default(self, caplog, op, status, error):
        _, h = _recorder(status, {"error": error, "message": "m"})
        b = _backend(h)
        with caplog.at_level(logging.WARNING, logger="sulci.backends.cloud"):
            out = b.delete_user("alice") if op == "delete_user" else b.clear()
        assert out is None
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(msgs) == 1, msgs
        assert op in msgs[0] and f"HTTP {status}" in msgs[0] and error in msgs[0], msgs[0]

    @pytest.mark.parametrize("status,error", [(422, "validation_error"), (503, "erasure_incomplete"), (404, "not_found")])
    @pytest.mark.parametrize("op", ["delete_user", "clear"])
    def test_raise_on_error_raises_with_the_status_and_code(self, op, status, error):
        body = {"error": error, "message": "m"}
        _, h = _recorder(status, body)
        b = _backend(h)
        with pytest.raises(SulciErasureError) as e:
            b.delete_user("alice", raise_on_error=True) if op == "delete_user" else b.clear(raise_on_error=True)
        assert (e.value.operation, e.value.status, e.value.error, e.value.body) == (op, status, error, body)

    @pytest.mark.parametrize("op", ["delete_user", "clear"])
    def test_a_transport_error_logs_and_returns_none_or_raises_on_request(self, caplog, op):
        def boom(request):
            raise httpx.ConnectError("connection refused", request=request)
        b = _backend(boom)
        with caplog.at_level(logging.WARNING, logger="sulci.backends.cloud"):
            assert (b.delete_user("alice") if op == "delete_user" else b.clear()) is None
        assert any("no response" in r.getMessage() for r in caplog.records)
        with pytest.raises(SulciErasureError) as e:
            b.delete_user("alice", raise_on_error=True) if op == "delete_user" else b.clear(raise_on_error=True)
        assert e.value.status is None and isinstance(e.value.__cause__, httpx.ConnectError)

    def test_a_non_json_failure_still_reports_its_status(self, caplog):
        b = _backend(lambda request: httpx.Response(502, text="<html>bad gateway</html>"))
        with caplog.at_level(logging.WARNING, logger="sulci.backends.cloud"):
            assert b.clear() is None
        assert any("HTTP 502" in r.getMessage() for r in caplog.records)

    def test_an_id_with_a_newline_cannot_forge_a_log_line(self, caplog):
        _, h = _recorder(503, {"error": "erasure_incomplete"})
        with caplog.at_level(logging.WARNING, logger="sulci.backends.cloud"):
            _backend(h).delete_user("alice\nWARNING forged line")
        assert all("\n" not in r.getMessage() for r in caplog.records)

    def test_cache_clear_still_never_raises(self):
        """Cache.clear() calls the backend with the default, so a failed erasure cannot crash an app."""
        from sulci import Cache
        cache = Cache(backend="sulci", api_key="sk-sulci-test0000000000")
        _, h = _recorder(503, {"error": "erasure_incomplete"})
        cache._backend._client.close()
        cache._backend._client = httpx.Client(base_url="https://api.sulci.io", transport=httpx.MockTransport(h))
        cache.clear()


class TestWhatTheReviewFound:
    """Independent review of the first cut (2026-10-09). Each was a real defect of that cut."""

    @pytest.mark.parametrize("bad", ["\ud800", "a\udcffb"])
    def test_an_unencodable_id_never_raises_by_default(self, caplog, bad):
        seen, h = _recorder()
        b = _backend(h)
        with caplog.at_level(logging.WARNING, logger="sulci.backends.cloud"):
            assert b.delete_user(bad) is None
        assert seen == [] and any("unencodable user_id" in r.getMessage() for r in caplog.records)
        with pytest.raises(SulciErasureError) as e:
            b.delete_user(bad, raise_on_error=True)
        assert e.value.status is None and e.value.error.startswith("unencodable user_id")
        assert isinstance(e.value.__cause__, UnicodeEncodeError)

    def test_an_id_whose_str_raises_never_raises_by_default(self):
        class Bad:
            def __str__(self):
                raise ValueError("no text")
        seen, h = _recorder()
        assert _backend(h).delete_user(Bad()) is None and seen == []

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
        call = (lambda **kw: b.delete_user("alice", **kw)) if op == "delete_user" else b.clear
        assert call() is None
        with pytest.raises(SulciErasureError) as e:
            call(raise_on_error=True)
        assert e.value.error == "unexpected_response"

    def test_an_error_code_with_control_characters_cannot_forge_a_line(self, caplog):
        _, h = _recorder(503, {"error": "x\nWARNING forged line"})
        with caplog.at_level(logging.WARNING, logger="sulci.backends.cloud"):
            _backend(h).clear()
        assert all("\n" not in r.getMessage() for r in caplog.records)

    def test_a_long_id_is_truncated_in_the_log(self, caplog):
        _, h = _recorder(503, {"error": "erasure_incomplete"})
        with caplog.at_level(logging.WARNING, logger="sulci.backends.cloud"):
            _backend(h).delete_user("x" * 100_000)
        assert all(len(r.getMessage()) < 1000 for r in caplog.records)

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
