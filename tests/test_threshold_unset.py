"""0.10.1 (oss-0032): the managed backend sends a threshold only when the caller set one.

Before 0.10.1 ``Cache(threshold=0.85)`` was the default, and the managed backend sent it on every lookup. The
gateway prefers a per-request threshold over the tenant's setting, so the threshold a Python SDK user saved in the
dashboard never applied (measured on 0.10.0: ``Cache(backend="sulci", api_key=...)`` sent ``"threshold": 0.85``).

Now ``threshold`` defaults to ``None``:
  * a local backend uses ``DEFAULT_THRESHOLD`` (0.85), unchanged;
  * the managed backend sends no ``threshold`` and the gateway applies the tenant's setting;
  * a threshold the caller set — at construction, by assignment, or per call — is sent as before.
"""
import json

import pytest

from sulci import AsyncCache, Cache
from sulci.backends.cloud import SulciCloudBackend
from sulci.core import DEFAULT_THRESHOLD

KEY = "sk-sulci-" + "t" * 32


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": None, "similarity": 0.0, "context_depth": 0}


def _sent_get(cache, **kw):
    """The JSON body of the /v1/cache/get request the managed backend sends for one lookup."""
    bodies = []
    cache._backend._client.post = lambda path, json=None, **k: (bodies.append((path, json)), _Resp())[1]
    cache.get("what is sulci?", **kw)
    gets = [b for p, b in bodies if p == "/v1/cache/get"]
    assert len(gets) == 1, bodies
    return gets[0]


def _managed(**kw):
    return Cache(backend="sulci", api_key=KEY, telemetry=False, **kw)


def test_the_default_is_unset_and_the_local_value_is_unchanged():
    assert DEFAULT_THRESHOLD == 0.85
    import inspect
    assert inspect.signature(Cache.__init__).parameters["threshold"].default is None


def test_a_local_cache_reads_the_default(tmp_path):
    assert Cache(backend="sqlite", db_path=str(tmp_path), telemetry=False).threshold == 0.85


def test_a_managed_cache_with_none_set_sends_no_threshold():
    c = _managed()
    assert c.threshold is None
    body = _sent_get(c)
    assert "threshold" not in body
    assert body["query"] == "what is sulci?"


@pytest.mark.parametrize("value", [0.9, 0.85, 0.0, 1.0])
def test_a_threshold_set_at_construction_is_sent(value):
    """0.85 too: a caller who names the old default chose it, and is not overridden by the dashboard."""
    assert _sent_get(_managed(threshold=value))["threshold"] == value


def test_a_threshold_assigned_after_construction_is_sent():
    c = _managed()
    c.threshold = 0.8
    assert c.threshold == 0.8 and _sent_get(c)["threshold"] == 0.8
    c.threshold = None                          # and unset again
    assert "threshold" not in _sent_get(c)


def test_a_per_call_threshold_is_sent_and_wins():
    assert _sent_get(_managed(), threshold=0.7)["threshold"] == 0.7
    assert _sent_get(_managed(threshold=0.9), threshold=0.7)["threshold"] == 0.7


def test_an_out_of_range_threshold_still_raises():
    with pytest.raises(ValueError):
        _managed().get("q", threshold=1.5)
    with pytest.raises(ValueError):
        _managed(threshold=-0.1).get("q")


def test_async_cache_forwards_the_same_way():
    a = AsyncCache(backend="sulci", api_key=KEY, telemetry=False)
    assert "threshold" not in _sent_get(a._cache)
    b = AsyncCache(backend="sulci", api_key=KEY, telemetry=False, threshold=0.9)
    assert _sent_get(b._cache)["threshold"] == 0.9


def test_remote_get_omits_a_none_threshold_on_the_wire():
    b = SulciCloudBackend(api_key=KEY)
    bodies = []
    b._client.post = lambda path, json=None, **k: (bodies.append(json), _Resp())[1]
    b.remote_get(query="x", threshold=None)
    b.remote_get(query="x", threshold=0.6)
    assert "threshold" not in bodies[0] and bodies[1]["threshold"] == 0.6
    assert json.dumps(bodies[0])                # still a JSON object the gateway parses


def test_the_local_path_is_unchanged(tmp_path):
    """A local lookup compares against 0.85 when none was set: the same entry is a miss at 0.86 and a hit at 0.85."""
    c = Cache(backend="sqlite", db_path=str(tmp_path), telemetry=False)
    calls = []
    real = c._backend.search

    def spy(*a, **k):
        calls.append(k.get("threshold"))
        return real(*a, **k)
    c._backend.search = spy
    if hasattr(c._backend, "search_match"):
        real_m = c._backend.search_match

        def spy_m(*a, **k):
            calls.append(k.get("threshold"))
            return real_m(*a, **k)
        c._backend.search_match = spy_m
    c.get("anything")
    assert calls == [0.85]


def test_a_custom_transport_without_the_flag_still_receives_a_number():
    """0.10.1's review: core treats any backend with remote_get/remote_set as a transport. One that has not declared
    ACCEPTS_UNSET_THRESHOLD is handed 0.85, as before, never None."""
    seen = []

    class Transport:
        def remote_get(self, query, threshold, *, user_id=None, session_id=None):
            seen.append(threshold)
            return None, 0.0, 0

        def remote_set(self, query, response, **kw):
            pass

        def clear(self):
            pass

    c = Cache(backend=Transport(), telemetry=False)
    c.get("q")
    assert c.threshold == 0.85 and seen == [0.85]
    assert SulciCloudBackend.ACCEPTS_UNSET_THRESHOLD is True


def test_a_subclass_with_a_class_level_threshold_still_works(tmp_path):
    """A subclass attribute shadows the descriptor; the instance then stores None, and a local lookup must still
    compare against a number (0.10.1's review)."""
    class Mine(Cache):
        threshold = 0.9
    c = Mine(backend="sqlite", db_path=str(tmp_path), telemetry=False)
    c.set("hello there", "hi")
    assert c.get("hello there")[0] == "hi"
