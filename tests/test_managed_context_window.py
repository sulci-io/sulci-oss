"""0.10.2 (oss-0037): the managed backend forwards the caller's context_window, and set() can say "standalone".

Measured on production 2026-10-19 (sulci-platform findings F2, F3): ``Cache(backend="sulci", context_window=4)``
blended nothing — every turn came back with ``context_depth`` 0 — because the SDK never sent the window and the
tenant's dashboard setting was 0. The gateway already accepted a per-request ``context_window`` (request > tenant
> 0) and a ``standalone`` flag on a set; only the SDK dropped them. And because a set whose session already holds
a turn stores no entry unless it says ``standalone``, no follow-up asked through the SDK was ever stored: in the
Acme demo's agent loop only step 1 was cached, and later steps could only miss or be served step 1's answer.

Now:
  * a window the caller set (construction or assignment) is sent with every managed lookup; one not set is not
    sent, so the dashboard's setting applies — the same rule as ``threshold`` (0.10.1);
  * a window above the gateway's 20 is sent as 20, and one that is not a whole number from 0 is not sent; each
    warns once, and neither raises (0.10.1 ignored the window entirely, so a Cache that worked keeps working);
  * ``set(..., standalone=True)`` and ``cached_call(..., standalone=True)`` send ``"standalone": true``; False
    sends nothing; AsyncCache mirrors both;
  * a custom transport that does not declare ``ACCEPTS_CONTEXT_WINDOW`` / ``ACCEPTS_STANDALONE`` is handed
    neither keyword;
  * nothing changes on a local backend.
"""
import asyncio
import inspect
import warnings

import pytest

from sulci import AsyncCache, Cache
from sulci.core import MAX_REMOTE_CONTEXT_WINDOW

KEY = "sk-sulci-" + "t" * 32


class _Resp:
    status_code = 200

    def __init__(self, body=None):
        self._body = body or {"response": None, "similarity": 0.0, "context_depth": 0}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _managed(**kw):
    return Cache(backend="sulci", api_key=KEY, telemetry=False, **kw)


def _capture(cache):
    bodies = []
    cache._backend._client.post = lambda path, json=None, **k: (bodies.append((path, json)), _Resp())[1]
    return bodies


def _sent(bodies, path):
    out = [b for p, b in bodies if p == path]
    assert len(out) == 1, bodies
    return out[0]


def _sent_get(cache, **kw):
    bodies = _capture(cache)
    cache.get("how do I fix it?", **kw)
    return _sent(bodies, "/v1/cache/get")


# ── context_window on the lookup ─────────────────────────────────────────────

def test_the_default_is_unset_and_reads_as_zero():
    assert inspect.signature(Cache.__init__).parameters["context_window"].default is None
    c = _managed()
    assert c.context_window == 0
    assert "context_window" not in _sent_get(c, session_id="s1")


@pytest.mark.parametrize("value", [0, 1, 4, 20])
def test_a_window_set_at_construction_is_sent(value):
    """0 too: a caller who asks for a stateless lookup is not overridden by a dashboard that says 4."""
    body = _sent_get(_managed(context_window=value), session_id="s1")
    assert body["context_window"] == value and isinstance(body["context_window"], int)
    assert body["session_id"] == "s1"


def test_a_window_assigned_after_construction_is_sent_and_can_be_unset():
    c = _managed()
    c.context_window = 6
    assert c.context_window == 6 and _sent_get(c)["context_window"] == 6
    c.context_window = None
    assert c.context_window == 0 and "context_window" not in _sent_get(c)


def test_a_window_above_the_gateway_limit_is_sent_as_the_limit_and_warns_once():
    assert MAX_REMOTE_CONTEXT_WINDOW == 20
    c = _managed(context_window=50)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _sent_get(c)["context_window"] == 20
        assert _sent_get(c)["context_window"] == 20
    msgs = [str(x.message) for x in w if "context_window" in str(x.message)]
    assert len(msgs) == 1 and "20 is sent" in msgs[0]


@pytest.mark.parametrize("value", [-1, 2.5, "4", True, float("nan"), float("inf")])
def test_a_window_that_is_not_a_whole_number_from_zero_is_not_sent(value):
    c = _managed()
    c.context_window = value
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert "context_window" not in _sent_get(c)
    assert any("is not sent" in str(x.message) for x in w)


def test_a_whole_float_is_sent_as_an_int():
    c = _managed()
    c.context_window = 4.0
    body = _sent_get(c)
    assert body["context_window"] == 4 and type(body["context_window"]) is int


def test_cached_call_forwards_the_window_too():
    c = _managed(context_window=4)
    bodies = _capture(c)
    c.cached_call("how do I fix it?", lambda q: "answer", session_id="s1")
    assert _sent(bodies, "/v1/cache/get")["context_window"] == 4


def test_a_transport_that_does_not_declare_it_is_not_handed_the_window():
    calls = []

    class Legacy:                      # written against 0.10.1's remote_get / remote_set
        ACCEPTS_UNSET_THRESHOLD = True

        def remote_get(self, query, threshold, *, user_id=None, session_id=None):
            calls.append(("get", query, threshold, user_id, session_id))
            return None, 0.0, 0

        def remote_set(self, query, response, *, user_id=None, session_id=None, ttl_seconds=None):
            calls.append(("set", query, response))

    c = Cache(backend=Legacy(), telemetry=False, context_window=4)
    c.get("q", session_id="s1")
    c.set("q", "a", session_id="s1", standalone=True)
    assert [x[0] for x in calls] == ["get", "set"]


# ── standalone on the write ──────────────────────────────────────────────────

def test_set_sends_standalone_only_when_true():
    c = _managed()
    bodies = _capture(c)
    c.set("How do I fix a Docker container that exits immediately?", "answer", session_id="s1", standalone=True)
    assert _sent(bodies, "/v1/cache/set")["standalone"] is True
    bodies = _capture(c)
    c.set("how do I fix it?", "answer", session_id="s1")
    assert "standalone" not in _sent(bodies, "/v1/cache/set")


def test_cached_call_forwards_standalone_and_not_to_llm_fn():
    c = _managed()
    bodies = _capture(c)
    seen = {}

    def llm(q, **kw):
        seen.update(kw)
        return "answer"

    r = c.cached_call("Task: deploy. Completed: none. Next step?", llm, session_id="s1", standalone=True, temperature=0)
    assert r["source"] == "llm"
    assert _sent(bodies, "/v1/cache/set")["standalone"] is True
    assert seen == {"temperature": 0}


def test_async_cache_mirrors_standalone():
    a = AsyncCache(backend="sulci", api_key=KEY, telemetry=False)
    bodies = _capture(a._cache)
    asyncio.run(a.aset("q1", "a1", session_id="s1", standalone=True))
    asyncio.run(a.acached_call("q2", lambda q: "a2", session_id="s1", standalone=True))
    a.set("q3", "a3", session_id="s1", standalone=True)
    a.cached_call("q4", lambda q: "a4", session_id="s1", standalone=True)
    sets = [b for p, b in bodies if p == "/v1/cache/set"]
    assert [b["query"] for b in sets] == ["q1", "q2", "q3", "q4"]
    assert all(b["standalone"] is True for b in sets)


def test_standalone_is_keyword_only_everywhere():
    for fn in (Cache.set, Cache.cached_call, AsyncCache.aset, AsyncCache.acached_call,
               AsyncCache.set, AsyncCache.cached_call):
        p = inspect.signature(fn).parameters["standalone"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False, fn


# ── nothing changes locally ──────────────────────────────────────────────────

def test_a_local_cache_is_unchanged(tmp_path, fake_embedder):
    c = Cache(backend="sqlite", db_path=str(tmp_path / "db"), telemetry=False, embedding_model=fake_embedder)
    assert c.context_window == 0 and c._sessions is None
    c2 = Cache(backend="sqlite", db_path=str(tmp_path / "db2"), telemetry=False, embedding_model=fake_embedder,
               context_window=4)
    assert c2.context_window == 4 and c2._sessions is not None
    c2.set("What is Python?", "A language.", session_id="s1", standalone=True)
    hit, sim, depth = c2.get("What is Python?", session_id="s1")
    assert hit == "A language." and depth >= 1


# ── 0.10.2's independent review ──────────────────────────────────────────────

def test_a_subclass_with_a_class_level_window_still_works(tmp_path, fake_embedder):
    """A class attribute shadows the descriptor. Before 0.10.2 the instance attribute then held the constructor's
    value (0 by default); it still does — None here would make `context_window > 0` raise TypeError."""
    class Mine(Cache):
        context_window = 6
    c = Mine(backend="sqlite", db_path=str(tmp_path / "db"), telemetry=False, embedding_model=fake_embedder)
    assert c.context_window == 0 and c._sessions is None
    c.set("hello there", "hi")
    assert c.get("hello there")[0] == "hi"
    m = Mine(backend="sulci", api_key=KEY, telemetry=False, context_window=3)
    assert m.context_window == 3 and _sent_get(m)["context_window"] == 3
    assert "context_window" not in _sent_get(Mine(backend="sulci", api_key=KEY, telemetry=False))


def test_a_transport_that_declares_it_receives_both_keywords():
    calls = []

    class Declared:
        ACCEPTS_UNSET_THRESHOLD = True
        ACCEPTS_CONTEXT_WINDOW = True
        ACCEPTS_STANDALONE = True

        def remote_get(self, query, threshold, *, user_id=None, session_id=None, context_window=None):
            calls.append(("get", context_window))
            return None, 0.0, 0

        def remote_set(self, query, response, *, user_id=None, session_id=None, ttl_seconds=None, standalone=False):
            calls.append(("set", standalone))

    c = Cache(backend=Declared(), telemetry=False, context_window=4)
    c.get("q", session_id="s1")
    c.set("q", "a", session_id="s1", standalone=True)
    c.set("q", "a", session_id="s1")
    assert calls == [("get", 4), ("set", True), ("set", False)]


def test_cached_call_without_standalone_sends_none():
    c = _managed()
    bodies = _capture(c)
    c.cached_call("how do I fix it?", lambda q: "answer", session_id="s1")
    assert "standalone" not in _sent(bodies, "/v1/cache/set")


@pytest.mark.parametrize("value,sent", [(10 ** 400, 20), (21, 20)])
def test_any_integer_above_the_limit_is_sent_as_the_limit(value, sent):
    c = _managed()
    c.context_window = value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _sent_get(c)["context_window"] == sent


def test_a_decimal_whole_number_is_sent():
    from decimal import Decimal
    c = _managed()
    c.context_window = Decimal(4)
    assert _sent_get(c)["context_window"] == 4


def test_each_new_value_can_warn_again():
    c = _managed(context_window=50)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _sent_get(c)
        c.context_window = -1
        _sent_get(c)
    msgs = [str(x.message) for x in w if "context_window" in str(x.message)]
    assert len(msgs) == 2 and "20 is sent" in msgs[0] and "is not sent" in msgs[1]


def _cli_kwargs(module_main, argv, monkeypatch, target):
    seen = {}
    monkeypatch.setattr(target[0], target[1], lambda **kw: seen.update(kw) or _Runner())
    module_main(argv)
    return seen


class _Runner:
    def run(self, **kw):
        pass


@pytest.mark.parametrize("argv,want", [
    (["--backend", "sulci"], None),           # the tenant's setting applies
    (["--backend", "sqlite"], 4),             # local: as before
    ([], 4),
    (["--backend", "sulci", "--context-window", "2"], 2),
    (["--backend", "sqlite", "--context-window", "0"], 0),
])
def test_the_cli_entry_points_do_not_override_the_dashboard(argv, want, monkeypatch):
    mcp_server = pytest.importorskip("sulci.integrations.mcp_server", reason="mcp extra not installed")
    monkeypatch.delenv("SULCI_BACKEND", raising=False)
    kw = _cli_kwargs(mcp_server.main, argv + ["--transport", "stdio"], monkeypatch, (mcp_server, "build_server"))
    assert kw["context_window"] == want

    proxy_main = pytest.importorskip("sulci.proxy.__main__", reason="proxy extra not installed")
    import types
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))
    import sulci.proxy.app as proxy_app
    seen = {}
    monkeypatch.setattr(proxy_app, "build_app", lambda **kw: seen.update(kw) or object())
    proxy_main.main(argv)
    assert seen["context_window"] == want


# ── per-call context_threshold (0.10.2): one threshold for no history, a stricter one once a lookup blends ─────────

def test_a_per_call_context_threshold_applies_only_when_the_lookup_blends(tmp_path, fake_embedder):
    c = Cache(backend="sqlite", db_path=str(tmp_path / "db"), telemetry=False, embedding_model=fake_embedder,
              context_window=4)
    c.set("What is Python?", "A language.")
    # no history: the per-call context threshold is ignored, `threshold` decides
    hit, sim, depth = c.get("Tell me what Python is", threshold=0.0, context_threshold=1.0, session_id="s1")
    assert depth == 0 and hit == "A language." and sim < 1.0
    c.set("Tell me about Rust", "Another language.", session_id="s1")
    # history: the context threshold wins over an explicit threshold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hit2, sim2, depth2 = c.get("What is Python?", threshold=0.0, context_threshold=1.0, session_id="s1")
        assert depth2 > 0 and hit2 is None and sim2 < 1.0
        hit3, _, depth3 = c.get("What is Python?", threshold=1.0, context_threshold=0.0, session_id="s1")
    assert depth3 > 0 and hit3 is not None


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_an_out_of_range_context_threshold_raises(bad):
    with pytest.raises(ValueError):
        _managed().get("q", context_threshold=bad)


def test_the_managed_backend_sends_the_context_threshold_only_when_set():
    assert "context_threshold" not in _sent_get(_managed(), session_id="s1")
    assert _sent_get(_managed(), session_id="s1", context_threshold=0.8)["context_threshold"] == 0.8
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _sent_get(_managed(context_threshold=0.82), session_id="s1")["context_threshold"] == 0.82
        assert _sent_get(_managed(context_threshold=0.82), context_threshold=0.9)["context_threshold"] == 0.9
    assert not any("context_threshold is not applied" in str(x.message) for x in w)


def test_cached_call_and_async_forward_the_context_threshold():
    c = _managed()
    bodies = _capture(c)
    c.cached_call("q", lambda q, **kw: "a", session_id="s1", context_threshold=0.8)
    assert _sent(bodies, "/v1/cache/get")["context_threshold"] == 0.8
    a = AsyncCache(backend="sulci", api_key=KEY, telemetry=False)
    bodies = _capture(a._cache)
    asyncio.run(a.aget("q1", session_id="s1", context_threshold=0.81))
    asyncio.run(a.acached_call("q2", lambda q: "a", session_id="s1", context_threshold=0.82))
    a.get("q3", session_id="s1", context_threshold=0.83)
    a.cached_call("q4", lambda q: "a", session_id="s1", context_threshold=0.84)
    got = [b["context_threshold"] for p, b in bodies if p == "/v1/cache/get"]
    assert got == [0.81, 0.82, 0.83, 0.84]


def test_a_transport_without_the_flag_is_not_handed_it_and_still_warns():
    calls = []

    class Legacy:
        ACCEPTS_UNSET_THRESHOLD = True

        def remote_get(self, query, threshold, *, user_id=None, session_id=None):
            calls.append(threshold)
            return None, 0.0, 0

        def remote_set(self, query, response, *, user_id=None, session_id=None, ttl_seconds=None):
            pass

    c = Cache(backend=Legacy(), telemetry=False, context_threshold=0.8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c.get("q", context_threshold=0.9)
    assert calls == [None]
    assert any("context_threshold is not applied" in str(x.message) for x in w)


# ── 0.10.2's second review ───────────────────────────────────────────────────

def test_an_explicit_threshold_beats_the_instance_context_threshold_on_the_wire_as_locally():
    body = _sent_get(_managed(context_threshold=0.7), session_id="s", threshold=0.9)
    assert body["threshold"] == 0.9 and "context_threshold" not in body
    body = _sent_get(_managed(context_threshold=0.7), session_id="s", threshold=0.9, context_threshold=0.8)
    assert body["context_threshold"] == 0.8


def test_a_per_call_context_threshold_alone_warns_on_a_transport_without_the_flag():
    class Legacy:
        ACCEPTS_UNSET_THRESHOLD = True

        def remote_get(self, query, threshold, *, user_id=None, session_id=None):
            return None, 0.0, 0

        def remote_set(self, query, response, *, user_id=None, session_id=None, ttl_seconds=None):
            pass

    c = Cache(backend=Legacy(), telemetry=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c.get("q", context_threshold=0.9)
    assert any("context_threshold is not applied" in str(x.message) for x in w)


def _telemetry_thresholds(monkeypatch):
    import sulci
    seen = []
    monkeypatch.setattr(sulci, "_emit", lambda name, payload: seen.append(payload.get("threshold")), raising=False)
    return seen


def test_telemetry_reports_the_context_threshold_that_decided_a_blended_lookup(tmp_path, fake_embedder, monkeypatch):
    seen = _telemetry_thresholds(monkeypatch)
    c = Cache(backend="sqlite", db_path=str(tmp_path / "db"), telemetry=True, embedding_model=fake_embedder,
              context_window=4)
    c.set("Tell me about Rust", "Another language.", session_id="s1")
    seen.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.get("What is Python?", threshold=0.5, context_threshold=0.95, session_id="s1")
    assert seen and seen[-1] == 0.95

    seen.clear()
    m = Cache(backend="sulci", api_key=KEY, telemetry=False)
    m._telemetry = True                 # no auto-connect: only the get-time emit is under test
    m._backend._client.post = lambda path, json=None, **k: _Resp({"response": None, "similarity": 0.1,
                                                                  "context_depth": 2})
    m.get("q", session_id="s1", context_threshold=0.8)
    assert seen and seen[-1] == 0.8


def test_a_subclass_window_assigned_after_construction_is_sent():
    class Mine(Cache):
        context_window = 6
    m = Mine(backend="sulci", api_key=KEY, telemetry=False)
    assert "context_window" not in _sent_get(m)
    m.context_window = 5
    assert _sent_get(m)["context_window"] == 5
