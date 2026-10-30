"""oss-0044: the token counts of the LLM call that produced a cached answer.

The dashboard's Cost Saved is an estimate (hits × a configured cost per call). A hit avoids exactly the call that
produced the stored answer, so recording that call's token counts with the entry lets each hit be credited with the
tokens it actually saved. sulci-platform 0325 records them on the managed backend.

Pinned here:
  * ``checked_usage`` reads the Anthropic, OpenAI and Gemini shapes, as a mapping or as attributes, adds Anthropic's
    prompt-cache input tokens, and refuses what is not a count — naming the field, never the value;
  * ``with_usage`` returns a ``str`` that carries the counts and never raises for unreadable counts (it runs after
    the LLM has been paid for): it warns and carries none;
  * ``Cache.set(usage=)`` checks before anything is sent; the managed backend sends ``"usage"`` only when there are
    counts, and only to a transport that declares ``ACCEPTS_USAGE``;
  * ``cached_call`` picks the counts up from an ``llm_fn`` that returns ``with_usage(...)`` and hands the caller
    plain text; AsyncCache mirrors it;
  * the body still validates against the gateway's request model (vendored below), and the response is stored as
    plain text on a local backend, where the counts are accepted and change nothing.
"""
import asyncio
import inspect
import pickle
import types
import warnings
from typing import Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

import sulci
from sulci import AsyncCache, Cache
from sulci._usage import MAX_TOKENS, UsageText, checked_usage, plain_text, usage_of, with_usage

KEY = "sk-sulci-" + "u" * 32


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": None, "similarity": 0.0, "context_depth": 0}


def _managed(**kw):
    return Cache(backend="sulci", api_key=KEY, telemetry=False, **kw)


def _capture(cache):
    bodies = []
    cache._backend._client.post = lambda path, json=None, **k: (bodies.append((path, json)), _Resp())[1]
    return bodies


def _sets(bodies):
    return [b for p, b in bodies if p == "/v1/cache/set"]


# ── checked_usage ────────────────────────────────────────────────────────────

def test_anthropic_mapping():
    assert checked_usage({"input_tokens": 12, "output_tokens": 340}) == {"input_tokens": 12, "output_tokens": 340}


def test_anthropic_prompt_cache_tokens_count_as_input():
    u = {"input_tokens": 5, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 2000, "output_tokens": 7}
    assert checked_usage(u) == {"input_tokens": 2105, "output_tokens": 7}


def test_anthropic_object_with_none_cache_fields():
    # The Anthropic SDK's Usage object carries the cache fields as None when prompt caching is off.
    u = types.SimpleNamespace(input_tokens=9, output_tokens=3, cache_creation_input_tokens=None,
                              cache_read_input_tokens=None)
    assert checked_usage(u) == {"input_tokens": 9, "output_tokens": 3}


def test_openai_object():
    u = types.SimpleNamespace(prompt_tokens=20, completion_tokens=30, total_tokens=50)
    assert checked_usage(u) == {"input_tokens": 20, "output_tokens": 30}


def test_openai_cache_fields_are_not_added():
    # OpenAI's prompt_tokens already includes cached tokens; a stray Anthropic-named field must not be added to it.
    u = {"prompt_tokens": 20, "completion_tokens": 30, "cache_read_input_tokens": 999}
    assert checked_usage(u) == {"input_tokens": 20, "output_tokens": 30}


def test_gemini_mapping():
    u = {"prompt_token_count": 11, "candidates_token_count": 22, "total_token_count": 33}
    assert checked_usage(u) == {"input_tokens": 11, "output_tokens": 22}


def test_gemini_thinking_tokens_count_as_output():
    u = types.SimpleNamespace(prompt_token_count=11, candidates_token_count=22, thoughts_token_count=300)
    assert checked_usage(u) == {"input_tokens": 11, "output_tokens": 322}


def test_thinking_tokens_are_not_added_to_other_shapes():
    assert checked_usage({"output_tokens": 5, "input_tokens": 1, "thoughts_token_count": 9})["output_tokens"] == 5


def test_zero_is_a_count():
    assert checked_usage({"input_tokens": 0, "output_tokens": 0}) == {"input_tokens": 0, "output_tokens": 0}


def test_none_is_none():
    assert checked_usage(None) is None


@pytest.mark.parametrize("bad, exc", [
    ({"input_tokens": 1}, ValueError),                                   # no output
    ({"output_tokens": 1}, ValueError),                                  # no input
    ({}, ValueError),
    (object(), ValueError),
    ({"input_tokens": -1, "output_tokens": 1}, ValueError),
    ({"input_tokens": 1, "output_tokens": MAX_TOKENS + 1}, ValueError),
    ({"input_tokens": MAX_TOKENS, "cache_read_input_tokens": 1, "output_tokens": 1}, ValueError),
    ({"input_tokens": True, "output_tokens": 1}, TypeError),
    ({"input_tokens": 1.5, "output_tokens": 1}, TypeError),
    ({"input_tokens": "12", "output_tokens": 1}, TypeError),
    ({"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": "x"}, TypeError),
    ("12", TypeError),
    (b"12", TypeError),
])
def test_refused(bad, exc):
    with pytest.raises(exc):
        checked_usage(bad)


def test_message_names_the_field_not_the_value():
    with pytest.raises(TypeError) as e:
        checked_usage({"input_tokens": "secret-value-7f3", "output_tokens": 1})
    assert "input_tokens" in str(e.value) and "secret-value-7f3" not in str(e.value)


def test_max_is_accepted():
    assert checked_usage({"input_tokens": MAX_TOKENS, "output_tokens": MAX_TOKENS})["output_tokens"] == MAX_TOKENS


# ── with_usage ───────────────────────────────────────────────────────────────

def test_with_usage_is_a_str_carrying_counts():
    t = sulci.with_usage("Paris.", {"input_tokens": 3, "output_tokens": 2})
    assert isinstance(t, str) and t == "Paris." and t.usage == {"input_tokens": 3, "output_tokens": 2}
    assert usage_of(t) == {"input_tokens": 3, "output_tokens": 2}


def test_derived_strings_are_plain():
    t = with_usage(" Paris. ", {"input_tokens": 3, "output_tokens": 2})
    for d in (t.strip(), t + "!", t[1:], plain_text(t)):
        assert type(d) is str and usage_of(d) is None
    assert plain_text(t) == " Paris. "


def test_plain_text_leaves_others_alone():
    assert plain_text("x") == "x"
    o = object()
    assert plain_text(o) is o


def test_with_usage_survives_pickle():
    # Imported here, not at the top: other suites purge and re-import `sulci`, and pickle finds the class by name in
    # sys.modules — a module-level import would hold the class of a module that is no longer there.
    from sulci._usage import UsageText, with_usage
    t = with_usage("Paris.", {"input_tokens": 3, "output_tokens": 2})
    t2 = pickle.loads(pickle.dumps(t))
    assert isinstance(t2, UsageText) and t2 == "Paris." and t2.usage == t.usage


def test_with_usage_warns_and_carries_nothing_on_unreadable_counts():
    with pytest.warns(UserWarning, match="not recorded"):
        t = with_usage("Paris.", {"input_tokens": "lots"})
    assert t == "Paris." and t.usage is None


def test_with_usage_none_is_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = with_usage("Paris.", None)
    assert t.usage is None


def test_with_usage_refuses_non_text():
    with pytest.raises(TypeError):
        with_usage(None, {"input_tokens": 1, "output_tokens": 1})


def test_exported():
    assert "with_usage" in sulci.__all__ and sulci.with_usage is with_usage


# ── Cache.set on the managed backend ─────────────────────────────────────────

def test_set_sends_usage():
    c = _managed()
    bodies = _capture(c)
    c.set("What is the capital of France?", "Paris.", usage={"prompt_tokens": 14, "completion_tokens": 3})
    (b,) = _sets(bodies)
    assert b["usage"] == {"input_tokens": 14, "output_tokens": 3}
    assert type(b["response"]) is str


def test_set_without_usage_sends_none():
    c = _managed()
    bodies = _capture(c)
    c.set("q", "a")
    (b,) = _sets(bodies)
    assert "usage" not in b


def test_set_reads_a_with_usage_response():
    c = _managed()
    bodies = _capture(c)
    c.set("q", with_usage("a", {"input_tokens": 4, "output_tokens": 5}))
    (b,) = _sets(bodies)
    assert b["usage"] == {"input_tokens": 4, "output_tokens": 5}
    assert type(b["response"]) is str and b["response"] == "a"


def test_counts_changed_after_with_usage_are_dropped_not_sent():
    # .usage is a plain attribute; a value broken after with_usage checked it must not reach the gateway (a 422
    # there would leave the answer uncached) and must not raise inside cached_call.
    c = _managed()
    bodies = _capture(c)
    t = with_usage("a", {"input_tokens": 4, "output_tokens": 5})
    t.usage = {"input_tokens": -4, "output_tokens": 5}
    with pytest.warns(UserWarning, match="not recorded"):
        r = c.cached_call("q", lambda q: t)
    (b,) = _sets(bodies)
    assert r["response"] == "a" and "usage" not in b and b["response"] == "a"


def test_explicit_usage_wins_over_the_response():
    c = _managed()
    bodies = _capture(c)
    c.set("q", with_usage("a", {"input_tokens": 4, "output_tokens": 5}), usage={"input_tokens": 1, "output_tokens": 1})
    (b,) = _sets(bodies)
    assert b["usage"] == {"input_tokens": 1, "output_tokens": 1}


@pytest.mark.parametrize("bad", [{"input_tokens": 1}, {"input_tokens": -1, "output_tokens": 1}, "12"])
def test_set_refuses_bad_usage_before_sending(bad):
    c = _managed()
    bodies = _capture(c)
    with pytest.raises((TypeError, ValueError)):
        c.set("q", "a", usage=bad)
    assert bodies == []


def test_a_transport_that_does_not_declare_it_is_not_handed_usage():
    c = _managed()
    calls = []

    class OldTransport:
        ACCEPTS_STANDALONE = True

        def remote_set(self, **kw):
            calls.append(kw)

        def remote_get(self, **kw):
            return None, 0.0, 0

    c._backend = OldTransport()
    c.set("q", "a", usage={"input_tokens": 1, "output_tokens": 1})
    assert calls and "usage" not in calls[0]


# ── cached_call ──────────────────────────────────────────────────────────────

def test_cached_call_records_usage_and_returns_plain_text():
    c = _managed()
    bodies = _capture(c)
    r = c.cached_call("q", lambda q: with_usage("a", {"input_tokens": 7, "output_tokens": 8}))
    assert r["source"] == "llm" and r["response"] == "a" and type(r["response"]) is str
    (b,) = _sets(bodies)
    assert b["usage"] == {"input_tokens": 7, "output_tokens": 8}


def test_cached_call_with_a_plain_str_sends_no_usage():
    c = _managed()
    bodies = _capture(c)
    r = c.cached_call("q", lambda q: "a")
    assert r["response"] == "a"
    (b,) = _sets(bodies)
    assert "usage" not in b


def test_cached_call_keeps_the_answer_when_counts_are_unreadable():
    c = _managed()
    bodies = _capture(c)
    with pytest.warns(UserWarning):
        r = c.cached_call("q", lambda q: with_usage("a", {"tokens": 3}))
    assert r["response"] == "a"
    (b,) = _sets(bodies)
    assert "usage" not in b and b["response"] == "a"


def test_async_cache_mirrors_usage():
    a = AsyncCache(backend="sulci", api_key=KEY, telemetry=False)
    bodies = _capture(a._cache)
    u = {"input_tokens": 1, "output_tokens": 2}
    asyncio.run(a.aset("q1", "a1", usage=u))
    asyncio.run(a.acached_call("q2", lambda q: with_usage("a2", u)))
    a.set("q3", "a3", usage=u)
    a.cached_call("q4", lambda q: with_usage("a4", u))
    sets = _sets(bodies)
    assert [b["query"] for b in sets] == ["q1", "q2", "q3", "q4"]
    assert all(b["usage"] == u for b in sets)


def test_usage_is_keyword_only_everywhere():
    for fn in (Cache.set, AsyncCache.aset, AsyncCache.set):
        p = inspect.signature(fn).parameters["usage"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None, fn


# ── the wire contract (sulci-platform 0325's models, vendored) ───────────────

class _GatewayCacheUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_tokens: int = Field(ge=0, le=MAX_TOKENS, strict=True)
    output_tokens: int = Field(ge=0, le=MAX_TOKENS, strict=True)


class _GatewayCacheSetRequest(BaseModel):
    query: str
    response: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ttl_seconds: Optional[int] = None
    metadata: Optional[dict] = None
    standalone: bool = False
    usage: Optional[_GatewayCacheUsage] = None


def test_body_validates_against_the_gateway_model():
    c = _managed()
    bodies = _capture(c)
    c.set("q", "a", session_id="s", standalone=True,
          usage={"input_tokens": 5, "cache_read_input_tokens": 10, "output_tokens": 6})
    (b,) = _sets(bodies)
    parsed = _GatewayCacheSetRequest.model_validate(b)
    assert parsed.usage.input_tokens == 15 and parsed.usage.output_tokens == 6


# ── local backends: accepted, stored as plain text, nothing else changes ─────

def test_local_backend_accepts_usage_and_stores_plain_text(tmp_path, fake_embedder):
    c = Cache(backend="sqlite", db_path=str(tmp_path / "db"), telemetry=False, embedding_model=fake_embedder)
    c.set("What is Python?", with_usage("A language.", {"input_tokens": 1, "output_tokens": 2}))
    c.set("What is Rust?", "Another language.", usage={"input_tokens": 1, "output_tokens": 2})
    hit, sim, _ = c.get("What is Python?")
    assert hit == "A language." and type(hit) is str
    r = c.cached_call("What is Go?", lambda q: with_usage("Also a language.", {"input_tokens": 1, "output_tokens": 1}))
    assert r["source"] == "llm" and type(r["response"]) is str
    with pytest.raises(ValueError):
        c.set("q", "a", usage={"input_tokens": 1})
