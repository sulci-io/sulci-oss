# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""oss-0053 (sulci 1.3.0): sulci.llm provider helpers report their own token counts.

Fake clients stand in for the Anthropic, OpenAI and Gemini SDKs: each records the call it received and answers in the
SDK's shape (attribute objects, as the SDKs return), so the tests see exactly what reaches the provider and what the
helper hands back.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

import sulci
from sulci import Cache, llm
from sulci._usage import UsageText


# ── fakes ────────────────────────────────────────────────────────────────────
class FakeAnthropic:
    def __init__(self, content=None, usage=NS(input_tokens=100, output_tokens=40,
                                              cache_creation_input_tokens=5, cache_read_input_tokens=20)):
        self.calls = []
        self._content = content if content is not None else [NS(type="text", text="Rotate it in Settings.")]
        self._usage = usage
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return NS(content=self._content, usage=self._usage)


class FakeOpenAI:
    def __init__(self, message=None, usage=NS(prompt_tokens=80, completion_tokens=25)):
        self.calls = []
        self._message = message if message is not None else NS(content="Rotate it in Settings.", tool_calls=None)
        self._usage = usage
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return NS(choices=[NS(message=self._message)], usage=self._usage)


class FakeGemini:
    def __init__(self, text="Rotate it in Settings.",
                 usage=NS(prompt_token_count=60, candidates_token_count=20, thoughts_token_count=15)):
        self.calls = []
        self._text = text
        self._usage = usage
        self.models = NS(generate_content=self._gen)

    def _gen(self, **kw):
        self.calls.append(kw)
        return NS(text=self._text, usage_metadata=self._usage, function_calls=None)


class AsyncAnthropic:
    def __init__(self):
        self.coros = []

        async def create(**kw):
            return NS(content=[NS(type="text", text="x")], usage=None)

        def start(**kw):
            coro = create(**kw)
            self.coros.append(coro)
            return coro
        self.messages = NS(create=start)


# ── each provider: the call it makes, the text and the counts it returns ─────
def test_anthropic_sends_one_user_message_and_returns_text_with_all_input_tokens():
    client = FakeAnthropic()
    ask = llm.anthropic(client, model="claude-haiku-4-5", system="Be brief.", temperature=0)
    out = ask("How do I rotate my key?")
    assert client.calls == [{"model": "claude-haiku-4-5", "max_tokens": 1024, "system": "Be brief.",
                             "temperature": 0, "messages": [{"role": "user", "content": "How do I rotate my key?"}]}]
    assert isinstance(out, UsageText) and out == "Rotate it in Settings."
    # Anthropic counts prompt-cache tokens beside input_tokens: an avoided call would have processed them too.
    assert out.usage == {"input_tokens": 125, "output_tokens": 40}


def test_anthropic_joins_text_blocks_and_skips_thinking():
    client = FakeAnthropic(content=[NS(type="thinking", thinking="hmm"), NS(type="text", text="Part one. "),
                                    NS(type="text", text="Part two.")])
    assert llm.anthropic(client, model="m")("q") == "Part one. Part two."


def test_openai_sends_system_then_user_and_returns_counts():
    client = FakeOpenAI()
    ask = llm.openai(client, model="gpt-4o-mini", system="Be brief.")
    out = ask("How do I rotate my key?", temperature=0.2)
    assert client.calls[0]["messages"] == [{"role": "system", "content": "Be brief."},
                                           {"role": "user", "content": "How do I rotate my key?"}]
    assert client.calls[0]["temperature"] == 0.2 and client.calls[0]["model"] == "gpt-4o-mini"
    assert out == "Rotate it in Settings." and out.usage == {"input_tokens": 80, "output_tokens": 25}


def test_gemini_counts_thinking_tokens_as_output_and_puts_system_in_config():
    client = FakeGemini()
    out = llm.gemini(client, model="gemini-2.5-flash", system="Be brief.")("q")
    assert client.calls == [{"model": "gemini-2.5-flash", "contents": "q",
                             "config": {"system_instruction": "Be brief."}}]
    assert out.usage == {"input_tokens": 60, "output_tokens": 35}


def test_call_time_params_override_the_helper_s():
    client = FakeOpenAI()
    llm.openai(client, model="m", temperature=0)("q", temperature=0.9)
    assert client.calls[0]["temperature"] == 0.9


# ── decision 1: streaming is refused, never guessed ─────────────────────────
@pytest.mark.parametrize("make", [
    lambda: llm.anthropic(FakeAnthropic(), model="m", stream=True),
    lambda: llm.openai(FakeOpenAI(), model="m", stream=True),
    lambda: llm.gemini(FakeGemini(), model="m", stream=True),
])
def test_stream_is_refused_at_build(make):
    with pytest.raises(TypeError, match="stream=True"):
        make()


def test_stream_is_refused_at_call_and_nothing_is_sent():
    client = FakeOpenAI()
    with pytest.raises(TypeError, match="stream=True"):
        llm.openai(client, model="m")("q", stream=True)
    assert client.calls == []


# ── decision 2: every saved token counted — so tools and extra choices are refused ──
@pytest.mark.parametrize("param", ["tools", "tool_choice", "functions", "function_call"])
def test_tool_params_are_refused(param):
    with pytest.raises(TypeError, match=param):
        llm.openai(FakeOpenAI(), model="m", **{param: [{"type": "function"}]})


def test_openai_n_above_one_is_refused_at_build_and_call():
    with pytest.raises(TypeError, match="n must be 1"):
        llm.openai(FakeOpenAI(), model="m", n=2)
    client = FakeOpenAI()
    with pytest.raises(TypeError, match="n must be 1"):
        llm.openai(client, model="m")("q", n=3)
    assert client.calls == []
    llm.openai(FakeOpenAI(), model="m", n=1)("q")   # n=1 is the default made explicit


@pytest.mark.parametrize("config", [{"tools": [{"google_search": {}}]}, {"automatic_function_calling": {}},
                                    NS(tools=[object()], candidate_count=None)])
def test_gemini_tools_in_config_are_refused_at_build_and_call(config):
    with pytest.raises(TypeError, match="config.tools|config.automatic_function_calling"):
        llm.gemini(FakeGemini(), model="m", config=config)
    client = FakeGemini()
    with pytest.raises(TypeError, match="not supported"):
        llm.gemini(client, model="m")("q", config=config)
    assert client.calls == []


def test_anthropic_mcp_servers_are_refused():
    with pytest.raises(TypeError, match="mcp_servers"):
        llm.anthropic(FakeAnthropic(), model="m", mcp_servers=[{"type": "url", "url": "https://x"}])


@pytest.mark.parametrize("kind", ["tool_use", "server_tool_use", "mcp_tool_use", "web_search_tool_result"])
def test_anthropic_reply_with_a_tool_block_is_not_cached(kind):
    client = FakeAnthropic(content=[NS(type="text", text="Let me search."), NS(type=kind)])
    with pytest.raises(llm.NoTextError):
        llm.anthropic(client, model="m")("q")


def test_gemini_candidate_count_above_one_is_refused():
    with pytest.raises(TypeError, match="candidate_count"):
        llm.gemini(FakeGemini(), model="m", config={"candidate_count": 2})


def test_a_tool_only_reply_raises_no_text_and_caches_nothing(tmp_path, fake_embedder):
    client = FakeOpenAI(message=NS(content=None, tool_calls=[NS(id="1")]))
    cache = Cache(backend="sqlite", db_path=str(tmp_path / "c"), embedding_model=fake_embedder, telemetry=False)
    with pytest.raises(llm.NoTextError):
        cache.cached_call("q", llm.openai(client, model="m"))
    assert cache.get("q")[0] is None


def test_an_empty_anthropic_reply_raises_no_text():
    with pytest.raises(llm.NoTextError):
        llm.anthropic(FakeAnthropic(content=[NS(type="text", text="  ")]), model="m")("q")


def test_a_blocked_gemini_reply_whose_text_raises_is_no_text():
    class Blocked:
        usage_metadata = None
        function_calls = None

        @property
        def text(self):
            raise ValueError("blocked")
    client = NS(models=NS(generate_content=lambda **kw: Blocked()))
    with pytest.raises(llm.NoTextError):
        llm.gemini(client, model="m")("q")


# ── async clients and odd input ──────────────────────────────────────────────
def test_an_async_client_is_refused_and_its_coroutine_closed():
    import inspect
    client = AsyncAnthropic()
    with pytest.raises(TypeError, match="async"):
        llm.anthropic(client, model="m")("q")
    # closed, so Python never warns "coroutine ... was never awaited"
    assert [inspect.getcoroutinestate(c) for c in client.coros] == [inspect.CORO_CLOSED]


def test_model_must_be_a_non_empty_str():
    with pytest.raises(TypeError, match="model"):
        llm.openai(FakeOpenAI(), model="")


def test_gemini_system_with_an_object_config_is_refused():
    with pytest.raises(TypeError, match="system_instruction"):
        llm.gemini(FakeGemini(), model="m", system="s", config=NS(candidate_count=None))


# ── counts stay optional ─────────────────────────────────────────────────────
def test_no_usage_object_still_returns_the_text_without_counts():
    out = llm.anthropic(FakeAnthropic(usage=None), model="m")("q")
    assert out == "Rotate it in Settings." and out.usage is None


def test_unreadable_usage_warns_and_keeps_the_text():
    with pytest.warns(UserWarning, match="token counts not recorded"):
        out = llm.openai(FakeOpenAI(usage=NS(prompt_tokens="lots", completion_tokens=3)), model="m")("q")
    assert out == "Rotate it in Settings." and out.usage is None


# ── through cached_call: the counts reach the managed backend, plain text comes back ──
KEY = "sk-sulci-" + "t" * 32


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_cached_call_on_the_managed_backend_sends_the_helper_s_counts():
    cache = Cache(backend="sulci", api_key=KEY, telemetry=False)
    sent = []

    def post(path, json=None, **k):
        sent.append((path, json))
        return _Resp({"response": None, "similarity": 0.0, "context_depth": 0} if path.endswith("/get")
                     else {"status": "ok", "stored": True})
    cache._backend._client.post = post
    result = cache.cached_call("How do I rotate my key?", llm.anthropic(FakeAnthropic(), model="m"))
    assert type(result["response"]) is str and result["response"] == "Rotate it in Settings."
    sets = [body for path, body in sent if path == "/v1/cache/set"]
    assert sets and sets[0]["usage"] == {"input_tokens": 125, "output_tokens": 40}


def test_async_cache_runs_a_helper_in_its_worker_thread(tmp_path, fake_embedder):
    acache = sulci.AsyncCache(backend="sqlite", db_path=str(tmp_path / "a"), embedding_model=fake_embedder,
                              telemetry=False)
    client = FakeOpenAI()
    result = asyncio.run(acache.acached_call("q", llm.openai(client, model="m")))
    assert result["source"] == "llm" and result["response"] == "Rotate it in Settings." and len(client.calls) == 1


def test_repr_names_the_provider_and_model():
    assert repr(llm.gemini(FakeGemini(), model="gemini-2.5-flash")) == "sulci.llm.gemini(model='gemini-2.5-flash')"
