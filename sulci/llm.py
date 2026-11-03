# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/llm.py
============
Ready-made LLM functions that report their own token counts (oss-0053, sulci 1.3.0).

``cached_call`` needs a function that takes the query and returns the answer.
Writing one by hand means remembering ``sulci.with_usage`` for the dashboard's
Tokens Saved; these helpers make the call through your own client and hand the
answer back with the call's token counts, so every entry they produce is
credited on the managed backend::

    import anthropic, sulci
    from sulci import llm

    cache = sulci.Cache(backend="sulci")
    ask = llm.anthropic(anthropic.Anthropic(), model="claude-haiku-4-5", max_tokens=1024)
    result = cache.cached_call("How do I rotate my API key?", ask)

    ask = llm.openai(openai.OpenAI(), model="gpt-4o-mini")
    ask = llm.gemini(genai.Client(), model="gemini-2.5-flash")

Each helper returns a callable ``fn(query, **params)``. ``params`` given to the
helper, and any passed through ``cached_call(..., **llm_kwargs)``, go to the
provider's create call (call-time values win). The answer is the response's
text; the counts are the provider's own usage object, read by the same rules
as ``with_usage`` (Anthropic's prompt-cache input tokens count as input;
reasoning and thinking tokens count as output — every token the avoided call
would have used).

What the helpers refuse, and why (owner decisions, 2026-11-03):

* **Streaming** (``stream=True``): a cache stores a complete answer. Stream
  outside the cache, or call ``Cache.set(..., usage=...)`` yourself once the
  stream ends. A stream's counts are never guessed.
* **Tools** (``tools``, ``tool_choice``, ``functions``, ``function_call``,
  Anthropic ``mcp_servers``, Gemini ``config.tools`` /
  ``config.automatic_function_calling``): a tool call is a control-flow
  instruction, not an answer, and a tool loop's answer comes from several calls
  whose counts one response does not carry — the proxy does not cache tool
  calls either. Write your own function for tool loops. A reply that still
  carries a tool or server-tool block raises :class:`NoTextError`.
* **More than one choice** (OpenAI ``n`` > 1, Gemini ``candidate_count`` > 1):
  only one answer is cached, and its counts would include choices nobody
  receives again.
* **An async client**: ``cached_call`` and ``AsyncCache.acached_call`` call the
  function synchronously (the latter in a worker thread). Pass the sync client.

A response with no text (a refusal with an empty body, a tool-only reply)
raises :class:`NoTextError` — a ``ValueError`` — so nothing is cached. A reply
cut short by ``max_tokens`` or a content filter is text, and is cached as it is
(as ``sulci-proxy`` does): raise ``max_tokens`` if your answers run long.

Counts are optional all the way down: a response without a usage object, or
one that cannot be read, still returns its text (``with_usage`` warns) and is
cached normally, with no Tokens Saved credit. On a self-hosted backend the
counts are accepted and stored nowhere.

No provider SDK is imported here: the helpers call the client you pass.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from sulci._usage import with_usage

__all__ = ["anthropic", "openai", "gemini", "NoTextError"]


class NoTextError(ValueError):
    """The provider's response carried no text to cache (``cached_call`` stores nothing)."""


#: Request parameters a helper never forwards, per the module docstring.
_TOOL_PARAMS = ("tools", "tool_choice", "functions", "function_call", "mcp_servers")
#: Gemini carries tools inside its config.
_GEMINI_TOOL_CONFIG = ("tools", "tool_config", "automatic_function_calling")


def _refuse(provider: str, params: dict) -> None:
    if params.get("stream"):
        raise TypeError(
            f"sulci.llm.{provider}: stream=True is not supported — a cache stores a complete answer. Stream outside "
            "the cache, or call Cache.set(query, text, usage=...) once the stream ends."
        )
    tools = [p for p in _TOOL_PARAMS if params.get(p) is not None]
    if tools:
        raise TypeError(
            f"sulci.llm.{provider}: {', '.join(tools)} not supported — a tool call is not an answer to cache. "
            "Write your own llm_fn for tool loops."
        )


def _not_awaitable(provider: str, resp: Any) -> Any:
    if inspect.isawaitable(resp):
        close = getattr(resp, "close", None)
        if callable(close):   # a coroutine: close it so Python does not warn it was never awaited
            close()
        raise TypeError(
            f"sulci.llm.{provider}: the client is async. cached_call runs the function synchronously "
            "(AsyncCache.acached_call in a worker thread) — pass the sync client."
        )
    return resp


class _Helper:
    """A ``cached_call``-ready function. Built by :func:`anthropic`, :func:`openai`, :func:`gemini`."""

    def __init__(self, provider: str, model: str, params: dict, call: Callable[[str, dict], Any],
                 text_of: Callable[[Any], str], usage_of: Callable[[Any], Any]):
        if not isinstance(model, str) or not model:
            raise TypeError(f"sulci.llm.{provider}: model must be a non-empty str")
        _refuse(provider, params)
        self.provider = provider
        self.model = model
        self._params = dict(params)
        self._call = call
        self._text_of = text_of
        self._usage_of = usage_of

    def __call__(self, query: str, **params: Any) -> str:
        merged = {**self._params, **params}
        _refuse(self.provider, merged)
        resp = _not_awaitable(self.provider, self._call(query, merged))
        text = self._text_of(resp)
        if not isinstance(text, str) or not text.strip():
            raise NoTextError(f"sulci.llm.{self.provider}: the response carried no text to cache")
        return with_usage(text, self._usage_of(resp))

    def __repr__(self) -> str:
        return f"sulci.llm.{self.provider}(model={self.model!r})"


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ── Anthropic ────────────────────────────────────────────────────────────────
def anthropic(client: Any, model: str, *, max_tokens: int = 1024, system: Optional[str] = None,
              **params: Any) -> _Helper:
    """An ``llm_fn`` that calls ``client.messages.create`` (the ``anthropic`` SDK's sync client).

    The query is sent as one user message; ``system`` as the system prompt. The answer is the response's text
    blocks joined (thinking and tool blocks are not answer text). The counts are ``response.usage``:
    ``input_tokens`` plus both prompt-cache counts, and ``output_tokens``, which includes thinking.
    """
    base = {"max_tokens": max_tokens, **params}
    if system is not None:
        base["system"] = system

    def call(query: str, p: dict) -> Any:
        return client.messages.create(model=model, messages=[{"role": "user", "content": query}], **p)

    def text_of(resp: Any) -> str:
        parts = []
        for block in _attr(resp, "content") or []:
            kind = _attr(block, "type")
            if isinstance(kind, str) and "tool" in kind:   # tool_use, server_tool_use, mcp_tool_use, *_tool_result
                return ""
            if kind == "text" and isinstance(_attr(block, "text"), str):
                parts.append(_attr(block, "text"))
        return "".join(parts)

    return _Helper("anthropic", model, base, call, text_of, lambda r: _attr(r, "usage"))


# ── OpenAI ───────────────────────────────────────────────────────────────────
def openai(client: Any, model: str, *, system: Optional[str] = None, **params: Any) -> _Helper:
    """An ``llm_fn`` that calls ``client.chat.completions.create`` (the ``openai`` SDK's sync client, or any
    OpenAI-compatible one — Azure, a local server — with the same shape).

    The query is one user message after an optional system message. The answer is the first choice's message
    content. The counts are ``response.usage``: ``prompt_tokens`` (cached prompt tokens included) and
    ``completion_tokens`` (reasoning tokens included). ``n`` > 1 is refused.
    """
    n = params.get("n")
    if n is not None and n != 1:
        raise TypeError("sulci.llm.openai: n must be 1 — only one answer is cached, and the counts would include "
                        "choices nobody receives again")
    prefix = [{"role": "system", "content": system}] if system is not None else []

    def call(query: str, p: dict) -> Any:
        if p.get("n") not in (None, 1):
            raise TypeError("sulci.llm.openai: n must be 1")
        return client.chat.completions.create(model=model, messages=prefix + [{"role": "user", "content": query}],
                                              **p)

    def text_of(resp: Any) -> str:
        choices = _attr(resp, "choices") or []
        if not choices:
            return ""
        message = _attr(choices[0], "message")
        if _attr(message, "tool_calls"):
            return ""
        content = _attr(message, "content")
        return content if isinstance(content, str) else ""

    return _Helper("openai", model, dict(params), call, text_of, lambda r: _attr(r, "usage"))


# ── Gemini ───────────────────────────────────────────────────────────────────
def _candidate_count(config: Any) -> Any:
    return _attr(config, "candidate_count") if config is not None else None


def _gemini_refuse(config: Any) -> None:
    if config is None:
        return
    if _candidate_count(config) not in (None, 1):
        raise TypeError("sulci.llm.gemini: candidate_count must be 1 — only one answer is cached")
    tools = [k for k in _GEMINI_TOOL_CONFIG if _attr(config, k) is not None]
    if tools:
        raise TypeError(f"sulci.llm.gemini: config.{', config.'.join(tools)} not supported — a tool loop's answer "
                        "comes from several calls. Write your own llm_fn for tool loops.")


def gemini(client: Any, model: str, *, system: Optional[str] = None, config: Any = None,
           **params: Any) -> _Helper:
    """An ``llm_fn`` that calls ``client.models.generate_content`` (the ``google-genai`` SDK's sync client).

    The query is the contents; ``system`` becomes ``config["system_instruction"]`` (``config`` may be a dict or the
    SDK's ``GenerateContentConfig``; with an object config, set ``system_instruction`` on it yourself). The answer is
    ``response.text``. The counts are ``response.usage_metadata``: ``prompt_token_count`` and
    ``candidates_token_count`` plus ``thoughts_token_count`` for thinking models. ``candidate_count`` > 1 is refused.
    """
    if system is not None:
        if config is None:
            config = {"system_instruction": system}
        elif isinstance(config, dict):
            config = {**config, "system_instruction": system}
        else:
            raise TypeError("sulci.llm.gemini: pass system= with a dict config, or set system_instruction on your "
                            "GenerateContentConfig")
    _gemini_refuse(config)
    base = dict(params)
    if config is not None:
        base["config"] = config

    def call(query: str, p: dict) -> Any:
        _gemini_refuse(p.get("config"))
        return client.models.generate_content(model=model, contents=query, **p)

    def text_of(resp: Any) -> str:
        if _attr(resp, "function_calls"):
            return ""
        try:
            text = _attr(resp, "text")
        except Exception:   # the SDK's .text raises on a blocked or empty response
            return ""
        return text if isinstance(text, str) else ""

    return _Helper("gemini", model, base, call, text_of, lambda r: _attr(r, "usage_metadata"))
