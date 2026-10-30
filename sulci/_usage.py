# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/_usage.py
===============
The token counts of the LLM call that produced a cached answer (oss-0044).

A cache hit avoids an LLM call, and the call it avoids is the one that produced
the stored answer. Recording that call's token counts beside the entry lets a
hit be credited with the tokens it saved: the counts the app reports (Sulci
records them as given and does not audit them), unlike the estimated
``cost_per_call``.

Two ways in:

* ``Cache.set(query, response, usage=...)`` — explicit. A value that cannot be
  read raises ``TypeError`` / ``ValueError`` before anything is stored, as
  ``metadata`` does.
* ``cached_call`` — the ``llm_fn`` returns ``sulci.with_usage(text, usage)``,
  a ``str`` that carries the counts. Everything that takes a ``str`` takes it
  unchanged, so an ``llm_fn`` that returns a plain ``str`` keeps working and
  simply records nothing.

What is read (Anthropic, OpenAI and Gemini shapes, as a mapping or as
attributes of the SDK's own usage object):

  input  = ``input_tokens``  (+ ``cache_creation_input_tokens``
           + ``cache_read_input_tokens``, which Anthropic counts apart from
           ``input_tokens``; an avoided call would have processed them too)
           | ``prompt_tokens`` (OpenAI; already includes cached tokens)
           | ``prompt_token_count`` (Gemini; already includes cached tokens)
  output = ``output_tokens`` | ``completion_tokens``
           | ``candidates_token_count`` (+ ``thoughts_token_count``, which Gemini's thinking models report apart
           from it; OpenAI and Anthropic already include reasoning tokens in their output counts)

Each count must be a non-negative ``int`` (not ``bool``) no larger than
``MAX_TOKENS``. Both an input and an output count are required.

Where it goes: in 1.x only the managed backend (``backend="sulci"``) records it
— the gateway keeps the counts with the entry and credits each hit on it to the
tenant's Tokens Saved. Self-hosted backends accept the argument and store
nothing for it. A gateway that predates the field ignores it.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Optional

#: The largest count accepted for either side of one call. Far above any model's context window today; the gateway
#: refuses larger values with 422, so the SDK refuses them first.
MAX_TOKENS = 10_000_000

_INPUT_NAMES = ("input_tokens", "prompt_tokens", "prompt_token_count")
_OUTPUT_NAMES = ("output_tokens", "completion_tokens", "candidates_token_count")
# Anthropic reports prompt-cache tokens beside input_tokens, not inside it.
_INPUT_EXTRA_NAMES = ("cache_creation_input_tokens", "cache_read_input_tokens")


def _read(usage: Any, name: str):
    if isinstance(usage, Mapping):
        return usage.get(name)
    return getattr(usage, name, None)


def _count(value: Any, name: str) -> int:
    # bool is an int subclass; True is not a token count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"usage {name} must be an int, not {type(value).__name__}")
    if value < 0 or value > MAX_TOKENS:
        raise ValueError(f"usage {name} must be between 0 and {MAX_TOKENS}")
    return int(value)


def checked_usage(usage: Any) -> Optional[dict]:
    """Return ``{"input_tokens": int, "output_tokens": int}``, ``None`` for ``None``, or raise.

    Reads a mapping or an object's attributes (see the module docstring for the names). Raises ``TypeError`` when
    a count is not an int (or the value is a ``str``/``bytes``), ``ValueError`` when a count is out of range or
    either side is missing. The message names the field, never the caller's whole object.
    """
    if usage is None:
        return None
    if isinstance(usage, (str, bytes)):
        raise TypeError(f"usage must be a mapping or a usage object, not {type(usage).__name__}")

    def first(names):
        for n in names:
            v = _read(usage, n)
            if v is not None:
                return n, v
        return None, None

    in_name, in_value = first(_INPUT_NAMES)
    out_name, out_value = first(_OUTPUT_NAMES)
    if in_name is None or out_name is None:
        missing = [side for side, n in (("input", in_name), ("output", out_name)) if n is None]
        raise ValueError(
            f"usage has no {' or '.join(missing)} token count "
            f"(input: {', '.join(_INPUT_NAMES)}; output: {', '.join(_OUTPUT_NAMES)})"
        )
    input_tokens = _count(in_value, in_name)
    if in_name == "input_tokens":
        for extra in _INPUT_EXTRA_NAMES:
            v = _read(usage, extra)
            if v is not None:
                input_tokens += _count(v, extra)
        if input_tokens > MAX_TOKENS:
            raise ValueError(f"usage input tokens must be between 0 and {MAX_TOKENS}")
    output_tokens = _count(out_value, out_name)
    if out_name == "candidates_token_count":
        v = _read(usage, "thoughts_token_count")
        if v is not None:
            output_tokens += _count(v, "thoughts_token_count")
        if output_tokens > MAX_TOKENS:
            raise ValueError(f"usage output tokens must be between 0 and {MAX_TOKENS}")
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


class UsageText(str):
    """A ``str`` carrying the token counts of the call that produced it (``.usage``). Made by :func:`with_usage`.

    It is a ``str`` in every way, so code that expects the ``llm_fn``'s answer to be a ``str`` is unaffected.
    Anything derived from it (``+``, ``.strip()``, slicing) is a plain ``str`` without counts.
    """

    usage: Optional[dict]

    def __new__(cls, text: str, usage: Optional[dict]):
        obj = super().__new__(cls, text)
        obj.usage = usage
        return obj

    def __reduce__(self):
        return (UsageText, (str(self), self.usage))


def with_usage(text: str, usage: Any) -> str:
    """Return ``text`` carrying the token counts of the LLM call that produced it.

    Use it as the return value of a ``cached_call`` ``llm_fn``::

        def ask_claude(q):
            msg = client.messages.create(model=..., max_tokens=..., messages=[{"role": "user", "content": q}])
            return sulci.with_usage(msg.content[0].text, msg.usage)

        cache.cached_call(question, ask_claude)

    ``usage`` is the provider's usage object or a mapping (see ``sulci._usage`` for the names read). On the managed
    backend the counts are stored with the entry and every hit on it is credited with them (the dashboard's Tokens
    Saved); self-hosted backends store nothing for them.

    ``text`` must be a ``str`` (``TypeError`` otherwise). A ``usage`` that cannot be read does NOT raise — this runs
    after the LLM call has been paid for, and losing the answer over a statistic would be the wrong trade: it warns
    (``UserWarning``) and the text carries no counts. Pass ``usage=`` to ``Cache.set`` for the strict check.
    """
    if not isinstance(text, str):
        raise TypeError(f"with_usage: text must be a str, not {type(text).__name__}")
    try:
        checked = checked_usage(usage)
    except (TypeError, ValueError) as exc:
        warnings.warn(f"sulci.with_usage: token counts not recorded — {exc}", UserWarning, stacklevel=2)
        checked = None
    return UsageText(str(text), checked)


def usage_of(response: Any) -> Optional[dict]:
    """The counts a :class:`UsageText` carries, checked again (``.usage`` is a plain attribute and can have been
    changed since ``with_usage`` checked it), else ``None``. Never raises — this runs inside ``cached_call`` after the
    LLM has been paid for: counts that no longer read warn and are dropped, and the answer is still cached."""
    if not isinstance(response, UsageText) or response.usage is None:
        return None
    try:
        return checked_usage(response.usage)
    except (TypeError, ValueError) as exc:
        warnings.warn(f"sulci: token counts not recorded — {exc}", UserWarning, stacklevel=3)
        return None


def plain_text(response: Any) -> Any:
    """A :class:`UsageText` as a plain ``str``; anything else unchanged."""
    return str.__str__(response) if isinstance(response, UsageText) else response
