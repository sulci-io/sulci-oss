# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci.integrations.litellm
───────────────────────────
Sulci as the cache layer inside LiteLLM.

LiteLLM is a routing/gateway layer: it normalises 100+ providers behind one
call and offers exact-match and stateless-semantic caches. It has no
context-aware cache. This adapter supplies one, so a LiteLLM deployment gets
multi-turn hit rates without changing any call site.

Usage:

    from sulci.integrations.litellm import install
    install(backend="sqlite", context_window=4)

    import litellm
    litellm.completion(model="gpt-4o", messages=[...])   # now cached

Or explicitly, if you want to own the wiring:

    import litellm
    from litellm.caching import Cache as LiteLLMCache
    from sulci.integrations.litellm import SulciLiteLLMCache

    litellm.cache = LiteLLMCache(type="local")
    litellm.cache.cache = SulciLiteLLMCache(backend="sqlite")

Install:
    pip install "sulci[litellm]"
    # which installs: sulci + litellm>=1.90.0

⚠️  API NOTE — MEASURED, NOT RECALLED (introspected 2026-08-11, litellm 1.96.0).
    **LiteLLM has no `custom` cache type.** `LiteLLMCacheType` is exactly
    {local, redis, redis-semantic, valkey-semantic, s3, disk, qdrant-semantic,
    azure-blob, gcs}. `Cache(type="custom", custom_cache=...)` does not exist
    and never did. The only injection point is *replacing the inner
    implementation after construction* — `litellm.cache.cache = <BaseCache>` —
    which is what `install()` does. If a future LiteLLM adds a first-class
    custom type, switch to it; until then this is the supported shape and it
    is the one their own semantic caches are wired through.

Token counts (1.3.0, oss-0053): on the managed backend (``backend="sulci"``)
    every stored response carries its own ``usage`` — LiteLLM normalises every
    provider to OpenAI's ``prompt_tokens`` / ``completion_tokens`` — so each
    later hit is credited to the dashboard's Tokens Saved. The whole call's
    counts: a hit avoids the whole call. Not sent for a streamed call:
    LiteLLM caches the assembled stream, and when the provider sent no usage
    in its chunks LiteLLM estimates the counts — never guessed here. Not sent
    for more than one choice (``n`` > 1), nor for embeddings (LiteLLM caches
    those without a ``usage``). A response whose counts cannot be read is
    stored without them. Local backends are sent none (they store no counts).

⚠️  The `key` argument LiteLLM passes to `get_cache`/`set_cache` is a hash of
    the whole request, so it is useless as a semantic lookup. The prompt
    arrives in `kwargs["messages"]` (chat) or `kwargs["input"]` (Responses
    API) — the same contract `RedisSemanticCache._get_prompt_from_kwargs`
    reads. This adapter mirrors that helper rather than inventing its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# litellm is optional — guard the import clearly.
try:
    from litellm.caching.base_cache import BaseCache
except ImportError as _ll_err:  # pragma: no cover
    raise ImportError(
        "litellm is required for sulci.integrations.litellm.\n"
        'Install: pip install "sulci[litellm]"\n'
        "or:      pip install litellm"
    ) from _ll_err

from sulci import Cache
from sulci._usage import accepts_usage, checked_usage
from sulci.sessions.errors import SulciSessionError

__all__ = ["SulciLiteLLMCache", "install"]


def _usage_from_value(value: Any) -> Any:
    """oss-0053: the raw usage object inside what LiteLLM hands ``set_cache`` — ``{"timestamp", "response"}`` with
    the response as a JSON string (``model_dump_json``) or a dict — or a response given directly. None if none."""
    if isinstance(value, dict) and "response" in value:
        value = value["response"]
    if isinstance(value, (bytes, str)):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    if isinstance(value, dict):
        return value.get("usage")
    return getattr(value, "usage", None)


def _prompt_from_kwargs(**kwargs: Any) -> Optional[str]:
    """
    Flatten a LiteLLM request into the string Sulci embeds.

    Mirrors ``RedisSemanticCache._get_prompt_from_kwargs``: chat requests
    carry ``messages``; Responses-API requests carry ``input``.
    """
    messages = kwargs.get("messages")
    if messages:
        parts = []
        for m in messages:
            if isinstance(m, dict):
                content = m.get("content")
            else:  # pydantic message objects
                content = getattr(m, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(
                        block.get("text"), str
                    ):
                        parts.append(block["text"])
        prompt = "\n".join(p for p in parts if p).strip()
        return prompt or None

    raw_input = kwargs.get("input")
    if raw_input is None:
        return None
    if isinstance(raw_input, str):
        return raw_input.strip() or None
    parts = []
    if isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and isinstance(
                            block.get("text"), str
                        ):
                            parts.append(block["text"])
    prompt = "\n".join(p for p in parts if p).strip()
    return prompt or None


class SulciLiteLLMCache(BaseCache):
    """
    Context-aware semantic cache for LiteLLM, backed by Sulci.

    Args:
        cache: An existing :class:`sulci.Cache`, or None to build one.
        namespace_by_model: When True (default), a response cached for one
            model is never served for another — the model name becomes the
            Sulci ``tenant_id``. Set False to share across models, which is
            cheaper and occasionally wrong. 1.0.1: on the managed backend
            (``backend="sulci"``) models cannot be kept apart — the API key is
            the only partition — so it is turned off there with a warning.
        session_key: Which LiteLLM metadata key carries the conversation id
            used for context blending. Defaults to ``"sulci_session_id"``;
            pass it as ``metadata={"sulci_session_id": ...}`` on the call.
        **cache_kwargs: Passed to :class:`sulci.Cache`.

    Mirrors the ``namespace_by_llm`` decision in
    :mod:`sulci.integrations.langchain` deliberately — the two adapters
    should not disagree about isolation defaults.
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        *,
        namespace_by_model: bool = True,
        session_key: str = "sulci_session_id",
        **cache_kwargs: Any,
    ) -> None:
        if cache is not None and cache_kwargs:
            raise TypeError(
                "SulciLiteLLMCache() takes either an existing `cache` or "
                "`**cache_kwargs`, not both."
            )
        self.sulci: Cache = cache if cache is not None else Cache(**cache_kwargs)
        # 1.0.1 (G33): the model was the tenant_id, which the managed backend never sent — answers were shared across
        # models while namespace_by_model said otherwise. It now refuses a tenant_id (TypeError), which this adapter
        # would turn into a miss on every call. Say so once and share, as SulciCache's namespace_by_llm does
        # (a log warning, not warnings.warn: a caller's -W error must not turn it into an exception).
        if namespace_by_model and getattr(self.sulci, "_is_remote_transport", False):
            logger.warning(
                "SulciLiteLLMCache: namespace_by_model=True cannot isolate models on the managed backend "
                "(backend='sulci'): the API key is the only partition. Responses are shared across models. "
                "Pass namespace_by_model=False to silence this, or use one API key per model."
            )
            namespace_by_model = False
        self.namespace_by_model = namespace_by_model
        self.session_key = session_key
        self._usage_warned = False   # oss-0053: say once that counts could not be read, not on every set

    # ── helpers ──────────────────────────────────────────────────────────
    def _tenant(self, **kwargs: Any) -> Optional[str]:
        if not self.namespace_by_model:
            return None
        model = kwargs.get("model")
        return str(model) if model else None

    def _standalone(self, **kwargs: Any) -> bool:
        """oss-0051: ``metadata={"sulci_standalone": True}`` stores this call's answer even when its session already
        holds a turn (sulci 1.1.1 stores no follow-up otherwise)."""
        meta = kwargs.get("metadata") or {}
        return isinstance(meta, dict) and meta.get("sulci_standalone") is True

    def _session(self, **kwargs: Any) -> Optional[str]:
        meta = kwargs.get("metadata") or {}
        if isinstance(meta, dict):
            sid = meta.get(self.session_key)
            if sid:
                return str(sid)
        return None

    def _usage(self, value: Any, **kwargs: Any) -> Optional[dict]:
        """oss-0053: the response's token counts, checked, on the managed backend only — else None. Never raises:
        the response has been paid for, and it is stored whether or not its counts are. None for a streamed call
        (LiteLLM may have estimated its counts) and for more than one choice (one is served on a hit)."""
        if not getattr(self.sulci, "_is_remote_transport", False):
            return None
        if kwargs.get("stream") or kwargs.get("n") not in (None, 1):
            return None
        raw = _usage_from_value(value)
        if raw is None:
            return None
        try:
            return checked_usage(raw)
        except (TypeError, ValueError) as exc:
            if not self._usage_warned:
                self._usage_warned = True
                logger.warning("sulci set_cache: token counts not recorded (stored without them): %s", exc)
            return None

    @staticmethod
    def _decode(stored: Any) -> Any:
        """Round-trip whatever ``set_cache`` stored back into a response."""
        if stored is None:
            return None
        if isinstance(stored, bytes):
            stored = stored.decode("utf-8")
        if isinstance(stored, str):
            try:
                return json.loads(stored)
            except json.JSONDecodeError:
                return stored
        return stored

    # ── BaseCache contract ───────────────────────────────────────────────
    def get_cache(self, key: str, **kwargs: Any) -> Any:
        prompt = _prompt_from_kwargs(**kwargs)
        if prompt is None:
            return None
        try:
            resp, similarity, _depth = self.sulci.get(
                prompt,
                tenant_id=self._tenant(**kwargs),
                session_id=self._session(**kwargs),
            )
        except Exception as exc:
            logger.warning("sulci get_cache failed, treating as miss: %s", exc)
            return None

        # LiteLLM's own semantic caches surface the score this way; match it
        # so existing dashboards keep working.
        kwargs.setdefault("metadata", {})["semantic-similarity"] = (
            similarity if resp is not None else 0.0
        )
        return self._decode(resp)

    def set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        prompt = _prompt_from_kwargs(**kwargs)
        if prompt is None:
            return
        try:
            payload = (
                value
                if isinstance(value, str)
                else json.dumps(value, default=str)
            )
        except (TypeError, ValueError) as exc:
            logger.warning("sulci set_cache: unserialisable value: %s", exc)
            return
        try:
            extra = {"standalone": True} if self._standalone(**kwargs) else {}   # oss-0051
            usage = self._usage(value, **kwargs)   # oss-0053
            if usage is not None and accepts_usage(self.sulci.set):
                extra["usage"] = usage
            self.sulci.set(
                prompt,
                payload,
                tenant_id=self._tenant(**kwargs),
                session_id=self._session(**kwargs),
                **extra,
            )
        except SulciSessionError as exc:
            logger.warning("sulci set_cache: the conversation turn was not recorded: %s", exc)   # says whether stored (oss-0049)
        except Exception as exc:
            logger.warning("sulci set_cache failed: %s", exc)

    async def async_get_cache(self, key: str, **kwargs: Any) -> Any:
        # sulci.Cache is sync; keep the event loop free.
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.get_cache(key, **kwargs)
        )

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.set_cache(key, value, **kwargs)
        )

    async def async_set_cache_pipeline(
        self, cache_list: list, **kwargs: Any
    ) -> None:
        for entry in cache_list:
            try:
                k, v = entry
            except (TypeError, ValueError):
                continue
            await self.async_set_cache(k, v, **kwargs)

    def disconnect(self) -> None:  # pragma: no cover - parity with BaseCache
        pass

    # ── observability ────────────────────────────────────────────────────
    def stats(self) -> dict:
        """Passthrough to :meth:`sulci.Cache.stats`."""
        return self.sulci.stats()


def install(
    cache: Optional[Cache] = None,
    *,
    namespace_by_model: bool = True,
    session_key: str = "sulci_session_id",
    **cache_kwargs: Any,
) -> "SulciLiteLLMCache":
    """
    Wire Sulci in as LiteLLM's cache and return the adapter.

    Constructs a ``litellm.caching.Cache`` (any type — the inner
    implementation is immediately replaced) and swaps in
    :class:`SulciLiteLLMCache`. See the module docstring for why this is the
    injection point rather than a ``type="custom"`` argument.
    """
    import litellm
    from litellm.caching import Cache as _LiteLLMCache

    adapter = SulciLiteLLMCache(
        cache,
        namespace_by_model=namespace_by_model,
        session_key=session_key,
        **cache_kwargs,
    )
    container = _LiteLLMCache(type="local")
    container.cache = adapter
    litellm.cache = container
    return adapter
