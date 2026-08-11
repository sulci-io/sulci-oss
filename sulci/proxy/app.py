# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
sulci.proxy.app
────────────────
An OpenAI- and Anthropic-compatible HTTP shim that caches on the way through.

This is the zero-code-change door. Point any SDK, CLI or containerised agent
at this process and every model call it makes is cached — no import, no
adapter, no cooperation from the caller:

    export OPENAI_BASE_URL=http://localhost:8787/v1
    export ANTHROPIC_BASE_URL=http://localhost:8787

Run:
    sulci-proxy --backend sqlite --db-path ./sulci_db --port 8787

Install:
    pip install "sulci[proxy]"
    # which installs: sulci + fastapi + uvicorn (httpx is already mandatory)

DESIGN DECISIONS, so they are not re-litigated
──────────────────────────────────────────────
1. **Streaming passes through UNCACHED.** Same call as
   ``SulciCacheLLM.stream_*`` in the LlamaIndex adapter. Reassembling SSE
   into a cacheable body, then re-emitting it as fake chunks, changes
   observable timing and token accounting. Requests with ``"stream": true``
   are forwarded verbatim and never stored.
2. **Only the last user turn is embedded**, with prior turns supplied as
   session context — that is what ``context_window`` is for. Embedding the
   whole flattened transcript would make every turn a unique key and defeat
   the cache.
3. **Tool-call responses are not cached.** A response with ``tool_calls`` /
   ``tool_use`` is a control-flow instruction whose arguments are usually
   state-dependent. Serving a stale one sends an agent to the wrong file.
4. **Auth headers are forwarded, never stored or logged.** The proxy holds
   no credentials of its own.
5. **The cache key includes the model** unless ``--share-across-models``.

⚠️  CORRECTNESS WARNING — the one that matters in CI.
    A cached answer is keyed on the prompt. The repository is not in the
    prompt. A cached "the auth module looks fine" from three commits ago is a
    confidently wrong review of new code. Pass ``x-sulci-tenant-id`` with a
    commit SHA or tree hash for any workload where the answer depends on
    state that changes. The proxy cannot infer this for you.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import httpx
    from fastapi import APIRouter, FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as _px_err:  # pragma: no cover
    raise ImportError(
        "fastapi and uvicorn are required for sulci.proxy.\n"
        'Install: pip install "sulci[proxy]"\n'
        "or:      pip install fastapi uvicorn"
    ) from _px_err

from sulci import Cache
from sulci.integrations._scope import warn_if_scope_unenforced

__all__ = ["build_app", "extract_prompt", "is_cacheable_response"]

OPENAI_UPSTREAM_DEFAULT = "https://api.openai.com"
ANTHROPIC_UPSTREAM_DEFAULT = "https://api.anthropic.com"

# Hop-by-hop and length-dependent headers must not be copied verbatim.
_STRIP_RESPONSE_HEADERS = {
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
}
_STRIP_REQUEST_HEADERS = {"host", "content-length", "accept-encoding"}


# ── payload helpers ──────────────────────────────────────────────────────
def _text_from_content(content: Any) -> str:
    """Flatten OpenAI/Anthropic content, which may be str or block list."""
    if isinstance(content, str):
        return content
    parts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
    return "\n".join(parts)


def extract_prompt(body: dict) -> Optional[str]:
    """
    Return the text to embed: the LAST user turn.

    Returns None when there is no user turn to key on, which the caller
    must treat as "do not cache this request" rather than as an error.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        text = _text_from_content(msg.get("content")).strip()
        if text:
            return text
    return None


def is_cacheable_response(payload: dict) -> bool:
    """
    False for tool-calling responses — see design decision 3.
    """
    if not isinstance(payload, dict):
        return False
    # OpenAI
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if isinstance(message, dict) and message.get("tool_calls"):
            return False
        if choice.get("finish_reason") == "tool_calls":
            return False
    # Anthropic
    if payload.get("stop_reason") == "tool_use":
        return False
    for block in payload.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return False
    return True


def _openai_envelope(text: str, model: str) -> dict:
    return {
        "id": "chatcmpl-sulci-cached",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        # Zero, honestly: no tokens were consumed upstream. Billing
        # reconciliations depend on this not being a fabricated number.
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _anthropic_envelope(text: str, model: str) -> dict:
    return {
        "id": "msg_sulci_cached",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _openai_text(payload: dict) -> Optional[str]:
    for choice in payload.get("choices") or []:
        if isinstance(choice, dict):
            msg = choice.get("message") or {}
            if isinstance(msg, dict):
                text = _text_from_content(msg.get("content"))
                if text:
                    return text
    return None


def _anthropic_text(payload: dict) -> Optional[str]:
    text = _text_from_content(payload.get("content"))
    return text or None


# ── app ──────────────────────────────────────────────────────────────────
def build_app(
    cache: Optional[Cache] = None,
    *,
    openai_upstream: Optional[str] = None,
    anthropic_upstream: Optional[str] = None,
    share_across_models: bool = False,
    timeout: float = 600.0,
    client: Optional[Any] = None,
    **cache_kwargs: Any,
) -> FastAPI:
    """
    Build the proxy app.

    Args:
        cache: An existing :class:`sulci.Cache`, or None to build one.
        openai_upstream / anthropic_upstream: Where misses are forwarded.
        share_across_models: If True, drop the model from the cache scope.
        client: Inject an ``httpx.AsyncClient`` (used by the test suite).
        **cache_kwargs: Passed to :class:`sulci.Cache`.
    """
    if cache is not None and cache_kwargs:
        raise TypeError(
            "build_app() takes either an existing `cache` or `**cache_kwargs`, "
            "not both."
        )
    sulci_cache = cache if cache is not None else Cache(**cache_kwargs)

    if not share_across_models:
        warn_if_scope_unenforced(sulci_cache, feature="per-model cache scoping")

    openai_upstream = (
        openai_upstream
        or os.environ.get("SULCI_OPENAI_UPSTREAM")
        or OPENAI_UPSTREAM_DEFAULT
    ).rstrip("/")
    anthropic_upstream = (
        anthropic_upstream
        or os.environ.get("SULCI_ANTHROPIC_UPSTREAM")
        or ANTHROPIC_UPSTREAM_DEFAULT
    ).rstrip("/")

    app = FastAPI(title="sulci-proxy", version="1")
    app.state.cache = sulci_cache
    app.state.client = client or httpx.AsyncClient(timeout=timeout)
    router = APIRouter()

    def _scope(request: Request, body: dict) -> tuple:
        tenant = request.headers.get("x-sulci-tenant-id")
        if not share_across_models:
            model = str(body.get("model") or "")
            tenant = f"{tenant}::{model}" if tenant else model or None
        return tenant, request.headers.get("x-sulci-session-id")

    def _fwd_headers(request: Request) -> dict:
        return {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _STRIP_REQUEST_HEADERS
        }

    async def _passthrough(request: Request, upstream: str, path: str, raw: bytes):
        """Forward verbatim. Used for streaming and uncacheable shapes."""
        req = app.state.client.build_request(
            request.method,
            f"{upstream}{path}",
            content=raw,
            headers=_fwd_headers(request),
            params=dict(request.query_params),
        )
        resp = await app.state.client.send(req, stream=True)
        headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in _STRIP_RESPONSE_HEADERS
        }
        headers["x-sulci-cache"] = "bypass"
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=headers,
            background=None,
        )

    async def _handle(
        request: Request, path: str, upstream: str, envelope, text_of
    ) -> Response:
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}

        if not isinstance(body, dict) or body.get("stream"):
            return await _passthrough(request, upstream, path, raw)

        prompt = extract_prompt(body)
        tenant, session = _scope(request, body)
        model = str(body.get("model") or "")

        if prompt:
            try:
                hit, similarity, depth = sulci_cache.get(
                    prompt, tenant_id=tenant, session_id=session
                )
            except Exception as exc:
                logger.warning("sulci lookup failed, forwarding: %s", exc)
                hit, similarity, depth = None, 0.0, 0
            if hit is not None:
                return JSONResponse(
                    envelope(hit, model),
                    headers={
                        "x-sulci-cache": "hit",
                        "x-sulci-similarity": str(similarity),
                        "x-sulci-context-depth": str(depth),
                    },
                )

        # Miss — forward and buffer.
        try:
            resp = await app.state.client.request(
                "POST",
                f"{upstream}{path}",
                content=raw,
                headers=_fwd_headers(request),
                params=dict(request.query_params),
            )
        except Exception as exc:
            logger.error("upstream request failed: %s", exc)
            return JSONResponse(
                {"error": {"message": f"upstream request failed: {exc}",
                           "type": "sulci_proxy_error"}},
                status_code=502,
                headers={"x-sulci-cache": "error"},
            )

        headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in _STRIP_RESPONSE_HEADERS
        }
        headers["x-sulci-cache"] = "miss"

        if resp.status_code == 200 and prompt:
            try:
                payload = resp.json()
            except Exception:
                payload = None
            if isinstance(payload, dict) and is_cacheable_response(payload):
                text = text_of(payload)
                if text:
                    try:
                        sulci_cache.set(
                            prompt, text, tenant_id=tenant, session_id=session
                        )
                    except Exception as exc:
                        logger.warning("sulci store failed: %s", exc)
            elif isinstance(payload, dict):
                headers["x-sulci-cache"] = "miss-uncacheable"

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=headers,
            media_type=resp.headers.get("content-type"),
        )

    @router.post("/v1/chat/completions")
    async def openai_chat(request: Request) -> Response:
        return await _handle(
            request,
            "/v1/chat/completions",
            openai_upstream,
            _openai_envelope,
            _openai_text,
        )

    @router.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        return await _handle(
            request,
            "/v1/messages",
            anthropic_upstream,
            _anthropic_envelope,
            _anthropic_text,
        )

    @router.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @router.get("/stats")
    async def stats() -> dict:
        try:
            return sulci_cache.stats()
        except Exception as exc:  # pragma: no cover
            return {"error": str(exc)}

    app.include_router(router)
    return app
