# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci.integrations.mcp_server
──────────────────────────────
Model Context Protocol server exposing a Sulci cache as MCP tools.

This is the door that agentic-CI runners actually use. A containerised
agent (Copilot CLI, Claude Code, Codex, gh-aw) cannot ``pip install sulci``
into its own process — but it can call an MCP server.

Usage (stdio, the transport every MCP client supports):

    sulci-mcp --backend sqlite --db-path ./sulci_db

Or programmatically:

    from sulci.integrations.mcp_server import build_server
    build_server(backend="sqlite").run(transport="stdio")

Install:
    pip install "sulci[mcp]"
    # which installs: sulci + mcp>=2.0.0

⚠️  API-VERSION NOTE — READ BEFORE CHANGING ANY IMPORT HERE.
    Written against **mcp 2.0.0**, introspected 2026-08-11, not recalled.
    In mcp 1.x the entry point was ``mcp.server.fastmcp.FastMCP``. That module
    **does not exist in 2.x**; the class is ``mcp.server.MCPServer`` and the
    decorator signature is ``.tool(name=..., title=..., description=...,
    annotations=...)``. The extra therefore pins ``mcp>=2.0.0`` deliberately —
    a ``>=1.0.0`` pin would resolve and then fail at import on 1.x.

TOOL SEMANTICS
──────────────
``cache_lookup`` and ``cache_stats`` are annotated ``readOnlyHint=True``.
``cache_store`` is a write, and is annotated as such — gh-aw's guidance is
that custom MCP servers should be read-only and writes should go through
safe outputs. ``cache_store`` writes only to the runner's own cache file,
not to repository state, but the annotation is honest so a reviewer can
make that call themselves rather than discovering it.

Set ``SULCI_MCP_READ_ONLY=1`` to register only the read-only tools.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# mcp is optional — guard the import clearly, in the house style of
# sulci.integrations.langchain / .llamaindex.
try:
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations
except ImportError as _mcp_err:  # pragma: no cover
    raise ImportError(
        "mcp>=2.0.0 is required for sulci.integrations.mcp_server.\n"
        'Install: pip install "sulci[mcp]"\n'
        "or:      pip install 'mcp>=2.0.0'\n"
        "Note: mcp 1.x is NOT supported — the FastMCP entry point it used "
        "was removed in 2.0."
    ) from _mcp_err

from sulci import Cache

__all__ = ["build_server", "main"]

_DEFAULT_INSTRUCTIONS = """\
Sulci is a context-aware semantic cache for LLM calls.

Call `cache_lookup` BEFORE doing expensive reasoning about a question that
may have been answered in a previous run. If it returns a hit, use the
cached answer instead of recomputing.

Call `cache_store` AFTER producing an answer you expect to be asked for
again, so a later run can skip the work.

Pass the same `session_id` for every call within one task so the cache can
blend prior turns into the lookup. Pass a `tenant_id` (for example
`owner/repo`) to keep one project's cache separate from another's.

IMPORTANT: a cached answer is keyed on the question, not on the state of
the world. If the answer depends on code that may have changed, include the
commit SHA or a tree hash in `tenant_id` so a stale answer cannot be
returned for new code.
"""


# 1.0.1 (G33): the managed backend's only partition is the API key, so these instructions do not mention tenant_id.
_MANAGED_INSTRUCTIONS = _DEFAULT_INSTRUCTIONS.split("Pass the same `session_id`")[0] + """\
Pass the same `session_id` for every call within one task so the cache can
blend prior turns into the lookup. Do not pass `tenant_id`: this cache is the
managed backend, where the Sulci account behind the API key is the only
partition (a tenant_id is refused, and every key on one account shares one
cache). One Sulci account per project keeps their caches apart.

IMPORTANT: a cached answer is keyed on the question, not on the state of
the world. If the answer depends on code that may have changed, say so in
the question (for example, name the commit) so a stale answer is not reused.
"""


def build_server(
    cache: Optional[Cache] = None,
    *,
    name: str = "sulci",
    read_only: Optional[bool] = None,
    default_tenant_id: Optional[str] = None,
    instructions: str = _DEFAULT_INSTRUCTIONS,
    **cache_kwargs: Any,
) -> MCPServer:
    """
    Build an :class:`MCPServer` exposing ``cache``.

    Args:
        cache: An existing :class:`sulci.Cache`. If omitted, one is built
            from ``cache_kwargs``.
        name: MCP server name, as advertised to the client.
        read_only: If True, ``cache_store`` is not registered. Defaults to
            ``SULCI_MCP_READ_ONLY == "1"``.
        default_tenant_id: Applied when a tool call omits ``tenant_id``.
            Defaults to ``SULCI_MCP_TENANT_ID``.
        instructions: Client-facing guidance on when to call the tools.
        **cache_kwargs: Passed straight to :class:`sulci.Cache`.

    Note that ``sulci.Cache`` defaults to ``backend="chroma"``. Inside a CI
    runner you almost certainly want ``backend="sqlite"``, which needs no
    server and no network egress.
    """
    if cache is None:
        cache = Cache(**cache_kwargs)
    elif cache_kwargs:
        raise TypeError(
            "build_server() takes either an existing `cache` or "
            "`**cache_kwargs`, not both."
        )

    if read_only is None:
        read_only = os.environ.get("SULCI_MCP_READ_ONLY") == "1"
    if default_tenant_id is None:
        default_tenant_id = os.environ.get("SULCI_MCP_TENANT_ID") or None
    # 0.9.3: a reserved default tenant ("global", "") would make every tool call fail; refuse it at build.
    from sulci._scope_ids import refuse_reserved_scope
    refuse_reserved_scope(default_tenant_id, None, personalized=False)
    # 1.0.1 (G33): the managed backend refuses tenant_id on every lookup; a default tenant would fail every tool call.
    if default_tenant_id is not None and getattr(cache, "_is_remote_transport", False):
        raise ValueError("build_server(): --tenant-id / SULCI_MCP_TENANT_ID is not supported with the managed backend "
                         "(backend=\"sulci\"): the Sulci account behind the API key is the only partition, and every "
                         "key on one account shares one cache. Use one Sulci account per tenant.")
    # 1.0.1 (G33): the default instructions tell the agent to pass a tenant_id, which the managed backend refuses.
    if instructions is _DEFAULT_INSTRUCTIONS and getattr(cache, "_is_remote_transport", False):
        instructions = _MANAGED_INSTRUCTIONS

    from sulci import __version__ as _sulci_version  # local: avoids cycle

    server = MCPServer(name=name, version=_sulci_version, instructions=instructions)

    def _tenant(explicit: Optional[str]) -> Optional[str]:
        return explicit if explicit is not None else default_tenant_id

    def _inner(c):
        # An AsyncCache wraps a Cache (its sync passthroughs call it); the tools are sync.
        from sulci.async_cache import AsyncCache
        return c._cache if isinstance(c, AsyncCache) else c

    def _plain_cache(c) -> bool:
        """oss-0049: True when `c` (or the Cache an AsyncCache wraps) uses Cache's own get/set, so the private
        _get/_set (which report what happened) are the same operations."""
        from sulci.async_cache import AsyncCache
        from sulci.core import Cache
        inner = _inner(c)
        if not isinstance(inner, Cache):
            return False
        if c is not inner and (type(c).set is not AsyncCache.set or type(c).get is not AsyncCache.get):
            return False
        return type(inner).set is Cache.set and type(inner).get is Cache.get

    @server.tool(
        name="cache_lookup",
        title="Look up a cached answer",
        description=(
            "Return a semantically similar cached answer for `query`, or a "
            "miss. Does not modify anything. Call this before doing expensive "
            "work on a question that may already have been answered."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    def cache_lookup(
        query: str,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> str:
        """Look up `query`. Returns a JSON object."""
        try:
            # Cache.get returns a 3-TUPLE (response, similarity, context_depth).
            # `response` is None on a miss. Verified against core.py:488-666 on
            # 2026-08-11 — do not "simplify" this to a scalar.
            # oss-0049: Cache.get records a hit's turn; this tool is annotated read-only and says "Does not
            # modify anything", so it looks up through _get(record_hit=False) when it can.
            lookup = getattr(_inner(cache), "_get", None) if _plain_cache(cache) else None
            kwargs = dict(threshold=threshold, tenant_id=_tenant(tenant_id), session_id=session_id)
            resp, similarity, depth = (lookup(query, record_hit=False, **kwargs) if lookup is not None
                                       else cache.get(query, **kwargs))
        except Exception as exc:  # never take the agent down with us
            logger.warning("sulci cache_lookup failed: %s", exc)
            return json.dumps({"cache_hit": False, "error": str(exc)})

        if resp is None:
            return json.dumps(
                {"cache_hit": False, "response": None, "similarity": similarity}
            )
        return json.dumps(
            {
                "cache_hit": True,
                "response": resp,
                "similarity": similarity,
                "context_depth": depth,
            }
        )

    @server.tool(
        name="cache_stats",
        title="Cache hit statistics",
        description=(
            "Hit/miss counts, hit rate and estimated cost saved for this "
            "cache instance. Read-only."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    def cache_stats() -> str:
        """Return `Cache.stats()` as JSON."""
        try:
            return json.dumps(cache.stats(), default=str)
        except Exception as exc:
            logger.warning("sulci cache_stats failed: %s", exc)
            return json.dumps({"error": str(exc)})

    if not read_only:

        @server.tool(
            name="cache_store",
            title="Store an answer in the cache",
            description=(
                "Store `response` as the answer to `query` so a later run can "
                "skip the work. WRITE: this modifies the cache file. It does "
                "not touch repository state. With `session_id`, a query asked "
                "once the session holds a turn (an earlier store, or a lookup "
                "that hit) is a follow-up: it is recorded but not stored "
                "(\"stored\": false), unless `standalone` is true — set it "
                "when `query` makes sense on its own."
            ),
            annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
        )
        def cache_store(
            query: str,
            response: str,
            session_id: Optional[str] = None,
            tenant_id: Optional[str] = None,
            standalone: bool = False,
        ) -> str:
            """Store `query` -> `response`. Returns a JSON object."""
            from sulci.sessions.errors import SulciSessionError
            # oss-0049: a follow-up (the session already holds a turn) is recorded but not stored, so "stored" is
            # read from Cache._set (True / False; None on the managed backend, whose transport does not report it,
            # where it stays true as before).
            # Only for a Cache whose set() is Cache.set itself: a subclass's override (redaction, auditing) must
            # run, and then "stored" cannot be read (None, reported as true, as before oss-0049).
            write = _inner(cache)._set if _plain_cache(cache) else cache.set
            try:
                stored = write(
                    query,
                    response,
                    tenant_id=_tenant(tenant_id),
                    session_id=session_id,
                    # oss-0051: store even when the session already holds a turn (a self-contained query).
                    **({"standalone": True} if standalone else {}),
                )
            except SulciSessionError as exc:
                # 0.10.0: the turn is missing; oss-0049: the entry is stored unless the set was a follow-up.
                logger.warning("sulci cache_store: %s", exc)
                return json.dumps({"stored": getattr(exc, "stored", True) is not False,
                                   "session_recorded": False, "error": str(exc)})
            except Exception as exc:
                logger.warning("sulci cache_store failed: %s", exc)
                return json.dumps({"stored": False, "error": str(exc)})
            return json.dumps({"stored": stored is not False})

    return server


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sulci-mcp",
        description="Serve a Sulci cache over the Model Context Protocol.",
    )
    p.add_argument(
        "--backend",
        default=os.environ.get("SULCI_BACKEND", "sqlite"),
        help="Vector backend (default: sqlite — no server, no egress).",
    )
    p.add_argument(
        "--db-path",
        default=os.environ.get("SULCI_DB_PATH", "./sulci_db"),
        help="Store location (default: ./sulci_db).",
    )
    # 0.10.1: no default here — unset is 0.85 on a local backend and the tenant's setting on backend="sulci".
    p.add_argument("--threshold", type=float, default=None)
    # 0.10.2: no default here either — unset is 4 on a local backend (as before) and, on backend="sulci", the
    # tenant's Context window setting: a default of 4 would be sent with every managed lookup and override it.
    p.add_argument("--context-window", type=int, default=None)
    p.add_argument(
        "--ttl-seconds",
        type=int,
        default=None,
        help="Entry TTL. Default is sulci's own (86400). In gh-aw, note that "
        "the surrounding actions/cache retains for 7 days.",
    )
    p.add_argument("--tenant-id", default=None)
    p.add_argument("--read-only", action="store_true")
    p.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport (default: stdio).",
    )
    return p


def main(argv: Optional[list] = None) -> None:
    """Console-script entry point for ``sulci-mcp``."""
    args = _build_arg_parser().parse_args(argv)

    # stdio transport speaks JSON-RPC on stdout. Anything else printed there
    # corrupts the stream, so force sulci's own chatter to stderr/off.
    if args.transport == "stdio":
        os.environ.setdefault("SULCI_QUIET", "1")
        logging.basicConfig(level=logging.WARNING)

    kwargs: dict = {
        "backend": args.backend,
        "db_path": args.db_path,
        "threshold": args.threshold,
        "context_window": (args.context_window if args.context_window is not None
                           else (None if args.backend == "sulci" else 4)),
    }
    if args.ttl_seconds is not None:
        kwargs["ttl_seconds"] = args.ttl_seconds

    server = build_server(
        read_only=args.read_only or None,
        default_tenant_id=args.tenant_id,
        **kwargs,
    )
    server.run(transport=args.transport)


if __name__ == "__main__":  # pragma: no cover
    main()
