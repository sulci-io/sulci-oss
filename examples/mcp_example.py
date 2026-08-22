# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
examples/mcp_example.py
────────────────────────
Drive the Sulci MCP server in-process, without a client.

    pip install "sulci[mcp,sqlite]"
    python examples/mcp_example.py

For the real thing, run `sulci-mcp` and point any MCP client at it over
stdio. See examples/gh_aw_sulci_mcp.md for a GitHub Agentic Workflow.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile

from sulci.integrations.mcp_server import build_server

Q = "What is semantic caching?"
A = "Semantic caching stores LLM responses indexed by meaning, not by exact text."


async def call(server, name: str, **args) -> dict:
    result = await server.call_tool(name, args)
    blocks = getattr(result, "content", result)
    return json.loads("".join(getattr(b, "text", "") for b in blocks))


# Per-run tempdir so re-running is idempotent, matching every other example
# in this directory (issue #19: "no db_path pollution accumulates across
# runs"). A fixed path would let a PREVIOUS run's entries satisfy this run's
# "miss" and the demo would silently prove nothing.
_DB_PATH = os.path.join(tempfile.mkdtemp(prefix="sulci_mcp_"), "cache")


async def main() -> None:
    server = build_server(backend="sqlite", db_path=_DB_PATH, context_window=4)

    print("tools:", [t.name for t in await server.list_tools()])
    print("miss :", await call(server, "cache_lookup", query=Q))
    print("store:", await call(server, "cache_store", query=Q, response=A))
    print("hit  :", await call(server, "cache_lookup", query=Q))
    print("stats:", await call(server, "cache_stats"))


if __name__ == "__main__":
    asyncio.run(main())
