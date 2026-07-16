#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
smoke_test_async.py
===================
End-to-end smoke test for sulci.AsyncCache.

Covers: create → aset → aget hit → aget miss → acached_call → astats →
        aclear → sync passthrough → partition kwargs (tenant_id / plan)

Run:
    python smoke_test_async.py
    make smoke-async

Exits 0 on success, 1 on any failure.
No API key required.
"""

import asyncio
import sys
import tempfile

def _check(label: str, condition: bool) -> None:
    if condition:
        print(f"  ✓  {label}")
    else:
        print(f"  ✗  {label}  ← FAILED")
        sys.exit(1)


async def main() -> None:
    print("◈ Sulci AsyncCache smoke test\n")

    try:
        from sulci import AsyncCache
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

    tmp = tempfile.mkdtemp(prefix="sulci_async_smoke_")

    # ── 1. Create ─────────────────────────────────────────────────────────────
    cache = AsyncCache(
        backend        = "sqlite",
        db_path        = tmp + "/cache",
        threshold      = 0.85,
        context_window = 4,
        session_ttl    = 3600,
    )
    print("Step 1 — Create AsyncCache")
    _check("isinstance AsyncCache", cache.__class__.__name__ == "AsyncCache")
    _check("repr starts with AsyncCache(", repr(cache).startswith("AsyncCache("))

    # ── 2. aset ───────────────────────────────────────────────────────────────
    print("\nStep 2 — aset()")
    q1 = "What is semantic caching?"
    a1 = "Semantic caching stores LLM responses by meaning, not exact text."
    await cache.aset(q1, a1, session_id="smoke-session")
    _check("aset completes without error", True)

    # ── 3. aget — exact hit ───────────────────────────────────────────────────
    print("\nStep 3 — aget() exact hit")
    response, sim, depth = await cache.aget(q1, session_id="smoke-session")
    _check("response matches stored answer", response == a1)
    _check("similarity >= 0.95 on exact match", sim >= 0.95)

    # ── 4. aget — miss ────────────────────────────────────────────────────────
    print("\nStep 4 — aget() cache miss")
    response, sim, depth = await cache.aget(
        "What is the capital of Australia?", session_id="smoke-session"
    )
    _check("response is None on miss", response is None)
    _check("similarity below threshold on miss", sim < 0.85)

    # ── 5. acached_call ───────────────────────────────────────────────────────
    print("\nStep 5 — acached_call()")
    calls = []

    def mock_llm(query: str) -> str:
        calls.append(query)
        return f"[Mock] {query}"

    # First call — miss, LLM invoked
    result = await cache.acached_call(
        "How does LangChain work?",
        mock_llm,
        session_id = "smoke-session",
    )
    _check("source is 'llm' on miss", result["source"] == "llm")
    _check("LLM was called once", len(calls) == 1)
    _check("result has all expected keys",
           all(k in result for k in
               ["response", "source", "similarity", "latency_ms",
                "cache_hit", "context_depth"]))

    # Second call — hit, LLM not invoked
    result2 = await cache.acached_call(
        "How does LangChain work?",
        mock_llm,
        session_id = "smoke-session",
    )
    _check("source is 'cache' on second call", result2["source"] == "cache")
    _check("LLM NOT called on cache hit", len(calls) == 1)

    # ── 6. Context methods ────────────────────────────────────────────────────
    print("\nStep 6 — context methods")
    ctx = await cache.aget_context("smoke-session")
    _check("aget_context returns ContextWindow", hasattr(ctx, "depth"))
    _check("context depth >= 1 after conversation", ctx.depth >= 1)

    await cache.aclear_context("smoke-session")
    ctx2 = await cache.aget_context("smoke-session")
    _check("aclear_context resets depth to 0", ctx2.depth == 0)

    summary = await cache.acontext_summary()
    _check("acontext_summary returns dict", isinstance(summary, dict))

    # ── 7. astats ─────────────────────────────────────────────────────────────
    print("\nStep 7 — astats()")
    s = await cache.astats()
    _check("hits >= 1", s["hits"] >= 1)
    _check("misses >= 1", s["misses"] >= 1)
    _check("hit_rate between 0 and 1", 0.0 <= s["hit_rate"] <= 1.0)
    _check("saved_cost >= 0", s["saved_cost"] >= 0.0)

    # ── 8. aclear ─────────────────────────────────────────────────────────────
    print("\nStep 8 — aclear()")
    await cache.aclear()
    s2 = await cache.astats()
    _check("hits reset to 0 after aclear", s2["hits"] == 0)
    _check("misses reset to 0 after aclear", s2["misses"] == 0)

    # ── 9. sync passthrough ───────────────────────────────────────────────────
    print("\nStep 9 — sync passthrough (get / set / stats / clear)")
    cache.set("What is asyncio?", "asyncio is Python's async I/O framework.")
    resp, sim, _ = cache.get("What is asyncio?")
    _check("sync set/get works on AsyncCache", resp is not None)
    s3 = cache.stats()
    _check("sync stats() returns dict", isinstance(s3, dict))

    # ── 10. partition kwargs — tenant_id / plan / metadata (v0.8.1, #108) ─────
    # AsyncCache now mirrors Cache's partition kwargs. Exercise the real async
    # path end-to-end (not just signatures): a tenant-scoped round-trip plus
    # the plan tier and metadata, then the same kwargs on the sync passthrough.
    print("\nStep 10 — partition kwargs (tenant_id / plan / metadata)")
    await cache.aset(
        "What is multi-tenancy?",
        "Multi-tenancy isolates each customer's data within shared infra.",
        tenant_id = "acme",
        plan      = "pro",
        metadata  = {"source": "kb"},
    )
    resp, sim, _ = await cache.aget("What is multi-tenancy?", tenant_id="acme", plan="pro")
    _check("aset/aget accept tenant_id + plan (+ metadata)", resp is not None)

    r = await cache.acached_call(
        "What is a tenant?",
        lambda q: "A tenant is an isolated customer partition.",
        tenant_id = "acme",
        plan      = "pro",
    )
    _check("acached_call accepts tenant_id + plan", r["source"] in ("cache", "llm"))

    cache.set("edge query", "edge answer", tenant_id="acme", plan="pro", metadata={"k": "v"})
    resp, _, _ = cache.get("edge query", tenant_id="acme", plan="pro")
    _check("sync passthrough accepts tenant_id + plan", resp is not None)

    # ── Done ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  ✅  All AsyncCache smoke tests passed")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
