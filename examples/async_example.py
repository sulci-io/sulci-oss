"""
examples/async_example.py
==========================
Sulci AsyncCache — non-blocking async caching demo.

Works with OpenAI or Anthropic API keys, or falls back to a mock LLM.

Requirements:
    pip install "sulci[sqlite]"
    pip install anthropic          # for Anthropic (optional)
    pip install openai             # for OpenAI (optional)

    export ANTHROPIC_API_KEY=sk-ant-...   # optional
    export OPENAI_API_KEY=sk-...          # optional
    # No key set → mock LLM used automatically

Run:
    python examples/async_example.py

How it works:
    AsyncCache wraps sulci.Cache and delegates all operations to a thread
    pool via asyncio.to_thread() — the event loop is never blocked.

    This is the correct pattern for FastAPI endpoints, LangChain async
    chains, LlamaIndex async agents, and any asyncio-based application.

Pattern:
    cache = AsyncCache(backend="sqlite", context_window=4)
    response, sim, depth = await cache.aget(query, session_id=session_id)
    await cache.aset(query, response, session_id=session_id)
    result = await cache.acached_call(query, llm_fn, session_id=session_id)

    # Multi-tenant (v0.8.1): tenant_id + plan mirror sulci.Cache, keyword-only.
    await cache.aset(query, response, session_id=sid, tenant_id="acme", plan="pro")
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sulci import AsyncCache

# ── Cache configuration ───────────────────────────────────────────────────────
cache = AsyncCache(
    backend        = "sqlite",
    db_path        = os.path.join(tempfile.mkdtemp(prefix="sulci_async_"), "cache"),
    threshold      = 0.85,
    context_window = 4,        # context-aware: remember last 4 turns per session
    query_weight   = 0.70,
    context_decay  = 0.60,
    session_ttl    = 3600,
)

# ── LLM setup: OpenAI → Anthropic → mock ─────────────────────────────────────
_has_openai    = bool(os.environ.get("OPENAI_API_KEY"))
_has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

print("── API key detection ──────────────────────────────")
print(f"  OPENAI_API_KEY    : {'✓ found' if _has_openai    else '✗ not set'}")
print(f"  ANTHROPIC_API_KEY : {'✓ found' if _has_anthropic else '✗ not set'}")

# Mock LLM body — defined first so the real-provider blocks below can fall
# back to it on AuthenticationError without redefining the responses (#20).
def _mock_call(query: str) -> str:
    q = query.lower()
    if "semantic" in q or "cache" in q:
        return "Semantic caching stores LLM responses indexed by meaning, not exact text."
    if "async" in q or "asyncio" in q or "fastapi" in q:
        return "AsyncCache wraps sulci.Cache with asyncio.to_thread() so the event loop is never blocked."
    if "langchain" in q:
        return "LangChain is a framework for building LLM-powered applications."
    if "cap" in q or "distributed" in q:
        return "The CAP theorem: Consistency, Availability, Partition tolerance — pick two."
    if "example" in q or "show" in q:
        return "cache = AsyncCache(backend='sqlite'); response, sim, depth = await cache.aget(query)."
    if "different" in q or "standard" in q:
        return "Unlike sync Cache.get(), AsyncCache.aget() yields the event loop — other requests proceed."
    if "benefit" in q or "advantage" in q:
        return "Benefits: non-blocking I/O, higher FastAPI throughput, compatible with all async frameworks."
    return f"[Mock] Answer to: {query[:60]}"

_call_llm   = None
_using_mock = False

# 1. Try OpenAI
if _has_openai:
    try:
        import openai
        _oa_client = openai.OpenAI()
        _oa_state  = {"rejected": False}
        def _call_llm(query: str) -> str:
            # #20 — once the key is rejected, fall back to mock so the rest of
            # the demo still runs (instead of raising AuthenticationError every
            # call).
            if _oa_state["rejected"]:
                return _mock_call(query)
            try:
                resp = _oa_client.chat.completions.create(
                    model    = "gpt-4o-mini",
                    messages = [{"role": "user", "content": query}],
                )
                return resp.choices[0].message.content
            except openai.AuthenticationError:
                print()
                print("⚠  OPENAI_API_KEY rejected by OpenAI (HTTP 401).")
                print("   Verify your key at https://platform.openai.com/api-keys")
                print("   Falling back to mock LLM for the rest of this demo.\n")
                _oa_state["rejected"] = True
                return _mock_call(query)
        print("  → Using: OpenAI gpt-4o-mini\n")
    except ImportError:
        print("  ✗ openai not installed — run: pip install openai")

# 2. Try Anthropic
if _call_llm is None and _has_anthropic:
    try:
        import anthropic
        _ant_client = anthropic.Anthropic()
        _ant_state  = {"rejected": False}
        def _call_llm(query: str) -> str:
            if _ant_state["rejected"]:
                return _mock_call(query)
            try:
                msg = _ant_client.messages.create(
                    model      = "claude-haiku-4-5-20251001",
                    max_tokens = 1024,
                    messages   = [{"role": "user", "content": query}],
                )
                return msg.content[0].text
            except anthropic.AuthenticationError:
                print()
                print("⚠  ANTHROPIC_API_KEY rejected by Anthropic (HTTP 401).")
                print("   Verify your key at https://console.anthropic.com/settings/keys")
                print("   Falling back to mock LLM for the rest of this demo.\n")
                _ant_state["rejected"] = True
                return _mock_call(query)
        print("  → Using: Anthropic claude-haiku-4-5-20251001\n")
    except ImportError:
        print("  ✗ anthropic not installed — run: pip install anthropic")

# 3. Mock fallback
if _call_llm is None:
    _using_mock = True
    print("  → Using: mock LLM (no API key — set OPENAI_API_KEY or ANTHROPIC_API_KEY)\n")
    _call_llm = _mock_call


# ── Chat class ────────────────────────────────────────────────────────────────
class Chat:
    """
    Multi-turn async chat with context-aware semantic caching.

    Demonstrates the idiomatic AsyncCache pattern:
    - await cache.acached_call() for LLM-backed queries
    - session_id scopes context to this conversation
    - hits return in <10ms without calling the LLM
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history    = []

    async def ask(self, question: str) -> dict:
        result = await cache.acached_call(
            question,
            _call_llm,
            session_id = self.session_id,
        )
        self.history.append({
            "question":      question,
            "answer":        result["response"],
            "source":        result["source"],
            "latency_ms":    result["latency_ms"],
            "similarity":    result["similarity"],
            "context_depth": result["context_depth"],
            "cache_hit":     result["cache_hit"],
        })
        return result

    def print_history(self) -> None:
        print(f"\n── Session '{self.session_id}' history ───────────────────────")
        for i, t in enumerate(self.history, 1):
            icon = "⚡" if t["source"] == "cache" else "🌐"
            ctx  = f"  ctx={t['context_depth']}" if t["context_depth"] else ""
            sim  = f"  sim={t['similarity']:.0%}" if t["source"] == "cache" else ""
            print(f"{i}. {icon} [{t['source'].upper()}] {t['latency_ms']:.0f}ms{sim}{ctx}")
            print(f"   Q: {t['question']}")
            print(f"   A: {t['answer'][:90]}...")
            print()


def print_result(label: str, r: dict) -> None:
    icon = "⚡" if r["cache_hit"] else "🌐"
    sim  = f" sim={r['similarity']:.0%}" if r["cache_hit"] else ""
    ctx  = f" ctx={r['context_depth']}" if r["context_depth"] else ""
    print(f"  {icon} [{r['source'].upper()}] {r['latency_ms']:.0f}ms{sim}{ctx} — {label}")


# ── Demo ──────────────────────────────────────────────────────────────────────
async def main():
    print("◈ Sulci AsyncCache — non-blocking async caching demo\n")

    # ── Round 1: fresh questions — all cache misses ───────────────────────────
    # Each question in its own session so context doesn't bleed across topics.
    print("Round 1: Fresh questions — all cache misses")
    print("-" * 54)
    fresh = [
        ("session-caching",  "What is semantic caching?"),
        ("session-async",    "How does AsyncCache work?"),
        ("session-cap",      "What is the CAP theorem?"),
    ]
    sessions = {}
    for sid, q in fresh:
        chat = Chat(session_id=sid)
        sessions[sid] = chat
        r = await chat.ask(q)
        print_result(q, r)

    # ── Round 2: paraphrases in same sessions — hits expected ─────────────────
    print("\nRound 2: Paraphrased questions — cache hits expected")
    print("-" * 54)
    paraphrases = [
        ("session-caching", "How does semantic cache work?"),    # ≈ Round 1 q1
        ("session-async",   "What is AsyncCache in sulci?"),     # ≈ Round 1 q2
        ("session-cap",     "Explain the CAP theorem"),          # ≈ Round 1 q3
    ]
    for sid, q in paraphrases:
        r = await sessions[sid].ask(q)
        print_result(q, r)

    # ── Round 3: context-aware follow-ups ─────────────────────────────────────
    print("\nRound 3: Context-aware follow-ups — single topic session")
    print("-" * 54)
    ctx_chat = Chat(session_id="session-context-demo")
    seed_q   = "What is semantic caching?"
    r = await ctx_chat.ask(seed_q)
    print_result(f"[seed]  {seed_q}", r)

    for q in [
        "Can you give me a simple example?",
        "How is this different from standard caching?",
        "What are the main benefits?",
    ]:
        r = await ctx_chat.ask(q)
        print_result(q, r)

    # ── Round 4: clearly unrelated — cache miss ───────────────────────────────
    print("\nRound 4: Clearly unrelated question — cache miss")
    print("-" * 54)
    r = await Chat(session_id="session-misc").ask("What is the capital of Australia?")
    print_result("What is the capital of Australia?", r)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print()
    s = await cache.astats()
    print("=" * 54)
    print(f"  Total queries  : {s['total_queries']}")
    print(f"  Cache hits     : {s['hits']}  ({s['hit_rate']:.0%} hit rate)")
    print(f"  LLM calls      : {s['misses']}")
    print(f"  Cost saved     : ${s['saved_cost']:.4f}")
    print(f"  Active sessions: {s.get('active_sessions', 'N/A')}")
    print("=" * 54)

    # History for context demo session
    ctx_chat.print_history()

    # ── FastAPI pattern note ───────────────────────────────────────────────────
    print("\n── FastAPI pattern ────────────────────────────────────")
    print("  from sulci import AsyncCache")
    print("  cache = AsyncCache(backend='sqlite', context_window=4)")
    print("")
    print("  @app.post('/chat')")
    print("  async def chat(query: str, session_id: str):")
    print("      response, sim, depth = await cache.aget(query, session_id=session_id)")
    print("      if response:")
    print("          return {'response': response, 'source': 'cache'}")
    print("      response = await call_llm(query)")
    print("      await cache.aset(query, response, session_id=session_id)")
    print("      return {'response': response, 'source': 'llm'}")
    print("")
    print("── Multi-tenant SaaS variant (v0.8.1) ─────────────────")
    print("  # tenant_id + plan mirror sulci.Cache — keyword-only, both async")
    print("  # methods and sync passthroughs. tenant_id partitions entries")
    print("  # (a HARD boundary on Qdrant / Sulci Cloud; a label on SQLite);")
    print("  # plan rides onto the emitted CacheEvent for per-plan telemetry.")
    print("  resp, sim, depth = await cache.aget(")
    print("      query, session_id=session_id, tenant_id=tenant, plan=plan)")
    print("  await cache.aset(")
    print("      query, response, session_id=session_id, tenant_id=tenant, plan=plan)")


if __name__ == "__main__":
    asyncio.run(main())
