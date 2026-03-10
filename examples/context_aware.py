"""
examples/context_aware.py
==========================
Demonstrates Sulci's context-aware caching for multi-turn conversations.

Without context, ambiguous follow-up queries like "How do I fix it?" match
whatever happens to be closest in the cache — which is often wrong.  With
context, sulci blends recent conversation history into the lookup vector so
the same follow-up resolves correctly based on what was discussed earlier.

Run:
    pip install "sulci[sqlite]"
    python examples/context_aware.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sulci import Cache

# ── Mock LLM — replace with your real LLM call ────────────────────────────
FAKE_ANSWERS = {
    "docker": "[LLM] Docker fix: check logs with 'docker logs <id>', ensure ports aren't in use, and verify your Dockerfile CMD.",
    "billing": "[LLM] Billing fix: update your payment method in Account Settings > Billing > Payment Methods.",
    "python": "[LLM] Python fix: check your virtual environment is activated and run 'pip install -r requirements.txt'.",
    "default": "[LLM] I need more context to answer that question well.",
}

def mock_llm(query: str) -> str:
    q = query.lower()
    for k in FAKE_ANSWERS:
        if k in q:
            return FAKE_ANSWERS[k]
    return FAKE_ANSWERS["default"]


# ── Cache configured with a 6-turn context window ─────────────────────────
cache = Cache(
    backend        = "sqlite",
    threshold      = 0.78,      # slightly lower to catch follow-up queries
    context_window = 6,         # remember last 6 turns per session
    query_weight   = 0.70,      # 70% current query, 30% history
    context_decay  = 0.60,      # moderate decay — recent turns matter most
    ttl_seconds    = 3600,
)


def show(label: str, result: dict) -> None:
    icon  = "⚡ HIT " if result["cache_hit"] else "🌐 MISS"
    ctx   = f"  ctx_depth={result['context_depth']}" if result["context_depth"] else ""
    sim   = f"  sim={result['similarity']:.0%}" if result["cache_hit"] else ""
    print(f"  {icon}  {result['latency_ms']:.1f}ms{sim}{ctx}")
    print(f"    Q: {label}")
    print(f"    A: {result['response'][:90]}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# DEMO 1 — Context resolves ambiguous follow-up correctly
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("DEMO 1: Ambiguous follow-up resolved by context")
print("=" * 60)

# ── Session A: Docker conversation ────────────────────────────
print("\n[Session A — Docker]")
r = cache.cached_call(
    "My Docker container crashes on startup",
    mock_llm,
    session_id="session-A",
)
show("My Docker container crashes on startup", r)

# Follow-up — context_depth > 0 here means history influenced the lookup
r = cache.cached_call(
    "How do I fix it?",
    mock_llm,
    session_id="session-A",
)
show("How do I fix it? [after Docker discussion]", r)
print(f"  → context_depth={r['context_depth']} (Docker context influenced lookup)\n")

# ── Session B: Billing conversation ───────────────────────────
print("[Session B — Billing]")
r = cache.cached_call(
    "My billing payment keeps failing",
    mock_llm,
    session_id="session-B",
)
show("My billing payment keeps failing", r)

r = cache.cached_call(
    "How do I fix it?",      # same query, DIFFERENT context → different result
    mock_llm,
    session_id="session-B",
)
show("How do I fix it? [after Billing discussion]", r)
print(f"  → context_depth={r['context_depth']} (Billing context influenced lookup)\n")


# ═══════════════════════════════════════════════════════════════════════════
# DEMO 2 — Context builds up across multiple turns
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("DEMO 2: Context builds over a multi-turn conversation")
print("=" * 60)
print()

session = "session-C"
turns = [
    "I'm building a Python web application",
    "I'm using FastAPI as my framework",
    "My dependencies aren't installing correctly",
    "How should I fix this?",   # highly ambiguous without context
]

for query in turns:
    r = cache.cached_call(query, mock_llm, session_id=session)
    show(query, r)

print("Context window state after 4 turns:")
summary = cache.context_summary(session)
for t in summary["turns"]:
    indicator = "👤" if t["role"] == "user" else "🤖"
    print(f"  {indicator} [{t['role']:9s}] {t['text']}")


# ═══════════════════════════════════════════════════════════════════════════
# DEMO 3 — Manual context injection (restore a saved conversation)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DEMO 3: Manual context injection")
print("=" * 60)
print()

ctx = cache.get_context("session-D")
ctx.add_turn("I'm running Python 3.9 on AWS Lambda", role="user")
ctx.add_turn("OK, I can help with Lambda deployments.", role="assistant")
ctx.add_turn("My Lambda function is timing out", role="user")

# Now ask a follow-up — context already set
r = cache.cached_call(
    "What's the best way to fix it?",
    mock_llm,
    session_id="session-D",
)
show("What's the best way to fix it? [pre-seeded context]", r)
print(f"  context_depth={r['context_depth']} — {summary['depth']} prior turns informed this")


# ═══════════════════════════════════════════════════════════════════════════
# DEMO 4 — clear_context starts a fresh topic in the same session
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DEMO 4: clear_context resets topic mid-session")
print("=" * 60)
print()

session = "session-E"
cache.cached_call("My PostgreSQL query is running slowly", mock_llm, session_id=session)
print("  [Topic 1: PostgreSQL]")
r = cache.cached_call("How do I fix it?", mock_llm, session_id=session)
print(f"  Follow-up resolved with context_depth={r['context_depth']}\n")

# New topic — clear context so follow-ups don't carry over
cache.clear_context(session)
print("  [Context cleared — new topic: billing]")
cache.cached_call("My invoice shows the wrong amount", mock_llm, session_id=session)
r = cache.cached_call("How do I fix it?", mock_llm, session_id=session)
print(f"  Follow-up resolved with context_depth={r['context_depth']} (fresh context)")


# ── Final stats ───────────────────────────────────────────────
print("\n" + "=" * 60)
s = cache.stats()
print(f"Total queries  : {s['total_queries']}")
print(f"Cache hits     : {s['hits']}  ({s['hit_rate']:.0%})")
print(f"LLM calls      : {s['misses']}")
print(f"Cost saved     : ${s['saved_cost']:.4f}")
print(f"Active sessions: {s.get('active_sessions', 'N/A')}")
print("=" * 60)
