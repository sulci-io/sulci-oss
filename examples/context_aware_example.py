"""
examples/context_aware_example.py
==================================
Demonstrates context-aware semantic caching with sulci.

The classic problem:
    "How do I fix it?" means something completely different depending
    on whether the previous turn was about Docker, Python, or billing.
    A stateless cache can't tell them apart - sulci's context window can.

Run:
    pip install "sulci[sqlite]"
    python examples/context_aware_example.py
"""

from sulci import Cache

# ── Fake LLM (no API key needed for this demo) ─────────────────────────
call_count = 0

def fake_llm(query: str) -> str:
    global call_count
    call_count += 1
    responses = {
        "docker":  "Check logs with `docker logs <id>`. Common fix: rebuild with `docker build --no-cache`.",
        "python":  "Run `pip install -r requirements.txt` and check your Python version with `python --version`.",
        "billing": "Go to Account Settings > Billing > Update Payment Method.",
        "default": f"[LLM response to: {query[:50]}]",
    }
    q = query.lower()
    if "docker" in q or "container" in q: return responses["docker"]
    if "python" in q or "import" in q:    return responses["python"]
    if "billing" in q or "payment" in q:  return responses["billing"]
    return responses["default"]


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ══════════════════════════════════════════════════════════════════
# 1. Context-naive vs context-aware comparison
# ══════════════════════════════════════════════════════════════════

separator("1. Setup: seed the cache with known Q&A pairs")

cache = Cache(backend="sqlite", threshold=0.82, context_window=6, db_path="/tmp/sulci_ctx_demo")
cache.clear()

# Seed specific Q&A pairs the cache will return
seed_pairs = [
    ("My Docker container crashes on startup",      fake_llm("docker container crash")),
    ("I have a Python import error in my script",   fake_llm("python import error")),
    ("My billing payment keeps failing",            fake_llm("billing payment")),
]
for q, r in seed_pairs:
    cache.set(q, r, session_id="seed")

print(f"  Seeded {len(seed_pairs)} entries")


# ══════════════════════════════════════════════════════════════════
# 2. Ambiguous follow-up: same query, different contexts
# ══════════════════════════════════════════════════════════════════

separator('2. Ambiguous query: "How do I fix it?" in different contexts')

test_cases = [
    ("docker-session",  "My Docker container crashes on startup"),
    ("python-session",  "I have a Python import error in my script"),
    ("billing-session", "My billing payment keeps failing"),
]

for session_id, first_query in test_cases:
    # First turn: specific problem
    r1 = cache.cached_call(first_query, fake_llm, session_id=session_id)
    print(f"\n  [{session_id}] Turn 1: '{first_query[:45]}...'")
    print(f"    → source={r1['source']}, context_depth={r1['context_depth']}")

    # Second turn: ambiguous follow-up
    r2 = cache.cached_call("How do I fix it?", fake_llm, session_id=session_id)
    print(f"  [{session_id}] Turn 2: 'How do I fix it?'")
    print(f"    → source={r2['source']}, similarity={r2['similarity']}, "
          f"context_depth={r2['context_depth']}")

    # Show what was returned
    snippet = r2['response'][:70].replace('\n', ' ')
    print(f"    → response: {snippet}...")


# ══════════════════════════════════════════════════════════════════
# 3. Manual context injection
# ══════════════════════════════════════════════════════════════════

separator("3. Manual context injection (pre-seeding a session)")

cache2 = Cache(backend="sqlite", context_window=4, db_path="/tmp/sulci_ctx_demo2")
cache2.clear()
cache2.set("What is the retry logic for Redis connections?",
           "Use exponential backoff: 1s, 2s, 4s, 8s. Max 3 retries.")

# Manually inject context before the first cached_call
ctx = cache2.get_context("infra-team")
ctx.add_turn("We are migrating from Redis to Valkey", role="user")
ctx.add_turn("Understood, I will keep that context in mind.", role="assistant")
ctx.add_turn("We use Python and asyncio for our workers", role="user")

print("\n  Pre-seeded context:")
for t in ctx.turns:
    print(f"    [{t.role:9s}] {t.text[:65]}")

result = cache2.cached_call(
    "What is the retry strategy for our cache connections?",
    fake_llm,
    session_id="infra-team",
)
print(f"\n  Query: 'What is the retry strategy for our cache connections?'")
print(f"  → source={result['source']}, context_depth={result['context_depth']}")
print(f"  → {result['response'][:80]}")


# ══════════════════════════════════════════════════════════════════
# 4. Context summary + session inspection
# ══════════════════════════════════════════════════════════════════

separator("4. Context summary")

summary = cache.context_summary("docker-session")
print(f"\n  Session: docker-session")
print(f"  Depth  : {summary['depth']}/{summary['max_turns']}")
print(f"  Turns  :")
for t in summary["turns"]:
    print(f"    [{t['role']:9s}] {t['text'][:55]}  (age={t['age_s']}s)")


# ══════════════════════════════════════════════════════════════════
# 5. Clear a session (new topic, same user)
# ══════════════════════════════════════════════════════════════════

separator("5. Clearing session context (user starts new topic)")

print(f"\n  Before clear: depth={cache.context_summary('docker-session')['depth']}")
cache.clear_context("docker-session")
print(f"  After  clear: depth={cache.context_summary('docker-session')['depth']}")
print("  → 'How do I fix it?' will now match without Docker context")


# ══════════════════════════════════════════════════════════════════
# 6. Stats
# ══════════════════════════════════════════════════════════════════

separator("6. Stats")

s = cache.stats()
print(f"\n  hits           : {s['hits']}")
print(f"  misses         : {s['misses']}")
print(f"  hit_rate       : {s['hit_rate']:.1%}")
print(f"  active_sessions: {s['active_sessions']}")
print(f"  saved_cost     : ${s['saved_cost']:.4f}")
print(f"\n  LLM calls made : {call_count}  (rest served from cache)")
print()
