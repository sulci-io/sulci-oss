"""
examples/anthropic_example.py
==============================
Production-ready Sulci + Anthropic Claude integration with context awareness.

Requirements:
    pip install "sulci[sqlite]" anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/anthropic_example.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sulci import Cache

# ── Cache configuration ───────────────────────────────────────────────────
# Per-run tempdir so the demo is idempotent across runs (issue #19).
_DB_PATH = os.path.join(tempfile.mkdtemp(prefix="sulci_anthropic_"), "cache")

cache = Cache(
    backend         = "sqlite",   # default-available; works with `pip install "sulci[sqlite]"`
    threshold       = 0.85,
    embedding_model = "minilm",
    ttl_seconds     = 86400,
    personalized    = False,
    db_path         = _DB_PATH,
    # context-awareness — remove or set context_window=0 for stateless mode
    context_window  = 6,         # remember last 6 turns per session
    query_weight    = 0.70,      # 70% current query, 30% history
    context_decay   = 0.60,
    session_ttl     = 3600,
)

# ── Anthropic client ──────────────────────────────────────────────────────
def _make_mock_call_claude():
    def call_claude(query: str, **_) -> str:
        return f"[Mock] Answer to: {query}"
    return call_claude

try:
    import anthropic
except ImportError:
    print("⚠  anthropic SDK not installed — using mock LLM")
    print("   To run with real Claude: pip install anthropic\n")
    call_claude = _make_mock_call_claude()
else:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠  ANTHROPIC_API_KEY not set — using mock LLM")
        print("   To run with real Claude: export ANTHROPIC_API_KEY=sk-ant-...")
        print("   Get a key at https://console.anthropic.com/\n")
        call_claude = _make_mock_call_claude()
    else:
        _client    = anthropic.Anthropic()
        _mock_call = _make_mock_call_claude()

        # #20 — fail fast (with a useful message) the first time a real
        # call is rejected. Subsequent calls fall back to mock so the
        # rest of the demo still completes; otherwise an expired or
        # invalid key produced a raw HTTPStatusError stack trace mid-run.
        _key_state = {"rejected": False}

        # Keep in step with sulci-web's Demo2.jsx: this file is vendored
        # to sulci.io, so a mismatch ships two different model strings
        # for the same demo. They diverged silently until 2026-08-09.
        def call_claude(query: str, model: str = "claude-sonnet-4-6") -> str:
            if _key_state["rejected"]:
                return _mock_call(query)
            try:
                msg = _client.messages.create(
                    model      = model,
                    max_tokens = 1024,
                    messages   = [{"role": "user", "content": query}],
                )
                return msg.content[0].text
            except anthropic.AuthenticationError:
                print()
                print("⚠  ANTHROPIC_API_KEY rejected by Anthropic (HTTP 401).")
                print("   Verify your key at https://console.anthropic.com/settings/keys")
                print("   Falling back to mock LLM for the rest of this demo.\n")
                _key_state["rejected"] = True
                return _mock_call(query)

        print("✓ Anthropic client ready\n")


# ── Context-aware chat wrapper ────────────────────────────────────────────
class Chat:
    """
    Multi-turn chat with context-aware semantic caching.

    Each call to .ask() passes the session_id so Sulci blends prior
    turns into the lookup vector.  Ambiguous follow-up queries like
    "Can you give me an example?" correctly resolve in the context of
    the ongoing conversation rather than matching unrelated cached entries.

    Usage:
        chat = Chat(user_id="alice")
        chat.ask("How do Python decorators work?")
        chat.ask("Can you give me a simple example?")  # context-resolved
    """

    def __init__(self, user_id: str = "default"):
        self.user_id    = user_id
        self.session_id = f"session-{user_id}"
        self.history    = []

    def ask(self, question: str) -> dict:
        result = cache.cached_call(
            question,
            call_claude,
            user_id    = self.user_id,
            session_id = self.session_id,
        )
        self.history.append({
            "question":      question,
            "answer":        result["response"],
            "source":        result["source"],
            "latency_ms":    result["latency_ms"],
            "similarity":    result["similarity"],
            "context_depth": result["context_depth"],
        })
        return result

    def new_topic(self) -> None:
        """Reset session context when the user switches topics."""
        cache.clear_context(self.session_id)
        print(f"  [Context cleared for {self.session_id}]")

    def print_history(self) -> None:
        print("\n── Conversation history ──────────────────────────────")
        for i, t in enumerate(self.history, 1):
            icon = "⚡" if t["source"] == "cache" else "🌐"
            ctx  = f"  ctx={t['context_depth']}" if t["context_depth"] else ""
            sim  = f"  sim={t['similarity']:.0%}" if t["source"] == "cache" else ""
            print(f"{i}. {icon} [{t['source'].upper()}] {t['latency_ms']:.0f}ms{sim}{ctx}")
            print(f"   Q: {t['question']}")
            print(f"   A: {t['answer'][:100]}...")
            print()


# ── Demo ──────────────────────────────────────────────────────────────────
def main():
    print("◈ Sulci + Anthropic Claude (context-aware)\n")
    chat = Chat(user_id="demo-user")

    # Round 1 — fresh questions (cache misses)
    print("Round 1: Fresh questions")
    print("-" * 50)
    for q in [
        "What is semantic caching?",
        "How do Python decorators work?",
        "Explain the CAP theorem",
    ]:
        r = chat.ask(q)
        icon = "⚡" if r["cache_hit"] else "🌐"
        print(f"{icon} [{r['source'].upper()}] {r['latency_ms']:.0f}ms — {q}")

    # Round 2 — paraphrases hit cache
    print("\nRound 2: Paraphrased questions")
    print("-" * 50)
    for q in [
        "How does semantic cache work?",         # ~ question 1
        "Python decorator pattern explained",    # ~ question 2
        "What does CAP theorem mean?",           # ~ question 3
    ]:
        r = chat.ask(q)
        icon = "⚡" if r["cache_hit"] else "🌐"
        sim  = f" sim={r['similarity']:.0%}" if r["cache_hit"] else ""
        print(f"{icon} [{r['source'].upper()}] {r['latency_ms']:.0f}ms{sim} — {q}")

    # Round 3 — ambiguous follow-ups resolved by context
    print("\nRound 3: Ambiguous follow-ups resolved by context")
    print("-" * 50)
    chat.ask("I'm trying to understand Python decorators better")
    for q in [
        "Can you give me a simple example?",     # context → Python decorators
        "How is this different from closures?",  # context → decorators vs closures
    ]:
        r = chat.ask(q)
        icon = "⚡" if r["cache_hit"] else "🌐"
        ctx  = f" ctx={r['context_depth']}" if r["context_depth"] else ""
        print(f"{icon} [{r['source'].upper()}] {r['latency_ms']:.0f}ms{ctx} — {q}")

    # Stats
    print()
    s = cache.stats()
    print("=" * 50)
    print(f"Queries        : {s['total_queries']}")
    print(f"Cache hits     : {s['hits']}  ({s['hit_rate']:.0%} hit rate)")
    print(f"LLM calls      : {s['misses']}")
    print(f"Cost saved     : ${s['saved_cost']:.4f}")
    print(f"Active sessions: {s.get('active_sessions', 'N/A')}")
    print("=" * 50)

    chat.print_history()


if __name__ == "__main__":
    main()
