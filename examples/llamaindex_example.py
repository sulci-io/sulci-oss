"""
examples/llamaindex_example.py
================================
Sulci Cache + LlamaIndex — context-aware semantic caching demo.

Works with OpenAI or Anthropic API keys, or falls back to a mock LLM.

Requirements:
    pip install "sulci[sqlite,llamaindex]"
    pip install llama-index-llms-openai     # for OpenAI
    pip install llama-index-llms-anthropic  # for Anthropic

    export OPENAI_API_KEY=sk-...            # OpenAI (checked first)
    export ANTHROPIC_API_KEY=sk-ant-...    # Anthropic (checked second)
    # No key set → mock LLM used automatically

Run:
    python examples/llamaindex_example.py

How it works:
    SulciCacheLLM wraps any LlamaIndex LLM and intercepts complete()
    and chat() calls. Set Settings.llm = SulciCacheLLM(...) and every
    LlamaIndex component — query engines, agents, RAG chains — caches
    automatically with no further code changes.

    Context-aware mode (context_window=4) blends prior turns into the
    lookup vector so ambiguous follow-up queries resolve correctly within
    the ongoing conversation. Each session_id keeps context isolated —
    questions in one session don't influence lookups in another.
"""
import os, sys, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.llm import LLM
from llama_index.core import Settings

from sulci.integrations.llamaindex import SulciCacheLLM

# ── LLM setup: OpenAI → Anthropic → mock ─────────────────────────────────────
_has_openai    = bool(os.environ.get("OPENAI_API_KEY"))
_has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

print("── API key detection ──────────────────────────────")
print(f"  OPENAI_API_KEY    : {'✓ found' if _has_openai    else '✗ not set'}")
print(f"  ANTHROPIC_API_KEY : {'✓ found' if _has_anthropic else '✗ not set'}")

_inner_llm  = None
_using_mock = False

# 1. Try OpenAI
if _has_openai:
    try:
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        _inner_llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0)
        print("  → Using: OpenAI gpt-4o-mini\n")
    except ImportError:
        print("  ✗ llama-index-llms-openai not installed — run: pip install llama-index-llms-openai")

# 2. Try Anthropic
if _inner_llm is None and _has_anthropic:
    try:
        from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
        _inner_llm = LlamaAnthropic(
            model   = "claude-haiku-4-5-20251001",
            api_key = os.environ["ANTHROPIC_API_KEY"],
        )
        print("  → Using: Anthropic claude-haiku-4-5-20251001\n")
    except ImportError:
        print("  ✗ llama-index-llms-anthropic not installed — run: pip install llama-index-llms-anthropic")

# Defined unconditionally so real-LLM calls can fall back to it when a
# provider rejects the API key mid-run (sibling followup to #20 from v0.5.4 —
# anthropic_example.py and async_example.py already ship this pattern).
_call_log: list = []   # external log — survives Pydantic's internal copy


class _MockLLM(LLM):
    """Deterministic mock — returns fixed answers for demo purposes."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="mock-llm")

    def _respond(self, text: str) -> str:
        _call_log.append(1)
        t = text.lower()
        if "semantic" in t or "cache" in t:
            return "Semantic caching stores LLM responses indexed by meaning, not exact text."
        if "llama" in t or "llamaindex" in t:
            return "LlamaIndex is a data framework for building LLM-powered agents over your data."
        if "cap" in t or "distributed" in t:
            return "The CAP theorem states a distributed system can guarantee only two of: Consistency, Availability, Partition tolerance."
        if "example" in t or "show" in t:
            return "Here is a simple example: cache = Cache(backend='sqlite'); cache.set('q', 'answer')."
        if "different" in t or "standard" in t or "traditional" in t:
            return "Unlike standard caches that match exact strings, semantic caching matches by meaning."
        if "benefit" in t or "advantage" in t:
            return "Benefits: lower LLM costs, faster responses, and improved hit rates in multi-turn conversations."
        return f"[Mock] Answer to: {text[:60]}"

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self._respond(prompt))

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        last = str(messages[-1].content) if messages else ""
        return ChatResponse(message=ChatMessage(
            role=MessageRole.ASSISTANT, content=self._respond(last)
        ))

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        result = self.complete(prompt, formatted, **kwargs)
        def gen():
            yield CompletionResponse(text=result.text, delta=result.text)
        return gen()

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        result = self.chat(messages, **kwargs)
        def gen():
            yield ChatResponse(message=result.message, delta=result.message.content or "")
        return gen()

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, formatted, **kwargs)

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        result = self.complete(prompt, formatted, **kwargs)
        async def gen():
            yield CompletionResponse(text=result.text, delta=result.text)
        return gen()

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        result = self.chat(messages, **kwargs)
        async def gen():
            yield ChatResponse(message=result.message, delta=result.message.content or "")
        return gen()


# 3. Mock fallback
if _inner_llm is None:
    _using_mock = True
    print("  → Using: mock LLM (no API key found — set OPENAI_API_KEY or ANTHROPIC_API_KEY)\n")
    _inner_llm = _MockLLM()


# Sentinel — set to True the first time a real-LLM call is rejected as
# unauthorised. All subsequent calls in this run route to the mock so the
# rest of the demo still completes. Same shape as _key_state in
# anthropic_example.py and async_example.py (both v0.5.4).
_key_state = {"rejected": False}


# ── Wrap with SulciCacheLLM and register globally ─────────────────────────────
llm = SulciCacheLLM(
    llm            = _inner_llm,
    backend        = "sqlite",
    db_path        = os.path.join(tempfile.mkdtemp(prefix="sulci_li_"), "cache"),
    threshold      = 0.90,    # strict — avoids false positives with real embeddings
    context_window = 4,       # context-aware: remember last 4 turns per session
    query_weight   = 0.70,
    context_decay  = 0.60,
    session_ttl    = 3600,
)

# Every LlamaIndex component — query engines, agents, RAG chains — picks this up
Settings.llm = llm
print("✓ SulciCacheLLM registered as Settings.llm\n")

# ── Per-call stats tracker ────────────────────────────────────────────────────
_stats = {"hits": 0, "misses": 0}


# ── Chat class ────────────────────────────────────────────────────────────────
class Chat:
    """
    Multi-turn chat using SulciCacheLLM via Settings.llm.

    Each Chat instance uses its own session_id so context blending stays
    scoped to that conversation — questions in one session don't influence
    lookup vectors in another.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history    = []
        self._cache     = llm._get_cache()

    def ask(self, question: str) -> dict:
        # Peek at cache to detect hit/miss accurately before the LLM call
        cached_resp, sim, depth = self._cache.get(
            question, session_id=self.session_id
        )
        is_hit = cached_resp is not None

        t0     = time.perf_counter()
        try:
            result = Settings.llm.complete(question, session_id=self.session_id)
        except Exception as exc:
            # Match by class name so we don't hard-import openai / anthropic
            # (they're optional deps here). Any real AuthenticationError from
            # either provider surfaces with that exact class name.
            name = type(exc).__name__
            if name == "AuthenticationError" and not _key_state["rejected"]:
                provider = "OpenAI" if _has_openai else "Anthropic"
                url      = ("https://platform.openai.com/api-keys"
                            if _has_openai
                            else "https://console.anthropic.com/settings/keys")
                print()
                print(f"⚠  {provider} rejected the API key (HTTP 401).")
                print(f"   Verify your key at {url}")
                print("   Falling back to mock LLM for the rest of this demo.\n")
                _key_state["rejected"] = True
                # Rebuild Settings.llm around the mock inner — keeps the
                # same SulciCacheLLM instance's cache/session state intact
                # for readers who inspect it after this point.
                Settings.llm = SulciCacheLLM(
                    llm            = _MockLLM(),
                    backend        = "sqlite",
                    db_path        = llm._sulci_kwargs.get("db_path", ""),
                    threshold      = 0.90,
                    context_window = 4,
                    query_weight   = 0.70,
                    context_decay  = 0.60,
                    session_ttl    = 3600,
                )
                # Update our local reference so self._cache still tracks
                # the active cache instance.
                self._cache = Settings.llm._get_cache()
                result = Settings.llm.complete(question, session_id=self.session_id)
            else:
                raise
        ms     = (time.perf_counter() - t0) * 1000

        if is_hit:
            _stats["hits"] += 1
        else:
            _stats["misses"] += 1
            self._cache.set(question, result.text, session_id=self.session_id)

        entry = {
            "question":      question,
            "answer":        result.text,
            "source":        "cache" if is_hit else "llm",
            "latency_ms":    round(ms, 1),
            "similarity":    round(sim, 4) if is_hit else 0.0,
            "context_depth": depth,
            "cache_hit":     is_hit,
        }
        self.history.append(entry)
        return entry

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
    print(f"{icon} [{r['source'].upper()}] {r['latency_ms']:.0f}ms{sim}{ctx} — {label}")


# ── Demo ──────────────────────────────────────────────────────────────────────
def main():
    print("◈ Sulci Cache + LlamaIndex — context-aware semantic cache demo\n")

    # ── Round 1: fresh questions — all cache misses ───────────────────────────
    # Each question in its own session so context doesn't bleed across topics.
    print("Round 1: Fresh questions — all cache misses")
    print("-" * 50)
    fresh = [
        ("session-caching",    "What is semantic caching?"),
        ("session-llamaindex", "How does LlamaIndex work?"),
        ("session-cap",        "What is the CAP theorem?"),
    ]
    sessions = {}
    for sid, q in fresh:
        chat = Chat(session_id=sid)
        sessions[sid] = chat
        r = chat.ask(q)
        print_result(q, r)

    # ── Round 2: paraphrases in same sessions — cache hits expected ───────────
    # Reusing the same session_id means prior context boosts similarity
    # for semantically equivalent follow-up questions.
    print("\nRound 2: Paraphrased questions in same sessions — hits expected")
    print("-" * 50)
    paraphrases = [
        ("session-caching",    "How does semantic cache work?"),   # ≈ Round 1 q1
        ("session-llamaindex", "What is LlamaIndex used for?"),    # ≈ Round 1 q2
        ("session-cap",        "Explain the CAP theorem"),         # ≈ Round 1 q3
    ]
    for sid, q in paraphrases:
        r = sessions[sid].ask(q)
        print_result(q, r)

    # ── Round 3: context-aware follow-ups — single topic session ─────────────
    # Ambiguous follow-ups resolve correctly because session context anchors
    # them to the ongoing topic — this is Sulci Cache's key differentiator.
    print("\nRound 3: Context-aware follow-ups — single topic session")
    print("-" * 50)
    ctx_chat = Chat(session_id="session-context-demo")
    seed_q   = "What is semantic caching?"
    r = ctx_chat.ask(seed_q)
    print_result(f"[seed]  {seed_q}", r)

    for q in [
        "Can you give me a simple example?",            # context → semantic caching
        "How is this different from standard caching?",
        "What are the main benefits?",
    ]:
        r = ctx_chat.ask(q)
        print_result(q, r)

    # ── Round 4: clearly unrelated — cache miss ───────────────────────────────
    print("\nRound 4: Clearly unrelated question — cache miss")
    print("-" * 50)
    r = Chat(session_id="session-misc").ask("What is the capital of Australia?")
    print_result("What is the capital of Australia?", r)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print()
    total = _stats["hits"] + _stats["misses"]
    rate  = _stats["hits"] / total if total else 0.0
    print("=" * 50)
    print(f"Total queries  : {total}")
    print(f"Cache hits     : {_stats['hits']}  ({rate:.0%} hit rate)")
    print(f"LLM calls      : {_stats['misses']}")
    print(f"Cost saved     : ${_stats['hits'] * 0.005:.4f}  (est. at $0.005/call)")
    print(f"Active sessions: {llm._get_cache().stats().get('active_sessions', 'N/A')}")
    print("=" * 50)

    ctx_chat.print_history()


if __name__ == "__main__":
    main()
