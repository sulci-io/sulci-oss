"""
examples/agent_example_crewai.py
=================================
Sulci + CrewAI — agent cost-saving demo.

One of several "agent_example_*" flavors. Each demonstrates Sulci
wrapping a different agent framework's LLM dispatch. The integration
pattern is the same — `set_llm_cache(SulciCache(...))` for any
framework that rides on LangChain, or a thin BaseLLM subclass that
routes through `cache.cached_call(query, llm_fn)` for frameworks that
don't, like CrewAI.

Companion examples:
    agent_example_langgraph.py    ← LangChain global cache pattern
    agent_example_crewai.py       ← you are here  (BaseLLM subclass pattern)
    agent_example_autogen.py      ← (planned)

Demonstrates Sulci wrapping a CrewAI Crew with 2 agents (researcher,
writer) collaborating on the same task across 3 runs. Watch the
cache hit rate climb as Sulci dedupes the Crew's structurally
repetitive planner/reasoner LLM calls:

    Run 1 (cold cache):    ~5-20% hit rate — Crew's first pass
    Run 2 (warm cache):   ~70-90% hit rate — same Crew, same task
    Run 3 (hot cache):    ~85-95% hit rate — near-full dedupe

CrewAI is the second framework Sulci demonstrates against and it's
structurally different from LangGraph: CrewAI doesn't ride on
LangChain's global cache, so `set_llm_cache()` won't intercept its
LLM calls. Instead we subclass CrewAI's `BaseLLM` to inject Sulci
between the Crew and the underlying LLM transport. This same pattern
works for any agent framework with a pluggable LLM interface
(AutoGen, LlamaIndex agents, hand-rolled ReAct, etc.).

Why this matters in production: a single CrewAI task typically fires
10-40 LLM calls (each agent's planning + tool-call decisions + the
delegation messages between agents). At $0.003-0.015/call against
frontier models, a multi-agent workflow run by 100 concurrent users
can rack up $30-200/hour. With 80%+ semantic dedupe on the structural
calls (planners, role prompts, delegation handshakes), that drops
4-5×. The agent overhead disappears; only the semantically-novel
reasoning steps still cost real money.

Requirements:
    pip install "sulci[sqlite]" crewai

    export ANTHROPIC_API_KEY=sk-ant-...
    # No key set → mock LLM used automatically; the demo runs to completion
    # but the timing numbers won't reflect real Claude latency.

Run:
    python examples/agent_example_crewai.py

What this demonstrates:
    1. Sulci wrapped as a CrewAI `BaseLLM` subclass — `CachedLLM`.
       Every LLM call the Crew dispatches (across both agents, across
       all tasks) goes through Sulci's `cached_call()`.

    2. Context-aware caching with context_window=4 + a per-Crew
       session_id. Each run of the same Crew shares context so that
       the second agent's prompt-conditioning hits the cache against
       the first agent's outputs from the prior run.

    3. Cumulative cost savings via `cache.stats()` — saved_cost is
       populated because the Cache below is constructed with
       `cost_per_call=COST_PER_CALL` (issue #88: an unconfigured Cache
       falls back to sulci's own default of $0.005, which is NOT what the
       per-run arithmetic in this file uses).

       ⚠️ This paragraph used to claim the value was passed to
       `cached_call()`. It was not passed anywhere. The constructor took
       sulci's $0.005 default while the print statements computed
       `hits * 0.003`, so one run printed BOTH "Aggregate cost saved
       $0.012" and "saved_cost=$0.020" for the same four hits, eleven
       lines apart — while the docstring said the discrepancy had been
       sidestepped. Fixed 2026-08-11: one constant, used by both.

    4. The CachedLLM pattern works for any non-LangChain agent
       framework. AutoGen, LlamaIndex agents, hand-rolled ReAct
       loops, custom orchestrators — same recipe, swap the framework.

The Crew itself is intentionally small — 2 agents, 1 sequential task
chain — so the focus stays on cache behavior, not on prompt
engineering or tool-use mechanics. Real production Crews have more
agents, tool-using nodes, and reflection steps; the cache savings
scale proportionally because more structural calls = more dedupe.
"""
import os, sys, time, tempfile, hashlib
from typing import Any
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Suppress CrewAI's first-run telemetry opt-in dialog, the background
# trace-upload connection, AND the per-task "view execution traces?" prompt.
# In CrewAI 1.14+, the post-kickoff prompt blocks for ~20s on stdin even
# after CREWAI_TRACING_ENABLED=false; CREWAI_DISABLE_TELEMETRY=true is
# the canonical full kill switch. The OTEL var disables the upstream
# OpenTelemetry exporter. Users who want CrewAI's tracing can re-enable
# any of these in their own code.
os.environ.setdefault("CREWAI_TRACING_ENABLED",      "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY",    "true")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT",    "true")
os.environ.setdefault("OTEL_SDK_DISABLED",           "true")

# Belt-and-suspenders: redirect stdin to /dev/null so if any interactive
# prompt slips through env-var suppression, it cannot block on TTY input.
# This is a safety net for CrewAI version drift — newer versions may add
# new opt-in prompts we don't know about yet.
try:
    sys.stdin = open(os.devnull, "r")
except OSError:
    pass

# ── Optional-dependency guards: friendly errors instead of stack traces ──────
try:
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.llms.base_llm import BaseLLM
except ImportError:
    sys.exit(
        "crewai is required for this example.\n"
        '  pip install "sulci[sqlite]" crewai\n'
    )

from sulci import Cache


# ── Cache configuration ──────────────────────────────────────────────────────
# Per-run tempdir so the example is idempotent — re-running this script does
# NOT pollute prior demo state. Pattern adopted in v0.5.4.
_DB_PATH = os.path.join(tempfile.mkdtemp(prefix="sulci_crewai_"), "cache")

# ONE source for the per-call price. The Cache uses it for saved_cost and the
# summary arithmetic below uses it for its own totals; if these ever diverge
# again the two printed dollar figures will disagree, as they did until
# 2026-08-11.
COST_PER_CALL = 0.003

sulci_cache = Cache(
    backend          = "sqlite",
    db_path          = _DB_PATH,
    threshold        = 0.85,
    context_window   = 4,        # context-aware blending across the Crew run
    query_weight     = 0.70,
    ttl_seconds      = 3600,
    cost_per_call    = COST_PER_CALL,   # issue #88 — see module docstring
)


# ── CachedLLM: BaseLLM subclass that routes through Sulci ────────────────────
# Pattern: wrap an inner LLM (real CrewAI LLM with Anthropic, or a mock for
# the no-API-key path) with a thin BaseLLM subclass that funnels every call
# through Sulci's cache.get() / cache.set(). On a miss, the inner LLM does
# the real work; on a hit, Sulci returns the cached response in ~0.7ms.
#
# This is the pattern for any agent framework that DOESN'T ride on LangChain.
# Subclass the framework's LLM base class, intercept call(), route through
# Sulci. One small class, zero changes to the rest of the agent code.

class CachedLLM(BaseLLM):
    """CrewAI BaseLLM that wraps an inner LLM with Sulci's semantic cache."""

    def __init__(self, inner_llm: Any, cache: Cache, session_id: str = "crew", **kwargs):
        # BaseLLM requires `model` — inherit from the inner LLM if present.
        model_name = getattr(inner_llm, "model", "sulci-wrapped")
        super().__init__(model=model_name, **kwargs)
        # Bypass Pydantic's field-only validation for these instance attrs.
        object.__setattr__(self, "_inner",      inner_llm)
        object.__setattr__(self, "_cache",      cache)
        object.__setattr__(self, "_session_id", session_id)
        object.__setattr__(self, "_run_hits",   0)
        object.__setattr__(self, "_run_misses", 0)

    def reset_run_counters(self):
        object.__setattr__(self, "_run_hits",   0)
        object.__setattr__(self, "_run_misses", 0)

    @staticmethod
    def _normalize(messages):
        """Flatten CrewAI's messages format to a single prompt string."""
        if isinstance(messages, str):
            return messages
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role    = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, str):
                    parts.append(f"[{role}] {content}")
            else:
                parts.append(str(m))
        return "\n".join(parts)

    def call(self, messages, **kwargs) -> str:
        """Cache-aware LLM dispatch. Inner LLM only fires on cache miss."""
        prompt = self._normalize(messages)

        # Check cache first. We use get()/set() directly (not cached_call())
        # so we can count per-run hits/misses for the demo's printed stats.
        resp, similarity, depth = self._cache.get(prompt, session_id=self._session_id)
        if resp is not None:
            object.__setattr__(self, "_run_hits", self._run_hits + 1)
            return resp

        # Miss → delegate to inner LLM and store result
        object.__setattr__(self, "_run_misses", self._run_misses + 1)
        response = self._inner.call(messages, **kwargs)
        if isinstance(response, str):
            self._cache.set(prompt, response, session_id=self._session_id)
        return response

    def supports_function_calling(self) -> bool:
        return getattr(self._inner, "supports_function_calling", lambda: False)()

    def supports_stop_words(self) -> bool:
        return getattr(self._inner, "supports_stop_words", lambda: True)()

    def get_context_window_size(self) -> int:
        return getattr(self._inner, "get_context_window_size", lambda: 8192)()

    @property
    def run_total(self):
        return self._run_hits + self._run_misses

    @property
    def run_hit_rate(self):
        return self._run_hits / self.run_total if self.run_total else 0.0


# ── Inner LLM setup: real CrewAI LLM if ANTHROPIC_API_KEY set, else mock ─────
_has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

if _has_anthropic:
    inner_llm  = LLM(model="anthropic/claude-haiku-4-5-20251001", temperature=0)
    _llm_label = "Anthropic claude-haiku-4-5-20251001 via CrewAI LLM (real API)"
else:
    # Deterministic mock — returns a hash-based stub response. Lets the demo
    # run end-to-end without an API key (CI, sandbox, quick iteration). The
    # 50ms sleep keeps the cache-vs-LLM timing contrast visible.
    class _MockInnerLLM(BaseLLM):
        def __init__(self):
            super().__init__(model="mock/echo-llm")

        def call(self, messages, **kwargs) -> str:
            time.sleep(0.05)
            if isinstance(messages, str):
                prompt = messages
            else:
                prompt = "\n".join(
                    m.get("content", "") for m in messages if isinstance(m, dict)
                )
            h = hashlib.md5(prompt.encode()).hexdigest()[:6]
            # CrewAI's executor expects substantive responses; keep it short
            # but multi-sentence so the writer agent has something to compress.
            return (f"[Mock {h}] Three key points: "
                    f"first, semantic caching dedupes LLM calls by meaning. "
                    f"Second, it cuts inference costs significantly. "
                    f"Third, context-aware blending makes it safe for agents.")

        def supports_function_calling(self): return False
        def supports_stop_words(self):       return True
        def get_context_window_size(self):    return 8192

    inner_llm  = _MockInnerLLM()
    _llm_label = "mock LLM (no ANTHROPIC_API_KEY set — set it for real call timings)"


cached_llm = CachedLLM(inner_llm=inner_llm, cache=sulci_cache, session_id="crew-demo-task-1")


# ── CrewAI agents + task ─────────────────────────────────────────────────────
# A small two-agent Crew: a researcher gathers points about a topic and hands
# them to a writer for a concise summary. Sequential process, same Crew run
# 3 times so the structural prompts (role context, task framing, delegation
# messages) cache cleanly.

def build_crew(task_topic: str) -> Crew:
    researcher = Agent(
        role            = "Research Analyst",
        goal            = f"Identify the 3 most important points about: {task_topic}",
        backstory       = "You are a careful research analyst who values brevity and accuracy. "
                          "Your answers are always structured as numbered lists.",
        llm             = cached_llm,
        verbose         = False,
        allow_delegation= False,
    )
    writer = Agent(
        role            = "Technical Writer",
        goal            = "Distill research findings into a tight 2-sentence summary",
        backstory       = "You are an experienced technical writer. You compress dense research "
                          "into clear, jargon-free prose without losing the key facts.",
        llm             = cached_llm,
        verbose         = False,
        allow_delegation= False,
    )
    research_task = Task(
        description     = f"Research the topic: {task_topic}. "
                          f"Identify the 3 most important points. Return them as a numbered list, no preamble.",
        expected_output = "A numbered list of 3 key points, one per line.",
        agent           = researcher,
    )
    write_task = Task(
        description     = "Take the researcher's 3 points and condense them into a 2-sentence summary. "
                          "No bullet points, no preamble, just the summary prose.",
        expected_output = "Two sentences summarizing the research.",
        agent           = writer,
        context         = [research_task],   # writer sees researcher's output
    )
    return Crew(
        agents          = [researcher, writer],
        tasks           = [research_task, write_task],
        process         = Process.sequential,
        verbose         = False,
        tracing         = False,    # explicit kill switch — also covered by env vars
    )


# ── Demo runner ──────────────────────────────────────────────────────────────
def run_task(task_topic: str, run_num: int) -> dict:
    """Run the Crew once; print per-run cache stats."""
    print(f"\n{'━' * 64}")
    print(f" Run {run_num} of 3 — {'cold' if run_num == 1 else 'warm' if run_num == 2 else 'hot'} cache")
    print(f" Topic: {task_topic}")
    print(f"{'━' * 64}")

    cached_llm.reset_run_counters()
    t0 = time.perf_counter()

    crew   = build_crew(task_topic)
    result = crew.kickoff()

    elapsed = time.perf_counter() - t0
    hits, misses = cached_llm._run_hits, cached_llm._run_misses

    print(f"\n  ── Run {run_num} stats ──────────────────────────")
    print(f"   LLM calls / cache lookups : {hits + misses}")
    print(f"   Cache hits                : {hits}")
    print(f"   Cache misses (LLM calls)  : {misses}")
    print(f"   Hit rate                  : {cached_llm.run_hit_rate:.0%}")
    print(f"   Wall time                 : {elapsed:.1f}s")
    print(f"   Cost saved (vs ${COST_PER_CALL}/call): ${hits * COST_PER_CALL:.3f}")

    # Truncated preview of the Crew output for visual confirmation
    output_str = str(result)[:140].replace("\n", " ")
    print(f"   Crew output (preview)     : {output_str}...")

    return {"hits": hits, "misses": misses, "elapsed": elapsed}


def main():
    print("◈ Sulci + CrewAI — agent cost-saving demo")
    print("─" * 64)
    print(f"  LLM   : {_llm_label}")
    print(f"  Cache : SQLite + context-aware (window=4, threshold=0.85)")
    print(f"  DB    : {_DB_PATH}")
    print()
    print("  3 runs of the same Crew task — watch the hit rate climb.")
    # Measured 2026-08-11 on the mock path: 2 calls per run, not 6-12. The
    # old copy asserted a range the example never produced. Printed from the
    # actual counter below rather than restated as prose.
    print("  Each Crew run dispatches 2 agents; the call count is printed per run.")

    TASK = "semantic caching for LLM applications"

    runs = [run_task(TASK, n) for n in (1, 2, 3)]

    # ── Summary ────────────────────────────────────────────────────────────
    total_calls = sum(r["hits"] + r["misses"] for r in runs)
    total_hits  = sum(r["hits"] for r in runs)
    total_miss  = sum(r["misses"] for r in runs)
    cold_time   = runs[0]["elapsed"]
    warm_time   = runs[2]["elapsed"]
    speedup     = cold_time / warm_time if warm_time > 0 else float("inf")

    print(f"\n{'━' * 64}")
    print(f" Summary across 3 runs")
    print(f"{'━' * 64}")
    print(f"   Total cache lookups       : {total_calls}")
    if total_calls > 0:
        print(f"   Total LLM calls (misses)  : {total_miss}  "
              f"({total_miss}/{total_calls} = {total_miss/total_calls:.0%})")
        print(f"   Total cache hits          : {total_hits}  "
              f"(would have been LLM calls without Sulci)")
        print(f"   Aggregate hit rate        : {total_hits/total_calls:.0%}")
    print(f"   Aggregate cost saved      : ${total_hits * COST_PER_CALL:.3f}")
    print(f"   Hot-run speedup vs cold   : {speedup:.1f}×")

    # Aggregate cache stats from Sulci's own counters (sanity check)
    s = sulci_cache.stats()
    print(f"\n  cache.stats() : hits={s['hits']}, misses={s['misses']}, "
          f"hit_rate={s['hit_rate']:.0%}, saved_cost=${s['saved_cost']:.3f}")


if __name__ == "__main__":
    main()
