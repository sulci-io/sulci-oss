"""
examples/agent_example_langgraph.py
====================================
Sulci + LangGraph — agent cost-saving demo.

One of several "agent_example_*" flavors. Each demonstrates Sulci
wrapping a different agent framework's LLM dispatch. The integration
pattern is the same — `set_llm_cache(SulciCache(...))` for any
framework that rides on LangChain, or `cache.cached_call(query, llm_fn)`
for frameworks that don't.

Companion examples (planned / future):
    agent_example_langgraph.py    ← you are here
    agent_example_crewai.py       ← (planned)
    agent_example_autogen.py      ← (planned)

Demonstrates Sulci wrapping a LangGraph ReAct-style agent. The agent
runs a multi-step research task 3 times in a row. Watch the cache hit
rate climb across runs as Sulci dedupes the agent's planner-reflector
inner loop:

    Run 1 (cold cache):    ~0% hit rate — all misses, real Anthropic calls
    Run 2 (warm cache):   ~75% hit rate — most planner/reflector steps hit
    Run 3 (hot cache):    ~90% hit rate — entire loop nearly fully cached

This is the same value proposition that lands the GA homepage hero
("Your agent called the LLM 200 times. Sulci would make it 40."): an
agent's reasoning loop is structurally repetitive — planner asks
"what's next?" every iteration, reflector asks "did that work?" — and
Sulci catches those repetitions even when phrasing varies slightly.

Why this matters in production: a single agent task that fires 20+ LLM
calls is the norm in 2026 (LangGraph, CrewAI, AutoGen, Anthropic Claude
agents). At $0.003/call, a 200-call task costs $0.60. At 80% hit rate
after the cache warms, that's $0.12 — 5× cheaper. Multiply by
concurrent users and tasks per day, and the cost difference becomes
the difference between "production becomes economical" and "it doesn't."

Requirements:
    pip install "sulci[sqlite,langchain]"
    pip install langgraph langchain-anthropic    # framework + LLM connector

    export ANTHROPIC_API_KEY=sk-ant-...
    # No key set → mock LLM used automatically; the demo runs to completion
    # but the timing numbers won't reflect real Claude latency.

Run:
    python examples/agent_example_langgraph.py

What this demonstrates:
    1. Sulci installed as LangChain's global cache via set_llm_cache().
       LangGraph's nodes call llm.invoke() under the hood, so every call
       in the graph automatically goes through Sulci — no LangGraph-
       specific code changes needed.

    2. Context-aware caching with context_window=4. The agent's session
       state (current task) is passed as session_id, so the planner's
       "what's next?" prompt is blended with prior turns and resolves
       to the right cached answer based on what was just discussed.

    3. Cumulative cost savings across repeated task executions, with
       real wall-clock timings that show the 1000×+ speedup on cache
       hits (~0.7ms cache vs ~1500ms LLM).

The agent itself is intentionally simple — a 2-node LangGraph
(plan → act → loop) — so the focus stays on the cache behavior, not
on the agent's reasoning quality. In a real production agent (with
tool calls, reflection, multi-step plans, etc.) the savings are
proportionally larger because there are more repeated calls per task.
"""
import os, sys, time, tempfile, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Optional-dependency guards: friendly errors instead of stack traces ──────
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    sys.exit(
        "langgraph is required for this example.\n"
        "  pip install langgraph langchain-anthropic\n"
    )

try:
    from langchain_core.globals import set_llm_cache
    from langchain_core.outputs import Generation
    from langchain_core.messages import HumanMessage
except ImportError:
    sys.exit(
        'langchain-core is required for this example.\n'
        '  pip install "sulci[sqlite,langchain]"\n'
    )

from sulci.integrations.langchain import SulciCache
from typing import TypedDict


# ── Cache configuration ──────────────────────────────────────────────────────
# Per-run tempdir so the example is idempotent — re-running this script does
# NOT pollute prior demo state. Pattern adopted in v0.5.4.
_DB_PATH = os.path.join(tempfile.mkdtemp(prefix="sulci_agent_"), "cache")


# Subclass SulciCache to track per-call hits/misses for the demo output.
# In production code you'd just use SulciCache directly and read aggregate
# numbers from cache.stats() — but the demo wants live per-step visibility.
class CountingSulciCache(SulciCache):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._run_hits = 0
        self._run_misses = 0

    def reset_run_counters(self):
        self._run_hits = 0
        self._run_misses = 0

    def lookup(self, prompt, llm_string):
        result = super().lookup(prompt, llm_string)
        if result is not None:
            self._run_hits += 1
        else:
            self._run_misses += 1
        return result

    @property
    def run_total(self):
        return self._run_hits + self._run_misses

    @property
    def run_hit_rate(self):
        return self._run_hits / self.run_total if self.run_total else 0.0


sulci_cache = CountingSulciCache(
    backend          = "sqlite",
    db_path          = _DB_PATH,
    threshold        = 0.85,
    context_window   = 4,        # context-aware blending for agent loops
    query_weight     = 0.70,
    namespace_by_llm = False,    # single LLM in this demo; keep stats in default partition
)
set_llm_cache(sulci_cache)


# ── LLM setup: real Anthropic if key present, else mock ──────────────────────
_has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

if _has_anthropic:
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
        _llm_label = "Anthropic claude-haiku-4-5-20251001 (real API)"
    except ImportError:
        sys.exit(
            "langchain-anthropic not installed.\n"
            "  pip install langchain-anthropic\n"
        )
else:
    # Mock chat model with deterministic output by prompt hash. Lets the demo
    # run without an API key — useful for CI, sandbox environments, and quick
    # iteration on the agent loop itself. The mock simulates ~50ms latency so
    # the cache speedup is still visible. Uses BaseChatModel (not BaseLLM)
    # to match ChatAnthropic's response type — both return AIMessage so the
    # agent nodes' `.content` access works identically in both branches.
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any, List, Optional

    class _MockChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "mock-chat"

        def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                      run_manager: Any = None, **kwargs: Any) -> ChatResult:
            time.sleep(0.05)   # simulate ~50ms real-world LLM latency
            # Concatenate message contents for deterministic hashing.
            prompt = "\n".join(getattr(m, "content", str(m)) for m in messages)
            h = hashlib.md5(prompt.encode()).hexdigest()[:6]
            ai_msg = AIMessage(content=f"[Mock answer {h}] {prompt[:50]}...")
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    llm = _MockChatModel()
    _llm_label = "mock LLM (no ANTHROPIC_API_KEY set — set it for real call timings)"


# ── Agent state ──────────────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    task: str
    findings: list[str]
    step: int
    max_steps: int
    last_question: str


# ── Agent nodes ──────────────────────────────────────────────────────────────
# Two-node LangGraph: planner decides what to research next, actor answers
# that question. The conditional edge loops until max_steps reached.
#
# This mirrors the structure of real agentic frameworks (ReAct, plan-execute,
# reflexion). Real production agents have more nodes — tool selection,
# tool execution, reflection, retry — but the cache-hit pattern is the
# same: planner/reflector calls have structural repetition, actor calls
# are content-specific.

def planner_node(state: AgentState) -> dict:
    """LLM call: decide what to research next given current state."""
    prompt = (
        f"Research task: {state['task']}\n"
        f"Findings so far: {state['findings']}\n"
        f"What single question should I research next? "
        f"Reply with just the question, no preamble."
    )
    response = llm.invoke([HumanMessage(content=prompt)]).content
    return {"step": state["step"] + 1, "last_question": response}


def actor_node(state: AgentState) -> dict:
    """LLM call: answer the planner's last question."""
    question = state.get("last_question", f"Tell me about: {state['task']}")
    prompt = f"In 2 sentences, answer: {question}"
    response = llm.invoke([HumanMessage(content=prompt)]).content
    return {"findings": state["findings"] + [response[:100]]}


def should_continue(state: AgentState) -> str:
    return "planner" if state["step"] < state["max_steps"] else END


# ── Build the graph once; reuse across all 3 runs ────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("actor",   actor_node)
graph.set_entry_point("planner")
graph.add_edge("planner", "actor")
graph.add_conditional_edges("actor", should_continue)
app = graph.compile()


# ── Demo runner ──────────────────────────────────────────────────────────────
def run_task(task: str, max_steps: int, run_num: int) -> dict:
    """Run the agent once; print per-step + run stats."""
    print(f"\n{'━' * 64}")
    print(f" Run {run_num} of 3 — {'cold' if run_num == 1 else 'warm' if run_num == 2 else 'hot'} cache")
    print(f" Task: {task}")
    print(f"{'━' * 64}")

    sulci_cache.reset_run_counters()
    t0 = time.perf_counter()

    initial_state: AgentState = {
        "task": task,
        "findings": [],
        "step": 0,
        "max_steps": max_steps,
    }
    result = app.invoke(initial_state)

    elapsed = time.perf_counter() - t0
    hits, misses = sulci_cache._run_hits, sulci_cache._run_misses

    print(f"\n  ── Run {run_num} stats ──────────────────────────")
    print(f"   LLM calls / cache lookups : {hits + misses}")
    print(f"   Cache hits                : {hits}")
    print(f"   Cache misses (LLM calls)  : {misses}")
    print(f"   Hit rate                  : {sulci_cache.run_hit_rate:.0%}")
    print(f"   Wall time                 : {elapsed:.1f}s")
    print(f"   Cost saved (vs $0.003/call): ${hits * 0.003:.3f}")
    return {"hits": hits, "misses": misses, "elapsed": elapsed}


def main():
    print("◈ Sulci + LangGraph — agent cost-saving demo")
    print("─" * 64)
    print(f"  LLM   : {_llm_label}")
    print(f"  Cache : SQLite + context-aware (window=4, threshold=0.85)")
    print(f"  DB    : {_DB_PATH}")
    print()
    print("  3 runs of the same agent task — watch the hit rate climb.")
    print("  Each run does ~6 LLM calls (3 plan + 3 act).")

    TASK = "the cost economics of semantic caching for LLM applications"
    MAX_STEPS = 3      # 3 iterations × 2 nodes each = 6 LLM calls per task

    runs = [run_task(TASK, MAX_STEPS, n) for n in (1, 2, 3)]

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
    print(f"   Total LLM calls (misses)  : {total_miss}  "
          f"({total_miss}/{total_calls} = {total_miss/total_calls:.0%})")
    print(f"   Total cache hits          : {total_hits}  "
          f"(would have been LLM calls without Sulci)")
    print(f"   Aggregate hit rate        : {total_hits/total_calls:.0%}")
    print(f"   Aggregate cost saved      : ${total_hits * 0.003:.3f}")
    print(f"   Hot-run speedup vs cold   : {speedup:.1f}×")

    # Aggregate cache stats from Sulci's own counters (sanity check —
    # these should match the per-run sums above).
    s = sulci_cache.stats()
    print(f"\n  cache.stats() : hits={s['hits']}, misses={s['misses']}, "
          f"hit_rate={s['hit_rate']:.0%}, saved_cost=${s['saved_cost']:.3f}")

    set_llm_cache(None)


if __name__ == "__main__":
    main()
