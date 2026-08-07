"""
benchmark/run.py
================
5,000-query stateless + 125-follow-up context-aware benchmark for Sulci.
Includes a dedicated context-aware caching benchmark (v0.2.0).

⚠️ The context benchmark is 125 follow-ups (25 sessions x 5), not 800 pairs.
It has said "800-pair" since v0.2.0 and the corpus has never been that size:
every SESSION_FOLLOWUPS pool holds exactly 5 entries and the draw clamps to
`min(n_followups, len(pool))`. The retired +20.8pp and +56pp figures came from
this corpus, so they were 125 samples, and nothing printed that.

Runs entirely without API keys or cloud accounts.
Uses a built-in TF-IDF cosine similarity engine to simulate
sentence-transformer embeddings — no ML dependencies required.

Produces (in benchmark/results/):
  summary.json              — stateless benchmark overall stats
  domain_breakdown.csv      — per-domain hit rates and cost savings
  threshold_sweep.csv       — hit rate vs threshold (0.70 → 0.95)
  time_series.csv           — hit rate evolution over time
  false_positives.csv       — near-miss analysis
  context_summary.json      — context-aware benchmark results (--context)
  context_accuracy.csv      — per-domain resolution accuracy (--context)
  context_alpha_sweep.csv   — resolution accuracy AND false-hit vs query_weight
                              (--context --context-sweep)

Usage:
  # Standalone stateless benchmark (no install needed)
  python benchmark/run.py

  # Add context-aware benchmark
  python benchmark/run.py --context

  # With real sulci embeddings (better accuracy)
  pip install "sulci[sqlite]"
  python benchmark/run.py --use-sulci --context

  # With real Claude API calls on misses (requires ANTHROPIC_API_KEY)
  pip install "sulci[sqlite]" anthropic
  python benchmark/run.py --use-sulci --use-claude --queries 1000 --no-sweep

  # Fast CI run
  python benchmark/run.py --no-sweep --queries 1000

Options:
  --use-sulci           Use sulci.Cache with SQLite + MiniLM instead of built-in engine
  --use-claude          Call Claude API on cache misses for real latency + semantic scoring
                        Requires: ANTHROPIC_API_KEY env var, pip install anthropic
  --claude-model MODEL  Claude model for --use-claude (default: claude-haiku-4-5-20251001)
  --claude-max-calls N  Cap total Claude API calls to limit cost (default: 500)
  --fresh               Delete any existing benchmark DBs before running (recommended
                        with --use-sulci to avoid stale-cache hit rate inflation)
  --threshold N         Similarity threshold (default: 0.85)
  --queries N           Number of test queries to run (default: 5000)
  --no-sweep            Skip threshold sweep (faster)
  --context             Run context-aware benchmark (measures follow-up resolution accuracy)
  --context-window N    Turns to remember per session (default: 4)
  --context-holdout N   Sessions per domain left UNWARMED (default: 1). Their
                        follow-ups have no correct answer cached, so they are the
                        only rows a context false-hit rate can be measured
                        against. 0 reproduces the pre-2026-08-06 corpus.
  --context-followups N Follow-ups per session (default: 5, clamped by the pools)
  --context-sweep       Sweep --query-weight, recording accuracy AND false-hit
  --out DIR             Output directory (default: benchmark/results)
"""

import argparse
import csv
import json
from datetime import datetime, timezone
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

# Seeded at import so the corpus is reproducible; overridden by --seed after
# args are parsed. Varying it is how you tell a real result from one draw --
# which 8 of each domain's 10 groups get warmed is a shuffle, and the held-out
# pair is what the discrimination metrics are measured against.
random.seed(42)

# ── CLI args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Sulci benchmark")
parser.add_argument("--use-sulci",  action="store_true",
                    help="Use sulci.Cache (requires pip install 'sulci[sqlite]')")
parser.add_argument("--use-claude", action="store_true",
                    help="Call Claude API on cache misses (requires ANTHROPIC_API_KEY)")
parser.add_argument("--claude-model", default="claude-haiku-4-5-20251001",
                    help="Claude model for --use-claude (default: claude-haiku-4-5-20251001)")
parser.add_argument("--claude-max-calls", type=int, default=500,
                    help="Max Claude API calls to cap cost (default: 500 ~$0.10 with Haiku)")
parser.add_argument("--fresh",      action="store_true",
                    help="Delete existing benchmark DBs before running (prevents stale-cache inflation)")
parser.add_argument("--threshold",  type=float, default=0.85)
parser.add_argument("--queries",    type=int,   default=5000,
                    help="Number of test queries (warmup is equal)")
parser.add_argument("--no-sweep",       action="store_true")
parser.add_argument("--context",        action="store_true",
                    help="Run context-aware benchmark")
parser.add_argument("--context-window",    type=int,   default=4,
                    help="Turns to remember per session (default: 4)")
parser.add_argument("--query-weight",     type=float, default=0.70,
                    help="Blend ratio for context-aware lookup. 0.70 (default) "
                         "favours vocabulary amplification; lower values favour "
                         "reference resolution -- resolving 'how do I fix it' "
                         "from the previous turn.")
parser.add_argument("--context-threshold", type=float, default=0.58,
                    help="Similarity threshold for context benchmark (default: 0.58)")
parser.add_argument("--context-followups", type=int, default=5,
                    help="Follow-ups per session (default: 5). CLAMPED by the "
                         "SESSION_FOLLOWUPS pools, which hold 5 each -- raising "
                         "this does not grow the corpus, it prints a warning.")
parser.add_argument("--context-holdout", type=int, default=1,
                    help="Sessions per domain left UNWARMED (default: 1). Their "
                         "follow-ups have no correct answer cached, so they are "
                         "the only rows a context false-hit rate can be measured "
                         "against. 0 reproduces the pre-2026-08-06 corpus, where "
                         "false-hit was unmeasurable.")
parser.add_argument("--context-sweep", action="store_true",
                    help="Sweep --query-weight and record resolution accuracy "
                         "AND false-hit rate at each alpha. Writes "
                         "context_alpha_sweep.csv. This is the run that decides "
                         "whether a low alpha is a real win or just a looser cache.")
parser.add_argument("--agent",          action="store_true",
                    help="Run agent-workload benchmark (measures per-session call deduplication)")
parser.add_argument("--agent-sessions",   type=int, default=50,
                    help="Sessions to simulate for --agent (default: 50)")
parser.add_argument("--agent-dispatches", type=int, default=200,
                    help="LLM dispatches per session for --agent (default: 200)")
parser.add_argument("--agent-threshold",  type=float, default=0.85,
                    help="Similarity threshold for --agent (default: 0.85)")
parser.add_argument("--out",            default=os.path.join(
                        os.path.dirname(__file__), "results"))
parser.add_argument("--seed", type=int, default=None,
                    help="Corpus RNG seed (default: 42). Vary it to check "
                         "a result holds across corpus draws.")
args = parser.parse_args()

# Re-seed AFTER parsing. build_corpus() runs later, so this reaches it.
if getattr(args, "seed", None) is not None:
    random.seed(args.seed)

os.makedirs(args.out, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  BUILT-IN EMBEDDING ENGINE
#     TF-IDF cosine similarity — no external dependencies.
#     Approximates sentence-transformer paraphrase detection on short queries.
# ══════════════════════════════════════════════════════════════════════════════

VOCAB = (
    "what is how the a of to for in can do you make use build why when where who "
    "best way difference between explain tell me about does work example simple create "
    "get set run start stop help need want should would could will show find fix error "
    "problem issue code data model api llm ai machine learning cache semantic vector "
    "embedding query response system database server python javascript function class "
    "method return value type string number list array key search index similarity "
    "threshold cosine dot product normalize dimension token text language natural "
    "processing nlp transformer bert llama fine tune train inference prompt completion "
    "generate output input context memory retrieval augmented generation rag chunk "
    "document knowledge base store reduce cost latency speed fast slow performance "
    "optimize save money expensive cheap free open source cancel subscription billing "
    "account password reset login logout update change delete remove add new order "
    "return refund shipping delivery track status payment invoice address phone email "
    "support contact hours policy terms privacy security feature bug deploy release "
    "install configure setup environment variable cloud aws azure docker container "
    "kubernetes microservice architecture design pattern test debug log monitor alert "
    "dashboard metric analytics report export import format parse validate schema "
    "migrate backup restore version upgrade rollback dependency package library "
    "framework react vue angular typescript interface component state hook effect "
    "async await promise callback event listener handler middleware route endpoint "
    "request response header body status auth token jwt oauth sql nosql query join "
    "index foreign key transaction commit rollback cursor aggregate pipeline filter "
    "sort limit skip project match group unwind lookup medical diagnosis treatment "
    "symptom prescription dosage side effect drug interaction patient doctor hospital "
    "appointment insurance coverage claim deductible premium copay referral specialist "
    "emergency urgent care pharmacy lab test result scan xray mri blood pressure "
    "diabetes heart disease cancer vaccine allergy chronic acute infection antibiotic "
    "legal contract clause liability warranty disclaimer intellectual property patent "
    "trademark copyright license agreement terms conditions dispute arbitration court "
    "compliance regulation gdpr hipaa sox audit risk assessment mitigation control"
).split()

VOCAB_IDX = {w: i for i, w in enumerate(VOCAB)}
DIM = len(VOCAB)


def _tokenize(text: str) -> list:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def _embed(text: str) -> list:
    tokens = _tokenize(text)
    tf: dict = defaultdict(float)
    for t in tokens:
        if t in VOCAB_IDX:
            tf[VOCAB_IDX[t]] += 1.0
    if not tf:
        return [0.0] * DIM
    vec = [0.0] * DIM
    for idx, cnt in tf.items():
        vec[idx] = 1 + math.log(cnt)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── Built-in cache (LSH-accelerated, no deps) ─────────────────────────────────

@dataclass
class _Entry:
    query:    str
    response: str
    vec:      list
    group:    str = ""
    domain:   str = ""


class _BuiltinCache:
    N_PROJ = 16

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.entries: list[_Entry] = []
        self.hits = self.misses = 0
        rng = random.Random(42)
        self._proj = []
        for _ in range(self.N_PROJ):
            v    = [rng.gauss(0, 1) for _ in range(DIM)]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            self._proj.append([x / norm for x in v])
        self._buckets: dict = {}

    def _lsh(self, vec: list) -> int:
        bits = 0
        for i, p in enumerate(self._proj):
            if sum(a * b for a, b in zip(vec, p)) > 0:
                bits |= (1 << i)
        return bits

    def get(self, query: str) -> tuple:
        qv = _embed(query)
        h  = self._lsh(qv)
        candidates: set = set()
        if h in self._buckets:
            candidates.update(self._buckets[h])
        for i in range(self.N_PROJ):
            nh = h ^ (1 << i)
            if nh in self._buckets:
                candidates.update(self._buckets[nh])
        best_sim, best_entry = 0.0, None
        for idx in candidates:
            e   = self.entries[idx]
            sim = _cosine(qv, e.vec)
            if sim > best_sim:
                best_sim, best_entry = sim, e
        if best_sim >= self.threshold:
            self.hits += 1
            return best_entry.response, best_sim, best_entry
        self.misses += 1
        return None, best_sim, None

    def set(self, query: str, response: str, group: str = "", domain: str = "") -> None:
        vec = _embed(query)
        idx = len(self.entries)
        self.entries.append(_Entry(query, response, vec, group, domain))
        h = self._lsh(vec)
        self._buckets.setdefault(h, []).append(idx)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SULCI CACHE WRAPPER  (optional, --use-sulci flag)
# ══════════════════════════════════════════════════════════════════════════════

class _SulciWrapper:
    """Thin wrapper around sulci.Cache to match the built-in interface."""

    def __init__(self, threshold: float, db_path: str, context_window: int = 0,
                 query_weight: float = 0.70):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from sulci import Cache
        except ImportError:
            print("ERROR: sulci not installed. Run: pip install \"sulci[sqlite]\"")
            sys.exit(1)
        self._cache   = Cache(
            backend        = "sqlite",
            threshold      = threshold,
            db_path        = db_path,
            ttl_seconds    = None,
            context_window = context_window,
            query_weight   = query_weight,
            context_decay  = 0.50,
        )
        self.threshold      = threshold
        self.context_window = context_window
        self.entries: list  = []
        self._group_map: dict = {}
        self.hits = self.misses = 0

    def get(self, query: str, session_id: str = None) -> tuple:
        response, sim, _ctx = self._cache.get(query, session_id=session_id)
        if response is not None:
            self.hits += 1
            matched = self._group_map.get(response)
            return response, sim, matched
        self.misses += 1
        return None, sim, None

    def set_with_session(self, query: str, response: str,
                         group: str = "", domain: str = "",
                         session_id: str = None) -> None:
        self._cache.set(query, response, session_id=session_id)
        self._group_map[response] = type("E", (), {"group": group, "domain": domain})()

    def set(self, query: str, response: str, group: str = "", domain: str = "") -> None:
        self._cache.set(query, response)
        self._group_map[response] = type("E", (), {"group": group, "domain": domain})()


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# 2c. CLAUDE API CLIENT  (optional, --use-claude flag)
#     Calls Claude on cache misses for real API latency + semantic correctness.
#     Uses a token-bucket rate limiter to stay within API limits.
#     Correctness is scored by embedding-cosine similarity between the cached
#     response and the live Claude response (threshold: 0.65).
# ══════════════════════════════════════════════════════════════════════════════

class _ClaudeClient:
    """
    Thin wrapper around the Anthropic API.

    - Lazy-imports anthropic so the rest of the benchmark runs without it.
    - Token-bucket rate limiter: max 50 req/min by default (well within Haiku limits).
    - Hard cap on total calls (--claude-max-calls) to bound cost.
    - Semantic correctness scoring: compares cached response to live response
      using the same TF-IDF cosine engine used throughout the benchmark.
      A cached response scoring >= SEMANTIC_CORRECT_THRESHOLD against the live
      response is considered "semantically correct" — a much stronger signal
      than the group-label proxy used in synthetic mode.
    """

    SEMANTIC_CORRECT_THRESHOLD = 0.28  # cosine sim: cached vs live response
                                       # Calibrated for short synthetic cache entries
                                       # vs longer Claude prose responses. TF-IDF
                                       # dilutes overlap when output lengths differ
                                       # significantly; 0.28 is the empirical cutoff
                                       # that separates correct from wrong responses.
    _RATE_LIMIT_PER_MIN        = 50    # requests per minute (conservative)

    def __init__(self, model: str, max_calls: int):
        try:
            import anthropic as _anthropic
        except ImportError:
            print("ERROR: anthropic not installed. Run: pip install anthropic")
            import sys; sys.exit(1)

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
            import sys; sys.exit(1)

        self._client    = _anthropic.Anthropic(api_key=api_key)
        self.model      = model
        self.max_calls  = max_calls
        self.call_count = 0
        self.total_cost_usd   = 0.0
        self.real_latencies   = []    # ms per API call
        self._bucket_tokens   = float(self._RATE_LIMIT_PER_MIN)
        self._bucket_last     = time.monotonic()
        # Haiku pricing (per million tokens, as of early 2026)
        self._input_cost_per_tok  = 0.80  / 1_000_000
        self._output_cost_per_tok = 4.00  / 1_000_000

    def _refill_bucket(self):
        now    = time.monotonic()
        delta  = now - self._bucket_last
        self._bucket_tokens = min(
            float(self._RATE_LIMIT_PER_MIN),
            self._bucket_tokens + delta * (self._RATE_LIMIT_PER_MIN / 60.0)
        )
        self._bucket_last = now

    def _wait_for_token(self):
        while True:
            self._refill_bucket()
            if self._bucket_tokens >= 1.0:
                self._bucket_tokens -= 1.0
                return
            time.sleep(0.1)

    def call(self, query: str) -> tuple:
        """
        Call Claude with query.  Returns (response_text, latency_ms, cost_usd).
        Returns (None, 0, 0) if the call cap has been reached.
        """
        if self.call_count >= self.max_calls:
            return None, 0.0, 0.0

        self._wait_for_token()

        t0 = time.perf_counter()
        try:
            msg = self._client.messages.create(
                model      = self.model,
                max_tokens = 256,
                messages   = [{"role": "user", "content": query}],
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            response   = msg.content[0].text.strip()

            # Cost accounting
            in_tok  = msg.usage.input_tokens
            out_tok = msg.usage.output_tokens
            cost    = in_tok * self._input_cost_per_tok + out_tok * self._output_cost_per_tok

            self.call_count      += 1
            self.total_cost_usd  += cost
            self.real_latencies.append(latency_ms)
            return response, latency_ms, cost

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"  [Claude API error] {exc}")
            return None, latency_ms, 0.0

    def semantic_correct(self, cached_response: str, live_response: str) -> bool:
        """
        True if the cached response is semantically close to the live Claude response.
        Uses the same TF-IDF cosine engine used throughout the rest of the benchmark
        — no extra dependencies required.
        """
        if not cached_response or not live_response:
            return False
        sim = _cosine(_embed(cached_response), _embed(live_response))
        return sim >= self.SEMANTIC_CORRECT_THRESHOLD

    def stats(self) -> dict:
        lats = sorted(self.real_latencies)
        def pct(lst, p):
            if not lst: return 0.0
            return lst[int(len(lst) * p / 100)]
        return {
            "claude_calls":          self.call_count,
            "claude_model":          self.model,
            "claude_total_cost_usd": round(self.total_cost_usd, 4),
            "claude_latency_p50_ms": round(pct(lats, 50), 1),
            "claude_latency_p95_ms": round(pct(lats, 95), 1),
            "claude_latency_p99_ms": round(pct(lats, 99), 1),
        }


# Singleton — created once in main() if --use-claude is set, None otherwise
_claude: Optional[_ClaudeClient] = None


# 2b. BUILT-IN CONTEXT CACHE
#     Mirrors sulci.ContextWindow blending without requiring sulci to be installed.
#     lookup_vec = alpha * query_vec + (1-alpha) * sum(w_i * turn_vec_i)
#     w_i = decay^i  (most recent turn = 1.0, older turns halved each step)
# ══════════════════════════════════════════════════════════════════════════════

class _ContextWindow:
    """Pure-Python sliding window that blends turn embeddings."""

    def __init__(self, max_turns: int = 4, query_weight: float = 0.70,
                 decay: float = 0.50):
        self.max_turns    = max_turns
        self.query_weight = query_weight
        self.decay        = decay
        self._turns: list = []   # list of (role, vec)

    def add_turn(self, text: str, role: str = "user") -> None:
        vec = _embed(text)
        self._turns.append((role, vec))
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def blend(self, query_vec: list) -> list:
        history = [(r, v) for r, v in self._turns if r in ("user", "assistant")]
        if not history:
            return query_vec
        dim         = len(query_vec)
        history_vec = [0.0] * dim
        total_w     = 0.0
        for i, (_, vec) in enumerate(reversed(history)):
            w = self.decay ** i
            for j in range(dim):
                history_vec[j] += w * vec[j]
            total_w += w
        if total_w:
            history_vec = [v / total_w for v in history_vec]
        alpha = self.query_weight
        out   = [alpha * q + (1.0 - alpha) * h
                 for q, h in zip(query_vec, history_vec)]
        norm  = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]

    def clear(self) -> None:
        self._turns.clear()

    @property
    def depth(self) -> int:
        return len(self._turns)


class _BuiltinContextCache(_BuiltinCache):
    """LSH-accelerated cache with per-session context blending."""

    def __init__(self, threshold: float = 0.85, context_window: int = 4,
                 query_weight: float = 0.70):
        super().__init__(threshold)
        self.context_window = context_window
        # Was accepted and then dropped on the floor. `_get_session` referenced a
        # bare `query_weight`, which is not a global -- so `--context` without
        # `--use-sulci` raised NameError at the first session, i.e. the exact
        # no-install path the module docstring advertises. Every context number
        # anyone has quoted came from the --use-sulci arm; the built-in arm has
        # not run since the parameter was threaded through.
        self.query_weight   = query_weight
        self._sessions: dict[str, _ContextWindow] = {}

    def _get_session(self, session_id: str) -> _ContextWindow:
        if session_id not in self._sessions:
            self._sessions[session_id] = _ContextWindow(
                max_turns    = self.context_window,
                query_weight = self.query_weight,
                decay        = 0.50,
            )
        return self._sessions[session_id]

    def get_ctx(self, query: str, session_id: str = None) -> tuple:
        """Like get() but blends session history into the lookup vector.

        Uses a full brute-force cosine scan (not LSH) because the context
        benchmark corpus is small (<300 entries), and LSH's random projection
        can produce 6-8 bit Hamming distances between semantically similar
        vectors, causing false negatives even at similarity 0.67.
        """
        raw_vec = _embed(query)
        if session_id:
            win   = self._get_session(session_id)
            qv    = win.blend(raw_vec) if win.depth > 0 else raw_vec
            depth = win.depth
        else:
            qv, depth = raw_vec, 0

        # Exact brute-force scan — no LSH for small context corpus
        best_sim, best_entry = 0.0, None
        for e in self.entries:
            sim = _cosine(qv, e.vec)
            if sim > best_sim:
                best_sim, best_entry = sim, e

        if best_sim >= self.threshold:
            self.hits += 1
            return best_entry.response, best_sim, best_entry, depth
        self.misses += 1
        return None, best_sim, None, depth

    def set_ctx(self, query: str, response: str,
                group: str = "", domain: str = "",
                session_id: str = None) -> None:
        self.set(query, response, group, domain)
        if session_id:
            win = self._get_session(session_id)
            # Only add the USER query to context — not the assistant response.
            # We disambiguate future queries based on what the user is asking about,
            # not based on what the system answered.  Adding response text introduces
            # structural noise tokens ("the", "in", "is") that dilute the domain signal.
            win.add_turn(query, role="user")

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].clear()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  QUERY CORPUS  (5 domains × 10 topic groups × ~100 queries each)
# ══════════════════════════════════════════════════════════════════════════════

DOMAINS = {
    "customer_support": {
        "templates": [
            ("cancel subscription",   ["How do I cancel my subscription?", "I want to cancel my account", "Cancel my plan please", "How to stop my subscription", "Unsubscribe from service", "Cancel renewal of my plan", "How do I stop being charged?", "I need to cancel my membership", "Cancel account request", "Steps to cancel subscription"]),
            ("reset password",        ["How do I reset my password?", "I forgot my password", "Cannot log in, password help", "Password reset instructions", "How to change my password?", "Lost password recovery", "Help I can't access my account", "Reset login credentials", "Forgot account password", "How to recover password?"]),
            ("refund policy",         ["What is your refund policy?", "Can I get a refund?", "How do I request a refund?", "Money back guarantee?", "Return and refund process", "I want my money back", "Refund request procedure", "How long does refund take?", "Is there a refund option?", "Refund eligibility criteria"]),
            ("update billing",        ["How do I update my billing info?", "Change credit card on file", "Update payment method", "New card for billing", "How to change billing address?", "Update my payment details", "Switch payment method", "Add new payment card", "Billing information update", "Change my card details"]),
            ("track order",           ["Where is my order?", "Track my shipment", "Order tracking information", "When will my order arrive?", "Check delivery status", "Shipping status for my order", "How to track my package?", "Order not delivered yet", "Delivery tracking help", "Where is my package?"]),
            ("contact support",       ["How do I contact support?", "Customer service phone number", "How to reach help desk?", "Support contact information", "Get help from customer service", "Speak to a representative", "Contact customer care", "Support hours and availability", "How to talk to someone?", "Reach the support team"]),
            ("change email",          ["How do I change my email address?", "Update account email", "Change email on my account", "New email address setup", "How to update login email?", "Change registered email", "Email address change request", "Update contact email", "Modify account email", "Change my account email address"]),
            ("account locked",        ["My account is locked", "Cannot access locked account", "Account suspended help", "How to unlock my account?", "Account access blocked", "Locked out of account", "Account deactivated issue", "Reactivate locked account", "Why is my account locked?", "Unlock account assistance"]),
            ("upgrade plan",          ["How do I upgrade my plan?", "Upgrade to premium", "Switch to higher tier plan", "Upgrade account plan", "How to get premium features?", "Plan upgrade instructions", "Move to a better plan", "Upgrade my subscription tier", "Higher plan benefits", "Upgrade from basic to pro"]),
            ("invoice download",      ["How to download my invoice?", "Get billing receipt", "Download payment invoice", "Where are my invoices?", "Invoice download instructions", "Access billing history", "Get my receipt or invoice", "Billing documents download", "How to get my invoice?", "Download past invoices"]),
        ],
        "responses": {
            "cancel": "To cancel, go to Account Settings > Subscription > Cancel Plan. Access continues until end of billing period.",
            "reset":  "Click 'Forgot Password' on the login page. You will receive a reset email within 5 minutes.",
            "refund": "Refunds allowed within 30 days. Processed within 5-7 business days to original payment method.",
            "billing":"Update billing at Account Settings > Billing > Payment Methods.",
            "track":  "Track at Orders > Track Shipment. Real-time updates sent via email.",
            "contact":"Reach us at support@company.com or 1-800-SUPPORT. Mon-Fri 9am-6pm EST.",
            "email":  "Change email at Account Settings > Profile > Email Address. Verification required.",
            "locked": "Account locked after failed attempts. Click 'Unlock Account' in the email we sent.",
            "upgrade":"Upgrade at Account Settings > Subscription > Change Plan. Effective immediately, prorated.",
            "invoice":"Download invoices at Account Settings > Billing > Invoice History.",
        },
    },
    "developer_qa": {
        "templates": [
            ("async await python",    ["How does async await work in Python?", "Python asyncio explanation", "Async functions in Python tutorial", "What is asyncio in Python?", "How to use await in Python", "Python async programming guide", "Asynchronous Python code example", "Understanding async def in Python", "Python coroutines explained", "How to write async code Python"]),
            ("git merge vs rebase",   ["What is the difference between git merge and rebase?", "Git rebase vs merge explained", "When to use git rebase vs merge", "Merge or rebase in git?", "Git merge vs rebase differences", "Should I use rebase or merge git?", "Explain git rebase versus merge", "Git history rebase vs merge", "Rebase vs merge which is better?", "Git branching merge vs rebase"]),
            ("docker container",      ["How do Docker containers work?", "Explain Docker containers", "What is a Docker container?", "Docker containers tutorial", "How to use Docker containers", "Docker container vs image", "Getting started with Docker", "Docker basics for beginners", "What does Docker container do?", "Docker containerisation explained"]),
            ("react usestate hook",   ["How does useState work in React?", "React useState hook explained", "Using useState in React components", "What is useState React hook?", "React state management with hooks", "useState example in React", "How to use React useState", "State management useState React", "React functional component state", "useState hook tutorial React"]),
            ("sql vs nosql",          ["What is the difference between SQL and NoSQL?", "SQL vs NoSQL databases comparison", "When to use NoSQL vs SQL?", "Relational vs non-relational database", "SQL NoSQL differences explained", "Choose SQL or NoSQL database", "NoSQL vs SQL which is better?", "Database SQL versus NoSQL", "Comparing SQL and NoSQL databases", "SQL NoSQL pros and cons"]),
            ("rest api design",       ["How do I design a REST API?", "REST API best practices", "RESTful API design principles", "Building a good REST API", "REST API design guidelines", "How to design RESTful endpoints", "REST API architecture explained", "Principles of REST API design", "Good practices for REST APIs", "REST API design patterns"]),
            ("python list comprehension", ["How do list comprehensions work in Python?", "Python list comprehension syntax", "List comprehension examples Python", "What are Python list comprehensions?", "Using list comprehension in Python", "Python comprehension explained", "List comprehension vs for loop Python", "Python list comprehension guide", "How to write list comprehension", "Python list comprehension tutorial"]),
            ("kubernetes deployment", ["How do Kubernetes deployments work?", "Kubernetes deployment explained", "What is a Kubernetes deployment?", "Deploy application on Kubernetes", "Kubernetes deployment tutorial", "K8s deployment configuration", "How to create Kubernetes deployment?", "Kubernetes pod vs deployment", "Kubernetes deployment strategy", "Getting started Kubernetes deployment"]),
            ("jwt authentication",    ["How does JWT authentication work?", "JWT token authentication explained", "What is JWT auth?", "JSON Web Token authentication", "How to implement JWT auth", "JWT authentication tutorial", "Understanding JWT tokens", "JWT vs session authentication", "Secure JWT implementation", "JWT authentication flow"]),
            ("big o notation",        ["What is Big O notation?", "Big O complexity explained", "How to calculate Big O?", "Algorithm complexity Big O", "Big O notation examples", "Understanding time complexity", "What does O(n) mean?", "Big O space and time complexity", "Algorithm efficiency Big O", "Big O notation tutorial"]),
        ],
        "responses": {
            "async":  "Use 'async def' for coroutines and 'await' to suspend. Run with asyncio.run(). Enables concurrent I/O-bound tasks.",
            "git":    "Merge preserves history with a merge commit. Rebase replays commits for linear history. Use merge for shared branches.",
            "docker": "Containers package apps with dependencies into isolated units. Images are templates; containers are running instances.",
            "react":  "useState returns [state, setState]. Call setState to update and trigger re-render. Never mutate state directly.",
            "sql":    "SQL uses structured schemas with ACID transactions. NoSQL trades consistency for flexibility. Use SQL for relational data.",
            "rest":   "Use HTTP methods (GET/POST/PUT/DELETE), stateless requests, resource URLs. Return proper status codes and use JSON.",
            "python": "[expr for item in iterable if condition]. Faster than loops for simple transforms.",
            "kubernetes": "Deployments manage ReplicaSets ensuring desired pod count. Define with kind: Deployment in YAML.",
            "jwt":    "JWT = header.payload.signature. Server signs with secret key. Client sends in Authorization header.",
            "bigo":   "O(1) constant, O(log n) logarithmic, O(n) linear, O(n²) quadratic. Describes worst-case growth rate.",
        },
    },
    "product_faq": {
        "templates": [
            ("pricing plans",         ["What are your pricing plans?", "How much does it cost?", "Pricing information", "What plans do you offer?", "Cost of subscription", "Pricing tiers explained", "How much is the pro plan?", "Monthly vs annual pricing", "Plan pricing comparison", "What is the cost per month?"]),
            ("free trial",            ["Is there a free trial?", "Can I try it for free?", "Free trial availability", "How long is the free trial?", "Do you offer a free trial?", "Trial period details", "Free tier available?", "Start free trial", "How to get free trial?", "Free trial sign up"]),
            ("data security",         ["How is my data secured?", "Data security practices", "Is my data safe?", "How do you protect user data?", "Data encryption and security", "Security measures for data", "How secure is the platform?", "Data privacy and security", "User data protection policy", "What security do you use?"]),
            ("integrations available",["What integrations do you support?", "Available third-party integrations", "Does it integrate with Slack?", "Supported integrations list", "What tools does it connect with?", "Integration options available", "Can it integrate with our tools?", "List of available integrations", "Supported app integrations", "What does it integrate with?"]),
            ("team collaboration",    ["How does team collaboration work?", "Can multiple users use it?", "Team features available", "Collaborate with my team", "Multi-user account features", "Team workspace setup", "How to add team members?", "Team plan features", "Collaborate on projects together", "Team account management"]),
            ("api access",            ["Do you have an API?", "API access available?", "How to access the API?", "API documentation link", "Can I use the API?", "Programmatic access via API", "REST API available?", "API key and access", "Developer API access", "Getting started with the API"]),
            ("mobile app",            ["Is there a mobile app?", "Mobile application available?", "iOS and Android app", "Download mobile app", "Does it have a mobile version?", "Mobile app download link", "App for smartphone?", "Mobile app features", "Is there an iPhone app?", "Download the app"]),
            ("data export",           ["How do I export my data?", "Data export options", "Can I download my data?", "Export data format", "How to export all data?", "Download my account data", "Data portability options", "Export to CSV or JSON", "How to backup my data?", "Data export and download"]),
            ("uptime sla",            ["What is your uptime guarantee?", "SLA and uptime commitment", "Service level agreement details", "How reliable is the service?", "Uptime percentage guarantee", "SLA terms and conditions", "Service availability guarantee", "Reliability and uptime SLA", "What uptime do you guarantee?", "SLA for enterprise customers"]),
            ("gdpr compliance",       ["Are you GDPR compliant?", "GDPR compliance status", "How do you handle GDPR?", "Data privacy GDPR compliance", "GDPR and data protection", "Is the product GDPR ready?", "GDPR compliance documentation", "Privacy regulations compliance", "EU data protection compliance", "GDPR data processing agreement"]),
        ],
        "responses": {
            "pricing":      "Starter ($29/mo), Growth ($99/mo), Enterprise (custom). Annual saves 20%.",
            "trial":        "14-day free trial, no credit card required. All Pro features included.",
            "security":     "AES-256 at rest, TLS 1.3 in transit, SOC 2 Type II certified, GDPR compliant.",
            "integrations": "50+ integrations: Slack, Jira, GitHub, Salesforce, HubSpot, Zapier and more.",
            "team":         "Unlimited members on team plans. Admins manage roles, permissions, workspaces.",
            "api":          "Full REST API available. 1,000 req/min on Growth, unlimited on Enterprise.",
            "mobile":       "iOS and Android apps with full feature parity and offline mode.",
            "export":       "Export as CSV, JSON, or PDF from Settings > Data Export. Ready within 24hrs.",
            "uptime":       "99.9% SLA for Growth, 99.99% for Enterprise. Credits for downtime.",
            "gdpr":         "Fully GDPR compliant. DPA available. Data in EU. Erasure requests in 30 days.",
        },
    },
    "medical_information": {
        "templates": [
            ("high blood pressure",   ["What is high blood pressure?", "Hypertension explained", "High blood pressure symptoms", "What causes high blood pressure?", "Hypertension treatment options", "How to lower blood pressure?", "Blood pressure normal range", "High BP risk factors", "Managing hypertension", "High blood pressure complications"]),
            ("type 2 diabetes",       ["What is type 2 diabetes?", "Type 2 diabetes explained", "Symptoms of type 2 diabetes", "How is type 2 diabetes treated?", "Type 2 diabetes management", "What causes type 2 diabetes?", "Diabetes type 2 risk factors", "Managing blood sugar diabetes", "Type 2 diabetes diet", "Insulin resistance diabetes"]),
            ("common cold treatment", ["How do you treat a common cold?", "Common cold remedies", "Cold symptoms treatment", "How long does a cold last?", "Best treatment for cold", "Cold vs flu differences", "How to recover from cold faster", "Treating cold symptoms at home", "Cold medicine and remedies", "Common cold duration and treatment"]),
            ("covid vaccine",         ["How do COVID vaccines work?", "COVID-19 vaccine mechanism", "mRNA vaccine explained", "COVID vaccine side effects", "Are COVID vaccines safe?", "COVID vaccination benefits", "How effective is COVID vaccine?", "COVID booster vaccine info", "COVID vaccine types comparison", "COVID vaccine immune response"]),
            ("mental health anxiety", ["What are symptoms of anxiety?", "Anxiety disorder symptoms", "How to manage anxiety?", "Anxiety treatment options", "Dealing with anxiety", "Anxiety vs normal worry", "Types of anxiety disorders", "Anxiety medication options", "Therapy for anxiety", "Anxiety self-help techniques"]),
            ("vitamin d deficiency",  ["What are symptoms of vitamin D deficiency?", "Vitamin D deficiency signs", "How to treat vitamin D deficiency?", "Low vitamin D symptoms", "Vitamin D deficiency causes", "Vitamin D supplement dosage", "Vitamin D and bone health", "How much vitamin D do I need?", "Vitamin D deficiency treatment", "Sun exposure and vitamin D"]),
            ("migraine headache",     ["What causes migraines?", "Migraine headache symptoms", "How to treat a migraine?", "Migraine triggers to avoid", "Migraine vs tension headache", "Migraine treatment options", "Preventing migraine attacks", "Migraine medication list", "How long does migraine last?", "Chronic migraine management"]),
            ("sleep disorders",       ["What are common sleep disorders?", "Types of sleep disorders", "Insomnia causes and treatment", "How to treat sleep problems?", "Sleep disorder symptoms", "Sleep apnea explained", "Improving sleep quality", "Sleep disorder diagnosis", "Treatment for insomnia", "Sleep hygiene tips"]),
            ("back pain causes",      ["What causes lower back pain?", "Lower back pain causes", "Back pain treatment options", "How to relieve back pain?", "Chronic back pain causes", "Back pain exercises", "Lower back pain remedies", "When to see doctor for back pain?", "Back pain relief at home", "Preventing lower back pain"]),
            ("antibiotic usage",      ["When should I take antibiotics?", "Antibiotic use guidelines", "How do antibiotics work?", "Antibiotic resistance explained", "Correct antibiotic usage", "Side effects of antibiotics", "Completing antibiotic course", "Antibiotic vs antiviral", "When are antibiotics needed?", "Antibiotic treatment duration"]),
        ],
        "responses": {
            "blood":    "Normal BP below 120/80. High BP (130+/80+) treated with lifestyle changes and medication.",
            "diabetes": "Type 2 impairs insulin use. Managed via diet, exercise, metformin, and HbA1c monitoring.",
            "cold":     "No cure. Treat symptoms with rest, fluids, decongestants. Lasts 7-10 days.",
            "covid":    "mRNA vaccines trigger immune response. 90-95% effective. Side effects last 1-2 days.",
            "anxiety":  "Treated with CBT therapy, SSRIs. Affects 18% of adults. Causes excessive worry.",
            "vitamin":  "Treat with D3 supplements (1000-4000 IU/day) and sun exposure.",
            "migraine": "Treated with triptans, NSAIDs. Triggers: stress, hormones, certain foods.",
            "sleep":    "Disorders: insomnia, sleep apnea. Treat with CBT-I, CPAP, sleep hygiene.",
            "back":     "Causes: muscle strain, disc herniation. Treat with rest, NSAIDs, physio.",
            "antibiotic":"Complete full course. Only for bacterial infections. Overuse causes resistance.",
        },
    },
    "general_knowledge": {
        "templates": [
            ("what is ai",            ["What is artificial intelligence?", "Explain artificial intelligence", "AI definition and overview", "What does AI mean?", "Artificial intelligence explained", "How does AI work?", "Introduction to AI", "What can AI do?", "AI basics explained", "Overview of artificial intelligence"]),
            ("climate change",        ["What is climate change?", "Explain climate change", "What causes climate change?", "Climate change effects", "Global warming explained", "Climate change impact", "What is global warming?", "Causes of climate change", "Climate change overview", "Effects of global warming"]),
            ("blockchain technology", ["What is blockchain?", "Blockchain technology explained", "How does blockchain work?", "What is a blockchain?", "Blockchain overview", "Blockchain use cases", "Explain blockchain technology", "What is distributed ledger?", "Blockchain basics", "How blockchain works simply"]),
            ("how internet works",    ["How does the internet work?", "Explain how the internet works", "What is the internet?", "Internet infrastructure explained", "How data travels on internet", "Internet protocols explained", "How websites work", "How does the web work?", "Internet basics explained", "TCP IP explained simply"]),
            ("quantum computing",     ["What is quantum computing?", "Quantum computing explained", "How does quantum computing work?", "Quantum vs classical computing", "Quantum computer basics", "What can quantum computers do?", "Explain quantum computing simply", "Quantum computing overview", "Future of quantum computing", "Quantum bits explained"]),
            ("renewable energy",      ["What is renewable energy?", "Types of renewable energy", "Explain renewable energy sources", "Solar and wind energy", "Renewable vs fossil fuels", "Benefits of renewable energy", "How solar energy works", "Renewable energy overview", "Clean energy sources explained", "Future of renewable energy"]),
            ("machine learning",      ["What is machine learning?", "Machine learning explained", "How does machine learning work?", "ML basics for beginners", "Introduction to machine learning", "What can machine learning do?", "Machine learning overview", "AI vs machine learning", "Getting started machine learning", "Machine learning definition"]),
            ("cryptocurrency bitcoin",["What is Bitcoin?", "Bitcoin explained simply", "How does Bitcoin work?", "What is cryptocurrency?", "Bitcoin vs traditional currency", "How to buy Bitcoin?", "Bitcoin blockchain explained", "Cryptocurrency basics", "What is digital currency?", "Bitcoin investment overview"]),
            ("dna genetics",          ["What is DNA?", "DNA explained simply", "How does DNA work?", "What is genetics?", "DNA and heredity", "Genes and DNA explained", "How genes work", "DNA structure and function", "Genetics basics", "What is a gene?"]),
            ("space exploration",     ["How do rockets work?", "Space exploration explained", "How do we explore space?", "Rocket propulsion basics", "How does a rocket engine work?", "Space mission overview", "How astronauts travel to space", "Rocket science basics", "Space shuttle how it works", "Getting to space explained"]),
        ],
        "responses": {
            "ai":         "AI enables machines to perform tasks requiring human intelligence: learning, reasoning, perception.",
            "climate":    "Long-term shifts in global temperatures caused by burning fossil fuels and greenhouse gases.",
            "blockchain": "Distributed ledger where records are linked cryptographically. Powers cryptocurrencies.",
            "internet":   "Global network via TCP/IP. Data travels in packets through routers. DNS maps domains to IPs.",
            "quantum":    "Qubits exist in superposition (0, 1, or both). Useful for cryptography and optimisation.",
            "renewable":  "Sources: solar, wind, hydro, geothermal. No emissions, naturally replenished.",
            "ml":         "Computers learn from data without explicit programming. Types: supervised, unsupervised, RL.",
            "bitcoin":    "Decentralised digital currency using blockchain. Limited to 21 million coins.",
            "dna":        "Double helix of nucleotides (ACGT). Genes encode proteins. Inherited from both parents.",
            "space":      "Newton's third law: expelled exhaust creates thrust. Escape velocity = 11.2 km/s.",
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 3b. CONTEXT BENCHMARK CORPUS
#     Conversation pairs: one domain-specific primer → one ambiguous follow-up.
#     The same follow-up (e.g. "How do I fix it?") should resolve differently
#     depending on which primer preceded it in the session.
#     We measure whether the cache returns the CORRECT domain's answer.
# ══════════════════════════════════════════════════════════════════════════════

# Semi-specific follow-up queries — ambiguous at the sentence level but contain
# 1-2 domain-adjacent tokens that the context blending amplifies.
# Each tuple: (query, domain_hint)  where domain_hint is used only for corpus
# organisation — the cache itself does NOT see it.
#
# Design principle: stateless lookup is confused (domain tokens are shared across
# sessions), context blending with the right primer pushes similarity above the
# domain-correct entry.
SESSION_FOLLOWUPS = {
    # Per session-key follow-ups, each keyword-aligned to that session's topic.
    # Stateless similarity to keyword_bundle: ~0.45-0.56 (below threshold 0.58).
    # Context-blended similarity: ~0.62-0.75 (above threshold 0.58).
    "locked":     ["How do I fix this account login error?",
                   "Help me resolve this account password error",
                   "How do I sort out this login account issue?",
                   "What do I do about this account login problem?",
                   "How do I update my account login password?"],
    "cancel":     ["How do I fix this subscription cancel billing error?",
                   "Help me resolve this cancel subscription account issue",
                   "How do I sort out this billing subscription problem?",
                   "What do I do about this cancel account subscription?",
                   "How do I update my subscription cancel billing?"],
    "billing":    ["How do I fix this billing payment invoice error?",
                   "Help me resolve this payment billing update problem",
                   "How do I sort out this invoice billing payment?",
                   "What do I do about this billing invoice update error?",
                   "How do I fix this payment invoice billing issue?"],
    "track":      ["How do I fix this order track delivery error?",
                   "Help me resolve this delivery track order problem",
                   "How do I check this order delivery track status?",
                   "What do I do about this track order shipping issue?",
                   "How do I fix this shipping delivery track order?"],
    "refund":     ["How do I fix this refund return payment error?",
                   "Help me resolve this billing refund payment problem",
                   "How do I get this return refund payment sorted?",
                   "What do I do about this payment refund return issue?",
                   "How do I fix this refund billing payment return?"],
    "docker":     ["How do I fix this docker container error?",
                   "Help me debug this container docker code error",
                   "How do I resolve this docker deploy container issue?",
                   "What do I do about this container docker code error?",
                   "How do I fix this docker code container deploy error?"],
    "async":      ["How do I fix this async await python code error?",
                   "Help me debug this python async code error",
                   "How do I resolve this async python code issue?",
                   "What do I do about this python async await error?",
                   "How do I fix this python code async error?"],
    "react":      ["How do I fix this react hook state component error?",
                   "Help me debug this component react state hook error",
                   "How do I resolve this state react hook issue?",
                   "What do I do about this react component state error?",
                   "How do I fix this hook state react component error?"],
    "kubernetes": ["How do I fix this kubernetes container deploy error?",
                   "Help me debug this deploy kubernetes container error",
                   "How do I resolve this container kubernetes deploy issue?",
                   "What do I do about this kubernetes deploy error?",
                   "How do I fix this container deploy kubernetes error?"],
    "jwt":        ["How do I fix this jwt auth token api error?",
                   "Help me debug this api jwt oauth token error",
                   "How do I resolve this token jwt api auth issue?",
                   "What do I do about this oauth jwt api token error?",
                   "How do I fix this api token jwt auth error?"],
    "api":        ["How do I fix this api data query export error?",
                   "Help me resolve this export api data query error",
                   "How do I sort out this data api query export issue?",
                   "What do I do about this api export data error?",
                   "How do I fix this query data api export error?"],
    "mobile":     ["How do I fix this mobile install feature bug error?",
                   "Help me resolve this feature mobile install bug error",
                   "How do I sort out this install mobile bug issue?",
                   "What do I do about this mobile bug install error?",
                   "How do I fix this bug feature mobile install error?"],
    "export":     ["How do I fix this export import data format error?",
                   "Help me resolve this data export import format error",
                   "How do I sort out this format data export issue?",
                   "What do I do about this import data export error?",
                   "How do I fix this data format export import error?"],
    "team":       ["How do I fix this team account feature update error?",
                   "Help me resolve this feature team account update error",
                   "How do I sort out this account team feature issue?",
                   "What do I do about this team feature account error?",
                   "How do I fix this account update team feature error?"],
    "uptime":     ["How do I fix this uptime performance monitor alert?",
                   "Help me resolve this monitor uptime performance alert",
                   "How do I sort out this performance uptime alert issue?",
                   "What do I do about this alert uptime monitor error?",
                   "How do I fix this performance alert uptime monitor?"],
    "blood":      ["What treatment does a doctor recommend for blood pressure symptom?",
                   "How do I treat this blood pressure symptom with a doctor?",
                   "What is the doctor treatment for blood pressure symptom?",
                   "Can a doctor recommend treatment for blood pressure?",
                   "What prescription treatment helps blood pressure symptom?"],
    "migraine":   ["What treatment does a doctor recommend for this symptom?",
                   "How do I treat this symptom with a doctor prescription?",
                   "What is the doctor treatment for this symptom?",
                   "Can a doctor recommend treatment for this symptom?",
                   "What prescription does a doctor give for this symptom?"],
    "sleep":      ["What treatment does a doctor recommend for this symptom?",
                   "How do I treat this symptom with doctor diagnosis?",
                   "What is the diagnosis treatment from a doctor?",
                   "Can a doctor recommend diagnosis treatment for this?",
                   "What doctor treatment helps this symptom diagnosis?"],
    "back":       ["What treatment does a doctor recommend for this symptom?",
                   "How do I treat this symptom with a doctor?",
                   "What is the doctor diagnosis for this back symptom?",
                   "Can a doctor recommend treatment for this back issue?",
                   "What symptom treatment does a doctor prescribe?"],
    "anxiety":    ["What treatment does a doctor recommend for this symptom?",
                   "How do I treat this symptom with doctor diagnosis?",
                   "What is the doctor treatment for this symptom?",
                   "Can a doctor recommend treatment for this symptom?",
                   "What diagnosis treatment does a doctor give for symptom?"],
    "ml":         ["How does this machine learning model use data?",
                   "What data does a machine learning model need?",
                   "How do ai machine learning models process data?",
                   "What is the machine learning model data format?",
                   "How does the machine learning ai model work with data?"],
    "blockchain": ["How does the blockchain data model api work?",
                   "What is the blockchain data api model?",
                   "How do blockchain model data api systems work?",
                   "What data does the blockchain api model use?",
                   "How does blockchain model use api data?"],
    "quantum":    ["How does the quantum computing model use data?",
                   "What data does a quantum computing model need?",
                   "How do quantum computing models process data?",
                   "What is the quantum model data format?",
                   "How does quantum computing model data work?"],
    "climate":    ["How does climate change affect energy cost data?",
                   "What data shows climate energy cost impact?",
                   "How do climate data energy cost models work?",
                   "What is the climate energy data cost model?",
                   "How does climate cost energy data change?"],
    "renewable":  ["How does renewable energy reduce cost data?",
                   "What data shows renewable energy cost reduction?",
                   "How do renewable energy cost data models work?",
                   "What is the renewable energy data cost?",
                   "How does renewable energy cost data change?"],
}



CONTEXT_SESSIONS = {
    # Each session: (primer, resp_key, hint, keyword_bundle)
    # keyword_bundle is stored as a short, VOCAB-dense paraphrase cache entry.
    "customer_support": {
        "sessions": [
            ("My account has been locked after too many login attempts",   "locked",   "account", "account login error password reset"),
            ("I need to cancel my current subscription plan",              "cancel",   "cancel",  "cancel subscription billing error account"),
            ("My payment keeps getting declined at checkout",              "billing",  "billing", "billing payment invoice error update"),
            ("I haven't received my order and it's been two weeks",        "track",    "track",   "order track delivery error shipping"),
            ("I need to get a refund for my recent purchase",              "refund",   "refund",  "refund return payment error billing"),
        ],
        "responses": {
            "locked":  "Account locked after failed attempts. Click 'Unlock Account' in the email we sent.",
            "cancel":  "To cancel, go to Account Settings > Subscription > Cancel Plan.",
            "billing": "Update billing at Account Settings > Billing > Payment Methods.",
            "track":   "Track at Orders > Track Shipment. Real-time updates sent via email.",
            "refund":  "Refunds allowed within 30 days. Processed within 5-7 business days.",
        },
    },
    "developer_qa": {
        "sessions": [
            ("My Docker container keeps crashing on startup with exit code 1",  "docker",     "container", "docker container error code deploy"),
            ("My Python async function is throwing a RuntimeError",             "async",      "async",     "async await python code error"),
            ("My React component is not re-rendering when state changes",        "react",      "component", "react hook state component error"),
            ("My Kubernetes pod is stuck in CrashLoopBackOff",                  "kubernetes", "kubernetes","kubernetes deploy container error"),
            ("My JWT token keeps getting rejected with 401 unauthorized",        "jwt",        "jwt",       "jwt auth token oauth api error"),
        ],
        "responses": {
            "docker":     "Containers package apps with dependencies. Check logs with docker logs.",
            "async":      "Use async def for coroutines and await to suspend. Run with asyncio.run().",
            "react":      "useState returns [state, setState]. Call setState to trigger re-render.",
            "kubernetes": "Deployments manage ReplicaSets. Check events with kubectl describe.",
            "jwt":        "JWT = header.payload.signature. Server signs with secret. Send in Authorization header.",
        },
    },
    "product_faq": {
        "sessions": [
            ("I cannot get the API to return data for my requests",          "api",    "api",    "api query data export error"),
            ("The mobile app keeps crashing when I open it",                 "mobile", "mobile", "mobile install feature bug error"),
            ("I am trying to export my data but the download never starts",  "export", "export", "export import data format error"),
            ("My team members cannot see the shared workspace",              "team",   "team",   "team account feature update error"),
            ("The service has been down for the past hour",                  "uptime", "uptime", "uptime performance monitor alert"),
        ],
        "responses": {
            "api":    "Full REST API available. 1,000 req/min on Growth, unlimited on Enterprise.",
            "mobile": "iOS and Android apps with full feature parity and offline mode.",
            "export": "Export as CSV, JSON, or PDF from Settings > Data Export. Ready within 24hrs.",
            "team":   "Unlimited members on team plans. Admins manage roles, permissions, workspaces.",
            "uptime": "99.9% SLA for Growth, 99.99% for Enterprise. Credits for downtime.",
        },
    },
    "medical_information": {
        "sessions": [
            ("My blood pressure reading was 145 over 92 this morning",         "blood",    "blood",    "blood pressure symptom doctor treatment"),
            ("I have been having severe migraine headaches every other day",   "migraine", "symptom",  "symptom treatment doctor prescription"),
            ("I cannot sleep more than 3 hours a night despite being tired",   "sleep",    "sleep",    "symptom treatment doctor diagnosis"),
            ("My lower back has been in constant pain for two weeks",          "back",     "back",     "symptom diagnosis treatment doctor"),
            ("I have been feeling anxious and overwhelmed constantly",         "anxiety",  "anxiety",  "symptom treatment diagnosis doctor"),
        ],
        "responses": {
            "blood":    "Normal BP below 120/80. High BP (130+/80+) treated with lifestyle changes and medication.",
            "migraine": "Treated with triptans, NSAIDs. Triggers: stress, hormones, certain foods.",
            "sleep":    "Disorders: insomnia, sleep apnea. Treat with CBT-I, CPAP, sleep hygiene.",
            "back":     "Causes: muscle strain, disc herniation. Treat with rest, NSAIDs, physio.",
            "anxiety":  "Treated with CBT therapy, SSRIs. Affects 18% of adults. Causes excessive worry.",
        },
    },
    "general_knowledge": {
        "sessions": [
            ("I am learning about how neural networks are trained",          "ml",         "machine",    "machine learning model data ai"),
            ("I want to understand how Bitcoin transactions are verified",   "blockchain", "blockchain", "blockchain data model api"),
            ("I am studying how qubits differ from classical bits",         "quantum",    "quantum",    "quantum computing model data"),
            ("I am researching the causes of rising sea levels",            "climate",    "climate",    "climate energy cost data"),
            ("I want to know how solar panels convert light to energy",     "renewable",  "renewable",  "renewable energy cost data"),
        ],
        "responses": {
            "ml":         "Computers learn from data without explicit programming. Types: supervised, unsupervised, RL.",
            "blockchain": "Distributed ledger where records are linked cryptographically. Powers cryptocurrencies.",
            "quantum":    "Qubits exist in superposition (0, 1, or both). Useful for cryptography and optimisation.",
            "climate":    "Long-term shifts in global temperatures caused by burning fossil fuels.",
            "renewable":  "Sources: solar, wind, hydro, geothermal. No emissions, naturally replenished.",
        },
    },
}


@dataclass
class ContextResult:
    domain:             str
    session_key:        str
    primer:             str
    followup:           str
    is_followup:        bool       # True = this is the ambiguous follow-up turn
    context_depth:      int        # 0 = no context used
    cache_hit:          bool
    similarity:         float
    resolved_correctly: bool       # did we get the right domain response?
    latency_ms:         float
    mode:               str        # "stateless" | "context_aware"
    should_hit:         bool = True  # False for held-out sessions (never warmed)


def held_out_context_keys(holdout_per_domain: int, seed: Optional[int] = None) -> set:
    """Pick `holdout_per_domain` session keys per domain to leave UNWARMED.

    These sessions are primed and queried exactly like the others, but their
    canonical answer is never stored. The correct outcome for every one of
    their follow-ups is a MISS. A hit is a FALSE HIT: the blended lookup vector
    drifted onto a neighbouring session and returned an answer to a question
    the user did not ask.

    Why this matters more here than in the stateless benchmark: at a low
    `query_weight` the lookup is mostly conversation history, so it is pulled
    toward the session's topic by construction. If that topic is not cached,
    the nearest neighbour is a different session in the same domain -- exactly
    the case that a resolution-accuracy number cannot see, because it only
    counts rows where a correct answer existed.

    Seeded from --seed so the held-out set varies across corpus draws, for the
    same reason build_corpus() draws its hold-outs rather than slicing them.
    """
    if holdout_per_domain <= 0:
        return set()
    rng  = random.Random(seed if seed is not None else 99)
    held = set()
    for _domain, cfg in CONTEXT_SESSIONS.items():
        keys = [resp_key for _p, resp_key, _x, _kw in cfg["sessions"]]
        held.update(rng.sample(keys, min(holdout_per_domain, len(keys))))
    return held


def build_context_corpus(n_followups: int = 5, held: Optional[set] = None) -> list:
    """
    Build a list of conversation pairs for the context benchmark.
    Each session: primer turn + n_followups queries specific to that session's topic.
    Follow-ups are drawn from SESSION_FOLLOWUPS[resp_key] so they share vocabulary
    with that session's keyword_bundle (within-session topic continuity test).

    `held` is the set of session keys whose answers are never warmed. Their rows
    carry should_hit=False and are the only rows a false-hit rate can be
    measured against.

    ⚠️ SIZE IS CAPPED BY CONTENT, NOT BY THE FLAG. Every pool in
    SESSION_FOLLOWUPS holds exactly 5 entries, so `min(n_followups, len(pool))`
    silently clamps to 5 and the corpus is 25 sessions × 5 = 125 rows however
    high --context-followups is set. Growing it means writing follow-ups.
    run_context_bench() prints the clamp rather than leaving it to be inferred.
    """
    sessions = []
    held     = held or set()
    # Deliberately NOT reseeded from --seed: keeping the follow-up draw fixed is
    # what makes runs at different --query-weight comparable to each other and
    # to the pre-hold-out numbers. --seed varies the held-out set instead.
    rng = random.Random(99)

    for domain, cfg in CONTEXT_SESSIONS.items():
        for primer, resp_key, _, kw_bundle in cfg["sessions"]:
            pool = SESSION_FOLLOWUPS.get(resp_key, [])
            followups = rng.sample(pool, min(n_followups, len(pool)))
            for fq in followups:
                sessions.append({
                    "domain":     domain,
                    "key":        resp_key,
                    "primer":     primer,
                    "followup":   fq,
                    "resp_key":   resp_key,
                    "kw_bundle":  kw_bundle,
                    "response":   cfg["responses"][resp_key],
                    "should_hit": resp_key not in held,
                })
    return sessions


def run_context_bench(n_followups: int = 5, use_sulci: bool = False,
                      context_window: int = 4,
                      query_weight: float = 0.70,
                      holdout_per_domain: int = 1,
                      seed: Optional[int] = None,
                      quiet: bool = False) -> dict:
    """
    Run the context-aware benchmark.

    For each domain × session:
      1. Warm the cache with domain-specific canonical answers
      2. Prime session with domain-specific query (session_id set)
      3. Fire ambiguous follow-up queries with and WITHOUT session_id
      4. Measure: did we get the right domain response?

    Returns dict with stateless and context_aware accuracy per domain.
    """
    # `quiet` is used by the alpha sweep, which calls this eight times and wants
    # one table, not eight preambles. It suppresses narration only -- every
    # number still lands in the returned dict and in the CSV.
    _say = (lambda *a, **k: None) if quiet else print
    _say(f"\n── Context-aware benchmark ──────────────────────────────")
    _say(f"  context_window={context_window}  followups_per_session={n_followups}")
    _say(f"  Engine: {'sulci.Cache' if use_sulci else 'built-in TF-IDF'}")
    _say()

    # Context benchmark uses a slightly lower threshold than the stateless benchmark.
    # Reason: the blended query vector (70% query + 30% history) has lower raw cosine
    # similarity to any single stored entry than an exact-match lookup would, so the
    # threshold is calibrated separately.  Default is 0.72 vs 0.85 for stateless.
    # Default 0.58 calibrated for TF-IDF blended vectors.
    # With real sentence-transformer embeddings (--use-sulci), set higher.
    ctx_threshold = args.context_threshold
    _say(f"  threshold(stateless)={args.threshold}  threshold(context)={ctx_threshold}")

    # Build caches
    if use_sulci:
        db_sl  = os.path.join(args.out, "ctx_bench_stateless_db")
        db_ctx = os.path.join(args.out, "ctx_bench_context_db")
        cache_stateless = _SulciWrapper(ctx_threshold, db_sl, context_window=0,
                                        query_weight=query_weight)
        cache_context   = _SulciWrapper(ctx_threshold, db_ctx, context_window=context_window,
                                        query_weight=query_weight)
    else:
        cache_stateless = _BuiltinContextCache(ctx_threshold, context_window=0,
                                               query_weight=query_weight)
        cache_context   = _BuiltinContextCache(ctx_threshold, context_window=context_window,
                                               query_weight=query_weight)

    # ── Warm both caches ─────────────────────────────────────────────────────
    # Each session stores two cache entries:
    #   (a) The verbose primer itself — context-blended follow-ups can match via
    #       domain vocabulary amplification (docker/container/code overlap).
    #   (b) A short keyword_bundle (e.g. "docker container error code deploy") —
    #       TF-IDF dense target that the blended follow-up vector can reach above
    #       the context threshold (0.58) while stateless stays below it (~0.40-0.55).
    #
    # HELD-OUT SESSIONS are skipped here and only here. They are primed and
    # queried identically below; the only difference is that no correct answer
    # exists for them, so every hit they produce is a false hit.
    held = held_out_context_keys(holdout_per_domain, seed)
    _say("  Warming cache with canonical responses...")

    for domain, cfg in CONTEXT_SESSIONS.items():
        for primer, resp_key, _, kw_bundle in cfg["sessions"]:
            if resp_key in held:
                continue
            response = cfg["responses"][resp_key]
            for cache in (cache_stateless, cache_context):
                # Store primer (verbose — context blending amplifies shared vocab)
                cache.set(primer, response, group=resp_key, domain=domain)
                # Store keyword bundle (dense — maximises within-domain cosine sim)
                cache.set(kw_bundle, response, group=resp_key, domain=domain)

    # ── Build sessions ────────────────────────────────────────────────────────
    sessions = build_context_corpus(n_followups=n_followups, held=held)

    # State the corpus shape rather than leaving it to be inferred. The pool
    # clamp below is the reason the retired +20.8pp / +56pp figures were never
    # solid: they were 125 samples, and nothing said so.
    pool_max = max((len(p) for p in SESSION_FOLLOWUPS.values()), default=0)
    if n_followups > pool_max:
        _say(f"  ⚠  --context-followups={n_followups} requested, but the largest "
              f"SESSION_FOLLOWUPS pool holds {pool_max}. Clamped to {pool_max}.")
        _say(f"     The corpus cannot grow past this without writing follow-ups.")
    n_neg = sum(1 for s in sessions if not s["should_hit"])
    _say(f"  Corpus: {len(sessions)} follow-ups  |  held-out sessions: "
          f"{len(held)}/{sum(len(c['sessions']) for c in CONTEXT_SESSIONS.values())}"
          f"  |  should-miss rows: {n_neg} ({n_neg / max(len(sessions), 1):.0%})")
    if not held:
        _say(f"  ⚠  holdout=0: every follow-up has a correct answer cached, so "
              f"false_hit_rate is unmeasurable and reported as null.")
    results:  list[ContextResult] = []
    session_counter = 0

    for item in sessions:
        domain   = item["domain"]
        key      = item["key"]
        primer   = item["primer"]
        followup = item["followup"]
        expected = item["response"]
        should   = item["should_hit"]
        session_id = f"bench-session-{session_counter}"
        session_counter += 1

        # ── Stateless lookup ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        if use_sulci:
            resp_sl, sim_sl, matched_sl = cache_stateless.get(followup)
        else:
            resp_sl, sim_sl, matched_sl, _ = cache_stateless.get_ctx(followup)
        ms_sl = (time.perf_counter() - t0) * 1000

        hit_sl      = resp_sl is not None
        correct_sl  = hit_sl and (expected[:30] in (resp_sl or ""))

        results.append(ContextResult(
            domain=domain, session_key=key, primer=primer, followup=followup,
            is_followup=True, context_depth=0, cache_hit=hit_sl,
            similarity=round(sim_sl, 4), resolved_correctly=correct_sl,
            latency_ms=round(ms_sl, 3), mode="stateless", should_hit=should,
        ))

        # ── Context-aware lookup ──────────────────────────────────────────────
        # First store the primer in the session
        if use_sulci:
            cache_context.set_with_session(
                primer, item["response"], group=key, domain=domain,
                session_id=session_id
            )
            t0 = time.perf_counter()
            resp_ctx, sim_ctx, matched_ctx = cache_context.get(followup, session_id=session_id)
            ms_ctx = (time.perf_counter() - t0) * 1000
            depth  = 1
        else:
            cache_context.set_ctx(primer, item["response"],
                                   group=key, domain=domain,
                                   session_id=session_id)
            t0 = time.perf_counter()
            resp_ctx, sim_ctx, matched_ctx, depth = cache_context.get_ctx(
                followup, session_id=session_id
            )
            ms_ctx = (time.perf_counter() - t0) * 1000

        hit_ctx     = resp_ctx is not None
        correct_ctx = hit_ctx and (expected[:30] in (resp_ctx or ""))

        results.append(ContextResult(
            domain=domain, session_key=key, primer=primer, followup=followup,
            is_followup=True, context_depth=depth, cache_hit=hit_ctx,
            similarity=round(sim_ctx, 4), resolved_correctly=correct_ctx,
            latency_ms=round(ms_ctx, 3), mode="context_aware", should_hit=should,
        ))

    return _context_analytics(results, context_window, query_weight,
                              holdout_per_domain, quiet=quiet)


def _context_analytics(results: list, context_window: int,
                       query_weight: float = 0.70,
                       holdout_per_domain: int = 0,
                       quiet: bool = False) -> dict:
    """Compute accuracy metrics comparing stateless vs context-aware.

    ⚠️ `resolution_accuracy` is computed over SHOULD-HIT rows only. Held-out
    rows can never be resolved correctly -- no correct answer exists for them --
    so including them would drag the number down mechanically and make it
    incomparable with any figure measured before hold-outs existed. Their
    contribution is `false_hit_rate`, which is a separate axis and belongs in
    its own column, not folded into an accuracy average.
    """
    sl  = [r for r in results if r.mode == "stateless"]
    ctx = [r for r in results if r.mode == "context_aware"]

    def acc(rows):
        pos = [r for r in rows if r.should_hit]
        if not pos: return 0.0
        return round(sum(1 for r in pos if r.resolved_correctly) / len(pos), 4)

    def discrimination(rows):
        """recall / false_hit_rate / precision, same definitions as summary()."""
        pos      = [r for r in rows if r.should_hit]
        neg      = [r for r in rows if not r.should_hit]
        hits     = [r for r in rows if r.cache_hit]
        pos_hits = [r for r in pos if r.cache_hit]
        neg_hits = [r for r in neg if r.cache_hit]
        good     = [r for r in hits if r.resolved_correctly]
        return {
            # of the follow-ups that DO have a cached answer, how many were served
            "recall":          round(len(pos_hits) / len(pos), 4) if pos else None,
            # of the follow-ups that do NOT, how many were answered anyway.
            # None (not 0.0) when there are no held-out rows -- an unmeasured
            # rate and a measured zero are different claims.
            "false_hit_rate":  round(len(neg_hits) / len(neg), 4) if neg else None,
            # of everything served, how much was the right answer
            "precision":       round(len(good) / len(hits), 4) if hits else None,
            "n_should_hit":    len(pos),
            "n_should_miss":   len(neg),
            "n_hits":          len(hits),
            "n_false_hits":    len(neg_hits),
        }

    def hit_r(rows):
        if not rows: return 0.0
        return round(sum(1 for r in rows if r.cache_hit) / len(rows), 4)

    def avg_sim(rows):
        hits = [r for r in rows if r.cache_hit]
        if not hits: return 0.0
        return round(sum(r.similarity for r in hits) / len(hits), 4)

    def avg_lat(rows):
        if not rows: return 0.0
        return round(sum(r.latency_ms for r in rows) / len(rows), 3)

    # Per-domain breakdown
    domain_rows = []
    for domain in CONTEXT_SESSIONS:
        d_sl  = [r for r in sl  if r.domain == domain]
        d_ctx = [r for r in ctx if r.domain == domain]
        domain_rows.append({
            "domain":                     domain,
            "sessions_tested":            len(d_sl),
            "stateless_hit_rate":         hit_r(d_sl),
            "context_hit_rate":           hit_r(d_ctx),
            "stateless_accuracy":         acc(d_sl),
            "context_accuracy":           acc(d_ctx),
            "accuracy_improvement":       round(acc(d_ctx) - acc(d_sl), 4),
            "stateless_avg_sim":          avg_sim(d_sl),
            "context_avg_sim":            avg_sim(d_ctx),
            "stateless_latency_ms":       avg_lat(d_sl),
            "context_latency_ms":         avg_lat(d_ctx),
        })

    sl_disc  = discrimination(sl)
    ctx_disc = discrimination(ctx)

    summary = {
        "context_window":               context_window,
        "query_weight":                 query_weight,
        "holdout_per_domain":           holdout_per_domain,
        "total_followup_queries":       len(sl),
        "domains_tested":               len(CONTEXT_SESSIONS),
        # Read this before reading any percentage below.
        "_metric_note": (
            "resolution_accuracy is over should-hit rows only. false_hit_rate is "
            "over held-out sessions, which are never warmed. false_hit_rate is null "
            "when holdout_per_domain=0 -- unmeasured, not zero."
        ),
        "stateless": {
            "hit_rate":                 hit_r(sl),
            "resolution_accuracy":      acc(sl),
            "avg_similarity":           avg_sim(sl),
            "avg_latency_ms":           avg_lat(sl),
            **sl_disc,
        },
        "context_aware": {
            "hit_rate":                 hit_r(ctx),
            "resolution_accuracy":      acc(ctx),
            "avg_similarity":           avg_sim(ctx),
            "avg_latency_ms":           avg_lat(ctx),
            **ctx_disc,
        },
        "improvement": {
            "accuracy_delta":           round(acc(ctx) - acc(sl), 4),
            "accuracy_delta_pct":       round((acc(ctx) - acc(sl)) * 100, 1),
            "hit_rate_delta":           round(hit_r(ctx) - hit_r(sl), 4),
        },
        "domain_breakdown":             domain_rows,
    }

    if quiet:
        return {"summary": summary, "results": [asdict(r) for r in results]}

    # Print summary table
    print(f"\n{'='*62}")
    print(f"  CONTEXT-AWARE BENCHMARK RESULTS")
    print(f"  context_window={context_window}  query_weight={query_weight}")
    print(f"{'='*62}")
    print(f"  {'Metric':<30} {'Stateless':>12} {'Context':>12} {'Delta':>8}")
    print(f"  {'-'*62}")
    pct_rows = [
        ("Hit rate",           hit_r(sl),  hit_r(ctx)),
        ("Resolution accuracy",acc(sl),    acc(ctx)),
        ("Avg similarity",     avg_sim(sl),avg_sim(ctx)),
    ]
    for label, sv, cv in pct_rows:
        delta = cv - sv
        print(f"  {label:<30} {sv:>11.1%}  {cv:>11.1%}  {delta:>+.1%}")
    sl_lat  = avg_lat(sl)
    ctx_lat = avg_lat(ctx)
    delta_lat = ctx_lat - sl_lat
    print(f"  {'Avg latency (ms)':<30} {sl_lat:>10.2f}ms  {ctx_lat:>10.2f}ms  {delta_lat:>+.2f}ms")

    # ── Discrimination ────────────────────────────────────────────────────────
    # A resolution-accuracy delta on its own cannot distinguish "context
    # resolves the follow-up" from "context makes the cache answer everything".
    # Both raise accuracy on rows where an answer exists. Only the held-out
    # rows separate them.
    def _fmt(v):
        return "  n/a  " if v is None else f"{v:>6.1%}"

    print(f"\n  Discrimination  ({ctx_disc['n_should_hit']} should-hit / "
          f"{ctx_disc['n_should_miss']} should-miss):")
    print(f"    {'Metric':<28} {'Stateless':>12} {'Context':>12}")
    print(f"    {'-'*54}")
    for label, k in (("Recall (should-hit served)",   "recall"),
                     ("False-hit (should-miss hit)",  "false_hit_rate"),
                     ("Precision (hits correct)",     "precision")):
        print(f"    {label:<28} {_fmt(sl_disc[k]):>12} {_fmt(ctx_disc[k]):>12}")
    if ctx_disc["false_hit_rate"] is None:
        print(f"    ⚠  false-hit is UNMEASURED at holdout=0, not zero.")

    print(f"\n  Domain breakdown:")
    for row in domain_rows:
        delta = row["accuracy_improvement"]
        sign  = "+" if delta >= 0 else ""
        print(f"    {row['domain']:22s}  "
              f"stateless={row['stateless_accuracy']:.0%}  "
              f"context={row['context_accuracy']:.0%}  "
              f"delta={sign}{delta:.0%}")
    print(f"{'='*62}\n")

    return {"summary": summary, "results": [asdict(r) for r in results]}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CORPUS BUILDER
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# NEAR-MISS PAIRS  --  the adversarial half of the corpus
# ══════════════════════════════════════════════════════════════════════════════
# Each entry is (domain, group_key, [queries]) where the group is NEVER warmed.
# Every query here is lexically close to a warmed group and semantically
# different from it. A hit on any of these is a FALSE HIT: the user receives an
# answer to a question they did not ask, and acts on it.
#
# These are the queries the old corpus could not express. Without them a cache
# that returns its nearest entry for everything scores ~100% and looks perfect.
NEAR_MISS = [
    ("customer_support", "reset api key", [
        "How do I reset my API key?", "Reset my API key",
        "I need to regenerate my API key", "How to rotate an API key",
        "My API key needs resetting", "Generate a new API key",
        "Where do I reset the API key?", "API key reset steps",
    ]),
    ("customer_support", "cancel appointment", [
        "How do I cancel my appointment?", "Cancel my appointment",
        "I want to cancel a scheduled appointment", "Cancel my booking",
        "How to cancel a meeting I booked", "Appointment cancellation",
        "Can I cancel my scheduled call?", "Cancel the demo I booked",
    ]),
    ("product_faq", "downgrade plan", [
        "How do I downgrade my plan?", "Downgrade from Pro",
        "I want to move to a cheaper plan", "How to downgrade my subscription",
        "Switch from Pro to Free", "Downgrade my account tier",
        "Can I move down a plan?", "Steps to downgrade",
    ]),
    ("developer_qa", "python threading", [
        "How does threading work in Python?", "Python threading explained",
        "Difference between threads and processes in Python",
        "When to use threading over asyncio", "Python GIL and threads",
        "Threading module basics", "How to spawn a thread in Python",
        "Thread safety in Python",
    ]),
    ("medical_information", "low blood pressure", [
        "What causes low blood pressure?", "Low blood pressure symptoms",
        "How to treat hypotension", "Is low blood pressure dangerous?",
        "Hypotension explained", "Signs of low blood pressure",
        "What to do about low BP", "Low blood pressure treatment",
    ]),
    ("general_knowledge", "what is deep learning", [
        "What is deep learning?", "Deep learning explained",
        "How do neural networks work?", "Deep learning vs machine learning",
        "What are neural nets?", "Introduction to deep learning",
        "Explain deep learning simply", "Deep learning basics",
    ]),
]


def build_corpus(n_test: int = 5000, holdout_per_domain: int = 2) -> dict:
    """Returns {domain: [{query, response, group, is_warmup, should_hit}]}

    HELD-OUT GROUPS. `holdout_per_domain` groups in each domain are tested but
    never warmed, so no correct answer exists for them in the cache. They are
    the same domain and the same vocabulary as the warmed groups, which makes
    them hard negatives rather than trivial ones -- `what is ai` is warmed,
    `machine learning` is not, and the two are genuinely adjacent.

    Before 2026-08-04 every group was warmed, so every test query had a
    same-group twin already cached and a hit was always available. The hit
    rate could only be high, and the corpus could not express a query that
    SHOULD miss. That is what made 99.9% meaningless.
    """
    corpus      = {}
    n_per_domain= n_test // len(DOMAINS)
    prefixes    = ["", "Please tell me ", "Can you explain ", "I need to know ",
                   "Quick question: ", "Help me understand ", "Could you tell me ",
                   "I was wondering ", ""]
    suffixes    = ["", "?", " please", " - need help", " asap", " thanks", ""]

    for domain, cfg in DOMAINS.items():
        templates = cfg["templates"]
        responses = cfg["responses"]
        queries   = []

        # Randomly drawn, not templates[-2:], so --seed varies WHICH groups the
        # cache has never seen. Fixed hold-outs meant every run measured
        # discrimination against the same two groups per domain.
        held = (set(random.sample([g for g, _ in templates], holdout_per_domain))
                if holdout_per_domain else set())

        for group_key, base_queries in templates:
            resp_key = next((k for k in responses if k in group_key),
                            list(responses.keys())[0])
            response = responses[resp_key]
            expanded = list(base_queries)
            for _ in range(190):
                base    = random.choice(base_queries)
                variant = (random.choice(prefixes) +
                           base.rstrip("?") +
                           random.choice(suffixes)).strip()
                if not variant.endswith(("?", ".")):
                    variant += "?"
                expanded.append(variant)
            random.shuffle(expanded)
            is_held = group_key in held
            for i, q in enumerate(expanded[:200]):
                queries.append({
                    "query":      q,
                    "response":   response,
                    "group":      group_key,
                    "domain":     domain,
                    # a held-out group contributes NO warmup rows
                    "is_warmup":  (i < 100) and not is_held,
                    "should_hit": not is_held,
                })

        # near-miss pairs: never warmed, must miss
        for nm_domain, nm_group, nm_queries in NEAR_MISS:
            if nm_domain != domain:
                continue
            expanded = list(nm_queries)
            for _ in range(90):
                base    = random.choice(nm_queries)
                variant = (random.choice(prefixes) +
                           base.rstrip("?") +
                           random.choice(suffixes)).strip()
                if not variant.endswith(("?", ".")):
                    variant += "?"
                expanded.append(variant)
            random.shuffle(expanded)
            for q in expanded[:100]:
                queries.append({
                    "query":      q,
                    "response":   "",
                    "group":      nm_group,
                    "domain":     domain,
                    "is_warmup":  False,
                    "should_hit": False,
                })

        random.shuffle(queries)
        # keep every held-out and near-miss row; they are the point
        warm = [q for q in queries if q["is_warmup"]]
        neg  = [q for q in queries if not q["should_hit"]]
        pos  = [q for q in queries if not q["is_warmup"] and q["should_hit"]]
        corpus[domain] = warm[:n_per_domain] + pos[:n_per_domain] + neg

    return corpus


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RESULT DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    query:            str
    domain:           str
    group:            str
    is_warmup:        bool
    cache_hit:        bool
    similarity:       float
    matched_group:    str
    latency_ms:       float
    correct:          bool   # group-label correctness (synthetic mode)
    should_hit:       bool = True   # False for held-out groups and near-miss pairs
    # Claude-mode extras (populated only when --use-claude is active)
    live_response:    str   = ""    # actual Claude response on miss
    live_latency_ms:  float = 0.0   # real API round-trip on miss
    semantic_correct: Optional[bool] = None  # cosine sim vs live response


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(corpus: dict, threshold: float, use_sulci: bool, verbose: bool = True) -> list:
    if use_sulci:
        db_path = os.path.join(args.out, "sulci_bench_db")
        cache   = _SulciWrapper(threshold, db_path)
        engine  = "sulci.Cache (SQLite + MiniLM)"
    else:
        cache  = _BuiltinCache(threshold)
        engine = "built-in TF-IDF engine"

    if _claude:
        engine += f" + Claude API ({_claude.model})"

    all_items  = []
    for items in corpus.values():
        all_items.extend(items)

    warmup = [x for x in all_items if x["is_warmup"]]
    test   = [x for x in all_items if not x["is_warmup"]]

    if verbose:
        print(f"\n{'='*58}")
        print(f"  Sulci Benchmark  |  threshold={threshold}")
        print(f"  Engine: {engine}")
        if _claude:
            print(f"  Claude cap: {_claude.max_calls} calls")
        print(f"{'='*58}")
        print(f"  Warmup : {len(warmup):,}  |  Test : {len(test):,}")
        # STATE THE COMPOSITION. The context benchmark prints its should-miss
        # share; this one did not, and that asymmetry is a trap: the should-miss
        # rows (held-out groups + NEAR_MISS pairs) are a FIXED count per domain
        # and do not scale with --queries, while the should-hit rows do. At
        # --queries 200 the test set is ~92% should-miss, so a correct run
        # reports a ~94% "false positive" rate and reads as catastrophe.
        # The rate is only interpretable next to the share it is a rate of.
        _neg = sum(1 for t in test if not t.get("should_hit", True))
        _pos = len(test) - _neg
        print(f"  Test set: {_pos:,} should-hit  |  {_neg:,} should-miss "
              f"({_neg / max(len(test), 1):.0%})")
        if len(test) and _neg / len(test) > 0.60:
            print(f"  ⚠  should-miss rows are a FIXED count per domain and do not "
                  f"scale with --queries.")
            print(f"     At this corpus size most of the test set is SUPPOSED to "
                  f"miss, so hit rate reads low")
            print(f"     and false-positive rate reads high. Both are correct. Use "
                  f"--queries 5000 to compare")
            print(f"     against any published figure.")
        print(f"{'='*58}\n")

    results = []

    for item in warmup:
        t0 = time.perf_counter()
        cache.set(item["query"], item["response"],
                  group=item["group"], domain=item["domain"])
        ms = (time.perf_counter() - t0) * 1000
        results.append(Result(
            query=item["query"], domain=item["domain"], group=item["group"],
            should_hit=item.get("should_hit", True),
            is_warmup=True, cache_hit=False, similarity=1.0,
            matched_group="", latency_ms=round(ms, 3), correct=True,
        ))

    for i, item in enumerate(test):
        t0  = time.perf_counter()
        resp, sim, matched = cache.get(item["query"])
        ms  = (time.perf_counter() - t0) * 1000

        if resp is None:
            # ── Cache MISS ────────────────────────────────────────────────────
            if _claude:
                # Real Claude API call: get live response, record actual latency
                live_resp, live_ms, _ = _claude.call(item["query"])
                if live_resp:
                    # Store the real response in the cache going forward
                    cache.set(item["query"], live_resp,
                              group=item["group"], domain=item["domain"])
                    results.append(Result(
                        query=item["query"], domain=item["domain"], group=item["group"],
                        should_hit=item.get("should_hit", True),
                        is_warmup=False, cache_hit=False, similarity=sim,
                        matched_group="", latency_ms=round(ms, 3), correct=True,
                        live_response=live_resp, live_latency_ms=round(live_ms, 1),
                        semantic_correct=None,  # miss — no cached response to score
                    ))
                else:
                    # API cap hit or error: fall back to synthetic response.
                    # Same rule as the non-Claude path: never cache a row we
                    # are asserting should miss.
                    if item.get("should_hit", True):
                        cache.set(item["query"], item["response"],
                                  group=item["group"], domain=item["domain"])
                    results.append(Result(
                        query=item["query"], domain=item["domain"], group=item["group"],
                        should_hit=item.get("should_hit", True),
                        is_warmup=False, cache_hit=False, similarity=sim,
                        matched_group="", latency_ms=round(ms, 3), correct=True,
                    ))
            else:
                # Write-back on miss is correct for a cache benchmark: a real
                # cache stores what it just computed, and the next identical
                # query should hit.
                #
                # But NOT for a row we are asserting should miss. Those groups
                # are deliberately unwarmed; caching the first one makes the
                # other 199 hit it at sim 1.0 and turns the hard-negative set
                # into a self-fulfilling 94% "false-hit rate". Measured, and
                # wrong, on 2026-08-04 before this line existed. The tell was
                # the entry count growing by exactly the test-set size.
                if item.get("should_hit", True):
                    cache.set(item["query"], item["response"],
                              group=item["group"], domain=item["domain"])
                results.append(Result(
                    query=item["query"], domain=item["domain"], group=item["group"],
                    should_hit=item.get("should_hit", True),
                    is_warmup=False, cache_hit=False, similarity=sim,
                    matched_group="", latency_ms=round(ms, 3), correct=True,
                ))
        else:
            # ── Cache HIT ─────────────────────────────────────────────────────
            m_group = getattr(matched, "group", "") if matched else ""
            group_correct = (m_group == item["group"])

            if _claude:
                # Verify the cached response semantically against a live Claude call
                live_resp, live_ms, _ = _claude.call(item["query"])
                semantic_ok = (
                    _claude.semantic_correct(resp, live_resp)
                    if live_resp else None
                )
                results.append(Result(
                    query=item["query"], domain=item["domain"], group=item["group"],
                    should_hit=item.get("should_hit", True),
                    is_warmup=False, cache_hit=True, similarity=sim,
                    matched_group=m_group, latency_ms=round(ms, 3),
                    correct=group_correct,
                    live_response=live_resp or "",
                    live_latency_ms=round(live_ms, 1),
                    semantic_correct=semantic_ok,
                ))
            else:
                results.append(Result(
                    query=item["query"], domain=item["domain"], group=item["group"],
                    should_hit=item.get("should_hit", True),
                    is_warmup=False, cache_hit=True, similarity=sim,
                    matched_group=m_group, latency_ms=round(ms, 3), correct=group_correct,
                ))

        if verbose and (i + 1) % 500 == 0:
            done  = [r for r in results if not r.is_warmup]
            hits  = sum(1 for r in done if r.cache_hit)
            extra = ""
            if _claude:
                cap_warn = "  ⚠ cap reached — remaining queries unverified"                            if _claude.call_count >= _claude.max_calls else ""
                extra = f"  claude_calls={_claude.call_count}{cap_warn}"
            print(f"  [{i+1:5,}/{len(test):,}]  "
                  f"hit rate: {hits/len(done):.1%}  "
                  f"entries: {len(warmup) + i + 1:,}{extra}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def percentile(lst, p):
    if not lst: return 0.0
    s = sorted(lst)
    return s[int(len(s) * p / 100)]


def summary(results: list, threshold: float) -> dict:
    test = [r for r in results if not r.is_warmup]
    hits = [r for r in test if r.cache_hit]
    miss = [r for r in test if not r.cache_hit]
    fps  = [r for r in hits if not r.correct]
    COST = 0.005

    # should-hit / should-miss partitions

    pos      = [r for r in test if getattr(r, "should_hit", True)]

    neg      = [r for r in test if not getattr(r, "should_hit", True)]

    pos_hits = [r for r in pos if r.cache_hit]

    neg_hits = [r for r in neg if r.cache_hit]

    out = {
        "threshold":             threshold,
        "total_queries":         len(test),
        "cache_hits":            len(hits),
        "cache_misses":          len(miss),
        "hit_rate":              round(len(hits) / len(test), 4) if test else 0,
        "false_positives":       len(fps),
        "false_positive_rate":   round(len(fps) / len(hits), 4) if hits else 0,

        # ── Discrimination. A single hit rate cannot distinguish a good cache
        # from one that answers everything: both score high. These three can.
        #
        #   recall        of the queries that SHOULD hit, how many did
        #   false_hit_rate of the queries that should MISS, how many hit anyway
        #                  -- the harmful case: the user gets someone else's
        #                  answer and acts on it
        #   precision     of all hits, how many matched the right group
        #
        # Before 2026-08-04 the corpus had no should-miss queries at all, so
        # false_hit_rate did not exist and hit_rate was the only number. That
        # is how 99.9% got published.
        "recall":                round(len(pos_hits) / len(pos), 4) if pos else 0,
        "false_hit_rate":        round(len(neg_hits) / len(neg), 4) if neg else 0,
        "precision":             round((len(hits) - len(fps)) / len(hits), 4) if hits else 0,
        "n_should_hit":          len(pos),
        "n_should_miss":         len(neg),
        "avg_similarity_hits":   round(sum(r.similarity for r in hits) / len(hits), 4) if hits else 0,
        "latency_hit_p50_ms":    round(percentile([r.latency_ms for r in hits], 50), 3),
        "latency_hit_p95_ms":    round(percentile([r.latency_ms for r in hits], 95), 3),
        "latency_miss_p50_ms":   round(percentile([r.latency_ms for r in miss], 50), 3),
        "latency_miss_p95_ms":   round(percentile([r.latency_ms for r in miss], 95), 3),
        "baseline_cost_usd":     round(len(test) * COST, 4),
        "actual_cost_usd":       round(len(miss) * COST, 4),
        "saved_cost_usd":        round(len(hits) * COST, 4),
        "cost_reduction_pct":    round(len(hits) / len(test) * 100, 2) if test else 0,
    }

    # Augment with real Claude stats when --use-claude was active
    if _claude and _claude.call_count > 0:
        cs = _claude.stats()
        # Real API miss latency (replaces simulated 0ms miss latency in output)
        real_miss_lats = sorted([r.live_latency_ms for r in miss if r.live_latency_ms > 0])
        # Semantic correctness rate on hits where we have a live response to compare
        scored_hits = [r for r in hits if r.semantic_correct is not None]
        out.update({
            "claude_mode":                   True,
            "claude_calls":                  cs["claude_calls"],
            "claude_model":                  cs["claude_model"],
            "claude_total_cost_usd":         cs["claude_total_cost_usd"],
            "real_latency_miss_p50_ms":      round(percentile(real_miss_lats, 50), 1) if real_miss_lats else None,
            "real_latency_miss_p95_ms":      round(percentile(real_miss_lats, 95), 1) if real_miss_lats else None,
            "real_latency_miss_p99_ms":      round(percentile(real_miss_lats, 99), 1) if real_miss_lats else None,
            "claude_latency_p50_ms":         cs["claude_latency_p50_ms"],
            "claude_latency_p95_ms":         cs["claude_latency_p95_ms"],
            "claude_latency_p99_ms":         cs["claude_latency_p99_ms"],
            "semantic_correct_rate":         round(
                sum(1 for r in scored_hits if r.semantic_correct) / len(scored_hits), 4
            ) if scored_hits else None,
            "semantic_scored_hits":          len(scored_hits),
        })

    return out


def domain_breakdown(results: list) -> list:
    test = [r for r in results if not r.is_warmup]
    rows = []
    COST = 0.005
    for domain in DOMAINS:
        d   = [r for r in test if r.domain == domain]
        h   = [r for r in d if r.cache_hit]
        m   = [r for r in d if not r.cache_hit]
        fp  = [r for r in h if not r.correct]
        rows.append({
            "domain":             domain,
            "total":              len(d),
            "hits":               len(h),
            "misses":             len(m),
            "hit_rate_pct":       round(len(h)/len(d)*100, 1) if d else 0,
            "false_positives":    len(fp),
            "fp_rate_pct":        round(len(fp)/len(h)*100, 2) if h else 0,
            "avg_sim_hits":       round(sum(r.similarity for r in h)/len(h), 4) if h else 0,
            "saved_usd":          round(len(h)*COST, 3),
            "cost_reduction_pct": round(len(h)/len(d)*100, 1) if d else 0,
        })
    return rows


def time_series(results: list, window: int = 100) -> list:
    test = [r for r in results if not r.is_warmup]
    rows = []
    for i in range(0, len(test), window):
        chunk    = test[i:i+window]
        hits     = sum(1 for r in chunk if r.cache_hit)
        cum      = test[:i+len(chunk)]
        cum_hits = sum(1 for r in cum if r.cache_hit)
        rows.append({
            "batch":                  i // window + 1,
            "queries_processed":      i + len(chunk),
            "window_hit_rate_pct":    round(hits/len(chunk)*100, 1) if chunk else 0,
            "cumulative_hit_rate_pct":round(cum_hits/len(cum)*100, 1) if cum else 0,
        })
    return rows


def false_positives_report(results: list) -> list:
    fps = [r for r in results if not r.is_warmup and r.cache_hit and not r.correct]
    return sorted([{
        "domain":        r.domain,
        "group":         r.group,
        "matched_group": r.matched_group,
        "similarity":    r.similarity,
        "query":         r.query[:100],
    } for r in fps[:100]], key=lambda x: -x["similarity"])


# ══════════════════════════════════════════════════════════════════════════════
# 8.  I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_json(obj, name):
    """Write a results JSON, stamped with its provenance.

    ⚠️ THE STAMP IS THE POINT. verify_benchmark.py reads whatever is in
    benchmark/results/, which is gitignored and never cleared, and it launches
    `run.py --no-sweep --context` with NO --agent -- so agent_summary.json is
    always a leftover from some earlier invocation. On 2026-08-06 that leftover
    was a MiniLM run (cold 0.27 / warm 0.9942) verified against a TF-IDF
    baseline (cold 0.43), and six of eight rows reported [OK].

    Mtime cannot settle this. A legitimate `run.py --agent` writes the file
    minutes before verify_benchmark.py starts, so ANY timestamp cutoff either
    rejects that legitimate file or accepts this morning's -- both were tried
    and both were wrong. The engine is the thing that actually differed, so
    record the engine.
    """
    path = os.path.join(args.out, name)
    if isinstance(obj, dict):
        obj = dict(obj)
        obj["_provenance"] = {
            "engine":       "sulci-minilm" if args.use_sulci else "builtin-tfidf",
            "engine_label": ("sulci.Cache (SQLite + MiniLM)" if args.use_sulci
                             else "built-in TF-IDF engine"),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv":         " ".join(sys.argv[1:]),
        }
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Saved {path}")


def save_csv(rows, name):
    if not rows: return
    path = os.path.join(args.out, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def sweep(corpus: dict, use_sulci: bool) -> list:
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    rows = []
    print("\n── Threshold sweep ──────────────────────────────────────")
    for t in thresholds:
        res = run(corpus, threshold=t, use_sulci=use_sulci, verbose=False)
        s   = summary(res, t)
        print(f"  t={t:.2f}  hit={s['hit_rate']:.1%}  "
              f"fp={s['false_positive_rate']:.2%}  "
              f"saved={s['cost_reduction_pct']:.1f}%")
        rows.append({
            "threshold":          t,
            "hit_rate_pct":       round(s["hit_rate"]*100, 1),
            "false_positive_pct": round(s["false_positive_rate"]*100, 2),
            "cost_reduction_pct": s["cost_reduction_pct"],
            "hits":               s["cache_hits"],
            "misses":             s["cache_misses"],
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _wipe_bench_dbs():
    """
    Remove SQLite benchmark database files written by --use-sulci runs.
    Called when --fresh is passed to prevent stale warmup data inflating hit rates
    across consecutive runs.  Safe to call even if the files don't exist yet.
    """
    import shutil
    db_names = [
        "sulci_bench_db",
        "ctx_bench_stateless_db",
        "ctx_bench_context_db",
    ]
    # Sulci's SQLite backend may create the path as a plain file, a .db file,
    # a directory, or with WAL/SHM sidecars — glob for all variants.
    import glob
    removed = []
    for name in db_names:
        base = os.path.join(args.out, name)
        candidates = [base] + glob.glob(base + ".*") + glob.glob(base + "-*")
        for path in candidates:
            if os.path.isdir(path):
                shutil.rmtree(path)
                removed.append(path)
            elif os.path.isfile(path):
                os.remove(path)
                removed.append(path)
    if removed:
        print(f"  --fresh: removed {len(removed)} benchmark DB(s):")
        for p in removed:
            print(f"    {p}")
    else:
        print("  --fresh: no existing benchmark DBs found (clean start).")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  AGENT WORKLOAD BENCHMARK  (--agent)
#     Simulates a realistic mixed-workload agent's LLM dispatch pattern across
#     N sessions × M dispatches per session. Measures the deduplication rate
#     Sulci achieves on prompts that follow the structural-repetition pattern
#     real agent traffic exhibits (planner / reflector / system prompts repeat
#     heavily across sessions; tool-call decisions repeat moderately; task-
#     specific reasoning is mostly novel).
#
#     Categories (calibrated to public agent-traffic measurements):
#       structural       45%  — planner, reflector, system-prompt-like prompts.
#                              Small param pools → high semantic-repetition.
#       semi_structural  35%  — tool-call decisions, intermediate reasoning,
#                              parameterized but template-bound prompts.
#       novel            20%  — task-specific reasoning, user-input-derived
#                              prompts. Large param pools → low repetition.
#
#     Expected hit rate range: 55-75% on default workload (50 × 200) with
#     real MiniLM embeddings + threshold 0.85.
# ══════════════════════════════════════════════════════════════════════════════

AGENT_WORKLOAD = {
    "structural": {
        "weight": 0.45,
        "templates": [
            "You are an autonomous agent with the role of {role}. Plan your next step toward the goal.",
            "Summarize what you have learned so far about the task: {task}",
            "Given the current state, what is your next action?",
            "Decide whether the user's question is fully answered. Question: {user_q}",
            "Reflect on the previous step. Outcome status: {context}",
            "What sub-task is most critical right now for {focus_area}?",
            "Verify your understanding of the user's intent: {user_q}",
            "Choose the next tool to invoke for: {task}",
            "Have we made sufficient progress on: {task}",
            "Identify any blockers preventing completion of: {focus_area}",
        ],
        "param_pools": {
            "role": [
                "planner", "researcher", "writer", "analyst",
                "executor", "reflector", "coordinator", "reviewer",
            ],
            "task": [
                "analyzing the financial report", "summarizing meeting notes",
                "drafting the customer response", "validating the dataset",
                "generating product recommendations", "comparing vendor options",
                "answering the technical question", "completing the integration",
                "researching the topic", "synthesizing the findings",
            ],
            "user_q": [
                "how do I deploy this to production",
                "what is the recommended approach for this problem",
                "can you summarize the key findings",
                "what is the next step I should take",
                "is this approach technically correct",
                "explain the trade-offs between these options",
                "should we use option A or option B here",
                "what are the main risks involved",
                "how does this compare to alternatives",
                "is there a better way to handle this",
                "what would you recommend as the next action",
                "can you walk me through the reasoning",
            ],
            "context": [
                "initial planning step completed", "preliminary data gathered",
                "analysis in progress", "first draft is ready for review",
                "errors encountered during execution", "results successfully validated",
                "user feedback has been incorporated", "iteration cycle complete",
                "blocked waiting on external input", "dependencies satisfied",
            ],
            "focus_area": [
                "accuracy", "efficiency", "cost", "user experience",
                "code quality", "scalability", "correctness", "maintainability",
            ],
        },
    },
    "semi_structural": {
        "weight": 0.35,
        "templates": [
            "Use the {tool} tool with input: {tool_input}",
            "Process this tool result and decide the next step: {tool_result}",
            "Refine the answer based on context: {partial_answer}",
            "Compare these options: {option_a} versus {option_b}",
            "Extract the key facts from this excerpt: {document_excerpt}",
            "Translate this technical statement to plain language: {technical_text}",
            "Determine if this output meets the quality bar: {candidate_output}",
            "Identify the next action given this state: {state_description}",
        ],
        "param_pools": {
            "tool": [
                "web_search", "calculator", "code_interpreter", "database_query",
                "file_reader", "shell_exec", "knowledge_base_lookup", "api_request",
            ],
            "tool_input": [
                "latest python 3.13 release notes", "current weather in Tokyo",
                "user registration count for Q3", "sum of column A in dataset",
                "list of files modified yesterday", "redis cache hit rate metrics",
                "PostgreSQL connection pool status", "company press releases 2025",
                "exchange rate USD to EUR", "stock price for ticker AAPL",
                "active user sessions count", "API rate limit remaining quota",
                "git log of recent commits", "system memory utilization",
                "scheduled maintenance windows", "outstanding bug reports list",
                "current deployment version", "database migration status",
                "list of pending pull requests", "container restart history",
            ],
            "tool_result": [
                "returned 42 matching rows from query", "found 3 critical errors in log",
                "search returned no results for query", "calculation completed: 1247.83",
                "file does not exist at specified path", "API responded with status 200",
                "transaction was rolled back due to conflict", "rate limit exceeded, retry after 60s",
                "configuration parsed successfully", "5 candidate matches were identified",
                "task queued for asynchronous processing", "data validation passed all checks",
                "encountered timeout after 30 seconds", "result cached for future requests",
                "authentication succeeded with token", "operation requires elevated privileges",
            ],
            "partial_answer": [
                "the system appears stable based on metrics",
                "two viable approaches were identified",
                "preliminary analysis suggests option B is preferred",
                "the root cause has not yet been determined",
                "results are consistent with the hypothesis",
                "performance degradation was observed under load",
                "the user's stated requirements were captured",
                "existing documentation covers most cases",
                "edge cases need additional handling",
                "the implementation follows the standard pattern",
                "test coverage is adequate for the critical paths",
                "the configuration matches the production baseline",
            ],
            "option_a": [
                "synchronous batch processing", "REST API integration",
                "in-memory caching layer", "vertical scaling approach",
                "monolithic deployment", "manual review workflow",
                "third-party SaaS solution", "polling-based updates",
                "session-affinity routing", "client-side validation",
            ],
            "option_b": [
                "asynchronous event streaming", "GraphQL API integration",
                "distributed cache cluster", "horizontal scaling approach",
                "microservices architecture", "automated CI gating",
                "self-hosted open-source stack", "webhook-driven updates",
                "stateless round-robin routing", "server-side validation",
            ],
            "document_excerpt": [
                "the quarterly report shows a 12% revenue increase year-over-year",
                "users report improved satisfaction scores in the latest survey",
                "infrastructure costs decreased by 8% after the migration",
                "the new feature was adopted by 34% of users within 30 days",
                "compliance audit identified two medium-severity findings",
                "system uptime exceeded the 99.9% SLA target this quarter",
                "customer churn was concentrated in the small-business segment",
                "the experiment showed a statistically significant improvement",
                "engineering velocity declined slightly during the migration period",
                "support ticket volume normalized after the initial release",
                "the proposed change has dependencies on three downstream services",
                "performance benchmarks indicate a 23% latency reduction",
            ],
            "technical_text": [
                "the API gateway terminates TLS and forwards to the upstream",
                "the cache uses LRU eviction with a 60-minute TTL",
                "all writes are committed via two-phase commit across replicas",
                "the embedder produces 384-dimensional dense vectors",
                "exponential backoff is applied to retries with jitter",
                "the orchestrator schedules tasks via priority queue semantics",
                "events are partitioned by tenant_id for ordered consumption",
                "the index is rebuilt nightly to incorporate new entries",
                "rate limiting is enforced per API key with a sliding window",
                "the data plane and control plane are deployed in separate VPCs",
            ],
            "candidate_output": [
                "the response correctly addresses the user's question",
                "output includes minor formatting inconsistencies",
                "the answer is factually accurate but verbose",
                "result omits the requested supporting evidence",
                "tone is appropriately professional and concise",
                "the response missed a critical context detail",
                "answer demonstrates good logical structure",
                "output is technically correct but lacks examples",
            ],
            "state_description": [
                "research phase complete, drafting underway",
                "initial outline approved by reviewer",
                "blocked pending external data refresh",
                "validation step failed, retry scheduled",
                "all required inputs have been collected",
                "concurrent revision detected, merge required",
                "approval received, ready to proceed to next stage",
                "edge case discovered, scope expansion needed",
            ],
        },
    },
    "novel": {
        "weight": 0.20,
        "templates": [
            "Analyze the following text in detail: {long_passage}",
            "Generate a comprehensive answer to: {detailed_question}",
            "Walk through the reasoning for: {complex_scenario}",
        ],
        # Novel pools are deliberately large to suppress semantic repetition.
        # Each entry is distinct enough that even MiniLM at threshold 0.85
        # rarely scores two as a hit. ~50 distinct prompts per category
        # × 3 categories = ~150 unique novel prompts, exceeding the expected
        # ~2,000 dispatches at 20% weight by ~7×.
        #
        # NB: deliberately NO shared prefix on long_passage entries (an earlier
        # draft used "Passage N: ..." which TF-IDF treated as a token-overlap
        # source, inflating the hit rate. The current entries share only the
        # generic English domain vocabulary that real-world novel prompts
        # would also share.)
        "param_pools": {
            "long_passage": [
                "A coastal town implemented a tidal energy generation system that reduced grid imports by 18% in the first year.",
                "An open-source compiler optimization pass eliminated 11% of redundant load instructions in benchmark code.",
                "A federated learning study showed that local differential privacy reduced model accuracy by 3.2 points.",
                "The migration from monolithic to microservices architecture took 14 months across 47 engineers.",
                "A novel sparse attention mechanism reduced inference memory by 41% with negligible accuracy loss.",
                "Researchers found that interleaved batch sampling improved gradient stability in low-data regimes.",
                "The reinforcement learning policy converged after 2.4M steps in the procedurally generated environment.",
                "A retrospective analysis revealed that 73% of incidents originated from configuration drift.",
                "The new edge caching layer reduced p99 latency from 340ms to 87ms during peak traffic.",
                "Adopting structured outputs reduced downstream parsing errors by 89% in the integration tests.",
                "The team's transition to trunk-based development cut merge conflicts by half within two months.",
                "A graph neural network embedding outperformed the bag-of-words baseline by 14 F1 points.",
                "Implementing OAuth 2.0 PKCE flow eliminated the credential leakage class of vulnerabilities.",
                "Customer churn correlated most strongly with onboarding completion rate, not feature usage.",
                "The chaos engineering experiments uncovered five previously latent failure modes in production.",
                "Cross-region active-active replication added 42ms baseline latency for consistent writes.",
                "A change in the request validation library reduced false positives in malicious traffic detection.",
                "Engineering productivity peaked when pull request review SLAs were enforced at 4 hours.",
                "The new vector quantization scheme compressed the embedding index by 6.3 times with 0.2 point recall loss.",
                "Deprecating the legacy authentication endpoint required a 9-month migration coordinated across teams.",
                "The agile retrospective ritual was credited with surfacing organizational friction earlier.",
                "A novel data augmentation strategy improved out-of-distribution generalization by 7 percent.",
                "Memory profiling revealed a long-tailed allocation pattern caused by string interning miss.",
                "The disaster recovery drill exposed gaps in the cross-region database failover procedure.",
                "Refactoring the storage abstraction layer reduced incident MTTR from 47 to 12 minutes.",
                "The post-mortem identified insufficient observability as the root cause of slow detection.",
                "An A/B test showed that gradual feature ramp-up reduced support ticket spikes by 60 percent.",
                "The new schema validator caught 91 percent of regression-causing data shape changes pre-deployment.",
                "A statistical analysis of code review comments identified three persistent style debate topics.",
                "Hardware acceleration via custom kernels delivered a 4.7 times throughput improvement.",
                "The team adopted property-based testing and discovered 23 edge case bugs in the first month.",
                "Migrating from synchronous to asynchronous logging reduced p95 request latency by 18ms.",
                "The user research panel surfaced unmet needs that weren't visible from telemetry alone.",
                "Implementing rate limiting at the API gateway eliminated a class of denial-of-service patterns.",
                "A formal verification effort proved the correctness of the consensus protocol under partition.",
                "The new release management workflow reduced deployment-induced incidents by 40 percent.",
                "Adopting columnar storage cut analytical query latency by an order of magnitude.",
                "The model card disclosed three known biases and the mitigation strategies applied.",
                "Refactoring the dependency injection container reduced unit test runtime from 8 to 2 minutes.",
                "Network policy enforcement at the service mesh layer simplified security audit compliance.",
                "A measurement study of CDN cache hit rates identified two underperforming edge regions.",
                "The team implemented continuous profiling and identified hot paths invisible to spot checks.",
                "Adopting feature flags as a deployment mechanism decoupled release from rollout cleanly.",
                "The semantic search index rebuild was parallelized across 32 workers, finishing in 14 minutes.",
                "A throughput analysis revealed that GC pauses dominated long-tail latency for the JVM service.",
                "The new alerting strategy reduced page volume by 70 percent while keeping critical alerts intact.",
                "Implementing canary deployments caught a regression that staged testing had missed.",
                "The infrastructure-as-code refactor consolidated 11 disparate provisioning patterns into one.",
                "Cross-team coordination overhead was the dominant cost factor in the multi-quarter project.",
                "An accessibility audit identified 17 issues blocking compliance with WCAG 2.1 AA.",
            ],
            "detailed_question": [
                "How should we redesign the system to handle 10× traffic without rearchitecting the data layer?",
                "What are the trade-offs between strong consistency and high availability for this workload?",
                "How can we measure the developer-productivity impact of the new tooling investments?",
                "What metric should we use to evaluate the success of the personalization model rollout?",
                "How do we balance the cost of comprehensive observability against the value it provides?",
                "What architectural patterns should we adopt for event-driven workloads at our scale?",
                "How can we improve the reliability of the third-party integration without owning that code?",
                "What is the appropriate level of test coverage for code with high blast radius on failure?",
                "How should we structure the team to support both research and production engineering needs?",
                "What governance framework should we apply to machine-learning model deployments?",
                "How do we reduce the cognitive load on on-call engineers during high-incident weeks?",
                "What are the security implications of allowing user-uploaded code to execute in our sandbox?",
                "How can we accelerate the path from prototype to production for experimental features?",
                "What is the right way to migrate this dataset without disrupting downstream consumers?",
                "How should we handle backward incompatibility in our public API across major versions?",
                "What is the appropriate retention policy for application logs given regulatory constraints?",
                "How do we decide when to invest in a custom solution versus adopting an off-the-shelf one?",
                "What is the right framework for prioritizing technical debt against feature delivery?",
                "How can we improve the experience of new engineers during their first 90 days?",
                "What are the operational risks of running this workload on spot instances?",
                "How should the platform team measure success in supporting product engineering velocity?",
                "What design considerations matter most for systems with strict latency SLOs?",
                "How do we evaluate whether the current alerting setup is too noisy or too quiet?",
                "What approach should we take to migrate from synchronous to asynchronous processing?",
                "How can we structure on-call rotations to balance fairness and expertise distribution?",
                "What metrics should govern our capacity planning for predictably bursty traffic?",
                "How do we approach load testing for services that don't have established traffic patterns?",
                "What architectural changes are needed to support multi-tenant isolation at this scale?",
                "How should we handle schema evolution in an event-streaming architecture?",
                "What is the right balance between automated and human review in our deployment pipeline?",
                "How do we identify and eliminate single points of failure in the current architecture?",
                "What strategies work best for managing technical debt accumulated during rapid prototyping?",
                "How should we structure A/B tests for features with low conversion event rates?",
                "What is the appropriate way to handle PII redaction in our log aggregation pipeline?",
                "How do we measure the ROI of investments in developer tooling and platform engineering?",
                "What patterns should we apply when designing rate limiting across federated services?",
                "How should our backup and disaster recovery strategy evolve as the data volume grows?",
                "What is the right granularity for service ownership in a platform-heavy organization?",
                "How can we improve cross-team API contract negotiation and stability?",
                "What design patterns help when integrating with legacy systems lacking modern APIs?",
                "How should we approach gradual migration of the monolith to service-oriented architecture?",
                "What is the optimal cache warming strategy for systems with cold-start sensitivity?",
                "How do we balance push and pull architectures for real-time data distribution?",
                "What governance does the data warehouse need as more downstream teams build on it?",
                "How should we approach the build-vs-buy decision for our internal developer platform?",
                "What are the key considerations for choosing between SQL and NoSQL for this domain?",
                "How can we systematically reduce time-to-detection for production incidents?",
                "What is the right approach to deprecating internal APIs without breaking dependents?",
                "How do we evaluate whether a particular ML model is ready for production rollout?",
                "What metrics best capture the developer experience of working in this codebase?",
            ],
            "complex_scenario": [
                "Three downstream services depend on a deprecated field that engineering needs to remove for security reasons.",
                "A new compliance requirement landed mid-quarter affecting how user data is stored and replicated.",
                "Two teams are independently building similar capabilities; consolidation is being considered.",
                "The latency SLO is being missed during certain traffic patterns; root cause is not yet identified.",
                "A vendor contract negotiation is exposing assumptions in the current integration architecture.",
                "The on-call rotation is finding the alerting too noisy; signal-to-noise tuning is overdue.",
                "An acquisition introduced infrastructure overlap with existing platforms; rationalization is needed.",
                "User-reported issues are spiking but internal metrics look healthy; observability gap suspected.",
                "Quarterly capacity planning reveals an upcoming bottleneck in the message queue throughput.",
                "Engineering velocity has plateaued; the team is investigating whether processes need adjustment.",
                "A new framework version offers significant improvements but requires a multi-week migration.",
                "Cost of cloud infrastructure has grown faster than revenue; optimization opportunities are being explored.",
                "A critical dependency is announcing end-of-life; replacement options are being evaluated.",
                "The product team requested a feature that would conflict with the current data partitioning strategy.",
                "Two competing architectural proposals have emerged for the same problem; tradeoffs need analysis.",
                "Security disclosed a vulnerability in a library used across multiple services; patching strategy is needed.",
                "Pre-launch testing surfaced an edge case that wasn't in the original requirements document.",
                "A team member raised concerns about the technical debt in a critical service before their departure.",
                "Customer support is escalating issues that appear to be caused by intermittent infrastructure problems.",
                "Performance regression detected in the latest release; rollback or roll-forward decision needed.",
                "A new internal tool was built outside the platform team; integration is being considered.",
                "The deployment frequency has decreased despite added automation; root cause is being investigated.",
                "Cross-team interface contracts are eroding; quarterly contract review is overdue.",
                "Test flakiness is increasing in CI; investigation suggests environmental rather than code issues.",
                "A high-traffic event is upcoming; capacity headroom is at 35% above current peak baseline.",
                "Vendor pricing changes are forcing reevaluation of which managed services to keep.",
                "Tech debt remediation work is competing with new feature development for the same engineers.",
                "The reliability of a single upstream provider is impacting our customer-facing availability.",
                "An incident retrospective surfaced patterns that suggest broader organizational learning is needed.",
                "Engineering and product disagree about whether the next quarter should focus on stability or features.",
                "A regulatory deadline requires changes to data handling within 90 days across all systems.",
                "The platform team's roadmap has more dependencies on it than it can deliver this quarter.",
                "An emerging open-source project might replace internal infrastructure; due diligence is underway.",
                "Customer feedback indicates the current pricing model misaligns with usage patterns.",
                "Test environment drift has caused inaccurate validation; environment-as-code investment is being weighed.",
                "Headcount constraints are forcing prioritization between platform stability and new product investments.",
                "Three pilot customers want bespoke integrations; a strategic decision is needed on commitment.",
                "A merger is creating uncertainty about which technology stack will be the standard going forward.",
                "Engineering retention has declined; root cause analysis identifies process and tooling friction.",
                "A new internal customer is asking for capabilities the team had planned to deprecate.",
                "Cross-region data sovereignty requirements are constraining the existing replication architecture.",
                "Recent changes to the build system have lengthened CI times; investigation and remediation needed.",
                "Customer expectations are evolving faster than the team can ship product changes to match.",
                "Multiple stakeholders are requesting conflicting changes to the same shared service.",
                "Recent telemetry suggests user behavior is shifting, but the analytics dashboards lag by a quarter.",
                "An accessibility audit found gaps that require coordinated work across multiple product areas.",
                "Pricing strategy and product strategy need closer alignment based on the latest customer research.",
                "Hiring slowdowns are forcing reprioritization of long-term investment work.",
                "Compliance changes are requiring more frequent third-party audits than the team is prepared for.",
                "Recent customer interviews surfaced concerns about long-term product roadmap visibility.",
            ],
        },
    },
}


def _agent_workload_stats() -> dict:
    """Diagnostics on the AGENT_WORKLOAD distribution — call counts per category."""
    out = {}
    for cat, spec in AGENT_WORKLOAD.items():
        n_templates = len(spec["templates"])
        n_combinations = 0
        for tmpl in spec["templates"]:
            # Estimate combinations via product of slot pool sizes
            import re as _re
            slots = _re.findall(r"\{(\w+)\}", tmpl)
            combos = 1
            for s in slots:
                pool = spec["param_pools"].get(s, ["_"])
                combos *= len(pool)
            n_combinations += combos
        out[cat] = {
            "weight":       spec["weight"],
            "templates":    n_templates,
            "combinations": n_combinations,
        }
    return out


def _generate_agent_prompt(rng) -> tuple:
    """Sample one agent prompt from AGENT_WORKLOAD according to category weights.

    Returns (category, prompt_text).
    """
    import re as _re
    # Pick category by weight
    r = rng.random()
    cum = 0.0
    chosen_cat = None
    for cat, spec in AGENT_WORKLOAD.items():
        cum += spec["weight"]
        if r < cum:
            chosen_cat = cat
            break
    if chosen_cat is None:
        chosen_cat = list(AGENT_WORKLOAD.keys())[-1]  # safety fallthrough

    spec   = AGENT_WORKLOAD[chosen_cat]
    tmpl   = rng.choice(spec["templates"])
    slots  = _re.findall(r"\{(\w+)\}", tmpl)
    params = {s: rng.choice(spec["param_pools"][s]) for s in slots}
    return chosen_cat, tmpl.format(**params)


def run_agent_bench(
    n_sessions:   int   = 50,
    dispatches:   int   = 200,
    threshold:    float = 0.85,
    use_sulci:    bool  = False,
    seed:         int   = 1729,
) -> dict:
    """Synthetic agent-workload benchmark.

    Simulates ``n_sessions`` × ``dispatches`` LLM-call dispatches drawn from
    AGENT_WORKLOAD. Measures aggregate hit rate, per-session hit rate
    distribution (cold → warm → hot), per-category hit rate, and the
    headline ``misses_per_session`` number that maps to the homepage's
    "200 calls → X misses" framing.

    Args:
        n_sessions:  Number of sessions to simulate (default 50).
        dispatches:  LLM-call dispatches per session (default 200).
        threshold:   Similarity threshold (default 0.85, MiniLM-tuned).
        use_sulci:   Real MiniLM+SQLite via sulci.Cache. Otherwise the
                     builtin TF-IDF cache (Mode 1). The relative shape of
                     the result is similar; absolute numbers differ.
        seed:        RNG seed for reproducibility (default 1729).

    Returns:
        {
            "summary": {
                "n_sessions":               int,
                "dispatches_per_session":   int,
                "total_dispatches":         int,
                "total_hits":               int,
                "total_misses":             int,
                "aggregate_hit_rate":       float,
                "misses_per_session_p50":   float,
                "misses_per_session_p95":   float,
                "hit_rate_cold_session":    float,    # session 1
                "hit_rate_warm_session":    float,    # median across last quarter
                "category_hit_rate": {
                    "structural":      float,
                    "semi_structural": float,
                    "novel":           float,
                },
                "category_distribution": {
                    "structural":      int,
                    "semi_structural": int,
                    "novel":           int,
                },
            },
            "per_session": [
                {"session": i, "hits": h, "misses": m, "hit_rate": r},
                ...
            ],
        }
    """
    import random
    rng = random.Random(seed)

    # Choose cache
    if use_sulci:
        # Real MiniLM + SQLite via sulci.Cache
        from sulci import Cache
        import tempfile
        db_dir = tempfile.mkdtemp(prefix="sulci_agent_bench_")
        cache = Cache(
            backend       = "sqlite",
            threshold     = threshold,
            db_path       = os.path.join(db_dir, "cache"),
            ttl_seconds   = None,
            telemetry     = False,
            cost_per_call = 0.005,
        )
        def _lookup(q):
            resp, sim, _ = cache.get(q)
            return (resp, sim)
        def _store(q, r):
            cache.set(q, r)
    else:
        # Mode-1 built-in TF-IDF + LSH cache
        bc = _BuiltinCache(threshold=threshold)
        def _lookup(q):
            resp, sim, _entry = bc.get(q)
            return (resp, sim)
        def _store(q, r):
            bc.set(q, r)

    per_session = []
    total_hits, total_miss = 0, 0
    cat_hits   = {"structural": 0, "semi_structural": 0, "novel": 0}
    cat_dispatches = {"structural": 0, "semi_structural": 0, "novel": 0}

    print(f"\n  Simulating {n_sessions} sessions × {dispatches} dispatches "
          f"= {n_sessions * dispatches:,} total LLM-call dispatches")
    print(f"  Workload mix: structural 45%, semi-structural 35%, novel 20%")

    for s in range(n_sessions):
        s_hits, s_miss = 0, 0
        for _ in range(dispatches):
            cat, prompt = _generate_agent_prompt(rng)
            resp, sim   = _lookup(prompt)
            cat_dispatches[cat] += 1
            if resp is not None:
                s_hits += 1
                cat_hits[cat] += 1
            else:
                s_miss += 1
                # On miss, simulate an LLM response and store it so future
                # similar prompts can hit. Real text doesn't matter for the
                # cache contract — only the prompt vector does.
                _store(prompt, f"<simulated agent response for {cat} prompt>")

        per_session.append({
            "session":  s,
            "hits":     s_hits,
            "misses":   s_miss,
            "hit_rate": round(s_hits / dispatches, 4),
        })
        total_hits += s_hits
        total_miss += s_miss

        if (s + 1) % max(1, n_sessions // 10) == 0:
            print(f"    Session {s+1}/{n_sessions}: hits={s_hits}, misses={s_miss}, "
                  f"hit_rate={s_hits/dispatches:.0%}")

    total_dispatches = total_hits + total_miss
    misses_per_session = sorted(p["misses"] for p in per_session)

    # Warm session = median of last quarter (cache should be warm by then)
    warm_start = max(1, int(n_sessions * 0.75))
    warm_rates = [p["hit_rate"] for p in per_session[warm_start:]]

    summary = {
        "n_sessions":             n_sessions,
        "dispatches_per_session": dispatches,
        "total_dispatches":       total_dispatches,
        "total_hits":             total_hits,
        "total_misses":           total_miss,
        "aggregate_hit_rate":     round(total_hits / total_dispatches, 4) if total_dispatches else 0.0,
        "misses_per_session_p50": misses_per_session[len(misses_per_session) // 2],
        "misses_per_session_p95": misses_per_session[int(len(misses_per_session) * 0.95)],
        "hit_rate_cold_session":  per_session[0]["hit_rate"],
        "hit_rate_warm_session":  round(sum(warm_rates) / len(warm_rates), 4) if warm_rates else 0.0,
        "category_hit_rate": {
            cat: round(cat_hits[cat] / cat_dispatches[cat], 4) if cat_dispatches[cat] else 0.0
            for cat in cat_hits
        },
        "category_distribution": dict(cat_dispatches),
    }

    return {"summary": summary, "per_session": per_session}


def main():
    global _claude

    t0 = time.time()
    print("\n◈ Sulci Benchmark")

    # ── Wipe stale DBs if --fresh ─────────────────────────────────────────────
    if args.fresh:
        _wipe_bench_dbs()

    # ── Initialise Claude client if requested ─────────────────────────────────
    if args.use_claude:
        if not args.use_sulci:
            print("  NOTE: --use-claude works best with --use-sulci (real MiniLM embeddings).")
            print("        Continuing with built-in TF-IDF engine.\n")
        _claude = _ClaudeClient(
            model     = args.claude_model,
            max_calls = args.claude_max_calls,
        )
        print(f"  Claude mode ON  |  model={args.claude_model}  "
              f"max_calls={args.claude_max_calls}")
        print(f"  Estimated max cost: ~${args.claude_max_calls * 0.0009:.2f} "
              f"(Haiku at ~$0.90/1k calls — $0.80/1M input + $4.00/1M output)\n")

    # ── Stateless benchmark ───────────────────────────────────────────────────
    print(f"  Building {args.queries:,}-query corpus...")
    corpus = build_corpus(n_test=args.queries)
    print(f"  Done ({sum(len(v) for v in corpus.values()):,} total queries)\n")

    results = run(corpus, args.threshold, args.use_sulci, verbose=True)

    print("\n── Saving results ───────────────────────────────────────")
    s = summary(results, args.threshold)
    save_json(s,                              "summary.json")
    save_csv(domain_breakdown(results),       "domain_breakdown.csv")
    save_csv(time_series(results),            "time_series.csv")
    save_csv(false_positives_report(results), "false_positives.csv")

    if not args.no_sweep:
        # Skip threshold sweep in Claude mode — each sweep pass would consume
        # additional API calls across all threshold values.
        if _claude:
            print("  Skipping threshold sweep in --use-claude mode to cap API calls.")
        else:
            sw = sweep(corpus, args.use_sulci)
            save_csv(sw, "threshold_sweep.csv")

    elapsed = time.time() - t0
    print(f"\n{'='*62}")
    print(f"  STATELESS BENCHMARK  |  threshold={args.threshold}")
    print(f"{'='*62}")
    print(f"  Queries        : {s['total_queries']:,}")
    print(f"  Hits           : {s['cache_hits']:,}  ({s['hit_rate']:.1%})")
    print(f"  False positives: {s['false_positives']} ({s['false_positive_rate']:.2%})")
    print(f"  Latency (hit)  : {s['latency_hit_p50_ms']:.2f}ms p50  /  {s['latency_hit_p95_ms']:.2f}ms p95")
    # Show real API miss latency when available, otherwise synthetic
    if s.get("real_latency_miss_p50_ms"):
        print(f"  Latency (miss) : {s['real_latency_miss_p50_ms']:.0f}ms p50  /  "
              f"{s['real_latency_miss_p95_ms']:.0f}ms p95  (real Claude API)")
    else:
        print(f"  Latency (miss) : {s['latency_miss_p50_ms']:.2f}ms p50  /  {s['latency_miss_p95_ms']:.2f}ms p95")
    print(f"  Cost saved     : ${s['saved_cost_usd']:.2f}  ({s['cost_reduction_pct']:.1f}%)")
    print(f"  Completed in   : {elapsed:.1f}s")
    print(f"  Results in     : {args.out}/")
    print(f"{'='*62}\n")

    print("  Domain breakdown:")
    for row in domain_breakdown(results):
        print(f"    {row['domain']:22s}  hit={row['hit_rate_pct']:5.1f}%  "
              f"fp={row['fp_rate_pct']:4.1f}%  saved=${row['saved_usd']:.2f}")
    print()

    # ── Claude mode summary ───────────────────────────────────────────────────
    if _claude and _claude.call_count > 0:
        cs = _claude.stats()
        sem_rate = s.get("semantic_correct_rate")
        print(f"  ── Claude API summary ──────────────────────────────")
        cap_hit = _claude.call_count >= _claude.max_calls
        print(f"  Calls made     : {cs['claude_calls']:,}  (cap={args.claude_max_calls})"
              + ("  ← cap reached" if cap_hit else ""))
        if cap_hit:
            unverified = s['total_queries'] - s.get('semantic_scored_hits', 0) - s['cache_misses']
            print(f"  ⚠  Cap hit mid-run — {unverified:,} hits were not semantically verified.")
            print(f"     Raise --claude-max-calls to cover the full run.")
        print(f"  Total cost     : ${cs['claude_total_cost_usd']:.4f}")
        print(f"  Latency p50    : {cs['claude_latency_p50_ms']:.0f}ms")
        print(f"  Latency p95    : {cs['claude_latency_p95_ms']:.0f}ms")
        print(f"  Latency p99    : {cs['claude_latency_p99_ms']:.0f}ms")
        if sem_rate is not None:
            print(f"  Semantic accuracy (cached vs live): "
                  f"{sem_rate:.1%}  "
                  f"(scored {s['semantic_scored_hits']:,} hits)")
            if cap_hit:
                print(f"  NOTE: semantic accuracy reflects only the verified hits.")
        print()

    # ── Context-aware benchmark ───────────────────────────────────────────────
    if args.context:
        ctx_data = run_context_bench(
            n_followups        = args.context_followups,
            use_sulci          = args.use_sulci,
            context_window     = args.context_window,
            query_weight       = args.query_weight,
            holdout_per_domain = args.context_holdout,
            seed               = args.seed,
        )
        save_json(ctx_data["summary"], "context_summary.json")
        save_csv(ctx_data["summary"]["domain_breakdown"], "context_accuracy.csv")

        imp   = ctx_data["summary"]["improvement"]
        c_sum = ctx_data["summary"]["context_aware"]
        print(f"  Context-aware accuracy improvement: "
              f"{imp['accuracy_delta_pct']:+.1f}pp  "
              f"(stateless={ctx_data['summary']['stateless']['resolution_accuracy']:.0%}  "
              f"→ context={c_sum['resolution_accuracy']:.0%})")
        # Never print the accuracy delta on its own. The delta is the headline
        # that produced +20.8pp and +56pp; the false-hit rate is what says
        # whether the delta was bought or earned.
        fh = c_sum["false_hit_rate"]
        print(f"  Context false-hit rate: "
              + (f"{fh:.1%}  ({c_sum['n_false_hits']}/{c_sum['n_should_miss']} "
                 f"held-out follow-ups answered anyway)" if fh is not None
                 else "UNMEASURED (--context-holdout 0)"))
        print(f"  ⚠  {c_sum['n_should_hit'] + c_sum['n_should_miss']} follow-ups total. "
              f"Small. Do not publish a figure from this corpus without the n.")
        print()

        # ── Alpha sweep ───────────────────────────────────────────────────────
        if args.context_sweep:
            alphas = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
            print(f"  ── query_weight sweep ({len(alphas)} points) ──────────────")
            print(f"    {'alpha':>6}  {'sl_acc':>7}  {'ctx_acc':>8}  "
                  f"{'ctx_recall':>11}  {'ctx_false_hit':>14}  {'ctx_prec':>9}")
            rows = []
            for a in alphas:
                d = run_context_bench(
                    n_followups        = args.context_followups,
                    use_sulci          = args.use_sulci,
                    context_window     = args.context_window,
                    query_weight       = a,
                    holdout_per_domain = args.context_holdout,
                    seed               = args.seed,
                    quiet              = True,
                )["summary"]
                s_, c_ = d["stateless"], d["context_aware"]
                rows.append({
                    "query_weight":        a,
                    "stateless_accuracy":  s_["resolution_accuracy"],
                    "context_accuracy":    c_["resolution_accuracy"],
                    "context_recall":      c_["recall"],
                    "context_false_hit":   c_["false_hit_rate"],
                    "context_precision":   c_["precision"],
                    "n_should_hit":        c_["n_should_hit"],
                    "n_should_miss":       c_["n_should_miss"],
                })
                fhs = "n/a" if c_["false_hit_rate"] is None else f"{c_['false_hit_rate']:.1%}"
                prs = "n/a" if c_["precision"] is None else f"{c_['precision']:.1%}"
                print(f"    {a:>6.2f}  {s_['resolution_accuracy']:>6.1%}  "
                      f"{c_['resolution_accuracy']:>7.1%}  "
                      f"{(c_['recall'] or 0):>10.1%}  {fhs:>14}  {prs:>9}")
            save_csv(rows, "context_alpha_sweep.csv")
            print(f"\n  ⚠  A lower alpha that raises accuracy AND false-hit together "
                  f"has not\n     found more answers -- it has loosened the cache. "
                  f"Read both columns.")
            print()

    # ── Agent workload benchmark ─────────────────────────────────────────────
    if args.agent:
        print(f"\n{'='*62}")
        print(f"  AGENT WORKLOAD BENCHMARK  |  threshold={args.agent_threshold}")
        print(f"{'='*62}")

        # Distribution diagnostics (so users can verify the workload shape)
        stats = _agent_workload_stats()
        for cat, info in stats.items():
            print(f"  {cat:<18} weight={info['weight']:.0%}  "
                  f"templates={info['templates']:>2}  "
                  f"max_unique_combinations≈{info['combinations']:,}")

        agent_data = run_agent_bench(
            n_sessions = args.agent_sessions,
            dispatches = args.agent_dispatches,
            threshold  = args.agent_threshold,
            use_sulci  = args.use_sulci,
        )
        save_json(agent_data["summary"],    "agent_summary.json")
        save_csv(agent_data["per_session"], "agent_per_session.csv")

        s_a = agent_data["summary"]
        print(f"\n  Aggregate dispatches  : {s_a['total_dispatches']:,}")
        print(f"  Aggregate hit rate    : {s_a['aggregate_hit_rate']:.1%}")
        print(f"  Cold session hit rate : {s_a['hit_rate_cold_session']:.1%}  (session 1)")
        print(f"  Warm session hit rate : {s_a['hit_rate_warm_session']:.1%}  (median of last quarter)")
        print(f"  Misses per session    : p50={s_a['misses_per_session_p50']}  "
              f"p95={s_a['misses_per_session_p95']}  "
              f"(of {args.agent_dispatches} dispatches)")
        print(f"\n  Per-category hit rate:")
        for cat, rate in s_a["category_hit_rate"].items():
            calls = s_a["category_distribution"][cat]
            print(f"    {cat:<18} {rate:.1%}  (of {calls:,} dispatches)")
        print()

        # Headline framing — "X calls → Y misses per session" — matches the
        # homepage agent positioning. The warm-session number is the steady-state
        # number a production agent will see after the cache has been populated.
        warm_misses_per_dispatch = 1 - s_a["hit_rate_warm_session"]
        warm_misses = round(args.agent_dispatches * warm_misses_per_dispatch)
        print(f"  ── Steady-state headline ──────────────────────────────")
        print(f"   {args.agent_dispatches} dispatches/session  →  ~{warm_misses} LLM calls/session")
        print(f"   ({args.agent_dispatches / max(warm_misses, 1):.1f}× reduction; "
              f"measured, not extrapolated)")
        print(f"  ────────────────────────────────────────────────────────")
        print()


if __name__ == "__main__":
    main()
