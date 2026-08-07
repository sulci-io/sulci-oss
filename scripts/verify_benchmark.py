#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
scripts/verify_benchmark.py
============================
Run the canonical TF-IDF benchmark and verify results against the committed
baseline (benchmark/baseline.json). Fails if any non-latency metric drifts
beyond tolerance.

Why this exists:
    The headline numbers on sulci.io and in benchmark/README.md (85.88% hit
    rate, +20.8pp resolution accuracy, $21.47 saved per 5000 queries) come
    from this benchmark. Any regression here is website-visible breakage.
    'make checkin' should catch it before a PR lands.

Tolerance:
    Percentages: ±1.0 percentage point absolute
    Counts:      ±2 absolute (dict-iteration tie-breaks differ across
                 Linux/macOS; we observed exactly one such off-by-one)
    Latency:     not checked (machine-dependent, varies 100x between
                 a fast Linux runner and a power-saving laptop)

Usage:
    python scripts/verify_benchmark.py
    python scripts/verify_benchmark.py --baseline path/to/other.json

Exit codes:
    0   all metrics within tolerance
    1   one or more metrics drifted beyond tolerance
    2   harness error (benchmark didn't run, JSON missing, etc.)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results"


# Tolerances
PCT_TOL    = 1.0   # 1.0 percentage point on rates
COUNT_TOL  = 2     # 2 absolute on integer counts
COST_TOL   = 0.50  # 50 cents on cost (5000-query budget is $25 baseline)


def run_benchmark(timeout_sec: int) -> int:
    print("Running canonical benchmark: python benchmark/run.py --no-sweep --context")
    print("(expected wall-clock: 10-20 seconds; TF-IDF engine, no MPS / network)")
    print()
    try:
        r = subprocess.run(
            [sys.executable, "benchmark/run.py", "--no-sweep", "--context"],
            cwd=REPO_ROOT,
            timeout=timeout_sec,
            capture_output=False,  # let it print to console for visibility
        )
        if r.returncode != 0:
            print(f"\nERROR: benchmark/run.py exited with code {r.returncode}",
                  file=sys.stderr)
            return r.returncode
        return 0
    except subprocess.TimeoutExpired:
        print(f"\nERROR: benchmark/run.py exceeded {timeout_sec}s timeout",
              file=sys.stderr)
        return -1


def load_results(run_started_at: float | None = None,
                 baseline_engine: str | None = None) -> tuple[dict, dict, dict]:
    """Load benchmark output JSONs.

    Returns (summary, context, agent_or_None). The agent file is optional —
    only present when benchmark/run.py was invoked with --agent.

    ⚠️ STALENESS IS THE POINT OF `run_started_at`. This function used to read
    whatever was on disk. `benchmark/results/` is gitignored and nothing ever
    clears it, and run_benchmark() invokes `run.py --no-sweep --context` with
    NO `--agent` — so the agent block was verified against a leftover
    `agent_summary.json` from some earlier session, and reported six of eight
    rows as [OK]. Observed 2026-08-06: cold 0.27 / warm 0.9942, which are the
    MiniLM figures, compared against a TF-IDF baseline of cold 0.43.

    That is the failure class from the vendored-book incident: **a checker that
    validates a file's properties cannot tell you it is the wrong artifact.**
    The schema was right, the fields were right, the engine was different.

    Files older than `run_started_at` are therefore not read. A required file
    that is stale is an error; the optional agent file is skipped with a named
    reason, because "not checked" and "checked and fine" must not look alike.

    ⚠️ THE AGENT FILE IS GUARDED BY ENGINE, NOT BY TIME, and two timestamp
    designs were tried and discarded first. This script never writes
    agent_summary.json -- it runs `run.py --no-sweep --context` with no
    --agent -- so that file is always produced by a separate earlier
    invocation. Timing it against the subprocess rejected a file the user had
    written 26 seconds before. Timing it against process start rejected the
    same file, because a legitimate `run.py --agent` necessarily finishes
    before this script begins. **Any mtime cutoff either rejects the legitimate
    file or accepts this morning's**; there is no threshold that separates them.

    The engine is what actually differed. The 2026-08-06 leftover was a MiniLM
    run (cold 0.27 / warm 0.9942) checked against a TF-IDF baseline (cold 0.43),
    and six of eight rows reported [OK]. `run.py` now stamps `_provenance.engine`
    into every results JSON, and the agent block is checked only when that
    matches the baseline's engine. A file with no stamp predates the stamp and
    is skipped by name.

    Age is still printed, because it is useful context -- but it decides
    nothing.
    """
    summary = RESULTS_DIR / "summary.json"
    context = RESULTS_DIR / "context_summary.json"
    agent   = RESULTS_DIR / "agent_summary.json"

    def is_stale(path: Path, cutoff: float | None) -> bool:
        if cutoff is None:
            return False          # --skip-run: caller has opted out, see main()
        # 1s slack for filesystems with coarse mtime granularity
        return path.stat().st_mtime < (cutoff - 1.0)

    def engine_of(data: dict) -> str | None:
        return (data.get("_provenance") or {}).get("engine")

    for required in (summary, context):
        if not required.exists():
            print(f"ERROR: {required} not found "
                  f"(benchmark didn't produce these results)", file=sys.stderr)
            sys.exit(2)
        if is_stale(required, run_started_at):
            print(f"ERROR: {required.name} predates this run "
                  f"({_ago(required)}) — the benchmark did not rewrite it.",
                  file=sys.stderr)
            print(f"       Refusing to verify a stale artifact. Delete "
                  f"{RESULTS_DIR}/ and re-run.", file=sys.stderr)
            sys.exit(2)

    agent_data = None
    if agent.exists():
        candidate  = json.loads(agent.read_text())
        found      = engine_of(candidate)
        if baseline_engine is None:
            agent_data = candidate        # baseline records no engine; nothing to compare
        elif found is None:
            print(f"  ⚠  agent_summary.json has no engine stamp ({_ago(agent)}) "
                  f"and is NOT checked.")
            print(f"     It predates the _provenance stamp. Re-run "
                  f"`run.py --agent` to produce a stamped file.")
        elif found != baseline_engine:
            print(f"  ⚠  agent_summary.json was produced by '{found}' but the "
                  f"baseline is '{baseline_engine}'.")
            print(f"     NOT checked ({_ago(agent)}). This is the 2026-08-06 "
                  f"defect: a leftover MiniLM run")
            print(f"     verified against TF-IDF numbers reported six of eight "
                  f"rows as [OK].")
        else:
            agent_data = candidate

    return (json.loads(summary.read_text()),
            json.loads(context.read_text()),
            agent_data)


def _ago(path: Path) -> str:
    """Human-readable age of a file, for staleness messages."""
    secs = max(0.0, time.time() - path.stat().st_mtime)
    if secs < 90:
        return f"{secs:.0f}s old"
    if secs < 5400:
        return f"{secs / 60:.0f}m old"
    if secs < 172800:
        return f"{secs / 3600:.0f}h old"
    return f"{secs / 86400:.0f}d old"


def compare(label: str, baseline_val, measured_val, tol, kind: str) -> tuple[bool, str]:
    """
    Compare a single metric. Returns (ok, formatted_diff_string).

    kind is "pct" (percentage in 0-100), "rate" (rate in 0-1),
            "count" (integer), or "money" (USD float).
    """
    delta = measured_val - baseline_val
    abs_delta = abs(delta)
    if kind == "pct":
        formatted = f"{baseline_val:>7.2f} → {measured_val:>7.2f}  Δ={delta:+.2f}pp"
        ok = abs_delta <= tol
    elif kind == "rate":
        formatted = f"{baseline_val:>7.4f} → {measured_val:>7.4f}  Δ={delta:+.4f}"
        ok = abs_delta * 100 <= tol  # tol is in pp; rate is 0-1
    elif kind == "count":
        formatted = f"{baseline_val:>7d} → {measured_val:>7d}  Δ={delta:+d}"
        ok = abs_delta <= tol
    elif kind == "money":
        formatted = f"${baseline_val:>6.2f} → ${measured_val:>6.2f}  Δ=${delta:+.2f}"
        ok = abs_delta <= tol
    else:
        raise ValueError(f"unknown kind: {kind}")

    marker = "OK" if ok else "DRIFT"
    return ok, f"  [{marker:5}]  {label:<46} {formatted}"


def verify_against_baseline(measured_summary: dict, measured_context: dict,
                            measured_agent: dict | None,
                            baseline: dict) -> bool:
    print("\n" + "=" * 72)
    print(" Verifying benchmark output against baseline")
    print(f" baseline: {baseline['_meta']['source']}")
    # A baseline that has been superseded still runs, and still goes red. Print
    # WHY at the top rather than leaving eight [DRIFT] lines to be interpreted --
    # an unexplained red check is a check people learn to skip.
    _status = baseline["_meta"].get("STATUS")
    if _status:
        print(f" ⚠  {_status}")
        if baseline["_meta"].get("superseded_on"):
            print(f"    superseded {baseline['_meta']['superseded_on']}. "
                  f"Drift below is EXPECTED until this file is regenerated.")
        if baseline["_meta"].get("to_regenerate"):
            print(f"    to regenerate: {baseline['_meta']['to_regenerate']}")
    print(f" tolerances: rates ±{PCT_TOL}pp, counts ±{COUNT_TOL}, money ±${COST_TOL:.2f}")
    print("=" * 72)

    all_ok = True

    # Stateless headline
    print("\n  Stateless (5000-query):")
    bs = baseline["stateless"]
    for label, key, kind, tol in [
        ("hit_rate",            "hit_rate",            "rate",  PCT_TOL),
        ("cache_hits",          "cache_hits",          "count", COUNT_TOL),
        ("false_positives",     "false_positives",     "count", COUNT_TOL),
        ("false_positive_rate", "false_positive_rate", "rate",  PCT_TOL),
        ("saved_cost_usd",      "saved_cost_usd",      "money", COST_TOL),
        ("cost_reduction_pct",  "cost_reduction_pct",  "pct",   PCT_TOL),
    ]:
        if key not in measured_summary:
            print(f"  [MISS ]  {label:<46} not present in measured output")
            all_ok = False
            continue
        ok, line = compare(label, bs[key], measured_summary[key], tol, kind)
        print(line)
        if not ok:
            all_ok = False

    # Context-aware headline
    print("\n  Context-aware (125-followup):")
    bc = baseline["context_aware"]
    msl = measured_context.get("stateless", {})
    mca = measured_context.get("context_aware", {})
    mim = measured_context.get("improvement", {})

    pairs = [
        ("stateless_hit_rate",            bc["stateless_hit_rate"],
            msl.get("hit_rate"),                "rate", PCT_TOL),
        ("stateless_resolution_accuracy", bc["stateless_resolution_accuracy"],
            msl.get("resolution_accuracy"),     "rate", PCT_TOL),
        ("context_hit_rate",              bc["context_hit_rate"],
            mca.get("hit_rate"),                "rate", PCT_TOL),
        ("context_resolution_accuracy",   bc["context_resolution_accuracy"],
            mca.get("resolution_accuracy"),     "rate", PCT_TOL),
        ("accuracy_delta_pct",            bc["accuracy_delta_pct"],
            mim.get("accuracy_delta_pct"),      "pct",  PCT_TOL),
        ("hit_rate_delta",                bc["hit_rate_delta"],
            mim.get("hit_rate_delta"),          "rate", PCT_TOL),
    ]
    for label, baseline_val, measured_val, kind, tol in pairs:
        if measured_val is None:
            print(f"  [MISS ]  {label:<46} not present in measured output")
            all_ok = False
            continue
        ok, line = compare(label, baseline_val, measured_val, tol, kind)
        print(line)
        if not ok:
            all_ok = False

    # Domain breakdown — accuracy improvement per domain
    print("\n  Per-domain accuracy improvement (context vs stateless):")
    measured_domains = {d["domain"]: d
                        for d in measured_context.get("domain_breakdown", [])}
    for entry in baseline["domain_breakdown_context_aware"]:
        d = entry["domain"]
        baseline_imp = entry["improvement"]
        if d not in measured_domains:
            print(f"  [MISS ]  domain {d:<32} not in measured output")
            all_ok = False
            continue
        measured_imp = measured_domains[d].get("accuracy_improvement", 0.0)
        ok, line = compare(f"{d} improvement", baseline_imp, measured_imp,
                           PCT_TOL, "rate")
        print(line)
        if not ok:
            all_ok = False

    # Agent workload — verified ONLY when benchmark was run with --agent.
    # Daily make checkin doesn't pass --agent so agent_summary.json is absent
    # and this block graceful-skips. Pre-release verification runs the agent
    # benchmark explicitly and validates against the pinned numbers.
    if measured_agent is None:
        print("\n  Agent workload: not measured this run (run benchmark/run.py --agent to check)")
    elif "agent_workload" not in baseline:
        print("\n  Agent workload: measured but no baseline pinned — recording, not checking")
    else:
        print(f"\n  Agent workload ({baseline['agent_workload']['n_sessions']}-session, "
              f"{baseline['agent_workload']['dispatches_per_session']}-dispatch):")
        ba = baseline["agent_workload"]
        ma = measured_agent
        # The TF-IDF agent benchmark involves randomness via the sampling seed,
        # so per-category hit rates can drift ±2pp run-to-run. Aggregate metric
        # uses a slightly wider tolerance than the stateless / context blocks.
        AGENT_PCT_TOL = max(PCT_TOL, 3.0)   # 3pp tolerance for agent metrics

        pairs = [
            ("aggregate_hit_rate",      ba["aggregate_hit_rate"],
                ma.get("aggregate_hit_rate"),                    "rate", AGENT_PCT_TOL),
            ("hit_rate_cold_session",   ba["hit_rate_cold_session"],
                ma.get("hit_rate_cold_session"),                 "rate", AGENT_PCT_TOL),
            ("hit_rate_warm_session",   ba["hit_rate_warm_session"],
                ma.get("hit_rate_warm_session"),                 "rate", AGENT_PCT_TOL),
            ("misses_per_session_p50",  ba["misses_per_session_p50"],
                ma.get("misses_per_session_p50"),                "count", 5),
            ("misses_per_session_p95",  ba["misses_per_session_p95"],
                ma.get("misses_per_session_p95"),                "count", 8),
            ("structural hit rate",     ba["category_hit_rate_structural"],
                ma.get("category_hit_rate", {}).get("structural"),       "rate", AGENT_PCT_TOL),
            ("semi_structural hit rate", ba["category_hit_rate_semi_structural"],
                ma.get("category_hit_rate", {}).get("semi_structural"),  "rate", AGENT_PCT_TOL),
            ("novel hit rate",          ba["category_hit_rate_novel"],
                ma.get("category_hit_rate", {}).get("novel"),            "rate", AGENT_PCT_TOL),
        ]
        for label, b_val, m_val, kind, tol in pairs:
            if m_val is None:
                print(f"  [MISS ]  {label:<46} not present in agent_summary.json")
                all_ok = False
                continue
            ok, line = compare(label, b_val, m_val, tol, kind)
            print(line)
            if not ok:
                all_ok = False

    print("\n" + "=" * 72)
    return all_ok


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", default="benchmark/baseline.json",
                   help="path to baseline JSON (default: benchmark/baseline.json)")
    p.add_argument("--timeout", type=int, default=120,
                   help="seconds to allow for benchmark/run.py (default: 120)")
    p.add_argument("--skip-run", action="store_true",
                   help="skip running the benchmark; verify against existing "
                        "benchmark/results/*.json (useful for re-checking)")
    args = p.parse_args()

    baseline_path = REPO_ROOT / args.baseline
    if not baseline_path.exists():
        print(f"ERROR: baseline not found at {baseline_path}", file=sys.stderr)
        return 2
    baseline = json.loads(baseline_path.read_text())

    if not args.skip_run:
        run_started_at = time.time()
        rc = run_benchmark(args.timeout)
        if rc != 0:
            return 2
    else:
        # Opting out of the run opts out of the staleness check — there is no
        # run to be stale relative to. Say so, rather than letting the absence
        # of a warning read as a clean bill of health.
        run_started_at = None
        print("  ⚠  --skip-run: verifying pre-existing benchmark/results/*.json.")
        print("     Their age and engine are not checked.")

    measured_summary, measured_context, measured_agent = load_results(
        run_started_at,
        baseline_engine=(baseline.get("_meta") or {}).get("engine"))
    ok = verify_against_baseline(measured_summary, measured_context,
                                 measured_agent, baseline)

    if ok:
        print("\n  ALL METRICS WITHIN TOLERANCE — no regression")
        print("=" * 72)
        return 0
    else:
        print("\n  ONE OR MORE METRICS DRIFTED — investigate before merging")
        print(f"  Baseline: {baseline_path}")
        print("=" * 72)
        return 1


if __name__ == "__main__":
    sys.exit(main())
