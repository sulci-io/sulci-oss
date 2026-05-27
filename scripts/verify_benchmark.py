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


def load_results() -> tuple[dict, dict, dict]:
    """Load benchmark output JSONs.

    Returns (summary, context, agent_or_None). The agent file is optional —
    only present when benchmark/run.py was invoked with --agent.
    """
    summary = RESULTS_DIR / "summary.json"
    context = RESULTS_DIR / "context_summary.json"
    agent   = RESULTS_DIR / "agent_summary.json"
    if not summary.exists():
        print(f"ERROR: {summary} not found (benchmark didn't produce stateless results)",
              file=sys.stderr)
        sys.exit(2)
    if not context.exists():
        print(f"ERROR: {context} not found (benchmark didn't produce context-aware results)",
              file=sys.stderr)
        sys.exit(2)
    agent_data = json.loads(agent.read_text()) if agent.exists() else None
    return (json.loads(summary.read_text()),
            json.loads(context.read_text()),
            agent_data)


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
        rc = run_benchmark(args.timeout)
        if rc != 0:
            return 2

    measured_summary, measured_context, measured_agent = load_results()
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
