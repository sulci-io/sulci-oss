#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
scripts/run_examples.py
========================
Run every example and smoke test in this repository, report pass/fail/timeout
per file, and produce a summary table.

What it covers:
    * examples/*.py              — usage examples (basic_usage, context_aware,
                                   anthropic, langchain, llamaindex, async,
                                   agent_example_langgraph)
    * examples/extending_sulci/  — protocol-implementation reference impls
    * smoke_test*.py             — end-to-end smoke tests at repo root

What it doesn't cover:
    * The pytest test suite — use scripts/run_tests_per_file.py for that
    * The provider-detection matrix — use scripts/verify_integration_examples.py

API key behavior:
    Examples that integrate with LLM providers (anthropic, langchain,
    llamaindex, async_example) detect ANTHROPIC_API_KEY / OPENAI_API_KEY
    and fall back to a mock LLM if neither is set. This script does NOT
    set or clear keys — it runs each example with whatever's in the
    environment, so by default mock fallback is exercised.

Default behavior:
    On Apple Silicon MPS, first MiniLM load is ~30-40s. Each example
    runs in its own Python subprocess paying its own cold-start. Default
    per-example timeout is 120s, override with --timeout.

Usage:
    python scripts/run_examples.py
    python scripts/run_examples.py --timeout 180
    python scripts/run_examples.py --filter examples/basic_usage.py

Exit codes:
    0   all selected examples completed (exit 0 in their own process)
    1   at least one example failed or timed out
    2   harness error (no examples found, etc.)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default discovery: examples/, examples/extending_sulci/, and smoke_test*.py.
# Order chosen to surface fast/standalone failures first; LLM-using examples
# come later because they're slowest (real or mock LLM calls).
DEFAULT_FILES = [
    # Quick, no LLM
    "examples/basic_usage.py",
    "examples/extending_sulci/custom_backend.py",

    # Smoke tests (also fast, run once per Python process)
    "smoke_test.py",
    "smoke_test_async.py",
    "smoke_test_langchain.py",
    "smoke_test_llamaindex.py",

    # Context-aware examples (no LLM, ~30-90s with MiniLM warmup)
    "examples/context_aware.py",
    "examples/context_aware_example.py",

    # LLM-using examples (mock fallback if no keys, real if keys are set)
    "examples/async_example.py",
    "examples/anthropic_example.py",
    "examples/langchain_example.py",
    "examples/llamaindex_example.py",
    "examples/agent_example_langgraph.py",
]


def run_one(target: str, timeout_sec: int, log_dir: Path) -> dict:
    """Run a single example/script and capture its outcome."""
    log_path = log_dir / (target.replace("/", "_") + ".log")
    cmd = [sys.executable, target]

    print(f"  running: {target} ... ", end="", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout_sec, cwd=REPO_ROOT)
        wallclock = time.time() - t0
        log_path.write_text(
            "=== STDOUT ===\n" + r.stdout +
            "\n=== STDERR ===\n" + r.stderr +
            f"\n=== exit: {r.returncode}, wallclock: {wallclock:.1f}s ===\n"
        )
        if r.returncode == 0:
            print(f"PASS    {wallclock:.1f}s")
            return {"target": target, "ok": True, "status": "PASS",
                    "wallclock": wallclock, "exit_code": 0,
                    "log": str(log_path)}
        else:
            print(f"FAIL    exit={r.returncode}  {wallclock:.1f}s")
            return {"target": target, "ok": False, "status": "FAIL",
                    "wallclock": wallclock, "exit_code": r.returncode,
                    "log": str(log_path)}
    except subprocess.TimeoutExpired as e:
        wallclock = time.time() - t0
        # Capture whatever was written before kill, for diagnosis.
        stdout_partial = (e.stdout.decode() if e.stdout else "") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr_partial = (e.stderr.decode() if e.stderr else "") if isinstance(e.stderr, bytes) else (e.stderr or "")
        log_path.write_text(
            "=== TIMEOUT ===\n"
            f"Killed after {wallclock:.1f}s (limit {timeout_sec}s)\n\n"
            "=== STDOUT (partial) ===\n" + stdout_partial +
            "\n=== STDERR (partial) ===\n" + stderr_partial
        )
        print(f"TIMEOUT  {wallclock:.1f}s (limit {timeout_sec}s)")
        return {"target": target, "ok": False, "status": "TIMEOUT",
                "wallclock": wallclock, "exit_code": -1,
                "log": str(log_path)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timeout", type=int, default=120,
                   help="per-example wall-clock timeout in seconds (default: 120). "
                        "Cold MiniLM load ~30-40s on Apple Silicon MPS, plus example "
                        "work, plus real LLM calls if keys are set.")
    p.add_argument("--files", nargs="+", default=None,
                   help="specific files to run (default: all examples + smoke tests)")
    p.add_argument("--filter", default=None,
                   help="run only files containing this substring in their path")
    p.add_argument("--log-dir", default="/tmp/sulci-examples-runner",
                   help="where to save per-file log output")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    targets = args.files if args.files else DEFAULT_FILES
    if args.filter:
        targets = [t for t in targets if args.filter in t]

    missing = [t for t in targets if not (REPO_ROOT / t).exists()]
    if missing:
        print(f"ERROR: missing target paths: {missing}", file=sys.stderr)
        return 2
    if not targets:
        print("ERROR: no targets selected", file=sys.stderr)
        return 2

    # Detect which API keys are set, just for an informational header.
    keys = []
    if os.environ.get("OPENAI_API_KEY"):    keys.append("OPENAI_API_KEY")
    if os.environ.get("ANTHROPIC_API_KEY"): keys.append("ANTHROPIC_API_KEY")
    keys_label = ", ".join(keys) if keys else "none (mock fallback for LLM examples)"

    print("=" * 70)
    print(" Sulci examples + smoke-test runner")
    print(f" timeout-per-file: {args.timeout}s   targets: {len(targets)}")
    print(f" API keys in env:  {keys_label}")
    print(f" logs: {log_dir}")
    print("=" * 70)

    results = []
    overall_t0 = time.time()
    for target in targets:
        results.append(run_one(target, args.timeout, log_dir))
    total = time.time() - overall_t0

    # Summary
    print("\n" + "=" * 70)
    print(f" SUMMARY ({total:.1f}s total wall-clock)")
    print("=" * 70)
    print(f"  {'Target':<48} {'Status':<8}  {'time':>7}")
    print(f"  {'-'*48} {'-'*8}  {'-'*7}")
    for r in results:
        print(f"  {r['target']:<48} {r['status']:<8}  {r['wallclock']:>6.1f}s")

    passed = sum(1 for r in results if r["ok"])
    total_count = len(results)
    failing = [r for r in results if not r["ok"]]

    print(f"\n  TOTAL: {passed}/{total_count} passed")
    if failing:
        print(f"\n  Failing targets (logs in {log_dir}):")
        for r in failing:
            print(f"    - {r['target']}  →  {r['log']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
