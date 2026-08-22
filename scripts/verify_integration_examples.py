#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
scripts/verify_integration_examples.py
================================
Verify that LLM-using examples (langchain_example.py, llamaindex_example.py)
correctly select their LLM provider across all credential configurations.

Cross product of:
    * examples: langchain_example.py, llamaindex_example.py
    * scenarios: no keys / OpenAI only / Anthropic only / both keys

→ 8 runs total. For each run, we override the API-key env vars in the
subprocess (your shell environment is never modified), execute the example
end-to-end, and parse its '→ Using: <provider>' output line to confirm the
expected provider was selected.

Why this matters:
    Both example files claim to detect available API keys and fall back to
    a mock LLM when neither is set. The matrix exercises the actual code
    path for every combination, including real-API end-to-end on whichever
    provider is selected. This is the kind of thing CI doesn't check
    (because CI doesn't have real keys), so it's a manual pre-release task.

Preconditions:
    * Both OPENAI_API_KEY and ANTHROPIC_API_KEY must be set in the shell
      env. The script bails with a clear error if either is missing.
    * The corresponding SDKs must be importable. Failures here surface as
      ImportError fallbacks within the example.

Cost:
    Real LLM calls are made for the OpenAI-only, Anthropic-only, and
    both-keys scenarios. Per scenario, a typical demo run is 10-30
    completions on small queries, costing on the order of $0.01-0.05.
    Full matrix: roughly $0.10-0.20 of total provider spend per run.

Usage:
    python scripts/verify_integration_examples.py
    python scripts/verify_integration_examples.py --timeout 240
    python scripts/verify_integration_examples.py --scenarios "no keys" "Anthropic only"

Exit codes:
    0   all 8 scenarios produced the expected provider and exited 0
    1   at least one scenario mismatch or non-zero exit
    2   harness error (missing keys, missing SDKs, etc.)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

EXAMPLES = [
    ("langchain_example.py",  "examples/langchain_example.py"),
    ("llamaindex_example.py", "examples/llamaindex_example.py"),
]

# (label, openai_value, anthropic_value, expected_provider)
# Empty string in the value column means "explicitly unset for this run".
def build_scenarios(real_openai: str, real_anthropic: str) -> List[Tuple[str, str, str, str]]:
    return [
        ("no keys",        "",            "",              "mock"),
        ("OpenAI only",    real_openai,   "",              "openai"),
        ("Anthropic only", "",            real_anthropic,  "anthropic"),
        ("both keys",      real_openai,   real_anthropic,  "openai"),  # OpenAI checked first
    ]

# Strict detector: anchor on "→ Using:" prefix and look at the *first word*
# that follows. Avoids matching substrings like "set OPENAI_API_KEY or ...".
USING_RE = re.compile(r"→\s*Using:\s*(\w+)", re.IGNORECASE)


def detect_provider(stdout: str) -> str:
    m = USING_RE.search(stdout)
    if not m:
        return "unknown"
    word = m.group(1).lower()
    # First word after "Using:" is the provider name as the example prints it:
    #   "Using: OpenAI gpt-4o-mini"            → "openai"
    #   "Using: Anthropic claude-haiku-..."    → "anthropic"
    #   "Using: mock LLM (...)"                → "mock"
    if word in ("openai", "anthropic", "mock"):
        return word
    return "unknown"


def run_scenario(example: Tuple[str, str], scenario: Tuple[str, str, str, str],
                 timeout_sec: int, log_dir: Path) -> dict:
    ex_name, ex_path = example
    scen_name, openai_val, anth_val, expected = scenario

    # Build env: copy current, then override the two keys per scenario.
    # We never write to os.environ — only the subprocess sees the override.
    env = os.environ.copy()
    if openai_val:
        env["OPENAI_API_KEY"] = openai_val
    else:
        env.pop("OPENAI_API_KEY", None)
    if anth_val:
        env["ANTHROPIC_API_KEY"] = anth_val
    else:
        env.pop("ANTHROPIC_API_KEY", None)

    log_path = log_dir / f"{ex_name}__{scen_name.replace(' ', '_')}.log"
    print(f"  [{ex_name:<26}] [{scen_name:<14}] running ...", end="", flush=True)

    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, ex_path], env=env,
                           capture_output=True, text=True,
                           timeout=timeout_sec, cwd=REPO_ROOT)
        wallclock = time.time() - t0
        log_path.write_text(
            "=== STDOUT ===\n" + r.stdout +
            "\n=== STDERR ===\n" + r.stderr +
            f"\n=== exit: {r.returncode}, wallclock: {wallclock:.1f}s ===\n"
        )
        detected = detect_provider(r.stdout)
        ok = (detected == expected and r.returncode == 0)
        marker = "OK" if ok else "MISMATCH"
        print(f" {marker:<8} detected={detected:<10} exit={r.returncode}  ({wallclock:.1f}s)")
        return {"example": ex_name, "scenario": scen_name,
                "expected": expected, "detected": detected,
                "exit_code": r.returncode, "wallclock": wallclock,
                "ok": ok, "log": str(log_path)}
    except subprocess.TimeoutExpired:
        wallclock = time.time() - t0
        log_path.write_text(f"TIMEOUT after {wallclock:.1f}s (limit {timeout_sec}s)\n")
        print(f" TIMEOUT  {wallclock:.1f}s")
        return {"example": ex_name, "scenario": scen_name,
                "expected": expected, "detected": "TIMEOUT",
                "exit_code": -1, "wallclock": wallclock,
                "ok": False, "log": str(log_path)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timeout", type=int, default=240,
                   help="per-scenario wall-clock timeout in seconds (default: 240). "
                        "Real LLM calls + cold MiniLM warmup can take 100-150s.")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help='filter to specific scenario labels: "no keys", "OpenAI only", '
                        '"Anthropic only", "both keys"')
    p.add_argument("--examples", nargs="+", default=None,
                   help="filter to specific example basenames "
                        "(e.g. langchain_example.py)")
    p.add_argument("--log-dir", default="/tmp/sulci-verify-integration-examples",
                   help="where to save per-scenario log output")
    args = p.parse_args()

    real_openai    = os.environ.get("OPENAI_API_KEY", "")
    real_anthropic = os.environ.get("ANTHROPIC_API_KEY", "")

    if not real_openai or not real_anthropic:
        missing = []
        if not real_openai:    missing.append("OPENAI_API_KEY")
        if not real_anthropic: missing.append("ANTHROPIC_API_KEY")
        print("ERROR: provider matrix requires both API keys.", file=sys.stderr)
        print(f"       Missing in environment: {', '.join(missing)}", file=sys.stderr)
        print(f"       Set them and re-run, or filter scenarios with --scenarios "
              f"to test only the no-keys path.", file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios(real_openai, real_anthropic)
    if args.scenarios:
        wanted = set(args.scenarios)
        scenarios = [s for s in scenarios if s[0] in wanted]
        if not scenarios:
            print(f"ERROR: no scenarios match {args.scenarios}. "
                  f"Available: 'no keys', 'OpenAI only', 'Anthropic only', 'both keys'",
                  file=sys.stderr)
            return 2

    examples = EXAMPLES
    if args.examples:
        wanted = set(args.examples)
        examples = [(name, path) for name, path in examples if name in wanted]
        if not examples:
            print(f"ERROR: no examples match {args.examples}", file=sys.stderr)
            return 2

    total = len(examples) * len(scenarios)

    print("=" * 84)
    print(f" Sulci integration-examples verifier (provider-detection matrix)")
    print(f" examples: {len(examples)}   scenarios: {len(scenarios)}   "
          f"runs: {total}   timeout: {args.timeout}s")
    print(f" logs: {log_dir}")
    print("=" * 84)

    results = []
    t0_overall = time.time()
    for example in examples:
        for scenario in scenarios:
            results.append(run_scenario(example, scenario, args.timeout, log_dir))
    total_wall = time.time() - t0_overall

    # Summary
    print("\n" + "=" * 84)
    print(f" MATRIX ({total_wall:.1f}s total wall-clock)")
    print("=" * 84)
    print(f"  {'Example':<26} {'Scenario':<16} {'Expected':<10} "
          f"{'Detected':<10} {'Exit':>4} {'Time':>7}  Result")
    print(f"  {'-'*26} {'-'*16} {'-'*10} {'-'*10} {'-'*4} {'-'*7}  ------")
    for r in results:
        print(f"  {r['example']:<26} {r['scenario']:<16} {r['expected']:<10} "
              f"{r['detected']:<10} {r['exit_code']:>4} {r['wallclock']:>6.1f}s  "
              f"{'PASS' if r['ok'] else 'FAIL'}")

    passed = sum(1 for r in results if r["ok"])
    failing = [r for r in results if not r["ok"]]
    print(f"\n  {passed}/{len(results)} scenarios passed")
    if failing:
        print(f"\n  Failing scenarios (logs in {log_dir}):")
        for r in failing:
            print(f"    - {r['example']} [{r['scenario']}]  →  {r['log']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
