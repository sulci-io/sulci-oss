#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
scripts/run_tests_per_file.py
==============================
Run each test file in tests/ in its own pytest invocation, sequentially.

Why one file at a time:
    Several integration tests (langchain, llamaindex) construct multiple
    MiniLMEmbedder instances in one process. On Apple Silicon (MPS), this
    occasionally deadlocks at embeddings.cpu() under memory pressure.
    Running each test file in a fresh Python process avoids that and gives
    each file a clean MiniLM cold-start.

Trade-off:
    Wall-clock time is longer than a single 'pytest tests/' because every
    test file pays the MiniLM warm-up cost (~10s on M-series Macs).
    On a CPU-only Linux runner the difference is much smaller.

Selection model:
    By default we DISCOVER test files via glob — `tests/test_*.py` and
    `tests/compat/`. This means newly-added test files are included
    automatically; no edits to this script needed when a new
    `tests/test_foo.py` is added. (Earlier versions had a hardcoded
    DEFAULT_FILES list that silently excluded any file not on it —
    same trap as platform's checkin.sh `--ignore=` lists.)

    `ORDER_HINT` below sets a preferred run order (fast first so failures
    fail loud and early; slowest last). Files not in the hint run after
    the hinted ones, in alphabetical order.

    `SLOW_FILES` is used by `--skip-slow` for faster local iteration.

Usage:
    python scripts/run_tests_per_file.py
    python scripts/run_tests_per_file.py --timeout 90
    python scripts/run_tests_per_file.py --files tests/test_core.py tests/test_context.py
    python scripts/run_tests_per_file.py --skip-slow
    python scripts/run_tests_per_file.py --list   # print the discovered file list and exit

Exit codes:
    0   all selected files passed (zero failures, zero errors)
    1   at least one file failed or errored
    2   harness error (bad arguments, missing files, etc.)
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
TESTS_DIR = REPO_ROOT / "tests"

#: Roots searched for suites. Kept in step BY HAND with SEARCH_ROOTS in
#: scripts/check_ci_test_coverage.py -- the two scripts answer different
#: questions (what does this runner execute / what does CI execute) and a
#: shared import would hide that they can legitimately differ. What they must
#: not do is differ BY ACCIDENT, which is #135: this script discovered
#: tests/test_*.py plus a hardcoded `tests/compat/`, so tests/integration/flows
#: and sulci/tests/compat were run by CI and never by `make checkin`.
#: Measured 2026-08-20: 26 targets here against 30 suites there.
#:
#: Declared, not recursive-from-repo-root -- see the SEARCH_ROOTS note in
#: check_ci_test_coverage.py for why build_figures.py v2 is the argument
#: against a blind recurse.
SEARCH_ROOTS = ("tests", "sulci/tests")

# Order hint — fast files first so failures fail loud and early; slowest last.
# Files not in this list still get included (after these, alphabetically) —
# the hint only affects ORDER, not membership. Add or remove freely without
# worrying about silently dropping files.
ORDER_HINT = [
    "tests/test_backends.py",
    "tests/test_cloud_backend.py",
    "tests/test_connect.py",
    "tests/test_context.py",
    "tests/test_core.py",
    "tests/test_async_cache.py",
    "tests/test_oss_connect.py",
    "tests/test_qdrant_tenant_isolation.py",
    "tests/test_integrations_langchain.py",
    "tests/test_integrations_llamaindex.py",
    "tests/compat/",
]

# Files that are typically slow (>1 min). Used by --skip-slow.
SLOW_FILES = {
    "tests/test_integrations_langchain.py",
    "tests/test_integrations_llamaindex.py",
    "tests/test_async_cache.py",
    "tests/test_core.py",
}

# Pytest summary tokens can appear in any order: "1 failed, 27 passed in 73s"
# or "28 passed in 55s" or "5 passed, 3 skipped in 12s" etc.
COUNT_PATTERNS = {
    "passed":  re.compile(r"(\d+)\s+passed"),
    "failed":  re.compile(r"(\d+)\s+failed"),
    "errors":  re.compile(r"(\d+)\s+error"),  # matches "error" and "errors"
    "skipped": re.compile(r"(\d+)\s+skipped"),
}
RUNTIME_PATTERN = re.compile(r"in\s+([\d.]+)s")


def discover_targets() -> List[str]:
    """Discover all test targets via glob, then order by ORDER_HINT.

    Discovery is deliberately lenient: any `tests/test_*.py` plus
    `tests/compat/` if it's a directory with at least one `test_*.py`
    inside. Anything that doesn't match doesn't get run.

    Returns posix-style paths relative to REPO_ROOT.
    """
    discovered: set[str] = set()

    for rel in SEARCH_ROOTS:
        root = REPO_ROOT / rel
        if not root.is_dir():
            continue

        # Files at the top level of the root, one invocation each.
        for p in root.glob("test_*.py"):
            discovered.add(str(p.relative_to(REPO_ROOT)).replace(os.sep, "/"))

        # Subdirectories holding suites, invoked as a DIRECTORY -- one pytest
        # call for the whole package rather than one per file. tests/compat
        # and sulci/tests/compat share conftest fixtures, and
        # tests/integration/flows imports flow_*.py helpers that are not
        # suites; splitting either per-file breaks them. rglob, so a suite
        # nested two deep is still found.
        for sub in sorted(d for d in root.iterdir() if d.is_dir()):
            if any(sub.rglob("test_*.py")):
                discovered.add(
                    str(sub.relative_to(REPO_ROOT)).replace(os.sep, "/") + "/"
                )

    # Order: ORDER_HINT first (in their listed order), then anything left
    # alphabetically. Items in ORDER_HINT that don't actually exist are
    # silently dropped — keeping ORDER_HINT a HINT, not a contract.
    ordered: List[str] = []
    seen: set[str] = set()
    for hint in ORDER_HINT:
        if hint in discovered:
            ordered.append(hint)
            seen.add(hint)
    for extra in sorted(discovered - seen):
        ordered.append(extra)
    return ordered


def parse_summary(stdout: str) -> Tuple[int, int, int, int, float]:
    """Extract (passed, failed, errors, skipped, runtime_sec) from pytest output."""
    last_summary = None
    for line in stdout.splitlines():
        if (line.startswith("=") and
                any(k in line for k in ("passed", "failed", "error", "skipped"))):
            last_summary = line
    if not last_summary:
        return (0, 0, 0, 0, 0.0)

    def _extract(pat: re.Pattern) -> int:
        m = pat.search(last_summary)
        return int(m.group(1)) if m else 0

    passed  = _extract(COUNT_PATTERNS["passed"])
    failed  = _extract(COUNT_PATTERNS["failed"])
    errors  = _extract(COUNT_PATTERNS["errors"])
    skipped = _extract(COUNT_PATTERNS["skipped"])

    rm = RUNTIME_PATTERN.search(last_summary)
    runtime = float(rm.group(1)) if rm else 0.0
    return (passed, failed, errors, skipped, runtime)


def run_one(target: str, timeout_sec: int, log_dir: Path) -> dict:
    """Run pytest against a single target and return a result dict."""
    log_path = log_dir / (target.replace("/", "_").rstrip("_") + ".log")
    cmd = [sys.executable, "-m", "pytest", target, "-v",
           f"--timeout={timeout_sec}", "--tb=short"]

    print(f"  running: {target} ... ", end="", flush=True)
    t0 = time.time()
    try:
        # Wall-clock budget for the whole subprocess: enough for many tests
        # to run, each with timeout_sec individually. Cap minimum at 10 min.
        wall_budget = max(timeout_sec * 30, 600)
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=wall_budget, cwd=REPO_ROOT)
        wallclock = time.time() - t0
        log_path.write_text(
            "=== STDOUT ===\n" + r.stdout +
            "\n=== STDERR ===\n" + r.stderr +
            f"\n=== exit: {r.returncode}, wallclock: {wallclock:.1f}s ===\n"
        )
        passed, failed, errors, skipped, _ = parse_summary(r.stdout)
        ok = (r.returncode == 0 and failed == 0 and errors == 0)
        status = "PASS" if ok else "FAIL"
        print(f"{status:5} ({passed}p, {failed}f, {errors}e, {skipped}s) {wallclock:.1f}s")
        return {
            "target": target, "passed": passed, "failed": failed,
            "errors": errors, "skipped": skipped, "wallclock": wallclock,
            "exit_code": r.returncode, "ok": ok, "log": str(log_path),
        }
    except subprocess.TimeoutExpired:
        wallclock = time.time() - t0
        log_path.write_text(f"TIMEOUT after {wallclock:.1f}s\n")
        print(f"TIMEOUT {wallclock:.1f}s")
        return {
            "target": target, "passed": 0, "failed": 0, "errors": 1,
            "skipped": 0, "wallclock": wallclock, "exit_code": -1,
            "ok": False, "log": str(log_path),
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timeout", type=int, default=120,
                   help="per-test pytest --timeout value in seconds (default: 120). "
                        "Each subprocess pays its own MiniLM cold-start (~30s on "
                        "Apple Silicon MPS), so 60s often trips on the first "
                        "test of a freshly-spawned subprocess.")
    p.add_argument("--files", nargs="+", default=None,
                   help="specific test files/dirs to run (default: all discovered)")
    p.add_argument("--skip-slow", action="store_true",
                   help="skip the slowest test files (langchain, llamaindex, async, core)")
    p.add_argument("--list", action="store_true",
                   help="print the discovered file list (in run order) and exit")
    p.add_argument("--log-dir", default="/tmp/sulci-test-runner",
                   help="where to save per-file log output")
    args = p.parse_args()

    if args.files:
        targets = args.files
    else:
        targets = discover_targets()

    if args.skip_slow:
        targets = [t for t in targets if t not in SLOW_FILES]

    if args.list:
        for t in targets:
            marker = "  [SLOW]" if t in SLOW_FILES else ""
            print(f"  {t}{marker}")
        return 0

    if not targets:
        print("ERROR: no test targets discovered. Is tests/ empty?", file=sys.stderr)
        return 2

    # Validate targets exist relative to repo root
    missing = [t for t in targets if not (REPO_ROOT / t.rstrip("/")).exists()]
    if missing:
        print(f"ERROR: missing test paths: {missing}", file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Sulci per-file test runner")
    print(f" timeout-per-file: {args.timeout}s   targets: {len(targets)}   logs: {log_dir}")
    print("=" * 70)

    results = []
    overall_t0 = time.time()
    for target in targets:
        results.append(run_one(target, args.timeout, log_dir))
    total_runtime = time.time() - overall_t0

    # Summary table
    print("\n" + "=" * 70)
    print(f" SUMMARY ({total_runtime:.1f}s total wall-clock)")
    print("=" * 70)
    print(f"  {'Target':<48} {'Result':<6}  {'p':>4} {'f':>3} {'e':>3} {'s':>4} {'time':>7}")
    print(f"  {'-'*48} {'-'*6}  {'-'*4} {'-'*3} {'-'*3} {'-'*4} {'-'*7}")
    for r in results:
        marker = "PASS" if r["ok"] else "FAIL"
        print(f"  {r['target']:<48} {marker:<6}  "
              f"{r['passed']:>4} {r['failed']:>3} {r['errors']:>3} "
              f"{r['skipped']:>4} {r['wallclock']:>6.1f}s")

    total_passed  = sum(r["passed"] for r in results)
    total_failed  = sum(r["failed"] for r in results)
    total_errors  = sum(r["errors"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    failing       = [r for r in results if not r["ok"]]

    print(f"\n  TOTAL: {total_passed} passed, {total_failed} failed, "
          f"{total_errors} errors, {total_skipped} skipped")
    if failing:
        print(f"\n  Failing files (logs in {log_dir}):")
        for r in failing:
            print(f"    - {r['target']}  →  {r['log']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
