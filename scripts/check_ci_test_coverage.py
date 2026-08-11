#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
scripts/check_ci_test_coverage.py
──────────────────────────────────
Fail if a test file exists on disk but no CI step runs it.

WHY THIS EXISTS
───────────────
`.github/workflows/tests.yml` names every test file it runs. That list is
maintained by hand, next to nothing that checks it, and it has already gone
wrong twice in one week:

  * 2026-08-11 — the three v0.9.0 suites (77 tests) were not in the list.
    All 12 matrix jobs reported SUCCESS while running none of them, and a
    merge landed on main on that basis.
  * The same audit found ELEVEN more suites that `make checkin` runs and CI
    never has, including all four telemetry suites and test_connect (59
    tests). Those had been invisible for months.

This is the same shape as `scripts/run_examples.py` omitting new examples and
`sulci-web`'s README hardcoding a script count over a directory: a hand-kept
list of what to run, sitting beside the thing it is meant to cover, with
nothing detecting the omission. The fix in each case is to make the omission
loud.

    python3 scripts/check_ci_test_coverage.py          # check
    python3 scripts/check_ci_test_coverage.py --list   # print the orphans

Add a suite to the workflow, or add it to DELIBERATELY_UNCOVERED below with a
reason. Silence is not an option this script offers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
TESTS = ROOT / "tests"

#: Files that are not suites, or are run by a directory-level step.
NOT_A_SUITE = {"conftest.py", "_fake_embedder.py"}

#: Suites intentionally outside CI. Each needs a reason, and the reason has to
#: be about the SUITE -- "it is slow" is one, "nobody got round to it" is not.
DELIBERATELY_UNCOVERED: dict = {
    # e.g. "test_something.py": "needs a live gateway; covered by staging",
}


def referenced_in_workflow() -> set:
    if not WORKFLOW.exists():
        print(f"check-ci-test-coverage: {WORKFLOW} not found", file=sys.stderr)
        raise SystemExit(2)
    text = WORKFLOW.read_text(encoding="utf-8")
    named = set(re.findall(r"tests/(test_[A-Za-z0-9_]+\.py)", text))
    # A directory-level step such as `pytest tests/` covers everything.
    if re.search(r"pytest\s+tests/\s", text) or re.search(r"pytest\s+tests/$", text, re.M):
        return {p.name for p in TESTS.glob("test_*.py")}
    return named


def main() -> int:
    on_disk = {
        p.name for p in TESTS.glob("*.py") if p.name not in NOT_A_SUITE
    }
    on_disk = {n for n in on_disk if n.startswith("test_")}
    covered = referenced_in_workflow()
    orphans = sorted(on_disk - covered - set(DELIBERATELY_UNCOVERED))

    if "--list" in sys.argv:
        for name in orphans:
            print(f"tests/{name}")
        return 0

    if not orphans:
        print(
            f"check-ci-test-coverage: OK -- all {len(on_disk)} suites in "
            f"tests/ are run by .github/workflows/tests.yml"
        )
        return 0

    print("check-ci-test-coverage: FAIL\n")
    print(f"  {len(orphans)} suite(s) exist on disk that NO CI step runs:\n")
    for name in orphans:
        print(f"    tests/{name}")
    print(
        "\n  A green matrix that never executes a suite is worse than a red "
        "one:\n  it reports confidence it has not earned. Either add a step to"
        "\n  .github/workflows/tests.yml, or add the file to"
        "\n  DELIBERATELY_UNCOVERED in this script with a reason.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
