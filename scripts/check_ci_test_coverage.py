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

#: Directories searched for suites, DECLARED rather than discovered.
#:
#: Until 2026-08-20 this was a single `TESTS.glob("*.py")` -- one level, top
#: level only -- and ELEVEN files lived outside it: tests/compat (2),
#: tests/integration/flows (1 suite + 6 flow fixtures) and sulci/tests/compat
#: (2). All were in fact run, by the directory-level steps at tests.yml:220
#: and :228, so nothing was unexecuted. What was missing is the ENFORCEMENT
#: that they stay wired -- which is the exact condition that let the
#: 2026-08-11 break through. See #135.
#:
#: Declared, not recursive-from-root, and that distinction is load-bearing.
#: sulci-platform's build_figures.py v1 globbed one level and found nothing;
#: v2 recursed and silently pooled seven directories including three
#: alpha-sweep runs, measuring recall 0.9300 where the four declared draws
#: give 0.9990. v3 made the declared set binding. This is v3's lesson, not
#: v2's: add a directory here on purpose or it is not searched.
SEARCH_ROOTS = ("tests", "sulci/tests")

#: Files that are not suites, or are run by a directory-level step.
NOT_A_SUITE = {"conftest.py", "_fake_embedder.py"}

#: Suites intentionally outside CI. Each needs a reason, and the reason has to
#: be about the SUITE -- "it is slow" is one, "nobody got round to it" is not.
DELIBERATELY_UNCOVERED: dict = {
    # e.g. "test_something.py": "needs a live gateway; covered by staging",
}


def referenced_in_workflow() -> set:
    """Relative paths of every suite some CI step runs.

    Two ways a step can cover a suite, and the second is why this is not just
    a filename match:

      * NAMED -- `pytest tests/test_core.py`
      * DIRECTORY-LEVEL -- `pytest tests/compat/ sulci/tests/compat/`, which
        covers everything beneath those directories including files added
        later. tests.yml:220 and :228 are both of this kind.

    Matching on bare filenames, as this did before 2026-08-20, also cannot
    tell tests/compat/test_x.py from sulci/tests/compat/test_x.py. Paths can.
    """
    if not WORKFLOW.exists():
        print(f"check-ci-test-coverage: {WORKFLOW} not found", file=sys.stderr)
        raise SystemExit(2)
    text = WORKFLOW.read_text(encoding="utf-8")

    covered = set(re.findall(r"((?:sulci/)?tests/[A-Za-z0-9_/]*test_[A-Za-z0-9_]+\.py)", text))

    # Directory-level steps. TOKENISE the command rather than matching a
    # pattern across it: an expression like
    #     pytest\\s+((?:[\\w./]+\\s+)*[\\w./]+/)
    # backtracks and captures `tests/` -- the PREFIX of the last filename --
    # out of `pytest tests/a.py tests/b.py tests/c.py`, making five ordinary
    # per-file steps each read as `pytest tests/` and every suite look
    # covered. Measured 2026-08-20 while writing this fix, which is the same
    # over-broad-matcher-reports-success shape the script exists to catch.
    for line in text.splitlines():
        m = re.search(r"\bpytest\s+(.*)$", line)
        if not m:
            continue
        for tok in m.group(1).split():
            if not tok.endswith("/"):
                continue
            root = ROOT / tok.rstrip("/")
            if root.is_dir():
                covered |= {
                    str(f.relative_to(ROOT))
                    for f in root.rglob("test_*.py")
                    if f.name not in NOT_A_SUITE
                }
    return covered


def suites_on_disk() -> set:
    found = set()
    for rel in SEARCH_ROOTS:
        root = ROOT / rel
        if not root.is_dir():
            print(f"check-ci-test-coverage: declared root {rel}/ does not "
                  f"exist. That is a finding, not a skip.", file=sys.stderr)
            raise SystemExit(2)
        found |= {
            str(f.relative_to(ROOT))
            for f in root.rglob("test_*.py")
            if f.name not in NOT_A_SUITE
        }
    return found


def main() -> int:
    on_disk = suites_on_disk()
    covered = referenced_in_workflow()
    orphans = sorted(on_disk - covered - set(DELIBERATELY_UNCOVERED))

    if "--list" in sys.argv:
        for name in orphans:
            print(name)
        return 0

    if not orphans:
        print(
            f"check-ci-test-coverage: OK -- all {len(on_disk)} suites under "
            f"{', '.join(r + '/' for r in SEARCH_ROOTS)} are run by "
            f".github/workflows/tests.yml"
        )
        print(
            "  Searched the DECLARED roots above, recursively. A suite in a "
            "directory\n  not listed in SEARCH_ROOTS is not searched and this "
            "pass says nothing\n  about it."
        )
        return 0

    print("check-ci-test-coverage: FAIL\n")
    print(f"  {len(orphans)} suite(s) exist on disk that NO CI step runs:\n")
    for name in orphans:
        print(f"    {name}")
    print(
        "\n  A green matrix that never executes a suite is worse than a red "
        "one:\n  it reports confidence it has not earned. Either add a step to"
        "\n  .github/workflows/tests.yml, or add the file to"
        "\n  DELIBERATELY_UNCOVERED in this script with a reason.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
