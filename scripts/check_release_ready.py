#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
scripts/check_release_ready.py
───────────────────────────────
Fail if this tree is not safe to publish to PyPI.

WHY THIS EXISTS
───────────────
`pyproject.toml` names `README.md` as the PyPI long description, so **cutting a
release publishes that file to the most-read page in the estate.** On
2026-08-13, preparing 0.9.0, README.md still carried `85.9%`, `0.74ms`,
`$21.47` and `56.8% → 77.6%` — four figures the claims register retired on
08-12. The release intended to *fix* the PyPI copy of a retired table would
have simultaneously republished four more.

The retired figures were already documented as outstanding, in this repo's own
CHANGELOG, under a heading that begins "Outstanding". Reading it was not the
missing step. **Nothing failed when it was ignored** — that was.

Two checks, both cheap:

  1. Every place that restates the version agrees with pyproject.toml.
  2. No retired figure appears outside a retraction note.
  3. Every dependency in the `all` extra can install on the Python floor
     this package declares.

    python3 scripts/check_release_ready.py

Run it before tagging. `make checkin` does not — a release is not a check-in,
and gating every commit on this would train people to skip it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Figures the register retired. A hit OUTSIDE a retraction note is a failure.
#: Add here when a figure is retired; never remove without a register entry.
RETIRED = {
    "85.9%":   "TF-IDF stateless hit rate; use 99.9% recall / 93.8% precision",
    "0.74ms":  "TF-IDF hit latency; use 87.19ms [86.96-87.73] on MiniLM",
    "$21.47":  "derived from the retired hit rate",
    "77.6%":   "TF-IDF context resolution accuracy",
    "56.8%":   "TF-IDF stateless context baseline",
    "+56pp":   "per-domain gain table, withdrawn entirely",
    "+56 pp":  "per-domain gain table, withdrawn entirely",
    "2,486":   "latency multiple with no committed source",
    "3,000×":  "latency multiple derived from the retired 0.74ms",
}

#: A retraction may name a retired figure. Markers are matched against a
#: WINDOW of surrounding lines, not the line itself: a real retraction is a
#: paragraph, and its verb frequently lands one or two lines away from the
#: number. Line-local matching produced three false positives on the first
#: run of this script, which is the sort of noise that gets a checker
#: disabled rather than fixed.
RETRACTION_WINDOW = 4

RETRACTION_MARKERS = (
    "withdrawn", "retired", "no longer", "used to", "previously",
    "do not cite", "not the shipped engine", "was measured on",
)

#: Files that ship to a reader. README.md is the PyPI page; the two adapter
#: docstrings ship inside the wheel and are read by anyone who opens the source.
PUBLISHED = [
    "README.md",
    "sulci/integrations/langchain.py",
    "sulci/integrations/llamaindex.py",
]

#: Files that restate the version and must agree with pyproject.toml.
VERSION_SITES = [
    ("docs/API-SURFACE.md", r"`sulci`\s+\**([0-9]+\.[0-9]+\.[0-9]+)\**"),
]


#: Dependencies whose own requires-python floor exceeds nothing we support,
#: keyed to the floor they declare. Checked against `all` -- an entry here
#: that appears UNMARKED in `all` breaks `pip install "sulci[all]"` on any
#: interpreter below its floor.
#:
#: ⚠️ THIS TABLE IS HAND-MAINTAINED AND CANNOT SEE PyPI. It catches the case
#: we know about, fast and offline. The real test is the
#: `sulci[all] resolves on the declared Python floor` step in publish.yml,
#: which resolves against the actual index on the actual floor. **This is not
#: a substitute for that step.** If you add a dependency to `all`, add its
#: floor here too -- nothing detects the omission, which is the same defect
#: this whole class of checker exists to answer.
DEP_PYTHON_FLOOR = {
    "mcp":      "3.10",
    "litellm":  "3.10",
    "fastapi":  "3.10",
    "uvicorn":  "3.9",
}


def _ver_tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.split(".")[:2])


def check_all_extra_floor() -> list:
    """Fail if `all` carries a dep that cannot install on requires-python."""
    try:
        import tomllib
    except ModuleNotFoundError:          # py3.9/3.10
        return []
    from packaging.requirements import Requirement

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    proj = data.get("project", {})
    floor_spec = proj.get("requires-python", "")
    m = re.search(r">=\s*([0-9]+\.[0-9]+)", floor_spec)
    if not m:
        return []
    floor = m.group(1)

    bad = []
    for raw in proj.get("optional-dependencies", {}).get("all", []):
        req = Requirement(raw)
        dep_floor = DEP_PYTHON_FLOOR.get(req.name.lower())
        if not dep_floor or _ver_tuple(dep_floor) <= _ver_tuple(floor):
            continue
        # It needs a newer Python than we claim. A marker excusing it is fine.
        if req.marker and not req.marker.evaluate({"python_version": floor}):
            continue
        bad.append(
            f"all: {req.name} needs Python >={dep_floor}, "
            f"pyproject declares requires-python {floor_spec}\n"
            f'      add a marker: "{raw}; python_version >= \'{dep_floor}\'"'
        )
    return bad


def pyproject_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"',
                  (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    if not m:
        print("check-release-ready: no version in pyproject.toml", file=sys.stderr)
        raise SystemExit(2)
    return m.group(1)


def check_versions(ver: str) -> list:
    bad = []
    for rel, pattern in VERSION_SITES:
        p = ROOT / rel
        if not p.exists():
            continue
        for found in set(re.findall(pattern, p.read_text(encoding="utf-8"))):
            if found != ver:
                bad.append(f"{rel}: says {found}, pyproject.toml says {ver}")
    return bad


def check_retired() -> list:
    bad = []
    for rel in PUBLISHED:
        p = ROOT / rel
        if not p.exists():
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines, 1):
            lo = max(0, n - 1 - RETRACTION_WINDOW)
            hi = min(len(lines), n + RETRACTION_WINDOW)
            window = " ".join(lines[lo:hi]).lower()
            if any(m in window for m in RETRACTION_MARKERS):
                continue          # inside a retraction paragraph
            for fig, why in RETIRED.items():
                if fig in line:
                    bad.append(f"{rel}:{n}: {fig} — {why}\n      {line.strip()[:90]}")
    return bad


def main() -> int:
    ver = pyproject_version()
    versions = check_versions(ver)
    retired = check_retired()
    floors = check_all_extra_floor()

    if not versions and not retired and not floors:
        print(f"check-release-ready: OK — {ver}, "
              f"{len(PUBLISHED)} published files carry no retired figure, "
              f"`all` installs on the declared floor")
        return 0

    print("check-release-ready: FAIL\n")
    if versions:
        print("  Version disagreement:")
        for b in versions:
            print(f"    {b}")
        print()
    if floors:
        print("  `all` cannot install on the Python floor this package claims:\n")
        for b in floors:
            print(f"    {b}")
        print("\n  `pip install \"sulci[all]\"` is the headline install line. "
              "Breaking it on\n  the declared floor is a silent regression for "
              "the users least able\n  to diagnose it -- the failure happens in "
              "pip's resolver, before any\n  Sulci code runs.\n")

    if retired:
        print("  Retired figures in files that ship to a reader:\n")
        for b in retired:
            print(f"    {b}")
        print("\n  pyproject.toml makes README.md the PyPI project page. "
              "Publishing\n  from this tree puts these figures on the most-read "
              "page in the estate.\n  See sulci-platform docs/marketing/CLAIMS.md "
              "for replacements.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
