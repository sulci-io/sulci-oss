#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
scripts/check_tag_version.py
────────────────────────────
Fail if the tag being built does not match `pyproject.toml` at the tagged
commit.

WHY THIS EXISTS
───────────────
`v0.9.0` was first pushed at a commit whose `pyproject.toml` said `0.8.3`.
Three things were true at once and none of them failed:

  * the release prompt printed `📦 v0.8.3`, which is the correct value for
    that tree and the wrong value for that tag;
  * branch protection fired specifically on tag creation, and `--admin`
    went through;
  * the ~20-minute matrix in `publish.yml` was already running, because
    nothing ahead of it looks at the tag name at all.

It was caught by a person watching the run. **The information existed and
nothing failed on it** — the same shape `check_release_ready.py` was written
for, one step further down the pipeline.

WHAT THIS IS NOT
────────────────
This is not `check_release_ready.py` and does not replace it. That script
answers *"is this tree safe to publish?"* — versions agree internally,
no retired figure ships, `all` resolves on the floor. This one answers a
strictly narrower question that only exists once a tag is in play:

    does the name on the tag match the version in the tree it points at?

`check_release_ready.py` cannot answer it. Run from a clean 0.8.3 tree it
returns OK, correctly, because every internal restatement of the version
agrees — with each other and with `pyproject.toml`. The tag is not one of
the places it reads, and it has no way to know a tag exists.

📌 Note what that means for the near-miss: **the release gate was green and
right.** Adding a sixth VERSION restatement would not have caught this.
The missing reading was the ref name.

USAGE
─────
    python3 scripts/check_tag_version.py v0.9.0
    python3 scripts/check_tag_version.py                # reads $GITHUB_REF_NAME

Exit 0 on match, 1 on mismatch, 2 if it could not read one of the two
values. **Exit 2 is not a pass.** A checker that cannot see the thing it
checks must say so rather than return green — see SULCI-PROJECT-DOC.md §0,
the `scrollWidth` entry and its two successors.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pyproject_version() -> str | None:
    """The `version = "..."` under [project], read positionally.

    Deliberately not a full TOML parse: this runs before dependencies are
    installed, on a bare runner, and `tomllib` is 3.11+ while this package
    declares a 3.9 floor. The regex is anchored to a line start so it cannot
    match a version pin inside a dependency string.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"^version\s*=\s*[\"']([^\"']+)[\"']", text, flags=re.M)
    return m.group(1) if m else None


def normalise(tag: str) -> str:
    """`v0.9.0` -> `0.9.0`. Leaves anything else alone so a malformed tag
    fails loudly on comparison rather than being silently repaired."""
    return tag[1:] if tag.startswith("v") else tag


def main(argv: list[str]) -> int:
    tag = argv[1] if len(argv) > 1 else os.environ.get("GITHUB_REF_NAME", "")
    tag = tag.strip()

    if not tag:
        print("check-tag-version: CANNOT READ — no tag given and "
              "$GITHUB_REF_NAME is unset.")
        print("  This is exit 2, not a pass. Pass the tag explicitly:")
        print("    python3 scripts/check_tag_version.py v0.9.0")
        return 2

    ver = pyproject_version()
    if ver is None:
        print("check-tag-version: CANNOT READ — no `version = \"...\"` line "
              "found in pyproject.toml.")
        print("  This is exit 2, not a pass.")
        return 2

    want = normalise(tag)
    if want == ver:
        print(f"check-tag-version: OK — tag {tag} matches "
              f"pyproject.toml version {ver}")
        return 0

    print("check-tag-version: FAIL\n")
    print(f"    tag being built     {tag}   (-> {want})")
    print(f"    pyproject.toml      {ver}")
    print()
    print("  The tag points at a commit whose declared version is different.")
    print("  This is the v0.9.0-at-0.8.3 near-miss. Nothing else in the")
    print("  pipeline reads the tag name, so nothing else will stop it.\n")
    print("  To fix: delete the tag, land the version bump, re-tag.\n")
    print("    git tag -d %s && git push --delete origin %s" % (tag, tag))
    print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
