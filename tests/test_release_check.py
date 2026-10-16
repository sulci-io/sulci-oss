# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_release_check.py
───────────────────────────
Cover how `make release-check` reads the version docs/API-SURFACE.md was
measured against: `scripts/check_release_ready.py` and
`scripts/check_api_surface.py`, through their one reader,
`check_api_surface.measured_version` (0.10.0, oss-0027).

WHY THIS SUITE EXISTS
─────────────────────
The doc dates itself on lines starting `**Measured:**`, newest first, and keeps
the older ones as history. Measured on main (cb67e19) before oss-0027:

  * `make release-check` was red on a tree ready to release: check_release_ready.py
    compared EVERY `sulci` + version in the file, so the 0.9.2 line disagreed
    with pyproject.toml's 0.9.3 (it was red at v0.9.3 too).
  * Both readers took `x.y.z` only. With a new top line and pyproject.toml at
    `0.10.0rc1`, check_release_ready.py read 0.10.0, 0.9.3 and 0.9.2, and
    check_api_surface.py -- which CI runs on every pull request -- skipped the
    new line and compared the next one down (0.9.3): the rc1 preparation would
    have turned CI red.
  * check_api_surface.py skipped its version check when its pattern found
    nothing, so a header it could not read reported green.

The scripts are run as SUBPROCESSES on a copy of the tree, because the exit
code is the contract (see tests/test_tag_version.py). The copy's README.md is
replaced with a line carrying no figure, so these cases turn on the version
alone; the retired-figure check is not under test here.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from check_api_surface import measured_version  # noqa: E402

SCRIPTS = ("check_release_ready.py", "check_api_surface.py")
MEASURED = re.compile(r"^\*\*Measured:\*\*", re.M)       # the real doc's measurement lines


# ── the reader ───────────────────────────────────────────────────────────────
HEAD_093 = ("**Measured:** 2026-10-10 against `sulci` **0.9.3** — "
            "`check_api_surface.py --show`; the 0.9.3 additions are the\n")
OLDER_092 = "**Measured:** 2026-10-01 against `sulci` **0.9.2**\n"


def doc(*lines: str) -> str:
    return "# Sulci — the measured public API surface\n\n" + "\n".join(lines)


def test_a_release_version_is_read():
    assert measured_version(doc(HEAD_093, OLDER_092)) == "0.9.3"


@pytest.mark.parametrize("v", ["0.10.0rc1", "0.10.0a1", "0.10.0b2", "1.0.0rc12"])
def test_a_pre_release_is_read_whole(v):
    head = f"**Measured:** 2026-10-16 against `sulci` **{v}** — `check_api_surface.py --show`\n"
    assert measured_version(doc(head, HEAD_093, OLDER_092)) == v


def test_only_the_newest_measurement_is_read():
    """The first `**Measured:**` line decides; the 0.9.2 line below is history."""
    assert measured_version(doc(OLDER_092, HEAD_093)) == "0.9.2"
    assert measured_version(doc(HEAD_093, OLDER_092)) == "0.9.3"


@pytest.mark.parametrize("head", [
    "**Measured:** 2026-10-16 against `sulci` 0.10.0rc1\n",          # not bolded
    "**Measured:** 2026-10-16 against `sulci` **0.10.0-rc1**\n",     # not how pyproject spells it
    "**Measured:** 2026-10-16 against `sulci` **0.10.0.dev1**\n",    # not a release or pre-release
    "**Measured:** 2026-10-16 against `sulci` **0.10**\n",           # not x.y.z
    "**Measured:** 2026-10-16, version to follow\n",
    "**Measured:** 2026-10-16 against **0.10.0rc1**\n",                # no `sulci` before it
])
def test_an_unreadable_newest_measurement_does_not_fall_back_to_an_older_one(head):
    """Falling through to the 0.9.3 line would compare history and could pass."""
    assert measured_version(doc(head, HEAD_093, OLDER_092)) is None


def test_no_measurement_line_reads_none():
    assert measured_version(doc("Some text naming `sulci` **0.9.3**.\n")) is None


UNSEEN = {
    "code span": "`check_api_surface.py` anchors on `**Measured:**` and `sulci` **99.0.0rc1** here\n",
    "code span, space before the marker": "See `` **Measured:** `sulci` **99.0.0rc1** `` below.\n",
    "code span, marker inside": "a `line with **Measured:** and `sulci` **99.0.0rc1**` shape\n",
    "HTML comment": "<!-- **Measured:** draft against `sulci` **99.0.0rc1** -->\n",
    "HTML comment, two lines": "<!--\n**Measured:** draft against `sulci` **99.0.0rc1**\n-->\n",
    "fenced block": "```\n**Measured:** 2026-10-16 against `sulci` **99.0.0rc1**\n```\n",
    "tilde fence": "~~~md\n**Measured:** 2026-10-16 against `sulci` **99.0.0rc1**\n~~~\n",
}


UNSEEN_BY_STRIPPING_ONLY_IN_PART = {      # the backstop's cases: Markdown hides them, the removal does not
    "indented code block": "    **Measured:** 2026-10-16 against `sulci` **99.0.0rc1**\n\n",
    "unclosed fence": "```\n**Measured:** 2026-10-16 against `sulci` **99.0.0rc1**\n\n",
    "code span across lines": "`a span\n**Measured:** 2026-10-16 against `sulci` **99.0.0rc1**` end\n",
    "fence, then a line with an info string": "```\n```python\n**Measured:** 2026-10-16 against `sulci` **99.0.0rc1**\n",
}


@pytest.mark.parametrize("unseen", list(UNSEEN) + list(UNSEEN_BY_STRIPPING_ONLY_IN_PART))
def test_a_marker_no_reader_sees_naming_another_version_fails_closed(unseen):
    """Read, it would decide the version while the doc shows 0.9.3 (review of oss-0027): refused, never green."""
    text = {**UNSEEN, **UNSEEN_BY_STRIPPING_ONLY_IN_PART}[unseen]
    assert measured_version(doc(text, HEAD_093, OLDER_092)) is None


@pytest.mark.parametrize("unseen", list(UNSEEN))
def test_a_marker_no_reader_sees_naming_the_same_version_is_harmless(unseen):
    text = UNSEEN[unseen].replace("99.0.0rc1", "0.9.3")
    assert measured_version(doc(text, HEAD_093, OLDER_092)) == "0.9.3"


def test_a_marker_named_in_prose_with_no_version_is_ignored():
    assert measured_version(doc("Notes name `**Measured:**`, no version.\n", HEAD_093)) == "0.9.3"


@pytest.mark.parametrize("prefix", ["> ", "- ", "* ", "## ", "   ", "Re-measured. "])
def test_a_newest_measurement_not_at_a_line_start_is_still_the_newest(prefix):
    """Skipping it would compare the 0.9.3 line below and could report green (review of oss-0027)."""
    head = f"{prefix}**Measured:** 2026-10-16 against `sulci` **0.10.0rc1**\n"
    assert measured_version(doc(head, HEAD_093, OLDER_092)) == "0.10.0rc1"


def test_the_version_is_the_one_after_sulci():
    head = "**Measured:** 2026-10-16 — the **0.9.3** rows re-read; against `sulci` **0.10.0**\n"
    assert measured_version(doc(head, HEAD_093)) == "0.10.0"


def test_the_real_doc_s_newest_measurement_is_readable():
    text = (ROOT / "docs" / "API-SURFACE.md").read_text(encoding="utf-8")
    assert measured_version(text) is not None


# ── both scripts, end to end, on a copy of the tree ──────────────────────────
@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """What the two scripts read: scripts/, pyproject.toml, docs/API-SURFACE.md
    and the sulci/ sources; README.md replaced by a line carrying no figure."""
    t = tmp_path / "tree"
    shutil.copytree(ROOT / "scripts", t / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "sulci", t / "sulci",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (t / "docs").mkdir()
    shutil.copy2(ROOT / "docs" / "API-SURFACE.md", t / "docs" / "API-SURFACE.md")
    shutil.copy2(ROOT / "pyproject.toml", t / "pyproject.toml")
    (t / "README.md").write_text("# sulci (test copy)\n", encoding="utf-8")
    return t


def run(tree: Path, script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(tree / "scripts" / script)],
                          capture_output=True, text=True, cwd=tree)


def set_version(tree: Path, v: str) -> None:
    p = tree / "pyproject.toml"
    s, n = re.subn(r'^(version\s*=\s*)"[^"]+"', rf'\g<1>"{v}"', p.read_text(encoding="utf-8"),
                   count=1, flags=re.M)
    assert n == 1
    p.write_text(s, encoding="utf-8")


def set_newest_measurement(tree: Path, bolded: str) -> None:
    """Rewrite the bolded version on the first `**Measured:**` line only."""
    p = tree / "docs" / "API-SURFACE.md"
    s = p.read_text(encoding="utf-8")
    m = MEASURED.search(s)
    assert m
    end = s.index("\n", m.start())
    line, n = re.subn(r"\*\*[0-9][^*]*\*\*", f"**{bolded}**", s[m.start():end], count=1)
    assert n == 1
    p.write_text(s[:m.start()] + line + s[end:], encoding="utf-8")


def older_versions(tree: Path) -> set:
    s = (tree / "docs" / "API-SURFACE.md").read_text(encoding="utf-8")
    lines = [s[m.start():s.index("\n", m.start())] for m in MEASURED.finditer(s)]
    return {measured_version(l + "\n") for l in lines[1:]}


@pytest.mark.parametrize("script", SCRIPTS)
def test_an_older_measurement_does_not_turn_the_check_red(tree, script):
    """The tree as committed: the newest line agrees, an older one names another version."""
    newest = measured_version((tree / "docs" / "API-SURFACE.md").read_text(encoding="utf-8"))
    assert older_versions(tree) - {newest, None}, "the doc keeps no older measurement: this case proves nothing"
    set_version(tree, newest)
    r = run(tree, script)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_quoted_newest_measurement_is_not_skipped(tree, script):
    """The false green the review found: a newest line that does not start its line, pyproject at the old version."""
    p = tree / "docs" / "API-SURFACE.md"
    s = p.read_text(encoding="utf-8")
    m = MEASURED.search(s)
    p.write_text(s[:m.start()] + "> **Measured:** 2026-10-16 against `sulci` **99.0.0rc1**\n\n" + s[m.start():],
                 encoding="utf-8")
    set_version(tree, measured_version(s))       # the version the line below it names: green if it were skipped
    r = run(tree, script)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "99.0.0rc1" in r.stdout, r.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_measurement_in_an_html_comment_does_not_count(tree, script):
    """The other false green the review found: pyproject at the version only a hidden line names."""
    p = tree / "docs" / "API-SURFACE.md"
    s = p.read_text(encoding="utf-8")
    m = MEASURED.search(s)
    p.write_text(s[:m.start()] + "<!-- **Measured:** draft against `sulci` **99.0.0rc1** -->\n\n" + s[m.start():],
                 encoding="utf-8")
    set_version(tree, "99.0.0rc1")
    r = run(tree, script)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "text no reader sees names another" in r.stdout, r.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_pre_release_that_agrees_passes(tree, script):
    set_version(tree, "0.10.0rc1")
    set_newest_measurement(tree, "0.10.0rc1")
    r = run(tree, script)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_pre_release_is_not_its_final_version(tree, script):
    set_version(tree, "0.10.0rc1")
    set_newest_measurement(tree, "0.10.0")
    r = run(tree, script)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "0.10.0rc1" in r.stdout and "against 0.10.0," in r.stdout, r.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_an_unreadable_newest_measurement_fails(tree, script):
    """Not a pass: before oss-0027, check_api_surface.py skipped the check and reported green."""
    set_version(tree, "0.10.0rc1")
    set_newest_measurement(tree, "0.10.0-rc1")
    r = run(tree, script)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "names no version this can read" in r.stdout, r.stdout
