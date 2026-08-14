# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
tests/test_tag_version.py
─────────────────────────
Cover `scripts/check_tag_version.py`, the release-guard.

WHY THIS SUITE EXISTS
─────────────────────
The guard was added after `v0.9.0` was first pushed at a commit whose
`pyproject.toml` said `0.8.3`. Its three exits were each watched at a prompt
on the day it landed, and **watching a thing work once is what this repo's
§9 pattern is made of** -- `run_examples.py` and `tests.yml` both went wrong
for exactly that reason. A guard with no test is a hand-kept assertion.

📌 The third case is the one that matters most and is easiest to leave out:
**exit 2 must not be treated as a pass.** A checker that cannot read the value
it checks and returns green converts an open question into a false answer --
the `scrollWidth` family, four instances and counting. `test_exit_2_*` below
is that lesson as an assertion.

The script is invoked as a SUBPROCESS rather than imported, because the exit
code is the contract. `main()` returning 2 and the process exiting 0 would
pass an import-level test and fail in CI, which is the gap the whole family
lives in.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_tag_version.py"


def run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Invoke the script as a process. The exit code is the contract."""
    import os
    e = dict(os.environ)
    # Never inherit a real ref name from the CI runner: on a tag push
    # GITHUB_REF_NAME is set, and it would silently satisfy the no-argument
    # cases below and turn them green for the wrong reason.
    e.pop("GITHUB_REF_NAME", None)
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, env=e,
    )


def pyproject_version() -> str:
    m = re.search(r"^version\s*=\s*[\"']([^\"']+)[\"']",
                  (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
                  flags=re.M)
    assert m, "pyproject.toml has no [project] version line"
    return m.group(1)


def test_script_exists():
    assert SCRIPT.is_file(), f"{SCRIPT} is missing"


# ── exit 0: the tag matches ──────────────────────────────────────────────────

def test_matching_tag_with_v_prefix_passes():
    r = run(f"v{pyproject_version()}")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_matching_tag_without_v_prefix_passes():
    """`0.9.0` and `v0.9.0` are the same tag for this purpose."""
    r = run(pyproject_version())
    assert r.returncode == 0, r.stdout + r.stderr


def test_surrounding_whitespace_is_tolerated():
    """GITHUB_REF_NAME arrives via shell interpolation; a stray newline must
    not read as a mismatch and block a legitimate release."""
    r = run(f"  v{pyproject_version()}  ")
    assert r.returncode == 0, r.stdout + r.stderr


def test_reads_github_ref_name_when_no_argument_given():
    r = run(env={"GITHUB_REF_NAME": f"v{pyproject_version()}"})
    assert r.returncode == 0, r.stdout + r.stderr


def test_explicit_argument_beats_the_environment():
    """The env var is the CI path; an argument is a human checking a specific
    tag. The argument must win, or `make check-tag TAG=...` is a no-op on a
    runner."""
    r = run("v0.0.1", env={"GITHUB_REF_NAME": f"v{pyproject_version()}"})
    assert r.returncode == 1, r.stdout + r.stderr


# ── exit 1: the near-miss ────────────────────────────────────────────────────

def test_the_v090_at_083_near_miss_fails():
    """The exact shape of the incident this guard exists for."""
    r = run("v0.8.3")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAIL" in r.stdout
    assert "0.8.3" in r.stdout and pyproject_version() in r.stdout


def test_failure_names_both_values():
    """A guard that says only 'mismatch' sends the reader back to the tree.
    Both values must be in the output or the message is not actionable."""
    r = run("v1.2.3")
    assert r.returncode == 1
    assert "1.2.3" in r.stdout
    assert pyproject_version() in r.stdout
    assert "git tag -d" in r.stdout, "no remediation offered"


@pytest.mark.parametrize("tag", ["v0.9", "0.9.0.0", "release-0.9.0", "latest"])
def test_malformed_tags_fail_rather_than_being_repaired(tag):
    """normalise() strips a leading `v` and nothing else, on purpose. A tag
    this script cannot parse must fail loudly, not be coerced into something
    that happens to match."""
    r = run(tag)
    assert r.returncode == 1, f"{tag!r} was accepted: {r.stdout}"


# ── exit 2: CANNOT READ, and it is NOT a pass ────────────────────────────────

def test_exit_2_when_no_tag_anywhere():
    r = run()
    assert r.returncode == 2, r.stdout + r.stderr
    assert "CANNOT READ" in r.stdout


def test_exit_2_when_ref_name_is_empty():
    r = run(env={"GITHUB_REF_NAME": ""})
    assert r.returncode == 2, r.stdout + r.stderr


def test_exit_2_when_argument_is_whitespace_only():
    r = run("   ")
    assert r.returncode == 2, r.stdout + r.stderr


def test_exit_2_is_not_zero():
    """Stated as its own assertion because it is the whole point.

    A CI step gates on a non-zero exit. If 'I could not read the version'
    ever returns 0, the guard reports a pass on the one occasion it has
    nothing to say -- which is worse than not existing, because it converts
    an open question into a false answer. Four instances of that family are
    recorded in SULCI-PROJECT-DOC.md §0.
    """
    assert run().returncode != 0
    assert run(env={"GITHUB_REF_NAME": ""}).returncode != 0


def test_exit_2_says_it_is_not_a_pass_in_the_output():
    """The reader of a red run is a person scrolling. The distinction between
    'mismatch' and 'could not read' has to be legible without the exit code."""
    r = run()
    assert "not a pass" in r.stdout.lower()


# ── the script's own independence ────────────────────────────────────────────

def test_script_imports_nothing_outside_the_stdlib():
    """It runs on a bare runner before `pip install`, as the first job in
    publish.yml. A third-party import would make the guard fail exactly when
    the environment is broken -- the case it most needs to survive."""
    src = SCRIPT.read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:import|from)\s+([a-zA-Z_][\w.]*)",
                              src, flags=re.M))
    allowed = {"__future__", "os", "re", "sys", "pathlib"}
    assert imported <= allowed, f"non-stdlib imports: {imported - allowed}"


def test_release_guard_runs_before_the_matrix_in_publish_yml():
    """The guard is worth seconds only if it gates the ~20-minute matrix.
    If `test` ever stops depending on it, it still runs -- in parallel,
    reporting after the thing it was meant to prevent has already started.

    ⚠️ This asserts a NAME in a file. It cannot tell you the job ran. See
    §0's 2026-08-12 entry: check_ci_test_coverage.py reported OK while
    tests.yml never fired at all.
    """
    wf = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8")
    assert "release-guard:" in wf, "release-guard job is gone"
    assert "check_tag_version.py" in wf, "the job no longer runs the script"
    assert re.search(r"^\s*test:\s*\n\s*needs:\s*release-guard\s*$",
                     wf, flags=re.M), "test no longer needs release-guard"
