# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""oss-0056: .github/workflows/release-gate.yml runs publish.yml's test job on every pull request.

The point is that a PR fails where a release tag would (v1.3.0's first publish run failed on 3.9 after green PR
checks). That only holds while the two jobs run the same steps, so this compares them as text: the `steps:` block of
publish.yml's `test` job and of release-gate.yml's `test` job must be identical. No YAML library needed.
"""
from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _job(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"^  {name}:\n(.*?)(?=^  \S|\Z)", text, re.S | re.M)
    assert m, f"no `{name}:` job in {path.name}"
    return m.group(1)


def _steps(job: str) -> str:
    i = job.find("    steps:\n")
    assert i >= 0, "job has no steps"
    return job[i:].rstrip("\n")   # a trailing blank line before the next job is not a step


def test_the_gate_runs_publish_yml_test_steps_byte_for_byte():
    publish = _steps(_job(WORKFLOWS / "publish.yml", "test"))
    gate = _steps(_job(WORKFLOWS / "release-gate.yml", "test"))
    assert gate == publish, "release-gate.yml's steps differ from publish.yml's test job — change both together"


def test_the_gate_runs_on_pull_requests_on_the_same_matrix_without_extra_pytest_options():
    text = (WORKFLOWS / "release-gate.yml").read_text(encoding="utf-8")
    assert re.search(r"^  pull_request:\n    branches: \[main\]", text, re.M)
    pub_matrix = re.search(r"python-version: (\[.*?\])", _job(WORKFLOWS / "publish.yml", "test")).group(1)
    assert f"python-version: {pub_matrix}" in _job(WORKFLOWS / "release-gate.yml", "test")
    assert not re.search(r"^\s+PYTEST_ADDOPTS:", text, re.M)   # publish.yml sets none; tests.yml's --timeout would differ


def test_publish_yml_still_runs_the_whole_suite():
    assert "pytest tests/ sulci/tests/" in _steps(_job(WORKFLOWS / "publish.yml", "test"))
