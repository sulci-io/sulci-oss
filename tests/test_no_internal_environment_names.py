"""This repository is public, so it doesn't name the managed platform's staging service: that is for Sulci's own
testing only. Point a development run at a local gateway (``SULCI_GATEWAY=http://localhost:8000``) or a test gateway
Sulci has given you, and in tests use example hosts (``gateway.example.test``, ``gateway.example.com``).

A line that says "staging" fails this test unless it is about a *user's own* staging environment, listed in ALLOWED
with the file it is in. Only files git tracks are read."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = re.compile(r"staging", re.IGNORECASE)
BINARY = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".whl", ".npy", ".pkl", ".db", ".sqlite")
THIS = "tests/test_no_internal_environment_names.py"

# (file, exact text): a user's own staging environment, not Sulci's.
ALLOWED = (
    ("README.md", "internal staging where dev traffic should not pollute production"),
    ("docs/architecture/adrs/0001-cache-auto-connect-telemetry.md", "internal staging that shouldn't"),
    ("docs/architecture/adrs/0001-cache-auto-connect-telemetry.md", "Internal staging environments where dev traffic"),
)


def _tracked():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [n for n in out.stdout.decode("utf-8").split("\0") if n]


def test_nothing_names_sulci_s_staging_service():
    names = _tracked()
    if names is None:   # an sdist or another copy without git: nothing to check against
        import pytest
        pytest.skip("not a git checkout")
    hits = []
    for name in names:
        if name == THIS or name.lower().endswith(BINARY):
            continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if not STAGING.search(text):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            rest = line
            for path, allowed in ALLOWED:
                if name == path:
                    rest = rest.replace(allowed, "")
            if STAGING.search(rest):
                hits.append(f"{name}:{n}: {line.strip()[:160]}")
    assert not hits, (
        "Sulci's staging service is named (use a local or test gateway, or example hosts; a user's own staging "
        "goes in ALLOWED):\n" + "\n".join(hits)
    )
