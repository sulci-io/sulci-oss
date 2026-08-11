"""
tests/conftest.py
─────────────────
Session-wide test isolation.

Disables telemetry for the entire SDK test suite by removing any
``SULCI_API_KEY`` from the environment at conftest load time. Without
this guard, a developer with ``SULCI_API_KEY`` exported in their shell
(common while iterating on the SDK) would have every test call to
``sulci.connect()`` that doesn\'t pass an explicit ``api_key=`` argument
silently emit telemetry to the production gateway — polluting that
account\'s request count and showing up on ``dashboard.sulci.io`` as
"deployment" rows that are really test runs.

Tests that explicitly want telemetry enabled — e.g. the
``test_telemetry_lifecycle.py`` atexit-hook regression suite added in
v0.6.4 — pass ``api_key="sk-sulci-test-key"`` directly to ``connect()``;
they\'re unaffected by this fixture because the SDK\'s key-resolution
order (explicit arg → env var → ~/.sulci/config) uses the explicit
argument first.

This file lives at the top of ``tests/`` so pytest auto-discovers it
before any test module imports ``sulci``. The env-var removal must
happen *before* sulci is imported by any test — once a test module
loads ``sulci`` with ``SULCI_API_KEY`` in env, the module-level state
can already be tainted by any module-level ``sulci.connect()`` call.

History
───────
Added in response to the v0.6.4 debugging session that surfaced the
"104-requests-from-nowhere" mystery: with v0.6.4\'s atexit hook now
reliably flushing telemetry on process exit, any pytest invocation
with a real ``SULCI_API_KEY`` in env was silently delivering events
to production with every test run that called ``sulci.connect()`` —
where pre-v0.6.4 those events had been lost when the daemon flush
thread died at process exit. The atexit fix didn\'t create the leak;
it made the existing leak visible.
"""
import os

# Remove production-key leak vector for the entire test session.
# Tests that need real telemetry must pass api_key= explicitly.
os.environ.pop("SULCI_API_KEY", None)


# ── shared fixtures for the v0.9.0 integration suites ────────────────────
#
# WHY THIS IS A FIXTURE AND NOT AN IMPORT
#
# tests/test_integrations_{mcp,litellm}.py and tests/test_proxy.py originally
# did `from tests._fake_embedder import FakeEmbedder`. That works under
# `python -m pytest` -- which is what the Makefile uses -- because `-m`
# inserts the CWD into sys.path, making the repo root importable as a package
# root. It does NOT work under a bare `pytest`, which is what
# .github/workflows/tests.yml uses: pytest inserts the TEST FILE'S directory
# (tests/), not the repo root, so `tests` is not a module and collection dies
# with ModuleNotFoundError before a single test runs.
#
# Local green, CI red, on identical code. Measured 2026-08-11 by reproducing
# with `PYTHONPATH=<sulci-only> pytest <abs path>`.
#
# conftest.py is loaded by pytest itself regardless of sys.path, so a fixture
# defined here is reachable from every test module with no import at all.
# _fake_embedder.py stays as the documented home of the class; only the route
# to it changes.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).parent))

import pytest as _pytest  # noqa: E402
from _fake_embedder import FakeEmbedder as _FakeEmbedder  # noqa: E402

FakeEmbedder = _FakeEmbedder


@_pytest.fixture
def fake_embedder():
    """Deterministic, offline Embedder. See tests/_fake_embedder.py."""
    return _FakeEmbedder()
