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
