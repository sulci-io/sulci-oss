# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
tests/compat/test_backend_conformance.py
=========================================
Conformance tests for the Backend protocol.

Tests are organized into three groups, each with different infrastructure
requirements:

  TestStructural   - inspect.signature checks; no instance needed.
                     Always runs for every registered backend class.

  TestRoundTrip    - store + search + clear with real data.
                     Skips if the backend can't be constructed locally.

  TestTenantIsolation - cross-tenant search returns no leak.
                     Runs only on backends with ENFORCES_TENANT_ISOLATION = True.
"""
from __future__ import annotations
import inspect
from typing import Any

import pytest

from sulci.backends import Backend


# =============================================================================
# Structural — runs against every registered backend class, no instance needed.
# =============================================================================

class TestStructural:
    """Signature/protocol contract — works on the class, no infra needed."""

    REQUIRED_METHODS = ("store", "search", "clear")

    def test_isinstance_backend(self, backend_class):
        # runtime_checkable only verifies method names exist
        instance_proxy = type("X", (backend_class,), {})
        # A bare class isn't an instance; Protocol structural check happens on
        # objects, but for class-level we use the safer inspect-based approach:
        for name in self.REQUIRED_METHODS:
            assert hasattr(backend_class, name), \
                f"{backend_class.__name__} missing required method {name!r}"

    def test_class_attribute_enforces_tenant_isolation(self, backend_class):
        """All registered backends must declare ENFORCES_TENANT_ISOLATION."""
        assert hasattr(backend_class, "ENFORCES_TENANT_ISOLATION"), (
            f"{backend_class.__name__} must declare ENFORCES_TENANT_ISOLATION "
            "as a class attribute (True or False)."
        )
        assert isinstance(backend_class.ENFORCES_TENANT_ISOLATION, bool)

    def test_store_signature_required_params(self, backend_class):
        sig = inspect.signature(backend_class.store)
        params = sig.parameters

        for required in ("self", "key", "query", "response", "embedding"):
            assert required in params, (
                f"{backend_class.__name__}.store() missing required "
                f"parameter {required!r}"
            )

    def test_store_signature_keyword_only_kwargs(self, backend_class):
        """
        tenant_id, user_id, expires, metadata must be keyword-only.
        Enforced via the `*,` separator in the protocol.
        """
        sig = inspect.signature(backend_class.store)
        params = sig.parameters

        for kw in ("tenant_id", "user_id", "expires", "metadata"):
            assert kw in params, (
                f"{backend_class.__name__}.store() missing kwarg {kw!r}"
            )
            assert params[kw].kind == inspect.Parameter.KEYWORD_ONLY, (
                f"{backend_class.__name__}.store() parameter {kw!r} must be "
                f"keyword-only (got {params[kw].kind})"
            )
            assert params[kw].default is None, (
                f"{backend_class.__name__}.store() parameter {kw!r} must "
                f"default to None (got {params[kw].default!r})"
            )

    def test_search_signature_required_params(self, backend_class):
        sig = inspect.signature(backend_class.search)
        params = sig.parameters

        for required in ("self", "embedding", "threshold"):
            assert required in params, (
                f"{backend_class.__name__}.search() missing required "
                f"parameter {required!r}"
            )

    def test_search_signature_keyword_only_kwargs(self, backend_class):
        sig = inspect.signature(backend_class.search)
        params = sig.parameters

        for kw in ("tenant_id", "user_id", "now"):
            assert kw in params, (
                f"{backend_class.__name__}.search() missing kwarg {kw!r}"
            )
            assert params[kw].kind == inspect.Parameter.KEYWORD_ONLY, (
                f"{backend_class.__name__}.search() parameter {kw!r} must "
                f"be keyword-only (got {params[kw].kind})"
            )
            assert params[kw].default is None, (
                f"{backend_class.__name__}.search() parameter {kw!r} must "
                f"default to None (got {params[kw].default!r})"
            )

    def test_clear_signature(self, backend_class):
        sig = inspect.signature(backend_class.clear)
        # Only `self`. No required positional args.
        params = list(sig.parameters.values())
        non_self = [p for p in params if p.name != "self"]
        for p in non_self:
            assert p.default is not inspect.Parameter.empty, (
                f"{backend_class.__name__}.clear() parameter {p.name!r} "
                "must have a default (clear takes no required args)"
            )


# =============================================================================
# Round-trip — needs a live backend instance.
# =============================================================================

class TestRoundTrip:
    """Store + search + retrieve. Skips when no local instance available."""

    EMB_DIM = 384  # MiniLM dimension; backends accept any size

    def _vec(self, fill: float = 1.0) -> list[float]:
        # Unit-norm vector pointing in one direction. Sufficient for cosine
        # similarity to return ~1.0 against itself.
        v = [0.0] * self.EMB_DIM
        v[0] = fill
        return v

    def test_empty_search_returns_miss(self, backend_instance):
        resp, sim = backend_instance.search(self._vec(), threshold=0.5)
        assert resp is None
        assert sim == 0.0

    def test_store_then_search_returns_hit(self, backend_instance):
        backend_instance.store(
            "k1",
            "what is python",
            "Python is a programming language.",
            self._vec(),
        )
        resp, sim = backend_instance.search(self._vec(), threshold=0.85)
        assert resp == "Python is a programming language."
        assert sim >= 0.85

    def test_clear_empties_backend(self, backend_instance):
        backend_instance.store(
            "k1", "q", "A", self._vec(),
        )
        backend_instance.clear()
        # After clear, lookup should miss. (Some backends may need a fresh
        # store afterwards to be functional; we only assert the miss here.)
        resp, _ = backend_instance.search(self._vec(), threshold=0.85)
        assert resp is None


# =============================================================================
# Tenant isolation — runs only on backends with ENFORCES_TENANT_ISOLATION=True.
# =============================================================================

class TestTenantIsolation:
    """
    Asserts the protocol's hard isolation guarantee:
        Entries from other tenants MUST NOT be returned, even if their
        similarity exceeds threshold.
    """

    EMB_DIM = 384

    def _vec(self, fill: float = 1.0) -> list[float]:
        v = [0.0] * self.EMB_DIM
        v[0] = fill
        return v

    def test_cross_tenant_search_misses(self, backend_instance):
        cls = type(backend_instance)
        if not getattr(cls, "ENFORCES_TENANT_ISOLATION", False):
            pytest.skip(
                f"{cls.__name__} does not enforce tenant isolation; "
                f"contract not applicable."
            )

        # Tenant A stores an entry.
        backend_instance.store(
            "k1", "q", "tenant_A_secret", self._vec(),
            tenant_id="tenant_a",
        )

        # Tenant B searches with the same embedding — must miss.
        resp, sim = backend_instance.search(
            self._vec(), threshold=0.5,
            tenant_id="tenant_b",
        )
        assert resp is None, (
            f"{cls.__name__} leaked tenant_a entry to tenant_b search "
            f"(got response={resp!r}, similarity={sim})"
        )

    def test_same_tenant_search_hits(self, backend_instance):
        cls = type(backend_instance)
        if not getattr(cls, "ENFORCES_TENANT_ISOLATION", False):
            pytest.skip(
                f"{cls.__name__} does not enforce tenant isolation."
            )

        backend_instance.store(
            "k1", "q", "tenant_A_data", self._vec(),
            tenant_id="tenant_a",
        )
        resp, sim = backend_instance.search(
            self._vec(), threshold=0.5,
            tenant_id="tenant_a",
        )
        assert resp == "tenant_A_data"
