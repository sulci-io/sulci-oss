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
import time
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


# =============================================================================
# User erasure — 0.10.0 (ADR 0005 Amendment 2). Runs for each backend that has delete_user; a backend without
# it skips BEFORE it is constructed, so the count of skips does not depend on whether its server is up.
# =============================================================================

@pytest.fixture
def erasable_backend(backend_class, request):
    if not callable(getattr(backend_class, "delete_user", None)):
        pytest.skip(f"{backend_class.__name__} has no delete_user (0.10.0: added backend by backend)")
    return request.getfixturevalue("backend_instance")     # constructs, clears before and after, or skips


class TestUserErasure:
    """``Backend.delete_user(user_id, *, tenant_id=None) -> int``: removes exactly the entries filed under
    (tenant_id or "global", user_id) — expired ones too — returns how many, and a second call returns 0."""

    EMB_DIM = 384

    def _vec(self, i: int) -> list[float]:
        v = [0.0] * self.EMB_DIM
        v[i % self.EMB_DIM] = 1.0
        return v

    #: Two kinds of expired entry. k3 expired in 1970: a backend that keeps expired entries until they are deleted
    #: must erase and count it; Redis never holds it (it removes a key at its EXPIREAT — oss-0017, measured).
    #: k4 expires in an hour: every backend holds it, and a lookup at LATER treats it as expired.
    LATER = 4_000_000_000.0                          # 2096

    def _seed(self, b) -> int:
        """Stores the user's four entries and the bystanders; returns how many of the user's the backend holds:
        4, or 3 for a backend that declares DROPS_EXPIRED_AT_STORE (Redis). Declared, not probed: a probe through
        the backend's own search() would let a search() that ignores `now` hide a delete_user that skips expired
        entries."""
        b.store("k1", "q1", "t1/u42/1", self._vec(1), tenant_id="t1", user_id="u42")
        b.store("k2", "q2", "t1/u42/2", self._vec(2), tenant_id="t1", user_id="u42")
        b.store("k3", "q3", "t1/u42/expired", self._vec(3), tenant_id="t1", user_id="u42", expires=1.0)
        b.store("k4", "q4", "t1/u42/expiring", self._vec(4), tenant_id="t1", user_id="u42", expires=time.time() + 3600)
        b.store("k1", "q1", "t2/u42", self._vec(1), tenant_id="t2", user_id="u42")
        b.store("k1", "q1", "t1/u7", self._vec(1), tenant_id="t1", user_id="u7")
        b.store("k1", "q1", "t1/shared", self._vec(1), tenant_id="t1")
        b.store("k1", "q1", "global/u42", self._vec(1), user_id="u42")
        # now=0.5 is before 1970's expiry, so a held k3 is found; LATER is after k4's.
        held_k3 = not getattr(type(b), "DROPS_EXPIRED_AT_STORE", False)
        found_k3 = b.search(self._vec(3), 0.5, tenant_id="t1", user_id="u42", now=0.5)[0] == "t1/u42/expired"
        assert found_k3 == held_k3, f"k3 (expired 1970) held: {found_k3}, but DROPS_EXPIRED_AT_STORE says {not held_k3}"
        assert b.search(self._vec(4), 0.5, tenant_id="t1", user_id="u42", now=self.LATER)[0] is None
        return 3 + held_k3

    def test_signature(self, backend_class):
        if not callable(getattr(backend_class, "delete_user", None)):
            pytest.skip(f"{backend_class.__name__} has no delete_user")
        params = inspect.signature(backend_class.delete_user).parameters
        assert list(params)[:2] == ["self", "user_id"]
        assert params["tenant_id"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["tenant_id"].default is None

    def test_returns_the_count_of_the_users_entries_expired_included(self, erasable_backend):
        held = self._seed(erasable_backend)
        assert erasable_backend.delete_user("u42", tenant_id="t1") == held

    def test_the_user_has_nothing_left(self, erasable_backend):
        b = erasable_backend
        self._seed(b)
        b.delete_user("u42", tenant_id="t1")
        for i in (1, 2, 3, 4):
            assert b.search(self._vec(i), 0.5, tenant_id="t1", user_id="u42", now=0.5)[0] is None

    def test_the_same_user_in_another_tenant_is_intact(self, erasable_backend):
        b = erasable_backend
        self._seed(b)
        b.delete_user("u42", tenant_id="t1")
        assert b.search(self._vec(1), 0.5, tenant_id="t2", user_id="u42")[0] == "t2/u42"
        assert b.search(self._vec(1), 0.5, user_id="u42")[0] == "global/u42"

    def test_other_users_and_the_shared_user_of_the_tenant_are_intact(self, erasable_backend):
        b = erasable_backend
        self._seed(b)
        b.delete_user("u42", tenant_id="t1")
        assert b.search(self._vec(1), 0.5, tenant_id="t1", user_id="u7")[0] == "t1/u7"
        assert b.search(self._vec(1), 0.5, tenant_id="t1")[0] == "t1/shared"

    def test_no_tenant_is_the_global_tenant(self, erasable_backend):
        b = erasable_backend
        self._seed(b)
        assert b.delete_user("u42") == 1
        assert b.search(self._vec(1), 0.5, user_id="u42")[0] is None
        assert b.search(self._vec(1), 0.5, tenant_id="t1", user_id="u42")[0] == "t1/u42/1"

    def test_a_second_call_returns_zero(self, erasable_backend):
        self._seed(erasable_backend)
        erasable_backend.delete_user("u42", tenant_id="t1")
        assert erasable_backend.delete_user("u42", tenant_id="t1") == 0

    def test_a_user_with_nothing_stored(self, erasable_backend):
        assert erasable_backend.delete_user("nobody", tenant_id="t1") == 0
