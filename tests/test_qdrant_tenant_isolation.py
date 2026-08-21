# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
tests/test_qdrant_tenant_isolation.py
======================================
End-to-end tenant isolation tests for QdrantBackend, framed around real
multi-tenant SaaS scenarios.

These tests verify the protocol's hard isolation guarantee:
    "Tenant isolation is a hard boundary — entries from other tenants
     MUST NOT be returned, even if their similarity exceeds threshold."

Test names are deliberately written as customer-support product scenarios
(rather than abstract test_filter_with_must_clause names) so that:
  - A failure tells the reader what user-impacting thing broke
  - The file doubles as documentation of the multi-tenancy contract
  - Auditors and security reviewers can read it as a behavioral spec

Backstory: HelpDesk AI is a SaaS support chatbot company. Their customers
are tenants — Acme Corp (industrial pumps), Globex Health (hospital
HIPAA), Initech Software (B2B API). The semantic cache sits in front of
the LLM; tenant isolation prevents one customer's cached responses from
leaking to another customer.

These scenarios run against an embedded Qdrant via db_path; if
qdrant-client is not installed, all tests skip cleanly.
"""
from __future__ import annotations
import math
import tempfile

import pytest

qdrant_client = pytest.importorskip("qdrant_client")
from sulci.backends.qdrant import QdrantBackend  # noqa: E402


# -----------------------------------------------------------------------------
# Test fixtures and helpers.
# -----------------------------------------------------------------------------

def _normalized(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


@pytest.fixture
def backend(tmp_path):
    """Embedded Qdrant backend, fresh per test."""
    return QdrantBackend(db_path=str(tmp_path / "qdrant"))


# Two semantically-similar embeddings — close enough that a broken filter
# would happily match across them. We use these to ensure that *similarity*
# alone never bypasses tenant isolation.
WARRANTY_QUERY_VEC  = _normalized([1.0, 0.05, 0.0] + [0.0] * 381)
WARRANTY_QUERY_VEC2 = _normalized([1.0, 0.06, 0.01] + [0.0] * 381)  # ~99% sim
PRICING_QUERY_VEC   = _normalized([0.0, 1.0, 0.0] + [0.0] * 381)   # orthogonal


# =============================================================================
# Scenario 1: Acme Corp agents share each other's cache (the value prop)
# =============================================================================

class TestAcmeAgentsShareCache:
    """Same-tenant agents benefit from each other's cached LLM responses."""

    def test_second_agent_gets_cache_hit_on_paraphrased_query(self, backend):
        # Agent #1 asks about warranty; cache stores the LLM's response.
        backend.store(
            key       = "acme-warranty-1",
            query     = "What's the warranty on the Acme P500 pump?",
            response  = "The Acme P500 has a 5-year limited warranty.",
            embedding = WARRANTY_QUERY_VEC,
            tenant_id = "acme_corp",
        )

        # Agent #2 asks a paraphrased version of the same question.
        # Vector is highly similar (~99%) — they should hit the cache.
        resp, sim = backend.search(
            WARRANTY_QUERY_VEC2,
            threshold = 0.85,
            tenant_id = "acme_corp",
        )
        assert resp == "The Acme P500 has a 5-year limited warranty."
        assert sim >= 0.85, f"Expected high similarity for paraphrase; got {sim}"


# =============================================================================
# Scenario 2: PHI must not leak from Globex Health to Initech Software
# =============================================================================

class TestGlobexPhiDoesNotLeakToInitech:
    """
    The single most important test in this file. If this fails, the
    multi-tenant SaaS deployment model is unsafe — a HIPAA breach and a
    company-ending lawsuit waiting to happen.
    """

    def test_initech_search_never_returns_globex_phi(self, backend):
        # Globex stores a response containing patient PHI.
        backend.store(
            key       = "globex-phi-1",
            query     = "Bill code for John Smith's surgery on March 15?",
            response  = "Patient John Smith MRN 4471823, bill code 49505 to UHC. Balance $1,247.83.",
            embedding = WARRANTY_QUERY_VEC,
            tenant_id = "globex_health",
        )

        # Initech, an unrelated tenant, asks something semantically similar.
        # Without tenant isolation, the high vector similarity would match.
        # WITH tenant isolation, this must miss — even though the vectors
        # are identical.
        resp, sim = backend.search(
            WARRANTY_QUERY_VEC,
            threshold = 0.85,
            tenant_id = "initech_software",
        )
        assert resp is None, (
            f"CRITICAL: tenant isolation breach. Globex PHI ({resp!r}) was "
            f"returned to Initech with similarity {sim}. Multi-tenant SaaS "
            f"deployment model is not safe."
        )
        assert sim == 0.0


# =============================================================================
# Scenario 3: Within-tenant user partitions when user_id is set
# =============================================================================

class TestAcmeAgentsIsolatedWhenUserIdSet:
    """
    Optional within-tenant user partitioning. Two agents at the same
    company but with different access permissions — caches partition by
    user_id so one agent's privileged responses don't leak to another
    agent who shouldn't see them.
    """

    def test_user_a_response_does_not_leak_to_user_b(self, backend):
        # Agent A (enterprise sales) caches a response with privileged data.
        backend.store(
            key       = "acme-pricing-a",
            query     = "What pricing tier does Northrop Industries get?",
            response  = "Northrop is Tier-3 enterprise: 22% discount, net-60 terms.",
            embedding = PRICING_QUERY_VEC,
            tenant_id = "acme_corp",
            user_id   = "agent_001_enterprise",
        )

        # Agent B (residential sales, no enterprise access) asks similarly.
        resp, sim = backend.search(
            PRICING_QUERY_VEC,
            threshold = 0.85,
            tenant_id = "acme_corp",
            user_id   = "agent_002_residential",
        )
        assert resp is None
        assert sim == 0.0

    def test_user_a_can_still_hit_their_own_cache(self, backend):
        # Sanity check: user_id partitioning doesn't break same-user hits.
        backend.store(
            key       = "acme-pricing-a",
            query     = "What pricing tier does Northrop Industries get?",
            response  = "Northrop is Tier-3 enterprise: 22% discount, net-60 terms.",
            embedding = PRICING_QUERY_VEC,
            tenant_id = "acme_corp",
            user_id   = "agent_001_enterprise",
        )

        resp, sim = backend.search(
            PRICING_QUERY_VEC,
            threshold = 0.85,
            tenant_id = "acme_corp",
            user_id   = "agent_001_enterprise",
        )
        assert resp == "Northrop is Tier-3 enterprise: 22% discount, net-60 terms."


# =============================================================================
# Scenario 4: Solo-developer single-tenant deployment (no tenant filter)
# =============================================================================

class TestSoloDeveloperNoTenantFilter:
    """
    Backwards-compatible single-tenant use. A solo developer using sulci
    as a personal cache passes neither tenant_id nor user_id; everything
    goes into the implicit 'global' partition.
    """

    def test_unfiltered_store_and_search_round_trips(self, backend):
        backend.store(
            key       = "solo-aws",
            query     = "How do I deploy a Python app to AWS Lambda?",
            response  = "Package code as a .zip, upload via aws lambda create-function.",
            embedding = WARRANTY_QUERY_VEC,
        )

        resp, sim = backend.search(WARRANTY_QUERY_VEC, threshold=0.85)
        assert resp == "Package code as a .zip, upload via aws lambda create-function."
        assert sim >= 0.99


# =============================================================================
# Scenario 5: tenant_id and user_id compose correctly (the four-quadrant test)
# =============================================================================

class TestTenantAndUserCompose:
    """
    Four-quadrant matrix: same/different tenant × same/different user.
    Only the (same-tenant, same-user) quadrant should hit. The other three
    must miss — including the ambiguous (different-tenant, same-user-id)
    case where the user_id collides across tenants.
    """

    def test_four_quadrants(self, backend):
        # Store one entry under tenant=acme, user=agent_001
        backend.store(
            key       = "compose-1",
            query     = "warranty question",
            response  = "acme-agent_001 response",
            embedding = WARRANTY_QUERY_VEC,
            tenant_id = "acme_corp",
            user_id   = "agent_001",
        )

        # Quadrant 1: same tenant, same user → HIT
        resp, _ = backend.search(
            WARRANTY_QUERY_VEC, threshold=0.85,
            tenant_id="acme_corp", user_id="agent_001",
        )
        assert resp == "acme-agent_001 response", "same-tenant same-user must hit"

        # Quadrant 2: same tenant, different user → MISS
        resp, _ = backend.search(
            WARRANTY_QUERY_VEC, threshold=0.85,
            tenant_id="acme_corp", user_id="agent_002",
        )
        assert resp is None, "same-tenant different-user must miss"

        # Quadrant 3: different tenant, same user_id → MISS
        # (This is the trap: a user_id that happens to match across tenants
        # must NOT bridge the tenant boundary. Tenant filter wins.)
        resp, _ = backend.search(
            WARRANTY_QUERY_VEC, threshold=0.85,
            tenant_id="globex_health", user_id="agent_001",
        )
        assert resp is None, "different-tenant same-user_id must miss"

        # Quadrant 4: different tenant, different user → MISS
        resp, _ = backend.search(
            WARRANTY_QUERY_VEC, threshold=0.85,
            tenant_id="globex_health", user_id="other_agent",
        )
        assert resp is None, "different-tenant different-user must miss"


# =============================================================================
# Scenario 6: 'global' default sentinel does not leak to named tenants
# =============================================================================

class TestGlobalDefaultDoesNotLeakToNamedTenants:
    """
    Operational migration safety: a deployment that started as
    single-tenant (no tenant_id) and later added multi-tenant features
    should not have its old un-tenanted entries suddenly start matching
    any specific tenant's searches.
    """

    def test_global_entry_does_not_match_named_tenant_search(self, backend):
        # Old entry stored before multi-tenancy was introduced.
        backend.store(
            key       = "legacy-1",
            query     = "How do I deploy?",
            response  = "Use docker-compose up.",
            embedding = WARRANTY_QUERY_VEC,
            # tenant_id=None — stored internally as "global"
        )

        # New code searches with a real tenant_id.
        resp, _ = backend.search(
            WARRANTY_QUERY_VEC,
            threshold = 0.85,
            tenant_id = "acme_corp",
        )
        assert resp is None, (
            "Legacy 'global' entry leaked into a named-tenant search. "
            "Operational migration to multi-tenancy is unsafe."
        )

    def test_named_tenant_entry_does_not_match_global_search(self, backend):
        # Reverse direction: a tenant-scoped entry must not appear in
        # an un-scoped (global) search either.
        backend.store(
            key       = "acme-1",
            query     = "warranty",
            response  = "acme warranty info",
            embedding = WARRANTY_QUERY_VEC,
            tenant_id = "acme_corp",
        )

        resp, _ = backend.search(
            WARRANTY_QUERY_VEC,
            threshold = 0.85,
            # tenant_id=None — searches "global" partition
        )
        assert resp is None


# =============================================================================
# Scenario 7: None and empty string tenant_id pin sentinel semantics
# =============================================================================

class TestNoneAndEmptyStringSentinelSemantics:
    """
    The protocol stores tenant_id=None as 'global'. Empty string ''
    is undefined in the protocol docstring; this test pins the actual
    behavior so future refactors don't accidentally change it.
    """

    def test_none_searches_match_none_stores(self, backend):
        backend.store(
            key       = "k1",
            query     = "q",
            response  = "r",
            embedding = WARRANTY_QUERY_VEC,
            tenant_id = None,
        )
        resp, _ = backend.search(
            WARRANTY_QUERY_VEC, threshold=0.85, tenant_id=None,
        )
        assert resp == "r"

    def test_empty_string_does_not_match_none(self, backend):
        # Pin current behavior: tenant_id="" is not the same as None.
        # tenant_id="" hits Qdrant's filter as a literal empty-string
        # match, while tenant_id=None falls through to no filter at all
        # (which then matches "global" payloads). They are distinct.
        backend.store(
            key       = "k1",
            query     = "q",
            response  = "r",
            embedding = WARRANTY_QUERY_VEC,
            tenant_id = None,  # stored as "global"
        )
        resp, _ = backend.search(
            WARRANTY_QUERY_VEC, threshold=0.85, tenant_id="",
        )
        # If this assertion fails, someone changed the sentinel semantics.
        # That's not necessarily wrong — but it's a contract change that
        # needs deliberate review and a docstring update.
        assert resp is None


# =============================================================================
# Scenario 8: Tenant isolation survives clear-and-restore lifecycle
# =============================================================================

class TestTenantIsolationSurvivesClear:
    """
    Operational test: clear() should not corrupt the tenant-isolation
    behavior on subsequent stores. Specifically guards against a
    regression where clear() destroys the collection (qdrant-client 1.x
    raises on subsequent operations) or leaves the filter index in a
    broken state.
    """

    def test_clear_then_restore_preserves_tenant_isolation(self, backend):
        # First lifecycle: store + verify isolation
        backend.store(
            key="k1", query="q", response="acme1",
            embedding=WARRANTY_QUERY_VEC, tenant_id="acme_corp",
        )
        backend.store(
            key="k2", query="q", response="globex1",
            embedding=WARRANTY_QUERY_VEC, tenant_id="globex_health",
        )

        resp, _ = backend.search(WARRANTY_QUERY_VEC, threshold=0.85, tenant_id="acme_corp")
        assert resp == "acme1"
        resp, _ = backend.search(WARRANTY_QUERY_VEC, threshold=0.85, tenant_id="globex_health")
        assert resp == "globex1"

        # Clear everything
        backend.clear()

        # After clear, both tenants must miss
        resp, _ = backend.search(WARRANTY_QUERY_VEC, threshold=0.85, tenant_id="acme_corp")
        assert resp is None
        resp, _ = backend.search(WARRANTY_QUERY_VEC, threshold=0.85, tenant_id="globex_health")
        assert resp is None

        # Re-store and verify isolation still works
        backend.store(
            key="k3", query="q", response="acme2",
            embedding=WARRANTY_QUERY_VEC, tenant_id="acme_corp",
        )
        resp, _ = backend.search(WARRANTY_QUERY_VEC, threshold=0.85, tenant_id="acme_corp")
        assert resp == "acme2"

        # And cross-tenant still misses
        resp, _ = backend.search(WARRANTY_QUERY_VEC, threshold=0.85, tenant_id="globex_health")
        assert resp is None
