# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_per_call_threshold.py
=================================
v0.8.0 (#34) — ``Cache.get(threshold=...)``.

Closes sulci-platform #44 option (a), sulci-platform #59 option 3, and ADR 0022
Open-7. All three are the same change seen from three angles.

WHY THIS IS A CORRECTNESS FIX, NOT AN ERGONOMIC ONE
---------------------------------------------------
Before this kwarg existed, a caller who wanted a threshold other than the
instance value had exactly one option: let ``Cache`` return a hit at ITS
threshold, then throw the hit away.

sulci-platform's gateway did precisely that. And it meant:

    the LIBRARY decided CacheEvent.cache_hit   (at the instance threshold)
    the CALLER decided what the customer saw   (at the effective threshold)

Those are two different numbers, so for similarity in ``[instance, effective)``
the library emitted a **hit** event while the customer received a **miss**.
Everything downstream of CacheEvent — Stripe meter events, usage_daily rollups,
the dashboard's hit rate, per-entry hit counters — counted a hit that never
happened. Shipped since v0.6.0.

The event and the answer are now decided by the same number, in one place.
``TestEventAgreesWithAnswer`` is the test that pins it, and it is the reason
this change exists.

NOT CHANGED: the Backend protocol. ``Backend.search(embedding, threshold, ...)``
has taken a per-call threshold since v0.4.0 — ``Cache`` simply never passed
anything but ``self.threshold`` to it. Custom backends need no changes, and the
conformance suite is untouched.
"""
from __future__ import annotations

import pytest

from sulci import Cache
from sulci.sinks.protocol import CacheEvent


class RecordingSink:
    """Captures every CacheEvent the Cache emits."""

    def __init__(self):
        self.events: list[CacheEvent] = []

    def emit(self, event: CacheEvent) -> None:
        self.events.append(event)

    @property
    def types(self) -> list[str]:
        return [e.event_type for e in self.events]


class StubBackend:
    """
    A Backend that returns a fixed similarity, and honours `threshold` exactly
    the way a real backend does: below it, nothing comes back.

    Records the threshold it was called with, so the tests can assert what the
    Cache actually *passed down* rather than only what came back.
    """

    ENFORCES_TENANT_ISOLATION = False

    def __init__(self, similarity: float = 0.75, response: str = "cached"):
        self.similarity = similarity
        self.response = response
        self.thresholds_seen: list[float] = []

    def store(self, key, query, response, embedding, **kw) -> None:
        pass

    def search(self, embedding, threshold, *, tenant_id=None, user_id=None, now=None):
        self.thresholds_seen.append(threshold)
        if self.similarity >= threshold:
            return self.response, self.similarity
        return None, 0.0

    def clear(self) -> None:
        pass


class StubEmbedder:
    dimension = 4

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def make_cache(instance_threshold: float, similarity: float, sink=None) -> Cache:
    return Cache(
        backend=StubBackend(similarity=similarity),
        embedding_model=StubEmbedder(),
        threshold=instance_threshold,
        event_sink=sink,
        telemetry=False,
    )


# ── The point of the whole change ────────────────────────────────────────────

class TestEventAgreesWithAnswer:
    """
    THE regression this exists for.

    The emitted CacheEvent must say the same thing the caller was told. Before
    #34 they could disagree, and billing believed the event.
    """

    def test_miss_at_stricter_threshold_emits_a_MISS_event(self):
        """
        Similarity 0.75. Instance threshold 0.65 (would hit). Caller asks for
        0.85 (a miss).

        PRE-#34 the only way to express this was to let the Cache hit at 0.65
        and discard the result — which emitted event_type="hit" while returning
        a miss to the user. Stripe billed for that hit.
        """
        sink = RecordingSink()
        cache = make_cache(instance_threshold=0.65, similarity=0.75, sink=sink)

        resp, sim, _ = cache.get("q", threshold=0.85)

        assert resp is None, "caller must see a miss at 0.85"
        assert sink.types == ["miss"], (
            "the EVENT must also be a miss. If this says 'hit', the library and "
            "the caller disagree, and billing believes the library."
        )

    def test_hit_at_looser_threshold_emits_a_HIT_event(self):
        """The mirror image — and it must be a real hit, not a coincidence."""
        sink = RecordingSink()
        cache = make_cache(instance_threshold=0.85, similarity=0.75, sink=sink)

        resp, sim, _ = cache.get("q", threshold=0.65)

        assert resp == "cached"
        assert sim == 0.75
        assert sink.types == ["hit"]

    def test_stats_agree_with_the_answer_too(self):
        """hits/misses feed saved_cost and the dashboard. Same invariant."""
        cache = make_cache(instance_threshold=0.65, similarity=0.75)

        cache.get("q", threshold=0.85)          # miss
        assert cache.stats()["hits"] == 0
        assert cache.stats()["misses"] == 1

        cache.get("q", threshold=0.65)          # hit
        assert cache.stats()["hits"] == 1


# ── Resolution semantics ─────────────────────────────────────────────────────

class TestThresholdResolution:

    def test_omitted_uses_the_instance_value(self):
        """Every pre-0.8.0 caller must be completely unaffected."""
        cache = make_cache(instance_threshold=0.70, similarity=0.75)
        backend = cache._backend

        cache.get("q")

        assert backend.thresholds_seen == [0.70]

    def test_per_call_overrides_the_instance_value(self):
        cache = make_cache(instance_threshold=0.70, similarity=0.75)
        backend = cache._backend

        cache.get("q", threshold=0.90)

        assert backend.thresholds_seen == [0.90]

    def test_zero_is_an_override_not_an_inherit(self):
        """
        The subtle one. `0.0` is falsy. Resolving with `threshold or
        self.threshold` would silently substitute the instance value and a
        caller asking to match ANYTHING would get the default instead.

        Uses `is None`. There is a test because this is the exact bug that makes
        people stop trusting a config system.
        """
        cache = make_cache(instance_threshold=0.90, similarity=0.10, sink=None)
        backend = cache._backend

        resp, sim, _ = cache.get("q", threshold=0.0)

        assert backend.thresholds_seen == [0.0], "0.0 must reach the backend, not 0.90"
        assert resp == "cached", "at threshold 0.0 even similarity 0.10 is a hit"

    def test_instance_threshold_is_not_mutated(self):
        """A per-call override is per-call. It must not leak into the next one."""
        cache = make_cache(instance_threshold=0.70, similarity=0.75)
        backend = cache._backend

        cache.get("q", threshold=0.95)
        cache.get("q")

        assert backend.thresholds_seen == [0.95, 0.70]
        assert cache.threshold == 0.70


class TestValidation:
    """
    An out-of-range threshold is a programming error — it can never be satisfied
    (>1) or never be missed (<0). Raising surfaces the bug in development;
    clamping would hide it and quietly change the caller's hit rate.

    Safe to raise: the value comes from the caller's own code, never from data.
    """

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0, -1.0])
    def test_out_of_range_raises(self, bad):
        cache = make_cache(instance_threshold=0.65, similarity=0.75)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            cache.get("q", threshold=bad)

    @pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
    def test_boundaries_are_accepted(self, ok):
        cache = make_cache(instance_threshold=0.65, similarity=0.75)
        cache.get("q", threshold=ok)   # must not raise


# ── Forwarding ───────────────────────────────────────────────────────────────

class TestCachedCallForwardsThreshold:
    """
    `cached_call` is the ergonomic front door. A kwarg it silently swallows is a
    kwarg that does not exist as far as most users are concerned.
    """

    def test_cached_call_honours_a_stricter_threshold(self):
        sink = RecordingSink()
        cache = make_cache(instance_threshold=0.65, similarity=0.75, sink=sink)
        called = []

        result = cache.cached_call(
            "q", lambda q: called.append(q) or "fresh", threshold=0.85,
        )

        assert result["source"] == "llm", "0.85 > 0.75 → miss → LLM must run"
        assert called == ["q"]
        assert "hit" not in sink.types

    def test_cached_call_honours_a_looser_threshold(self):
        cache = make_cache(instance_threshold=0.85, similarity=0.75)
        called = []

        result = cache.cached_call(
            "q", lambda q: called.append(q) or "fresh", threshold=0.65,
        )

        assert result["source"] == "cache"
        assert result["response"] == "cached"
        assert called == [], "LLM must NOT be called on a hit"


class TestBackendProtocolUnchanged:
    """
    #34 required NO change to the public Backend protocol.

    `Backend.search(embedding, threshold, ...)` has accepted a per-call threshold
    since v0.4.0. `Cache` simply never passed anything but `self.threshold` down.
    So no custom backend needs updating and the conformance suite is untouched —
    which is why this landed as a minor rather than a major.
    """

    def test_search_signature_still_takes_threshold_positionally(self):
        import inspect

        from sulci.backends.protocol import Backend

        params = list(inspect.signature(Backend.search).parameters)
        assert params[:3] == ["self", "embedding", "threshold"]
