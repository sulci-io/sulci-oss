# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
tests/test_context_threshold.py
=================================
2026-08-12 — ``Cache(context_threshold=...)``.

WHAT THE DEFECT WAS
-------------------
``benchmark/run.py`` has calibrated its own ``--context-threshold`` separately
since it was written, and states the reason in a comment above the assignment:
the blended lookup vector is 70% query + 30% history, so its raw cosine to any
stored entry is structurally lower than an exact-match lookup's.

``Cache`` never read that. One ``threshold``, default 0.85, decided both. A user
following the class docstring's own ``Cache(backend="sqlite",
context_window=6)`` example ran every blended lookup at exact-match calibration.

WHAT THE FIX IS NOT
-------------------
It is not a new default. Nothing moves for an existing caller: unset means
``threshold``, byte for byte.

And it does NOT key off ``context_window``. That is the trap, and
``TestDiscriminatorIsDepthNotWindow`` is the suite that pins it. A
``Cache(context_window=6)`` serves plenty of lookups with no ``session_id`` or
an empty window; ``_context_vec`` hands those back unblended, and they are
exact-match lookups whatever the constructor said. Keying off ``context_window``
would apply the blended threshold to all of them — which is the defect, not a
version of the fix. It would also pass ``make checkin-fast`` green, because with
``context_threshold`` unset the two numbers are equal and nothing moves.

ONE NUMBER, ONE DECISION POINT
------------------------------
``TestEventAgreesWithAnswer`` mirrors ``tests/test_per_call_threshold.py``'s
suite of the same name, for the same reason. Whichever threshold serves the
answer must be the one in the telemetry payload and the emitted ``CacheEvent``.
v0.8.0 (#34) exists because those were once two numbers, and billing believed
the event.
"""
from __future__ import annotations

import warnings

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
    Returns a fixed similarity and honours ``threshold`` the way a real backend
    does. Records every threshold it was handed, so a test can assert what the
    Cache passed DOWN, not merely what came back.
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


def make_cache(
    *,
    threshold: float = 0.85,
    context_threshold=None,
    context_window: int = 0,
    similarity: float = 0.75,
    sink=None,
) -> Cache:
    return Cache(
        backend=StubBackend(similarity=similarity),
        embedding_model=StubEmbedder(),
        threshold=threshold,
        context_threshold=context_threshold,
        context_window=context_window,
        event_sink=sink,
        telemetry=False,
    )


def _blend(cache: Cache, session_id: str = "s1") -> None:
    """Put one prior turn in the window so the next get() actually blends."""
    cache.set("prior turn", "prior answer", session_id=session_id)


@pytest.fixture(autouse=True)
def _quiet():
    """These suites assert on behaviour; TestWarning asserts on the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


# ── The trap ────────────────────────────────────────────────────────────────

class TestDiscriminatorIsDepthNotWindow:
    """
    THE regression this file exists for.

    context_threshold applies when the lookup BLENDED, not when the Cache is
    capable of blending.
    """

    def test_context_window_set_but_no_session_uses_threshold(self):
        cache = make_cache(threshold=0.85, context_threshold=0.10,
                           context_window=6)
        cache.get("q")                       # no session_id -> depth 0
        assert cache._backend.thresholds_seen == [0.85], (
            "a lookup with no session did not blend; it must be judged at "
            "`threshold`, not at the blended-path value"
        )

    def test_context_window_set_but_empty_window_uses_threshold(self):
        cache = make_cache(threshold=0.85, context_threshold=0.10,
                           context_window=6)
        cache.get("q", session_id="fresh")   # session exists, window empty
        assert cache._backend.thresholds_seen == [0.85]

    def test_blended_lookup_uses_context_threshold(self):
        cache = make_cache(threshold=0.85, context_threshold=0.10,
                           context_window=6)
        _blend(cache)
        _resp, _sim, depth = cache.get("follow-up", session_id="s1")
        assert depth > 0, "fixture did not produce a blended lookup"
        assert cache._backend.thresholds_seen[-1] == 0.10

    def test_mixed_traffic_on_one_instance_gets_both(self):
        """The whole point: one Cache, two kinds of lookup, two thresholds."""
        cache = make_cache(threshold=0.85, context_threshold=0.10,
                           context_window=6)
        _blend(cache)
        cache.get("stateless")                      # depth 0
        cache.get("follow-up", session_id="s1")     # depth > 0
        assert cache._backend.thresholds_seen == [0.85, 0.10]


# ── Nothing moves when it is unset ──────────────────────────────────────────

class TestDefaultUnchanged:

    def test_default_is_none_not_a_number(self):
        assert make_cache().context_threshold is None, (
            "a context default must not ship: 125 synthetic follow-ups across "
            "5 domains cannot size a default every context user inherits"
        )

    def test_unset_blended_lookup_uses_threshold(self):
        cache = make_cache(threshold=0.85, context_window=6)
        _blend(cache)
        cache.get("follow-up", session_id="s1")
        assert cache._backend.thresholds_seen[-1] == 0.85

    def test_unset_is_identical_to_pinning_it_to_threshold(self):
        a = make_cache(threshold=0.85, context_window=6)
        b = make_cache(threshold=0.85, context_threshold=0.85, context_window=6)
        for c in (a, b):
            _blend(c)
            c.get("follow-up", session_id="s1")
        assert a._backend.thresholds_seen == b._backend.thresholds_seen


# ── Precedence ──────────────────────────────────────────────────────────────

class TestPrecedence:

    def test_explicit_per_call_threshold_beats_context_threshold(self):
        cache = make_cache(threshold=0.85, context_threshold=0.10,
                           context_window=6)
        _blend(cache)
        cache.get("follow-up", session_id="s1", threshold=0.99)
        assert cache._backend.thresholds_seen[-1] == 0.99, (
            "a caller who names a number gets that number"
        )

    def test_zero_is_a_legitimate_per_call_threshold(self):
        """`is None`, never `or` — 0.0 must not fall through to the blend value."""
        cache = make_cache(threshold=0.85, context_threshold=0.10,
                           context_window=6)
        _blend(cache)
        cache.get("follow-up", session_id="s1", threshold=0.0)
        assert cache._backend.thresholds_seen[-1] == 0.0


# ── One number, one decision point ──────────────────────────────────────────

class TestEventAgreesWithAnswer:
    """
    Mirrors test_per_call_threshold.py::TestEventAgreesWithAnswer. If the
    blended threshold served the answer, it must be the number in the event.
    """

    def test_blended_hit_below_instance_threshold_emits_a_HIT_event(self):
        sink = RecordingSink()
        cache = make_cache(threshold=0.85, context_threshold=0.50,
                           similarity=0.75, context_window=6, sink=sink)
        _blend(cache)
        sink.events.clear()
        resp, _sim, _d = cache.get("follow-up", session_id="s1")
        assert resp is not None, "0.75 clears the 0.50 blended threshold"
        assert sink.types == ["hit"]

    def test_blended_miss_above_blend_threshold_emits_a_MISS_event(self):
        sink = RecordingSink()
        cache = make_cache(threshold=0.50, context_threshold=0.90,
                           similarity=0.75, context_window=6, sink=sink)
        _blend(cache)
        sink.events.clear()
        resp, _sim, _d = cache.get("follow-up", session_id="s1")
        assert resp is None, "0.75 does not clear the 0.90 blended threshold"
        assert sink.types == ["miss"], (
            "the event must agree with the answer even when the blended "
            "threshold is STRICTER than the instance one"
        )


# ── Validation ──────────────────────────────────────────────────────────────

class TestValidation:

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
    def test_out_of_range_raises_at_construction(self, bad):
        with pytest.raises(ValueError, match="context_threshold"):
            make_cache(context_threshold=bad)

    @pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
    def test_in_range_accepted(self, ok):
        assert make_cache(context_threshold=ok).context_threshold == ok

    def test_int_is_coerced_to_float(self):
        assert isinstance(make_cache(context_threshold=1).context_threshold, float)


# ── The signpost ────────────────────────────────────────────────────────────

class TestWarning:
    """The escape hatch has to be signposted, or it is not an escape hatch."""

    def _warns(self, cache, **kw):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache.get("follow-up", **kw)
        return [str(x.message) for x in w
                if issubclass(x.category, UserWarning)
                and "context_threshold" in str(x.message)]

    def test_warns_on_first_blended_lookup_when_unset(self):
        cache = make_cache(threshold=0.85, context_window=6)
        _blend(cache)
        assert self._warns(cache, session_id="s1")

    def test_does_not_warn_on_a_stateless_lookup(self):
        """A context-capable Cache serving stateless traffic has no problem."""
        cache = make_cache(threshold=0.85, context_window=6)
        assert not self._warns(cache)

    def test_warns_at_most_once_per_instance(self):
        cache = make_cache(threshold=0.85, context_window=6)
        _blend(cache)
        seen = sum(len(self._warns(cache, session_id="s1")) for _ in range(5))
        assert seen <= 1

    def test_silent_when_context_threshold_is_set(self):
        cache = make_cache(threshold=0.85, context_threshold=0.50,
                           context_window=6)
        _blend(cache)
        assert not self._warns(cache, session_id="s1")

    def test_warning_names_no_recommended_value(self):
        """
        §4: the corpus cannot discriminate at any threshold. A warning that
        suggests a number would be publishing a figure the evidence does not
        support, one level away from the caveat that qualifies it.
        """
        cache = make_cache(threshold=0.85, context_window=6)
        _blend(cache)
        msg = self._warns(cache, session_id="s1")[0]
        for forbidden in ("0.70", "0.7,", "try 0.", "recommend 0.", "use 0."):
            assert forbidden not in msg
        assert "no context-specific default" in msg or "no measurement" in msg
