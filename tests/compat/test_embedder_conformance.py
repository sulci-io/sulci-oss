# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
tests/compat/test_embedder_conformance.py
==========================================
Conformance tests for the Embedder protocol.
"""
from __future__ import annotations
import inspect
import math


class TestStructural:
    """Signature/protocol contract — works on the class, no infra needed."""

    def test_has_dimension_property(self, embedder_class):
        assert hasattr(embedder_class, "dimension"), (
            f"{embedder_class.__name__} missing 'dimension' attribute"
        )
        # Protocol says dimension is a property
        attr = inspect.getattr_static(embedder_class, "dimension")
        assert isinstance(attr, property), (
            f"{embedder_class.__name__}.dimension must be a @property "
            f"(got {type(attr).__name__})"
        )

    def test_has_embed_method(self, embedder_class):
        assert hasattr(embedder_class, "embed")
        sig = inspect.signature(embedder_class.embed)
        params = sig.parameters
        assert "self" in params
        assert "text" in params

    def test_has_embed_batch_method(self, embedder_class):
        assert hasattr(embedder_class, "embed_batch")
        sig = inspect.signature(embedder_class.embed_batch)
        params = sig.parameters
        assert "self" in params
        assert "texts" in params


class TestBehavior:
    """Round-trip behavior — needs a live embedder instance."""

    def test_dimension_is_positive_int(self, embedder_instance):
        d = embedder_instance.dimension
        assert isinstance(d, int)
        assert d > 0

    def test_dimension_is_stable(self, embedder_instance):
        assert embedder_instance.dimension == embedder_instance.dimension

    def test_embed_returns_list_of_dimension_floats(self, embedder_instance):
        vec = embedder_instance.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == embedder_instance.dimension
        assert all(isinstance(x, float) for x in vec)

    def test_embed_is_l2_normalized(self, embedder_instance):
        vec = embedder_instance.embed("hello world")
        norm = math.sqrt(sum(x * x for x in vec))
        # L2 norm should be ~1.0 (allow some float drift)
        assert abs(norm - 1.0) < 1e-3, (
            f"{type(embedder_instance).__name__} produced un-normalized "
            f"vector (||v|| = {norm:.6f}, expected ~1.0)"
        )

    def test_embed_batch_length_matches_input(self, embedder_instance):
        texts = ["one", "two", "three"]
        vecs = embedder_instance.embed_batch(texts)
        assert len(vecs) == len(texts)
        assert all(len(v) == embedder_instance.dimension for v in vecs)
