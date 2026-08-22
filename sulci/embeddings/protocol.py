# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/embeddings/protocol.py
============================
Public Embedder protocol — first introduced in v0.4.0.

STABLE API — modifications require api-reviewer approval per ADR 0005.
Changes to this protocol are BREAKING CHANGES for all customer-authored
embedder implementations. Do not modify without a superseding ADR.

This protocol formalizes the shape that MiniLMEmbedder and OpenAIEmbedder
already have. No embedder implementation changed to match this protocol.

Usage
-----
Verify your custom embedder conforms:

    from sulci.embeddings import Embedder

    class MyEmbedder:
        @property
        def dimension(self) -> int: return 768
        def embed(self, text: str) -> list[float]: ...
        def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    isinstance(MyEmbedder(), Embedder)  # True

Inject into Cache:

    from sulci import Cache
    cache = Cache(embedding_model=MyEmbedder())
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """
    Protocol every sulci embedder must satisfy.

    Implementations shipped in v0.4.0:
      - MiniLMEmbedder  sulci/embeddings/minilm.py
         (supports model_name "minilm", "mpnet", "bge")
      - OpenAIEmbedder  sulci/embeddings/openai.py

    Custom implementations: any class matching this surface.
    Verify conformance: sulci.tests.compat.test_embedder_conformance
    """

    @property
    def dimension(self) -> int:
        """
        Vector dimensionality this embedder produces.

        MiniLM: 384
        MPNet: 768
        BGE: 768
        OpenAI text-embedding-3-small: 1536

        Cache uses this to initialize backends with matching vector sizes.
        MUST be stable for the lifetime of the embedder instance.
        """
        ...

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Returns:
            L2-normalized vector of length `self.dimension`.

        Normalization is required because sulci backends use cosine
        similarity and assume unit-length vectors. Unnormalized vectors
        produce incorrect similarity scores.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts efficiently in a single call.

        Returns:
            List of L2-normalized vectors, one per input text,
            each of length `self.dimension`.

        This is the hot path for bulk-loading caches and for
        benchmark workloads. Implementations should batch internally
        when possible (e.g., sentence-transformers supports batch_size).
        """
        ...
