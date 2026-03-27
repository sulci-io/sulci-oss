# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/embeddings/minilm.py
Local sentence-transformers embeddings — free, no API key, CPU-friendly.

Models:
    "minilm"  →  all-MiniLM-L6-v2      384-dim   fast    good quality
    "mpnet"   →  all-mpnet-base-v2      768-dim   slower  better quality
    "bge"     →  BAAI/bge-base-en-v1.5  768-dim   slower  best open-source

First call downloads the model (~90 MB for MiniLM). Cached locally after.
"""
from __future__ import annotations


class MiniLMEmbedder:

    MODELS = {
        "minilm": "all-MiniLM-L6-v2",
        "mpnet":  "all-mpnet-base-v2",
        "bge":    "BAAI/bge-base-en-v1.5",
    }

    def __init__(self, model_name: str = "minilm"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not found.\n"
                "Install with: pip install \"sulci[chroma]\"  (or any backend extra)"
            )
        model_id    = self.MODELS.get(model_name, model_name)
        self._model = SentenceTransformer(model_id)
        self._dim   = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns L2-normalised vector."""
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one efficient batch call."""
        return self._model.encode(
            texts, normalize_embeddings=True, batch_size=64
        ).tolist()
