# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/embeddings/openai.py
OpenAI embeddings — highest quality, requires API key.

Model: text-embedding-3-small (1536-dim, $0.02 / 1M tokens)
       text-embedding-3-large (3072-dim, $0.13 / 1M tokens)

Requires: OPENAI_API_KEY env var or explicit api_key argument.
"""
from __future__ import annotations
import os


class OpenAIEmbedder:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model:   str           = "text-embedding-3-small",
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not found.\n"
                "Install with: pip install \"sulci[openai]\""
            )
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model  = model
        self._dim    = 3072 if "large" in model else 1536

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Embed a single string via OpenAI API."""
        r = self._client.embeddings.create(model=self._model, input=text)
        return r.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in a single API call."""
        r = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in sorted(r.data, key=lambda x: x.index)]


from typing import Optional  # noqa: E402 — keep import at bottom for clarity
