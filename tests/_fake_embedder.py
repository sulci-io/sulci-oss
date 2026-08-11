# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
tests/_fake_embedder.py
────────────────────────
A deterministic, dependency-free :class:`sulci.embeddings.Embedder`.

WHY THIS EXISTS
───────────────
``tests/test_integrations_langchain.py`` and ``..._llamaindex.py`` build a
real ``Cache``, which loads ``all-MiniLM-L6-v2`` from huggingface.co. On any
machine or CI runner without model access those suites do not fail — they
*error*, 37 of 56, and the distinction between "blocked" and "passing" is
exactly the one this project has been burned by. The 2026-08-05 reconciliation
could not upgrade a marker for that reason.

The three integration surfaces added in v0.9.0 (MCP, LiteLLM, proxy) are
adapter code: their job is translating between somebody else's contract and
``Cache``'s. That is fully testable without a real embedder, so these suites
inject this one and run anywhere, offline, in about a second.

⚠️  This is NOT a substitute for the embedder-backed suites. It proves the
    plumbing, not the retrieval quality. Bag-of-characters similarity is not
    semantic similarity — do not draw hit-rate conclusions from it.
"""

from __future__ import annotations

import math
import re
import zlib

_TOKEN = re.compile(r"[a-z0-9]+")
_DIM = 64

# zlib.crc32, not builtin hash(): PYTHONHASHSEED randomises str hashing per
# process, so a persisted SQLite store written by one process would not match
# vectors computed by the next. Stable hashing keeps the fixture honest.
def _bucket(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) % _DIM


class FakeEmbedder:
    """Hashed bag-of-words, L2-normalised. Same text -> same vector."""

    @property
    def dimension(self) -> int:
        return _DIM

    def embed(self, text: str) -> list:
        vec = [0.0] * _DIM
        for tok in _TOKEN.findall((text or "").lower()):
            vec[_bucket(tok)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # Never emit a zero vector — cosine similarity is undefined and
            # backends differ on what they do with it.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_batch(self, texts: list) -> list:
        return [self.embed(t) for t in texts]
