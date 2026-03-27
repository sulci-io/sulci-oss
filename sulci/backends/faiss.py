# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/backends/faiss.py
FAISS backend — fastest local option, MIT license, zero infra.

Install  : pip install "sulci[faiss]"
Free tier: fully self-hosted, no limits
Latency  : <2 ms at 100k entries (HNSW index)
"""
from __future__ import annotations
import os, pickle, time
from typing import Optional


class FAISSBackend:

    def __init__(self, db_path: str = "./sulci_faiss"):
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "faiss-cpu not found.\n"
                "Install with: pip install \"sulci[faiss]\""
            )
        self._faiss = faiss
        self._np    = np
        self._path  = db_path
        self._index = None
        self._meta: list[dict] = []
        os.makedirs(db_path, exist_ok=True)
        self._load()

    def _idx_path(self):  return os.path.join(self._path, "index.faiss")
    def _meta_path(self): return os.path.join(self._path, "meta.pkl")

    def _load(self):
        if os.path.exists(self._meta_path()):
            with open(self._meta_path(), "rb") as f:
                self._meta = pickle.load(f)
        if os.path.exists(self._idx_path()):
            self._index = self._faiss.read_index(self._idx_path())

    def _save(self):
        self._faiss.write_index(self._index, self._idx_path())
        with open(self._meta_path(), "wb") as f:
            pickle.dump(self._meta, f)

    def _ensure(self, dim: int):
        if self._index is None:
            self._index = self._faiss.IndexHNSWFlat(dim, 32)
            self._index.hnsw.efSearch = 64

    def store(
        self,
        key: str, query: str, response: str, embedding: list[float],
        user_id: Optional[str] = None, expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._ensure(len(embedding))
        vec = self._np.array([embedding], dtype="float32")
        self._index.add(vec)
        self._meta.append({
            "key": key, "query": query, "response": response,
            "user_id": user_id or "global",
            "expires": expires or 0.0,
            **(metadata or {}),
        })
        self._save()

    def search(
        self,
        embedding: list[float], threshold: float,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        if self._index is None or self._index.ntotal == 0:
            return None, 0.0
        now = now or time.time()
        vec = self._np.array([embedding], dtype="float32")
        k   = min(5, self._index.ntotal)
        D, I= self._index.search(vec, k)
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            meta = self._meta[idx]
            if meta.get("expires") and now > meta["expires"]:
                continue
            if user_id and meta.get("user_id") != user_id:
                continue
            sim = float(1.0 - dist)
            if sim >= threshold:
                return meta["response"], sim
        return None, 0.0

    def clear(self) -> None:
        self._index = None
        self._meta  = []
        for p in [self._idx_path(), self._meta_path()]:
            if os.path.exists(p):
                os.remove(p)
