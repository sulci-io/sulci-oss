# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/backends/chroma.py
ChromaDB backend — recommended for getting started.

Install  : pip install "sulci[chroma]"
Free tier: self-hosted (unlimited) or Chroma Cloud (1 M embeddings free)
Latency  : ~4 ms at 10k entries
"""
from __future__ import annotations
import time
from typing import Optional


class ChromaBackend:

    def __init__(self, db_path: str = "./sulci_db"):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB not found.\n"
                "Install with: pip install \"sulci[chroma]\""
            )
        self._client = chromadb.PersistentClient(path=db_path)
        self._col    = self._client.get_or_create_collection(
            name     = "sulci",
            metadata = {"hnsw:space": "cosine"},
        )

    def store(
        self,
        key: str, query: str, response: str, embedding: list[float],
        user_id: Optional[str] = None, expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._col.upsert(
            ids        = [key],
            embeddings = [embedding],
            documents  = [response],
            metadatas  = [{
                "query":   query,
                "expires": expires or 0.0,
                "user_id": user_id or "global",
                **(metadata or {}),
            }],
        )

    def search(
        self,
        embedding: list[float], threshold: float,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        now   = now or time.time()
        where = {"user_id": user_id} if user_id else None
        try:
            res = self._col.query(
                query_embeddings = [embedding],
                n_results        = 5,
                where            = where,
                include          = ["documents", "distances", "metadatas"],
            )
        except Exception:
            return None, 0.0

        for doc, dist, meta in zip(
            res["documents"][0],
            res["distances"][0],
            res["metadatas"][0],
        ):
            if meta.get("expires") and now > meta["expires"]:
                continue
            sim = 1.0 - dist          # cosine distance → similarity
            if sim >= threshold:
                return doc, sim
        return None, 0.0

    def clear(self) -> None:
        self._client.delete_collection("sulci")
        self._col = self._client.get_or_create_collection(
            name     = "sulci",
            metadata = {"hnsw:space": "cosine"},
        )
