# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

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

from sulci._metadata import without_reserved
from sulci.backends._scope import scoped_id


class ChromaBackend:
    #: True if this backend enforces tenant_id partition isolation.
    #: When True, search() must not return entries with mismatched tenant_id.
    ENFORCES_TENANT_ISOLATION: bool = True     # 0.9.3: search() filters on tenant AND user

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
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._col.upsert(
            # 0.9.3: scoped to tenant and user. `key` comes from the query alone, so upserting by it let
            # another tenant or user storing the same question replace this entry.
            ids        = [scoped_id(key, tenant_id, user_id)],
            embeddings = [embedding],
            documents  = [response],
            metadatas  = [{
                # 0.9.3: metadata first and without reserved names; the entry's fields win.
                **without_reserved(metadata),
                "query":     query,
                "expires":   expires or 0.0,
                "user_id":   user_id or "global",
                "tenant_id": tenant_id or "global",
            }],
        )

    def search(
        self,
        embedding: list[float], threshold: float,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        now   = now or time.time()
        # 0.9.3: a lookup reads its own scope only — tenant AND user, None meaning the shared scope "global".
        # In the query's filter, so other scopes' copies of a question cannot crowd this one out.
        where = {"$and": [{"tenant_id": tenant_id or "global"}, {"user_id": user_id or "global"}]}
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
