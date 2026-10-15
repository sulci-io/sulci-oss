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

from sulci._metadata import checked_metadata
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
                # 0.9.3: metadata is refused if it uses a reserved name, and goes first; the entry's fields win.
                **checked_metadata(metadata),
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

    def delete_user(self, user_id: str, *, tenant_id: Optional[str] = None) -> int:
        """0.10.0: remove every entry filed under (``tenant_id or "global"``, ``user_id or "global"``) — expired ones
        too: Chroma keeps them until deleted — and return how many (the ``Backend`` protocol's contract).

        The ids are read a page at a time with the same filter ``search`` uses, then deleted by id, until none is
        left; the count is the number of ids read and deleted. Chroma's own answer is not used: measured on
        chromadb 1.5.9, ``delete(...)`` answers ``{"deleted": N}`` with N the number of ids passed, existing or not,
        and the 0.4 / 0.5 releases ``sulci[chroma]`` allows answer nothing. So the count is exact for one erasure
        at a time; two erasures of the same user running at once (or a ``clear()`` beside one) may both count the
        same entries. If an id already deleted comes back — Chroma did not delete it, or the same entry was stored
        again for the user while this ran — it raises rather than loop; repeat the call. Entries written before
        sulci 0.4.0 have no ``tenant_id`` and are neither found by ``search`` nor erased here; ``clear()`` removes
        them. Deleted entries stay in Chroma's files until it compacts them."""
        where = {"$and": [{"tenant_id": tenant_id or "global"}, {"user_id": user_id or "global"}]}
        try:
            page = min(1000, int(self._client.get_max_batch_size()))
        except Exception:
            page = 1000
        removed, deleted = 0, set()
        while True:
            ids = self._col.get(where=where, include=[], limit=page)["ids"]
            if not ids:
                return removed
            if deleted.intersection(ids):
                raise RuntimeError("ChromaBackend.delete_user: an entry already deleted was found again (Chroma did "
                                   "not delete it, or it was stored again while the erasure ran); the erasure did "
                                   "not complete — repeat the call")
            self._col.delete(ids=ids)
            deleted.update(ids)
            removed += len(ids)

    def clear(self) -> None:
        self._client.delete_collection("sulci")
        self._col = self._client.get_or_create_collection(
            name     = "sulci",
            metadata = {"hnsw:space": "cosine"},
        )
