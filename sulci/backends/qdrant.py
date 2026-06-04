# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/backends/qdrant.py
Qdrant backend — best performance for production.

Install  : pip install "sulci[qdrant]"
Free tier: 1 GB cluster free forever at cloud.qdrant.io
Latency  : <5 ms local, sub-ms with quantization
"""
from __future__ import annotations
import time, uuid
from typing import Optional


class QdrantBackend:
    #: True if this backend enforces tenant_id partition isolation.
    #: When True, search() must not return entries with mismatched tenant_id.
    ENFORCES_TENANT_ISOLATION: bool = True

    COLLECTION = "sulci"

    def __init__(
        self,
        db_path:   str           = "./sulci_qdrant",
        url:       Optional[str] = None,
        api_key:   Optional[str] = None,
        dimension: int           = 384,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client not found.\n"
                "Install with: pip install \"sulci[qdrant]\""
            )
        self._client = (
            QdrantClient(url=url, api_key=api_key) if url
            else QdrantClient(path=db_path)
        )
        existing = [c.name for c in self._client.get_collections().collections]
        if self.COLLECTION not in existing:
            self._client.create_collection(
                collection_name = self.COLLECTION,
                vectors_config  = VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def store(
        self,
        key: str, query: str, response: str, embedding: list[float],
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name = self.COLLECTION,
            points = [PointStruct(
                id      = str(uuid.uuid4()),
                vector  = embedding,
                payload = {
                    "key": key, "query": query, "response": response,
                    "tenant_id": tenant_id or "global",
                    "user_id": user_id or "global",
                    "expires": expires or 0.0,
                    # v0.7.2 — store-time timestamp so consumers (e.g.
                    # sulci-platform top_queries) can report an honest
                    # "last seen" for entries that have never been served,
                    # instead of fabricating one at aggregation time.
                    "created": time.time(),
                    **(metadata or {}),
                },
            )],
        )

    def search(
        self,
        embedding: list[float], threshold: float,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        resp, score, _matched = self.search_match(
            embedding, threshold,
            tenant_id=tenant_id, user_id=user_id, now=now,
        )
        return resp, score

    def search_match(
        self,
        embedding: list[float], threshold: float,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float, Optional[str]]:
        """
        Like search(), but also returns the STORED query text of the
        matched entry as the third element (None on miss).

        v0.7.2 — optional backend extension, feature-detected by
        Cache.get via hasattr(). The matched query identifies which
        cached entry was served, enabling per-entry hit counting by
        downstream consumers (CacheEvent.matched_query_hash). search()
        keeps its 2-tuple contract by delegating here, so the Backend
        protocol and the other backends are untouched.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        now = now or time.time()

        # Build payload filter conditions. Tenant isolation is a hard
        # boundary — entries from other tenants must never be returned,
        # even if their similarity exceeds threshold. The store path
        # writes tenant_id="global" when None is passed; the read path
        # mirrors that by filtering to "global" for None — otherwise
        # an unscoped read would silently see entries from named tenants
        # and break the operational migration to multi-tenancy.
        must_conditions = [
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id if tenant_id is not None else "global"),
            ),
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id if user_id is not None else "global"),
            ),
        ]
        filter_ = Filter(must=must_conditions)

        results = self._client.query_points(
            collection_name = self.COLLECTION,
            query           = embedding,
            query_filter    = filter_,
            limit           = 5,
            with_payload    = True,
        )
        for r in results.points:
            p = r.payload or {}
            if p.get("expires") and now > p["expires"]:
                continue
            if r.score >= threshold:
                return p.get("response"), r.score, p.get("query")
        return None, 0.0, None

    def clear(self) -> None:
        # Delete all points but keep the collection (and its HNSW index)
        # intact. qdrant-client 1.x raises ValueError on subsequent
        # operations against a missing collection, so a recreate-on-clear
        # would also work but costs a full index rebuild on first store.
        from qdrant_client.models import Filter
        try:
            self._client.delete(
                collection_name = self.COLLECTION,
                points_selector = Filter(must=[]),
            )
        except Exception:
            # Empty collection or other transient error — clear is
            # idempotent so we swallow.
            pass
