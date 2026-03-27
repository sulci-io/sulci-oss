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
                    "user_id": user_id or "global",
                    "expires": expires or 0.0,
                    **(metadata or {}),
                },
            )],
        )

    def search(
        self,
        embedding: list[float], threshold: float,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        now     = now or time.time()
        filter_ = Filter(must=[FieldCondition(
            key="user_id", match=MatchValue(value=user_id)
        )]) if user_id else None

        results = self._client.search(
            collection_name = self.COLLECTION,
            query_vector    = embedding,
            query_filter    = filter_,
            limit           = 5,
            with_payload    = True,
        )
        for r in results:
            p = r.payload or {}
            if p.get("expires") and now > p["expires"]:
                continue
            if r.score >= threshold:
                return p.get("response"), r.score
        return None, 0.0

    def clear(self) -> None:
        self._client.delete_collection(self.COLLECTION)
