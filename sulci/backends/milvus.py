# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/backends/milvus.py
Milvus Lite backend — embedded vector DB, no server required.

Install  : pip install "sulci[milvus]"
Free tier: Milvus Lite (embedded) or Zilliz Cloud free tier
Latency  : 5–20 ms embedded, <5 ms Zilliz Cloud
"""
from __future__ import annotations
import time
from typing import Optional


class MilvusBackend:

    COLLECTION = "sulci"

    def __init__(
        self,
        db_path: str           = "./sulci_milvus.db",
        uri:     Optional[str] = None,
        token:   Optional[str] = None,
    ):
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError(
                "pymilvus not found.\n"
                "Install with: pip install \"sulci[milvus]\""
            )
        self._client = (
            MilvusClient(uri=uri, token=token) if uri
            else MilvusClient(db_path)
        )
        self._ready = False

    def _ensure(self, dim: int):
        if self._ready:
            return
        if not self._client.has_collection(self.COLLECTION):
            self._client.create_collection(
                collection_name = self.COLLECTION,
                dimension       = dim,
                metric_type     = "COSINE",
                auto_id         = True,
            )
        self._ready = True

    def store(
        self,
        key: str, query: str, response: str, embedding: list[float],
        user_id: Optional[str] = None, expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._ensure(len(embedding))
        self._client.insert(self.COLLECTION, [{
            "vector":   embedding,
            "query":    query,
            "response": response,
            "user_id":  user_id or "global",
            "expires":  expires or 0.0,
        }])

    def search(
        self,
        embedding: list[float], threshold: float,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        if not self._ready:
            return None, 0.0
        now     = now or time.time()
        filter_ = f'user_id == "{user_id}"' if user_id else ""
        results = self._client.search(
            collection_name = self.COLLECTION,
            data            = [embedding],
            limit           = 5,
            output_fields   = ["response", "expires"],
            filter          = filter_,
        )
        for r in results[0]:
            entity = r.get("entity", {})
            exp    = entity.get("expires", 0)
            if exp and now > exp:
                continue
            sim = r.get("distance", 0)
            if sim >= threshold:
                return entity.get("response"), sim
        return None, 0.0

    def clear(self) -> None:
        self._client.drop_collection(self.COLLECTION)
        self._ready = False
