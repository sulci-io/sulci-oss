# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/backends/redis.py
Redis backend — sub-millisecond latency, battle-tested at scale.

Install  : pip install "sulci[redis]"
Free tier: Upstash Redis — 10,000 requests/day free
Latency  : <1 ms local Redis, ~5 ms Upstash

Usage:
    # Local Redis
    cache = Cache(backend="redis")

    # Upstash
    from sulci.backends.redis import RedisBackend
    backend = RedisBackend(url="rediss://default:TOKEN@host.upstash.io:6380")
"""
from __future__ import annotations
import time, struct, math
from typing import Optional


class RedisBackend:
    #: True if this backend enforces tenant_id partition isolation.
    #: When True, search() must not return entries with mismatched tenant_id.
    ENFORCES_TENANT_ISOLATION: bool = False

    def __init__(
        self,
        db_path:    str = "./sulci_db",
        url:        str = "redis://localhost:6379",
        key_prefix: str = "sulci:",
    ):
        try:
            import redis as redis_lib
        except ImportError:
            raise ImportError(
                "redis not found.\n"
                "Install with: pip install \"sulci[redis]\""
            )
        self._redis  = redis_lib.from_url(url, decode_responses=False)
        # Namespace prefix for all keys this instance writes/reads. Default
        # "sulci:" matches v0.4.x behavior. Tests and multi-tenant deployments
        # can pass a unique prefix (e.g. "sulci:test:abc12345:") to coexist
        # on a shared Redis daemon without cross-instance interference.
        self._prefix = key_prefix

    def _key(self, k: str) -> str:
        return f"{self._prefix}{k}"

    def _pack(self, vec: list[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    def _unpack(self, blob: bytes) -> list[float]:
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na  = math.sqrt(sum(x * x for x in a)) or 1.0
        nb  = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def store(
        self,
        key: str, query: str, response: str, embedding: list[float],
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        rkey = self._key(key)
        self._redis.hset(rkey, mapping={
            "query":    query.encode(),
            "response": response.encode(),
            "user_id":  (user_id or "global").encode(),
            "expires":  str(expires or 0.0).encode(),
            "embedding":self._pack(embedding),
        })
        if expires:
            self._redis.expireat(rkey, int(expires))

    def search(
        self,
        embedding: list[float], threshold: float,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        now       = now or time.time()
        best_sim  = 0.0
        best_resp = None
        cursor    = 0
        while True:
            cursor, keys = self._redis.scan(cursor, match=f"{self._prefix}*", count=200)
            for rkey in keys:
                try:
                    data = self._redis.hgetall(rkey)
                    exp  = float(data.get(b"expires", 0))
                    if exp and now > exp:
                        continue
                    uid = (data.get(b"user_id") or b"global").decode()
                    if user_id and uid != user_id:
                        continue
                    stored = self._unpack(data[b"embedding"])
                    sim    = self._cosine(embedding, stored)
                    if sim > best_sim:
                        best_sim  = sim
                        best_resp = data[b"response"].decode()
                except Exception:
                    continue
            if cursor == 0:
                break
        if best_sim >= threshold:
            return best_resp, best_sim
        return None, best_sim

    def clear(self) -> None:
        keys = self._redis.keys(f"{self._prefix}*")
        if keys:
            self._redis.delete(*keys)
