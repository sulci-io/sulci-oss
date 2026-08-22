# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sessions/redis.py — RedisSessionStore (v0.5.0)

Redis-backed session store for horizontally-scaled deployments.
Multiple gateway replicas share session history via Redis Lists.

Key schema:
    sulci:session:{session_id}   — Redis List of JSON-encoded vectors

Install:
    pip install "sulci[redis]"  (or "sulci[all]")
"""
from __future__ import annotations
import json
from typing import Optional, List, Sequence

from sulci.sessions.protocol import SessionStore


class RedisSessionStore(SessionStore):
    """
    Redis-backed session store.

    Each session is a Redis List where each entry is a JSON-encoded vector.
    LPUSH appends to the front; LTRIM keeps history bounded.

    Args:
        redis_client: Configured redis.Redis client instance
        key_prefix:   Prefix for Redis keys (default "sulci:session:")
        ttl_seconds:  TTL for session keys. Default None = no expiry.
                      Set to e.g. 86400 for 24-hour session lifetime.
    """

    def __init__(
        self,
        redis_client,
        key_prefix: str = "sulci:session:",
        ttl_seconds: Optional[int] = None,
        tenant_id: Optional[str] = None,
    ):
        try:
            import redis   # noqa: F401
        except ImportError:
            raise ImportError(
                "redis package not installed. "
                'Install with: pip install "sulci[redis]"'
            )
        self._redis = redis_client
        self._prefix = key_prefix
        self._ttl = ttl_seconds
        self._tenant_id = tenant_id

    def _key(self, session_id: str) -> str:
        if self._tenant_id:
            return f"{self._prefix}{self._tenant_id}:{session_id}"
        return f"{self._prefix}{session_id}"

    def get(self, session_id: str) -> List[Sequence[float]]:
        key = self._key(session_id)
        # LRANGE returns entries with most-recent-first if we LPUSH.
        # We store oldest-first by RPUSH, so LRANGE 0 -1 returns chronological.
        raw = self._redis.lrange(key, 0, -1)
        if not raw:
            return []
        return [json.loads(item) for item in raw]

    def append(
        self,
        session_id: str,
        vector: Sequence[float],
        max_turns: int = 8,
    ) -> None:
        key = self._key(session_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, json.dumps(list(vector)))
        pipe.ltrim(key, -max_turns, -1)   # Keep last max_turns entries
        if self._ttl:
            pipe.expire(key, self._ttl)
        pipe.execute()

    def clear(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))

    def summary(self, session_id: Optional[str] = None) -> dict:
        if session_id is not None:
            turns = self._redis.llen(self._key(session_id))
            return {
                "sessions": 1 if turns > 0 else 0,
                "total_turns": turns,
                "avg_turns": float(turns),
            }

        # Aggregate — uses SCAN, could be expensive on huge Redis instances
        cursor = 0
        sessions = 0
        total_turns = 0
        pattern = (f"{self._prefix}{self._tenant_id}:*"
                   if self._tenant_id else f"{self._prefix}*")
        while True:
            cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=100)
            for k in keys:
                sessions += 1
                total_turns += self._redis.llen(k)
            if cursor == 0:
                break

        avg = total_turns / sessions if sessions else 0.0
        return {"sessions": sessions, "total_turns": total_turns, "avg_turns": avg}
