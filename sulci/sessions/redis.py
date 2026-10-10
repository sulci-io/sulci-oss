# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sessions/redis.py — RedisSessionStore (v0.5.0)

Redis-backed session store for horizontally-scaled deployments.
Multiple gateway replicas share session history via Redis Lists.

Key schema (0.9.3; the default prefix changed from "sulci:session:" with it):
    {prefix}s:{session_id}   — Redis List of JSON-encoded vectors, oldest first
    {prefix}i:               — Sorted set: every session id this store holds
    {prefix}i:{scope}        — Sorted set: the session ids of one tenant scope or one user scope

``Cache`` hands the store scoped ids (``sulci.sessions.scope``). Each ``append`` writes the
session and records its id in the store's index and in its tenant's and user's indexes in
one MULTI/EXEC transaction (nothing interleaves with it), scored by when the session expires
plus a grace period, and drops index entries whose session has expired — so an index lists
the live sessions, not every id ever written. ``clear_scope`` and ``clear_all`` remove exactly
the indexed sessions, in transactions, without a keyspace SCAN (a SCAN-based erasure was
measured on the managed platform saturating Redis); a write that races an erasure is either
removed by it or stays indexed. The ``s:`` and ``i:`` namespaces cannot collide, whatever id
a caller uses. The grace period (``_INDEX_GRACE_SECONDS``) also absorbs clock differences
between the processes sharing the store; it must exceed them.

Needs Redis 6.2 or later (``ZADD … GT``). Not for Redis Cluster: one write spans the session
key and three index keys.

Install:
    pip install "sulci[redis]"  (or "sulci[all]")
"""
from __future__ import annotations
import json
import time
from typing import Iterable, Optional, List, Sequence

from sulci.sessions.protocol import SessionStore
from sulci.sessions.scope import scopes_of

#: How long an index entry outlives its session, so an index never forgets a key that still exists.
_INDEX_GRACE_SECONDS = 300
_NEVER = float("inf")
_BATCH = 500


def _text(v) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else v


class RedisSessionStore(SessionStore):
    """
    Redis-backed session store.

    Each session is a Redis List where each entry is a JSON-encoded vector.
    RPUSH appends; LTRIM keeps history bounded.

    Args:
        redis_client: Configured redis.Redis client instance
        key_prefix:   Prefix for Redis keys (default "sulci:sessions:"; 0.9.3 — keys written by
                      earlier versions under "sulci:session:" are never read)
        ttl_seconds:  TTL for session keys. Default None = no expiry.
                      Set to e.g. 86400 for 24-hour session lifetime.
    """

    def __init__(
        self,
        redis_client,
        key_prefix: str = "sulci:sessions:",
        ttl_seconds: Optional[int] = None,
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

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}s:{session_id}"

    def _index(self, scope: str = "") -> str:
        return f"{self._prefix}i:{scope}"

    def get(self, session_id: str) -> List[Sequence[float]]:
        raw = self._redis.lrange(self._key(session_id), 0, -1)
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
        indexes = [self._index()] + [self._index(s) for s in scopes_of(session_id)]
        now = time.time()
        until = now + self._ttl + _INDEX_GRACE_SECONDS if self._ttl else _NEVER
        pipe = self._redis.pipeline(transaction=True)
        pipe.rpush(key, json.dumps(list(vector)))
        pipe.ltrim(key, -max_turns, -1)   # Keep last max_turns entries
        if self._ttl:
            pipe.expire(key, self._ttl)
        for idx in indexes:
            # GT: a writer whose clock is behind cannot move an entry's expiry earlier. Indexes themselves never
            # expire: stores with different TTLs may share a prefix, and an empty sorted set disappears anyway.
            pipe.zadd(idx, {session_id: until}, gt=True)
            pipe.zremrangebyscore(idx, "-inf", now)            # sessions that have expired
        pipe.execute()

    def _remove(self, session_ids: Iterable[str]) -> int:
        """Delete these sessions and their index entries in one transaction; return how many existed."""
        ids = list(session_ids)
        if not ids:
            return 0
        pipe = self._redis.pipeline(transaction=True)
        pipe.delete(*[self._key(s) for s in ids])
        pipe.zrem(self._index(), *ids)
        for sid in ids:
            for scope in scopes_of(sid):
                pipe.zrem(self._index(scope), sid)
        return int(pipe.execute()[0])

    def _members(self, index: str):
        for member, _score in self._redis.zscan_iter(index, count=_BATCH):
            yield _text(member)

    def _remove_indexed(self, index: str) -> int:
        n, batch = 0, []
        for member in self._members(index):
            batch.append(member)
            if len(batch) >= _BATCH:
                n += self._remove(batch); batch = []
        return n + self._remove(batch)

    def clear(self, session_id: str) -> None:
        self._remove([session_id])

    def clear_scope(self, scope: str) -> int:
        return self._remove_indexed(self._index(scope))

    def clear_all(self) -> int:
        return self._remove_indexed(self._index())

    def summary(self, session_id: Optional[str] = None) -> dict:
        if session_id is not None:
            turns = self._redis.llen(self._key(session_id))
            return {
                "sessions": 1 if turns > 0 else 0,
                "total_turns": turns,
                "avg_turns": float(turns),
            }
        # The store's own index, never a keyspace SCAN; an indexed session that expired counts as none.
        sessions = total_turns = 0
        for member in self._members(self._index()):
            turns = self._redis.llen(self._key(member))
            if turns:
                sessions += 1
                total_turns += turns
        avg = total_turns / sessions if sessions else 0.0
        return {"sessions": sessions, "total_turns": total_turns, "avg_turns": avg}
