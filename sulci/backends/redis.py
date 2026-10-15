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
import re, time, struct, math
from itertools import islice
from typing import Optional

from sulci.backends._scope import scoped_id

#: 0.10.0: the name of an entry's hash after the prefix — 32 hex digits (``scoped_id``, 0.9.3) or 16 (the query
#: digest, before 0.9.3). Every read, ``clear`` and ``delete_user`` take only keys of this shape, under a prefix whose
#: glob characters are escaped. Before, ``SCAN MATCH prefix*`` with the prefix unescaped reached other instances'
#: entries on the same Redis — measured: an instance with prefix ``P[x]:`` served the answer of one with ``Px:``, and
#: ``clear()`` on ``X:`` deleted the entries of ``X:a:`` — and every other key under the prefix (a
#: ``RedisSessionStore`` on its default prefix keeps its lists and sorted sets inside ``sulci:``).
_ENTRY_NAME = re.compile(rb"[0-9a-f]{16}(?:[0-9a-f]{16})?")
_GLOB = re.compile(r"([\\*?\[\]])")


class RedisBackend:
    #: True if this backend enforces tenant_id partition isolation.
    #: When True, search() must not return entries with mismatched tenant_id.
    ENFORCES_TENANT_ISOLATION: bool = True     # 0.9.3: search() filters on tenant AND user
    #: 0.10.0: an entry stored already past its expiry is never held — store() sets EXPIREAT and Redis removes the
    #: key at once. The conformance suite's TestUserErasure reads this to expect that entry not to be counted.
    DROPS_EXPIRED_AT_STORE: bool = True

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
        if not key_prefix:
            # 0.10.0: an empty prefix made every key in the database this backend's to read, clear and erase.
            raise ValueError("RedisBackend: key_prefix must not be empty (default 'sulci:'); end it with a "
                             "separator such as ':' so it is not the beginning of another instance's prefix")
        self._redis  = redis_lib.from_url(url, decode_responses=False)
        # Namespace prefix for all keys this instance writes/reads. Default
        # "sulci:" matches v0.4.x behavior. Tests and multi-tenant deployments
        # can pass a unique prefix (e.g. "sulci:test:abc12345:") to coexist
        # on a shared Redis daemon without cross-instance interference.
        self._prefix = key_prefix

    def _key(self, k: str) -> str:
        return f"{self._prefix}{k}"

    def _entry_keys(self):
        """Every entry's key under this instance's prefix, and nothing else (0.10.0; see ``_ENTRY_NAME``)."""
        prefix = self._prefix.encode()
        for rkey in self._redis.scan_iter(match=_GLOB.sub(r"\\\1", self._prefix) + "*", count=500):
            if rkey.startswith(prefix) and _ENTRY_NAME.fullmatch(rkey, len(prefix)):
                yield rkey

    def _entry_key_pages(self, size: int = 500):
        keys = self._entry_keys()
        while True:
            page = list(islice(keys, size))
            if not page:
                return
            yield page

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
        # 0.9.3: scoped to tenant and user. `key` comes from the query alone, so naming the hash by it let
        # another tenant or user storing the same question replace this entry.
        rkey = self._key(scoped_id(key, tenant_id, user_id))
        self._redis.hset(rkey, mapping={
            "query":    query.encode(),
            "response": response.encode(),
            "tenant_id":(tenant_id or "global").encode(),
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
        scope     = ((tenant_id or "global").encode(), (user_id or "global").encode())
        for keys in self._entry_key_pages(200):
            for rkey in keys:
                try:
                    data = self._redis.hgetall(rkey)
                    exp  = float(data.get(b"expires", 0))
                    if exp and now > exp:
                        continue
                    # 0.9.3: a lookup reads its own scope only — tenant AND user, None meaning "global". An
                    # entry without a stored scope matches no scope.
                    if (data.get(b"tenant_id"), data.get(b"user_id")) != scope:
                        continue
                    stored = self._unpack(data[b"embedding"])
                    sim    = self._cosine(embedding, stored)
                    if sim > best_sim:
                        best_sim  = sim
                        best_resp = data[b"response"].decode()
                except Exception:
                    continue
        if best_sim >= threshold:
            return best_resp, best_sim
        return None, best_sim

    def clear(self) -> None:
        # 0.10.0: this instance's entries only (see _ENTRY_NAME), by SCAN rather than the blocking KEYS.
        for page in self._entry_key_pages():
            self._redis.delete(*page)

    def delete_user(self, user_id: str, *, tenant_id: Optional[str] = None) -> int:
        """0.10.0: remove every entry filed under (``tenant_id or "global"``, ``user_id or "global"``) and return how
        many (the ``Backend`` protocol's contract).

        An entry's hash name is a digest of its scope and key, so no key pattern narrows the search to one user:
        this SCANs this instance's entry keys (``_entry_keys``: the prefix escaped, the rest of the key a digest —
        never another instance's entries, a session store's keys or anything else under the prefix), reads each
        one's ``tenant_id`` and ``user_id`` fields (pipelined) and deletes the matches — O(keys under the prefix),
        not on the lookup path. The count is what ``DEL`` removed. A key named like an entry that is not a hash
        answers ``WRONGTYPE`` and is skipped; any other error propagates, and a failure after some pages leaves
        those pages erased (repeat the call). An entry past its expiry has usually been removed by Redis already
        (``store`` sets ``EXPIREAT``) and so is not counted; one still held is erased and counted. A snapshot or AOF
        written earlier still holds what was deleted. Not for Redis Cluster: one ``DEL`` names many keys."""
        from redis.exceptions import ResponseError
        scope = [(tenant_id or "global").encode(), (user_id or "global").encode()]
        removed = 0
        for keys in self._entry_key_pages():
            pipe = self._redis.pipeline(transaction=False)
            for rkey in keys:
                pipe.hmget(rkey, "tenant_id", "user_id")
            doomed = []
            for rkey, fields in zip(keys, pipe.execute(raise_on_error=False)):
                if isinstance(fields, Exception):
                    if isinstance(fields, ResponseError) and str(fields).startswith("WRONGTYPE"):
                        continue                          # not a hash: not an entry
                    raise fields
                if list(fields) == scope:
                    doomed.append(rkey)
            if doomed:
                removed += self._redis.delete(*doomed)
        return removed
