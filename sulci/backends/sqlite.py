# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/backends/sqlite.py
SQLite backend — zero infrastructure, embedded database.

Install  : pip install "sulci[sqlite]"
Free tier: fully embedded, no server, no cost ever
Latency  : 5–50 ms (slower than FAISS/Chroma but needs nothing)
Best for : prototyping, low-traffic apps, edge deployments, CI/CD
"""
from __future__ import annotations
import os, sqlite3, json, struct, time, math, warnings
from typing import Optional


# PRAGMA user_version of the current schema.
#   0 — through 0.9.1: `key` alone was UNIQUE, and no tenant_id column.
#   1 — 0.9.2: a row is unique on (key, tenant_id, user_id).
SCHEMA_VERSION = 1

_CREATE_TABLE = """
    CREATE TABLE {name} (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        key       TEXT NOT NULL,
        query     TEXT NOT NULL,
        response  TEXT NOT NULL,
        embedding BLOB NOT NULL,
        tenant_id TEXT NOT NULL DEFAULT 'global',
        user_id   TEXT NOT NULL DEFAULT 'global',
        expires   REAL NOT NULL DEFAULT 0,
        created   REAL NOT NULL,
        metadata  TEXT NOT NULL DEFAULT '{{}}',
        UNIQUE (key, tenant_id, user_id)
    )
"""
_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_user    ON cache(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires)",
)


class SQLiteBackend:
    #: True if this backend enforces tenant_id partition isolation.
    #: When True, search() must not return entries with mismatched tenant_id.
    #: tenant_id is part of a row's identity here, so two tenants never
    #: overwrite each other's entry, but search() does not filter on it.
    ENFORCES_TENANT_ISOLATION: bool = False

    def __init__(self, db_path: str = "./sulci_db"):
        os.makedirs(db_path, exist_ok=True)
        db_file     = os.path.join(db_path, "sulci.db")
        self._db_file = db_file
        self._conn  = sqlite3.connect(db_file, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self):
        # A current file needs no write lock (and may be read-only).
        conn = self._conn
        if conn.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
            return
        # Explicit transaction: BEGIN IMMEDIATE takes the write lock before
        # the version is re-read, so two processes opening the same 0.9.1
        # file cannot both migrate it.
        discarded = 0
        prev_isolation, conn.isolation_level = conn.isolation_level, None
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                exists  = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cache'"
                ).fetchone()
                if not exists:
                    conn.execute(_CREATE_TABLE.format(name="cache"))
                elif version < SCHEMA_VERSION:
                    discarded = self._migrate_v0(conn)
                for stmt in _CREATE_INDEXES:
                    conn.execute(stmt)
                if version < SCHEMA_VERSION:
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.isolation_level = prev_isolation
        if discarded:
            warnings.warn(
                f"sulci: upgraded the SQLite cache schema at {self._db_file}; "
                f"discarded {discarded} entr{'y' if discarded == 1 else 'ies'} "
                "written before 0.9.2, whose scope could not be reliably "
                "determined. The cache starts cold and re-populates on misses.",
                RuntimeWarning,
                stacklevel=3,
            )

    @staticmethod
    def _migrate_v0(conn: sqlite3.Connection) -> int:
        """
        Replace a pre-0.9.2 table with an empty one keyed on
        (key, tenant_id, user_id). Returns the number of rows discarded.

        Every old row is discarded, not only those carrying a user_id: under
        the old key no row's scope can be reliably determined, and a row
        stored without a user_id may hold a response written for one. It is
        a cache, so a cold start is the safe cost.
        """
        discarded = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        conn.execute("DROP TABLE cache")
        conn.execute(_CREATE_TABLE.format(name="cache"))
        return discarded

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
        # The key is derived from the query text alone (Cache.set), so the
        # scope must be part of the conflict target: the same query stored
        # for two users or two tenants is two rows, never one overwritten.
        self._conn.execute("""
            INSERT INTO cache
                (key, query, response, embedding, tenant_id, user_id,
                 expires, created, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key, tenant_id, user_id) DO UPDATE SET
                response  = excluded.response,
                embedding = excluded.embedding,
                expires   = excluded.expires
        """, (
            key, query, response,
            self._pack(embedding),
            tenant_id or "global",
            user_id or "global",
            expires or 0.0,
            time.time(),
            json.dumps(metadata or {}),
        ))
        self._conn.commit()

    def search(
        self,
        embedding: list[float], threshold: float,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        now       = now or time.time()
        rows      = self._conn.execute(
            "SELECT response, embedding, expires, user_id FROM cache"
        ).fetchall()
        best_sim  = 0.0
        best_resp = None
        for row in rows:
            if row["expires"] and now > row["expires"]:
                continue
            if user_id and row["user_id"] != user_id:
                continue
            sim = self._cosine(embedding, self._unpack(row["embedding"]))
            if sim > best_sim:
                best_sim  = sim
                best_resp = row["response"]
        if best_sim >= threshold:
            return best_resp, best_sim
        return None, best_sim

    def clear(self) -> None:
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()
