# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/backends/sqlite.py
SQLite backend — zero infrastructure, embedded database.

Install  : pip install "sulci[sqlite]"
Free tier: fully embedded, no server, no cost ever
Latency  : 5–50 ms (slower than FAISS/Chroma but needs nothing)
Best for : prototyping, low-traffic apps, edge deployments, CI/CD
"""
from __future__ import annotations
import os, sqlite3, json, struct, time, math
from typing import Optional


class SQLiteBackend:

    def __init__(self, db_path: str = "./sulci_db"):
        os.makedirs(db_path, exist_ok=True)
        db_file     = os.path.join(db_path, "sulci.db")
        self._conn  = sqlite3.connect(db_file, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS cache (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                key       TEXT UNIQUE NOT NULL,
                query     TEXT NOT NULL,
                response  TEXT NOT NULL,
                embedding BLOB NOT NULL,
                user_id   TEXT NOT NULL DEFAULT 'global',
                expires   REAL NOT NULL DEFAULT 0,
                created   REAL NOT NULL,
                metadata  TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_user    ON cache(user_id);
            CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires);
        """)
        self._conn.commit()

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
        user_id: Optional[str] = None, expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._conn.execute("""
            INSERT INTO cache
                (key, query, response, embedding, user_id, expires, created, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                response  = excluded.response,
                embedding = excluded.embedding,
                expires   = excluded.expires
        """, (
            key, query, response,
            self._pack(embedding),
            user_id or "global",
            expires or 0.0,
            time.time(),
            json.dumps(metadata or {}),
        ))
        self._conn.commit()

    def search(
        self,
        embedding: list[float], threshold: float,
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
