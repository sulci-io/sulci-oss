# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/backends/faiss.py
FAISS backend — fastest local option, MIT license, zero infra.

Install  : pip install "sulci[faiss]"
Free tier: fully self-hosted, no limits
Latency  : a lookup searches the caller's own (tenant, user) entries exactly (0.9.3); the HNSW index holds
           the vectors. An approximate search over every scope let other scopes crowd a caller's entry out.
"""
from __future__ import annotations
import os, pickle, threading, time
from typing import Optional

from sulci._metadata import checked_metadata


class FAISSBackend:
    #: True if this backend enforces tenant_id partition isolation.
    #: When True, search() must not return entries with mismatched tenant_id.
    ENFORCES_TENANT_ISOLATION: bool = True     # 0.9.3: search() filters on tenant AND user

    # 0.10.0 (oss-0022): store, search and clear — every call that reads or changes the index, the entries or the files
    # once the object exists — hold self._lock, so calls from several threads run one at a time (delete_user, when FAISS
    # gains it, holds it too). An RLock, so a test can ask whether the calling thread holds it. AsyncCache runs each
    # backend call on a thread of its own, and neither a faiss index nor this class's lists may be changed by two
    # threads at once. Measured on main (oss-0021, faiss-cpu 1.15.1, Linux): 400 concurrent AsyncCache.aset calls
    # aborted the Python process ("corrupted size vs. prev_size", "double free or corruption") or raised in 4 of 5 runs,
    # and the one that finished answered 4 of 20 lookups with another entry's response; 8 threads calling store()
    # directly aborted it in 5 of 5. One process per db_path still: two processes, or two FAISSBackend objects on one
    # directory, each write the files over the other.

    def __init__(self, db_path: str = "./sulci_faiss"):
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "faiss-cpu not found.\n"
                "Install with: pip install \"sulci[faiss]\""
            )
        self._faiss = faiss
        self._np    = np
        self._path  = db_path
        self._index = None
        self._meta: list[dict] = []
        self._scopes: dict = {}     # 0.9.3: (tenant_id, user_id) -> positions of that scope's entries
        self._lock  = threading.RLock()
        os.makedirs(db_path, exist_ok=True)
        self._load()

    def _idx_path(self):  return os.path.join(self._path, "index.faiss")
    def _meta_path(self): return os.path.join(self._path, "meta.pkl")

    def _load(self):
        if os.path.exists(self._meta_path()):
            with open(self._meta_path(), "rb") as f:
                self._meta = pickle.load(f)
            self._index_scopes()
        if os.path.exists(self._idx_path()):
            self._index = self._faiss.read_index(self._idx_path())

    def _index_scopes(self):
        """Rebuild the scope map from the stored entries; one without a stored scope belongs to none."""
        self._scopes = {}
        for i, m in enumerate(self._meta):
            if "tenant_id" in m and "user_id" in m:
                self._scopes.setdefault((m["tenant_id"], m["user_id"]), []).append(i)

    def _save(self):
        self._faiss.write_index(self._index, self._idx_path())
        with open(self._meta_path(), "wb") as f:
            pickle.dump(self._meta, f)

    def _ensure(self, dim: int):
        if self._index is None:
            self._index = self._faiss.IndexHNSWFlat(dim, 32)
            self._index.hnsw.efSearch = 64

    def store(
        self,
        key: str, query: str, response: str, embedding: list[float],
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        with self._lock:
            metadata = checked_metadata(metadata)     # 0.9.3: before anything — _ensure creates the index
            self._ensure(len(embedding))
            vec = self._np.array([embedding], dtype="float32")
            self._index.add(vec)
            self._meta.append({
                # 0.9.3: metadata is refused if it uses a reserved name, and goes first; the entry's fields win.
                **checked_metadata(metadata),
                "key": key, "query": query, "response": response,
                "tenant_id": tenant_id or "global",
                "user_id": user_id or "global",
                "expires": expires or 0.0,
            })
            self._scopes.setdefault((tenant_id or "global", user_id or "global"), []).append(len(self._meta) - 1)
            self._save()

    def search(
        self,
        embedding: list[float], threshold: float,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return None, 0.0
            now = now or time.time()
            # 0.9.3: a lookup reads its own scope only — tenant AND user, None meaning the shared scope "global" —
            # and searches that scope's vectors EXACTLY (the index's squared-L2 metric), so no number of entries in
            # other scopes can crowd this one out. An entry without a stored scope matches no scope.
            scope = (tenant_id or "global", user_id or "global")
            ids   = self._scopes.get(scope)
            if not ids:
                return None, 0.0
            vecs  = self._index.reconstruct_batch(self._np.array(ids, dtype="int64"))
            dists = ((vecs - self._np.array(embedding, dtype="float32")) ** 2).sum(axis=1)
            order = self._np.argsort(dists)[:5]
            D, I  = [dists[order]], [[ids[j] for j in order]]
            for dist, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self._meta):
                    continue
                meta = self._meta[idx]
                if meta.get("expires") and now > meta["expires"]:
                    continue
                if (meta.get("tenant_id"), meta.get("user_id")) != scope:
                    continue
                sim = float(1.0 - dist)
                if sim >= threshold:
                    return meta["response"], sim
            return None, 0.0

    def clear(self) -> None:
        with self._lock:
            self._index = None
            self._meta  = []
            self._scopes = {}
            for p in [self._idx_path(), self._meta_path()]:
                if os.path.exists(p):
                    os.remove(p)
