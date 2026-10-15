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
import logging, os, pickle, threading, time
from typing import Optional

from sulci._metadata import checked_metadata

_log = logging.getLogger("sulci")


class FAISSBackend:
    #: True if this backend enforces tenant_id partition isolation.
    #: When True, search() must not return entries with mismatched tenant_id.
    ENFORCES_TENANT_ISOLATION: bool = True     # 0.9.3: search() filters on tenant AND user

    # 0.10.0 (oss-0022): store, search and clear — and delete_user (oss-0024) — every call that reads or changes the
    # index, the entries or the files once the object exists — hold self._lock, so calls from several threads run one at
    # a time. An RLock, so a test can ask whether the calling thread holds it. AsyncCache runs each backend call on a
    # thread of its own, and neither a faiss index nor this class's lists may be changed by two threads at once.
    # Measured on main (oss-0021, faiss-cpu 1.15.1, Linux): 400 concurrent AsyncCache.aset calls aborted the Python
    # process ("corrupted size vs. prev_size", "double free or corruption") or raised in 4 of 5 runs, and the one that
    # finished answered 4 of 20 lookups with another entry's response; 8 threads calling store() directly aborted it in
    # 5 of 5. One process per db_path still: two processes, or two FAISSBackend objects on one directory, each write the
    # files over the other.

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
        self._finish_interrupted_write()
        if os.path.exists(self._meta_path()):
            with open(self._meta_path(), "rb") as f:
                self._meta = pickle.load(f)
            self._index_scopes()
        if os.path.exists(self._idx_path()):
            self._index = self._faiss.read_index(self._idx_path())
        self._reconcile()

    def _reconcile(self):
        """0.10.0 (oss-0025): entry i is position i of the index, so the two must hold the same number. They can
        disagree in files written before 0.10.0 — a store() stopped between its two files left the index one longer, a
        clear() stopped between them left the entries without an index — after a power cut (store() does not flush),
        or when one file is copied or restored alone; each later store then filed its vector away from its entry
        (measured on main: 4 of 10 lookups answered with another entry's response). Stores only append, and the one
        write that compacts (delete_user) flushes between its two replaces, so the first min(index, entries) of each
        still belong together: the rest is dropped, with a WARNING, and the files are rewritten and flushed."""
        held = self._index.ntotal if self._index is not None else 0
        if held == len(self._meta):
            return
        keep = min(held, len(self._meta))
        _log.warning("sulci FAISSBackend: %s holds %d vectors but %d entries (a write stopped part way before sulci "
                     "0.10.0, a power cut, or a file copied alone); keeping the first %d of each", self._path, held,
                     len(self._meta), keep)
        if keep == 0:
            self._index, self._meta, self._scopes = None, [], {}
            for p in (self._meta_path(), self._idx_path()):
                if os.path.exists(p):
                    os.remove(p)
            self._sync_dir()
            return
        index = self._faiss.IndexHNSWFlat(self._index.d, 32)
        index.hnsw.efSearch = 64
        index.add(self._index.reconstruct_batch(self._np.arange(keep, dtype="int64")))
        self._write_files(index, self._meta[:keep])

    def _index_scopes(self):
        """Rebuild the scope map from the stored entries; one without a stored scope belongs to none."""
        self._scopes = {}
        for i, m in enumerate(self._meta):
            if "tenant_id" in m and "user_id" in m:
                self._scopes.setdefault((m["tenant_id"], m["user_id"]), []).append(i)

    def _save(self):
        self._write_files(self._index, self._meta, durable=False)    # 0.10.0 (oss-0025): see _PENDING

    # 0.10.0: every write replaces both files the same way — delete_user's since oss-0024, store's since oss-0025 (it
    # wrote index.faiss and then meta.pkl in place: stopped while writing either, it left a file empty or short and the
    # directory would not open again — "EOFError: Ran out of input", "read error in index.faiss", measured on main;
    # stopped between the two, it left them out of step). The new index and entries are each written in full to a
    # ".pending" file beside the old one — and, for delete_user and repairs, flushed to disk; store() does not flush
    # (see _write_files) — then the index replaces index.faiss (os.replace: whole or not at all), and from that moment
    # this object holds the new index and entries; then the entries replace meta.pkl. Whatever stops it part way — the
    # process killed, or an error — leaves one of two states on disk:
    #   both ".pending" files, or only index.faiss.pending: neither old file was replaced; the ".pending" files are
    #   removed, meta.pkl.pending first;
    #   only meta.pkl.pending: the index was replaced; the entries are put in place.
    # _finish_interrupted_write settles it before anything reads (_load) or writes (_write_files, clear, delete_user)
    # the files, so an index of one generation is never used with the entries of another — whose positions would not
    # match, and a lookup would answer with another entry's response. On Windows a directory cannot be flushed: the
    # replaces are still whole or not at all, but a power cut may lose the last of them.
    _PENDING = ".pending"

    def _finish_interrupted_write(self, *, durable: bool = True):
        idx_new, meta_new = self._idx_path() + self._PENDING, self._meta_path() + self._PENDING
        if os.path.exists(idx_new):                  # the index was not replaced: nothing was written
            if os.path.exists(meta_new):
                os.remove(meta_new)                  # first: index.faiss.pending alone still means "not replaced"
                self._sync_dir()
            os.remove(idx_new)
            self._sync_dir()
        elif os.path.exists(meta_new):               # the index was replaced: put its entries in place
            os.replace(meta_new, self._meta_path())
            if durable:
                self._sync_dir()

    def _sync_dir(self):
        if os.name == "nt":                          # a directory cannot be opened to flush it on Windows
            return
        fd = os.open(self._path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _write_files(self, index, meta, *, durable: bool = True):
        """Write this index and these entries (see ``_PENDING``) and hold them from the moment the index is replaced.
        ``durable`` flushes each file and the directory to disk, so the write survives a power cut: delete_user and the
        repair of mismatched files. store() passes False — measured on Linux (ext4), flushing took a store at 1,000
        entries from 2.1 to 5.2 ms, and the files and their replacing cost nothing. Stopping the process is safe either
        way. A power cut after a store that was not flushed may lose the latest stores or, on a file system that does
        not flush a file replaced by rename (ext4 does), leave a file empty so the directory will not open — as before
        0.10.0, when nothing was flushed."""
        self._finish_interrupted_write(durable=durable)    # defensive: in one process, memory is always the newest
        idx_new, meta_new = self._idx_path() + self._PENDING, self._meta_path() + self._PENDING
        self._faiss.write_index(index, idx_new)
        if durable:
            with open(idx_new, "rb") as f:
                os.fsync(f.fileno())
        with open(meta_new, "wb") as f:
            pickle.dump(meta, f)
            if durable:
                f.flush()
                os.fsync(f.fileno())
        if durable:
            self._sync_dir()
        os.replace(idx_new, self._idx_path())
        if index is not self._index or meta is not self._meta:
            self._index, self._meta = index, meta    # the index on disk is the new one: so is this object's
            self._index_scopes()
        if durable:
            self._sync_dir()
        self._finish_interrupted_write(durable=durable)    # puts meta.pkl.pending in place

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

    def delete_user(self, user_id: str, *, tenant_id: Optional[str] = None) -> int:
        """0.10.0: remove every entry filed under (``tenant_id or "global"``, ``user_id or "global"``) — expired ones
        too: they are kept until deleted — and return how many (the ``Backend`` protocol's contract).

        A FAISS HNSW index cannot remove a vector, so the index is rebuilt from every other entry's vector, in order,
        and written with the entries as ``_PENDING`` describes. It holds the backend's lock throughout, so stores and
        lookups wait for it. Measured on faiss-cpu 1.15.1 (Linux, 384 dimensions, one machine): about 0.06 s at 1,000
        entries, 1 s at 10,000 and 6.4 s at 50,000; building the HNSW graph grows a little faster than the number of
        entries. A user with nothing stored returns 0 without rewriting anything. If it raises after the new index was
        put in place, the erasure has happened in this object and is completed on disk by the next write or the next
        FAISSBackend on the directory; a repeated call returns 0. Entries written before sulci 0.9.3 carry no scope and
        are not reached; ``clear()`` removes them. The bytes of the old files may stay on disk until the file system
        reuses them."""
        with self._lock:
            self._finish_interrupted_write()         # an earlier write's entries, if their replace failed
            doomed = set(self._scopes.get((tenant_id or "global", user_id or "global"), ()))
            if not doomed:
                return 0
            keep  = [i for i in range(len(self._meta)) if i not in doomed]
            index = self._faiss.IndexHNSWFlat(self._index.d, 32)
            index.hnsw.efSearch = 64
            if keep:
                index.add(self._index.reconstruct_batch(self._np.array(keep, dtype="int64")))
            meta = [self._meta[i] for i in keep]
            self._write_files(index, meta)
            return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._index = None
            self._meta  = []
            self._scopes = {}
            # 0.10.0 (oss-0024): a stopped write's leftovers go first, meta.pkl.pending first (see _PENDING)
            for p in [self._meta_path() + self._PENDING, self._idx_path() + self._PENDING,
                      self._idx_path(), self._meta_path()]:
                if os.path.exists(p):
                    os.remove(p)
