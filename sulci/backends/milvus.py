# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/backends/milvus.py
Milvus Lite backend — embedded vector DB, no server required.

Install  : pip install "sulci[milvus]"
Free tier: Milvus Lite (embedded) or Zilliz Cloud free tier
Latency  : 5–20 ms embedded, <5 ms Zilliz Cloud
"""
from __future__ import annotations
import re
import time
from typing import Optional

from sulci.sessions.scope import user_scope

# 0.10.0: an entry's scope is filed as ONE key — ``sulci.sessions.scope.user_scope(tenant, user)``, built from
# SHA-256 digests — and a lookup filters on that key alone. No caller-supplied text reaches a filter expression.
#
# Through 0.9.3 the filter quoted the tenant and user ids themselves. Measured on milvus-lite 2.5.1 (installed
# beside pymilvus 2.6.12 on a maintainer's machine, and with pymilvus 3.0.2 here): an id holding a newline, carriage
# return or tab missed its own entries — every escape form of those characters failed to match, so no quoting could
# fix it; milvus-lite 3.2.2 matched them. A key of hex digits, "t", "u", ":" and "-" needs no quoting on any version.
# The raw ids are no longer stored either: nothing reads them, and the key is all a lookup needs.
_SCOPE_KEY = re.compile(r"\At(?:-|[0-9a-f]{32}):u(?:-|[0-9a-f]{32}):\Z")


def _scope_key(tenant_id: Optional[str], user_id: Optional[str]) -> str:
    """The key an entry of this tenant and user is filed under. ``None`` and ``""`` both mean the shared scope.
    Unlike the backends that file ``id or "global"``, the string ``"global"`` is an id like any other here — it does
    NOT reach the shared scope. ``Cache`` refuses it before any backend is called (``sulci._scope_ids``)."""
    key = user_scope(tenant_id or None, user_id or None)
    if not _SCOPE_KEY.match(key):               # the filter relies on it: no quote, backslash or space
        raise RuntimeError(f"not a scope key: {key!r}")
    return key


class MilvusBackend:
    #: True if this backend enforces tenant_id partition isolation.
    #: When True, search() must not return entries with mismatched tenant_id.
    ENFORCES_TENANT_ISOLATION: bool = True     # 0.9.3: search() filters on tenant AND user

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

    def _open(self) -> bool:
        """0.10.0: use the collection if it exists — check its schema and LOAD it — and say whether it does.

        Through 0.9.3 an existing collection was never opened: every lookup missed until this process stored
        something (``search`` returned at once while ``_ready`` was False). It was never loaded either: a Milvus
        server — and milvus-lite 3.x, which the extra excludes — refuse to search a collection that is not loaded
        ("Collection 'sulci' is in state 'released'", measured on milvus-lite 3.2.2 for a collection another
        process wrote); milvus-lite 2.5.1 searches it anyway. Measured 2026-10-14."""
        if self._ready:
            return True
        if not self._client.has_collection(self.COLLECTION):
            return False
        # 0.10.0: every entry carries a "scope" field. A collection sulci created has dynamic fields; one created
        # elsewhere with a fixed schema and no "scope" field would fail every insert and every lookup with a
        # Milvus error — say what to do instead.
        info   = self._client.describe_collection(self.COLLECTION)
        fields = {f.get("name") for f in info.get("fields", [])}
        if "scope" not in fields and not info.get("enable_dynamic_field", False):
            raise RuntimeError(
                f"Milvus collection {self.COLLECTION!r} has a fixed schema without a 'scope' field. sulci "
                "0.10.0 files every entry under one: drop the collection (MilvusBackend.clear()) and let sulci "
                "create it, or add a VARCHAR 'scope' field (max_length >= 68) or enable dynamic fields."
            )
        # Load only if it is not loaded: loading needs the Load privilege, which a read-only role on a Milvus
        # server may lack, and a collection this or another sulci process already loaded needs nothing.
        state = self._client.get_load_state(self.COLLECTION).get("state")
        if getattr(state, "name", str(state)) != "Loaded":
            self._client.load_collection(self.COLLECTION)
        self._ready = True
        return True

    def _ensure(self, dim: int):
        if self._open():
            return
        self._client.create_collection(           # created loaded
            collection_name = self.COLLECTION,
            dimension       = dim,
            metric_type     = "COSINE",
            auto_id         = True,
        )
        self._ready = True

    def store(
        self,
        key: str, query: str, response: str, embedding: list[float],
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._ensure(len(embedding))
        self._client.insert(self.COLLECTION, [{
            "vector":   embedding,
            "query":    query,
            "response": response,
            "scope":    _scope_key(tenant_id, user_id),     # 0.10.0: what search() filters on; no raw ids
            "expires":  expires or 0.0,
        }])

    def search(
        self,
        embedding: list[float], threshold: float,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        # 0.10.0: an existing collection, not only one this process wrote. Until a collection exists, each lookup
        # asks the server whether it does (one round trip) and misses.
        if not self._open():
            return None, 0.0
        now     = now or time.time()
        # 0.9.3: a lookup reads its own scope only — tenant AND user, None meaning the shared scope.
        # 0.10.0: by the scope key alone. An entry written before 0.10.0 has no "scope" field, so it never
        # matches: it is a miss, and stays in the collection until clear().
        filter_ = f'scope == "{_scope_key(tenant_id, user_id)}"'
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
