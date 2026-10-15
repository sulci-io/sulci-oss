# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/backends/qdrant.py
Qdrant backend — best performance for production.

Install  : pip install "sulci[qdrant]"
Free tier: 1 GB cluster free forever at cloud.qdrant.io
Latency  : <5 ms local. `quantization` and `on_disk` change this and the
           direction is not the same for both — quantization trades a little
           recall for speed and memory, offload trades latency for cost.
           ❌ This line previously read "sub-ms with quantization". Neither
           lever was reachable from this constructor when that was written, so
           the figure described a configuration nobody using this class could
           produce. Both are reachable as of v0.9.1; the FIGURE is still
           unmeasured and is deliberately not restated here.
"""
from __future__ import annotations
import contextlib, threading, time, uuid, warnings
from typing import Any, Optional

from sulci._metadata import checked_metadata


class QdrantBackend:
    #: True if this backend enforces tenant_id partition isolation.
    #: When True, search() must not return entries with mismatched tenant_id.
    ENFORCES_TENANT_ISOLATION: bool = True

    COLLECTION = "sulci"

    #: Accepted shorthands for the `quantization` kwarg. A raw qdrant-client
    #: quantization model may be passed instead; it is forwarded untouched.
    #: `product` is deliberately absent — it needs a compression ratio this
    #: class has no defensible default for, and inventing one would be worse
    #: than making the caller pass the model.
    _QUANTIZATION_SHORTHANDS = ("scalar", "binary")

    def __init__(
        self,
        db_path:   str           = "./sulci_qdrant",
        url:       Optional[str] = None,
        api_key:   Optional[str] = None,
        dimension: int           = 384,
        *,
        on_disk:      Optional[bool] = None,
        quantization: Optional[Any]  = None,
    ):
        """
        on_disk
            Store vectors on disk instead of resident in RAM. `None` (the
            default) passes nothing and leaves Qdrant's own default in force,
            so an existing deployment is unaffected by upgrading.

        quantization
            `"scalar"`, `"binary"`, or a qdrant-client quantization model
            (`ScalarQuantization`, `BinaryQuantization`, `ProductQuantization`)
            forwarded untouched. `None` leaves quantization off.

        ⛔ **Both apply at COLLECTION CREATION ONLY.** This constructor is a
        no-op when the collection already exists, so passing either kwarg
        against a live collection does NOT reconfigure it. That is the
        fail-open this class is most likely to produce, so it does not stay
        silent: when the collection exists and the stored config differs from
        what was asked for, a `RuntimeWarning` names both values. Reconfigure
        with `update_collection` or rebuild the collection.

        ⚠️ Neither lever's effect on latency has been measured on this engine.
        They are exposed so the choice is available, not because a number is
        being claimed for either.
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client not found.\n"
                "Install with: pip install \"sulci[qdrant]\""
            )

        quantization_config = self._resolve_quantization(quantization)

        self._client = (
            QdrantClient(url=url, api_key=api_key) if url
            else QdrantClient(path=db_path)
        )
        # 0.10.0 (oss-0023): in local mode (db_path=) store, search_match, delete_user and clear hold self._lock, so
        # calls from several threads reach qdrant-client one at a time. AsyncCache runs each backend call on a thread of
        # its own. qdrant-client's local mode writes its points through one SQLite connection that two threads must not
        # use at once, and keeps them in in-memory arrays that a lookup reads while a write may be resizing them:
        # lookups need the lock too (an independent review measured IndexError with search_match alone unlocked).
        # Measured on main (oss-0022): 400 concurrent AsyncCache.aset calls raised sqlite3.InterfaceError ("bad
        # parameter or other API misuse") inside qdrant-client in 5 of 5 runs on Linux (qdrant-client 1.19.1), and
        # killed the Python process with a segmentation fault in 3 of 3 on a maintainer's Mac (1.17.1); with the lock,
        # every run on both found all its entries. Both SQLite builds were THREADSAFE=1, so qdrant-client let any thread
        # use the connection; where it is built otherwise (qdrant-client says THREADSAFE=2 is common on macOS),
        # qdrant-client refuses calls from any thread but the one that opened it, lock or not. A Qdrant server (url=) is
        # reached over HTTP and serves concurrent requests itself: no lock. An RLock, so a test can ask whether the
        # calling thread holds it.
        self._lock = threading.RLock() if not url else contextlib.nullcontext()

        # Only pass what the caller actually asked for. Passing on_disk=None
        # and quantization_config=None is equivalent to omitting them today,
        # but building the kwargs explicitly keeps that an intentional
        # property of this call rather than a coincidence of qdrant-client's
        # current defaults.
        vector_kwargs: dict = {"size": dimension, "distance": Distance.COSINE}
        if on_disk is not None:
            vector_kwargs["on_disk"] = on_disk
        if quantization_config is not None:
            vector_kwargs["quantization_config"] = quantization_config

        existing = [c.name for c in self._client.get_collections().collections]
        if self.COLLECTION not in existing:
            self._client.create_collection(
                collection_name = self.COLLECTION,
                vectors_config  = VectorParams(**vector_kwargs),
            )
        elif on_disk is not None or quantization_config is not None:
            self._warn_if_config_differs(on_disk, quantization_config)

    @classmethod
    def _resolve_quantization(cls, quantization: Optional[Any]) -> Optional[Any]:
        """Map a shorthand to a qdrant-client model, or forward a model.

        Raises ValueError on an unrecognised string. A typo'd shorthand must
        not silently fall through to "no quantization" — that is a green run
        with the lever off, which is indistinguishable from a green run with
        the lever on unless something refuses.
        """
        if quantization is None:
            return None
        if not isinstance(quantization, str):
            return quantization          # assume a qdrant-client model

        key = quantization.strip().lower()
        from qdrant_client.models import (
            BinaryQuantization, BinaryQuantizationConfig,
            ScalarQuantization, ScalarQuantizationConfig, ScalarType,
        )
        if key == "scalar":
            # always_ram is left unset. Qdrant's default is what the cost
            # modelling in sulci-platform's COGS.md priced; setting it here
            # would change the memory profile the saving was measured against.
            return ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8)
            )
        if key == "binary":
            return BinaryQuantization(binary=BinaryQuantizationConfig())
        raise ValueError(
            f"quantization={quantization!r} is not recognised. Pass one of "
            f"{cls._QUANTIZATION_SHORTHANDS}, or a qdrant-client quantization "
            f"model (e.g. ProductQuantization) to use anything else."
        )

    def _warn_if_config_differs(
        self, on_disk: Optional[bool], quantization_config: Optional[Any]
    ) -> None:
        """Compare the live collection against what was asked for, and say so.

        The collection already exists, so nothing here changes it. The only
        useful thing this can do is refuse to be silent about the gap.
        """
        try:
            params = self._client.get_collection(self.COLLECTION).config.params.vectors
        except Exception:
            # Named vectors, an older server, or a transient error. Do not
            # guess; say that the check could not run rather than reporting a
            # match it never made.
            warnings.warn(
                f"QdrantBackend: collection {self.COLLECTION!r} already exists "
                f"and its vector config could not be read, so the requested "
                f"on_disk/quantization settings were NOT verified and have "
                f"NOT been applied.",
                RuntimeWarning, stacklevel=3,
            )
            return

        drift = []
        if on_disk is not None and getattr(params, "on_disk", None) != on_disk:
            drift.append(f"on_disk: live={getattr(params, 'on_disk', None)!r} requested={on_disk!r}")
        if quantization_config is not None:
            live_q = getattr(params, "quantization_config", None)
            if live_q != quantization_config:
                drift.append(f"quantization: live={live_q!r} requested={quantization_config!r}")

        if drift:
            warnings.warn(
                f"QdrantBackend: collection {self.COLLECTION!r} already exists and was "
                f"NOT reconfigured — create_collection is a no-op here. "
                + "; ".join(drift)
                + ". Apply with update_collection, or drop and rebuild the collection.",
                RuntimeWarning, stacklevel=3,
            )

    def store(
        self,
        key: str, query: str, response: str, embedding: list[float],
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, expires: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        with self._lock:
            from qdrant_client.models import PointStruct
            self._client.upsert(
                collection_name = self.COLLECTION,
                points = [PointStruct(
                    id      = str(uuid.uuid4()),
                    vector  = embedding,
                    payload = {
                        # 0.9.3: the caller's metadata is refused if it uses a reserved name, and goes
                        # FIRST, so the entry's own fields below always win (it used to come last).
                        **checked_metadata(metadata),
                        "key": key, "query": query, "response": response,
                        "tenant_id": tenant_id or "global",
                        "user_id": user_id or "global",
                        "expires": expires or 0.0,
                        # v0.7.2 — store-time timestamp so consumers (e.g.
                        # sulci-platform top_queries) can report an honest
                        # "last seen" for entries that have never been served,
                        # instead of fabricating one at aggregation time.
                        "created": time.time(),
                    },
                )],
            )

    def search(
        self,
        embedding: list[float], threshold: float,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        resp, score, _matched = self.search_match(
            embedding, threshold,
            tenant_id=tenant_id, user_id=user_id, now=now,
        )
        return resp, score

    def search_match(
        self,
        embedding: list[float], threshold: float,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None, now: Optional[float] = None,
    ) -> tuple[Optional[str], float, Optional[str]]:
        """
        Like search(), but also returns the STORED query text of the
        matched entry as the third element (None on miss).

        v0.7.2 — optional backend extension, feature-detected by
        Cache.get via hasattr(). The matched query identifies which
        cached entry was served, enabling per-entry hit counting by
        downstream consumers (CacheEvent.matched_query_hash). search()
        keeps its 2-tuple contract by delegating here, so the Backend
        protocol and the other backends are untouched.
        """
        with self._lock:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            now = now or time.time()

            # Build payload filter conditions. Tenant isolation is a hard
            # boundary — entries from other tenants must never be returned,
            # even if their similarity exceeds threshold. The store path
            # writes tenant_id="global" when None is passed; the read path
            # mirrors that by filtering to "global" for None — otherwise
            # an unscoped read would silently see entries from named tenants
            # and break the operational migration to multi-tenancy.
            must_conditions = [
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id if tenant_id is not None else "global"),
                ),
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id if user_id is not None else "global"),
                ),
            ]
            filter_ = Filter(must=must_conditions)

            results = self._client.query_points(
                collection_name = self.COLLECTION,
                query           = embedding,
                query_filter    = filter_,
                limit           = 5,
                with_payload    = True,
            )
            for r in results.points:
                p = r.payload or {}
                if p.get("expires") and now > p["expires"]:
                    continue
                if r.score >= threshold:
                    return p.get("response"), r.score, p.get("query")
            return None, 0.0, None

    def delete_user(self, user_id: str, *, tenant_id: Optional[str] = None) -> int:
        """0.10.0: remove every entry filed under (``tenant_id or "global"``, ``user_id or "global"``) — expired ones
        too: Qdrant keeps them until deleted — and return how many (the ``Backend`` protocol's contract).

        Two calls: an exact ``count`` with the scope's filter (the scope ``store`` files under), then one ``delete``
        by that filter with ``wait=True``, which returns once the deletion is applied. Qdrant's answer to a delete
        carries no count (measured on qdrant-client 1.10.0 and 1.19.1, local mode: ``UpdateResult(status=completed)``
        whether or not anything matched), so the count is the one taken first. An entry stored for the user between
        the two calls is erased without being counted, and two erasures of the same user at once may both count the
        same entries. An answer other than ``completed`` raises: the deletion was not confirmed; repeat the call.
        Against a server, a client timeout while a large deletion runs raises too, and the server may finish it; a
        repeated call then counts what is left. Entries written before sulci 0.4.0 have no ``tenant_id`` and are
        neither found by ``search`` nor erased here; ``clear()`` removes them."""
        with self._lock:
            from qdrant_client.models import (FieldCondition, Filter, FilterSelector, MatchValue,
                                              UpdateStatus)
            scope = Filter(must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id or "global")),
                FieldCondition(key="user_id", match=MatchValue(value=user_id or "global")),
            ])
            held = self._client.count(collection_name=self.COLLECTION, count_filter=scope, exact=True).count
            result = self._client.delete(collection_name=self.COLLECTION,
                                         points_selector=FilterSelector(filter=scope), wait=True)
            if getattr(result, "status", None) != UpdateStatus.COMPLETED:
                raise RuntimeError(f"QdrantBackend.delete_user: Qdrant answered {result!r} rather than a completed "
                                   f"deletion; the erasure is not confirmed — repeat the call")
            return held

    def clear(self) -> None:
        with self._lock:
            # Delete all points but keep the collection (and its HNSW index)
            # intact. qdrant-client 1.x raises ValueError on subsequent
            # operations against a missing collection, so a recreate-on-clear
            # would also work but costs a full index rebuild on first store.
            from qdrant_client.models import Filter
            try:
                self._client.delete(
                    collection_name = self.COLLECTION,
                    points_selector = Filter(must=[]),
                )
            except Exception:
                # Empty collection or other transient error — clear is
                # idempotent so we swallow.
                pass
