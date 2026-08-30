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
import time, uuid, warnings
from typing import Any, Optional


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
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name = self.COLLECTION,
            points = [PointStruct(
                id      = str(uuid.uuid4()),
                vector  = embedding,
                payload = {
                    "key": key, "query": query, "response": response,
                    "tenant_id": tenant_id or "global",
                    "user_id": user_id or "global",
                    "expires": expires or 0.0,
                    # v0.7.2 — store-time timestamp so consumers (e.g.
                    # sulci-platform top_queries) can report an honest
                    # "last seen" for entries that have never been served,
                    # instead of fabricating one at aggregation time.
                    "created": time.time(),
                    **(metadata or {}),
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

    def clear(self) -> None:
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
