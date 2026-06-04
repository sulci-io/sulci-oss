# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
sulci/sinks/protocol.py — EventSink protocol + CacheEvent (v0.5.0)

STABLE API — modifications require superseding ADR per ADR 0005.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Optional, Dict, Any


def query_hash(text: str) -> str:
    """
    Stable identity hash for a stored query's text.

    CONTRACT — shared with consumers outside this repo: sulci-platform's
    top-queries pipeline (workers/top_queries + shared/hashing.py)
    computes the same value independently when aggregating Qdrant
    payloads, and joins it against CacheEvent.matched_query_hash to
    attribute cache serves to stored entries. Changing the algorithm or
    truncation length breaks that join silently; both repos pin the
    scheme with a literal-value test.

    sha256 of the UTF-8 text, first 32 hex chars.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:32]


@dataclass
class CacheEvent:
    """
    Emitted by Cache on every hit/miss/set/clear.

    Privacy discipline: sinks shipped with sulci MUST NOT emit query text,
    response text, or embedding vectors externally. Only the metadata
    envelope shown below leaves the process. TelemetrySink enforces this
    via an explicit field allowlist.
    """
    event_type: str                             # 'hit' | 'miss' | 'set' | 'clear'
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    backend_id: Optional[str] = None            # e.g. 'qdrant', 'chroma', 'sulci'
    embedding_model: Optional[str] = None       # e.g. 'minilm', 'openai'
    similarity: Optional[float] = None          # for 'hit' events
    latency_ms: Optional[int] = None            # for 'hit' and 'miss'
    context_depth: int = 0                      # number of session turns consulted
    timestamp: Optional[float] = None           # unix timestamp
    # ── v0.5.6 addition (sulci-oss issue #36) ──
    # Customer plan tier ('free' | 'pro' | 'business' | 'enterprise' |
    # 'oss_connect'). Populated by callers who know the tenant's plan
    # at emit time (sulci-platform's gateway passes it from the auth
    # lookup); defaults to None for users of the OSS library who don't
    # have plan context. Added per ADR 0005's "additive kwarg with
    # backward-compatible default" rule — pre-0.5.6 callers see no
    # behavior change.
    #
    # Why the field exists: the sulci-platform billing pipeline reads
    # cache events from a Redis stream and routes them by tenant +
    # plan. Without this field, plan attribution required a separate
    # Postgres lookup per event, which created a join-at-consume-time
    # burden that two realworld E2E tests caught (test_09 / test_j09).
    # Carrying plan on the event closes that gap.
    plan: Optional[str] = None
    # ── v0.7.2 addition (true hit-count semantics) ──
    # Identity hash (see query_hash() above) of the STORED query whose
    # entry was served, populated only on 'hit' events and only when the
    # backend exposes search_match() (QdrantBackend does). None on
    # misses, on backends without search_match, and for pre-0.7.2
    # callers — additive per ADR 0005, same pattern as `plan` above.
    #
    # Why the field exists: sulci-platform's Top Queries pipeline needs
    # to count how many times each cached entry was actually SERVED,
    # not how many times it was stored — otherwise the dashboard's
    # "Hits" column contradicts the hit-rate stat computed from these
    # very events. Carrying the matched entry's hash on the existing
    # hit event closes that gap with zero extra hot-path I/O.
    #
    # Privacy: this is a hash, never text. It is deliberately NOT in the
    # sink allowlist, so TelemetrySink and RedisStreamSink scrub it —
    # it is consumable only by in-process sinks injected by the caller.
    matched_query_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)   # extension point


@runtime_checkable
class EventSink(Protocol):
    """
    Receives CacheEvent on every cache operation.

    Implementations shipped in v0.5.0:
      - NullSink         — no-op, default
      - TelemetrySink    — HTTPS POST to endpoint with field allowlist
      - RedisStreamSink  — writes to Redis Stream for billing/observability

    Custom implementations: any class matching this surface.
    """

    def emit(self, event: CacheEvent) -> None:
        """
        Handle a cache event. Called on every hit/miss/set/clear.

        MUST NOT raise on delivery failure — degrade gracefully
        (log and continue). A failing sink must never break the
        caller's cache operation.
        """
        ...

    def flush(self) -> None:
        """
        Force-flush any buffered events.

        Called at Cache.__del__ and on explicit user flush.
        May be a no-op for sinks that don't buffer.
        """
        ...
