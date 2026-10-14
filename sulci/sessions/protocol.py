# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sessions/protocol.py — SessionStore protocol (v0.5.0)

STABLE API — modifications require superseding ADR per ADR 0005.

📌 0.9.3: ``clear_scope`` and ``clear_all`` added (OSS-3 — session history was keyed
by ``session_id`` alone and ``Cache.clear()`` could not reach an injected store's).
Recorded as ADR 0005 Amendment 1 (sulci-platform, 2026-10-11).
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable, Optional, List, Sequence


@runtime_checkable
class SessionStore(Protocol):
    """
    Per-session conversation history for context-aware caching.

    Implementations shipped in v0.5.0:
      - InMemorySessionStore — process-local dict (default, zero config)
      - RedisSessionStore    — Redis Lists for horizontal scaling

    Custom implementations: any class matching this surface.
    """

    def get(self, session_id: str) -> List[Sequence[float]]:
        """
        Return the conversation history for this session as a list of
        embedding vectors. Most recent last. Empty list if session is new.
        """
        ...

    def append(
        self,
        session_id: str,
        vector: Sequence[float],
        max_turns: int = 8,
    ) -> None:
        """
        Append an embedding vector to this session's history.
        If len(history) > max_turns, the oldest entries are dropped.
        """
        ...

    def clear(self, session_id: str) -> None:
        """Remove all history for this session. Idempotent."""
        ...

    def clear_scope(self, scope: str) -> int:
        """
        0.9.3. Remove every session whose id starts with ``scope`` — a tenant's
        (``sulci.sessions.scope.tenant_scope``) or a user's (``user_scope``) — and
        return how many were removed. ``Cache`` hands a store scoped ids
        (``scoped_session_id``), so this is how a tenant's or a user's
        conversation history is erased. Must not scan another tenant's keys
        to do it when the store is shared.
        """
        ...

    def clear_all(self) -> int:
        """
        0.9.3. Remove every session this store holds and return how many.
        ``Cache.clear()`` calls it; ``Cache`` refuses at construction a store
        that does not have it, rather than leave history ``clear()`` cannot reach.
        """
        ...

    def summary(self, session_id: Optional[str] = None) -> dict:
        """
        Return stats about the store.
        If session_id is given, stats for that session only.
        If None, aggregate stats across all sessions.

        Return shape:
          {
            "sessions": int,        # number of distinct session_ids (or 1 if scoped)
            "total_turns": int,     # total history entries across sessions
            "avg_turns": float,     # average per session
          }
        """
        ...
