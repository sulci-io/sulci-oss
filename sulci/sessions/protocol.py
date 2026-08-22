# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sessions/protocol.py — SessionStore protocol (v0.5.0)

STABLE API — modifications require superseding ADR per ADR 0005.
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
