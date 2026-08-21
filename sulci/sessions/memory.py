# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sessions/memory.py — InMemorySessionStore (v0.5.0)

Extracted from sulci/context.py to match the SessionStore protocol.
Behavior is identical to the v0.3.x context.SessionStore class.
sulci.context.SessionStore still importable via backward-compat shim.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Optional, List, Sequence

from sulci.sessions.protocol import SessionStore


class InMemorySessionStore(SessionStore):
    """
    Process-local session store using a dict.

    Appropriate for:
      - Single-process deployments (notebooks, Streamlit, local dev)
      - Self-hosted single-gateway configurations

    NOT appropriate for:
      - Multi-replica gateway (use RedisSessionStore instead)
      - Long-running processes that may leak memory (use max_total_sessions)
    """

    def __init__(
        self,
        max_total_sessions: Optional[int] = None,
        tenant_id: Optional[str] = None,
    ):
        """
        Args:
            max_total_sessions: If set, evicts the oldest session when
                                exceeded (LRU-ish). Default None = unbounded.
            tenant_id:          Bind this store to a tenant. When set, all
                                session_ids are namespaced internally so two
                                tenants using the same session_id never collide.
                                Default None = single-tenant (no scoping).
        """
        self._data: dict[str, List[Sequence[float]]] = defaultdict(list)
        self._max_total_sessions = max_total_sessions
        self._tenant_id = tenant_id

    def _scoped(self, session_id: str) -> str:
        return f"{self._tenant_id}::{session_id}" if self._tenant_id else session_id

    def get(self, session_id: str) -> List[Sequence[float]]:
        return list(self._data.get(self._scoped(session_id), []))

    def append(
        self,
        session_id: str,
        vector: Sequence[float],
        max_turns: int = 8,
    ) -> None:
        history = self._data[self._scoped(session_id)]
        history.append(list(vector))
        if len(history) > max_turns:
            # Trim from the front — keep most recent max_turns entries
            del history[:-max_turns]

        if (self._max_total_sessions
                and len(self._data) > self._max_total_sessions):
            # Evict the first-inserted session (Python 3.7+ dicts preserve order)
            oldest_id = next(iter(self._data))
            del self._data[oldest_id]

    def clear(self, session_id: str) -> None:
        self._data.pop(self._scoped(session_id), None)

    def summary(self, session_id: Optional[str] = None) -> dict:
        if session_id is not None:
            scoped = self._scoped(session_id)
            turns = len(self._data.get(scoped, []))
            return {
                "sessions": 1 if scoped in self._data else 0,
                "total_turns": turns,
                "avg_turns": float(turns),
            }

        total_sessions = len(self._data)
        total_turns = sum(len(h) for h in self._data.values())
        avg_turns = total_turns / total_sessions if total_sessions else 0.0
        return {
            "sessions": total_sessions,
            "total_turns": total_turns,
            "avg_turns": avg_turns,
        }
