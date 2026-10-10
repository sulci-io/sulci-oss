# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sessions/memory.py — InMemorySessionStore (v0.5.0)

Extracted from sulci/context.py to match the SessionStore protocol.

0.9.3: the store no longer takes a ``tenant_id``. ``Cache`` scopes every session id by
tenant and user (``sulci.sessions.scope``), and ``clear_scope`` / ``clear_all`` remove a
tenant's, a user's or every session. Every operation holds one lock: ``AsyncCache`` and
threaded servers call a store from several threads, and ``clear_scope`` walks the dict.
"""
from __future__ import annotations
import threading
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

    def __init__(self, max_total_sessions: Optional[int] = None):
        """
        Args:
            max_total_sessions: If set, evicts the oldest session when
                                exceeded (LRU-ish). Default None = unbounded.
        """
        self._data: dict[str, List[Sequence[float]]] = defaultdict(list)
        self._max_total_sessions = max_total_sessions
        self._lock = threading.Lock()

    def get(self, session_id: str) -> List[Sequence[float]]:
        with self._lock:
            return list(self._data.get(session_id, []))

    def append(
        self,
        session_id: str,
        vector: Sequence[float],
        max_turns: int = 8,
    ) -> None:
        with self._lock:
            history = self._data[session_id]
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
        with self._lock:
            self._data.pop(session_id, None)

    def clear_scope(self, scope: str) -> int:
        with self._lock:
            doomed = [sid for sid in self._data if sid.startswith(scope)]
            for sid in doomed:
                del self._data[sid]
            return len(doomed)

    def clear_all(self) -> int:
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n

    def summary(self, session_id: Optional[str] = None) -> dict:
        with self._lock:
            if session_id is not None:
                turns = len(self._data.get(session_id, []))
                return {
                    "sessions": 1 if session_id in self._data else 0,
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
