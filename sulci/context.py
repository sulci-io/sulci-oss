# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/context.py
================
Context-aware semantic caching for multi-turn LLM conversations.

The core problem
----------------
Without context, "How do I fix it?" is ambiguous — the cache might return a
Docker fix when the user is actually asking about billing, or miss a valid
hit entirely because the bare query doesn't match anything closely enough.

The solution
------------
Each conversation turn is embedded and held in a sliding window.  When
looking up a cache entry, sulci blends the current query embedding with a
*decayed* summary of recent turns:

    lookup_vec = α · embed(query) + (1-α) · Σ wᵢ · embed(turnᵢ)

    where  wᵢ  decays exponentially with turn age  (older → less weight)
    and    α   (query_weight) defaults to 0.70 so the current query always
           dominates — context just nudges the search direction.

Result
------
  - "How do I fix it?" after "My Docker container crashes" → Docker fixes ✓
  - "How do I fix it?" after "My payment failed" → billing solutions ✓
  - Isolated "How do I fix it?" → most common standalone match ✓
  - First turn in any session → pure query embedding, zero overhead ✓

Sessions
--------
A session groups turns for one conversation.  Identified by a session_id
string (UUID, user ID + thread ID, etc.).  Multiple independent
conversations run in parallel, each with their own isolated window.

Quick usage
-----------
    from sulci import Cache

    cache = Cache(backend="sqlite", context_window=6)

    cache.cached_call(
        "My Docker container crashes on startup",
        llm_fn,
        session_id="user-42",
    )
    result = cache.cached_call(
        "How do I fix it?",       # resolved in Docker context
        llm_fn,
        session_id="user-42",
    )
    print(result["context_depth"])  # 1 — one prior turn influenced this

    # Manual injection (e.g. restore a saved conversation)
    ctx = cache.get_context("user-42")
    ctx.add_turn("What Python version should I use?", role="user")
    ctx.add_turn("Python 3.11+ is recommended.", role="assistant")

    # Inspect
    print(cache.context_summary("user-42"))
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Turn ──────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """One message in a conversation (user query or assistant response)."""
    text:      str
    role:      str                    # "user" | "assistant" | "system"
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[list[float]] = field(default=None, repr=False)


# ── ContextWindow ─────────────────────────────────────────────────────────────

class ContextWindow:
    """
    Sliding window of conversation turns for one session.

    Args:
        max_turns:    Maximum turns to keep (oldest dropped first).
        query_weight: Weight of current query vs blended history (0.0–1.0).
                      0.70 = query dominates, history nudges direction.
                      Lower values make context influence stronger.
        decay:        Exponential decay per turn (0.0–1.0).
                      0.5 = each older turn contributes half as much.
                      0.8 = history stays influential for longer.
        roles:        Which roles contribute to context blending.
                      Default: both user and assistant turns.
    """

    def __init__(
        self,
        max_turns:    int   = 6,
        query_weight: float = 0.70,
        decay:        float = 0.50,
        roles:        tuple = ("user", "assistant"),
    ):
        self.max_turns    = max_turns
        self.query_weight = query_weight
        self.decay        = decay
        self.roles        = roles
        self._turns: list[Turn] = []
        self._session_id: str   = ""

    # ── mutations ─────────────────────────────────────────────────

    def add_turn(
        self,
        text:      str,
        role:      str = "user",
        embedding: Optional[list[float]] = None,
    ) -> Turn:
        """Append a turn. Drops oldest if window is full."""
        turn = Turn(text=text, role=role, embedding=embedding)
        self._turns.append(turn)
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)
        return turn

    def clear(self) -> None:
        """Reset conversation history."""
        self._turns.clear()

    # ── context blending ──────────────────────────────────────────

    def blend(
        self,
        query_vec: list[float],
        embedder=None,
    ) -> list[float]:
        """
        Return a context-blended embedding for the current query.

        Combines:
          - query_vec     (weight = query_weight, e.g. 0.70)
          - history vecs  (weight = 1 - query_weight, decayed by turn age)

        Most-recent turn has decay weight 1.0; each older turn is
        multiplied by the decay factor (default 0.5).

        If there is no history, returns query_vec unchanged — zero
        overhead for the first turn or stateless use.

        Args:
            query_vec: L2-normalised embedding of the current query.
            embedder:  Embedder instance used to lazily embed turns that
                       were added without pre-computed vectors.
        """
        history = [t for t in self._turns if t.role in self.roles]
        if not history:
            return query_vec

        # Lazily embed turns missing vectors
        if embedder is not None:
            for turn in history:
                if turn.embedding is None:
                    try:
                        turn.embedding = embedder.embed(turn.text)
                    except Exception:
                        pass  # skip un-embeddable turns gracefully

        embedded = [t for t in history if t.embedding is not None]
        if not embedded:
            return query_vec

        dim = len(query_vec)

        # Decayed history vector (most recent = weight 1.0)
        history_vec = [0.0] * dim
        total_w     = 0.0
        for i, turn in enumerate(reversed(embedded)):
            w = self.decay ** i          # 1.0, 0.5, 0.25, ...
            for j in range(dim):
                history_vec[j] += w * turn.embedding[j]
            total_w += w

        if total_w > 0:
            history_vec = [v / total_w for v in history_vec]

        # alpha * query + (1-alpha) * history
        alpha = self.query_weight
        out   = [alpha * q + (1.0 - alpha) * h for q, h in zip(query_vec, history_vec)]

        # Re-normalise so cosine similarity still works correctly
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]

    # ── introspection ─────────────────────────────────────────────

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    @property
    def depth(self) -> int:
        """Number of turns currently in the window."""
        return len(self._turns)

    def last_user_query(self) -> Optional[str]:
        """Return the text of the most recent user turn, or None."""
        for t in reversed(self._turns):
            if t.role == "user":
                return t.text
        return None

    def summary(self) -> dict:
        """Human-readable snapshot of the current window state."""
        now = time.time()
        return {
            "depth":        self.depth,
            "max_turns":    self.max_turns,
            "query_weight": self.query_weight,
            "decay":        self.decay,
            "turns": [
                {
                    "role":      t.role,
                    "text":      t.text[:80] + ("..." if len(t.text) > 80 else ""),
                    "has_embed": t.embedding is not None,
                    "age_s":     round(now - t.timestamp, 1),
                }
                for t in self._turns
            ],
        }

    def __repr__(self) -> str:
        return (
            f"ContextWindow(depth={self.depth}/{self.max_turns}, "
            f"query_weight={self.query_weight}, decay={self.decay})"
        )


# ── SessionStore ──────────────────────────────────────────────────────────────

class SessionStore:
    """
    Manager of ContextWindows for concurrent sessions.

    Automatically creates a new ContextWindow for unknown session IDs and
    evicts sessions that have been idle longer than ``ttl_seconds``.

    Args:
        max_turns:    Forwarded to each ContextWindow.
        query_weight: Forwarded to each ContextWindow.
        decay:        Forwarded to each ContextWindow.
        ttl_seconds:  Idle sessions older than this are evicted automatically.
                      None = no eviction (not recommended for long-running apps).
    """

    def __init__(
        self,
        max_turns:    int           = 6,
        query_weight: float         = 0.70,
        decay:        float         = 0.50,
        ttl_seconds:  Optional[int] = 3600,
    ):
        self._cfg = dict(
            max_turns    = max_turns,
            query_weight = query_weight,
            decay        = decay,
        )
        self.ttl_seconds   = ttl_seconds
        self._windows:     dict[str, ContextWindow] = {}
        self._last_active: dict[str, float]         = {}

    def get(self, session_id: str) -> ContextWindow:
        """Return the ContextWindow for this session, creating one if new."""
        self._evict_stale()
        if session_id not in self._windows:
            w = ContextWindow(**self._cfg)
            w._session_id             = session_id
            self._windows[session_id] = w
        self._last_active[session_id] = time.time()
        return self._windows[session_id]

    def delete(self, session_id: str) -> None:
        """Explicitly remove a session and its history."""
        self._windows.pop(session_id, None)
        self._last_active.pop(session_id, None)

    def clear_all(self) -> None:
        """Remove all sessions."""
        self._windows.clear()
        self._last_active.clear()

    def active_sessions(self) -> list[str]:
        """List currently active session IDs."""
        self._evict_stale()
        return list(self._windows.keys())

    def _evict_stale(self) -> None:
        if not self.ttl_seconds:
            return
        cutoff = time.time() - self.ttl_seconds
        stale  = [sid for sid, t in self._last_active.items() if t < cutoff]
        for sid in stale:
            self._windows.pop(sid, None)
            self._last_active.pop(sid, None)

    def summary(self) -> dict:
        """Overview of all active sessions and their turn counts."""
        self._evict_stale()
        return {
            "active_sessions": len(self._windows),
            "ttl_seconds":     self.ttl_seconds,
            "sessions": {
                sid: win.summary()
                for sid, win in self._windows.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"SessionStore(sessions={len(self._windows)}, "
            f"ttl={self.ttl_seconds}s)"
        )
