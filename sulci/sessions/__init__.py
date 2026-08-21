# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci.sessions — v0.5.0 session store protocols and implementations.

Public API:
    SessionStore            (protocol)
    InMemorySessionStore    (default, process-local)
    RedisSessionStore       (for horizontal scaling)
"""
from sulci.sessions.protocol import SessionStore
from sulci.sessions.memory import InMemorySessionStore
from sulci.sessions.redis import RedisSessionStore

__all__ = ["SessionStore", "InMemorySessionStore", "RedisSessionStore"]
