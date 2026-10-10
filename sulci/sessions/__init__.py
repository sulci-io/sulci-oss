# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci.sessions — v0.5.0 session store protocols and implementations.

Public API:
    SessionStore            (protocol)
    InMemorySessionStore    (default, process-local)
    RedisSessionStore       (for horizontal scaling)
    scoped_session_id, tenant_scope, user_scope   (0.9.3: the tenant and user a session belongs to)
"""
from sulci.sessions.protocol import SessionStore
from sulci.sessions.memory import InMemorySessionStore
from sulci.sessions.redis import RedisSessionStore
from sulci.sessions.scope import scoped_session_id, tenant_scope, user_scope

__all__ = ["SessionStore", "InMemorySessionStore", "RedisSessionStore",
           "scoped_session_id", "tenant_scope", "user_scope"]
