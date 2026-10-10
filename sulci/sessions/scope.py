# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sessions/scope.py — the tenant and user a session belongs to (0.9.3)

Conversation history used to be keyed by the caller's ``session_id`` alone, so one
store serving several tenants (or users, with ``personalized=True``) blended one
tenant's turns into another's lookups whenever they used the same id, and nothing
could find a tenant's or a user's sessions to erase them.

``Cache`` now hands every session store a SCOPED id:

    t{tenant}:u{user}:{session_id}

- ``{tenant}`` and ``{user}`` are the first 32 hex digits of the SHA-256 of the id's
  UTF-8 text, or ``-`` for ``None`` (the shared scope; ``user`` is ``None`` whenever
  the Cache is not personalized). Fixed width and hex, so an id holding ``:`` or glob
  characters cannot shift the key, and a hex digest is never ``-``.
- the caller's ``session_id`` last and verbatim, never parsed.

``tenant_scope(tenant_id)`` and ``user_scope(tenant_id, user_id)`` are the prefixes a
store's ``clear_scope`` takes: every session of that tenant, or of that user within it.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional, Tuple

NO_ID = "-"
_SCOPED = re.compile(r"\A(t(?:-|[0-9a-f]{32}):)(u(?:-|[0-9a-f]{32}):)")


def _part(value: Optional[str]) -> str:
    if value is None:
        return NO_ID
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:32]


def tenant_scope(tenant_id: Optional[str]) -> str:
    """The prefix of every scoped session id of this tenant (``None``: the shared scope)."""
    return f"t{_part(tenant_id)}:"


def user_scope(tenant_id: Optional[str], user_id: Optional[str]) -> str:
    """The prefix of every scoped session id of this user within this tenant (``None``: no user)."""
    return f"{tenant_scope(tenant_id)}u{_part(user_id)}:"


def scoped_session_id(tenant_id: Optional[str], user_id: Optional[str], session_id: str) -> str:
    """The id a session store is given for this tenant's (and user's) ``session_id``."""
    return f"{user_scope(tenant_id, user_id)}{session_id}"


def scopes_of(session_id: str) -> Tuple[str, ...]:
    """The tenant and user scopes a scoped session id belongs to; ``()`` for an id that is not scoped
    (a store used directly, without a Cache)."""
    m = _SCOPED.match(session_id)
    return (m.group(1), m.group(1) + m.group(2)) if m else ()
