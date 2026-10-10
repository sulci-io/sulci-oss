# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/_scope_ids.py
===================
``"global"`` and ``""`` are not ids a caller may pass (0.9.3).

Every backend files an entry stored without a tenant or user under the sentinel
``"global"`` — the SHARED scope. A caller that passed ``user_id="global"`` or
``tenant_id="global"`` therefore read and wrote the shared scope as if it were
its own; ``""`` was stored there too (``"" or "global"``), and on Qdrant a lookup
with ``""`` could not even read back what it had stored. Measured through
``Cache`` on Qdrant, Chroma, FAISS, SQLite and Redis.

``Cache.get`` and ``Cache.set`` now refuse them with ``ValueError`` before
anything is read, written or sent: ``tenant_id`` always, and ``user_id`` when it
takes effect (``personalized=True``; otherwise ``Cache`` ignores it). Refused:
the string ``"global"``, and ANY value that is falsy but not ``None`` — ``""``,
``0``, ``False``, ``b""`` — because every backend files ``id or "global"``, so
each of them landed in the shared scope too. Omit the argument (``None``) to mean
"no tenant" or "no user". The managed gateway refuses ``"global"`` and ``""``
as user ids with 422.
"""

from __future__ import annotations

from typing import Any

#: The string values that address the shared scope. (Any other falsy value but None does too: see _reserved.)
RESERVED_SCOPE_IDS = frozenset({"global", ""})


def _reserved(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value in RESERVED_SCOPE_IDS
    try:
        return not value                      # 0, False, b"", ... are filed as "global" by every backend
    except Exception:
        return False


def _not_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
        return False
    except UnicodeEncodeError:              # a lone surrogate: no backend can store or filter on it
        return True


def refuse_reserved_scope(tenant_id: Any, user_id: Any, *, personalized: bool) -> None:
    """Raise ``ValueError`` if ``tenant_id``, or ``user_id`` when ``personalized``, names the shared scope."""
    if _reserved(tenant_id):
        raise ValueError(
            f"tenant_id={tenant_id!r} is reserved: it names the shared scope that entries without a tenant are "
            "filed under. Omit tenant_id (None) for no tenant, or pass a tenant's own id."
        )
    if _not_text(tenant_id):
        raise ValueError("tenant_id is not valid Unicode text (it contains a lone surrogate)")
    if personalized and _not_text(user_id):
        raise ValueError("user_id is not valid Unicode text (it contains a lone surrogate)")
    if personalized and _reserved(user_id):
        raise ValueError(
            f"user_id={user_id!r} is reserved: it names the shared scope that entries without a user are "
            "filed under. Omit user_id (None) for no user, or pass a user's own id."
        )
