# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/backends/_scope.py
========================
The storage id of a cache entry, scoped to its tenant and user (0.9.3).

``Cache.set`` derives an entry's ``key`` from the query text alone. The Chroma
backend upserted by that key and the Redis backend used it as the hash name, so
when a second tenant or user stored the same question, its entry REPLACED the
first one: one tenant could erase another tenant's entries, and one user
another user's, just by asking the same thing. (SQLite had the same flaw and
was fixed in 0.9.2 with ``UNIQUE (key, tenant_id, user_id)``; Qdrant, FAISS and
Milvus never upsert by key.)

``scoped_id`` combines the tenant, the user and the key with an unambiguous
encoding (an ASCII-escaped JSON array, so ``("a:b", "c")`` and ``("a", "b:c")``
differ, and any string encodes, lone surrogates included), and maps a falsy
tenant or user to ``"global"``, as the backends do for the scope fields they
store (every backend stores both since 0.9.3). The same question
stored again in the SAME scope still replaces its entry.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional


def scoped_id(key: str, tenant_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """A 32-hex-digit id that differs whenever the tenant, the user or the key differs."""
    scope = [tenant_id or "global", user_id or "global", key]
    return hashlib.sha256(json.dumps(scope, ensure_ascii=True).encode("ascii")).hexdigest()[:32]
