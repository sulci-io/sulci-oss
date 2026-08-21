# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci.integrations._scope
──────────────────────────
Internal helper: warn when a caller asks for ``tenant_id`` scoping on a
backend that does not enforce it.

WHY THIS EXISTS — read before deleting it
─────────────────────────────────────────
``Cache.get`` / ``Cache.set`` accept ``tenant_id`` on every backend. Only
**Qdrant** filters on it. Measured 2026-08-11::

    chroma  ENFORCES_TENANT_ISOLATION = False
    cloud   ENFORCES_TENANT_ISOLATION = False
    faiss   ENFORCES_TENANT_ISOLATION = False
    milvus  ENFORCES_TENANT_ISOLATION = False
    qdrant  ENFORCES_TENANT_ISOLATION = True     <-- the only True

⚠️  ``cloud`` (``backend="sulci"``) is a FALSE NEGATIVE in that table. See
    ``backends/cloud.py:85-89``: the Sulci Cloud gateway DOES enforce
    isolation server-side, keyed off the api_key, and the flag is declared
    False only because the OSS conformance suite cannot reach the gateway to
    verify it locally. ``API-SURFACE.md`` is right to say "only qdrant and
    sulci enforce tenant isolation." This helper therefore exempts the cloud
    backend BY NAME — reading the flag alone would warn wrongly on the managed
    backend, which is the one customers pay for isolation on.
    redis   ENFORCES_TENANT_ISOLATION = False
    sqlite  ENFORCES_TENANT_ISOLATION = False

The argument is accepted and ignored. That is survivable when a human
passes it as a hint. It is **not** survivable in the three surfaces added in
v0.9.0, because all three sell scoping as a safety property:

* the MCP server tells agents to put a commit SHA in ``tenant_id`` so a
  stale review cannot be served for new code;
* the LiteLLM adapter defaults to ``namespace_by_model=True``, so a GPT-4
  answer is not served for Claude;
* the proxy scopes by model unless ``--share-across-models``.

On the default backend (``sqlite``, and it is the default for both new CLIs)
**every one of those is a no-op**. Shipping them silently would be the
``delete_user`` defect again: a documented capability the code does not have.

So: warn, once per process per backend, loudly, with the fix. Do not raise —
the cache still works and is still useful; it is the isolation that is
absent, and the caller may not need it.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = ["backend_enforces_isolation", "warn_if_scope_unenforced"]

_WARNED: set = set()

#: Backends that enforce isolation somewhere the local flag cannot see.
_ENFORCED_BY_NAME = frozenset({"SulciCloudBackend"})


class ScopeNotEnforcedWarning(UserWarning):
    """A tenant/namespace scope was requested on a backend that ignores it."""


def backend_enforces_isolation(cache: Any) -> bool:
    """
    True if ``cache``'s backend filters on ``tenant_id``.

    Returns True on anything we cannot introspect — a test double or a
    customer-authored backend gets the benefit of the doubt rather than a
    spurious warning.
    """
    backend = getattr(cache, "_backend", None) or getattr(cache, "backend", None)
    if backend is None:
        return True
    # Cloud enforces server-side; its False is a conformance-suite artefact.
    if type(backend).__name__ in _ENFORCED_BY_NAME:
        return True
    flag = getattr(backend, "ENFORCES_TENANT_ISOLATION", None)
    if flag is None:
        flag = getattr(type(backend), "ENFORCES_TENANT_ISOLATION", None)
    return True if flag is None else bool(flag)


def warn_if_scope_unenforced(cache: Any, *, feature: str) -> bool:
    """
    Emit :class:`ScopeNotEnforcedWarning` if scoping will not be enforced.

    Args:
        cache: The :class:`sulci.Cache` about to be scoped.
        feature: What the caller called the feature, for the message.

    Returns:
        True if isolation IS enforced, False if the warning fired.
    """
    if backend_enforces_isolation(cache):
        return True

    backend = getattr(cache, "_backend", None) or getattr(cache, "backend", None)
    name = type(backend).__name__ if backend is not None else "this backend"
    key = (name, feature)
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(
            f"{feature} was requested, but {name} does not enforce tenant_id "
            f"isolation — the scope will be ACCEPTED AND IGNORED, and entries "
            f"from different scopes can be served for each other. "
            f"Qdrant is currently the only backend that enforces it "
            f"(ENFORCES_TENANT_ISOLATION = True); use backend='qdrant' or the managed 'sulci' backend, or "
            f"give each scope its own db_path, or accept the sharing "
            f"deliberately by turning this feature off.",
            ScopeNotEnforcedWarning,
            stacklevel=3,
        )
    return False
