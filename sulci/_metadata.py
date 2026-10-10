# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
sulci/_metadata.py
==================
The names a caller's ``metadata`` may not use (0.9.3).

``Cache.set(..., metadata=...)`` stores arbitrary extra fields beside an entry.
The Qdrant, Chroma and FAISS backends used to spread those fields into the
entry AFTER their own, so a metadata key named like one of the entry's own
fields replaced it: ``{"user_id": "bob"}`` filed another user's entry under
bob, ``{"tenant_id": "t2"}`` planted it in another tenant (Qdrant),
``{"expires": 0}`` made it permanent, ``{"response": ...}`` replaced the
answer served (Qdrant, FAISS).

A ``metadata`` that uses one of these names is refused with ``ValueError`` —
by ``Cache.set`` before anything is embedded, written or sent, on every
backend, and again by each backend's ``store`` that keeps ``metadata`` — as
the managed gateway refuses it with ``422``. The error names the keys, never
their values. A ``metadata`` that is not a mapping is refused with
``TypeError``. The backends that merge ``metadata`` into the entry also write
it first, so the entry's own fields would win even without the refusal.

``RESERVED_METADATA_KEYS`` is every entry field written by the backends that
store ``metadata`` beside it (Qdrant, Chroma, FAISS).
``tests/test_metadata_reserved.py`` pins it against what those backends actually
write, so a backend that starts writing a new field fails a test until the field
is listed here. The managed backend (``backend="sulci"``) does not send
``metadata`` to the gateway at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Fields of the cache entry itself. A metadata key with one of these names is refused.
RESERVED_METADATA_KEYS = frozenset({
    "key", "query", "response", "tenant_id", "user_id", "expires", "created",
})


def checked_metadata(metadata: Any) -> dict:
    """Return ``metadata`` as a new dict, or raise.

    ``None`` and an empty mapping give ``{}``. An object that ``**`` can unpack
    (it has ``keys()`` and ``__getitem__``) is read like a mapping, whether or not
    it is registered as one; anything else raises ``TypeError``. A key that names
    a field of the entry itself raises ``ValueError`` naming every such key (never
    a value). The caller's object is never modified.
    """
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        if not (hasattr(metadata, "keys") and hasattr(metadata, "__getitem__")):
            raise TypeError(f"metadata must be a mapping, not {type(metadata).__name__}")
    out = {k: metadata[k] for k in metadata.keys()}
    # str.__str__ reads the text itself: a str subclass cannot hide a reserved name behind its own __eq__/__hash__.
    reserved = sorted(str.__str__(k) for k in out if isinstance(k, str) and str.__str__(k) in RESERVED_METADATA_KEYS)
    if reserved:
        raise ValueError(
            f"metadata keys {', '.join(reserved)} name fields of the cache entry itself; rename them "
            f"(reserved: {', '.join(sorted(RESERVED_METADATA_KEYS))})"
        )
    return out
