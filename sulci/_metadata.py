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

The entry's own fields now always win. These names are dropped from
``metadata``, with one WARNING that names the dropped keys (never their
values); every other key is stored exactly as before.

``RESERVED_METADATA_KEYS`` is every entry field written by the backends that
store ``metadata`` beside it (Qdrant, Chroma, FAISS).
``tests/test_metadata_reserved.py`` pins it against what those backends actually
write, so a backend that starts writing a new field fails a test until the field
is listed here. The managed backend (``backend="sulci"``) does not send
``metadata`` to the gateway at all.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

log = logging.getLogger(__name__)

#: Fields of the cache entry itself. A metadata key with one of these names is ignored.
RESERVED_METADATA_KEYS = frozenset({
    "key", "query", "response", "tenant_id", "user_id", "expires", "created",
})


def without_reserved(metadata: Any) -> Any:
    """Return ``metadata`` as a new dict without the reserved keys.

    ``None`` and an empty mapping give ``{}``. An object that ``**`` can unpack
    (it has ``keys()`` and ``__getitem__``) is filtered like a mapping, whether or
    not it is registered as one. Anything else is returned unchanged, so a backend
    fails on it exactly as it did before 0.9.3. The caller's object is never
    modified.
    """
    if not metadata:
        return {}
    if not isinstance(metadata, Mapping):
        if not (hasattr(metadata, "keys") and hasattr(metadata, "__getitem__")):
            return metadata
        metadata = {k: metadata[k] for k in metadata.keys()}
    dropped = sorted(k for k in metadata if isinstance(k, str) and k in RESERVED_METADATA_KEYS)
    if dropped:
        log.warning(
            "metadata keys %s name fields of the cache entry itself and were ignored; "
            "the entry keeps its own values (reserved: %s)",
            ", ".join(dropped), ", ".join(sorted(RESERVED_METADATA_KEYS)),
        )
    return {k: v for k, v in metadata.items() if not (isinstance(k, str) and k in RESERVED_METADATA_KEYS)}
