# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
tests/compat/conftest.py
========================
Fixtures and registries for the Backend/Embedder conformance suite.

To validate a custom backend or embedder, add the class to the
appropriate registry below and run `pytest tests/compat/`.
"""
from __future__ import annotations
import importlib
import os
import tempfile
from typing import Any, Optional, Type

import pytest
import uuid


# -----------------------------------------------------------------------------
# Per-pytest-session UUID prefix for Redis-backed test fixtures.
# -----------------------------------------------------------------------------
# Prevents cross-project Redis interference: tests namespace their keys
# under "sulci:test:<run-id>:*" instead of plain "sulci:*", so they
# never collide with sulci-platform's runtime services or other test
# runs against the same Redis daemon. See sulci-io/sulci-oss#29.
TEST_RUN_ID     = uuid.uuid4().hex[:8]
TEST_KEY_PREFIX = f"sulci:test:{TEST_RUN_ID}:"


# -----------------------------------------------------------------------------
# Registries — extend these to test custom implementations.
# -----------------------------------------------------------------------------

def _import_or_none(module_path: str, attr: str) -> Optional[Type]:
    """Try to import attr from module_path; return None if module missing."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    except ImportError:
        return None


BACKEND_CLASSES = [
    cls for cls in [
        _import_or_none("sulci.backends.sqlite", "SQLiteBackend"),
        _import_or_none("sulci.backends.chroma", "ChromaBackend"),
        _import_or_none("sulci.backends.faiss",  "FAISSBackend"),
        _import_or_none("sulci.backends.redis",  "RedisBackend"),
        _import_or_none("sulci.backends.qdrant", "QdrantBackend"),
        _import_or_none("sulci.backends.milvus", "MilvusBackend"),
        # v0.6.0 (sulci-oss #62, umbrella #63) — SulciCloudBackend is
        # deliberately NOT in this list. It is a TRANSPORT for the entire
        # Cache.get/set call, not a Backend protocol implementer in the
        # vector-search sense. It exposes remote_get / remote_set rather
        # than search / store / clear, and Cache.get/set detects it via
        # `hasattr(backend, "remote_get")` (see Cache._is_remote_transport).
        #
        # The transport surface is covered by tests in
        # tests/test_cloud_backend.py:
        #   - TestRemoteGet, TestRemoteSet — method shape + payload
        #   - TestCanonicalGatewayPaths   — wire-contract pydantic round-trip
        #   - TestCloudTransportShortCircuit — Cache.get/set routing
        #
        # If a second transport implementation emerges, consider adding a
        # parallel TRANSPORT_CLASSES registry here with a dedicated
        # transport-conformance suite.
    ]
    if cls is not None
]

EMBEDDER_CLASSES = [
    cls for cls in [
        _import_or_none("sulci.embeddings.minilm", "MiniLMEmbedder"),
        _import_or_none("sulci.embeddings.openai", "OpenAIEmbedder"),
    ]
    if cls is not None
]


# -----------------------------------------------------------------------------
# Construction helpers — return a live instance, or None if local construction
# isn't possible (missing dep, no running server, no api key, etc.). Tests use
# the None signal to skip behavioral checks while still running structural ones.
# -----------------------------------------------------------------------------

def _try_construct_backend(cls: Type) -> Optional[Any]:
    """
    Try to construct a backend instance with sensible local defaults.
    Return None if the backend can't run locally in this environment.
    """
    name = cls.__name__

    if name == "SQLiteBackend":
        # SQLite always works — uses a temp directory.
        return cls(db_path=tempfile.mkdtemp(prefix="sulci_compat_"))

    if name == "FAISSBackend":
        try:
            import faiss  # noqa: F401
        except ImportError:
            return None
        return cls(db_path=tempfile.mkdtemp(prefix="sulci_compat_"))

    if name == "ChromaBackend":
        try:
            import chromadb  # noqa: F401
        except ImportError:
            return None
        return cls(db_path=tempfile.mkdtemp(prefix="sulci_compat_"))

    if name == "QdrantBackend":
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            return None
        # QdrantBackend supports embedded mode via db_path — same wire
        # format as a remote Qdrant server, no infrastructure needed.
        # If the embedded client fails to construct (rare), skip
        # behavioral tests rather than blocking the suite.
        try:
            return cls(db_path=tempfile.mkdtemp(prefix="sulci_compat_"))
        except Exception:
            return None

    if name == "RedisBackend":
        try:
            import redis  # noqa: F401
        except ImportError:
            return None
        # Requires a running redis-server on localhost. Probe and skip if absent.
        try:
            instance = cls(key_prefix=TEST_KEY_PREFIX)
            instance._redis.ping()  # connection check
            return instance
        except Exception:
            return None

    if name == "MilvusBackend":
        try:
            import pymilvus  # noqa: F401
        except ImportError:
            return None
        try:
            return cls(db_path=tempfile.mkdtemp(prefix="sulci_compat_"))
        except Exception:
            return None

    return None


def _try_construct_embedder(cls: Type) -> Optional[Any]:
    """
    Try to construct an embedder instance. Return None if not possible.
    """
    name = cls.__name__

    if name == "MiniLMEmbedder":
        # Always works locally; first call may download the MiniLM model.
        try:
            return cls()
        except Exception:
            return None

    if name == "OpenAIEmbedder":
        # Constructs without a key, but behavioral tests need OPENAI_API_KEY
        # to actually call the API. Return the instance; tests will skip
        # behavioral assertions when the env var is missing.
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        try:
            return cls()
        except Exception:
            return None

    return None


# -----------------------------------------------------------------------------
# Pytest parametrize fixtures — yield (cls, instance_or_None) tuples.
# -----------------------------------------------------------------------------

@pytest.fixture(params=BACKEND_CLASSES, ids=lambda c: c.__name__)
def backend_class(request) -> Type:
    """Yields each registered backend class (no instance)."""
    return request.param


@pytest.fixture
def backend_instance(backend_class) -> Any:
    """
    Yields a live backend instance, or skips if the backend cannot be
    constructed locally (missing dep, no running server, etc.).
    """
    inst = _try_construct_backend(backend_class)
    if inst is None:
        pytest.skip(f"{backend_class.__name__}: no local construction available")
    # Defensive setup clear — guards against state leaked by an earlier
    # test that crashed before reaching teardown. SQLite/Qdrant get a
    # fresh tmp_path/collection each fixture call, so this is mostly a
    # no-op for them; matters for Redis where the daemon is shared.
    try:
        inst.clear()
    except Exception:
        pass
    yield inst
    # Best-effort cleanup
    try:
        inst.clear()
    except Exception:
        pass


@pytest.fixture(params=EMBEDDER_CLASSES, ids=lambda c: c.__name__)
def embedder_class(request) -> Type:
    """Yields each registered embedder class (no instance)."""
    return request.param


@pytest.fixture
def embedder_instance(embedder_class) -> Any:
    """
    Yields a live embedder instance, or skips if the embedder cannot be
    constructed locally (missing dep, missing api key, etc.).
    """
    inst = _try_construct_embedder(embedder_class)
    if inst is None:
        pytest.skip(f"{embedder_class.__name__}: no local construction available")
    yield inst
