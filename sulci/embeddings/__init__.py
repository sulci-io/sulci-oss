# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

# sulci/embeddings/__init__.py
# Embedding models are loaded dynamically by core.py.
# The Embedder protocol is exported here for type-hint and conformance use.

from sulci.embeddings.protocol import Embedder

__all__ = ["Embedder"]