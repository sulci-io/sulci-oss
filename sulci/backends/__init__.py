# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

# sulci/backends/__init__.py
# Backend implementations are loaded dynamically by core.py.
# The Backend protocol is exported here for type-hint and conformance use.

from sulci.backends.protocol import Backend

__all__ = ["Backend"]