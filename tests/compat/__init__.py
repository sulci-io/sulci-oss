# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

# tests/compat/__init__.py
# Conformance test suite for the Backend and Embedder protocols (v0.4.0+).
#
# Custom backend/embedder authors can verify their implementation conforms
# to the sulci protocols by adding their class to BACKEND_CLASSES /
# EMBEDDER_CLASSES in conftest.py and running:
#
#   pytest tests/compat/ -v
