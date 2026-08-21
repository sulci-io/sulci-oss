# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci.proxy
────────────
OpenAI- and Anthropic-compatible caching shim. See :mod:`sulci.proxy.app`.

Importing this package requires the ``proxy`` extra::

    pip install "sulci[proxy]"
"""
from sulci.proxy.app import build_app, extract_prompt, is_cacheable_response

__all__ = ["build_app", "extract_prompt", "is_cacheable_response"]
