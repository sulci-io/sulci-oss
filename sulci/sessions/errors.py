# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sessions/errors.py — 0.10.0.

``SulciSessionError``: an injected session store (``Cache(session_store=...)``) failed a WRITE. Raised by
``Cache.set`` (the cache entry WAS stored — only the conversation turn was not recorded), ``Cache.clear_context``
(the history was NOT cleared), ``Cache.clear`` (the entries WERE cleared, the histories were NOT), and a write
through a window from ``Cache.get_context``. The store's own exception is ``__cause__``. Only a USER turn WITH an
embedding is written to an injected store; assistant turns and text-only turns stay in the window.

A failed READ does not raise: ``Cache.get``, ``get_context`` and ``context_summary`` log a WARNING on the
``sulci`` logger and carry on without the history. ``Cache.cached_call`` never raises it either — it has the
response in hand, so it logs a WARNING and returns it.

Before 0.10.0 all of these failures were swallowed without a word (measured on 0.9.3: ``set``,
``clear_context`` and ``get`` against a store whose every call raised — each returned normally, nothing logged).
"""
from __future__ import annotations


class SulciSessionError(RuntimeError):
    """An injected session store failed to record or clear conversation history. See the module docstring."""
