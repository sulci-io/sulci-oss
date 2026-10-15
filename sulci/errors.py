# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/errors.py — 0.10.0.

``SulciErasureError``: an erasure that Sulci could not confirm. Raised by ``Cache.delete_user`` (and
``AsyncCache.adelete_user``) on every backend, and by the managed backend's ``SulciCloudBackend.delete_user`` and
``clear`` — and so by ``Cache.clear()`` there. 0.9.3 defined it in ``sulci.backends.cloud``; it lives here because
both paths raise it, and ``sulci.backends.cloud`` imports it from here.
"""
from __future__ import annotations


def _short(value, limit: int = 200) -> str:
    """repr() for a log line or message: control characters escaped, so a value cannot forge a line;
    truncated, so a 100 kB id does not become a 100 kB log line."""
    r = repr(value)
    return r if len(r) <= limit else r[:limit] + f"…(+{len(r) - limit} chars)"


class SulciErasureError(RuntimeError):
    """An erasure that did not complete, or could not be confirmed. The message names the operation and what
    failed, never the user id.

    Raised by:

    - ``Cache.delete_user`` / ``AsyncCache.adelete_user`` (0.10.0) when the backend fails to erase the user's
      entries (``entries`` and ``sessions`` are ``None``: the session store was not touched), or when the session
      store fails after the entries were erased (``entries`` is the count erased, ``sessions`` is ``None``).
      Either way the call can be repeated: it is idempotent, and a retry completes the erasure.
    - ``SulciCloudBackend.delete_user`` / ``clear`` (0.9.3) — and so ``Cache.delete_user`` and ``Cache.clear()``
      on the managed backend — whenever the gateway does not confirm the erasure: a transport error, any non-2xx
      answer, a 2xx answer that is not ``{"status": "ok", ...}``, or a user id that cannot be encoded (nothing is
      sent). Through ``Cache.delete_user``, a ``503 erasure_incomplete`` answer sets ``entries`` / ``sessions``
      for the half the gateway reports complete; the half that failed stays ``None``, and any partial count the
      gateway gives for it is in ``body`` (``deleted``, ``sessions_deleted``) and in the message.

    Attributes:
        operation: ``"delete_user"`` or ``"clear"``.
        status:    the HTTP status, or ``None`` when no response arrived or the erasure was local.
        error:     the gateway's error code (``"validation_error"``, ``"erasure_incomplete"``,
                   ``"unexpected_response"``, …); with no response, the transport error's TYPE
                   (``"ConnectError"``) — its text, which can carry the URL and so the user id,
                   stays on ``__cause__``; ``"unencodable user_id (…)"`` when nothing was sent; for a local
                   erasure, the type of the backend's or the session store's exception (its text stays on
                   ``__cause__``).
        body:      the parsed JSON answer, when there was one.
        entries:   the count of cache entries erased when that half of the erasure completed; ``None`` when it
                   did not (or was not attempted).
        sessions:  the count of conversation sessions erased when that half completed; ``None`` when it did not.
    """

    def __init__(self, operation: str, status, error: str, body=None, entries=None, sessions=None,
                 detail=None):
        self.operation, self.status, self.error, self.body = operation, status, error, body
        self.entries, self.sessions, self.detail = entries, sessions, detail
        if detail is not None:
            super().__init__(f"sulci {operation} failed: {detail}")
        else:
            where = "no response" if status is None else f"HTTP {status}"
            super().__init__(f"sulci {operation} failed: {where}: {_short(error)}")

    def __reduce__(self):  # picklable (multiprocessing, task queues)
        return (type(self), (self.operation, self.status, self.error, self.body, self.entries, self.sessions,
                             self.detail))
