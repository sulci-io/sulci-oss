# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan

"""
sulci/config.py — persistent SDK config (D14 / v0.5.2)
=======================================================

Reads and writes a small JSON config at ``~/.sulci/config``.

Used by:
  - :func:`sulci.connect` (future v0.6.0 device-code flow) to persist the
    ``api_key`` resolved through the browser handshake, so subsequent
    process invocations don't need to re-authenticate.
  - :mod:`sulci.telemetry` to persist a stable, anonymous ``machine_id``
    used as one input to the per-deployment fingerprint sent to
    ``/v1/analytics/deployments``.

Design rules (do not relax without an ADR):

  1. **0600 permissions.** The file may contain an api_key. We refuse to
     write a world-readable config. On read we tolerate any mode (the
     user may have legitimately tightened it further).
  2. **Silent fallback on corruption.** A malformed file must never
     prevent ``import sulci`` from succeeding or a ``Cache(...)`` call
     from working. :func:`load` returns ``{}`` on any read failure.
  3. **No PII by construction.** ``machine_id`` is a freshly-generated
     ``uuid4`` — it does not encode the MAC address, hostname, or any
     filesystem path. The same machine running two unrelated installs
     in two unrelated venvs would produce the same id only because they
     share ``$HOME``, which is the desired "deployment identity" notion.
  4. **No locking.** The config is written by one local process at a
     time during normal use (``sulci.connect()`` and the first call to
     :func:`get_machine_id` after install). Last-write-wins is fine.

This module is intentionally dependency-free — only stdlib — so it
imports cleanly even in the bare ``__init__.py``-only test environments
described in ``sulci/__init__.py``.
"""
from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any, Dict


# ── Paths ─────────────────────────────────────────────────────────────────────

# We resolve ``~`` lazily on every call so tests can monkeypatch ``$HOME``
# without import-order surprises. Cheap (one stat) — not a hot path.

def _config_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".sulci"


def _config_path() -> Path:
    return _config_dir() / "config"


# ── Public API ────────────────────────────────────────────────────────────────

def load() -> Dict[str, Any]:
    """Return the parsed config dict, or ``{}`` on any failure.

    Failures swallowed silently (per design rule 2):
      - file does not exist
      - file is not valid JSON
      - file is unreadable for any OSError reason
      - file contents are not a JSON object (e.g. ``[1, 2, 3]``)
    """
    try:
        path = _config_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def save(data: Dict[str, Any]) -> bool:
    """Write ``data`` to ``~/.sulci/config`` with mode 0600.

    Returns ``True`` on success, ``False`` on any IOError. Never raises.

    The directory is created with mode 0700 if missing. The file is
    written atomically via a tempfile + rename so a partial write
    cannot corrupt an existing config — important because :func:`load`
    is the only line of defense between a corrupt file and a crash on
    next import.

    .. warning::
       **The 0700/0600 hardening is POSIX-only.** ``os.chmod`` on Windows
       toggles the read-only bit and does not apply POSIX mode bits, so on
       Windows this file is left at whatever the profile directory grants —
       measured as 0o777/0o666 on 2026-08-11, the first time the assertions
       ran there. The file holds the API key.

       On a single-user Windows machine the profile ACL is usually adequate.
       It is not the guarantee the paragraph above implies, and restricting
       it properly needs an ACL call (pywin32 / ``icacls``), not ``chmod``.
       See SECURITY.md.
    """
    try:
        cfg_dir = _config_dir()
        cfg_dir.mkdir(mode=0o700, exist_ok=True)
        # Tighten dir mode in case it pre-existed with looser perms.
        try:
            os.chmod(cfg_dir, 0o700)
        except OSError:
            pass

        path = _config_path()
        tmp = path.with_suffix(".tmp")
        # Open the tempfile with 0600 from the start — never world-readable
        # even between create and chmod.
        fd = os.open(
            str(tmp),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            mode=stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
        # Belt-and-braces: ensure final file is 0600 (older umasks etc.)
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        return True
    except OSError:
        return False


def update(**fields: Any) -> bool:
    """Read-modify-write helper. Merges ``fields`` into the existing config.

    A corrupt or missing file is treated as ``{}`` (so ``update`` can be
    called as a first-write). Returns ``True`` on a successful save.

    When ``api_key`` is among the fields being written, auto-stamps a
    ``written_at`` UTC ISO-8601 timestamp on the saved dict. This lets
    :func:`sulci.connect` detect stale persisted keys and refuse to
    use a config that's older than the staleness threshold (90 days).
    Added 2026-05-13 (sulci-oss #80). Other field writes (e.g.
    ``machine_id``-only via :func:`get_machine_id`) do not stamp,
    because they don't represent a fresh authentication event.
    """
    import datetime
    data = load()
    data.update(fields)
    if "api_key" in fields:
        data["written_at"] = datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat(timespec="seconds")
    return save(data)


def get_machine_id() -> str:
    """Return a stable, anonymous ``machine_id`` for this install.

    Generated on first call (a fresh ``uuid4``) and persisted to
    ``~/.sulci/config``. Subsequent calls return the same value.

    If the config is unwritable (read-only home, weird permissions, etc.)
    we still return a UUID — but that UUID is process-local, not stable
    across restarts. The fingerprint built on top of it will still be
    valid (same shape, same bytes), it just won't deduplicate across
    runs on the dashboard. A degraded but functional fallback.
    """
    data = load()
    mid = data.get("machine_id")
    if isinstance(mid, str) and mid:
        return mid
    new_mid = uuid.uuid4().hex
    update(machine_id=new_mid)   # may fail silently — that's OK
    return new_mid
