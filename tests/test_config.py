"""
tests/test_config.py
====================
Unit tests for sulci.config — D14 (v0.5.2).

Coverage
--------
- load() returns {} on missing file
- load() returns {} on malformed JSON (silent fallback)
- load() returns {} on non-object JSON (e.g. [1,2,3])
- load() returns parsed dict on valid JSON
- save() creates ~/.sulci/ with 0700 permissions
- save() writes file with 0600 permissions
- save() is atomic (tempfile + rename)
- save() round-trips through load()
- update() merges into existing config
- update() works as first-write (no existing file)
- get_machine_id() generates a uuid on first call, persists it, returns same on second call
- get_machine_id() degrades gracefully when home is unwritable (returns a UUID anyway)
"""
from __future__ import annotations

import json
import os
import stat
import pytest

import sulci.config as cfg


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """Redirect ~ to a tempdir so we never touch the real ~/.sulci."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # On some systems os.path.expanduser also consults USERPROFILE.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


# ── load() ────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_missing_file_returns_empty_dict(self, tmp_home):
        assert cfg.load() == {}

    def test_malformed_json_returns_empty_dict(self, tmp_home):
        cfg_dir = tmp_home / ".sulci"
        cfg_dir.mkdir()
        (cfg_dir / "config").write_text("{not json")
        assert cfg.load() == {}

    def test_non_object_json_returns_empty_dict(self, tmp_home):
        """A JSON list/string/number should not crash — silent fallback."""
        cfg_dir = tmp_home / ".sulci"
        cfg_dir.mkdir()
        (cfg_dir / "config").write_text("[1, 2, 3]")
        assert cfg.load() == {}

    def test_valid_object_returns_parsed_dict(self, tmp_home):
        cfg_dir = tmp_home / ".sulci"
        cfg_dir.mkdir()
        (cfg_dir / "config").write_text(json.dumps({"api_key": "sk-x", "machine_id": "abc"}))
        loaded = cfg.load()
        assert loaded == {"api_key": "sk-x", "machine_id": "abc"}

    def test_unreadable_file_returns_empty_dict(self, tmp_home):
        """Permission errors must not propagate."""
        cfg_dir = tmp_home / ".sulci"
        cfg_dir.mkdir()
        f = cfg_dir / "config"
        f.write_text(json.dumps({"x": 1}))
        os.chmod(f, 0)
        try:
            # Root in CI containers can read regardless of mode.
            # Skip the assertion in that case rather than fake a result.
            # os.geteuid() does not exist on Windows, and chmod(f, 0) there
            # only toggles the read-only bit -- the file stays readable. This
            # assertion is about POSIX mode bits, so it is POSIX-only.
            if not hasattr(os, "geteuid"):
                pytest.skip("POSIX-only: chmod(0) does not deny read on Windows")
            if os.geteuid() == 0:
                pytest.skip("Cannot test unreadable file as root")
            assert cfg.load() == {}
        finally:
            os.chmod(f, stat.S_IRUSR | stat.S_IWUSR)


# ── save() ────────────────────────────────────────────────────────────────────

class TestSave:
    def test_creates_directory(self, tmp_home):
        assert cfg.save({"api_key": "sk-test"}) is True
        assert (tmp_home / ".sulci").is_dir()

    # ⚠️ THESE TWO ARE POSIX-ONLY, AND THAT IS A REAL GAP, NOT A TEST QUIRK.
    #
    # sulci.config.save() hardens ~/.sulci to 0700 and ~/.sulci/config to
    # 0600 because that file holds the API key. os.chmod on Windows only
    # toggles the read-only bit; the POSIX mode bits are not applied. First
    # measured 2026-08-11, when these assertions ran on Windows for the very
    # first time and reported 0o777 and 0o666.
    #
    # So on Windows the key file is NOT restricted by sulci. It inherits the
    # ACL of the user profile directory, which is usually adequate on a
    # single-user machine and is NOT the guarantee the docstring implies.
    # Recorded in SECURITY.md and in the config.save() docstring.
    #
    # Skipping is right for the ASSERTION -- 0600 is not expressible via
    # chmod on Windows -- and wrong as a resolution. Proper hardening needs
    # an ACL call (pywin32 / icacls). Tracked, not fixed here.
    @pytest.mark.skipif(
        os.name == "nt",
        reason="POSIX mode bits; Windows needs ACLs -- see SECURITY.md",
    )
    def test_directory_mode_is_0700(self, tmp_home):
        cfg.save({"api_key": "sk-test"})
        mode = stat.S_IMODE((tmp_home / ".sulci").stat().st_mode)
        assert mode == 0o700, f"expected 0o700, got {oct(mode)}"

    @pytest.mark.skipif(
        os.name == "nt",
        reason="POSIX mode bits; Windows needs ACLs -- see SECURITY.md",
    )
    def test_file_mode_is_0600(self, tmp_home):
        cfg.save({"api_key": "sk-test"})
        mode = stat.S_IMODE((tmp_home / ".sulci" / "config").stat().st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_round_trips_through_load(self, tmp_home):
        original = {"api_key": "sk-roundtrip", "machine_id": "x" * 32}
        cfg.save(original)
        assert cfg.load() == original

    def test_overwrites_existing_file(self, tmp_home):
        cfg.save({"a": 1})
        cfg.save({"b": 2})
        assert cfg.load() == {"b": 2}

    def test_no_tempfile_left_behind(self, tmp_home):
        cfg.save({"x": 1})
        tmps = list((tmp_home / ".sulci").glob("*.tmp"))
        assert tmps == []


# ── update() ──────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_merges_into_existing_config(self, tmp_home):
        cfg.save({"api_key": "sk-x"})
        cfg.update(machine_id="abc123")
        loaded = cfg.load()
        assert loaded == {"api_key": "sk-x", "machine_id": "abc123"}

    def test_update_overwrites_existing_field(self, tmp_home):
        cfg.save({"api_key": "old"})
        cfg.update(api_key="new")
        assert cfg.load()["api_key"] == "new"

    def test_update_works_as_first_write(self, tmp_home):
        # No existing file
        cfg.update(machine_id="firstwrite")
        assert cfg.load() == {"machine_id": "firstwrite"}

    def test_update_treats_corrupt_file_as_empty(self, tmp_home):
        """A corrupt file should not block an update — treated as {}."""
        cfg_dir = tmp_home / ".sulci"
        cfg_dir.mkdir()
        (cfg_dir / "config").write_text("garbage")
        cfg.update(machine_id="recovery")
        assert cfg.load() == {"machine_id": "recovery"}


# ── get_machine_id() ──────────────────────────────────────────────────────────

class TestGetMachineId:
    def test_generates_on_first_call(self, tmp_home):
        mid = cfg.get_machine_id()
        assert isinstance(mid, str)
        assert len(mid) == 32   # uuid4().hex
        assert int(mid, 16) >= 0   # must be valid hex

    def test_persists_across_calls(self, tmp_home):
        mid1 = cfg.get_machine_id()
        mid2 = cfg.get_machine_id()
        assert mid1 == mid2

    def test_persists_across_module_reloads(self, tmp_home):
        """Re-importing the module must not regenerate the id."""
        import importlib, sulci.config as c
        mid1 = c.get_machine_id()
        importlib.reload(c)
        mid2 = c.get_machine_id()
        assert mid1 == mid2

    def test_returns_uuid_even_on_unwritable_home(self, tmp_home, monkeypatch):
        """If save fails, we still return a (process-local) UUID — degraded but functional."""
        monkeypatch.setattr(cfg, "save", lambda *a, **kw: False)
        mid = cfg.get_machine_id()
        assert isinstance(mid, str)
        assert len(mid) == 32

    def test_existing_machine_id_is_preserved(self, tmp_home):
        """If config already has a machine_id, return it verbatim."""
        cfg.save({"machine_id": "preexisting"})
        assert cfg.get_machine_id() == "preexisting"


class TestWrittenAtStamping:
    """update() auto-stamps written_at when api_key is among the fields.

    Added 2026-05-13 for sulci-oss #80 — the config staleness guard. The
    api_key/written_at pair must always travel together so the resolution
    chain in sulci.connect() can detect stale persisted keys.

    Other field writes (machine_id-only, etc.) must NOT trip the stamp,
    because they don't represent a fresh authentication event.
    """

    def test_update_with_api_key_stamps_written_at(self, tmp_home):
        """A cfg.update(api_key=...) call must persist a written_at
        timestamp alongside the key."""
        cfg.update(api_key="sk-sulci-fresh-aaaaaaaaaaaaaaaaaaaa")
        data = cfg.load()
        assert data["api_key"] == "sk-sulci-fresh-aaaaaaaaaaaaaaaaaaaa"
        assert "written_at" in data
        assert "T" in data["written_at"]
        assert data["written_at"].endswith("+00:00")

    def test_update_without_api_key_does_not_stamp(self, tmp_home):
        """A cfg.update(machine_id=...) call (the get_machine_id() path)
        must NOT stamp written_at."""
        cfg.update(machine_id="abc123")
        data = cfg.load()
        assert data["machine_id"] == "abc123"
        assert "written_at" not in data

    def test_update_with_both_api_key_and_other_fields_stamps(self, tmp_home):
        """Mixed-field update — api_key plus extras — still stamps written_at."""
        cfg.update(api_key="sk-sulci-bbbbbbbbbbbbbbbbbbbbbbbbb",
                      machine_id="def456")
        data = cfg.load()
        assert data["api_key"] == "sk-sulci-bbbbbbbbbbbbbbbbbbbbbbbbb"
        assert data["machine_id"] == "def456"
        assert "written_at" in data

    def test_update_with_api_key_refreshes_existing_written_at(self, tmp_home):
        """When a config already has a written_at from a previous write,
        the next api_key write must REFRESH it (not keep the old one)."""
        import time
        cfg.update(api_key="sk-sulci-first-aaaaaaaaaaaaaaaaaaaa")
        first = cfg.load()["written_at"]
        time.sleep(1.1)
        cfg.update(api_key="sk-sulci-second-aaaaaaaaaaaaaaaaaaaa")
        second = cfg.load()["written_at"]
        assert second > first  # ISO timestamps sort lexically
