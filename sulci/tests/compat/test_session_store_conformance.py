# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""Public conformance suite for SessionStore implementations."""
import pytest
from sulci.sessions import SessionStore


class TestSessionStoreProtocol:
    def test_conforms_to_protocol(self, session_store):
        assert isinstance(session_store, SessionStore)


class TestBasicOperations:
    def test_empty_session_returns_empty_list(self, session_store):
        assert session_store.get("new-session") == []

    def test_append_then_get(self, session_store):
        session_store.append("s1", [0.1, 0.2, 0.3])
        result = session_store.get("s1")
        assert len(result) == 1

    def test_multiple_appends_return_all(self, session_store):
        for i in range(5):
            session_store.append("s1", [float(i)])
        assert len(session_store.get("s1")) == 5

    def test_max_turns_limits_history(self, session_store):
        for i in range(10):
            session_store.append("s1", [float(i)], max_turns=3)
        assert len(session_store.get("s1")) == 3

    def test_sessions_are_isolated(self, session_store):
        session_store.append("a", [1.0])
        session_store.append("b", [2.0])
        assert len(session_store.get("a")) == 1
        assert len(session_store.get("b")) == 1

    def test_clear_removes_session(self, session_store):
        session_store.append("s1", [1.0])
        session_store.clear("s1")
        assert session_store.get("s1") == []

    def test_clear_nonexistent_is_idempotent(self, session_store):
        session_store.clear("never-existed")   # Must not raise


class TestSummary:
    def test_summary_empty_store(self, session_store):
        s = session_store.summary()
        assert s["sessions"] == 0
        assert s["total_turns"] == 0

    def test_summary_counts_sessions(self, session_store):
        session_store.append("a", [1.0])
        session_store.append("b", [2.0])
        s = session_store.summary()
        assert s["sessions"] == 2

    def test_scoped_summary(self, session_store):
        session_store.append("a", [1.0])
        session_store.append("a", [2.0])
        s = session_store.summary(session_id="a")
        assert s["total_turns"] == 2
