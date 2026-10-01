"""
tests/test_backends.py
======================
Backend-specific tests. Each backend is skipped if its
dependency is not installed — no test failures for missing extras.

Run all:           pytest tests/test_backends.py -v
Run SQLite only:   pytest tests/test_backends.py -v -k sqlite
Run Chroma only:   pytest tests/test_backends.py -v -k chroma
"""
import pytest
import uuid

# Per-pytest-session UUID prefix for Redis-backed test fixtures. See
# tests/compat/conftest.py and sulci-io/sulci-oss#29 for context.
_TEST_RUN_ID     = uuid.uuid4().hex[:8]
_TEST_KEY_PREFIX = f"sulci:test:{_TEST_RUN_ID}:"
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Skip helpers ──────────────────────────────────────────────

def has_package(name: str) -> bool:
    import importlib
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

skip_chroma  = pytest.mark.skipif(not has_package("chromadb"),         reason="chromadb not installed")
skip_qdrant  = pytest.mark.skipif(not has_package("qdrant_client"),     reason="qdrant-client not installed")
skip_faiss   = pytest.mark.skipif(not has_package("faiss"),             reason="faiss-cpu not installed")
skip_redis   = pytest.mark.skipif(not has_package("redis"),             reason="redis not installed")
skip_milvus  = pytest.mark.skipif(not has_package("pymilvus"),          reason="pymilvus not installed")

# ── Shared test logic ─────────────────────────────────────────

DUMMY_VEC = [0.1] * 384   # 384-dim zero vector (MiniLM dimension)


def _run_backend_contract(backend):
    """
    Shared contract tests — every backend must pass these.
    Operates at the raw backend level (no embedder needed).
    """
    import math

    # normalise dummy vec so cosine similarity works
    norm = math.sqrt(sum(x * x for x in DUMMY_VEC))
    vec  = [x / norm for x in DUMMY_VEC]

    # 1. Empty search returns None
    result, sim = backend.search(vec, threshold=0.85)
    assert result is None
    assert sim == 0.0

    # 2. Store and retrieve exact match
    backend.store(
        key       = "abc123",
        query     = "What is Python?",
        response  = "Python is a language.",
        embedding = vec,
    )
    result, sim = backend.search(vec, threshold=0.85)
    assert result == "Python is a language."
    assert sim >= 0.99

    # 3. Clear removes all entries
    backend.clear()
    result, sim = backend.search(vec, threshold=0.85)
    assert result is None

    # 4. TTL expiry
    backend.store(
        key       = "ttl_key",
        query     = "expiring query",
        response  = "expiring response",
        embedding = vec,
        expires   = time.time() - 1,   # already expired
    )
    result, sim = backend.search(vec, threshold=0.85)
    assert result is None, "Expired entry should not be returned"

    # 5. User-scoped search
    backend.store(
        key       = "user_key",
        query     = "scoped query",
        response  = "alice response",
        embedding = vec,
        user_id   = "alice",
    )
    # Alice gets her entry
    result, _ = backend.search(vec, threshold=0.85, user_id="alice")
    assert result == "alice response"
    # Bob does not
    result, _ = backend.search(vec, threshold=0.85, user_id="bob")
    assert result is None

    # 6. Final cleanup — leave no entries behind for shared backends like Redis
    backend.clear()


# ── SQLite ────────────────────────────────────────────────────

class TestSQLiteBackend:

    def test_contract(self, tmp_path):
        from sulci.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(db_path=str(tmp_path / "sqlite_test"))
        _run_backend_contract(backend)

    def test_persistence(self, tmp_path):
        """Data survives re-opening the database."""
        import math
        vec  = [0.1] * 384
        norm = math.sqrt(sum(x * x for x in vec))
        vec  = [x / norm for x in vec]

        from sulci.backends.sqlite import SQLiteBackend
        db_path = str(tmp_path / "persist_db")

        # Write
        b1 = SQLiteBackend(db_path=db_path)
        b1.store("k1", "What is Python?", "Python is a language.", vec)

        # Re-open and read
        b2 = SQLiteBackend(db_path=db_path)
        result, sim = b2.search(vec, threshold=0.85)
        assert result == "Python is a language."


class TestSQLiteScopeKey:
    """
    Cache.set derives the backend key from the query text alone. The
    SQLite row must therefore be unique on (key, tenant_id, user_id):
    the same query stored under two scopes is two rows, each served
    only its own response.
    """

    QUERY = "what is my account balance"

    def _cache(self, tmp_path, fake_embedder):
        from sulci import Cache
        return Cache(
            backend="sqlite", db_path=str(tmp_path / "scope"),
            embedding_model=fake_embedder, personalized=True,
            telemetry=False,
        )

    def test_same_query_two_users_each_get_their_own(self, tmp_path, fake_embedder):
        cache = self._cache(tmp_path, fake_embedder)
        cache.set(self.QUERY, "alice: $10", user_id="alice")
        cache.set(self.QUERY, "bob: $99",   user_id="bob")

        assert cache.get(self.QUERY, user_id="alice")[0] == "alice: $10"
        assert cache.get(self.QUERY, user_id="bob")[0]   == "bob: $99"

    def test_same_query_two_tenants_each_get_their_own(self, tmp_path, fake_embedder):
        cache = self._cache(tmp_path, fake_embedder)
        cache.set(self.QUERY, "acme: $10",   tenant_id="acme",   user_id="u-acme")
        cache.set(self.QUERY, "globex: $99", tenant_id="globex", user_id="u-globex")

        assert cache.get(self.QUERY, tenant_id="acme",   user_id="u-acme")[0]   == "acme: $10"
        assert cache.get(self.QUERY, tenant_id="globex", user_id="u-globex")[0] == "globex: $99"

    def test_same_key_two_tenants_is_two_rows(self, tmp_path):
        """
        Backend level, same user: a second tenant's write must not
        replace the first tenant's row. (search() does not filter on
        tenant_id — ENFORCES_TENANT_ISOLATION is False — so this is
        asserted on the stored rows, not on a lookup.)
        """
        from sulci.backends.sqlite import SQLiteBackend
        b = SQLiteBackend(db_path=str(tmp_path / "rows"))
        b.store("k", self.QUERY, "acme",   [1.0, 0.0], tenant_id="acme",   user_id="u")
        b.store("k", self.QUERY, "globex", [1.0, 0.0], tenant_id="globex", user_id="u")

        rows = b._conn.execute(
            "SELECT tenant_id, user_id, response FROM cache ORDER BY id"
        ).fetchall()
        assert [tuple(r) for r in rows] == [
            ("acme", "u", "acme"), ("globex", "u", "globex"),
        ]

    def test_same_scope_rewrite_still_updates_in_place(self, tmp_path):
        from sulci.backends.sqlite import SQLiteBackend
        b = SQLiteBackend(db_path=str(tmp_path / "upsert"))
        b.store("k", self.QUERY, "old", [1.0, 0.0], tenant_id="t", user_id="u")
        b.store("k", self.QUERY, "new", [1.0, 0.0], tenant_id="t", user_id="u")

        rows = b._conn.execute("SELECT response FROM cache").fetchall()
        assert [r[0] for r in rows] == ["new"]
        assert b.search([1.0, 0.0], 0.85, user_id="u")[0] == "new"


class TestSQLiteSchemaMigration:
    """A database written by 0.9.1 (user_version 0) opens under 0.9.2."""

    V0_SCHEMA = """
        CREATE TABLE cache (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            key       TEXT UNIQUE NOT NULL,
            query     TEXT NOT NULL,
            response  TEXT NOT NULL,
            embedding BLOB NOT NULL,
            user_id   TEXT NOT NULL DEFAULT 'global',
            expires   REAL NOT NULL DEFAULT 0,
            created   REAL NOT NULL,
            metadata  TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX idx_user    ON cache(user_id);
        CREATE INDEX idx_expires ON cache(expires);
    """

    def _write_v0(self, db_dir):
        import sqlite3, struct
        os.makedirs(db_dir)
        conn = sqlite3.connect(os.path.join(db_dir, "sulci.db"))
        conn.executescript(self.V0_SCHEMA)
        vec = struct.pack("2f", 1.0, 0.0)
        conn.executemany(
            "INSERT INTO cache (key, query, response, embedding, user_id, created)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("g1", "shared q",   "shared answer", vec, "global", 1.0),
                ("p1", "personal q", "someone's",     vec, "alice",  2.0),
                ("p2", "other q",    "someone else's", vec, "bob",   3.0),
            ],
        )
        conn.commit()
        conn.close()

    def test_discards_every_pre_upgrade_row(self, tmp_path):
        """No pre-0.9.2 row survives — with or without a user_id."""
        from sulci.backends.sqlite import SQLiteBackend, SCHEMA_VERSION
        db_dir = str(tmp_path / "v091")
        self._write_v0(db_dir)

        with pytest.warns(RuntimeWarning, match="discarded 3 entries written before 0.9.2"):
            b = SQLiteBackend(db_path=db_dir)

        assert b._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0
        assert b._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert b.search([1.0, 0.0], 0.85)[0] is None
        cols = [r[1] for r in b._conn.execute("PRAGMA table_info(cache)")]
        assert "tenant_id" in cols
        # The old single-column UNIQUE is gone: two users can now hold the key.
        b.store("p1", "personal q", "alice's", [1.0, 0.0], user_id="alice")
        b.store("p1", "personal q", "bob's",   [1.0, 0.0], user_id="bob")
        assert b.search([1.0, 0.0], 0.85, user_id="alice")[0] == "alice's"
        assert b.search([1.0, 0.0], 0.85, user_id="bob")[0]   == "bob's"

    def test_reopen_after_migration_is_a_no_op(self, tmp_path):
        import warnings
        from sulci.backends.sqlite import SQLiteBackend
        db_dir = str(tmp_path / "v091")
        self._write_v0(db_dir)
        with pytest.warns(RuntimeWarning):
            b1 = SQLiteBackend(db_path=db_dir)
        b1.store("k", "q", "written after the upgrade", [1.0, 0.0])
        b1._conn.close()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            b = SQLiteBackend(db_path=db_dir)
        assert b.search([1.0, 0.0], 0.85)[0] == "written after the upgrade"

    def test_empty_v0_database_migrates_silently(self, tmp_path):
        import sqlite3, warnings
        from sulci.backends.sqlite import SQLiteBackend
        db_dir = str(tmp_path / "v091")
        self._write_v0(db_dir)
        conn = sqlite3.connect(os.path.join(db_dir, "sulci.db"))
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            b = SQLiteBackend(db_path=db_dir)
        cols = [r[1] for r in b._conn.execute("PRAGMA table_info(cache)")]
        assert "tenant_id" in cols


# ── ChromaDB ──────────────────────────────────────────────────

class TestChromaBackend:

    @skip_chroma
    def test_contract(self, tmp_path):
        from sulci.backends.chroma import ChromaBackend
        backend = ChromaBackend(db_path=str(tmp_path / "chroma_test"))
        _run_backend_contract(backend)

    @skip_chroma
    def test_multiple_entries_ranked(self, tmp_path):
        """More similar entry should be returned over less similar."""
        import math
        from sulci.backends.chroma import ChromaBackend

        def unit(v):
            norm = math.sqrt(sum(x * x for x in v))
            return [x / norm for x in v]

        backend = ChromaBackend(db_path=str(tmp_path / "chroma_rank"))
        vec_a   = unit([1.0] + [0.0] * 383)           # pure dim-0
        vec_b   = unit([0.0, 1.0] + [0.0] * 382)      # pure dim-1
        query   = unit([0.05, 0.999] + [0.0] * 382)   # near dim-1 → matches vec_b

        backend.store("a", "query A", "Response A", vec_a)
        backend.store("b", "query B", "Response B", vec_b)

        result, sim = backend.search(query, threshold=0.5)
        assert result == "Response B"


# ── FAISS ─────────────────────────────────────────────────────

class TestFAISSBackend:

    @skip_faiss
    def test_contract(self, tmp_path):
        from sulci.backends.faiss import FAISSBackend
        backend = FAISSBackend(db_path=str(tmp_path / "faiss_test"))
        _run_backend_contract(backend)

    @skip_faiss
    def test_persistence(self, tmp_path):
        import math
        from sulci.backends.faiss import FAISSBackend
        vec  = [0.1] * 384
        norm = math.sqrt(sum(x * x for x in vec))
        vec  = [x / norm for x in vec]

        db = str(tmp_path / "faiss_persist")
        FAISSBackend(db_path=db).store("k1", "q", "FAISS response", vec)

        b2 = FAISSBackend(db_path=db)
        result, _ = b2.search(vec, threshold=0.85)
        assert result == "FAISS response"


# ── Qdrant ────────────────────────────────────────────────────

class TestQdrantBackend:

    @skip_qdrant
    def test_contract(self, tmp_path):
        from sulci.backends.qdrant import QdrantBackend
        backend = QdrantBackend(db_path=str(tmp_path / "qdrant_test"))
        _run_backend_contract(backend)


# ── Redis ─────────────────────────────────────────────────────

class TestRedisBackend:

    @skip_redis
    def test_contract_local(self):
        """Requires local Redis on localhost:6379."""
        from sulci.backends.redis import RedisBackend
        try:
            backend = RedisBackend(url="redis://localhost:6379", key_prefix=_TEST_KEY_PREFIX)
            backend._redis.ping()
        except Exception:
            pytest.skip("No local Redis instance available")
        _run_backend_contract(backend)


# ── Milvus ────────────────────────────────────────────────────

class TestMilvusBackend:

    @skip_milvus
    def test_contract(self, tmp_path):
        from sulci.backends.milvus import MilvusBackend
        backend = MilvusBackend(db_path=str(tmp_path / "milvus_test.db"))
        _run_backend_contract(backend)

    # Values a caller might pass as user_id. Each must match only the entry
    # stored under exactly that value.
    SCOPE_VALUES = [
        "alice", 'x" or user_id != "x', 'a\\" or user_id != "', "a\\",
        "o'neil", "line\nbreak", "cr\rx", "tab\tx", "nul\x00x", "uni \u00fc", "",
    ]

    def test_filter_literal_is_a_single_quoted_string(self):
        """Runs without pymilvus: the expression must not depend on the value."""
        from sulci.backends.milvus import _filter_literal
        for value in self.SCOPE_VALUES:
            lit = _filter_literal(value)
            assert lit[0] == lit[-1] == '"'
            body = lit[1:-1]
            # Every quote and backslash in the body is escaped, so the
            # literal cannot close early and nothing after it is parsed.
            i = 0
            while i < len(body):
                if body[i] == "\\":
                    assert body[i + 1] in '\\"nrt'
                    i += 2
                    continue
                assert body[i] not in '"\n\r'
                i += 1

    @skip_milvus
    def test_user_scope_matches_exactly(self, tmp_path):
        from sulci.backends.milvus import MilvusBackend
        b = MilvusBackend(db_path=str(tmp_path / "milvus_scope.db"))
        vec = [1.0, 0.0]
        for i, value in enumerate(self.SCOPE_VALUES[:-1]):
            b.store(f"k{i}", "q", f"resp-{i}", vec, user_id=value)

        for i, value in enumerate(self.SCOPE_VALUES[:-1]):
            assert b.search(vec, 0.85, user_id=value)[0] == f"resp-{i}", value
        assert b.search(vec, 0.85, user_id="nobody")[0] is None
