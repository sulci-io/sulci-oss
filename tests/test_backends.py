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
# 0.10.0: and milvus-lite — the tests open a local file, which pymilvus alone cannot (not installed on Windows).
# By find_spec, not import: an installed milvus-lite that fails to import (2.5.1 without pkg_resources) must FAIL
# the tests, not skip them.
def _installed(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


skip_milvus  = pytest.mark.skipif(not (_installed("pymilvus") and _installed("milvus_lite")),
                                  reason="pymilvus or milvus-lite not installed")

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
        replace the first tenant's row, asserted on the stored rows. (Lookups
        are scoped by tenant and user since 0.9.3: tests/test_backend_isolation.py.)
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
        assert b.search([1.0, 0.0], 0.85, tenant_id="t", user_id="u")[0] == "new"   # 0.9.3: scope = tenant AND user


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


class TestFAISSThreads:
    """0.10.0 (oss-0022): calls from several threads run one at a time. Measured on main before it: 8 threads calling
    store() aborted the process in 5 of 5 runs, and 400 concurrent AsyncCache.aset calls aborted or raised in 4 of 5.
    test_every_call_holds_the_lock is the guard for search, clear and the file writes: the concurrency test alone
    catches an unlocked store, not those (measured with mutants)."""

    @staticmethod
    def _vec(i):
        import random
        r = random.Random(i)
        v = [r.gauss(0, 1) for _ in range(64)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    @skip_faiss
    def test_every_call_holds_the_lock(self, tmp_path):
        from sulci.backends.faiss import FAISSBackend
        b = FAISSBackend(db_path=str(tmp_path / "f"))
        b.store("k0", "q", "r0", self._vec(0), user_id="u")
        real, held = b._index, []

        class Index:                                 # the index, recording whether each use held the lock
            def __getattr__(self, name):
                held.append((name, b._lock._is_owned()))
                return getattr(real, name)
        b._index = Index()
        real_idx_path = b._idx_path

        def idx_path():
            held.append(("files", b._lock._is_owned()))
            return real_idx_path()
        b._idx_path = idx_path
        import types, faiss
        b._faiss = types.SimpleNamespace(write_index=lambda index, path: faiss.write_index(real, path))
        b.store("k1", "q", "r1", self._vec(1), user_id="u")
        assert b.search(self._vec(1), 0.9, user_id="u")[0] == "r1"
        b.clear()
        assert {n for n, _ in held} >= {"add", "reconstruct_batch", "ntotal", "files"}
        assert all(owned for _, owned in held), [n for n, owned in held if not owned]

    @skip_faiss
    def test_concurrent_stores_and_lookups_keep_entries_and_vectors_together(self, tmp_path):
        import threading
        from sulci.backends.faiss import FAISSBackend
        b = FAISSBackend(db_path=str(tmp_path / "f"))
        errors = []

        def worker(k):
            try:
                for i in range(k, 160, 8):
                    b.store(f"k{i}", f"q{i}", f"r{i}", self._vec(i), user_id=f"u{i % 3}")
                    b.search(self._vec(i), 0.99, user_id=f"u{i % 3}")
            except Exception as e:                   # surfaced below, not lost in the thread
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert b._index.ntotal == len(b._meta) == 160
        for pos, m in enumerate(b._meta):            # each position's vector is its own entry's
            i = int(m["response"][1:])
            assert max(abs(x - y) for x, y in zip(b._index.reconstruct(pos), self._vec(i))) < 1e-6
        assert [b.search(self._vec(i), 0.99, user_id=f"u{i % 3}")[0] for i in range(160)] == \
            [f"r{i}" for i in range(160)]
        b2 = FAISSBackend(db_path=str(tmp_path / "f"))   # and the files agree
        assert b2._index.ntotal == len(b2._meta) == 160


# ── Qdrant ────────────────────────────────────────────────────

class TestQdrantBackend:

    @skip_qdrant
    def test_contract(self, tmp_path):
        from sulci.backends.qdrant import QdrantBackend
        backend = QdrantBackend(db_path=str(tmp_path / "qdrant_test"))
        _run_backend_contract(backend)


class TestQdrantThreads:
    """0.10.0 (oss-0023): in local mode, calls from several threads reach qdrant-client one at a time. Measured on main
    before it: 400 concurrent AsyncCache.aset calls raised sqlite3.InterfaceError inside qdrant-client in 5 of 5 runs.
    test_every_client_call_holds_the_lock is the guard for each method; the concurrency test shows the failure."""

    @staticmethod
    def _vec(i):
        import random
        r = random.Random(i)
        v = [r.gauss(0, 1) for _ in range(64)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    @skip_qdrant
    def test_every_client_call_holds_the_lock(self, tmp_path):
        from sulci.backends.qdrant import QdrantBackend
        b = QdrantBackend(db_path=str(tmp_path / "q"), dimension=64)
        real, held = b._client, []

        class Client:                                # the client, recording whether each call held the lock
            def __getattr__(self, name):
                held.append((name, b._lock._is_owned()))
                return getattr(real, name)
        b._client = Client()
        b.store("k1", "q", "r1", self._vec(1), tenant_id="t", user_id="u")
        assert b.search(self._vec(1), 0.9, tenant_id="t", user_id="u")[0] == "r1"
        assert b.delete_user("u", tenant_id="t") == 1
        b.clear()
        assert {n for n, _ in held} >= {"upsert", "query_points", "count", "delete"}
        assert all(owned for _, owned in held), [n for n, owned in held if not owned]

    @skip_qdrant
    def test_concurrent_stores_and_lookups(self, tmp_path):
        import threading
        from sulci.backends.qdrant import QdrantBackend
        b = QdrantBackend(db_path=str(tmp_path / "q"), dimension=64)
        errors = []

        def worker(k):
            try:
                for i in range(k, 160, 8):
                    b.store(f"k{i}", f"q{i}", f"r{i}", self._vec(i), user_id=f"u{i % 3}")
                    b.search(self._vec(i), 0.99, user_id=f"u{i % 3}")
            except Exception as e:                   # surfaced below, not lost in the thread
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert [b.search(self._vec(i), 0.99, user_id=f"u{i % 3}")[0] for i in range(160)] == \
            [f"r{i}" for i in range(160)]

    @skip_qdrant
    def test_a_server_is_not_locked(self, monkeypatch):
        import contextlib
        import qdrant_client
        from sulci.backends.qdrant import QdrantBackend

        class Server:                                # a client for url=, without a server behind it
            def __init__(self, **kw):
                pass

            def get_collections(self):
                return type("C", (), {"collections": [type("N", (), {"name": "sulci"})()]})()
        monkeypatch.setattr(qdrant_client, "QdrantClient", Server)
        b = QdrantBackend(url="http://qdrant.invalid:6333")
        assert isinstance(b._lock, contextlib.nullcontext)


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

    def test_no_caller_text_reaches_the_filter(self):
        """Runs without pymilvus. 0.10.0: the filter names a scope key of digests, whatever the ids hold."""
        import re
        from sulci.backends.milvus import _scope_key
        keys = set()
        for tenant in (None, "t1", 't" or "1"=="1'):
            for value in self.SCOPE_VALUES:
                key = _scope_key(tenant, value)
                assert re.fullmatch(r"t(-|[0-9a-f]{32}):u(-|[0-9a-f]{32}):", key), (tenant, value)
                keys.add(key)
        # one key per (tenant, value): the ten ids, and "" naming the shared user
        assert len(keys) == 3 * len(self.SCOPE_VALUES)
        assert _scope_key(None, None) == _scope_key("", "")

    def test_global_is_an_id_not_the_shared_scope(self):
        """Runs without pymilvus. Cache refuses "global"; a direct caller of the backend gets a scope of its own."""
        from sulci.backends.milvus import _scope_key
        assert _scope_key(None, "global") != _scope_key(None, None)
        assert _scope_key("global", None) != _scope_key(None, None)

    @skip_milvus
    def test_a_fixed_schema_without_a_scope_field_is_refused_clearly(self, tmp_path):
        import pytest
        from pymilvus import MilvusClient, DataType
        from sulci.backends.milvus import MilvusBackend
        b = MilvusBackend(db_path=str(tmp_path / "milvus_fixed.db"))
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
        index = b._client.prepare_index_params()
        index.add_index("vector", index_type="FLAT", metric_type="COSINE")
        b._client.create_collection(b.COLLECTION, schema=schema, index_params=index)
        with pytest.raises(RuntimeError, match="fixed schema without a 'scope' field"):
            b.store("k", "q", "r", [1.0, 0.0])

    @staticmethod
    def _flaky_client(monkeypatch, failures):
        """A MilvusClient stand-in that fails its first `failures` constructions like a server that did not come
        up, and records every milvus-lite release."""
        from pymilvus.exceptions import MilvusException
        from sulci.backends import milvus as mb
        calls = {"made": 0, "released": []}

        class _Client:
            def __init__(self, *a, **k):
                calls["made"] += 1
                if calls["made"] <= failures:
                    raise MilvusException(code=2, message="Fail connecting to server on unix:/tmp/x.sock")

        monkeypatch.setattr("pymilvus.MilvusClient", _Client)
        monkeypatch.setattr(mb, "_release_lite_server", lambda path: calls["released"].append(path))
        monkeypatch.setattr(mb.time, "sleep", lambda s: None)
        return calls

    @skip_milvus
    def test_a_local_server_that_failed_to_start_is_retried(self, monkeypatch, tmp_path):
        """0.10.0: measured in CI — milvus-lite 2.5.1's first server on a cold macOS runner was gone when pymilvus
        connected, and the backend could not be constructed."""
        from sulci.backends.milvus import MilvusBackend
        calls = self._flaky_client(monkeypatch, failures=2)
        db = str(tmp_path / "m.db")
        MilvusBackend(db_path=db)
        assert calls["made"] == 3 and calls["released"] == [db, db]

    @skip_milvus
    def test_a_local_server_that_never_starts_raises_its_error(self, monkeypatch, tmp_path):
        import pytest
        from pymilvus.exceptions import MilvusException
        from sulci.backends.milvus import MilvusBackend, _LITE_START_ATTEMPTS
        calls = self._flaky_client(monkeypatch, failures=99)
        with pytest.raises(MilvusException, match="Fail connecting"):
            MilvusBackend(db_path=str(tmp_path / "m.db"))
        assert calls["made"] == _LITE_START_ATTEMPTS

    @skip_milvus
    def test_a_server_that_was_never_started_is_not_retried(self, monkeypatch, tmp_path):
        """A file held by another process: retrying on milvus-lite 2.5.1 started a second server on the same file."""
        import pytest
        from pymilvus.exceptions import ConnectionConfigException
        from sulci.backends import milvus as mb
        calls = {"made": 0, "released": []}

        class _Client:
            def __init__(self, *a, **k):
                calls["made"] += 1
                raise ConnectionConfigException(message="Open local milvus failed")

        monkeypatch.setattr("pymilvus.MilvusClient", _Client)
        monkeypatch.setattr(mb, "_release_lite_server", lambda path: calls["released"].append(path))
        with pytest.raises(ConnectionConfigException):
            mb.MilvusBackend(db_path=str(tmp_path / "m.db"))
        assert calls == {"made": 1, "released": []}

    @skip_milvus
    def test_a_milvus_server_uri_is_not_retried_here(self, monkeypatch):
        import pytest
        from pymilvus.exceptions import MilvusException
        from sulci.backends.milvus import MilvusBackend
        calls = self._flaky_client(monkeypatch, failures=99)
        with pytest.raises(MilvusException):
            MilvusBackend(uri="http://127.0.0.1:1")
        assert calls["made"] == 1 and calls["released"] == []

    @skip_milvus
    def test_a_new_backend_finds_what_an_earlier_one_stored(self, tmp_path):
        """0.10.0: a lookup opens the existing collection; through 0.9.3 it missed until this instance stored."""
        from sulci.backends.milvus import MilvusBackend
        db = str(tmp_path / "milvus_reopen.db")
        first = MilvusBackend(db_path=db)
        first.store("k", "q", "stored-earlier", [1.0, 0.0], tenant_id="t1", user_id="alice")
        first._client.close()
        again = MilvusBackend(db_path=db)
        assert again.search([1.0, 0.0], 0.85, tenant_id="t1", user_id="alice")[0] == "stored-earlier"

    @skip_milvus
    def test_a_collection_written_by_another_process_is_loaded(self, tmp_path):
        """0.10.0: a collection written by another process is found and searched. On milvus-lite 3.2.2 (which the
        extra now excludes) this raised "in state 'released'" until the backend loaded it; 2.5.1 searches an
        unloaded collection anyway, so here it guards the open path, and the recorded-client test the load."""
        import subprocess
        import sys
        from sulci.backends.milvus import MilvusBackend
        db = str(tmp_path / "milvus_other.db")
        writer = ("from sulci.backends.milvus import MilvusBackend\n"
                  f"MilvusBackend(db_path={db!r}).store('k', 'q', 'from-the-writer', [1.0, 0.0], tenant_id='t1')\n")
        subprocess.run([sys.executable, "-c", writer], check=True)
        b = MilvusBackend(db_path=db)
        b.store("k2", "q2", "from-the-reader", [0.0, 1.0], tenant_id="t1")
        assert b.search([1.0, 0.0], 0.85, tenant_id="t1")[0] == "from-the-writer"
        assert b.search([0.0, 1.0], 0.85, tenant_id="t1")[0] == "from-the-reader"

    @skip_milvus
    def test_an_entry_written_before_the_scope_key_is_a_miss(self, tmp_path):
        """0.10.0: a row without a "scope" field (as 0.9.3 wrote them) matches no lookup — not an error."""
        from sulci.backends.milvus import MilvusBackend
        b = MilvusBackend(db_path=str(tmp_path / "milvus_old.db"))
        vec = [1.0, 0.0]
        b.store("k", "q", "new", vec, tenant_id="t1", user_id="alice")
        b._client.insert(b.COLLECTION, [{"vector": vec, "query": "q", "response": "old",
                                         "tenant_id": "t1", "user_id": "bob", "expires": 0.0}])
        assert b.search(vec, 0.85, tenant_id="t1", user_id="bob")[0] is None
        assert b.search(vec, 0.85, tenant_id="t1", user_id="alice")[0] == "new"

    @skip_milvus
    def test_user_scope_matches_exactly(self, tmp_path):
        from sulci.backends.milvus import MilvusBackend
        b = MilvusBackend(db_path=str(tmp_path / "milvus_scope.db"))
        vec = [1.0, 0.0]
        for i, value in enumerate(self.SCOPE_VALUES[:-1]):
            b.store(f"k{i}", "q", f"resp-{i}", vec, user_id=value)

        for i, value in enumerate(self.SCOPE_VALUES[:-1]):
            assert b.search(vec, 0.85, user_id=value)[0] == f"resp-{i}", value
            # 0.10.0: and within its tenant — the same user id under another tenant is someone else
            assert b.search(vec, 0.85, tenant_id="t2", user_id=value)[0] is None, value
        assert b.search(vec, 0.85, user_id="nobody")[0] is None


# ── Milvus Lite: a stored entry is readable before store() returns (oss-0019) ──────────────────────────────────

class _LagClient:
    """Insert returns ids; get() answers nothing for the first `lag` reads of each id, as Milvus Lite did on a Mac."""
    def __init__(self, lag):
        self.lag, self.gets, self.next_id = lag, [], 0

    def insert(self, collection, rows):
        self.next_id += 1
        self._left = self.lag
        return {"insert_count": 1, "ids": [self.next_id]}

    def get(self, collection, ids, output_fields=None, timeout=None):
        assert timeout is not None and timeout > 0           # every read is bounded (oss-0019)
        self.gets.append(list(ids))
        if self._left > 0:
            self._left -= 1
            return []
        return [{"id": ids[0], "scope": "s"}]


def _lag_backend(lag, lite=True):
    from sulci.backends.milvus import MilvusBackend
    b = MilvusBackend.__new__(MilvusBackend)
    b._client, b._ready, b._lite = _LagClient(lag), True, lite
    b._ensure = lambda dim: None
    return b


class TestMilvusLiteReadsBack:
    """Measured on a maintainer's Mac (milvus-lite 2.5.1): a lookup straight after an insert missed the entry in 6 of
    50 trials and found it 0.5 s later; reading the row back by id before returning closed it (0 of 50)."""

    def test_store_returns_once_the_row_reads_back(self):
        b = _lag_backend(lag=2)
        b.store("k", "q", "a", [1.0, 0.0])
        assert b._client.gets == [[1], [1], [1]]               # two empty reads, then the row

    def test_a_row_visible_at_once_costs_one_read(self):
        b = _lag_backend(lag=0)
        b.store("k", "q", "a", [1.0, 0.0])
        assert b._client.gets == [[1]]

    def test_a_milvus_server_is_not_polled(self):
        b = _lag_backend(lag=2, lite=False)
        b.store("k", "q", "a", [1.0, 0.0])
        assert b._client.gets == []

    def test_a_row_that_never_reads_back_warns_once_and_returns_in_time(self, monkeypatch, caplog):
        import logging
        import sulci.backends.milvus as m
        monkeypatch.setattr(m, "_LITE_VISIBLE_TIMEOUT_SECONDS", 0.1)
        b = _lag_backend(lag=10 ** 6)
        t0 = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="sulci"):
            b.store("k", "q", "a", [1.0, 0.0])                  # returns: the entry WAS stored
        assert time.monotonic() - t0 < 0.1 + 0.2
        assert len(b._client.gets) >= 2
        warned = [r.getMessage() for r in caplog.records if r.name == "sulci"]
        assert len(warned) == 1 and "not readable after" in warned[0]

    def test_a_failing_read_back_warns_and_does_not_fail_the_store(self, caplog):
        import logging
        b = _lag_backend(lag=0)

        def broken(*a, **k):
            raise ConnectionError("lite went away")
        b._client.get = broken
        with caplog.at_level(logging.WARNING, logger="sulci"):
            b.store("k", "q", "a", [1.0, 0.0])
        warned = [r.getMessage() for r in caplog.records if r.name == "sulci"]
        assert len(warned) == 1 and "ConnectionError" in warned[0]

    def test_an_insert_without_ids_warns_once_and_is_not_polled(self, caplog):
        import logging
        b = _lag_backend(lag=2)
        b._client.insert = lambda collection, rows: {"insert_count": 1}
        with caplog.at_level(logging.WARNING, logger="sulci"):
            b.store("k", "q", "a", [1.0, 0.0])
            b.store("k2", "q2", "a2", [0.0, 1.0])
        assert b._client.gets == []
        assert len([r for r in caplog.records if "returned no ids" in r.getMessage()]) == 1

    @pytest.mark.parametrize("uri,lite", [(None, True), ("", True), ("./cache.db", True), ("/tmp/x.db", True),
                                          ("cache.db", True), ("http://localhost:19530", False),
                                          ("https://x.zillizcloud.com", False), ("tcp://h:19530", False),
                                          ("grpc://h:19530", False), ("unix:/tmp/m.sock", False),
                                          ("http://host/path.db", False), ("HTTP://host/x.db", False),
                                          ("http:x.db", False), ("tcp:x.db", False), ("unix:///tmp/m.db", False),
                                          ("x.DB", False), ("file:///tmp/x.db", True)])
    def test_which_uris_are_milvus_lite(self, uri, lite):
        from sulci.backends.milvus import _is_lite
        assert _is_lite(uri) is lite

    @skip_milvus
    def test_on_milvus_lite_every_store_reads_back_before_it_returns(self, tmp_path):
        from sulci.backends.milvus import MilvusBackend
        b = MilvusBackend(db_path=str(tmp_path / "lite.db"))
        assert b._lite is True
        real_get, reads = b._client.get, []

        def counted(*a, **k):
            rows = real_get(*a, **k)
            reads.append(bool(rows))
            return rows
        b._client.get = counted
        try:
            for i in range(20):
                v = [0.0] * 8
                v[i % 8] = 1.0
                b.store(f"k{i}", f"q{i}", f"a{i}", v, tenant_id="t", user_id=f"u{i}")
                assert reads and reads[-1] is True                  # it returned on a read that found the row
                assert b.search(v, 0.9, tenant_id="t", user_id=f"u{i}")[0] == f"a{i}"
        finally:
            b.clear()
