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
            backend = RedisBackend(url="redis://localhost:6379")
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
