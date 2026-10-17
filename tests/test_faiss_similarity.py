"""0.10.1 (oss-0032, #172): FAISSBackend reports cosine similarity, as every other backend does.

Before 0.10.1 ``search`` reported ``1 - d`` where ``d`` is the squared L2 distance: for unit vectors that is
``2·cos − 1`` (measured on 0.10.0: cos 0.895 reported as 0.79, a miss at threshold 0.85 that SQLite serves), and a
vector not of unit length was scored by its length (``a`` against ``3a``, cos 1, reported as −3: never a hit).
"""
import math

import pytest

faiss = pytest.importorskip("faiss")

from sulci.backends.faiss import FAISSBackend           # noqa: E402
from sulci.backends.sqlite import SQLiteBackend         # noqa: E402


def _at(cos):
    th = math.acos(cos)
    return [1.0, 0.0, 0.0], [math.cos(th), math.sin(th), 0.0]


def _faiss(tmp_path, *vecs):
    b = FAISSBackend(db_path=str(tmp_path / "f"))
    for i, v in enumerate(vecs):
        b.store(f"k{i}", f"q{i}", f"r{i}", v)
    return b


@pytest.mark.parametrize("cos", [0.999, 0.95, 0.895, 0.85, 0.5, 0.0, -0.5])
def test_the_reported_similarity_is_the_cosine(tmp_path, cos):
    a, b = _at(cos)
    resp, sim = _faiss(tmp_path, a).search(b, threshold=-1.0)
    assert resp == "r0" and sim == pytest.approx(cos, abs=1e-6)


def test_a_match_at_the_threshold_is_a_hit_as_on_sqlite(tmp_path):
    """The case #172 named: cos 0.895 at threshold 0.85 — SQLite serves it; FAISS used to miss it."""
    a, b = _at(0.895)
    f = _faiss(tmp_path, a)
    s = SQLiteBackend(db_path=str(tmp_path / "s"))
    s.store("k0", "q0", "r0", a)
    assert f.search(b, threshold=0.85) == pytest.approx(s.search(b, threshold=0.85))
    assert f.search(b, threshold=0.85)[0] == "r0"
    assert f.search(b, threshold=0.90) == (None, 0.0)


def test_length_does_not_change_the_score(tmp_path):
    assert _faiss(tmp_path, [1.0, 0.0, 0.0]).search([3.0, 0.0, 0.0], threshold=0.99)[1] == pytest.approx(1.0)
    assert _faiss(tmp_path / "2", [0.0, 2.0, 0.0]).search([0.0, 0.5, 0.0], threshold=0.99)[1] == pytest.approx(1.0)


def test_an_identical_vector_meets_threshold_one(tmp_path):
    v = [0.1, 0.2, 0.3, 0.4]
    assert _faiss(tmp_path, v).search(v, threshold=1.0 - 1e-9)[0] == "r0"


def test_a_zero_vector_scores_zero_never_nan(tmp_path):
    resp, sim = _faiss(tmp_path, [0.0, 0.0, 0.0]).search([2.0, 0.0, 0.0], threshold=-1.0)
    assert resp == "r0" and sim == 0.0          # 1 - |d|² would be -3 here (0.10.1's review: [1,0,0] hid it)


def test_an_inf_query_is_a_miss_without_a_numpy_warning(tmp_path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _faiss(tmp_path, [1.0, 0.0, 0.0]).search([float("inf"), 0.0, 0.0], threshold=0.5) == (None, 0.0)


def test_only_the_best_five_are_rescored_and_the_best_wins_among_many(tmp_path):
    import random
    rnd = random.Random(7)
    vecs = [[rnd.uniform(-1, 1) for _ in range(16)] for _ in range(300)]
    b = _faiss(tmp_path, *vecs)
    q = vecs[123]
    assert b.search(q, threshold=0.999)[0] == "r123"


def test_the_most_similar_entry_wins(tmp_path):
    a, near = _at(0.97)
    _, far = _at(0.9)
    b = _faiss(tmp_path, far, near)
    assert b.search(a, threshold=0.85) == ("r1", pytest.approx(0.97, abs=1e-6))
