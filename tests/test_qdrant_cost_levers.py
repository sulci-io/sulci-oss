# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.

"""
tests/test_qdrant_cost_levers.py
=================================
`on_disk` and `quantization` on `QdrantBackend.__init__`.

Both are cost levers. sulci-platform's `docs/marketing/pricing/COGS.md` §4
prices them and finds them interchangeable at 4M vectors and compounding when
stacked. Neither was reachable from this constructor before v0.9.1, so the
cost model described a configuration nobody using this class could produce.

⛔ **The defect these tests exist to prevent is a SILENT one.** Three of the
four ways to get this wrong produce no error at all:

  - a typo'd shorthand falling through to "no quantization"
  - passing either kwarg against a collection that already exists, where
    `create_collection` is a no-op
  - a caller who passes nothing having their existing collection reconfigured
    by an upgrade

In every case the process runs, the cache works, and the cluster is not what
the operator believes. So each test asserts the STORED collection config read
back from the client, not the constructor's arguments — asserting the input
would pass against a constructor that dropped them on the floor.

📌 **On the embedded-Qdrant lock:** `QdrantClient(path=...)` takes an
exclusive lock on the storage folder, so a test that reopens the same path
must close the first client. That is a property of the harness, not of the
code under test; `_close()` below exists for that and nothing else.
"""
from __future__ import annotations
import math
import warnings

import pytest

qdrant_client = pytest.importorskip("qdrant_client")
from sulci.backends.qdrant import QdrantBackend  # noqa: E402


def _close(backend: QdrantBackend) -> None:
    """Release the embedded storage lock so the same path can be reopened."""
    backend._client.close()


def _vectors_config(backend: QdrantBackend):
    """The config as the SERVER holds it — the only thing worth asserting."""
    return backend._client.get_collection(
        QdrantBackend.COLLECTION
    ).config.params.vectors


def _normalized(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


ANSWER_VEC = _normalized([1.0, 0.2, 0.0] + [0.0] * 381)


# -----------------------------------------------------------------------------
# Defaults — an upgrade must not reconfigure anybody's cluster.
# -----------------------------------------------------------------------------

class TestDefaultsAreInert:
    def test_omitting_both_kwargs_leaves_qdrant_defaults_in_force(self, tmp_path):
        """The upgrade path. A caller who passes nothing gets what they had.

        This is the test that makes the release safe to ship: `on_disk` and
        `quantization` default to None, nothing is passed to VectorParams, and
        Qdrant's own defaults decide. If this ever fails, upgrading the library
        silently changes the memory profile of every existing deployment.
        """
        backend = QdrantBackend(db_path=str(tmp_path / "q"))
        params = _vectors_config(backend)
        assert params.on_disk is None
        assert params.quantization_config is None


# -----------------------------------------------------------------------------
# The levers reach the collection.
# -----------------------------------------------------------------------------

class TestLeversReachTheCollection:
    def test_on_disk_true_is_stored_on_the_collection(self, tmp_path):
        backend = QdrantBackend(db_path=str(tmp_path / "q"), on_disk=True)
        assert _vectors_config(backend).on_disk is True

    def test_on_disk_false_is_stored_and_is_not_the_same_as_omitting_it(self, tmp_path):
        """`False` and `None` are different requests and must not collapse.

        A caller explicitly pinning vectors to RAM is making a statement; a
        caller who says nothing is not. Storing False for both would make the
        explicit case unassertable.
        """
        backend = QdrantBackend(db_path=str(tmp_path / "q"), on_disk=False)
        assert _vectors_config(backend).on_disk is False

    def test_scalar_shorthand_produces_int8_scalar_quantization(self, tmp_path):
        from qdrant_client.models import ScalarType

        backend = QdrantBackend(db_path=str(tmp_path / "q"), quantization="scalar")
        stored = _vectors_config(backend).quantization_config
        assert stored is not None
        assert stored.scalar.type == ScalarType.INT8

    def test_binary_shorthand_produces_binary_quantization(self, tmp_path):
        backend = QdrantBackend(db_path=str(tmp_path / "q"), quantization="binary")
        stored = _vectors_config(backend).quantization_config
        assert stored is not None
        assert getattr(stored, "binary", None) is not None

    def test_a_raw_quantization_model_is_forwarded_untouched(self, tmp_path):
        """The escape hatch. `product` has no shorthand on purpose — it needs a
        compression ratio this class has no defensible default for — so the
        model must pass straight through or product quantization is
        unreachable, which is the state this change exists to end.
        """
        from qdrant_client.models import (
            CompressionRatio, ProductQuantization, ProductQuantizationConfig,
        )

        backend = QdrantBackend(
            db_path=str(tmp_path / "q"),
            quantization=ProductQuantization(
                product=ProductQuantizationConfig(compression=CompressionRatio.X16)
            ),
        )
        stored = _vectors_config(backend).quantization_config
        assert stored is not None
        assert getattr(stored, "product", None) is not None

    def test_both_levers_stack(self, tmp_path):
        """COGS §4's cheapest cell is both levers on, not either alone."""
        backend = QdrantBackend(
            db_path=str(tmp_path / "q"), on_disk=True, quantization="scalar"
        )
        params = _vectors_config(backend)
        assert params.on_disk is True
        assert params.quantization_config is not None


# -----------------------------------------------------------------------------
# The cache still works with the levers on.
# -----------------------------------------------------------------------------

class TestStoreAndSearchSurviveTheLevers:
    def test_round_trip_with_both_levers_on(self, tmp_path):
        """Configuring for cost must not break the thing being configured.

        Quantization is lossy by construction, so this asserts a hit above
        threshold on a self-similar query rather than an exact score.
        """
        backend = QdrantBackend(
            db_path=str(tmp_path / "q"), on_disk=True, quantization="scalar"
        )
        backend.store("k1", "how do I reset my password", "Use the reset link.", ANSWER_VEC)

        response, score = backend.search(ANSWER_VEC, threshold=0.9)
        assert response == "Use the reset link."
        assert score >= 0.9


# -----------------------------------------------------------------------------
# A typo must refuse. Silence here is indistinguishable from success.
# -----------------------------------------------------------------------------

class TestUnrecognisedShorthandRefuses:
    @pytest.mark.parametrize("bad", ["scaler", "int8", "SCALAR_", "none", ""])
    def test_unknown_string_raises_rather_than_disabling_quantization(self, tmp_path, bad):
        """`quantization="scaler"` must not mean `quantization=None`.

        A shorthand that falls through to "off" produces a cluster with the
        lever disabled, a green test run, and a cost line nobody can explain.
        The failure has to happen at the call, where the typo is.
        """
        with pytest.raises(ValueError) as excinfo:
            QdrantBackend(db_path=str(tmp_path / "q"), quantization=bad)
        assert repr(bad) in str(excinfo.value)

    def test_shorthands_are_case_and_whitespace_insensitive(self, tmp_path):
        backend = QdrantBackend(db_path=str(tmp_path / "q"), quantization="  Scalar ")
        assert _vectors_config(backend).quantization_config is not None


# -----------------------------------------------------------------------------
# The no-op. This is the fail-open the docstring warns about.
# -----------------------------------------------------------------------------

class TestExistingCollectionIsNotSilentlyIgnored:
    def test_requesting_new_config_against_an_existing_collection_warns(self, tmp_path):
        """⛔ The whole point of this class of test.

        `create_collection` is skipped when the collection exists, so the
        kwargs do nothing. An operator who sets `on_disk=True`, sees no error
        and a working cache will believe the cluster is offloaded. It is not.
        The warning is the only thing standing between them and that belief.
        """
        path = str(tmp_path / "q")
        first = QdrantBackend(db_path=path)
        _close(first)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            second = QdrantBackend(db_path=path, on_disk=True, quantization="scalar")

        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        message = str(caught[0].message)
        # Both the live value and the requested one, so the reader can see the
        # gap without going to look it up.
        assert "on_disk" in message and "requested=True" in message
        assert "quantization" in message
        assert "update_collection" in message or "rebuild" in message

        # And the collection really is unchanged — the warning is honest.
        assert _vectors_config(second).on_disk is None

    def test_matching_config_against_an_existing_collection_is_silent(self, tmp_path):
        """No warning when there is nothing to warn about.

        A warning that fires on every restart is a warning people filter, and
        then it is not there on the day it matters.
        """
        path = str(tmp_path / "q")
        first = QdrantBackend(db_path=path, on_disk=True, quantization="scalar")
        _close(first)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            QdrantBackend(db_path=path, on_disk=True, quantization="scalar")

        assert caught == []

    def test_passing_no_kwargs_against_an_existing_collection_is_silent(self, tmp_path):
        """The overwhelmingly common path — a process restart — stays quiet."""
        path = str(tmp_path / "q")
        first = QdrantBackend(db_path=path)
        _close(first)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            QdrantBackend(db_path=path)

        assert caught == []

    def test_partial_drift_is_reported_for_the_field_that_drifted(self, tmp_path):
        """Only `on_disk` differs. The message should say so and not blame
        quantization, which matches."""
        path = str(tmp_path / "q")
        first = QdrantBackend(db_path=path, quantization="scalar")
        _close(first)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            QdrantBackend(db_path=path, on_disk=True, quantization="scalar")

        assert len(caught) == 1
        message = str(caught[0].message)
        assert "on_disk" in message
        assert "quantization" not in message
