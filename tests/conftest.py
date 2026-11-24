"""Test configuration. All tests use synthetic inputs only; none reads files outside a temporary directory.

BLAS/OpenMP threads are limited to one before numpy is imported: on many-core machines multithreaded LAPACK makes
the small least-squares solves inside shrinkage LDA much slower. This affects speed only.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from eareeg_decoder import synthetic  # noqa: E402


@pytest.fixture(scope="session")
def small_cohort():
    """Six synthetic participants, 10 streams each (200 presentations), contingent blinks only."""
    es, truth = synthetic.synthetic_epochs(n_participants=6, n_streams=10, seed=7)
    return es, truth


@pytest.fixture
def rng():
    return np.random.default_rng(20261123)
