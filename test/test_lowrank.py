"""Test for low rank sinkhorn solvers"""

# Author: Laurène DAVID <laurene.david@ip-paris.fr>
#
# License: MIT License

import ot
import numpy as np
import pytest
from ot.lowrank import sklearn_import  # check sklearn installation


def test_compute_lr_sqeuclidean_matrix():
    # test computation of low rank cost matrices M1 and M2
    n = 100
    X_s = np.reshape(1.0 * np.arange(2 * n), (n, 2))
    X_t = np.reshape(1.0 * np.arange(2 * n), (n, 2))

    M1, M2 = ot.lowrank.compute_lr_sqeuclidean_matrix(X_s, X_t, rescale_cost=False)
    M = ot.dist(X_s, X_t, metric="sqeuclidean")  # original cost matrix

    np.testing.assert_allclose(np.dot(M1, M2.T), M, atol=1e-05)


def test_lowrank_sinkhorn():
    # test low rank sinkhorn
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(n), (n, 1))

    Q, R, g, log = ot.lowrank.lowrank_sinkhorn(
        X_s, X_t, a, b, reg=0.1, log=True, rescale_cost=False
    )
    P = log["lazy_plan"][:]
    value_linear = log["value_linear"]

    # check constraints for P
    np.testing.assert_allclose(a, P.sum(1), atol=1e-05)
    np.testing.assert_allclose(b, P.sum(0), atol=1e-05)

    # check if lazy_plan is equal to the fully computed plan
    P_true = np.dot(Q, np.dot(np.diag(1 / g), R.T))
    np.testing.assert_allclose(P, P_true, atol=1e-05)

    # check if value_linear is correct with its original formula
    M = ot.dist(X_s, X_t, metric="sqeuclidean")
    value_linear_true = np.sum(M * P_true)
    np.testing.assert_allclose(value_linear, value_linear_true, atol=1e-05)

    # check warn parameter when Dykstra algorithm doesn't converge
    with pytest.warns(UserWarning):
        ot.lowrank.lowrank_sinkhorn(X_s, X_t, a, b, reg=0.1, stopThr=0, numItermax=1)


@pytest.mark.parametrize(("init"), ("random", "deterministic", "kmeans"))
def test_lowrank_sinkhorn_init(init):
    # test lowrank inits
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(n), (n, 1))

    # test ImportError if init="kmeans" and sklearn not imported
    if init in ["random", "deterministic"] or (
        (init == "kmeans") and (sklearn_import is True)
    ):
        Q, R, g, log = ot.lowrank.lowrank_sinkhorn(
            X_s, X_t, a, b, reg=0.1, init=init, log=True
        )
        P = log["lazy_plan"][:]

        # check constraints for P
        np.testing.assert_allclose(a, P.sum(1), atol=1e-05)
        np.testing.assert_allclose(b, P.sum(0), atol=1e-05)

    else:
        with pytest.raises(ImportError):
            Q, R, g = ot.lowrank.lowrank_sinkhorn(X_s, X_t, a, b, reg=0.1, init=init)


@pytest.mark.parametrize(("alpha, rank"), ((0.8, 2), (0.5, 3), (0.2, 6)))
def test_lowrank_sinkhorn_alpha_error(alpha, rank):
    # Test warning for value of alpha
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(0, n), (n, 1))

    with pytest.raises(ValueError):
        ot.lowrank.lowrank_sinkhorn(
            X_s, X_t, a, b, reg=0.1, rank=rank, alpha=alpha, warn=False
        )


@pytest.mark.parametrize(("gamma_init"), ("rescale", "theory"))
def test_lowrank_sinkhorn_gamma_init(gamma_init):
    # Test lr sinkhorn with different init strategies
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(n), (n, 1))

    Q, R, g, log = ot.lowrank.lowrank_sinkhorn(
        X_s, X_t, a, b, reg=0.1, gamma_init=gamma_init, log=True
    )
    P = log["lazy_plan"][:]

    # check constraints for P
    np.testing.assert_allclose(a, P.sum(1), atol=1e-05)
    np.testing.assert_allclose(b, P.sum(0), atol=1e-05)


@pytest.skip_backend("tf")
def test_lowrank_sinkhorn_backends(nx):
    # Test low rank sinkhorn for different backends
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(0, n), (n, 1))

    ab, bb, X_sb, X_tb = nx.from_numpy(a, b, X_s, X_t)

    Q, R, g, log = ot.lowrank.lowrank_sinkhorn(X_sb, X_tb, ab, bb, reg=0.1, log=True)
    lazy_plan = log["lazy_plan"]
    P = lazy_plan[:]

    np.testing.assert_allclose(ab, P.sum(1), atol=1e-05)
    np.testing.assert_allclose(bb, P.sum(0), atol=1e-05)
