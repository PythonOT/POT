"""Tests for gromov._lowrank.py"""

# Author: Laurène DAVID <laurene.david@ip-paris.fr>
#
# License: MIT License

import ot
import numpy as np
import pytest


def test__flat_product_operator():
    # test flat product operator
    n, d = 100, 2
    X = np.reshape(1.0 * np.arange(2 * n), (n, d))
    A1, A2 = ot.lowrank.compute_lr_sqeuclidean_matrix(X, X, rescale_cost=False)

    A1_ = ot.gromov._flat_product_operator(A1)
    A2_ = ot.gromov._flat_product_operator(A2)
    cost = ot.dist(X, X)

    # test value
    np.testing.assert_allclose(cost**2, np.dot(A1_, A2_.T), atol=1e-05)


def test_lowrank_gromov_wasserstein_samples():
    # test low rank gromov wasserstein
    n_samples = 20  # nb samples
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    X_s = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=1)
    X_t = X_s[::-1].copy()

    a = ot.unif(n_samples)
    b = ot.unif(n_samples)

    Q, R, g, log = ot.gromov.lowrank_gromov_wasserstein_samples(
        X_s, X_t, a, b, reg=0.1, log=True, rescale_cost=False
    )
    P = log["lazy_plan"][:]

    # check constraints for P
    np.testing.assert_allclose(a, P.sum(1), atol=1e-04)
    np.testing.assert_allclose(b, P.sum(0), atol=1e-04)

    # check if lazy_plan is equal to the fully computed plan
    P_true = np.dot(Q, np.dot(np.diag(1 / g), R.T))
    np.testing.assert_allclose(P, P_true, atol=1e-05)

    # check warn parameter when low rank GW algorithm doesn't converge
    with pytest.warns(UserWarning):
        ot.gromov.lowrank_gromov_wasserstein_samples(
            X_s,
            X_t,
            a,
            b,
            reg=0.1,
            stopThr=0,
            numItermax=1,
            warn=True,
            warn_dykstra=False,
        )

    # check warn parameter when Dykstra algorithm doesn't converge
    with pytest.warns(UserWarning):
        ot.gromov.lowrank_gromov_wasserstein_samples(
            X_s,
            X_t,
            a,
            b,
            reg=0.1,
            stopThr_dykstra=0,
            numItermax_dykstra=1,
            warn=False,
            warn_dykstra=True,
        )


@pytest.mark.parametrize(("alpha, rank"), ((0.8, 2), (0.5, 3), (0.2, 6), (0.1, -1)))
def test_lowrank_gromov_wasserstein_samples_alpha_error(alpha, rank):
    # Test warning for value of alpha and rank
    n_samples = 20  # nb samples
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    X_s = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=1)
    X_t = X_s[::-1].copy()

    a = ot.unif(n_samples)
    b = ot.unif(n_samples)

    with pytest.raises(ValueError):
        ot.gromov.lowrank_gromov_wasserstein_samples(
            X_s, X_t, a, b, reg=0.1, rank=rank, alpha=alpha, warn=False
        )


@pytest.mark.parametrize(("gamma_init"), ("rescale", "theory", "other"))
def test_lowrank_wasserstein_samples_gamma_init(gamma_init):
    # Test lr sinkhorn with different init strategies
    n_samples = 20  # nb samples
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    X_s = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=1)
    X_t = X_s[::-1].copy()

    a = ot.unif(n_samples)
    b = ot.unif(n_samples)

    if gamma_init not in ["rescale", "theory"]:
        with pytest.raises(NotImplementedError):
            ot.gromov.lowrank_gromov_wasserstein_samples(
                X_s, X_t, a, b, reg=0.1, gamma_init=gamma_init, log=True
            )

    else:
        Q, R, g, log = ot.gromov.lowrank_gromov_wasserstein_samples(
            X_s, X_t, a, b, reg=0.1, gamma_init=gamma_init, log=True
        )
        P = log["lazy_plan"][:]

        # check constraints for P
        np.testing.assert_allclose(a, P.sum(1), atol=1e-04)
        np.testing.assert_allclose(b, P.sum(0), atol=1e-04)


@pytest.skip_backend("tf")
def test_lowrank_gromov_wasserstein_samples_backends(nx):
    # Test low rank sinkhorn for different backends
    n_samples = 20  # nb samples
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    X_s = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=1)
    X_t = X_s[::-1].copy()

    a = ot.unif(n_samples)
    b = ot.unif(n_samples)

    ab, bb, X_sb, X_tb = nx.from_numpy(a, b, X_s, X_t)

    Q, R, g, log = ot.gromov.lowrank_gromov_wasserstein_samples(
        X_sb, X_tb, ab, bb, reg=0.1, log=True
    )
    lazy_plan = log["lazy_plan"]
    P = lazy_plan[:]

    np.testing.assert_allclose(ab, P.sum(1), atol=1e-04)
    np.testing.assert_allclose(bb, P.sum(0), atol=1e-04)
