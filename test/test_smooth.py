"""Tests for ot.smooth model"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import ot
import pytest
from scipy.optimize import check_grad


def test_smooth_ot_dual():
    # get data
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    with pytest.raises(NotImplementedError):
        Gl2, log = ot.smooth.smooth_ot_dual(u, u, M, 1, reg_type="none")

    # squared l2 regularisation
    Gl2, log = ot.smooth.smooth_ot_dual(
        u, u, M, 1, reg_type="l2", log=True, stopThr=1e-10
    )

    # check constraints
    np.testing.assert_allclose(u, Gl2.sum(1), atol=1e-05)  # cf convergence sinkhorn
    np.testing.assert_allclose(u, Gl2.sum(0), atol=1e-05)  # cf convergence sinkhorn

    # kl regularisation
    G = ot.smooth.smooth_ot_dual(u, u, M, 1, reg_type="kl", stopThr=1e-10)

    # check constraints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn

    G2 = ot.sinkhorn(u, u, M, 1, stopThr=1e-10)
    np.testing.assert_allclose(G, G2, atol=1e-05)

    # sparsity-constrained regularisation
    max_nz = 2
    Gsc, log = ot.smooth.smooth_ot_dual(
        u,
        u,
        M,
        1,
        max_nz=max_nz,
        log=True,
        reg_type="sparsity_constrained",
        stopThr=1e-10,
    )

    # check marginal constraints
    np.testing.assert_allclose(u, Gsc.sum(1), atol=1e-03)
    np.testing.assert_allclose(u, Gsc.sum(0), atol=1e-03)

    # check sparsity constraints
    np.testing.assert_array_less(np.sum(Gsc > 0, axis=0), np.ones(n) * max_nz + 1)


def test_smooth_ot_semi_dual():
    # get data
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    with pytest.raises(NotImplementedError):
        Gl2, log = ot.smooth.smooth_ot_semi_dual(u, u, M, 1, reg_type="none")

    # squared l2 regularisation
    Gl2, log = ot.smooth.smooth_ot_semi_dual(
        u, u, M, 1, reg_type="l2", log=True, stopThr=1e-10
    )

    # check constraints
    np.testing.assert_allclose(u, Gl2.sum(1), atol=1e-05)  # cf convergence sinkhorn
    np.testing.assert_allclose(u, Gl2.sum(0), atol=1e-05)  # cf convergence sinkhorn

    # kl regularisation
    G = ot.smooth.smooth_ot_semi_dual(u, u, M, 1, reg_type="kl", stopThr=1e-10)

    # check constraints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn

    G2 = ot.sinkhorn(u, u, M, 1, stopThr=1e-10)
    np.testing.assert_allclose(G, G2, atol=1e-05)

    # sparsity-constrained regularisation
    max_nz = 2
    Gsc = ot.smooth.smooth_ot_semi_dual(
        u, u, M, 1, reg_type="sparsity_constrained", max_nz=max_nz, stopThr=1e-10
    )

    # check marginal constraints
    np.testing.assert_allclose(u, Gsc.sum(1), atol=1e-03)
    np.testing.assert_allclose(u, Gsc.sum(0), atol=1e-03)

    # check sparsity constraints
    np.testing.assert_array_less(np.sum(Gsc > 0, axis=0), np.ones(n) * max_nz + 1)


def test_sparsity_constrained_gradient():
    max_nz = 5
    regularizer = ot.smooth.SparsityConstrained(max_nz=max_nz)
    rng = np.random.RandomState(0)
    X = rng.randn(
        10,
    )
    b = 0.5

    def delta_omega_func(X):
        return regularizer.delta_Omega(X)[0]

    def delta_omega_grad(X):
        return regularizer.delta_Omega(X)[1]

    dual_grad_err = check_grad(delta_omega_func, delta_omega_grad, X)
    np.testing.assert_allclose(dual_grad_err, 0.0, atol=1e-07)

    def max_omega_func(X, b):
        return regularizer.max_Omega(X, b)[0]

    def max_omega_grad(X, b):
        return regularizer.max_Omega(X, b)[1]

    semi_dual_grad_err = check_grad(max_omega_func, max_omega_grad, X, b)
    np.testing.assert_allclose(semi_dual_grad_err, 0.0, atol=1e-07)
