"""Tests for module Unbalanced OT with entropy regularization"""

# Author: Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License


import itertools
import numpy as np
import ot
import pytest
from ot.gromov._unbalanced import fused_unbalanced_gromov_wasserstein, fused_unbalanced_gromov_wasserstein2


@pytest.mark.parametrize("unbalanced_solver, divergence", itertools.product(["mm", "lbfgsb"], ["kl", "l2"]))
def test_fused_unbalanced_gromov_wasserstein(nx, unbalanced_solver, divergence):
    n_samples = 20  # nb samples
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=1)
    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    G0 = (1 / (1.0 * n_samples)) * np.eye(n_samples, n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)
    # G = ot.gromov.gromov_wasserstein(
    #     C1, C2, None, q, 'square_loss', G0=G0, verbose=True,
    #     alpha_min=0., alpha_max=1.)
    # Gb = nx.to_numpy(ot.gromov.gromov_wasserstein(
    #     C1b, C2b, pb, None, 'square_loss', symmetric=True, G0=G0b, verbose=True))

    reg_m = (20, 10)
    eps = 0
    max_iter_ot = 10000
    max_iter = 1000
    tol = 1e-7
    tol_ot = 1e-7

    pi_samp, pi_feat, log_fugw = fused_unbalanced_gromov_wasserstein(
        C1, C1, wx=p, wy=p, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=0, M=None, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot, method_sinkhorn="sinkhorn",
        log=True    , verbose=False
    )
    print("fugw = {:.6f}".format(log_fugw["fugw_cost"].item()))
    fugw, log_fugw = fused_unbalanced_gromov_wasserstein2(
        C1, C1, wx=p, wy=p, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=0, M=None, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot, method_sinkhorn="sinkhorn",
        log=True, verbose=False
    )

    np.testing.assert_allclose(fugw, 0, atol=1e-04)

    # Id = (1 / (1.0 * n_samples)) * np.eye(n_samples, n_samples)
    # np.testing.assert_allclose(nx.to_numpy(pi_samp), Id, atol=1e-03)

    # # check constraints
    # np.testing.assert_allclose(G, Gb, atol=1e-06)
    # np.testing.assert_allclose(
    #     p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    # np.testing.assert_allclose(
    #     q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    # Id = (1 / (1.0 * n_samples)) * np.eye(n_samples, n_samples)

    # np.testing.assert_allclose(Gb, np.flipud(Id), atol=1e-04)

    # gw, log = ot.gromov.gromov_wasserstein2(C1, C2, None, q, 'kl_loss', armijo=True, log=True)
    # gwb, logb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, None, 'kl_loss', armijo=True, log=True)
    # gwb = nx.to_numpy(gwb)

    # gw_val = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'kl_loss', armijo=True, G0=G0, log=False)
    # gw_valb = nx.to_numpy(
    #     ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', armijo=True, G0=G0b, log=False)
    # )

    # G = log['T']
    # Gb = nx.to_numpy(logb['T'])

    # np.testing.assert_allclose(gw, gwb, atol=1e-06)
    # np.testing.assert_allclose(gwb, 0, atol=1e-1, rtol=1e-1)

    # np.testing.assert_allclose(gw_val, gw_valb, atol=1e-06)
    # np.testing.assert_allclose(gwb, gw_valb, atol=1e-1, rtol=1e-1)  # cf log=False

    # # check constraints
    # np.testing.assert_allclose(G, Gb, atol=1e-06)
    # np.testing.assert_allclose(
    #     p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    # np.testing.assert_allclose(
    #     q, Gb.sum(0), atol=1e-04)  # cf convergence gromov
