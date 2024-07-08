"""Tests for module Fused Unbalanced Gromov-Wasserstein"""

# Author: Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License


import itertools
import numpy as np
import ot
import pytest
from ot.gromov._unbalanced import fused_unbalanced_gromov_wasserstein, fused_unbalanced_gromov_wasserstein2


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize("unbalanced_solver, divergence", itertools.product(["mm", "lbfgsb"], ["kl", "l2"]))
def test_sanity(nx, unbalanced_solver, divergence):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(
        n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    # linear part
    M_samp = np.ones((n_samples, n_samples))
    np.fill_diagonal(np.fliplr(M_samp), 0)
    M_samp_nx = nx.from_numpy(M_samp)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    reg_m = (100, 50)
    eps = 0
    alpha = 0.5
    max_iter_ot = 10000
    max_iter = 10000
    tol = 1e-7
    tol_ot = 1e-7

    # test couplings
    anti_id_sample = np.flipud(np.eye(n_samples, n_samples)) / n_samples

    pi_sample, pi_feature = fused_unbalanced_gromov_wasserstein(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=G0, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    pi_sample_nx, pi_feature_nx = fused_unbalanced_gromov_wasserstein(
        C1b, C2b, wx=pb, wy=qb, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp_nx, init_duals=None, init_pi=G0b, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample, anti_id_sample, atol=1e-03)
    np.testing.assert_allclose(pi_sample_nx, pi_sample, atol=1e-06)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

    # test divergence

    fugw = fused_unbalanced_gromov_wasserstein2(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=G0, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    fugw_nx = fused_unbalanced_gromov_wasserstein2(
        C1b, C2b, wx=pb, wy=qb, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp_nx, init_duals=None, init_pi=G0b, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    fugw_nx = nx.to_numpy(fugw_nx)
    np.testing.assert_allclose(fugw, fugw_nx, atol=1e-08)
    np.testing.assert_allclose(fugw, 0, atol=1e-02)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize("unbalanced_solver, divergence, eps", itertools.product(["scaling", "mm", "lbfgsb"], ["kl", "l2"], [0, 1e-2]))
def test_init_plans(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(
        n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    # linear part
    M_samp = np.ones((n_samples, n_samples))
    np.fill_diagonal(np.fliplr(M_samp), 0)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    reg_m = (100, 50)
    alpha = 0.5
    max_iter_ot = 1000
    max_iter = 1000
    tol = 1e-5
    tol_ot = 1e-5

    pi_sample, pi_feature = fused_unbalanced_gromov_wasserstein(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=G0, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    pi_sample_nx, pi_feature_nx = fused_unbalanced_gromov_wasserstein(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample, pi_sample_nx, atol=1e-06)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

    # test divergence

    fugw = fused_unbalanced_gromov_wasserstein2(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=G0, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    fugw_nx = fused_unbalanced_gromov_wasserstein2(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    fugw_nx = nx.to_numpy(fugw_nx)
    np.testing.assert_allclose(fugw, fugw_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize("unbalanced_solver, divergence, eps", itertools.product(["scaling", "mm", "lbfgsb"], ["kl", "l2"], [0, 1e-2]))
def test_init_duals(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(
        n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    dual1, dual2 = nx.from_numpy(np.zeros_like(p), np.zeros_like(q))
    init_duals = (dual1, dual2)

    # linear part
    M_samp = np.ones((n_samples, n_samples))
    np.fill_diagonal(np.fliplr(M_samp), 0)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    C1, C2, p, q, M_samp = nx.from_numpy(C1, C2, p, q, M_samp)

    reg_m = (100, 50)
    alpha = 0.5
    max_iter_ot = 1000
    max_iter = 1000
    tol = 1e-5
    tol_ot = 1e-5

    pi_sample, pi_feature = fused_unbalanced_gromov_wasserstein(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    pi_sample_nx, pi_feature_nx = fused_unbalanced_gromov_wasserstein(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=init_duals, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample, pi_sample_nx, atol=1e-06)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

    # test divergence
    fugw = fused_unbalanced_gromov_wasserstein2(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    fugw_nx = fused_unbalanced_gromov_wasserstein2(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=init_duals, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    fugw_nx = nx.to_numpy(fugw_nx)
    np.testing.assert_allclose(fugw, fugw_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize("unbalanced_solver, divergence, eps", itertools.product(["scaling", "mm", "lbfgsb"], ["kl", "l2"], [0, 1e-2]))
def test_reg_marginals(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(
        n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    # linear part
    M_samp = np.ones((n_samples, n_samples))
    np.fill_diagonal(np.fliplr(M_samp), 0)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    alpha = 0.5
    max_iter_ot = 1000
    max_iter = 1000
    tol = 1e-5
    tol_ot = 1e-5

    reg_m = 100
    full_list_reg_m = [reg_m, reg_m]
    full_tuple_reg_m = (reg_m, reg_m)
    tuple_reg_m, list_reg_m = (reg_m), [reg_m]

    list_options = [full_tuple_reg_m, tuple_reg_m, full_list_reg_m, list_reg_m]

    pi_sample, pi_feature = fused_unbalanced_gromov_wasserstein(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=G0, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    fugw = fused_unbalanced_gromov_wasserstein2(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=G0, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    for opt in list_options:
        pi_sample_nx, pi_feature_nx = fused_unbalanced_gromov_wasserstein(
            C1, C2, wx=p, wy=q, reg_marginals=opt, epsilon=eps,
            divergence=divergence, unbalanced_solver=unbalanced_solver,
            alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
            tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
            method_sinkhorn="sinkhorn", log=False, verbose=False
        )
        pi_sample_nx = nx.to_numpy(pi_sample_nx)
        pi_feature_nx = nx.to_numpy(pi_feature_nx)

        np.testing.assert_allclose(pi_sample, pi_sample_nx, atol=1e-06)
        np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

        # test divergence
        fugw_nx = fused_unbalanced_gromov_wasserstein2(
            C1, C2, wx=p, wy=q, reg_marginals=opt, epsilon=eps,
            divergence=divergence, unbalanced_solver=unbalanced_solver,
            alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
            tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
            method_sinkhorn="sinkhorn", log=False, verbose=False
        )

        fugw_nx = nx.to_numpy(fugw_nx)
        np.testing.assert_allclose(fugw, fugw_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize("unbalanced_solver, divergence, eps", itertools.product(["scaling", "mm", "lbfgsb"], ["kl", "l2"], [0, 1e-2]))
def test_log(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(
        n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    # linear part
    M_samp = np.ones((n_samples, n_samples))
    np.fill_diagonal(np.fliplr(M_samp), 0)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    reg_m = (100, 50)
    alpha = 0.5
    max_iter_ot = 1000
    max_iter = 1000
    tol = 1e-5
    tol_ot = 1e-5

    pi_sample, pi_feature = fused_unbalanced_gromov_wasserstein(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    pi_sample_nx, pi_feature_nx, log = fused_unbalanced_gromov_wasserstein(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=True, verbose=False
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample, pi_sample_nx, atol=1e-06)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

    # test divergence

    fugw = fused_unbalanced_gromov_wasserstein2(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    fugw_nx, log = fused_unbalanced_gromov_wasserstein2(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=True, verbose=False
    )

    fugw_nx = nx.to_numpy(fugw_nx)
    np.testing.assert_allclose(fugw, fugw_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize("unbalanced_solver, divergence, eps", itertools.product(["scaling", "mm", "lbfgsb"], ["kl", "l2"], [0, 1e-2]))
def test_marginals(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(
        n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    # linear part
    M_samp = np.ones((n_samples, n_samples))
    np.fill_diagonal(np.fliplr(M_samp), 0)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    reg_m = (100, 50)
    alpha = 0.5
    max_iter_ot = 1000
    max_iter = 1000
    tol = 1e-5
    tol_ot = 1e-5

    pi_sample, pi_feature = fused_unbalanced_gromov_wasserstein(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    pi_sample_nx, pi_feature_nx = fused_unbalanced_gromov_wasserstein(
        C1, C2, wx=None, wy=None, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample, pi_sample_nx, atol=1e-06)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

    # test divergence

    fugw = fused_unbalanced_gromov_wasserstein2(
        C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    fugw_nx = fused_unbalanced_gromov_wasserstein2(
        C1, C2, wx=None, wy=None, reg_marginals=reg_m, epsilon=eps,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M_samp, init_duals=None, init_pi=None, max_iter=max_iter,
        tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn="sinkhorn", log=False, verbose=False
    )

    fugw_nx = nx.to_numpy(fugw_nx)
    np.testing.assert_allclose(fugw, fugw_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
def test_raise_value_error(nx):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    eps = 1e-2
    reg_m = (10, 100)
    max_iter_ot = 1000
    max_iter = 1000
    tol = 1e-6
    tol_ot = 1e-6

    # raise error of divergence
    def fugw_div(divergence):
        return fused_unbalanced_gromov_wasserstein(
            C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
            divergence=divergence, unbalanced_solver="mm",
            alpha=0, M=None, init_duals=None, init_pi=G0, max_iter=max_iter,
            tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
            method_sinkhorn="sinkhorn", log=False, verbose=False
        )

    def fugw_div_nx(divergence):
        return fused_unbalanced_gromov_wasserstein(
            C1b, C2b, wx=pb, wy=qb, reg_marginals=reg_m, epsilon=eps,
            divergence=divergence, unbalanced_solver="mm",
            alpha=0, M=None, init_duals=None, init_pi=G0b, max_iter=max_iter,
            tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
            method_sinkhorn="sinkhorn", log=False, verbose=False
        )

    np.testing.assert_raises(NotImplementedError, fugw_div, "div_not_existed")
    np.testing.assert_raises(NotImplementedError, fugw_div_nx, "div_not_existed")

    # raise error of solver
    def fugw_solver(unbalanced_solver):
        return fused_unbalanced_gromov_wasserstein(
            C1, C2, wx=p, wy=q, reg_marginals=reg_m, epsilon=eps,
            divergence="kl", unbalanced_solver=unbalanced_solver,
            alpha=0, M=None, init_duals=None, init_pi=G0, max_iter=max_iter,
            tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
            method_sinkhorn="sinkhorn", log=False, verbose=False
        )

    def fugw_solver_nx(unbalanced_solver):
        return fused_unbalanced_gromov_wasserstein(
            C1b, C2b, wx=pb, wy=qb, reg_marginals=reg_m, epsilon=eps,
            divergence="kl", unbalanced_solver=unbalanced_solver,
            alpha=0, M=None, init_duals=None, init_pi=G0b, max_iter=max_iter,
            tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
            method_sinkhorn="sinkhorn", log=False, verbose=False
        )

    np.testing.assert_raises(NotImplementedError, fugw_solver, "solver_not_existed")
    np.testing.assert_raises(NotImplementedError, fugw_solver_nx, "solver_not_existed")
