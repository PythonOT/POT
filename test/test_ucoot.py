"""Tests for module Unbalanced Co-Optimal Transport"""

# Author: Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import itertools
import numpy as np
import ot
import pytest
from ot.gromov._unbalanced import (
    unbalanced_co_optimal_transport,
    unbalanced_co_optimal_transport2,
)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize(
    "unbalanced_solver, divergence", itertools.product(["mm", "lbfgsb"], ["kl", "l2"])
)
def test_sanity(nx, unbalanced_solver, divergence):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    xs_nx, xt_nx = nx.from_numpy(xs, xt)
    px_s_nx, px_f_nx, py_s_nx, py_f_nx = nx.from_numpy(px_s, px_f, py_s, py_f)

    reg_m = (10, 5)
    eps = 0
    max_iter_ot = 200
    max_iter = 200
    tol = 1e-7
    tol_ot = 1e-7

    # test couplings
    anti_id_sample = np.flipud(np.eye(n_samples, n_samples)) / n_samples
    id_feature = np.eye(2, 2) / 2

    pi_sample, pi_feature = unbalanced_co_optimal_transport(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=0,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    pi_sample_nx, pi_feature_nx = unbalanced_co_optimal_transport(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=0,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample, anti_id_sample, atol=1e-05)
    np.testing.assert_allclose(pi_sample_nx, pi_sample, atol=1e-06)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)
    np.testing.assert_allclose(pi_feature, id_feature, atol=1e-05)

    # test divergence
    ucoot = unbalanced_co_optimal_transport2(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=0,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = unbalanced_co_optimal_transport2(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=0,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = nx.to_numpy(ucoot_nx)
    np.testing.assert_allclose(ucoot, ucoot_nx, atol=1e-08)
    np.testing.assert_allclose(ucoot, 0, atol=1e-06)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize(
    "unbalanced_solver, divergence, eps",
    itertools.product(
        ["sinkhorn", "sinkhorn_log", "mm", "lbfgsb"], ["kl", "l2"], [0, 1]
    ),
)
def test_init_plans(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)
    G0_samp = px_s[:, None] * py_s[None, :]
    G0_feat = px_f[:, None] * py_f[None, :]

    xs_nx, xt_nx, G0_samp_nx, G0_feat_nx = nx.from_numpy(xs, xt, G0_samp, G0_feat)
    px_s_nx, px_f_nx, py_s_nx, py_f_nx = nx.from_numpy(px_s, px_f, py_s, py_f)

    reg_m = (1, 5)
    alpha = (0.1, 0.2)
    max_iter_ot = 5
    max_iter = 5
    tol = 1e-7
    tol_ot = 1e-7

    # test couplings
    pi_sample, pi_feature = unbalanced_co_optimal_transport(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    pi_sample_nx, pi_feature_nx = unbalanced_co_optimal_transport(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=(G0_samp_nx, G0_feat_nx),
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample_nx, pi_sample, atol=1e-03)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-03)

    # test divergence
    ucoot = unbalanced_co_optimal_transport2(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = unbalanced_co_optimal_transport2(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=(G0_samp_nx, G0_feat_nx),
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = nx.to_numpy(ucoot_nx)
    np.testing.assert_allclose(ucoot, ucoot_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize(
    "unbalanced_solver, divergence, eps",
    itertools.product(
        ["sinkhorn", "sinkhorn_log", "mm", "lbfgsb"], ["kl", "l2"], [0, 1]
    ),
)
def test_init_duals(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    xs_nx, xt_nx = nx.from_numpy(xs, xt)
    px_s_nx, px_f_nx, py_s_nx, py_f_nx = nx.from_numpy(px_s, px_f, py_s, py_f)

    init_duals_samp = nx.from_numpy(np.zeros(n_samples), np.zeros(n_samples))
    init_duals_feat = nx.from_numpy(np.zeros(2), np.zeros(2))
    init_duals = (init_duals_samp, init_duals_feat)

    reg_m = (10, 5)
    alpha = (0.1, 0.2)
    max_iter_ot = 5
    max_iter = 5
    tol = 1e-7
    tol_ot = 1e-7

    # test couplings
    pi_sample, pi_feature = unbalanced_co_optimal_transport(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    pi_sample_nx, pi_feature_nx = unbalanced_co_optimal_transport(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=init_duals,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample_nx, pi_sample, atol=1e-03)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-03)

    # test divergence
    ucoot = unbalanced_co_optimal_transport2(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = unbalanced_co_optimal_transport2(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=init_duals,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = nx.to_numpy(ucoot_nx)
    np.testing.assert_allclose(ucoot, ucoot_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize(
    "unbalanced_solver, divergence, eps",
    itertools.product(
        ["sinkhorn", "sinkhorn_log", "mm", "lbfgsb"], ["kl", "l2"], [0, 1e-2]
    ),
)
def test_linear_part(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    xs_nx, xt_nx = nx.from_numpy(xs, xt)
    px_s_nx, px_f_nx, py_s_nx, py_f_nx = nx.from_numpy(px_s, px_f, py_s, py_f)

    # linear part
    M_samp = np.ones((n_samples, n_samples))
    np.fill_diagonal(np.fliplr(M_samp), 0)
    M_feat = np.ones((2, 2))
    np.fill_diagonal(M_feat, 0)
    M_samp_nx, M_feat_nx = nx.from_numpy(M_samp, M_feat)

    reg_m = (10, 5)
    alpha = (0.1, 0.2)
    max_iter_ot = 5
    max_iter = 5
    tol = 1e-7
    tol_ot = 1e-7

    # test couplings
    pi_sample, pi_feature = unbalanced_co_optimal_transport(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=M_samp,
        M_feat=M_feat,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    pi_sample_nx, pi_feature_nx = unbalanced_co_optimal_transport(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=M_samp_nx,
        M_feat=M_feat_nx,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample_nx, pi_sample, atol=1e-06)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

    # test divergence
    ucoot = unbalanced_co_optimal_transport2(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=M_samp,
        M_feat=M_feat,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = unbalanced_co_optimal_transport2(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=M_samp_nx,
        M_feat=M_feat_nx,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = nx.to_numpy(ucoot_nx)
    np.testing.assert_allclose(ucoot, ucoot_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize(
    "unbalanced_solver, divergence, eps",
    itertools.product(
        ["sinkhorn", "sinkhorn_log", "mm", "lbfgsb"], ["kl", "l2"], [0, 1]
    ),
)
def test_reg_marginals(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    xs_nx, xt_nx = nx.from_numpy(xs, xt)
    px_s_nx, px_f_nx, py_s_nx, py_f_nx = nx.from_numpy(px_s, px_f, py_s, py_f)

    alpha = (0.1, 0.2)
    max_iter_ot = 5
    max_iter = 5
    tol = 1e-7
    tol_ot = 1e-7

    reg_m = 100
    full_list_reg_m = [reg_m, reg_m]
    full_tuple_reg_m = (reg_m, reg_m)
    tuple_reg_m, list_reg_m = (reg_m), [reg_m]

    list_options = [full_tuple_reg_m, tuple_reg_m, full_list_reg_m, list_reg_m]

    # test couplings
    pi_sample, pi_feature = unbalanced_co_optimal_transport(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    # test divergence
    ucoot = unbalanced_co_optimal_transport2(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    for opt in list_options:
        pi_sample_nx, pi_feature_nx = unbalanced_co_optimal_transport(
            X=xs_nx,
            Y=xt_nx,
            wx_samp=px_s_nx,
            wx_feat=px_f_nx,
            wy_samp=py_s_nx,
            wy_feat=py_f_nx,
            reg_marginals=opt,
            epsilon=eps,
            divergence=divergence,
            unbalanced_solver=unbalanced_solver,
            alpha=alpha,
            M_samp=None,
            M_feat=None,
            init_pi=None,
            init_duals=None,
            max_iter=max_iter,
            tol=tol,
            max_iter_ot=max_iter_ot,
            tol_ot=tol_ot,
            log=False,
            verbose=False,
        )
        pi_sample_nx = nx.to_numpy(pi_sample_nx)
        pi_feature_nx = nx.to_numpy(pi_feature_nx)

        np.testing.assert_allclose(pi_sample_nx, pi_sample, atol=1e-06)
        np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

        # test divergence
        ucoot_nx = unbalanced_co_optimal_transport2(
            X=xs_nx,
            Y=xt_nx,
            wx_samp=px_s_nx,
            wx_feat=px_f_nx,
            wy_samp=py_s_nx,
            wy_feat=py_f_nx,
            reg_marginals=opt,
            epsilon=eps,
            divergence=divergence,
            unbalanced_solver=unbalanced_solver,
            alpha=alpha,
            M_samp=None,
            M_feat=None,
            init_pi=None,
            init_duals=None,
            max_iter=max_iter,
            tol=tol,
            max_iter_ot=max_iter_ot,
            tol_ot=tol_ot,
            method_sinkhorn="sinkhorn",
            log=False,
            verbose=False,
        )

        ucoot_nx = nx.to_numpy(ucoot_nx)
        np.testing.assert_allclose(ucoot, ucoot_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize(
    "unbalanced_solver, divergence, alpha",
    itertools.product(
        ["sinkhorn", "sinkhorn_log", "mm", "lbfgsb"], ["kl", "l2"], [0, 1]
    ),
)
def test_eps(nx, unbalanced_solver, divergence, alpha):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    xs_nx, xt_nx = nx.from_numpy(xs, xt)
    px_s_nx, px_f_nx, py_s_nx, py_f_nx = nx.from_numpy(px_s, px_f, py_s, py_f)

    reg_m = (10, 5)
    alpha = (0.1, 0.2)
    max_iter_ot = 5
    max_iter = 5
    tol = 1e-7
    tol_ot = 1e-7

    eps = 1
    full_list_eps = [eps, eps]
    full_tuple_eps = (eps, eps)
    tuple_eps, list_eps = (eps), [eps]

    list_options = [full_list_eps, full_tuple_eps, tuple_eps, list_eps]

    # test couplings
    pi_sample, pi_feature = unbalanced_co_optimal_transport(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    # test divergence
    ucoot = unbalanced_co_optimal_transport2(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    for opt in list_options:
        pi_sample_nx, pi_feature_nx = unbalanced_co_optimal_transport(
            X=xs_nx,
            Y=xt_nx,
            wx_samp=px_s_nx,
            wx_feat=px_f_nx,
            wy_samp=py_s_nx,
            wy_feat=py_f_nx,
            reg_marginals=reg_m,
            epsilon=opt,
            divergence=divergence,
            unbalanced_solver=unbalanced_solver,
            alpha=alpha,
            M_samp=None,
            M_feat=None,
            init_pi=None,
            init_duals=None,
            max_iter=max_iter,
            tol=tol,
            max_iter_ot=max_iter_ot,
            tol_ot=tol_ot,
            log=False,
            verbose=False,
        )
        pi_sample_nx = nx.to_numpy(pi_sample_nx)
        pi_feature_nx = nx.to_numpy(pi_feature_nx)

        np.testing.assert_allclose(pi_sample_nx, pi_sample, atol=1e-06)
        np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

        # test divergence
        ucoot_nx = unbalanced_co_optimal_transport2(
            X=xs_nx,
            Y=xt_nx,
            wx_samp=px_s_nx,
            wx_feat=px_f_nx,
            wy_samp=py_s_nx,
            wy_feat=py_f_nx,
            reg_marginals=reg_m,
            epsilon=opt,
            divergence=divergence,
            unbalanced_solver=unbalanced_solver,
            alpha=alpha,
            M_samp=None,
            M_feat=None,
            init_pi=None,
            init_duals=None,
            max_iter=max_iter,
            tol=tol,
            max_iter_ot=max_iter_ot,
            tol_ot=tol_ot,
            log=False,
            verbose=False,
        )

        ucoot_nx = nx.to_numpy(ucoot_nx)
        np.testing.assert_allclose(ucoot, ucoot_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize(
    "unbalanced_solver, divergence, eps",
    itertools.product(
        ["sinkhorn", "sinkhorn_log", "mm", "lbfgsb"], ["kl", "l2"], [0, 1e-2]
    ),
)
def test_alpha(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    xs_nx, xt_nx = nx.from_numpy(xs, xt)
    px_s_nx, px_f_nx, py_s_nx, py_f_nx = nx.from_numpy(px_s, px_f, py_s, py_f)

    # linear part
    M_samp = np.ones((n_samples, n_samples))
    np.fill_diagonal(np.fliplr(M_samp), 0)
    M_feat = np.ones((2, 2))
    np.fill_diagonal(M_feat, 0)
    M_samp_nx, M_feat_nx = nx.from_numpy(M_samp, M_feat)

    reg_m = (10, 5)
    max_iter_ot = 5
    max_iter = 5
    tol = 1e-7
    tol_ot = 1e-7

    alpha = 1
    full_list_alpha = [alpha, alpha]
    full_tuple_alpha = (alpha, alpha)
    tuple_alpha, list_alpha = (alpha), [alpha]

    list_options = [full_list_alpha, full_tuple_alpha, tuple_alpha, list_alpha]

    # test couplings
    pi_sample, pi_feature = unbalanced_co_optimal_transport(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=M_samp,
        M_feat=M_feat,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    # test divergence
    ucoot = unbalanced_co_optimal_transport2(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=M_samp,
        M_feat=M_feat,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    for opt in list_options:
        pi_sample_nx, pi_feature_nx = unbalanced_co_optimal_transport(
            X=xs_nx,
            Y=xt_nx,
            wx_samp=px_s_nx,
            wx_feat=px_f_nx,
            wy_samp=py_s_nx,
            wy_feat=py_f_nx,
            reg_marginals=reg_m,
            epsilon=eps,
            divergence=divergence,
            unbalanced_solver=unbalanced_solver,
            alpha=opt,
            M_samp=M_samp_nx,
            M_feat=M_feat_nx,
            init_pi=None,
            init_duals=None,
            max_iter=max_iter,
            tol=tol,
            max_iter_ot=max_iter_ot,
            tol_ot=tol_ot,
            log=False,
            verbose=False,
        )
        pi_sample_nx = nx.to_numpy(pi_sample_nx)
        pi_feature_nx = nx.to_numpy(pi_feature_nx)

        np.testing.assert_allclose(pi_sample_nx, pi_sample, atol=1e-06)
        np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

        ucoot_nx = unbalanced_co_optimal_transport2(
            X=xs_nx,
            Y=xt_nx,
            wx_samp=px_s_nx,
            wx_feat=px_f_nx,
            wy_samp=py_s_nx,
            wy_feat=py_f_nx,
            reg_marginals=reg_m,
            epsilon=eps,
            divergence=divergence,
            unbalanced_solver=unbalanced_solver,
            alpha=opt,
            M_samp=M_samp_nx,
            M_feat=M_feat_nx,
            init_pi=None,
            init_duals=None,
            max_iter=max_iter,
            tol=tol,
            max_iter_ot=max_iter_ot,
            tol_ot=tol_ot,
            log=False,
            verbose=False,
        )

        ucoot_nx = nx.to_numpy(ucoot_nx)
        np.testing.assert_allclose(ucoot, ucoot_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize(
    "unbalanced_solver, divergence, eps",
    itertools.product(
        ["sinkhorn", "sinkhorn_log", "mm", "lbfgsb"], ["kl", "l2"], [0, 1]
    ),
)
def test_log(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    xs_nx, xt_nx = nx.from_numpy(xs, xt)
    px_s_nx, px_f_nx, py_s_nx, py_f_nx = nx.from_numpy(px_s, px_f, py_s, py_f)

    reg_m = (10, 5)
    alpha = (0.1, 0.2)
    max_iter_ot = 5
    max_iter = 5
    tol = 1e-7
    tol_ot = 1e-7

    # test couplings
    pi_sample, pi_feature = unbalanced_co_optimal_transport(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    pi_sample_nx, pi_feature_nx, log = unbalanced_co_optimal_transport(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=True,
        verbose=False,
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample_nx, pi_sample, atol=1e-06)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

    # test divergence
    ucoot = unbalanced_co_optimal_transport2(
        X=xs,
        Y=xt,
        wx_samp=px_s,
        wx_feat=px_f,
        wy_samp=py_s,
        wy_feat=py_f,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = unbalanced_co_optimal_transport2(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = nx.to_numpy(ucoot_nx)
    np.testing.assert_allclose(ucoot, ucoot_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
@pytest.mark.parametrize(
    "unbalanced_solver, divergence, eps",
    itertools.product(
        ["sinkhorn", "sinkhorn_log", "mm", "lbfgsb"], ["kl", "l2"], [0, 1]
    ),
)
def test_marginals(nx, unbalanced_solver, divergence, eps):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    xs_nx, xt_nx = nx.from_numpy(xs, xt)
    px_s_nx, px_f_nx, py_s_nx, py_f_nx = nx.from_numpy(px_s, px_f, py_s, py_f)

    reg_m = (10, 5)
    alpha = (0.1, 0.2)
    max_iter_ot = 5
    max_iter = 5
    tol = 1e-7
    tol_ot = 1e-7

    # test couplings
    pi_sample, pi_feature = unbalanced_co_optimal_transport(
        X=xs,
        Y=xt,
        wx_samp=None,
        wx_feat=None,
        wy_samp=None,
        wy_feat=None,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    pi_sample_nx, pi_feature_nx = unbalanced_co_optimal_transport(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample_nx, pi_sample, atol=1e-06)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-06)

    # test divergence
    ucoot = unbalanced_co_optimal_transport2(
        X=xs,
        Y=xt,
        wx_samp=None,
        wx_feat=None,
        wy_samp=None,
        wy_feat=None,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = unbalanced_co_optimal_transport2(
        X=xs_nx,
        Y=xt_nx,
        wx_samp=px_s_nx,
        wx_feat=px_f_nx,
        wy_samp=py_s_nx,
        wy_feat=py_f_nx,
        reg_marginals=reg_m,
        epsilon=eps,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=None,
        M_feat=None,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=False,
        verbose=False,
    )

    ucoot_nx = nx.to_numpy(ucoot_nx)
    np.testing.assert_allclose(ucoot, ucoot_nx, atol=1e-08)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tensorflow backend")
def test_raise_value_error(nx):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()

    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    xs_nx, xt_nx = nx.from_numpy(xs, xt)
    px_s_nx, px_f_nx, py_s_nx, py_f_nx = nx.from_numpy(px_s, px_f, py_s, py_f)

    reg_m = (10, 5)
    eps = 0
    max_iter_ot = 5
    max_iter = 5
    tol = 1e-7
    tol_ot = 1e-7

    # raise error of divergence
    def ucoot_div(divergence):
        return unbalanced_co_optimal_transport(
            X=xs,
            Y=xt,
            wx_samp=px_s,
            wx_feat=px_f,
            wy_samp=py_s,
            wy_feat=py_f,
            reg_marginals=reg_m,
            epsilon=eps,
            divergence=divergence,
            unbalanced_solver="mm",
            alpha=0,
            M_samp=None,
            M_feat=None,
            init_pi=None,
            init_duals=None,
            max_iter=max_iter,
            tol=tol,
            max_iter_ot=max_iter_ot,
            tol_ot=tol_ot,
            log=False,
            verbose=False,
        )

    def ucoot_div_nx(divergence):
        return unbalanced_co_optimal_transport(
            X=xs_nx,
            Y=xt_nx,
            wx_samp=px_s_nx,
            wx_feat=px_f_nx,
            wy_samp=py_s_nx,
            wy_feat=py_f_nx,
            reg_marginals=reg_m,
            epsilon=eps,
            divergence=divergence,
            unbalanced_solver="mm",
            alpha=0,
            M_samp=None,
            M_feat=None,
            init_pi=None,
            init_duals=None,
            max_iter=max_iter,
            tol=tol,
            max_iter_ot=max_iter_ot,
            tol_ot=tol_ot,
            log=False,
            verbose=False,
        )

    np.testing.assert_raises(NotImplementedError, ucoot_div, "div_not_existed")
    np.testing.assert_raises(NotImplementedError, ucoot_div_nx, "div_not_existed")

    # raise error of solver
    def ucoot_solver(unbalanced_solver):
        return unbalanced_co_optimal_transport(
            X=xs,
            Y=xt,
            wx_samp=px_s,
            wx_feat=px_f,
            wy_samp=py_s,
            wy_feat=py_f,
            reg_marginals=reg_m,
            epsilon=eps,
            divergence="kl",
            unbalanced_solver=unbalanced_solver,
            alpha=0,
            M_samp=None,
            M_feat=None,
            init_pi=None,
            init_duals=None,
            max_iter=max_iter,
            tol=tol,
            max_iter_ot=max_iter_ot,
            tol_ot=tol_ot,
            log=False,
            verbose=False,
        )

    def ucoot_solver_nx(unbalanced_solver):
        return unbalanced_co_optimal_transport(
            X=xs_nx,
            Y=xt_nx,
            wx_samp=px_s_nx,
            wx_feat=px_f_nx,
            wy_samp=py_s_nx,
            wy_feat=py_f_nx,
            reg_marginals=reg_m,
            epsilon=eps,
            divergence="kl",
            unbalanced_solver=unbalanced_solver,
            alpha=0,
            M_samp=None,
            M_feat=None,
            init_pi=None,
            init_duals=None,
            max_iter=max_iter,
            tol=tol,
            max_iter_ot=max_iter_ot,
            tol_ot=tol_ot,
            log=False,
            verbose=False,
        )

    np.testing.assert_raises(NotImplementedError, ucoot_solver, "solver_not_existed")
    np.testing.assert_raises(NotImplementedError, ucoot_solver_nx, "solver_not_existed")
