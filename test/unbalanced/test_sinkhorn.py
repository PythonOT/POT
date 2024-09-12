"""Tests for module Unbalanced OT with entropy regularization"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Laetitia Chapel <laetitia.chapel@univ-ubs.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License


import itertools
import numpy as np
import ot
import pytest
from ot.unbalanced import barycenter_unbalanced


@pytest.mark.parametrize("method,reg_type", itertools.product(["sinkhorn", "sinkhorn_stabilized", "sinkhorn_reg_scaling"], ["kl", "entropy"]))
def test_unbalanced_convergence(nx, method, reg_type):
    # test generalized sinkhorn for unbalanced OT
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = ot.utils.unif(n) * 1.5
    M = ot.dist(x, x)
    a, b, M = nx.from_numpy(a, b, M)

    epsilon = 1.
    reg_m = 1.

    G, log = ot.unbalanced.sinkhorn_unbalanced(
        a, b, M, reg=epsilon, reg_m=reg_m, method=method,
        reg_type=reg_type, log=True, verbose=True
    )
    loss = nx.to_numpy(ot.unbalanced.sinkhorn_unbalanced2(
        a, b, M, reg=epsilon, reg_m=reg_m, method=method,
        reg_type=reg_type, verbose=True
    ))
    # check fixed point equations
    # in log-domain
    fi = reg_m / (reg_m + epsilon)
    logb = nx.log(b + 1e-16)
    loga = nx.log(a + 1e-16)
    if reg_type == "entropy":
        logKtu = nx.logsumexp(log["logu"][None, :] - M.T / epsilon, axis=1)
        logKv = nx.logsumexp(log["logv"][None, :] - M / epsilon, axis=1)
    elif reg_type == "kl":
        log_ab = loga[:, None] + logb[None, :]
        logKtu = nx.logsumexp(log["logu"][None, :] - M.T / epsilon + log_ab.T, axis=1)
        logKv = nx.logsumexp(log["logv"][None, :] - M / epsilon + log_ab, axis=1)
    v_final = fi * (logb - logKtu)
    u_final = fi * (loga - logKv)

    np.testing.assert_allclose(
        nx.to_numpy(u_final), nx.to_numpy(log["logu"]), atol=1e-05)
    np.testing.assert_allclose(
        nx.to_numpy(v_final), nx.to_numpy(log["logv"]), atol=1e-05)

    # check if sinkhorn_unbalanced2 returns the correct loss
    np.testing.assert_allclose(nx.to_numpy(nx.sum(G * M)), loss, atol=1e-5)

    # check in case no histogram is provided
    M_np = nx.to_numpy(M)
    a_np, b_np = np.array([]), np.array([])
    a, b = nx.from_numpy(a_np, b_np)

    G = ot.unbalanced.sinkhorn_unbalanced(
        a, b, M, reg=epsilon, reg_m=reg_m,
        method=method, reg_type=reg_type, verbose=True
    )
    G_np = ot.unbalanced.sinkhorn_unbalanced(
        a_np, b_np, M_np, reg=epsilon, reg_m=reg_m,
        method=method, reg_type=reg_type, verbose=True
    )
    np.testing.assert_allclose(G_np, nx.to_numpy(G))


@pytest.mark.parametrize("method,reg_type", itertools.product(["sinkhorn", "sinkhorn_stabilized", "sinkhorn_reg_scaling"], ["kl", "entropy"]))
def test_unbalanced_warmstart(nx, method, reg_type):
    # test generalized sinkhorn for unbalanced OT
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)
    b = ot.utils.unif(n)
    M = ot.dist(x, x)
    a, b, M = nx.from_numpy(a, b, M)

    epsilon = 1.
    reg_m = 1.

    G0, log0 = ot.unbalanced.sinkhorn_unbalanced(
        a, b, M, reg=epsilon, reg_m=reg_m, method=method,
        reg_type=reg_type, warmstart=None, log=True, verbose=True
    )
    loss0 = ot.unbalanced.sinkhorn_unbalanced2(
        a, b, M, reg=epsilon, reg_m=reg_m, method=method,
        reg_type=reg_type, warmstart=None, verbose=True
    )

    dim_a, dim_b = M.shape
    warmstart = (nx.zeros(dim_a, type_as=M), nx.zeros(dim_b, type_as=M))
    G, log = ot.unbalanced.sinkhorn_unbalanced(
        a, b, M, reg=epsilon, reg_m=reg_m, method=method,
        reg_type=reg_type, warmstart=warmstart, log=True, verbose=True
    )
    loss = ot.unbalanced.sinkhorn_unbalanced2(
        a, b, M, reg=epsilon, reg_m=reg_m, method=method,
        reg_type=reg_type, warmstart=warmstart, verbose=True
    )

    _, log_emd = ot.lp.emd(a, b, M, log=True)
    warmstart1 = (log_emd["u"], log_emd["v"])
    G1, log1 = ot.unbalanced.sinkhorn_unbalanced(
        a, b, M, reg=epsilon, reg_m=reg_m, method=method,
        reg_type=reg_type, warmstart=warmstart1, log=True, verbose=True
    )
    loss1 = ot.unbalanced.sinkhorn_unbalanced2(
        a, b, M, reg=epsilon, reg_m=reg_m, method=method,
        reg_type=reg_type, warmstart=warmstart1, verbose=True
    )

    np.testing.assert_allclose(
        nx.to_numpy(log["logu"]), nx.to_numpy(log0["logu"]), atol=1e-05)
    np.testing.assert_allclose(
        nx.to_numpy(log["logv"]), nx.to_numpy(log0["logv"]), atol=1e-05)
    np.testing.assert_allclose(
        nx.to_numpy(log0["logu"]), nx.to_numpy(log1["logu"]), atol=1e-05)
    np.testing.assert_allclose(
        nx.to_numpy(log0["logv"]), nx.to_numpy(log1["logv"]), atol=1e-05)

    np.testing.assert_allclose(nx.to_numpy(G), nx.to_numpy(G0), atol=1e-05)
    np.testing.assert_allclose(nx.to_numpy(G0), nx.to_numpy(G1), atol=1e-05)

    np.testing.assert_allclose(nx.to_numpy(loss), nx.to_numpy(loss0), atol=1e-5)
    np.testing.assert_allclose(nx.to_numpy(loss0), nx.to_numpy(loss1), atol=1e-5)


@pytest.mark.parametrize("method,reg_type, log", itertools.product(["sinkhorn", "sinkhorn_stabilized", "sinkhorn_reg_scaling"], ["kl", "entropy"], [True, False]))
def test_sinkhorn_unbalanced2(nx, method, reg_type, log):
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = ot.utils.unif(n) * 1.5
    M = ot.dist(x, x)
    a, b, M = nx.from_numpy(a, b, M)

    epsilon = 1.
    reg_m = 1.

    loss = nx.to_numpy(ot.unbalanced.sinkhorn_unbalanced2(
        a, b, M, reg=epsilon, reg_m=reg_m, method=method,
        reg_type=reg_type, log=False, verbose=True
    ))

    res = ot.unbalanced.sinkhorn_unbalanced2(
        a, b, M, reg=epsilon, reg_m=reg_m, method=method,
        reg_type=reg_type, log=log, verbose=True
    )
    loss0 = res[0] if log else res

    np.testing.assert_allclose(nx.to_numpy(loss), nx.to_numpy(loss0), atol=1e-5)


@pytest.mark.parametrize("method,reg_m", itertools.product(["sinkhorn", "sinkhorn_stabilized", "sinkhorn_reg_scaling"], [1, float("inf")]))
def test_unbalanced_relaxation_parameters(nx, method, reg_m):
    # test generalized sinkhorn for unbalanced OT
    n = 100
    rng = np.random.RandomState(50)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = rng.rand(n, 2)

    M = ot.dist(x, x)
    epsilon = 1.

    a, b, M = nx.from_numpy(a, b, M)

    # options for reg_m
    full_list_reg_m = [reg_m, reg_m]
    full_tuple_reg_m = (reg_m, reg_m)
    tuple_reg_m, list_reg_m = (reg_m), [reg_m]
    nx_reg_m = reg_m * nx.ones(1)
    list_options = [nx_reg_m, full_tuple_reg_m,
                    tuple_reg_m, full_list_reg_m, list_reg_m]

    loss, log = ot.unbalanced.sinkhorn_unbalanced(
        a, b, M, reg=epsilon, reg_m=reg_m,
        method=method, log=True, verbose=True
    )

    for opt in list_options:
        loss_opt, log_opt = ot.unbalanced.sinkhorn_unbalanced(
            a, b, M, reg=epsilon, reg_m=opt,
            method=method, log=True, verbose=True
        )

        np.testing.assert_allclose(
            nx.to_numpy(log["logu"]), nx.to_numpy(log_opt["logu"]), atol=1e-05)
        np.testing.assert_allclose(
            nx.to_numpy(log["logv"]), nx.to_numpy(log_opt["logv"]), atol=1e-05)
        np.testing.assert_allclose(
            nx.to_numpy(loss), nx.to_numpy(loss_opt), atol=1e-05)


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized", "sinkhorn_reg_scaling"])
def test_unbalanced_multiple_inputs(nx, method):
    # test generalized sinkhorn for unbalanced OT
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = rng.rand(n, 2)

    M = ot.dist(x, x)
    epsilon = 1.
    reg_m = 1.

    a, b, M = nx.from_numpy(a, b, M)

    G, log = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=epsilon,
                                               reg_m=reg_m, method=method,
                                               log=True, verbose=True)

    # check fixed point equations
    # in log-domain
    fi = reg_m / (reg_m + epsilon)
    logb = nx.log(b + 1e-16)
    loga = nx.log(a + 1e-16)[:, None]
    logKtu = nx.logsumexp(
        log["logu"][:, None, :] - M[:, :, None] / epsilon, axis=0
    )
    logKv = nx.logsumexp(log["logv"][None, :] - M[:, :, None] / epsilon, axis=1)
    v_final = fi * (logb - logKtu)
    u_final = fi * (loga - logKv)

    np.testing.assert_allclose(
        nx.to_numpy(u_final), nx.to_numpy(log["logu"]), atol=1e-05)
    np.testing.assert_allclose(
        nx.to_numpy(v_final), nx.to_numpy(log["logv"]), atol=1e-05)


def test_stabilized_vs_sinkhorn(nx):
    # test if stable version matches sinkhorn
    n = 100

    # Gaussian distributions
    a = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    b1 = ot.datasets.make_1D_gauss(n, m=60, s=8)
    b2 = ot.datasets.make_1D_gauss(n, m=30, s=4)

    # creating matrix A containing all distributions
    b = np.vstack((b1, b2)).T

    M = ot.utils.dist0(n)
    M /= np.median(M)
    epsilon = 0.1
    reg_m = 1.

    ab, bb, Mb = nx.from_numpy(a, b, M)

    G, _ = ot.unbalanced.sinkhorn_unbalanced2(
        ab, bb, Mb, epsilon, reg_m, method="sinkhorn_stabilized", log=True
    )
    G2, _ = ot.unbalanced.sinkhorn_unbalanced2(
        ab, bb, Mb, epsilon, reg_m, method="sinkhorn", log=True
    )
    G2_np, _ = ot.unbalanced.sinkhorn_unbalanced2(
        a, b, M, epsilon, reg_m, method="sinkhorn", log=True
    )
    G = nx.to_numpy(G)
    G2 = nx.to_numpy(G2)

    np.testing.assert_allclose(G, G2, atol=1e-5)
    np.testing.assert_allclose(G2, G2_np, atol=1e-5)


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized"])
def test_unbalanced_barycenter(nx, method):
    # test generalized sinkhorn for unbalanced OT barycenter
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    A = rng.rand(n, 2)

    # make dists unbalanced
    A = A * np.array([1, 2])[None, :]
    M = ot.dist(x, x)
    epsilon = 1.
    reg_m = 1.

    A, M = nx.from_numpy(A, M)

    q, log = barycenter_unbalanced(
        A, M, reg=epsilon, reg_m=reg_m, method=method, log=True, verbose=True
    )
    # check fixed point equations
    fi = reg_m / (reg_m + epsilon)
    logA = nx.log(A + 1e-16)
    logq = nx.log(q + 1e-16)[:, None]
    logKtu = nx.logsumexp(
        log["logu"][:, None, :] - M[:, :, None] / epsilon, axis=0
    )
    logKv = nx.logsumexp(log["logv"][None, :] - M[:, :, None] / epsilon, axis=1)
    v_final = fi * (logq - logKtu)
    u_final = fi * (logA - logKv)

    np.testing.assert_allclose(
        nx.to_numpy(u_final), nx.to_numpy(log["logu"]), atol=1e-05)
    np.testing.assert_allclose(
        nx.to_numpy(v_final), nx.to_numpy(log["logv"]), atol=1e-05)


def test_barycenter_stabilized_vs_sinkhorn(nx):
    # test generalized sinkhorn for unbalanced OT barycenter
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    A = rng.rand(n, 2)

    # make dists unbalanced
    A = A * np.array([1, 4])[None, :]
    M = ot.dist(x, x)
    epsilon = 0.5
    reg_m = 10

    Ab, Mb = nx.from_numpy(A, M)

    qstable, _ = barycenter_unbalanced(
        Ab, Mb, reg=epsilon, reg_m=reg_m, log=True, tau=100,
        method="sinkhorn_stabilized", verbose=True
    )
    q, _ = barycenter_unbalanced(
        Ab, Mb, reg=epsilon, reg_m=reg_m, method="sinkhorn", log=True
    )
    q_np, _ = barycenter_unbalanced(
        A, M, reg=epsilon, reg_m=reg_m, method="sinkhorn", log=True
    )
    q, qstable = nx.to_numpy(q, qstable)
    np.testing.assert_allclose(q, qstable, atol=1e-05)
    np.testing.assert_allclose(q, q_np, atol=1e-05)


def test_wrong_method(nx):

    n = 10
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = ot.utils.unif(n) * 1.5

    M = ot.dist(x, x)
    epsilon = 1.
    reg_m = 1.

    a, b, M = nx.from_numpy(a, b, M)

    with pytest.raises(ValueError):
        ot.unbalanced.sinkhorn_unbalanced(
            a, b, M, reg=epsilon, reg_m=reg_m, method='badmethod',
            log=True, verbose=True
        )
    with pytest.raises(ValueError):
        ot.unbalanced.sinkhorn_unbalanced2(
            a, b, M, epsilon, reg_m, method='badmethod', verbose=True
        )


def test_implemented_methods(nx):
    IMPLEMENTED_METHODS = ['sinkhorn', 'sinkhorn_stabilized']
    TO_BE_IMPLEMENTED_METHODS = ['sinkhorn_reg_scaling']
    NOT_VALID_TOKENS = ['foo']
    # test generalized sinkhorn for unbalanced OT barycenter
    n = 3
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = ot.utils.unif(n) * 1.5
    A = rng.rand(n, 2)
    M = ot.dist(x, x)
    epsilon = 1.
    reg_m = 1.

    a, b, M, A = nx.from_numpy(a, b, M, A)

    for method in IMPLEMENTED_METHODS:
        ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, reg_m,
                                          method=method)
        ot.unbalanced.sinkhorn_unbalanced2(a, b, M, epsilon, reg_m,
                                           method=method)
        barycenter_unbalanced(A, M, reg=epsilon, reg_m=reg_m,
                              method=method)
    with pytest.warns(UserWarning, match='not implemented'):
        for method in set(TO_BE_IMPLEMENTED_METHODS):
            ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, reg_m,
                                              method=method)
            ot.unbalanced.sinkhorn_unbalanced2(a, b, M, epsilon, reg_m,
                                               method=method)
            barycenter_unbalanced(A, M, reg=epsilon, reg_m=reg_m,
                                  method=method)
    with pytest.raises(ValueError):
        for method in set(NOT_VALID_TOKENS):
            ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, reg_m,
                                              method=method)
            ot.unbalanced.sinkhorn_unbalanced2(a, b, M, epsilon, reg_m,
                                               method=method)
            barycenter_unbalanced(A, M, reg=epsilon, reg_m=reg_m,
                                  method=method)


@pytest.mark.parametrize("reg_div,regm_div", itertools.product(['kl', 'l2', 'entropy'], ['kl', 'l2']))
def test_lbfgsb_unbalanced(nx, reg_div, regm_div):

    np.random.seed(42)

    xs = np.random.randn(5, 2)
    xt = np.random.randn(6, 2)

    M = ot.dist(xs, xt)

    a = ot.unif(5)
    b = ot.unif(6)

    G, log = ot.unbalanced.lbfgsb_unbalanced(a, b, M, 1, 10, reg_div=reg_div, regm_div=regm_div, log=True, verbose=False)

    ab, bb, Mb = nx.from_numpy(a, b, M)

    Gb, log = ot.unbalanced.lbfgsb_unbalanced(ab, bb, Mb, 1, 10, reg_div=reg_div, regm_div=regm_div, log=True, verbose=False)

    np.testing.assert_allclose(G, nx.to_numpy(Gb))


@pytest.mark.parametrize("reg_div,regm_div", itertools.product(['kl', 'l2', 'entropy'], ['kl', 'l2']))
def test_lbfgsb_unbalanced_relaxation_parameters(nx, reg_div, regm_div):

    np.random.seed(42)

    xs = np.random.randn(5, 2)
    xt = np.random.randn(6, 2)

    M = ot.dist(xs, xt)

    a = ot.unif(5)
    b = ot.unif(6)

    a, b, M = nx.from_numpy(a, b, M)

    reg_m = 10
    full_list_reg_m = [reg_m, reg_m]
    full_tuple_reg_m = (reg_m, reg_m)
    tuple_reg_m, list_reg_m = (reg_m), [reg_m]
    np1_reg_m = reg_m * np.ones(1)
    np2_reg_m = reg_m * np.ones(2)

    list_options = [np1_reg_m, np2_reg_m, full_tuple_reg_m,
                    tuple_reg_m, full_list_reg_m, list_reg_m]

    G = ot.unbalanced.lbfgsb_unbalanced(a, b, M, 1, reg_m=reg_m,
                                        reg_div=reg_div, regm_div=regm_div,
                                        log=False, verbose=False)

    for opt in list_options:
        G0 = ot.unbalanced.lbfgsb_unbalanced(
            a, b, M, 1, reg_m=opt, reg_div=reg_div,
            regm_div=regm_div, log=False, verbose=False
        )

        np.testing.assert_allclose(nx.to_numpy(G), nx.to_numpy(G0), atol=1e-06)


@pytest.mark.parametrize("reg_div,regm_div", itertools.product(['kl', 'l2', 'entropy'], ['kl', 'l2']))
def test_lbfgsb_reference_measure(nx, reg_div, regm_div):

    np.random.seed(42)

    xs = np.random.randn(5, 2)
    xt = np.random.randn(6, 2)
    M = ot.dist(xs, xt)
    a = ot.unif(5)
    b = ot.unif(6)

    a, b, M = nx.from_numpy(a, b, M)
    c = a[:, None] * b[None, :]

    G, _ = ot.unbalanced.lbfgsb_unbalanced(a, b, M, reg=1, reg_m=10, c=None,
                                           reg_div=reg_div, regm_div=regm_div,
                                           log=True, verbose=False)

    G0, _ = ot.unbalanced.lbfgsb_unbalanced(a, b, M, reg=1, reg_m=10, c=c,
                                            reg_div=reg_div, regm_div=regm_div,
                                            log=True, verbose=False)

    np.testing.assert_allclose(nx.to_numpy(G), nx.to_numpy(G0), atol=1e-06)


@pytest.mark.parametrize("div", ["kl", "l2"])
def test_mm_convergence(nx, div):
    n = 100
    rng = np.random.RandomState(42)
    x = rng.randn(n, 2)
    rng = np.random.RandomState(75)
    y = rng.randn(n, 2)
    a_np = ot.utils.unif(n)
    b_np = ot.utils.unif(n)

    M = ot.dist(x, y)
    M = M / M.max()
    reg_m = 100
    a, b, M = nx.from_numpy(a_np, b_np, M)

    G, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div=div,
                                       verbose=False, log=True)
    loss = nx.to_numpy(
        ot.unbalanced.mm_unbalanced2(a, b, M, reg_m, div=div, verbose=True)
    )

    # check if the marginals come close to the true ones when large reg
    np.testing.assert_allclose(np.sum(nx.to_numpy(G), 1), a_np, atol=1e-03)
    np.testing.assert_allclose(np.sum(nx.to_numpy(G), 0), b_np, atol=1e-03)

    # check if mm_unbalanced2 returns the correct loss
    np.testing.assert_allclose(nx.to_numpy(nx.sum(G * M)), loss, atol=1e-5)

    # check in case no histogram is provided
    a_np, b_np = np.array([]), np.array([])
    a, b = nx.from_numpy(a_np, b_np)

    G_null = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div=div, verbose=False)
    np.testing.assert_allclose(nx.to_numpy(G_null), nx.to_numpy(G))

    # test when G0 is given
    G0 = ot.emd(a, b, M)
    G0_np = nx.to_numpy(G0)
    reg_m = 10000
    G = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div=div, G0=G0, verbose=False)
    np.testing.assert_allclose(G0_np, nx.to_numpy(G), atol=1e-05)


@pytest.mark.parametrize("div", ["kl", "l2"])
def test_mm_relaxation_parameters(nx, div):
    n = 100
    rng = np.random.RandomState(42)
    x = rng.randn(n, 2)
    rng = np.random.RandomState(75)
    y = rng.randn(n, 2)
    a_np = ot.utils.unif(n)
    b_np = ot.utils.unif(n)

    M = ot.dist(x, y)
    M = M / M.max()
    a, b, M = nx.from_numpy(a_np, b_np, M)

    reg = 1e-2

    reg_m = 100
    full_list_reg_m = [reg_m, reg_m]
    full_tuple_reg_m = (reg_m, reg_m)
    tuple_reg_m, list_reg_m = (reg_m), [reg_m]
    nx1_reg_m = reg_m * nx.ones(1)
    nx2_reg_m = reg_m * nx.ones(2)

    list_options = [nx1_reg_m, nx2_reg_m, full_tuple_reg_m,
                    tuple_reg_m, full_list_reg_m, list_reg_m]

    G0, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, reg=reg,
                                        div=div, verbose=False, log=True)
    loss_0 = nx.to_numpy(
        ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=reg_m, reg=reg,
                                     div=div, verbose=True)
    )

    for opt in list_options:
        G1, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=opt,
                                            reg=reg, div=div,
                                            verbose=False, log=True)
        loss_1 = nx.to_numpy(
            ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=opt,
                                         reg=reg, div=div, verbose=True)
        )

        np.testing.assert_allclose(nx.to_numpy(G0), nx.to_numpy(G1), atol=1e-05)
        np.testing.assert_allclose(loss_0, loss_1, atol=1e-5)


@pytest.mark.parametrize("div", ["kl", "l2"])
def test_mm_reference_measure(nx, div):
    n = 100
    rng = np.random.RandomState(42)
    x = rng.randn(n, 2)
    rng = np.random.RandomState(75)
    y = rng.randn(n, 2)
    a_np = ot.utils.unif(n)
    b_np = ot.utils.unif(n)

    M = ot.dist(x, y)
    M = M / M.max()
    a, b, M = nx.from_numpy(a_np, b_np, M)
    c = a[:, None] * b[None, :]

    reg = 1e-2
    reg_m = 100

    G0, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, c=None, reg=reg,
                                        div=div, verbose=False, log=True)
    loss_0 = ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=reg_m, c=None, reg=reg,
                                          div=div, verbose=True)
    loss_0 = nx.to_numpy(loss_0)

    G1, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, c=c,
                                        reg=reg, div=div,
                                        verbose=False, log=True)
    loss_1 = ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=reg_m, c=c,
                                          reg=reg, div=div, verbose=True)
    loss_1 = nx.to_numpy(loss_1)

    np.testing.assert_allclose(nx.to_numpy(G0), nx.to_numpy(G1), atol=1e-05)
    np.testing.assert_allclose(loss_0, loss_1, atol=1e-5)


def test_mm_wrong_divergence(nx):

    n = 100
    rng = np.random.RandomState(42)
    x = rng.randn(n, 2)
    rng = np.random.RandomState(75)
    y = rng.randn(n, 2)
    a_np = ot.utils.unif(n)
    b_np = ot.utils.unif(n)

    M = ot.dist(x, y)
    M = M / M.max()
    a, b, M = nx.from_numpy(a_np, b_np, M)

    reg = 1e-2
    reg_m = 100

    G0, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, reg=reg,
                                        div="kl", verbose=False, log=True)
    loss_0 = nx.to_numpy(
        ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=reg_m, reg=reg,
                                     div="kl", verbose=True)
    )

    G1, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, reg=reg,
                                        div="wrong_div", verbose=False, log=True)
    loss_1 = nx.to_numpy(
        ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=reg_m, reg=reg,
                                     div="wrong_div", verbose=True)
    )

    np.testing.assert_allclose(nx.to_numpy(G0), nx.to_numpy(G1), atol=1e-05)
    np.testing.assert_allclose(loss_0, loss_1, atol=1e-5)
