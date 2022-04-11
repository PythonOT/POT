"""Tests for module Unbalanced OT with entropy regularization"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Laetitia Chapel <laetitia.chapel@univ-ubs.fr>
#
# License: MIT License

import numpy as np
import ot
import pytest
from ot.unbalanced import barycenter_unbalanced


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized"])
def test_unbalanced_convergence(nx, method):
    # test generalized sinkhorn for unbalanced OT
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = ot.utils.unif(n) * 1.5

    M = ot.dist(x, x)
    epsilon = 1.
    reg_m = 1.

    a, b, M = nx.from_numpy(a, b, M)

    G, log = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=epsilon,
                                               reg_m=reg_m,
                                               method=method,
                                               log=True,
                                               verbose=True)
    loss = nx.to_numpy(ot.unbalanced.sinkhorn_unbalanced2(
        a, b, M, epsilon, reg_m, method=method, verbose=True
    ))
    # check fixed point equations
    # in log-domain
    fi = reg_m / (reg_m + epsilon)
    logb = nx.log(b + 1e-16)
    loga = nx.log(a + 1e-16)
    logKtu = nx.logsumexp(log["logu"][None, :] - M.T / epsilon, axis=1)
    logKv = nx.logsumexp(log["logv"][None, :] - M / epsilon, axis=1)

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
        a, b, M, reg=epsilon, reg_m=reg_m, method=method, verbose=True
    )
    G_np = ot.unbalanced.sinkhorn_unbalanced(
        a_np, b_np, M_np, reg=epsilon, reg_m=reg_m, method=method, verbose=True
    )
    np.testing.assert_allclose(G_np, nx.to_numpy(G))


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized"])
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

    loss, log = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=epsilon,
                                                  reg_m=reg_m,
                                                  method=method,
                                                  log=True,
                                                  verbose=True)
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

    assert len(loss) == b.shape[1]


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


def test_mm_convergence(nx):
    n = 100
    rng = np.random.RandomState(42)
    x = rng.randn(n, 2)
    rng = np.random.RandomState(75)
    y = rng.randn(n, 2)
    a = ot.utils.unif(n)
    b = ot.utils.unif(n)

    M = ot.dist(x, y)
    M = M / M.max()
    reg_m = 100
    a, b, M = nx.from_numpy(a, b, M)

    G_kl, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div='kl',
                                          verbose=True, log=True)
    loss_kl = nx.to_numpy(ot.unbalanced.mm_unbalanced2(
                          a, b, M, reg_m, div='kl', verbose=True))
    G_l2, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div='l2',
                                          verbose=False, log=True)

    # check if the marginals come close to the true ones when large reg
    np.testing.assert_allclose(np.sum(nx.to_numpy(G_kl), 1), a, atol=1e-03)
    np.testing.assert_allclose(np.sum(nx.to_numpy(G_kl), 0), b, atol=1e-03)
    np.testing.assert_allclose(np.sum(nx.to_numpy(G_l2), 1), a, atol=1e-03)
    np.testing.assert_allclose(np.sum(nx.to_numpy(G_l2), 0), b, atol=1e-03)

    # check if mm_unbalanced2 returns the correct loss
    np.testing.assert_allclose(nx.to_numpy(nx.sum(G_kl * M)), loss_kl,
                               atol=1e-5)

    # check in case no histogram is provided
    a_np, b_np = np.array([]), np.array([])
    a, b = nx.from_numpy(a_np, b_np)

    G_kl_null = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div='kl')
    G_l2_null = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div='l2')
    np.testing.assert_allclose(G_kl_null, G_kl)
    np.testing.assert_allclose(G_l2_null, G_l2)

    # test when G0 is given
    G0 = ot.emd(a, b, M)
    reg_m = 10000
    G_kl = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div='kl', G0=G0)
    G_l2 = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div='l2', G0=G0)
    np.testing.assert_allclose(G0, G_kl, atol=1e-05)
    np.testing.assert_allclose(G0, G_l2, atol=1e-05)
