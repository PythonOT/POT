"""Tests for module bregman on OT with bregman projections """

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.backend import get_backend_list
from ot.backend import torch

backend_list = get_backend_list()


def test_sinkhorn():
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.sinkhorn(u, u, M, 1, stopThr=1e-10)

    # check constratints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn


@pytest.mark.parametrize('nx', backend_list)
def test_sinkhorn_backends(nx):
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)

    M = ot.dist(x, y)

    G = ot.sinkhorn(a, a, M, 1)

    ab = nx.from_numpy(a)
    Mb = nx.from_numpy(M)

    Gb = ot.sinkhorn(ab, ab, Mb, 1)

    np.allclose(G, nx.to_numpy(Gb))


@pytest.mark.parametrize('nx', backend_list)
def test_sinkhorn2_backends(nx):
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)

    M = ot.dist(x, y)

    G = ot.sinkhorn(a, a, M, 1)

    ab = nx.from_numpy(a)
    Mb = nx.from_numpy(M)

    Gb = ot.sinkhorn2(ab, ab, Mb, 1)

    np.allclose(G, nx.to_numpy(Gb))


def test_sinkhorn2_gradients():
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)

    M = ot.dist(x, y)

    if torch:

        a1 = torch.tensor(a, requires_grad=True)
        b1 = torch.tensor(a, requires_grad=True)
        M1 = torch.tensor(M, requires_grad=True)

        val = ot.sinkhorn2(a1, b1, M1, 1)

        val.backward()

        assert a1.shape == a1.grad.shape
        assert b1.shape == b1.grad.shape
        assert M1.shape == M1.grad.shape


def test_sinkhorn_empty():
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G, log = ot.sinkhorn([], [], M, 1, stopThr=1e-10, verbose=True, log=True)
    # check constratints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)

    G, log = ot.sinkhorn([], [], M, 1, stopThr=1e-10,
                         method='sinkhorn_stabilized', verbose=True, log=True)
    # check constratints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)

    G, log = ot.sinkhorn(
        [], [], M, 1, stopThr=1e-10, method='sinkhorn_epsilon_scaling',
        verbose=True, log=True)
    # check constratints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)

    # test empty weights greenkhorn
    ot.sinkhorn([], [], M, 1, method='greenkhorn', stopThr=1e-10, log=True)


@pytest.mark.parametrize("nx", backend_list)
def test_sinkhorn_variants(nx):
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    ub = nx.from_numpy(u)
    Mb = nx.from_numpy(M)

    G = ot.sinkhorn(u, u, M, 1, method='sinkhorn', stopThr=1e-10)
    G0 = nx.to_numpy(ot.sinkhorn(ub, ub, Mb, 1, method='sinkhorn', stopThr=1e-10))
    Gs = nx.to_numpy(ot.sinkhorn(ub, ub, Mb, 1, method='sinkhorn_stabilized', stopThr=1e-10))
    Ges = nx.to_numpy(ot.sinkhorn(
        ub, ub, Mb, 1, method='sinkhorn_epsilon_scaling', stopThr=1e-10))
    G_green = nx.to_numpy(ot.sinkhorn(ub, ub, Mb, 1, method='greenkhorn', stopThr=1e-10))

    # check values
    np.testing.assert_allclose(G, G0, atol=1e-05)
    np.testing.assert_allclose(G0, Gs, atol=1e-05)
    np.testing.assert_allclose(G0, Ges, atol=1e-05)
    np.testing.assert_allclose(G0, G_green, atol=1e-5)
    print(G0, G_green)


def test_sinkhorn_variants_log():
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G0, log0 = ot.sinkhorn(u, u, M, 1, method='sinkhorn', stopThr=1e-10, log=True)
    Gs, logs = ot.sinkhorn(u, u, M, 1, method='sinkhorn_stabilized', stopThr=1e-10, log=True)
    Ges, loges = ot.sinkhorn(
        u, u, M, 1, method='sinkhorn_epsilon_scaling', stopThr=1e-10, log=True)
    G_green, loggreen = ot.sinkhorn(u, u, M, 1, method='greenkhorn', stopThr=1e-10, log=True)

    # check values
    np.testing.assert_allclose(G0, Gs, atol=1e-05)
    np.testing.assert_allclose(G0, Ges, atol=1e-05)
    np.testing.assert_allclose(G0, G_green, atol=1e-5)
    print(G0, G_green)


@pytest.mark.parametrize("nx", backend_list)
@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized"])
def test_barycenter(nx, method):
    n_bins = 100  # nb bins

    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n_bins, m=30, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n_bins, m=40, s=10)

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T

    # loss matrix + normalization
    M = ot.utils.dist0(n_bins)
    M /= M.max()

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    A = nx.from_numpy(A)
    M = nx.from_numpy(M)
    weights = nx.from_numpy(weights)

    # wasserstein
    reg = 1e-2
    bary_wass, log = ot.bregman.barycenter(A, M, reg, weights, method=method, log=True)

    np.testing.assert_allclose(1, np.sum(nx.to_numpy(bary_wass)))

    ot.bregman.barycenter(A, M, reg, log=True, verbose=True)


@pytest.mark.parametrize("nx", backend_list)
def test_barycenter_stabilization(nx):
    n_bins = 100  # nb bins

    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n_bins, m=30, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n_bins, m=40, s=10)

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T

    # loss matrix + normalization
    M = ot.utils.dist0(n_bins)
    M /= M.max()

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    Ab = nx.from_numpy(A)
    Mb = nx.from_numpy(M)
    weights_b = nx.from_numpy(weights)

    # wasserstein
    reg = 1e-2
    bar_np = ot.bregman.barycenter(A, M, reg, weights, method="sinkhorn", stopThr=1e-8, verbose=True)
    bar_stable = nx.to_numpy(ot.bregman.barycenter(
        Ab, Mb, reg, weights_b, method="sinkhorn_stabilized",
        stopThr=1e-8, verbose=True
    ))
    bar = nx.to_numpy(ot.bregman.barycenter(
        Ab, Mb, reg, weights_b, method="sinkhorn",
        stopThr=1e-8, verbose=True
    ))
    np.testing.assert_allclose(bar, bar_stable)
    np.testing.assert_allclose(bar, bar_np)


@pytest.mark.parametrize("nx", backend_list)
def test_wasserstein_bary_2d(nx):
    size = 100  # size of a square image
    a1 = np.random.randn(size, size)
    a1 += a1.min()
    a1 = a1 / np.sum(a1)
    a2 = np.random.randn(size, size)
    a2 += a2.min()
    a2 = a2 / np.sum(a2)
    # creating matrix A containing all distributions
    A = np.zeros((2, size, size))
    A[0, :, :] = a1
    A[1, :, :] = a2

    Ab = nx.from_numpy(A)

    # wasserstein
    reg = 1e-2
    bary_wass = nx.to_numpy(ot.bregman.convolutional_barycenter2d(Ab, reg))

    np.testing.assert_allclose(1, np.sum(bary_wass))

    # help in checking if log and verbose do not bug the function
    ot.bregman.convolutional_barycenter2d(A, reg, log=True, verbose=True)


@pytest.mark.parametrize("nx", backend_list)
def test_unmix(nx):
    n_bins = 50  # nb bins

    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n_bins, m=20, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n_bins, m=40, s=10)

    a = ot.datasets.make_1D_gauss(n_bins, m=30, s=10)

    # creating matrix A containing all distributions
    D = np.vstack((a1, a2)).T

    # loss matrix + normalization
    M = ot.utils.dist0(n_bins)
    M /= M.max()

    M0 = ot.utils.dist0(2)
    M0 /= M0.max()
    h0 = ot.unif(2)

    ab = nx.from_numpy(a)
    Db = nx.from_numpy(D)
    Mb = nx.from_numpy(M)
    M0b = nx.from_numpy(M0)
    h0b = nx.from_numpy(h0)

    # wasserstein
    reg = 1e-3
    um = ot.bregman.unmix(ab, Db, Mb, M0b, h0b, reg, 1, alpha=0.01, )
    um = nx.to_numpy(um)

    np.testing.assert_allclose(1, np.sum(um), rtol=1e-03, atol=1e-03)
    np.testing.assert_allclose([0.5, 0.5], um, rtol=1e-03, atol=1e-03)

    ot.bregman.unmix(ab, Db, Mb, M0b, h0b, reg,
                     1, alpha=0.01, log=True, verbose=True)


@pytest.mark.parametrize("nx", backend_list)
def test_empirical_sinkhorn(nx):
    # test sinkhorn
    n = 10
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(np.arange(n), (n, 1))
    X_t = np.reshape(np.arange(0, n), (n, 1))
    M = ot.dist(X_s, X_t)
    M_m = ot.dist(X_s, X_t, metric='minkowski')

    ab = nx.from_numpy(a)
    bb = nx.from_numpy(b)
    X_sb = nx.from_numpy(X_s)
    X_tb = nx.from_numpy(X_t)
    Mb = nx.from_numpy(M, type_as=ab)
    M_mb = nx.from_numpy(M_m, type_as=ab)

    G_sqe = nx.to_numpy(ot.bregman.empirical_sinkhorn(X_sb, X_tb, 1))
    sinkhorn_sqe = nx.to_numpy(ot.sinkhorn(ab, bb, Mb, 1))
    sinkhorn_sqe_np = ot.sinkhorn(a, b, M, 1)

    G_log, log_es = ot.bregman.empirical_sinkhorn(X_sb, X_tb, 0.1, log=True)
    G_log = nx.to_numpy(G_log)
    sinkhorn_log, log_s = ot.sinkhorn(ab, bb, Mb, 0.1, log=True)
    sinkhorn_log = nx.to_numpy(sinkhorn_log)
    sinkhorn_log_np, log_s = ot.sinkhorn(a, b, M, 0.1, log=True)

    G_m = nx.to_numpy(ot.bregman.empirical_sinkhorn(X_sb, X_tb, 1, metric='minkowski'))
    sinkhorn_m = nx.to_numpy(ot.sinkhorn(ab, bb, M_mb, 1))
    sinkhorn_m_np = ot.sinkhorn(a, b, M_m, 1)

    loss_emp_sinkhorn = nx.to_numpy(ot.bregman.empirical_sinkhorn2(X_sb, X_tb, 1))
    loss_sinkhorn = nx.to_numpy(ot.sinkhorn2(ab, bb, Mb, 1))
    loss_sinkhorn_np = ot.sinkhorn2(a, b, M, 1)

    # check constratints
    np.testing.assert_allclose(sinkhorn_sqe, sinkhorn_sqe_np, atol=1e-05)
    np.testing.assert_allclose(
        sinkhorn_sqe.sum(1), G_sqe.sum(1), atol=1e-05)  # metric sqeuclidian
    np.testing.assert_allclose(
        sinkhorn_sqe.sum(0), G_sqe.sum(0), atol=1e-05)  # metric sqeuclidian
    np.testing.assert_allclose(sinkhorn_log, sinkhorn_log_np, atol=1e-05)
    np.testing.assert_allclose(
        sinkhorn_log.sum(1), G_log.sum(1), atol=1e-05)  # log
    np.testing.assert_allclose(
        sinkhorn_log.sum(0), G_log.sum(0), atol=1e-05)  # log
    np.testing.assert_allclose(sinkhorn_m, sinkhorn_m_np, atol=1e-05)
    np.testing.assert_allclose(
        sinkhorn_m.sum(1), G_m.sum(1), atol=1e-05)  # metric euclidian
    np.testing.assert_allclose(
        sinkhorn_m.sum(0), G_m.sum(0), atol=1e-05)  # metric euclidian
    np.testing.assert_allclose(loss_sinkhorn, loss_sinkhorn_np, atol=1e-05)
    np.testing.assert_allclose(loss_emp_sinkhorn, loss_sinkhorn, atol=1e-05)


@pytest.mark.parametrize("nx", backend_list)
def test_lazy_empirical_sinkhorn(nx):
    # test sinkhorn
    n = 10
    a = ot.unif(n)
    b = ot.unif(n)
    numIterMax = 1000

    X_s = np.reshape(np.arange(n), (n, 1))
    X_t = np.reshape(np.arange(0, n), (n, 1))
    M = ot.dist(X_s, X_t)
    M_m = ot.dist(X_s, X_t, metric='minkowski')

    ab = nx.from_numpy(a)
    bb = nx.from_numpy(b)
    X_sb = nx.from_numpy(X_s)
    X_tb = nx.from_numpy(X_t)
    Mb = nx.from_numpy(M, type_as=ab)
    M_mb = nx.from_numpy(M_m, type_as=ab)

    f, g = ot.bregman.empirical_sinkhorn(X_sb, X_tb, 1, numIterMax=numIterMax, isLazy=True, batchSize=(1, 3), verbose=True)
    f, g = nx.to_numpy(f), nx.to_numpy(g)
    G_sqe = np.exp(f[:, None] + g[None, :] - M / 1)
    sinkhorn_sqe = nx.to_numpy(ot.sinkhorn(ab, bb, Mb, 1))
    sinkhorn_sqe_np = ot.sinkhorn(a, b, M, 1)

    f, g, log_es = ot.bregman.empirical_sinkhorn(X_sb, X_tb, 0.1, numIterMax=numIterMax, isLazy=True, batchSize=1, log=True)
    f, g = nx.to_numpy(f), nx.to_numpy(g)
    G_log = np.exp(f[:, None] + g[None, :] - M / 0.1)
    sinkhorn_log, log_s = ot.sinkhorn(ab, bb, Mb, 0.1, log=True)
    sinkhorn_log = nx.to_numpy(sinkhorn_log)
    sinkhorn_log_np, log_s_np = ot.sinkhorn(a, b, M, 0.1, log=True)

    f, g = ot.bregman.empirical_sinkhorn(X_sb, X_tb, 1, metric='minkowski', numIterMax=numIterMax, isLazy=True, batchSize=1)
    f, g = nx.to_numpy(f), nx.to_numpy(g)
    G_m = np.exp(f[:, None] + g[None, :] - M_m / 1)
    sinkhorn_m = nx.to_numpy(ot.sinkhorn(ab, bb, M_mb, 1))
    sinkhorn_m_np = ot.sinkhorn(a, b, M_m, 1)

    loss_emp_sinkhorn, log = ot.bregman.empirical_sinkhorn2(X_sb, X_tb, 1, numIterMax=numIterMax, isLazy=True, batchSize=1, log=True)
    loss_emp_sinkhorn = nx.to_numpy(loss_emp_sinkhorn)
    loss_sinkhorn = nx.to_numpy(ot.sinkhorn2(ab, bb, Mb, 1))
    loss_sinkhorn_np = ot.sinkhorn2(a, b, M, 1)

    # check constratints
    np.testing.assert_allclose(sinkhorn_sqe, sinkhorn_sqe_np, atol=1e-05)
    np.testing.assert_allclose(
        sinkhorn_sqe.sum(1), G_sqe.sum(1), atol=1e-05)  # metric sqeuclidian
    np.testing.assert_allclose(
        sinkhorn_sqe.sum(0), G_sqe.sum(0), atol=1e-05)  # metric sqeuclidian
    np.testing.assert_allclose(sinkhorn_log, sinkhorn_log_np, atol=1e-05)
    np.testing.assert_allclose(
        sinkhorn_log.sum(1), G_log.sum(1), atol=1e-05)  # log
    np.testing.assert_allclose(
        sinkhorn_log.sum(0), G_log.sum(0), atol=1e-05)  # log
    np.testing.assert_allclose(sinkhorn_m, sinkhorn_m_np, atol=1e-05)
    np.testing.assert_allclose(
        sinkhorn_m.sum(1), G_m.sum(1), atol=1e-05)  # metric euclidian
    np.testing.assert_allclose(
        sinkhorn_m.sum(0), G_m.sum(0), atol=1e-05)  # metric euclidian
    np.testing.assert_allclose(loss_sinkhorn, loss_sinkhorn_np, atol=1e-05)
    np.testing.assert_allclose(loss_emp_sinkhorn, loss_sinkhorn, atol=1e-05)


@pytest.mark.parametrize("nx", backend_list)
def test_empirical_sinkhorn_divergence(nx):
    # Test sinkhorn divergence
    n = 10
    a = np.linspace(1, n, n)
    a /= a.sum()
    b = ot.unif(n)
    X_s = np.reshape(np.arange(n), (n, 1))
    X_t = np.reshape(np.arange(0, n * 2, 2), (n, 1))
    M = ot.dist(X_s, X_t)
    M_s = ot.dist(X_s, X_s)
    M_t = ot.dist(X_t, X_t)

    ab = nx.from_numpy(a)
    bb = nx.from_numpy(b)
    X_sb = nx.from_numpy(X_s)
    X_tb = nx.from_numpy(X_t)
    Mb = nx.from_numpy(M, type_as=ab)
    M_sb = nx.from_numpy(M_s, type_as=ab)
    M_tb = nx.from_numpy(M_t, type_as=ab)

    emp_sinkhorn_div = nx.to_numpy(ot.bregman.empirical_sinkhorn_divergence(X_sb, X_tb, 1, a=ab, b=bb))
    sinkhorn_div = nx.to_numpy(
        ot.sinkhorn2(ab, bb, Mb, 1)
        - 1 / 2 * ot.sinkhorn2(ab, ab, M_sb, 1)
        - 1 / 2 * ot.sinkhorn2(bb, bb, M_tb, 1)
    )
    emp_sinkhorn_div_np = ot.bregman.empirical_sinkhorn_divergence(X_s, X_t, 1, a=a, b=b)

    emp_sinkhorn_div_log, log_es = ot.bregman.empirical_sinkhorn_divergence(
        X_sb, X_tb, 1, a=ab, b=bb, log=True
    )
    emp_sinkhorn_div_log = nx.to_numpy(emp_sinkhorn_div_log)
    sink_div_log_ab, log_s_ab = ot.sinkhorn2(ab, bb, Mb, 1, log=True)
    sink_div_log_a, log_s_a = ot.sinkhorn2(ab, ab, M_sb, 1, log=True)
    sink_div_log_b, log_s_b = ot.sinkhorn2(bb, bb, M_tb, 1, log=True)
    sink_div_log = sink_div_log_ab - 1 / 2 * (sink_div_log_a + sink_div_log_b)
    sink_div_log = nx.to_numpy(sink_div_log)
    sink_div_log_np = (
        ot.sinkhorn2(a, b, M, 1)
        - (1 / 2) * (ot.sinkhorn2(a, a, M_s, 1) + ot.sinkhorn2(b, b, M_t, 1))
    )
    # check constraints
    np.testing.assert_allclose(emp_sinkhorn_div, emp_sinkhorn_div_np, atol=1e-05)
    np.testing.assert_allclose(
        emp_sinkhorn_div, sinkhorn_div, atol=1e-05)  # cf conv emp sinkhorn
    np.testing.assert_allclose(sink_div_log, sink_div_log_np, atol=1e-05)
    np.testing.assert_allclose(
        emp_sinkhorn_div_log, sink_div_log, atol=1e-05)  # cf conv emp sinkhorn


@pytest.mark.parametrize("nx", backend_list)
def test_stabilized_vs_sinkhorn_multidim(nx):
    # test if stable version matches sinkhorn
    # for multidimensional inputs
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

    ab = nx.from_numpy(a)
    bb = nx.from_numpy(b)
    Mb = nx.from_numpy(M, type_as=ab)

    G_np, _ = ot.bregman.sinkhorn(a, b, M, reg=epsilon, method="sinkhorn", log=True)
    G, log = ot.bregman.sinkhorn(ab, bb, Mb, reg=epsilon,
                                 method="sinkhorn_stabilized",
                                 log=True)
    G = nx.to_numpy(G)
    G2, log2 = ot.bregman.sinkhorn(ab, bb, Mb, epsilon,
                                   method="sinkhorn", log=True)
    G2 = nx.to_numpy(G2)

    np.testing.assert_allclose(G_np, G2)
    np.testing.assert_allclose(G, G2)


def test_implemented_methods():
    IMPLEMENTED_METHODS = ['sinkhorn', 'sinkhorn_stabilized']
    ONLY_1D_methods = ['greenkhorn', 'sinkhorn_epsilon_scaling']
    NOT_VALID_TOKENS = ['foo']
    # test generalized sinkhorn for unbalanced OT barycenter
    n = 3
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = ot.utils.unif(n)
    A = rng.rand(n, 2)
    A /= A.sum(0, keepdims=True)
    M = ot.dist(x, x)
    epsilon = 1.0

    for method in IMPLEMENTED_METHODS:
        ot.bregman.sinkhorn(a, b, M, epsilon, method=method)
        ot.bregman.sinkhorn2(a, b, M, epsilon, method=method)
        ot.bregman.barycenter(A, M, reg=epsilon, method=method)
    with pytest.raises(ValueError):
        for method in set(NOT_VALID_TOKENS):
            ot.bregman.sinkhorn(a, b, M, epsilon, method=method)
            ot.bregman.sinkhorn2(a, b, M, epsilon, method=method)
            ot.bregman.barycenter(A, M, reg=epsilon, method=method)
    for method in ONLY_1D_methods:
        ot.bregman.sinkhorn(a, b, M, epsilon, method=method)
        with pytest.raises(ValueError):
            ot.bregman.sinkhorn2(a, b, M, epsilon, method=method)


@pytest.mark.parametrize("nx", backend_list)
@pytest.mark.filterwarnings("ignore:Bottleneck")
def test_screenkhorn(nx):
    # test screenkhorn
    rng = np.random.RandomState(0)
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    x = rng.randn(n, 2)
    M = ot.dist(x, x)

    ab = nx.from_numpy(a)
    bb = nx.from_numpy(b)
    Mb = nx.from_numpy(M, type_as=ab)

    # np sinkhorn
    G_sink_np = ot.sinkhorn(a, b, M, 1e-03)
    # sinkhorn
    G_sink = nx.to_numpy(ot.sinkhorn(ab, bb, Mb, 1e-03))
    # screenkhorn
    G_screen = nx.to_numpy(ot.bregman.screenkhorn(ab, bb, Mb, 1e-03, uniform=True, verbose=True))
    # check marginals
    np.testing.assert_allclose(G_sink.sum(0), G_sink_np.sum(0), atol=1e-02)
    np.testing.assert_allclose(G_sink.sum(1), G_sink_np.sum(1), atol=1e-02)
    np.testing.assert_allclose(G_sink.sum(0), G_screen.sum(0), atol=1e-02)
    np.testing.assert_allclose(G_sink.sum(1), G_screen.sum(1), atol=1e-02)


@pytest.mark.parametrize("nx", backend_list)
def test_convolutional_barycenter_non_square(nx):
    # test for image with height not equal width
    A = np.ones((2, 2, 3)) / (2 * 3)
    b = nx.to_numpy(ot.bregman.convolutional_barycenter2d(nx.from_numpy(A), 1e-03))
    np.testing.assert_allclose(np.ones((2, 3)) / (2 * 3), b, atol=1e-02)
    np.testing.assert_allclose(np.ones((2, 3)) / (2 * 3), b, atol=1e-02)
