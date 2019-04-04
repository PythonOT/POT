"""Tests for module bregman on OT with bregman projections """

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#
# License: MIT License

import numpy as np
import ot


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


def test_sinkhorn_variants():
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G0 = ot.sinkhorn(u, u, M, 1, method='sinkhorn', stopThr=1e-10)
    Gs = ot.sinkhorn(u, u, M, 1, method='sinkhorn_stabilized', stopThr=1e-10)
    Ges = ot.sinkhorn(
        u, u, M, 1, method='sinkhorn_epsilon_scaling', stopThr=1e-10)
    Gerr = ot.sinkhorn(u, u, M, 1, method='do_not_exists', stopThr=1e-10)
    G_green = ot.sinkhorn(u, u, M, 1, method='greenkhorn', stopThr=1e-10)

    # check values
    np.testing.assert_allclose(G0, Gs, atol=1e-05)
    np.testing.assert_allclose(G0, Ges, atol=1e-05)
    np.testing.assert_allclose(G0, Gerr)
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
    Gerr, logerr = ot.sinkhorn(u, u, M, 1, method='do_not_exists', stopThr=1e-10, log=True)
    G_green, loggreen = ot.sinkhorn(u, u, M, 1, method='greenkhorn', stopThr=1e-10, log=True)

    # check values
    np.testing.assert_allclose(G0, Gs, atol=1e-05)
    np.testing.assert_allclose(G0, Ges, atol=1e-05)
    np.testing.assert_allclose(G0, Gerr)
    np.testing.assert_allclose(G0, G_green, atol=1e-5)
    print(G0, G_green)


def test_bary():

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

    # wasserstein
    reg = 1e-3
    bary_wass = ot.bregman.barycenter(A, M, reg, weights)

    np.testing.assert_allclose(1, np.sum(bary_wass))

    ot.bregman.barycenter(A, M, reg, log=True, verbose=True)


def test_wasserstein_bary_2d():

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

    # wasserstein
    reg = 1e-2
    bary_wass = ot.bregman.convolutional_barycenter2d(A, reg)

    np.testing.assert_allclose(1, np.sum(bary_wass))

    # help in checking if log and verbose do not bug the function
    ot.bregman.convolutional_barycenter2d(A, reg, log=True, verbose=True)


def test_unmix():

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

    # wasserstein
    reg = 1e-3
    um = ot.bregman.unmix(a, D, M, M0, h0, reg, 1, alpha=0.01,)

    np.testing.assert_allclose(1, np.sum(um), rtol=1e-03, atol=1e-03)
    np.testing.assert_allclose([0.5, 0.5], um, rtol=1e-03, atol=1e-03)

    ot.bregman.unmix(a, D, M, M0, h0, reg,
                     1, alpha=0.01, log=True, verbose=True)


def test_empirical_sinkhorn():
    # test sinkhorn
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(np.arange(n), (n, 1))
    X_t = np.reshape(np.arange(0, n), (n, 1))
    M = ot.dist(X_s, X_t)
    M_m = ot.dist(X_s, X_t, metric='minkowski')

    G_sqe = ot.bregman.empirical_sinkhorn(X_s, X_t, 1)
    sinkhorn_sqe = ot.sinkhorn(a, b, M, 1)

    G_log, log_es = ot.bregman.empirical_sinkhorn(X_s, X_t, 0.1, log=True)
    sinkhorn_log, log_s = ot.sinkhorn(a, b, M, 0.1, log=True)

    G_m = ot.bregman.empirical_sinkhorn(X_s, X_t, 1, metric='minkowski')
    sinkhorn_m = ot.sinkhorn(a, b, M_m, 1)

    loss_emp_sinkhorn = ot.bregman.empirical_sinkhorn2(X_s, X_t, 1)
    loss_sinkhorn = ot.sinkhorn2(a, b, M, 1)

    # check constratints
    np.testing.assert_allclose(
        sinkhorn_sqe.sum(1), G_sqe.sum(1), atol=1e-05)  # metric sqeuclidian
    np.testing.assert_allclose(
        sinkhorn_sqe.sum(0), G_sqe.sum(0), atol=1e-05)  # metric sqeuclidian
    np.testing.assert_allclose(
        sinkhorn_log.sum(1), G_log.sum(1), atol=1e-05)  # log
    np.testing.assert_allclose(
        sinkhorn_log.sum(0), G_log.sum(0), atol=1e-05)  # log
    np.testing.assert_allclose(
        sinkhorn_m.sum(1), G_m.sum(1), atol=1e-05)  # metric euclidian
    np.testing.assert_allclose(
        sinkhorn_m.sum(0), G_m.sum(0), atol=1e-05)  # metric euclidian
    np.testing.assert_allclose(loss_emp_sinkhorn, loss_sinkhorn, atol=1e-05)


def test_empirical_sinkhorn_divergence():
    #Test sinkhorn divergence
    n = 10
    a = ot.unif(n)
    b = ot.unif(n)
    X_s = np.reshape(np.arange(n), (n, 1))
    X_t = np.reshape(np.arange(0, n * 2, 2), (n, 1))
    M = ot.dist(X_s, X_t)
    M_s = ot.dist(X_s, X_s)
    M_t = ot.dist(X_t, X_t)

    emp_sinkhorn_div = ot.bregman.empirical_sinkhorn_divergence(X_s, X_t, 1)
    sinkhorn_div = (ot.sinkhorn2(a, b, M, 1) - 1 / 2 * ot.sinkhorn2(a, a, M_s, 1) - 1 / 2 * ot.sinkhorn2(b, b, M_t, 1))

    emp_sinkhorn_div_log, log_es = ot.bregman.empirical_sinkhorn_divergence(X_s, X_t, 1, log=True)
    sink_div_log_ab, log_s_ab = ot.sinkhorn2(a, b, M, 1, log=True)
    sink_div_log_a, log_s_a = ot.sinkhorn2(a, a, M_s, 1, log=True)
    sink_div_log_b, log_s_b = ot.sinkhorn2(b, b, M_t, 1, log=True)
    sink_div_log = sink_div_log_ab - 1 / 2 * (sink_div_log_a + sink_div_log_b)

    # check constratints
    np.testing.assert_allclose(
        emp_sinkhorn_div, sinkhorn_div, atol=1e-05)  # cf conv emp sinkhorn
    np.testing.assert_allclose(
        emp_sinkhorn_div_log, sink_div_log, atol=1e-05)  # cf conv emp sinkhorn
