"""Tests for module bregman on OT with bregman projections """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
from ot.utils import unif, dist, dist0
from ot.bregman import sinkhorn, barycenter, unmix
from ot.datasets import get_1D_gauss


def test_sinkhorn():
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = unif(n)

    M = dist(x, x)

    G = sinkhorn(u, u, M, 1, stopThr=1e-10)

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
    u = unif(n)

    M = dist(x, x)

    G, log = sinkhorn([], [], M, 1, stopThr=1e-10, verbose=True, log=True)
    # check constratints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)

    G, log = sinkhorn([], [], M, 1, stopThr=1e-10,
                      method='sinkhorn_stabilized', verbose=True, log=True)
    # check constratints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)

    G, log = sinkhorn(
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
    u = unif(n)

    M = dist(x, x)

    G0 = sinkhorn(u, u, M, 1, method='sinkhorn', stopThr=1e-10)
    Gs = sinkhorn(u, u, M, 1, method='sinkhorn_stabilized', stopThr=1e-10)
    Ges = sinkhorn(
        u, u, M, 1, method='sinkhorn_epsilon_scaling', stopThr=1e-10)
    Gerr = sinkhorn(u, u, M, 1, method='do_not_exists', stopThr=1e-10)

    # check values
    np.testing.assert_allclose(G0, Gs, atol=1e-05)
    np.testing.assert_allclose(G0, Ges, atol=1e-05)
    np.testing.assert_allclose(G0, Gerr)


def test_bary():

    n_bins = 100  # nb bins

    # Gaussian distributions
    a1 = get_1D_gauss(n_bins, m=30, s=10)  # m= mean, s= std
    a2 = get_1D_gauss(n_bins, m=40, s=10)

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T

    # loss matrix + normalization
    M = dist0(n_bins)
    M /= M.max()

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    # wasserstein
    reg = 1e-3
    bary_wass = barycenter(A, M, reg, weights)

    np.testing.assert_allclose(1, np.sum(bary_wass))

    barycenter(A, M, reg, log=True, verbose=True)


def test_unmix():

    n_bins = 50  # nb bins

    # Gaussian distributions
    a1 = get_1D_gauss(n_bins, m=20, s=10)  # m= mean, s= std
    a2 = get_1D_gauss(n_bins, m=40, s=10)

    a = get_1D_gauss(n_bins, m=30, s=10)

    # creating matrix A containing all distributions
    D = np.vstack((a1, a2)).T

    # loss matrix + normalization
    M = dist0(n_bins)
    M /= M.max()

    M0 = dist0(2)
    M0 /= M0.max()
    h0 = unif(2)

    # wasserstein
    reg = 1e-3
    um = unmix(a, D, M, M0, h0, reg, 1, alpha=0.01,)

    np.testing.assert_allclose(1, np.sum(um), rtol=1e-03, atol=1e-03)
    np.testing.assert_allclose([0.5, 0.5], um, rtol=1e-03, atol=1e-03)

    unmix(a, D, M, M0, h0, reg,
          1, alpha=0.01, log=True, verbose=True)
