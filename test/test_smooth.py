"""Tests for ot.smooth model """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np

import ot


def test_smooth_ot_dual():

    # get data
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    Gl2 = ot.smooth.smooth_ot_dual(u, u, M, 1, reg_type='l2', stopThr=1e-10)

    # check constratints
    np.testing.assert_allclose(
        u, Gl2.sum(1), atol=1e-05)  # cf convergence sinkhorn
    np.testing.assert_allclose(
        u, Gl2.sum(0), atol=1e-05)  # cf convergence sinkhorn

    # kl regyularisation
    G = ot.smooth.smooth_ot_dual(u, u, M, 1, reg_type='kl', stopThr=1e-10)

    # check constratints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn

    G2 = ot.sinkhorn(u, u, M, 1, stopThr=1e-10)
    np.testing.assert_allclose(G, G2, atol=1e-05)
