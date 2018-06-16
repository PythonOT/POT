"""
==========================
Stochastic test
==========================

This example is designed to test the stochatic optimization algorithms module
for descrete and semicontinous measures from the POT library.

"""

# Author: Kilian Fatras <kilian.fatras@gmail.com>
#
# License: MIT License

import numpy as np
import ot

#############################################################################
#
# TEST SAG algorithm
# ---------------------------------------------
# 2 identical discrete measures u defined on the same space with a
# regularization term, a learning rate and a number of iteration


def test_stochastic_sag():
    # test sag
    n = 15
    reg = 1
    numItermax = 300000
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.stochastic.transportation_matrix_entropic(u, u, M, reg, "sag",
                                                     numItermax=numItermax)

    # check constratints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-04)  # cf convergence sag
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-04)  # cf convergence sag


#############################################################################
#
# TEST ASGD algorithm
# ---------------------------------------------
# 2 identical discrete measures u defined on the same space with a
# regularization term, a learning rate and a number of iteration

def test_stochastic_asgd():
    # test asgd
    n = 15
    reg = 1
    numItermax = 300000
    lr = 1
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.stochastic.transportation_matrix_entropic(u, u, M, reg, "asgd",
                                                     numItermax=numItermax,
                                                     lr=lr)

    # check constratints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-03)  # cf convergence asgd
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-03)  # cf convergence asgd


#############################################################################
#
# TEST Convergence SAG and ASGD toward Sinkhorn's solution
# --------------------------------------------------------
# 2 identical discrete measures u defined on the same space with a
# regularization term, a learning rate and a number of iteration


def test_sag_asgd_sinkhorn():
    # test all algorithms
    n = 15
    reg = 1
    nb_iter = 300000
    lr = 1
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)
    zero = np.zeros(n)
    M = ot.dist(x, x)

    G_asgd = ot.stochastic.transportation_matrix_entropic(u, u, M, reg, "asgd",
                                                          numItermax=nb_iter,
                                                          lr=1)
    G_sag = ot.stochastic.transportation_matrix_entropic(u, u, M, reg, "sag",
                                                         numItermax=nb_iter)
    G_sinkhorn = ot.sinkhorn(u, u, M, reg)

    # check constratints
    np.testing.assert_allclose(
        zero, (G_sag - G_sinkhorn).sum(1), atol=1e-03)  # cf convergence sag
    np.testing.assert_allclose(
        zero, (G_sag - G_sinkhorn).sum(0), atol=1e-03)  # cf convergence sag
    np.testing.assert_allclose(
        zero, (G_asgd - G_sinkhorn).sum(1), atol=1e-03)  # cf convergence asgd
    np.testing.assert_allclose(
        zero, (G_asgd - G_sinkhorn).sum(0), atol=1e-03)  # cf convergence asgd
