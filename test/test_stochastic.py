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
# COMPUTE TEST FOR SEMI-DUAL PROBLEM
#############################################################################

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

    G = ot.stochastic.solve_semi_dual_entropic(u, u, M, reg, "sag",
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

    G = ot.stochastic.solve_semi_dual_entropic(u, u, M, reg, "asgd",
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

    G_asgd = ot.stochastic.solve_semi_dual_entropic(u, u, M, reg, "asgd",
                                                    numItermax=nb_iter, lr=lr)
    G_sag = ot.stochastic.solve_semi_dual_entropic(u, u, M, reg, "sag",
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
    np.testing.assert_allclose(
        G_sag, G_sinkhorn, atol=1e-03)  # cf convergence sag
    np.testing.assert_allclose(
        G_asgd, G_sinkhorn, atol=1e-03)  # cf convergence asgd


#############################################################################
# COMPUTE TEST FOR DUAL PROBLEM
#############################################################################

#############################################################################
#
# TEST SGD algorithm
# ---------------------------------------------
# 2 identical discrete measures u defined on the same space with a
# regularization term, a batch_size and a number of iteration


def test_stochastic_dual_sgd():
    # test sgd
    n = 10
    reg = 1
    numItermax = 300000
    batch_size = 8
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.stochastic.solve_dual_entropic(u, u, M, reg, batch_size,
                                          numItermax=numItermax)

    # check constratints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-02)  # cf convergence sgd
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-02)  # cf convergence sgd


#############################################################################
#
# TEST Convergence SGD toward Sinkhorn's solution
# --------------------------------------------------------
# 2 identical discrete measures u defined on the same space with a
# regularization term, a batch_size and a number of iteration


def test_dual_sgd_sinkhorn():
    # test all dual algorithms
    n = 10
    reg = 1
    nb_iter = 300000
    batch_size = 8
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)
    zero = np.zeros(n)
    M = ot.dist(x, x)

    G_sgd = ot.stochastic.solve_dual_entropic(u, u, M, reg, batch_size,
                                              numItermax=nb_iter)

    G_sinkhorn = ot.sinkhorn(u, u, M, reg)

    # check constratints
    np.testing.assert_allclose(
        zero, (G_sgd - G_sinkhorn).sum(1), atol=1e-02)  # cf convergence sgd
    np.testing.assert_allclose(
        zero, (G_sgd - G_sinkhorn).sum(0), atol=1e-02)  # cf convergence sgd
    np.testing.assert_allclose(
        G_sgd, G_sinkhorn, atol=1e-02)  # cf convergence sgd
