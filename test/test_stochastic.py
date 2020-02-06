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
    n_samples = 15
    reg = 1
    max_iter = 30000
    rng = np.random.RandomState(0)

    X = rng.randn(n_samples, 2)
    a = ot.utils.unif(n_samples)

    M = ot.dist(X, X)

    G = ot.stochastic.solve_semi_dual_entropic(a, a, M, reg, "sag",
                                               max_iter=max_iter)

    # check constratints
    np.testing.assert_allclose(
        a, G.sum(1), atol=1e-04)  # cf convergence sag
    np.testing.assert_allclose(
        a, G.sum(0), atol=1e-04)  # cf convergence sag

    # new variable name
    Gamma_ent = ot.stochastic.solve_semi_dual_entropic(a, a, M, reg, "sag",
                                                       max_iter=max_iter)

    # check constratints
    np.testing.assert_allclose(
        a, Gamma_ent.sum(1), atol=1e-04)  # cf convergence sag
    np.testing.assert_allclose(
        a, Gamma_ent.sum(0), atol=1e-04)  # cf convergence sag


#############################################################################
#
# TEST ASGD algorithm
# ---------------------------------------------
# 2 identical discrete measures u defined on the same space with a
# regularization term, a learning rate and a number of iteration


def test_stochastic_asgd():
    # test asgd
    n_samples = 15
    reg = 1
    max_iter = 100000
    rng = np.random.RandomState(0)

    X = rng.randn(n_samples, 2)
    a = ot.utils.unif(n_samples)

    M = ot.dist(X, X)

    G = ot.stochastic.solve_semi_dual_entropic(a, a, M, reg, "asgd",
                                               max_iter=max_iter)

    # check constratints
    np.testing.assert_allclose(
        a, G.sum(1), atol=1e-03)  # cf convergence asgd
    np.testing.assert_allclose(
        a, G.sum(0), atol=1e-03)  # cf convergence asgd

    # new variable name
    Gamma_ent = ot.stochastic.solve_semi_dual_entropic(a, a, M, reg, "asgd",
                                                       max_iter=max_iter)

    # check constratints
    np.testing.assert_allclose(
        a, Gamma_ent.sum(1), atol=1e-03)  # cf convergence asgd
    np.testing.assert_allclose(
        a, Gamma_ent.sum(0), atol=1e-03)  # cf convergence asgd


#############################################################################
#
# TEST Convergence SAG and ASGD toward Sinkhorn's solution
# --------------------------------------------------------
# 2 identical discrete measures u defined on the same space with a
# regularization term, a learning rate and a number of iteration


def test_sag_asgd_sinkhorn():
    # test all algorithms
    n_samples = 15
    reg = 1
    max_iter = 100000
    rng = np.random.RandomState(0)

    X = rng.randn(n_samples, 2)
    a = ot.utils.unif(n_samples)
    M = ot.dist(X, X)

    G_asgd = ot.stochastic.solve_semi_dual_entropic(a, a, M, reg, "asgd",
                                                    max_iter=max_iter)
    G_sag = ot.stochastic.solve_semi_dual_entropic(a, a, M, reg, "sag",
                                                   max_iter=max_iter)
    G_sinkhorn = ot.sinkhorn(a, a, M, reg)

    # check constratints
    np.testing.assert_allclose(
        G_sag.sum(1), G_sinkhorn.sum(1), atol=1e-03)
    np.testing.assert_allclose(
        G_sag.sum(0), G_sinkhorn.sum(0), atol=1e-03)
    np.testing.assert_allclose(
        G_asgd.sum(1), G_sinkhorn.sum(1), atol=1e-03)
    np.testing.assert_allclose(
        G_asgd.sum(0), G_sinkhorn.sum(0), atol=1e-03)
    np.testing.assert_allclose(
        G_sag, G_sinkhorn, atol=1e-03)  # cf convergence sag
    np.testing.assert_allclose(
        G_asgd, G_sinkhorn, atol=1e-03)  # cf convergence asgd

    # new variable name
    Gamma_ent_asgd = ot.stochastic.solve_semi_dual_entropic(a, a, M, reg, "asgd",
                                                            max_iter=max_iter)
    Gamma_ent_sag = ot.stochastic.solve_semi_dual_entropic(a, a, M, reg, "sag",
                                                           max_iter=max_iter)
    Gamma_ent_sinkhorn = ot.sinkhorn(a, a, M, reg)

    # check constratints
    np.testing.assert_allclose(
        Gamma_ent_sag.sum(1), Gamma_ent_sinkhorn.sum(1), atol=1e-03)
    np.testing.assert_allclose(
        Gamma_ent_sag.sum(0), Gamma_ent_sinkhorn.sum(0), atol=1e-03)
    np.testing.assert_allclose(
        Gamma_ent_asgd.sum(1), Gamma_ent_sinkhorn.sum(1), atol=1e-03)
    np.testing.assert_allclose(
        Gamma_ent_asgd.sum(0), Gamma_ent_sinkhorn.sum(0), atol=1e-03)
    np.testing.assert_allclose(
        Gamma_ent_sag, Gamma_ent_sinkhorn, atol=1e-03)  # cf convergence sag
    np.testing.assert_allclose(
        Gamma_ent_asgd, Gamma_ent_sinkhorn, atol=1e-03)  # cf convergence asgd


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
    n_samples = 10
    reg = 1
    max_iter = 15000
    batch_size = 10
    rng = np.random.RandomState(0)

    X = rng.randn(n_samples, 2)
    a = ot.utils.unif(n_samples)

    M = ot.dist(X, X)

    G = ot.stochastic.solve_dual_entropic(a, a, M, reg, batch_size,
                                          max_iter=max_iter)

    # check constratints
    np.testing.assert_allclose(
        a, G.sum(1), atol=1e-03)  # cf convergence sgd
    np.testing.assert_allclose(
        a, G.sum(0), atol=1e-03)  # cf convergence sgd

    # new variable name
    Gamma_ent = ot.stochastic.solve_dual_entropic(a, a, M, reg, batch_size,
                                                  max_iter=max_iter)

    # check constratints
    np.testing.assert_allclose(
        a, Gamma_ent.sum(1), atol=1e-03)  # cf convergence sgd
    np.testing.assert_allclose(
        a, Gamma_ent.sum(0), atol=1e-03)  # cf convergence sgd


#############################################################################
#
# TEST Convergence SGD toward Sinkhorn's solution
# --------------------------------------------------------
# 2 identical discrete measures u defined on the same space with a
# regularization term, a batch_size and a number of iteration


def test_dual_sgd_sinkhorn():
    # test all dual algorithms
    n_samples = 10
    reg = 1
    max_iter = 15000
    batch_size = 10
    rng = np.random.RandomState(0)

# Test uniform
    X = rng.randn(n_samples, 2)
    a = ot.utils.unif(n_samples)
    M = ot.dist(X, X)

    G_sgd = ot.stochastic.solve_dual_entropic(a, a, M, reg, batch_size,
                                              max_iter=max_iter)

    G_sinkhorn = ot.sinkhorn(a, a, M, reg)

    # check constraints
    np.testing.assert_allclose(
        G_sgd.sum(1), G_sinkhorn.sum(1), atol=1e-03)
    np.testing.assert_allclose(
        G_sgd.sum(0), G_sinkhorn.sum(0), atol=1e-03)
    np.testing.assert_allclose(
        G_sgd, G_sinkhorn, atol=1e-03)  # cf convergence sgd

    # new variable name
    Gamma_ent_sgd = ot.stochastic.solve_dual_entropic(a, a, M, reg, batch_size,
                                                      max_iter=max_iter)

    Gamma_ent_sinkhorn = ot.sinkhorn(a, a, M, reg)

    # check constraints
    np.testing.assert_allclose(
        Gamma_ent_sgd.sum(1), Gamma_ent_sinkhorn.sum(1), atol=1e-03)
    np.testing.assert_allclose(
        Gamma_ent_sgd.sum(0), Gamma_ent_sinkhorn.sum(0), atol=1e-03)
    np.testing.assert_allclose(
        Gamma_ent_sgd, Gamma_ent_sinkhorn, atol=1e-03)  # cf convergence sgd

# Test gaussian
    n_samples = 30
    reg = 1
    batch_size = 30

    a = ot.datasets.make_1D_gauss(n_samples, 15, 5)  # m= mean, s= std
    b = ot.datasets.make_1D_gauss(n_samples, 15, 5)
    X_source = np.arange(n_samples, dtype=np.float64)
    Y_target = np.arange(n_samples, dtype=np.float64)
    M = ot.dist(X_source.reshape((n_samples, 1)), Y_target.reshape((n_samples, 1)))
    M /= M.max()

    G_sgd = ot.stochastic.solve_dual_entropic(a, b, M, reg, batch_size,
                                              max_iter=max_iter)

    G_sinkhorn = ot.sinkhorn(a, b, M, reg)

    # check constraints
    np.testing.assert_allclose(
        G_sgd.sum(1), G_sinkhorn.sum(1), atol=1e-03)
    np.testing.assert_allclose(
        G_sgd.sum(0), G_sinkhorn.sum(0), atol=1e-03)
    np.testing.assert_allclose(
        G_sgd, G_sinkhorn, atol=1e-03)  # cf convergence sgd

    # new variable name
    Gamma_ent_sgd = ot.stochastic.solve_dual_entropic(a, b, M, reg, batch_size,
                                                      max_iter=max_iter)

    Gamma_ent_sinkhorn = ot.sinkhorn(a, b, M, reg)

    # check constraints
    np.testing.assert_allclose(
        Gamma_ent_sgd.sum(1), Gamma_ent_sinkhorn.sum(1), atol=1e-03)
    np.testing.assert_allclose(
        Gamma_ent_sgd.sum(0), Gamma_ent_sinkhorn.sum(0), atol=1e-03)
    np.testing.assert_allclose(
        Gamma_ent_sgd, Gamma_ent_sinkhorn, atol=1e-03)  # cf convergence sgd
