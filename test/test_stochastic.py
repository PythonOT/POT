"""
==========================
Stochastic test
==========================

This example is designed to test the stochatic optimization algorithms module
for descrete and semicontinous measures from the POT library.

"""

# Authors: Kilian Fatras <kilian.fatras@gmail.com>
#          Rémi Flamary <remi.flamary@polytechnique.edu>
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
    n = 10
    reg = 1
    numItermax = 30000
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.stochastic.solve_semi_dual_entropic(u, u, M, reg, "sag",
                                               numItermax=numItermax)

    # check constraints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-03)  # cf convergence sag
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-03)  # cf convergence sag


#############################################################################
#
# TEST ASGD algorithm
# ---------------------------------------------
# 2 identical discrete measures u defined on the same space with a
# regularization term, a learning rate and a number of iteration


def test_stochastic_asgd():
    # test asgd
    n = 10
    reg = 1
    numItermax = 10000
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G, log = ot.stochastic.solve_semi_dual_entropic(u, u, M, reg, "asgd",
                                                    numItermax=numItermax, log=True)

    # check constraints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-02)  # cf convergence asgd
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-02)  # cf convergence asgd


#############################################################################
#
# TEST Convergence SAG and ASGD toward Sinkhorn's solution
# --------------------------------------------------------
# 2 identical discrete measures u defined on the same space with a
# regularization term, a learning rate and a number of iteration


def test_sag_asgd_sinkhorn():
    # test all algorithms
    n = 10
    reg = 1
    nb_iter = 10000
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)
    M = ot.dist(x, x)

    G_asgd = ot.stochastic.solve_semi_dual_entropic(u, u, M, reg, "asgd",
                                                    numItermax=nb_iter)
    G_sag = ot.stochastic.solve_semi_dual_entropic(u, u, M, reg, "sag",
                                                   numItermax=nb_iter)
    G_sinkhorn = ot.sinkhorn(u, u, M, reg)

    # check constraints
    np.testing.assert_allclose(
        G_sag.sum(1), G_sinkhorn.sum(1), atol=1e-02)
    np.testing.assert_allclose(
        G_sag.sum(0), G_sinkhorn.sum(0), atol=1e-02)
    np.testing.assert_allclose(
        G_asgd.sum(1), G_sinkhorn.sum(1), atol=1e-02)
    np.testing.assert_allclose(
        G_asgd.sum(0), G_sinkhorn.sum(0), atol=1e-02)
    np.testing.assert_allclose(
        G_sag, G_sinkhorn, atol=1e-02)  # cf convergence sag
    np.testing.assert_allclose(
        G_asgd, G_sinkhorn, atol=1e-02)  # cf convergence asgd


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
    numItermax = 5000
    batch_size = 10
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G, log = ot.stochastic.solve_dual_entropic(u, u, M, reg, batch_size,
                                               numItermax=numItermax, log=True)

    # check constraints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-03)  # cf convergence sgd
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-03)  # cf convergence sgd


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
    nb_iter = 5000
    batch_size = 10
    rng = np.random.RandomState(0)

# Test uniform
    x = rng.randn(n, 2)
    u = ot.utils.unif(n)
    M = ot.dist(x, x)

    G_sgd = ot.stochastic.solve_dual_entropic(u, u, M, reg, batch_size,
                                              numItermax=nb_iter)

    G_sinkhorn = ot.sinkhorn(u, u, M, reg)

    # check constraints
    np.testing.assert_allclose(
        G_sgd.sum(1), G_sinkhorn.sum(1), atol=1e-02)
    np.testing.assert_allclose(
        G_sgd.sum(0), G_sinkhorn.sum(0), atol=1e-02)
    np.testing.assert_allclose(
        G_sgd, G_sinkhorn, atol=1e-02)  # cf convergence sgd

# Test gaussian
    n = 30
    reg = 1
    batch_size = 30

    a = ot.datasets.make_1D_gauss(n, 15, 5)  # m= mean, s= std
    b = ot.datasets.make_1D_gauss(n, 15, 5)
    X_source = np.arange(n, dtype=np.float64)
    Y_target = np.arange(n, dtype=np.float64)
    M = ot.dist(X_source.reshape((n, 1)), Y_target.reshape((n, 1)))
    M /= M.max()

    G_sgd = ot.stochastic.solve_dual_entropic(a, b, M, reg, batch_size,
                                              numItermax=nb_iter)

    G_sinkhorn = ot.sinkhorn(a, b, M, reg)

    # check constraints
    np.testing.assert_allclose(
        G_sgd.sum(1), G_sinkhorn.sum(1), atol=1e-03)
    np.testing.assert_allclose(
        G_sgd.sum(0), G_sinkhorn.sum(0), atol=1e-03)
    np.testing.assert_allclose(
        G_sgd, G_sinkhorn, atol=1e-03)  # cf convergence sgd


def test_loss_dual_entropic(nx):

    nx.seed(0)

    xs = nx.randn(50, 2)
    xt = nx.randn(40, 2) + 2

    ws = nx.rand(50)
    ws = ws / nx.sum(ws)

    wt = nx.rand(40)
    wt = wt / nx.sum(wt)

    u = nx.randn(50)
    v = nx.randn(40)

    def metric(x, y):
        return -nx.dot(x, y.T)

    ot.stochastic.loss_dual_entropic(u, v, xs, xt)

    ot.stochastic.loss_dual_entropic(u, v, xs, xt, ws=ws, wt=wt, metric=metric)


def test_plan_dual_entropic(nx):

    nx.seed(0)

    xs = nx.randn(50, 2)
    xt = nx.randn(40, 2) + 2

    ws = nx.rand(50)
    ws = ws / nx.sum(ws)

    wt = nx.rand(40)
    wt = wt / nx.sum(wt)

    u = nx.randn(50)
    v = nx.randn(40)

    def metric(x, y):
        return -nx.dot(x, y.T)

    G1 = ot.stochastic.plan_dual_entropic(u, v, xs, xt)

    assert np.all(nx.to_numpy(G1) >= 0)
    assert G1.shape[0] == 50
    assert G1.shape[1] == 40

    G2 = ot.stochastic.plan_dual_entropic(u, v, xs, xt, ws=ws, wt=wt, metric=metric)

    assert np.all(nx.to_numpy(G2) >= 0)
    assert G2.shape[0] == 50
    assert G2.shape[1] == 40


def test_loss_dual_quadratic(nx):

    nx.seed(0)

    xs = nx.randn(50, 2)
    xt = nx.randn(40, 2) + 2

    ws = nx.rand(50)
    ws = ws / nx.sum(ws)

    wt = nx.rand(40)
    wt = wt / nx.sum(wt)

    u = nx.randn(50)
    v = nx.randn(40)

    def metric(x, y):
        return -nx.dot(x, y.T)

    ot.stochastic.loss_dual_quadratic(u, v, xs, xt)

    ot.stochastic.loss_dual_quadratic(u, v, xs, xt, ws=ws, wt=wt, metric=metric)


def test_plan_dual_quadratic(nx):

    nx.seed(0)

    xs = nx.randn(50, 2)
    xt = nx.randn(40, 2) + 2

    ws = nx.rand(50)
    ws = ws / nx.sum(ws)

    wt = nx.rand(40)
    wt = wt / nx.sum(wt)

    u = nx.randn(50)
    v = nx.randn(40)

    def metric(x, y):
        return -nx.dot(x, y.T)

    G1 = ot.stochastic.plan_dual_quadratic(u, v, xs, xt)

    assert np.all(nx.to_numpy(G1) >= 0)
    assert G1.shape[0] == 50
    assert G1.shape[1] == 40

    G2 = ot.stochastic.plan_dual_quadratic(u, v, xs, xt, ws=ws, wt=wt, metric=metric)

    assert np.all(nx.to_numpy(G2) >= 0)
    assert G2.shape[0] == 50
    assert G2.shape[1] == 40
