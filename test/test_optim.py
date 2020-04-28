"""Tests for module optim fro OT optimization """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import ot


def test_conditional_gradient():

    n_bins = 100  # nb bins
    np.random.seed(0)
    # bin positions
    x = np.arange(n_bins, dtype=np.float64)

    # Gaussian distributions
    a = ot.datasets.make_1D_gauss(n_bins, m=20, s=5)  # m= mean, s= std
    b = ot.datasets.make_1D_gauss(n_bins, m=60, s=10)

    # loss matrix
    M = ot.dist(x.reshape((n_bins, 1)), x.reshape((n_bins, 1)))
    M /= M.max()

    def f(G):
        return 0.5 * np.sum(G**2)

    def df(G):
        return G

    reg = 1e-1

    G, log = ot.optim.cg(a, b, M, reg, f, df, verbose=True, log=True)

    np.testing.assert_allclose(a, G.sum(1))
    np.testing.assert_allclose(b, G.sum(0))


def test_conditional_gradient2():
    n = 1000  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([4, 4])
    cov_t = np.array([[1, -.8], [-.8, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
    xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

    a, b = np.ones((n,)) / n, np.ones((n,)) / n

    # loss matrix
    M = ot.dist(xs, xt)
    M /= M.max()

    def f(G):
        return 0.5 * np.sum(G**2)

    def df(G):
        return G

    reg = 1e-1

    G, log = ot.optim.cg(a, b, M, reg, f, df, numItermaxEmd=200000,
                         verbose=True, log=True)

    np.testing.assert_allclose(a, G.sum(1))
    np.testing.assert_allclose(b, G.sum(0))


def test_generalized_conditional_gradient():

    n_bins = 100  # nb bins
    np.random.seed(0)
    # bin positions
    x = np.arange(n_bins, dtype=np.float64)

    # Gaussian distributions
    a = ot.datasets.make_1D_gauss(n_bins, m=20, s=5)  # m= mean, s= std
    b = ot.datasets.make_1D_gauss(n_bins, m=60, s=10)

    # loss matrix
    M = ot.dist(x.reshape((n_bins, 1)), x.reshape((n_bins, 1)))
    M /= M.max()

    def f(G):
        return 0.5 * np.sum(G**2)

    def df(G):
        return G

    reg1 = 1e-3
    reg2 = 1e-1

    G, log = ot.optim.gcg(a, b, M, reg1, reg2, f, df, verbose=True, log=True)

    np.testing.assert_allclose(a, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(b, G.sum(0), atol=1e-05)


def test_solve_1d_linesearch_quad_funct():
    np.testing.assert_allclose(ot.optim.solve_1d_linesearch_quad(1, -1, 0), 0.5)
    np.testing.assert_allclose(ot.optim.solve_1d_linesearch_quad(-1, 5, 0), 0)
    np.testing.assert_allclose(ot.optim.solve_1d_linesearch_quad(-1, 0.5, 0), 1)
