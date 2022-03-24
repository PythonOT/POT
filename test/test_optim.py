"""Tests for module optim fro OT optimization """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import ot


def test_conditional_gradient(nx):

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

    def fb(G):
        return 0.5 * nx.sum(G ** 2)

    ab, bb, Mb = nx.from_numpy(a, b, M)

    reg = 1e-1

    G, log = ot.optim.cg(a, b, M, reg, f, df, verbose=True, log=True)
    Gb, log = ot.optim.cg(ab, bb, Mb, reg, fb, df, verbose=True, log=True)
    Gb = nx.to_numpy(Gb)

    np.testing.assert_allclose(Gb, G)
    np.testing.assert_allclose(a, Gb.sum(1))
    np.testing.assert_allclose(b, Gb.sum(0))


def test_conditional_gradient_itermax(nx):
    n = 100  # nb samples

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

    def fb(G):
        return 0.5 * nx.sum(G ** 2)

    ab, bb, Mb = nx.from_numpy(a, b, M)

    reg = 1e-1

    G, log = ot.optim.cg(a, b, M, reg, f, df, numItermaxEmd=10000,
                         verbose=True, log=True)
    Gb, log = ot.optim.cg(ab, bb, Mb, reg, fb, df, numItermaxEmd=10000,
                          verbose=True, log=True)
    Gb = nx.to_numpy(Gb)

    np.testing.assert_allclose(Gb, G)
    np.testing.assert_allclose(a, Gb.sum(1))
    np.testing.assert_allclose(b, Gb.sum(0))


def test_generalized_conditional_gradient(nx):

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

    def fb(G):
        return 0.5 * nx.sum(G ** 2)

    reg1 = 1e-3
    reg2 = 1e-1

    ab, bb, Mb = nx.from_numpy(a, b, M)

    G, log = ot.optim.gcg(a, b, M, reg1, reg2, f, df, verbose=True, log=True)
    Gb, log = ot.optim.gcg(ab, bb, Mb, reg1, reg2, fb, df, verbose=True, log=True)
    Gb = nx.to_numpy(Gb)

    np.testing.assert_allclose(Gb, G)
    np.testing.assert_allclose(a, Gb.sum(1), atol=1e-05)
    np.testing.assert_allclose(b, Gb.sum(0), atol=1e-05)


def test_solve_1d_linesearch_quad_funct():
    np.testing.assert_allclose(ot.optim.solve_1d_linesearch_quad(1, -1, 0), 0.5)
    np.testing.assert_allclose(ot.optim.solve_1d_linesearch_quad(-1, 5, 0), 0)
    np.testing.assert_allclose(ot.optim.solve_1d_linesearch_quad(-1, 0.5, 0), 1)


def test_line_search_armijo(nx):
    xk = np.array([[0.25, 0.25], [0.25, 0.25]])
    pk = np.array([[-0.25, 0.25], [0.25, -0.25]])
    gfk = np.array([[23.04273441, 23.0449082], [23.04273441, 23.0449082]])
    old_fval = -123

    xkb, pkb, gfkb = nx.from_numpy(xk, pk, gfk)

    # Should not throw an exception and return 0. for alpha
    alpha, a, b = ot.optim.line_search_armijo(
        lambda x: 1, xkb, pkb, gfkb, old_fval
    )
    alpha_np, anp, bnp = ot.optim.line_search_armijo(
        lambda x: 1, xk, pk, gfk, old_fval
    )
    assert a == anp
    assert b == bnp
    assert alpha == 0.

    # check line search armijo
    def f(x):
        return nx.sum((x - 5.0) ** 2)

    def grad(x):
        return 2 * (x - 5.0)

    xk = nx.from_numpy(np.array([[[-5.0, -5.0]]]))
    pk = nx.from_numpy(np.array([[[100.0, 100.0]]]))
    gfk = grad(xk)
    old_fval = f(xk)

    # chech the case where the optimum is on the direction
    alpha, _, _ = ot.optim.line_search_armijo(f, xk, pk, gfk, old_fval)
    np.testing.assert_allclose(alpha, 0.1)

    # check the case where the direction is not far enough
    pk = nx.from_numpy(np.array([[[3.0, 3.0]]]))
    alpha, _, _ = ot.optim.line_search_armijo(f, xk, pk, gfk, old_fval, alpha0=1.0)
    np.testing.assert_allclose(alpha, 1.0)

    # check the case where checking the wrong direction
    alpha, _, _ = ot.optim.line_search_armijo(f, xk, -pk, gfk, old_fval)
    assert alpha <= 0

    # check the case where the point is not a vector
    xk = nx.from_numpy(np.array(-5.0))
    pk = nx.from_numpy(np.array(100.0))
    gfk = grad(xk)
    old_fval = f(xk)
    alpha, _, _ = ot.optim.line_search_armijo(f, xk, pk, gfk, old_fval)
    np.testing.assert_allclose(alpha, 0.1)
