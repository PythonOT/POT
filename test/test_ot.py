"""Tests for main module ot """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np

import ot
from ot.datasets import get_1D_gauss as gauss


def test_doctest():
    import doctest

    # test lp solver
    doctest.testmod(ot.lp, verbose=True)

    # test bregman solver
    doctest.testmod(ot.bregman, verbose=True)


def test_emd_emd2():
    # test emd and emd2 for simple identity
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.emd(u, u, M)

    # check G is identity
    np.testing.assert_allclose(G, np.eye(n) / n)
    # check constratints
    np.testing.assert_allclose(u, G.sum(1))  # cf convergence sinkhorn
    np.testing.assert_allclose(u, G.sum(0))  # cf convergence sinkhorn

    w = ot.emd2(u, u, M)
    # check loss=0
    np.testing.assert_allclose(w, 0)


def test_emd_empty():
    # test emd and emd2 for simple identity
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.emd([], [], M)

    # check G is identity
    np.testing.assert_allclose(G, np.eye(n) / n)
    # check constratints
    np.testing.assert_allclose(u, G.sum(1))  # cf convergence sinkhorn
    np.testing.assert_allclose(u, G.sum(0))  # cf convergence sinkhorn

    w = ot.emd2([], [], M)
    # check loss=0
    np.testing.assert_allclose(w, 0)


def test_emd2_multi():
    n = 1000  # nb bins

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=20, s=5)  # m= mean, s= std

    ls = np.arange(20, 1000, 20)
    nb = len(ls)
    b = np.zeros((n, nb))
    for i in range(nb):
        b[:, i] = gauss(n, m=ls[i], s=10)

    # loss matrix
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
    # M/=M.max()

    print('Computing {} EMD '.format(nb))

    # emd loss 1 proc
    ot.tic()
    emd1 = ot.emd2(a, b, M, 1)
    ot.toc('1 proc : {} s')

    # emd loss multipro proc
    ot.tic()
    emdn = ot.emd2(a, b, M)
    ot.toc('multi proc : {} s')

    np.testing.assert_allclose(emd1, emdn)


def test_dual_variables():
    # %% parameters

    n = 5000  # nb bins
    m = 6000  # nb bins

    mean1 = 1000
    mean2 = 1100

    # bin positions
    x = np.arange(n, dtype=np.float64)
    y = np.arange(m, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=mean1, s=5)  # m= mean, s= std

    b = gauss(m, m=mean2, s=10)

    # loss matrix
    M = ot.dist(x.reshape((-1, 1)), y.reshape((-1, 1))) ** (1. / 2)
    # M/=M.max()

    # %%

    print('Computing {} EMD '.format(1))

    # emd loss 1 proc
    ot.tic()
    G, alpha, beta = ot.emd(a, b, M, dual_variables=True)
    ot.toc('1 proc : {} s')

    cost1 = (G * M).sum()
    cost_dual = np.vdot(a, alpha) + np.vdot(b, beta)

    # emd loss 1 proc
    ot.tic()
    cost_emd2 = ot.emd2(a, b, M)
    ot.toc('1 proc : {} s')

    ot.tic()
    G2 = ot.emd(b, a, np.ascontiguousarray(M.T))
    ot.toc('1 proc : {} s')

    cost2 = (G2 * M.T).sum()

    # Check that both cost computations are equivalent
    np.testing.assert_almost_equal(cost1, cost_emd2)
    # Check that dual and primal cost are equal
    np.testing.assert_almost_equal(cost1, cost_dual)
    # Check symmetry
    np.testing.assert_almost_equal(cost1, cost2)
    # Check with closed-form solution for gaussians
    np.testing.assert_almost_equal(cost1, np.abs(mean1 - mean2))

    [ind1, ind2] = np.nonzero(G)

    # Check that reduced cost is zero on transport arcs
    np.testing.assert_array_almost_equal((M - alpha.reshape(-1, 1) - beta.reshape(1, -1))[ind1, ind2],
                                         np.zeros(ind1.size))
