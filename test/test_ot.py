"""Tests for main module ot """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import warnings

import numpy as np

import ot
from ot.datasets import make_1D_gauss as gauss
import pytest


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
    n = 500  # nb bins

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=20, s=5)  # m= mean, s= std

    ls = np.arange(20, 500, 20)
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

    # emd loss multipro proc with log
    ot.tic()
    emdn = ot.emd2(a, b, M, log=True, return_matrix=True)
    ot.toc('multi proc : {} s')

    for i in range(len(emdn)):
        emd = emdn[i]
        log = emd[1]
        cost = emd[0]
        check_duality_gap(a, b[:, i], M, log['G'], log['u'], log['v'], cost)
        emdn[i] = cost

    emdn = np.array(emdn)
    np.testing.assert_allclose(emd1, emdn)


def test_lp_barycenter():

    a1 = np.array([1.0, 0, 0])[:, None]
    a2 = np.array([0, 0, 1.0])[:, None]

    A = np.hstack((a1, a2))
    M = np.array([[0, 1.0, 4.0], [1.0, 0, 1.0], [4.0, 1.0, 0]])

    # obvious barycenter between two diracs
    bary0 = np.array([0, 1.0, 0])

    bary = ot.lp.barycenter(A, M, [.5, .5])

    np.testing.assert_allclose(bary, bary0, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(bary.sum(), 1)


def test_free_support_barycenter():

    measures_locations = [np.array([-1.]).reshape((1, 1)), np.array([1.]).reshape((1, 1))]
    measures_weights = [np.array([1.]), np.array([1.])]

    X_init = np.array([-12.]).reshape((1, 1))

    # obvious barycenter location between two diracs
    bar_locations = np.array([0.]).reshape((1, 1))

    X = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init)

    np.testing.assert_allclose(X, bar_locations, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not ot.lp.cvx.cvxopt, reason="No cvxopt available")
def test_lp_barycenter_cvxopt():

    a1 = np.array([1.0, 0, 0])[:, None]
    a2 = np.array([0, 0, 1.0])[:, None]

    A = np.hstack((a1, a2))
    M = np.array([[0, 1.0, 4.0], [1.0, 0, 1.0], [4.0, 1.0, 0]])

    # obvious barycenter between two diracs
    bary0 = np.array([0, 1.0, 0])

    bary = ot.lp.barycenter(A, M, [.5, .5], solver=None)

    np.testing.assert_allclose(bary, bary0, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(bary.sum(), 1)


def test_warnings():
    n = 100  # nb bins
    m = 100  # nb bins

    mean1 = 30
    mean2 = 50

    # bin positions
    x = np.arange(n, dtype=np.float64)
    y = np.arange(m, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=mean1, s=5)  # m= mean, s= std

    b = gauss(m, m=mean2, s=10)

    # loss matrix
    M = ot.dist(x.reshape((-1, 1)), y.reshape((-1, 1))) ** (1. / 2)

    print('Computing {} EMD '.format(1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        print('Computing {} EMD '.format(1))
        ot.emd(a, b, M, numItermax=1)
        assert "numItermax" in str(w[-1].message)
        assert len(w) == 1
        a[0] = 100
        print('Computing {} EMD '.format(2))
        ot.emd(a, b, M)
        assert "infeasible" in str(w[-1].message)
        assert len(w) == 2
        a[0] = -1
        print('Computing {} EMD '.format(2))
        ot.emd(a, b, M)
        assert "infeasible" in str(w[-1].message)
        assert len(w) == 3


def test_dual_variables():
    n = 500  # nb bins
    m = 600  # nb bins

    mean1 = 300
    mean2 = 400

    # bin positions
    x = np.arange(n, dtype=np.float64)
    y = np.arange(m, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=mean1, s=5)  # m= mean, s= std

    b = gauss(m, m=mean2, s=10)

    # loss matrix
    M = ot.dist(x.reshape((-1, 1)), y.reshape((-1, 1))) ** (1. / 2)

    print('Computing {} EMD '.format(1))

    # emd loss 1 proc
    ot.tic()
    G, log = ot.emd(a, b, M, log=True)
    ot.toc('1 proc : {} s')

    ot.tic()
    G2 = ot.emd(b, a, np.ascontiguousarray(M.T))
    ot.toc('1 proc : {} s')

    cost1 = (G * M).sum()
    # Check symmetry
    np.testing.assert_array_almost_equal(cost1, (M * G2.T).sum())
    # Check with closed-form solution for gaussians
    np.testing.assert_almost_equal(cost1, np.abs(mean1 - mean2))

    # Check that both cost computations are equivalent
    np.testing.assert_almost_equal(cost1, log['cost'])
    check_duality_gap(a, b, M, G, log['u'], log['v'], log['cost'])


def check_duality_gap(a, b, M, G, u, v, cost):
    cost_dual = np.vdot(a, u) + np.vdot(b, v)
    # Check that dual and primal cost are equal
    np.testing.assert_almost_equal(cost_dual, cost)

    [ind1, ind2] = np.nonzero(G)

    # Check that reduced cost is zero on transport arcs
    np.testing.assert_array_almost_equal((M - u.reshape(-1, 1) - v.reshape(1, -1))[ind1, ind2],
                                         np.zeros(ind1.size))
