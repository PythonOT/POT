

import ot
import numpy as np

# import pytest


def test_doctest():

    import doctest

    # test lp solver
    doctest.testmod(ot.lp, verbose=True)

    # test bregman solver
    doctest.testmod(ot.bregman, verbose=True)


def test_emd_emd2():
    # test emd and emd2 for simple identity
    n = 100
    np.random.seed(0)

    x = np.random.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.emd(u, u, M)

    # check G is identity
    assert np.allclose(G, np.eye(n) / n)

    w = ot.emd2(u, u, M)

    # check loss=0
    assert np.allclose(w, 0)


def test_emd2_multi():

    from ot.datasets import get_1D_gauss as gauss

    n = 1000  # nb bins
    np.random.seed(0)

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=20, s=5)  # m= mean, s= std

    ls = np.arange(20, 1000, 10)
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

    assert np.allclose(emd1, emdn)
