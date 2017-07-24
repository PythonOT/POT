

import ot
import numpy as np

#import pytest


def test_doctest():

    import doctest

    # test lp solver
    doctest.testmod(ot.lp, verbose=True)

    # test bregman solver
    doctest.testmod(ot.bregman, verbose=True)


#@pytest.mark.skip(reason="Seems to be a conflict between pytest and multiprocessing")
def test_emd_multi():

    from ot.datasets import get_1D_gauss as gauss

    n = 1000  # nb bins

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
