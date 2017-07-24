

import ot
import numpy as np

# import pytest


def test_parmap():

    n = 100

    def f(i):
        return 1.0 * i * i

    a = np.arange(n)

    l1 = map(f, a)

    l2 = ot.utils.parmap(f, a)

    assert np.allclose(l1, l2)


def test_tic_toc():

    import time

    ot.tic()
    time.sleep(0.5)
    t = ot.toc()
    t2 = ot.toq()

    # test timing
    assert np.allclose(0.5, t, rtol=1e-2, atol=1e-2)

    # test toc vs toq
    assert np.allclose(t, t2, rtol=1e-2, atol=1e-2)


def test_kernel():

    n = 100

    x = np.random.randn(n, 2)

    K = ot.utils.kernel(x, x)

    # gaussian kernel  has ones on the diagonal
    assert np.allclose(np.diag(K), np.ones(n))


def test_unif():

    n = 100

    u = ot.unif(n)

    assert np.allclose(1, np.sum(u))


def test_dist():

    n = 100

    x = np.random.randn(n, 2)

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sum(np.square(x[i, :] - x[j, :]))

    D2 = ot.dist(x, x)

    # dist shoul return squared euclidean
    assert np.allclose(D, D2)
