import ot
import numpy as np
import pytest

try:  # test if cudamat installed
    import ot.dr
    nogo = False
except ImportError:
    nogo = True


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_fda():

    n = 100  # nb samples in source and target datasets
    nz = 0.2
    np.random.seed(0)

    # generate circle dataset
    t = np.random.rand(n) * 2 * np.pi
    ys = np.floor((np.arange(n) * 1.0 / n * 3)) + 1
    xs = np.concatenate(
        (np.cos(t).reshape((-1, 1)), np.sin(t).reshape((-1, 1))), 1)
    xs = xs * ys.reshape(-1, 1) + nz * np.random.randn(n, 2)

    nbnoise = 8

    xs = np.hstack((xs, np.random.randn(n, nbnoise)))

    p = 2

    Pfda, projfda = ot.dr.fda(xs, ys, p)

    projfda(xs)

    assert np.allclose(np.sum(Pfda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda():

    n = 100  # nb samples in source and target datasets
    nz = 0.2
    np.random.seed(0)

    # generate circle dataset
    t = np.random.rand(n) * 2 * np.pi
    ys = np.floor((np.arange(n) * 1.0 / n * 3)) + 1
    xs = np.concatenate(
        (np.cos(t).reshape((-1, 1)), np.sin(t).reshape((-1, 1))), 1)
    xs = xs * ys.reshape(-1, 1) + nz * np.random.randn(n, 2)

    nbnoise = 8

    xs = np.hstack((xs, np.random.randn(n, nbnoise)))

    p = 2

    Pwda, projwda = ot.dr.wda(xs, ys, p, maxiter=10)

    projwda(xs)

    assert np.allclose(np.sum(Pwda**2, 0), np.ones(p))
