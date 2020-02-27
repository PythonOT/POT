"""Tests for module gpu for gpu acceleration """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import ot
import pytest

try:  # test if cudamat installed
    import ot.gpu
    nogpu = False
except ImportError:
    nogpu = True


@pytest.mark.skipif(nogpu, reason="No GPU available")
def test_gpu_old_doctests():
    a = [.5, .5]
    b = [.5, .5]
    M = [[0., 1.], [1., 0.]]
    G = ot.sinkhorn(a, b, M, 1)
    np.testing.assert_allclose(G, np.array([[0.36552929, 0.13447071],
                                            [0.13447071, 0.36552929]]))


@pytest.mark.skipif(nogpu, reason="No GPU available")
def test_gpu_dist():

    rng = np.random.RandomState(0)

    for n_samples in [50, 100, 500, 1000]:
        print(n_samples)
        a = rng.rand(n_samples // 4, 100)
        b = rng.rand(n_samples, 100)

        M = ot.dist(a.copy(), b.copy())
        M2 = ot.gpu.dist(a.copy(), b.copy())

        np.testing.assert_allclose(M, M2, rtol=1e-10)

        M2 = ot.gpu.dist(a.copy(), b.copy(), metric='euclidean', to_numpy=False)

        # check raise not implemented wrong metric
        with pytest.raises(NotImplementedError):
            M2 = ot.gpu.dist(a.copy(), b.copy(), metric='cityblock', to_numpy=False)


@pytest.mark.skipif(nogpu, reason="No GPU available")
def test_gpu_sinkhorn():

    rng = np.random.RandomState(0)

    for n_samples in [50, 100, 500, 1000]:
        a = rng.rand(n_samples // 4, 100)
        b = rng.rand(n_samples, 100)

        wa = ot.unif(n_samples // 4)
        wb = ot.unif(n_samples)

        wb2 = np.random.rand(n_samples, 20)
        wb2 /= wb2.sum(0, keepdims=True)

        M = ot.dist(a.copy(), b.copy())
        M2 = ot.gpu.dist(a.copy(), b.copy(), to_numpy=False)

        reg = 1

        G = ot.sinkhorn(wa, wb, M, reg)
        G1 = ot.gpu.sinkhorn(wa, wb, M, reg)

        np.testing.assert_allclose(G1, G, rtol=1e-10)

        # run all on gpu
        ot.gpu.sinkhorn(wa, wb, M2, reg, to_numpy=False, log=True)

        # run sinkhorn for multiple targets
        ot.gpu.sinkhorn(wa, wb2, M2, reg, to_numpy=False, log=True)


@pytest.mark.skipif(nogpu, reason="No GPU available")
def test_gpu_sinkhorn_lpl1():

    rng = np.random.RandomState(0)

    for n_samples in [50, 100, 500]:
        print(n_samples)
        a = rng.rand(n_samples // 4, 100)
        labels_a = np.random.randint(10, size=(n_samples // 4))
        b = rng.rand(n_samples, 100)

        wa = ot.unif(n_samples // 4)
        wb = ot.unif(n_samples)

        M = ot.dist(a.copy(), b.copy())
        M2 = ot.gpu.dist(a.copy(), b.copy(), to_numpy=False)

        reg = 1

        G = ot.da.sinkhorn_lpl1_mm(wa, labels_a, wb, M, reg)
        G1 = ot.gpu.da.sinkhorn_lpl1_mm(wa, labels_a, wb, M, reg)

        np.testing.assert_allclose(G1, G, rtol=1e-10)

        ot.gpu.da.sinkhorn_lpl1_mm(wa, labels_a, wb, M2, reg, to_numpy=False, log=True)
