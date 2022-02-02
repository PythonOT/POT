"""Tests for module dr on Dimensionality Reduction """

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Minhui Huang <mhhuang@ucdavis.edu>
#
# License: MIT License

import numpy as np
import ot
import pytest

try:  # test if autograd and pymanopt are installed
    import ot.dr
    nogo = False
except ImportError:
    nogo = True


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_fda():

    n_samples = 90  # nb samples in source and target datasets
    np.random.seed(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif('gaussrot', n_samples)

    n_features_noise = 8

    xs = np.hstack((xs, np.random.randn(n_samples, n_features_noise)))

    p = 1

    Pfda, projfda = ot.dr.fda(xs, ys, p)

    projfda(xs)

    np.testing.assert_allclose(np.sum(Pfda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda():

    n_samples = 100  # nb samples in source and target datasets
    np.random.seed(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif('gaussrot', n_samples)

    n_features_noise = 8

    xs = np.hstack((xs, np.random.randn(n_samples, n_features_noise)))

    p = 2

    Pwda, projwda = ot.dr.wda(xs, ys, p, maxiter=10)

    projwda(xs)

    np.testing.assert_allclose(np.sum(Pwda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda_low_reg():

    n_samples = 100  # nb samples in source and target datasets
    np.random.seed(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif('gaussrot', n_samples)

    n_features_noise = 8

    xs = np.hstack((xs, np.random.randn(n_samples, n_features_noise)))

    p = 2

    Pwda, projwda = ot.dr.wda(xs, ys, p, reg=0.01, maxiter=10, sinkhorn_method='sinkhorn_log')

    projwda(xs)

    np.testing.assert_allclose(np.sum(Pwda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda_normalized():

    n_samples = 100  # nb samples in source and target datasets
    np.random.seed(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif('gaussrot', n_samples)

    n_features_noise = 8

    xs = np.hstack((xs, np.random.randn(n_samples, n_features_noise)))

    p = 2

    P0 = np.random.randn(10, p)
    P0 /= P0.sum(0, keepdims=True)

    Pwda, projwda = ot.dr.wda(xs, ys, p, maxiter=10, P0=P0, normalize=True)

    projwda(xs)

    np.testing.assert_allclose(np.sum(Pwda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_prw():
    d = 100  # Dimension
    n = 100  # Number samples
    k = 3  # Subspace dimension
    dim = 3

    def fragmented_hypercube(n, d, dim):
        assert dim <= d
        assert dim >= 1
        assert dim == int(dim)

        a = (1. / n) * np.ones(n)
        b = (1. / n) * np.ones(n)

        # First measure : uniform on the hypercube
        X = np.random.uniform(-1, 1, size=(n, d))

        # Second measure : fragmentation
        tmp_y = np.random.uniform(-1, 1, size=(n, d))
        Y = tmp_y + 2 * np.sign(tmp_y) * np.array(dim * [1] + (d - dim) * [0])
        return a, b, X, Y

    a, b, X, Y = fragmented_hypercube(n, d, dim)

    tau = 0.002
    reg = 0.2

    pi, U = ot.dr.projection_robust_wasserstein(X, Y, a, b, tau, reg=reg, k=k, maxiter=1000, verbose=1)

    U0 = np.random.randn(d, k)
    U0, _ = np.linalg.qr(U0)

    pi, U = ot.dr.projection_robust_wasserstein(X, Y, a, b, tau, U0=U0, reg=reg, k=k, maxiter=1000, verbose=1)
