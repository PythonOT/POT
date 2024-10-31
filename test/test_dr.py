"""Tests for module dr on Dimensionality Reduction"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Minhui Huang <mhhuang@ucdavis.edu>
#         Antoine Collas <antoine.collas@inria.fr>
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
    rng = np.random.RandomState(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif("gaussrot", n_samples, random_state=rng)

    n_features_noise = 8

    xs = np.hstack((xs, rng.randn(n_samples, n_features_noise)))

    p = 1

    Pfda, projfda = ot.dr.fda(xs, ys, p)

    projfda(xs)

    np.testing.assert_allclose(np.sum(Pfda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda():
    n_samples = 100  # nb samples in source and target datasets
    rng = np.random.RandomState(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif("gaussrot", n_samples, random_state=rng)

    n_features_noise = 8

    xs = np.hstack((xs, rng.randn(n_samples, n_features_noise)))

    p = 2

    Pwda, projwda = ot.dr.wda(xs, ys, p, maxiter=10)

    projwda(xs)

    np.testing.assert_allclose(np.sum(Pwda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda_low_reg():
    n_samples = 100  # nb samples in source and target datasets
    rng = np.random.RandomState(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif("gaussrot", n_samples, random_state=rng)

    n_features_noise = 8

    xs = np.hstack((xs, rng.randn(n_samples, n_features_noise)))

    p = 2

    Pwda, projwda = ot.dr.wda(
        xs, ys, p, reg=0.01, maxiter=10, sinkhorn_method="sinkhorn_log"
    )

    projwda(xs)

    np.testing.assert_allclose(np.sum(Pwda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda_normalized():
    n_samples = 100  # nb samples in source and target datasets
    rng = np.random.RandomState(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif("gaussrot", n_samples, random_state=rng)

    n_features_noise = 8

    xs = np.hstack((xs, rng.randn(n_samples, n_features_noise)))

    p = 2

    P0 = rng.randn(10, p)
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

    def fragmented_hypercube(n, d, dim, rng):
        assert dim <= d
        assert dim >= 1
        assert dim == int(dim)

        a = (1.0 / n) * np.ones(n)
        b = (1.0 / n) * np.ones(n)

        # First measure : uniform on the hypercube
        X = rng.uniform(-1, 1, size=(n, d))

        # Second measure : fragmentation
        tmp_y = rng.uniform(-1, 1, size=(n, d))
        Y = tmp_y + 2 * np.sign(tmp_y) * np.array(dim * [1] + (d - dim) * [0])
        return a, b, X, Y

    rng = np.random.RandomState(42)
    a, b, X, Y = fragmented_hypercube(n, d, dim, rng)

    tau = 0.002
    reg = 0.2

    pi, U = ot.dr.projection_robust_wasserstein(
        X, Y, a, b, tau, reg=reg, k=k, maxiter=1000, verbose=1
    )

    U0 = rng.randn(d, k)
    U0, _ = np.linalg.qr(U0)

    pi, U = ot.dr.projection_robust_wasserstein(
        X, Y, a, b, tau, U0=U0, reg=reg, k=k, maxiter=1000, verbose=1
    )


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_ewca():
    d = 5
    n_samples = 50
    k = 3
    rng = np.random.RandomState(0)

    # generate gaussian dataset
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    D = rng.normal(size=d)
    D = (D / np.linalg.norm(D)) ** 4
    cov = Q @ np.diag(D) @ Q.T
    X = rng.multivariate_normal(np.zeros(d), cov, size=n_samples)
    X = X - X.mean(0, keepdims=True)
    assert X.shape == (n_samples, d)

    # compute first 3 components with BCD
    pi, U = ot.dr.ewca(
        X, reg=0.01, method="BCD", k=k, verbose=1, sinkhorn_method="sinkhorn_log"
    )
    assert pi.shape == (n_samples, n_samples)
    assert (pi >= 0).all()
    assert np.allclose(pi.sum(0), 1 / n_samples, atol=1e-3)
    assert np.allclose(pi.sum(1), 1 / n_samples, atol=1e-3)
    assert U.shape == (d, k)
    assert np.allclose(U.T @ U, np.eye(k), atol=1e-3)

    # test that U contains the principal components
    U_first_eigvec = np.linalg.svd(X.T, full_matrices=False)[0][:, :k]
    _, cos, _ = np.linalg.svd(U.T @ U_first_eigvec, full_matrices=False)
    assert np.allclose(cos, np.ones(k), atol=1e-3)

    # compute first 3 components with MM
    pi, U = ot.dr.ewca(
        X, reg=0.01, method="MM", k=k, verbose=1, sinkhorn_method="sinkhorn_log"
    )
    assert pi.shape == (n_samples, n_samples)
    assert (pi >= 0).all()
    assert np.allclose(pi.sum(0), 1 / n_samples, atol=1e-3)
    assert np.allclose(pi.sum(1), 1 / n_samples, atol=1e-3)
    assert U.shape == (d, k)
    assert np.allclose(U.T @ U, np.eye(k), atol=1e-3)

    # test that U contains the principal components
    U_first_eigvec = np.linalg.svd(X.T, full_matrices=False)[0][:, :k]
    _, cos, _ = np.linalg.svd(U.T @ U_first_eigvec, full_matrices=False)
    assert np.allclose(cos, np.ones(k), atol=1e-3)

    # compute last 3 components
    pi, U = ot.dr.ewca(
        X, reg=100000, method="MM", k=k, verbose=1, sinkhorn_method="sinkhorn_log"
    )

    # test that U contains the last principal components
    U_last_eigvec = np.linalg.svd(X.T, full_matrices=False)[0][:, -k:]
    _, cos, _ = np.linalg.svd(U.T @ U_last_eigvec, full_matrices=False)
    assert np.allclose(cos, np.ones(k), atol=1e-3)
