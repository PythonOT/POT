"""Tests for module partial  """

# Author:
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#
# License: MIT License

import numpy as np
import scipy as sp
import ot
import pytest


def test_raise_errors():

    n_samples = 20  # nb samples (gaussian)
    n_noise = 20  # nb of samples (noise)

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 2]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)
    xs = np.append(xs, (np.random.rand(n_noise, 2) + 1) * 4).reshape((-1, 2))
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)
    xt = np.append(xt, (np.random.rand(n_noise, 2) + 1) * -3).reshape((-1, 2))

    M = ot.dist(xs, xt)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    with pytest.raises(ValueError):
        ot.partial.partial_wasserstein_lagrange(p + 1, q, M, 1, log=True)

    with pytest.raises(ValueError):
        ot.partial.partial_wasserstein(p, q, M, m=2, log=True)

    with pytest.raises(ValueError):
        ot.partial.partial_wasserstein(p, q, M, m=-1, log=True)

    with pytest.raises(ValueError):
        ot.partial.entropic_partial_wasserstein(p, q, M, reg=1, m=2, log=True)

    with pytest.raises(ValueError):
        ot.partial.entropic_partial_wasserstein(p, q, M, reg=1, m=-1, log=True)

    with pytest.raises(ValueError):
        ot.partial.partial_gromov_wasserstein(M, M, p, q, m=2, log=True)

    with pytest.raises(ValueError):
        ot.partial.partial_gromov_wasserstein(M, M, p, q, m=-1, log=True)

    with pytest.raises(ValueError):
        ot.partial.entropic_partial_gromov_wasserstein(M, M, p, q, reg=1, m=2, log=True)

    with pytest.raises(ValueError):
        ot.partial.entropic_partial_gromov_wasserstein(M, M, p, q, reg=1, m=-1, log=True)


def test_partial_wasserstein_lagrange():

    n_samples = 20  # nb samples (gaussian)
    n_noise = 20  # nb of samples (noise)

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 2]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)
    xs = np.append(xs, (np.random.rand(n_noise, 2) + 1) * 4).reshape((-1, 2))
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)
    xt = np.append(xt, (np.random.rand(n_noise, 2) + 1) * -3).reshape((-1, 2))

    M = ot.dist(xs, xt)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    w0, log0 = ot.partial.partial_wasserstein_lagrange(p, q, M, 1, log=True)


def test_partial_wasserstein():

    n_samples = 20  # nb samples (gaussian)
    n_noise = 20  # nb of samples (noise)

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 2]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)
    xs = np.append(xs, (np.random.rand(n_noise, 2) + 1) * 4).reshape((-1, 2))
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)
    xt = np.append(xt, (np.random.rand(n_noise, 2) + 1) * -3).reshape((-1, 2))

    M = ot.dist(xs, xt)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    m = 0.5

    w0, log0 = ot.partial.partial_wasserstein(p, q, M, m=m, log=True)
    w, log = ot.partial.entropic_partial_wasserstein(p, q, M, reg=1, m=m,
                                                     log=True, verbose=True)

    # check constratints
    np.testing.assert_equal(
        w0.sum(1) - p <= 1e-5, [True] * len(p))  # cf convergence wasserstein
    np.testing.assert_equal(
        w0.sum(0) - q <= 1e-5, [True] * len(q))  # cf convergence wasserstein
    np.testing.assert_equal(
        w.sum(1) - p <= 1e-5, [True] * len(p))  # cf convergence wasserstein
    np.testing.assert_equal(
        w.sum(0) - q <= 1e-5, [True] * len(q))  # cf convergence wasserstein

    # check transported mass
    np.testing.assert_allclose(
        np.sum(w0), m, atol=1e-04)
    np.testing.assert_allclose(
        np.sum(w), m, atol=1e-04)

    w0, log0 = ot.partial.partial_wasserstein2(p, q, M, m=m, log=True)
    w0_val = ot.partial.partial_wasserstein2(p, q, M, m=m, log=False)

    G = log0['T']

    np.testing.assert_allclose(w0, w0_val, atol=1e-1, rtol=1e-1)

    # check constratints
    np.testing.assert_equal(
        G.sum(1) <= p, [True] * len(p))  # cf convergence wasserstein
    np.testing.assert_equal(
        G.sum(0) <= q, [True] * len(q))  # cf convergence wasserstein
    np.testing.assert_allclose(
        np.sum(G), m, atol=1e-04)


def test_partial_gromov_wasserstein():
    n_samples = 20  # nb samples
    n_noise = 10  # nb of samples (noise)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([0, 0, 0])
    cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s)
    xs = np.concatenate((xs, ((np.random.rand(n_noise, 2) + 1) * 4)), axis=0)
    P = sp.linalg.sqrtm(cov_t)
    xt = np.random.randn(n_samples, 3).dot(P) + mu_t
    xt = np.concatenate((xt, ((np.random.rand(n_noise, 3) + 1) * 10)), axis=0)
    xt2 = xs[::-1].copy()

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C3 = ot.dist(xt2, xt2)

    m = 2 / 3
    res0, log0 = ot.partial.partial_gromov_wasserstein(C1, C3, p, q, m=m,
                                                       log=True, verbose=True)
    np.testing.assert_allclose(res0, 0, atol=1e-1, rtol=1e-1)

    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(xt, xt)

    m = 1
    res0, log0 = ot.partial.partial_gromov_wasserstein(C1, C2, p, q, m=m,
                                                       log=True)
    G = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss')
    np.testing.assert_allclose(G, res0, atol=1e-04)

    res, log = ot.partial.entropic_partial_gromov_wasserstein(C1, C2, p, q, 10,
                                                              m=m, log=True)
    G = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q, 'square_loss', epsilon=10)
    np.testing.assert_allclose(G, res, atol=1e-02)

    w0, log0 = ot.partial.partial_gromov_wasserstein2(C1, C2, p, q, m=m,
                                                      log=True)
    w0_val = ot.partial.partial_gromov_wasserstein2(C1, C2, p, q, m=m,
                                                    log=False)
    G = log0['T']
    np.testing.assert_allclose(w0, w0_val, atol=1e-1, rtol=1e-1)

    m = 2 / 3
    res0, log0 = ot.partial.partial_gromov_wasserstein(C1, C2, p, q, m=m,
                                                       log=True)
    res, log = ot.partial.entropic_partial_gromov_wasserstein(C1, C2, p, q,
                                                              100, m=m,
                                                              log=True)

    # check constratints
    np.testing.assert_equal(
        res0.sum(1) <= p, [True] * len(p))  # cf convergence wasserstein
    np.testing.assert_equal(
        res0.sum(0) <= q, [True] * len(q))  # cf convergence wasserstein
    np.testing.assert_allclose(
        np.sum(res0), m, atol=1e-04)

    np.testing.assert_equal(
        res.sum(1) <= p, [True] * len(p))  # cf convergence wasserstein
    np.testing.assert_equal(
        res.sum(0) <= q, [True] * len(q))  # cf convergence wasserstein
    np.testing.assert_allclose(
        np.sum(res), m, atol=1e-04)
