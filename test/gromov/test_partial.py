""" Tests for gromov._partial.py """

# Author:
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#         Cédric Vincent-Cuat <cedvincentcuaz@gmail.com>
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

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=rng)
    xs = np.append(xs, (rng.rand(n_noise, 2) + 1) * 4).reshape((-1, 2))
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=rng)
    xt = np.append(xt, (rng.rand(n_noise, 2) + 1) * -3).reshape((-1, 2))

    M = ot.dist(xs, xt)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    with pytest.raises(ValueError):
        ot.gromov.partial_gromov_wasserstein(M, M, p, q, m=2, log=True)

    with pytest.raises(ValueError):
        ot.gromov.partial_gromov_wasserstein(M, M, p, q, m=-1, log=True)

    with pytest.raises(ValueError):
        ot.gromov.entropic_partial_gromov_wasserstein(M, M, p, q, reg=1, m=2,
                                                      log=True)

    with pytest.raises(ValueError):
        ot.gromov.entropic_partial_gromov_wasserstein(M, M, p, q, reg=1, m=-1,
                                                      log=True)


def test_partial_gromov_wasserstein():
    rng = np.random.RandomState(42)
    n_samples = 20  # nb samples
    n_noise = 10  # nb of samples (noise)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([0, 0, 0])
    cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=rng)
    xs = np.concatenate((xs, ((rng.rand(n_noise, 2) + 1) * 4)), axis=0)
    P = sp.linalg.sqrtm(cov_t)
    xt = rng.randn(n_samples, 3).dot(P) + mu_t
    xt = np.concatenate((xt, ((rng.rand(n_noise, 3) + 1) * 10)), axis=0)
    xt2 = xs[::-1].copy()

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C3 = ot.dist(xt2, xt2)

    m = 2 / 3
    res0, log0 = ot.gromov.partial_gromov_wasserstein(C1, C3, p, q, m=m,
                                                      log=True, verbose=True)
    np.testing.assert_allclose(res0, 0, atol=1e-1, rtol=1e-1)

    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(xt, xt)

    m = 1
    res0, log0 = ot.gromov.partial_gromov_wasserstein(C1, C2, p, q, m=m,
                                                      log=True)
    G = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss')
    np.testing.assert_allclose(G, res0, atol=1e-04)

    w0, log0 = ot.gromov.partial_gromov_wasserstein2(C1, C2, p, q, m=m,
                                                     log=True)
    w0_val = ot.gromov.partial_gromov_wasserstein2(C1, C2, p, q, m=m,
                                                   log=False)
    G = log0['T']
    np.testing.assert_allclose(w0, w0_val, atol=1e-1, rtol=1e-1)

    m = 2 / 3
    res0, log0 = ot.gromov.partial_gromov_wasserstein(C1, C2, p, q, m=m,
                                                      log=True)

    # check constraints
    np.testing.assert_equal(
        res0.sum(1) <= p, [True] * len(p))  # cf convergence wasserstein
    np.testing.assert_equal(
        res0.sum(0) <= q, [True] * len(q))  # cf convergence wasserstein
    np.testing.assert_allclose(
        np.sum(res0), m, atol=1e-04)


def test_entropic_partial_gromov_wasserstein():
    rng = np.random.RandomState(42)
    n_samples = 20  # nb samples
    n_noise = 10  # nb of samples (noise)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([0, 0, 0])
    cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=rng)
    xs = np.concatenate((xs, ((rng.rand(n_noise, 2) + 1) * 4)), axis=0)
    P = sp.linalg.sqrtm(cov_t)
    xt = rng.randn(n_samples, 3).dot(P) + mu_t
    xt = np.concatenate((xt, ((rng.rand(n_noise, 3) + 1) * 10)), axis=0)
    xt2 = xs[::-1].copy()

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C3 = ot.dist(xt2, xt2)

    m = 1

    res, log = ot.gromov.entropic_partial_gromov_wasserstein(C1, C2, p, q, 1e4,
                                                             m=m, log=True)
    np.testing.assert_allclose(
        np.sum(res), m, atol=1e-04)
