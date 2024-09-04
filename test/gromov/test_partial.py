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


def test_partial_gromov_wasserstein(nx):
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

    m = 2. / 3.

    C1b, C2b, C3b, pb, qb = nx.from_numpy(C1, C2, C3, p, q)
    G0 = np.outer(p, q) * m / (np.sum(p) * np.sum(q))  # make sure |G0|=m, G01_m\leq p, G0.T1_n\leq q.
    G0b = nx.from_numpy(G0)

    # check consistency across backends and stability w.r.t loss/marginals/sym
    list_sym = [True, None]
    for i, loss_fun in enumerate(['square_loss', 'kl_loss']):
        res, log = ot.gromov.partial_gromov_wasserstein(
            C1, C3, p=p, q=None, m=m, G0=None, log=True, symmetric=list_sym[i],
            warn=True, verbose=True)

        resb, logb = ot.gromov.partial_gromov_wasserstein(
            C1b, C3b, p=None, q=qb, m=m, G0=G0b, log=True, symmetric=False,
            warn=True, verbose=True)

        resb_ = nx.to_numpy(resb)
        np.testing.assert_allclose(res, 0, atol=1e-1, rtol=1e-1)
        np.testing.assert_allclose(res, resb_, atol=1e-15)
        assert np.all(res.sum(1) <= p)  # cf convergence wasserstein
        assert np.all(res.sum(0) <= q)  # cf convergence wasserstein
        np.testing.assert_allclose(
            np.sum(res), m, atol=1e-15)

    # Edge cases - tests with m=1 set by default (coincide with gw)
    m = 1
    res0, log0 = ot.gromov.partial_gromov_wasserstein(
        C1, C2, p, q, m=m, log=True)
    res0b, log0b = ot.gromov.partial_gromov_wasserstein(
        C1b, C2b, pb, qb, m=None, log=True)
    G = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss')
    np.testing.assert_allclose(G, res0, atol=1e-04)
    np.testing.assert_allclose(res0b, res0, atol=1e-04)

    # tests for pGW2
    for loss_fun in ['square_loss', 'kl_loss']:
        w0, log0 = ot.gromov.partial_gromov_wasserstein2(
            C1, C2, p=None, q=q, m=m, loss_fun=loss_fun, log=True)
        w0_val = ot.gromov.partial_gromov_wasserstein2(
            C1b, C2b, p=pb, q=None, m=m, loss_fun=loss_fun, log=False)
        np.testing.assert_allclose(w0, w0_val, rtol=1e-8)


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

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    m = 1

    res, log = ot.gromov.entropic_partial_gromov_wasserstein(C1, C2, p, q, 1e4,
                                                             m=m, log=True)
    np.testing.assert_allclose(
        np.sum(res), m, atol=1e-04)
