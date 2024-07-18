"""Tests for module partial"""

# Author:
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#
# License: MIT License

import numpy as np
import scipy as sp
import ot
from ot.backend import to_numpy, torch
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
        ot.partial.entropic_partial_gromov_wasserstein(
            M, M, p, q, reg=1, m=-1, log=True
        )


def test_partial_wasserstein_lagrange():
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

    w0, log0 = ot.partial.partial_wasserstein_lagrange(p, q, M, 1, log=True)

    w0, log0 = ot.partial.partial_wasserstein_lagrange(p, q, M, 100, log=True)


def test_partial_wasserstein(nx):
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

    m = 0.5

    p, q, M = nx.from_numpy(p, q, M)

    w0, log0 = ot.partial.partial_wasserstein(p, q, M, m=m, log=True)
    w, log = ot.partial.entropic_partial_wasserstein(
        p, q, M, reg=1, m=m, log=True, verbose=True
    )

    # check constraints
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=0) - q) <= 1e-5, [True] * len(q))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=0) - q) <= 1e-5, [True] * len(q))

    # check transported mass
    np.testing.assert_allclose(np.sum(to_numpy(w0)), m, atol=1e-04)
    np.testing.assert_allclose(np.sum(to_numpy(w)), m, atol=1e-04)

    w0, log0 = ot.partial.partial_wasserstein2(p, q, M, m=m, log=True)
    w0_val = ot.partial.partial_wasserstein2(p, q, M, m=m, log=False)

    G = log0["T"]

    np.testing.assert_allclose(w0, w0_val, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_equal(to_numpy(nx.sum(G, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(G, axis=0) - q) <= 1e-5, [True] * len(q))
    np.testing.assert_allclose(np.sum(to_numpy(G)), m, atol=1e-04)

    empty_array = nx.zeros(0, type_as=M)
    w = ot.partial.partial_wasserstein(empty_array, empty_array, M=M, m=None)

    # check constraints
    np.testing.assert_equal(to_numpy(nx.sum(w, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w, axis=0) - q) <= 1e-5, [True] * len(q))
    np.testing.assert_equal(to_numpy(nx.sum(w, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w, axis=0) - q) <= 1e-5, [True] * len(q))

    # check transported mass
    np.testing.assert_allclose(np.sum(to_numpy(w)), 1, atol=1e-04)

    w0 = ot.partial.entropic_partial_wasserstein(
        empty_array, empty_array, M=M, reg=10, m=None
    )

    # check constraints
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=0) - q) <= 1e-5, [True] * len(q))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=0) - q) <= 1e-5, [True] * len(q))

    # check transported mass
    np.testing.assert_allclose(np.sum(to_numpy(w0)), 1, atol=1e-04)


def test_partial_wasserstein2_gradient():
    if torch:
        n_samples = 40

        mu = np.array([0, 0])
        cov = np.array([[1, 0], [0, 2]])

        xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)
        xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)

        M = torch.tensor(ot.dist(xs, xt), requires_grad=True, dtype=torch.float64)

        p = torch.tensor(ot.unif(n_samples), dtype=torch.float64)
        q = torch.tensor(ot.unif(n_samples), dtype=torch.float64)

        m = 0.5

        w, log = ot.partial.partial_wasserstein2(p, q, M, m=m, log=True)

        w.backward()

        assert M.grad is not None
        assert M.grad.shape == M.shape


def test_entropic_partial_wasserstein_gradient():
    if torch:
        n_samples = 40

        mu = np.array([0, 0])
        cov = np.array([[1, 0], [0, 2]])

        xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)
        xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)

        M = torch.tensor(ot.dist(xs, xt), requires_grad=True, dtype=torch.float64)

        p = torch.tensor(ot.unif(n_samples), requires_grad=True, dtype=torch.float64)
        q = torch.tensor(ot.unif(n_samples), requires_grad=True, dtype=torch.float64)

        m = 0.5
        reg = 1

        _, log = ot.partial.entropic_partial_wasserstein(
            p, q, M, m=m, reg=reg, log=True
        )

        log["partial_w_dist"].backward()

        assert M.grad is not None
        assert p.grad is not None
        assert q.grad is not None
        assert M.grad.shape == M.shape
        assert p.grad.shape == p.shape
        assert q.grad.shape == q.shape


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
    res0, log0 = ot.partial.partial_gromov_wasserstein(
        C1, C3, p, q, m=m, log=True, verbose=True
    )
    np.testing.assert_allclose(res0, 0, atol=1e-1, rtol=1e-1)

    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(xt, xt)

    m = 1
    res0, log0 = ot.partial.partial_gromov_wasserstein(C1, C2, p, q, m=m, log=True)
    G = ot.gromov.gromov_wasserstein(C1, C2, p, q, "square_loss")
    np.testing.assert_allclose(G, res0, atol=1e-04)

    res, log = ot.partial.entropic_partial_gromov_wasserstein(
        C1, C2, p, q, 10, m=m, log=True
    )
    G = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, "square_loss", epsilon=10)
    np.testing.assert_allclose(G, res, atol=1e-02)

    w0, log0 = ot.partial.partial_gromov_wasserstein2(C1, C2, p, q, m=m, log=True)
    w0_val = ot.partial.partial_gromov_wasserstein2(C1, C2, p, q, m=m, log=False)
    G = log0["T"]
    np.testing.assert_allclose(w0, w0_val, atol=1e-1, rtol=1e-1)

    m = 2 / 3
    res0, log0 = ot.partial.partial_gromov_wasserstein(C1, C2, p, q, m=m, log=True)
    res, log = ot.partial.entropic_partial_gromov_wasserstein(
        C1, C2, p, q, 100, m=m, log=True
    )

    # check constraints
    np.testing.assert_equal(
        res0.sum(1) <= p, [True] * len(p)
    )  # cf convergence wasserstein
    np.testing.assert_equal(
        res0.sum(0) <= q, [True] * len(q)
    )  # cf convergence wasserstein
    np.testing.assert_allclose(np.sum(res0), m, atol=1e-04)

    np.testing.assert_equal(
        res.sum(1) <= p, [True] * len(p)
    )  # cf convergence wasserstein
    np.testing.assert_equal(
        res.sum(0) <= q, [True] * len(q)
    )  # cf convergence wasserstein
    np.testing.assert_allclose(np.sum(res), m, atol=1e-04)
