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
    psub = ot.unif(n_samples - 5 + n_noise)
    q = ot.unif(n_samples + n_noise)

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([0, 0, 0])
    cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # clean samples
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=rng)
    P = sp.linalg.sqrtm(cov_t)
    xt = rng.randn(n_samples, 3).dot(P) + mu_t
    # add noise
    xs = np.concatenate((xs, ((rng.rand(n_noise, 2) + 1) * 4)), axis=0)
    xt = np.concatenate((xt, ((rng.rand(n_noise, 3) + 1) * 10)), axis=0)
    xt2 = xs[::-1].copy()

    C1 = ot.dist(xs, xs)
    C1sub = ot.dist(xs[5:], xs[5:])

    C2 = ot.dist(xt, xt)
    C3 = ot.dist(xt2, xt2)

    m = 2. / 3.

    C1b, C1subb, C2b, C3b, pb, psubb, qb = nx.from_numpy(C1, C1sub, C2, C3, p, psub, q)
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
        np.testing.assert_allclose(res, 0, rtol=1e-4)
        np.testing.assert_allclose(res, resb_, rtol=1e-4)
        assert np.all(res.sum(1) <= p)  # cf convergence wasserstein
        assert np.all(res.sum(0) <= q)  # cf convergence wasserstein
        np.testing.assert_allclose(
            np.sum(res), m, atol=1e-15)

    # tests with different number of samples across spaces
    m = 0.5
    res, log = ot.gromov.partial_gromov_wasserstein(
        C1, C1sub, p=p, q=psub, m=m, log=True)

    resb, logb = ot.gromov.partial_gromov_wasserstein(
        C1b, C1subb, p=pb, q=psubb, m=m, log=True)

    resb_ = nx.to_numpy(resb)
    np.testing.assert_allclose(res, resb_, rtol=1e-4)
    assert np.all(res.sum(1) <= p)  # cf convergence wasserstein
    assert np.all(res.sum(0) <= psub)  # cf convergence wasserstein
    np.testing.assert_allclose(
        np.sum(res), m, atol=1e-15)

    # Edge cases - tests with m=1 set by default (coincide with gw)
    m = 1
    res0 = ot.gromov.partial_gromov_wasserstein(
        C1, C2, p, q, m=m, log=False)
    res0b, log0b = ot.gromov.partial_gromov_wasserstein(
        C1b, C2b, pb, qb, m=None, log=True)
    G = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss')
    np.testing.assert_allclose(G, res0, rtol=1e-4)
    np.testing.assert_allclose(res0b, res0, rtol=1e-4)

    # tests for pGW2
    for loss_fun in ['square_loss', 'kl_loss']:
        w0, log0 = ot.gromov.partial_gromov_wasserstein2(
            C1, C2, p=None, q=q, m=m, loss_fun=loss_fun, log=True)
        w0_val = ot.gromov.partial_gromov_wasserstein2(
            C1b, C2b, p=pb, q=None, m=m, loss_fun=loss_fun, log=False)
        np.testing.assert_allclose(w0, w0_val, rtol=1e-4)

    # tests integers
    C1_int = C1.astype(int)
    C1b_int = nx.from_numpy(C1_int)
    C2_int = C2.astype(int)
    C2b_int = nx.from_numpy(C2_int)

    res0b, log0b = ot.gromov.partial_gromov_wasserstein(
        C1b_int, C2b_int, pb, qb, m=m, log=True)

    assert nx.to_numpy(res0b).dtype == C1_int.dtype


def test_partial_partial_gromov_linesearch(nx):
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

    # computing necessary inputs to the line-search
    Gb, _ = ot.gromov.partial_gromov_wasserstein(
        C1b, C2b, pb, qb, m=m, log=True)

    deltaGb = Gb - G0b
    fC1, fC2, hC1, hC2 = ot.gromov._utils._transform_matrix(C1b, C2b, 'square_loss')
    fC2t = fC2.T

    ones_p = nx.ones(p.shape[0], type_as=pb)
    ones_q = nx.ones(p.shape[0], type_as=pb)

    constC1 = nx.outer(nx.dot(fC1, pb), ones_q)
    constC2 = nx.outer(ones_p, nx.dot(qb, fC2t))
    cost_G0b = ot.gromov.gwloss(constC1 + constC2, hC1, hC2, G0b)

    df_G0b = ot.gromov.gwggrad(constC1 + constC2, hC1, hC2, G0b)

    # perform line-search
    alpha, _, cost_Gb = ot.gromov.solve_partial_gromov_linesearch(
        G0b, deltaGb, cost_G0b, df_G0b, fC1, fC2, hC1, hC2, 0., 1.,
        alpha_min=0., alpha_max=1.)

    np.testing.assert_allclose(alpha, 1., rtol=1e-4)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_partial_gromov_wasserstein(nx):
    rng = np.random.RandomState(42)
    n_samples = 20  # nb samples
    n_noise = 10  # nb of samples (noise)

    p = ot.unif(n_samples + n_noise)
    psub = ot.unif(n_samples - 5 + n_noise)
    q = ot.unif(n_samples + n_noise)

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([0, 0, 0])
    cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # clean samples
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=rng)
    P = sp.linalg.sqrtm(cov_t)
    xt = rng.randn(n_samples, 3).dot(P) + mu_t
    # add noise
    xs = np.concatenate((xs, ((rng.rand(n_noise, 2) + 1) * 4)), axis=0)
    xt = np.concatenate((xt, ((rng.rand(n_noise, 3) + 1) * 10)), axis=0)
    xt2 = xs[::-1].copy()

    C1 = ot.dist(xs, xs)
    C1sub = ot.dist(xs[5:], xs[5:])

    C2 = ot.dist(xt, xt)
    C3 = ot.dist(xt2, xt2)

    m = 2. / 3.

    C1b, C1subb, C2b, C3b, pb, psubb, qb = nx.from_numpy(C1, C1sub, C2, C3, p, psub, q)
    G0 = np.outer(p, q) * m / (np.sum(p) * np.sum(q))  # make sure |G0|=m, G01_m\leq p, G0.T1_n\leq q.
    G0b = nx.from_numpy(G0)

    # check consistency across backends and stability w.r.t loss/marginals/sym
    list_sym = [True, None]
    for i, loss_fun in enumerate(['square_loss', 'kl_loss']):
        res, log = ot.gromov.entropic_partial_gromov_wasserstein(
            C1, C3, p=p, q=None, reg=1e4, m=m, G0=None, log=True,
            symmetric=list_sym[i], verbose=True)

        resb, logb = ot.gromov.entropic_partial_gromov_wasserstein(
            C1b, C3b, p=None, q=qb, reg=1e4, m=m, G0=G0b, log=True,
            symmetric=False, verbose=True)

        resb_ = nx.to_numpy(resb)
        np.testing.assert_allclose(res, 0, rtol=1e-4)
        np.testing.assert_allclose(res, resb_, rtol=1e-4)
        assert np.all(res.sum(1) <= p)  # cf convergence wasserstein
        assert np.all(res.sum(0) <= q)  # cf convergence wasserstein
        np.testing.assert_allclose(
            np.sum(res), m, rtol=1e-4)

    # tests with m is None
    res = ot.gromov.entropic_partial_gromov_wasserstein(
        C1, C3, p=p, q=None, reg=1e4, G0=None, log=False,
        symmetric=list_sym[i], verbose=True)

    resb = ot.gromov.entropic_partial_gromov_wasserstein(
        C1b, C3b, p=None, q=qb, reg=1e4, G0=None, log=False,
        symmetric=False, verbose=True)

    resb_ = nx.to_numpy(resb)
    np.testing.assert_allclose(res, 0, atol=1e-1, rtol=1e-1)
    np.testing.assert_allclose(res, resb_, atol=1e-7)
    np.testing.assert_allclose(
        np.sum(res), 1., rtol=1e-4)

    # tests with different number of samples across spaces
    m = 0.5
    res, log = ot.gromov.entropic_partial_gromov_wasserstein(
        C1, C1sub, p=p, q=psub, reg=1e4, m=m, log=True)

    resb, logb = ot.gromov.entropic_partial_gromov_wasserstein(
        C1b, C1subb, p=pb, q=psubb, reg=1e4, m=m, log=True)

    resb_ = nx.to_numpy(resb)
    np.testing.assert_allclose(res, resb_, rtol=1e-4)
    assert np.all(res.sum(1) <= p)  # cf convergence wasserstein
    assert np.all(res.sum(0) <= psub)  # cf convergence wasserstein
    np.testing.assert_allclose(
        np.sum(res), m, rtol=1e-4)

    # tests for pGW2
    for loss_fun in ['square_loss', 'kl_loss']:
        w0, log0 = ot.gromov.entropic_partial_gromov_wasserstein2(
            C1, C2, p=None, q=q, reg=1e4, m=m, loss_fun=loss_fun, log=True)
        w0_val = ot.gromov.entropic_partial_gromov_wasserstein2(
            C1b, C2b, p=pb, q=None, reg=1e4, m=m, loss_fun=loss_fun, log=False)
        np.testing.assert_allclose(w0, w0_val, rtol=1e-8)
