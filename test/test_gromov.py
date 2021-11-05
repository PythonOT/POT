"""Tests for module gromov  """

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#
# License: MIT License

import numpy as np
import ot
from ot.backend import NumpyBackend
from ot.backend import torch

import pytest


def test_gromov(nx):
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b = nx.from_numpy(C1)
    C2b = nx.from_numpy(C2)
    pb = nx.from_numpy(p)
    qb = nx.from_numpy(q)

    G = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', verbose=True)
    Gb = nx.to_numpy(ot.gromov.gromov_wasserstein(C1b, C2b, pb, qb, 'square_loss', verbose=True))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    Id = (1 / (1.0 * n_samples)) * np.eye(n_samples, n_samples)

    np.testing.assert_allclose(Gb, np.flipud(Id), atol=1e-04)

    gw, log = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'kl_loss', log=True)
    gwb, logb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', log=True)

    gw_val = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'kl_loss', log=False)
    gw_valb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', log=False)

    G = log['T']
    Gb = nx.to_numpy(logb['T'])

    np.testing.assert_allclose(gw, gwb, atol=1e-06)
    np.testing.assert_allclose(gwb, 0, atol=1e-1, rtol=1e-1)

    np.testing.assert_allclose(gw_val, gw_valb, atol=1e-06)
    np.testing.assert_allclose(gwb, gw_valb, atol=1e-1, rtol=1e-1)  # cf log=False

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


def test_gromov_dtype_device(nx):
    # setup
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        C1b = nx.from_numpy(C1, type_as=tp)
        C2b = nx.from_numpy(C2, type_as=tp)
        pb = nx.from_numpy(p, type_as=tp)
        qb = nx.from_numpy(q, type_as=tp)

        Gb = ot.gromov.gromov_wasserstein(C1b, C2b, pb, qb, 'square_loss', verbose=True)
        gw_valb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', log=False)

        nx.assert_same_dtype_device(C1b, Gb)
        nx.assert_same_dtype_device(C1b, gw_valb)


def test_gromov2_gradients():
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)

    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=5)

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    if torch:

        p1 = torch.tensor(p, requires_grad=True)
        q1 = torch.tensor(q, requires_grad=True)
        C11 = torch.tensor(C1, requires_grad=True)
        C12 = torch.tensor(C2, requires_grad=True)

        val = ot.gromov_wasserstein2(C11, C12, p1, q1)

        val.backward()

        assert q1.shape == q1.grad.shape
        assert p1.shape == p1.grad.shape
        assert C11.shape == C11.grad.shape
        assert C12.shape == C12.grad.shape


@pytest.skip_backend("jax", reason="test very slow with jax backend")
def test_entropic_gromov(nx):
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b = nx.from_numpy(C1)
    C2b = nx.from_numpy(C2)
    pb = nx.from_numpy(p)
    qb = nx.from_numpy(q)

    G = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q, 'square_loss', epsilon=5e-4, verbose=True)
    Gb = nx.to_numpy(ot.gromov.entropic_gromov_wasserstein(
        C1b, C2b, pb, qb, 'square_loss', epsilon=5e-4, verbose=True
    ))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    gw, log = ot.gromov.entropic_gromov_wasserstein2(
        C1, C2, p, q, 'kl_loss', epsilon=1e-2, log=True)
    gwb, logb = ot.gromov.entropic_gromov_wasserstein2(
        C1b, C2b, pb, qb, 'kl_loss', epsilon=1e-2, log=True)

    G = log['T']
    Gb = nx.to_numpy(logb['T'])

    np.testing.assert_allclose(gw, gwb, atol=1e-06)
    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


@pytest.skip_backend("jax", reason="test very slow with jax backend")
def test_entropic_gromov_dtype_device(nx):
    # setup
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        C1b = nx.from_numpy(C1, type_as=tp)
        C2b = nx.from_numpy(C2, type_as=tp)
        pb = nx.from_numpy(p, type_as=tp)
        qb = nx.from_numpy(q, type_as=tp)

        Gb = ot.gromov.entropic_gromov_wasserstein(
            C1b, C2b, pb, qb, 'square_loss', epsilon=5e-4, verbose=True
        )
        gw_valb = ot.gromov.entropic_gromov_wasserstein2(
            C1b, C2b, pb, qb, 'square_loss', epsilon=5e-4, verbose=True
        )

        nx.assert_same_dtype_device(C1b, Gb)
        nx.assert_same_dtype_device(C1b, gw_valb)


def test_pointwise_gromov(nx):
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b = nx.from_numpy(C1)
    C2b = nx.from_numpy(C2)
    pb = nx.from_numpy(p)
    qb = nx.from_numpy(q)

    def loss(x, y):
        return np.abs(x - y)

    def lossb(x, y):
        return nx.abs(x - y)

    G, log = ot.gromov.pointwise_gromov_wasserstein(
        C1, C2, p, q, loss, max_iter=100, log=True, verbose=True, random_state=42)
    G = NumpyBackend().todense(G)
    Gb, logb = ot.gromov.pointwise_gromov_wasserstein(
        C1b, C2b, pb, qb, lossb, max_iter=100, log=True, verbose=True, random_state=42)
    Gb = nx.to_numpy(nx.todense(Gb))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    np.testing.assert_allclose(logb['gw_dist_estimated'], 0.0, atol=1e-08)
    np.testing.assert_allclose(logb['gw_dist_std'], 0.0, atol=1e-08)

    G, log = ot.gromov.pointwise_gromov_wasserstein(
        C1, C2, p, q, loss, max_iter=100, alpha=0.1, log=True, verbose=True, random_state=42)
    G = NumpyBackend().todense(G)
    Gb, logb = ot.gromov.pointwise_gromov_wasserstein(
        C1b, C2b, pb, qb, lossb, max_iter=100, alpha=0.1, log=True, verbose=True, random_state=42)
    Gb = nx.to_numpy(nx.todense(Gb))

    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(logb['gw_dist_estimated'], 0.10342276348494964, atol=1e-8)
    np.testing.assert_allclose(logb['gw_dist_std'], 0.0015952535464736394, atol=1e-8)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
def test_sampled_gromov(nx):
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0], dtype=np.float64)
    cov_s = np.array([[1, 0], [0, 1]], dtype=np.float64)

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b = nx.from_numpy(C1)
    C2b = nx.from_numpy(C2)
    pb = nx.from_numpy(p)
    qb = nx.from_numpy(q)

    def loss(x, y):
        return np.abs(x - y)

    def lossb(x, y):
        return nx.abs(x - y)

    G, log = ot.gromov.sampled_gromov_wasserstein(
        C1, C2, p, q, loss, max_iter=100, epsilon=1, log=True, verbose=True, random_state=42)
    Gb, logb = ot.gromov.sampled_gromov_wasserstein(
        C1b, C2b, pb, qb, lossb, max_iter=100, epsilon=1, log=True, verbose=True, random_state=42)
    Gb = nx.to_numpy(Gb)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    np.testing.assert_allclose(logb['gw_dist_estimated'], 0.05679474884977278, atol=1e-08)
    np.testing.assert_allclose(logb['gw_dist_std'], 0.0005986592106971995, atol=1e-08)


def test_gromov_barycenter(nx):
    ns = 10
    nt = 20

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    p1 = ot.unif(ns)
    p2 = ot.unif(nt)
    n_samples = 3
    p = ot.unif(n_samples)

    C1b = nx.from_numpy(C1)
    C2b = nx.from_numpy(C2)
    p1b = nx.from_numpy(p1)
    p2b = nx.from_numpy(p2)
    pb = nx.from_numpy(p)

    Cb = ot.gromov.gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'square_loss', max_iter=100, tol=1e-3, verbose=True, random_state=42
    )
    Cbb = nx.to_numpy(ot.gromov.gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'square_loss', max_iter=100, tol=1e-3, verbose=True, random_state=42
    ))
    np.testing.assert_allclose(Cb, Cbb, atol=1e-06)
    np.testing.assert_allclose(Cbb.shape, (n_samples, n_samples))

    Cb2 = ot.gromov.gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'kl_loss', max_iter=100, tol=1e-3, random_state=42
    )
    Cb2b = nx.to_numpy(ot.gromov.gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'kl_loss', max_iter=100, tol=1e-3, random_state=42
    ))
    np.testing.assert_allclose(Cb2, Cb2b, atol=1e-06)
    np.testing.assert_allclose(Cb2b.shape, (n_samples, n_samples))


@pytest.mark.filterwarnings("ignore:divide")
def test_gromov_entropic_barycenter(nx):
    ns = 10
    nt = 20

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    p1 = ot.unif(ns)
    p2 = ot.unif(nt)
    n_samples = 2
    p = ot.unif(n_samples)

    C1b = nx.from_numpy(C1)
    C2b = nx.from_numpy(C2)
    p1b = nx.from_numpy(p1)
    p2b = nx.from_numpy(p2)
    pb = nx.from_numpy(p)

    Cb = ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'square_loss', 1e-3, max_iter=50, tol=1e-3, verbose=True, random_state=42
    )
    Cbb = nx.to_numpy(ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'square_loss', 1e-3, max_iter=50, tol=1e-3, verbose=True, random_state=42
    ))
    np.testing.assert_allclose(Cb, Cbb, atol=1e-06)
    np.testing.assert_allclose(Cbb.shape, (n_samples, n_samples))

    Cb2 = ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'kl_loss', 1e-3, max_iter=100, tol=1e-3, random_state=42
    )
    Cb2b = nx.to_numpy(ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'kl_loss', 1e-3, max_iter=100, tol=1e-3, random_state=42
    ))
    np.testing.assert_allclose(Cb2, Cb2b, atol=1e-06)
    np.testing.assert_allclose(Cb2b.shape, (n_samples, n_samples))


def test_fgw(nx):
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    ys = np.random.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)
    M /= M.max()

    Mb = nx.from_numpy(M)
    C1b = nx.from_numpy(C1)
    C2b = nx.from_numpy(C2)
    pb = nx.from_numpy(p)
    qb = nx.from_numpy(q)

    G, log = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, 'square_loss', alpha=0.5, log=True)
    Gb, logb = ot.gromov.fused_gromov_wasserstein(Mb, C1b, C2b, pb, qb, 'square_loss', alpha=0.5, log=True)
    Gb = nx.to_numpy(Gb)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence fgw
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence fgw

    Id = (1 / (1.0 * n_samples)) * np.eye(n_samples, n_samples)

    np.testing.assert_allclose(
        Gb, np.flipud(Id), atol=1e-04)  # cf convergence gromov

    fgw, log = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, p, q, 'square_loss', alpha=0.5, log=True)
    fgwb, logb = ot.gromov.fused_gromov_wasserstein2(Mb, C1b, C2b, pb, qb, 'square_loss', alpha=0.5, log=True)

    G = log['T']
    Gb = nx.to_numpy(logb['T'])

    np.testing.assert_allclose(fgw, fgwb, atol=1e-08)
    np.testing.assert_allclose(fgwb, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


def test_fgw2_gradients():
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)

    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=5)

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    M = ot.dist(xs, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    if torch:

        p1 = torch.tensor(p, requires_grad=True)
        q1 = torch.tensor(q, requires_grad=True)
        C11 = torch.tensor(C1, requires_grad=True)
        C12 = torch.tensor(C2, requires_grad=True)
        M1 = torch.tensor(M, requires_grad=True)

        val = ot.fused_gromov_wasserstein2(M1, C11, C12, p1, q1)

        val.backward()

        assert q1.shape == q1.grad.shape
        assert p1.shape == p1.grad.shape
        assert C11.shape == C11.grad.shape
        assert C12.shape == C12.grad.shape
        assert M1.shape == M1.grad.shape


def test_fgw_barycenter(nx):
    np.random.seed(42)

    ns = 50
    nt = 60

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    ys = np.random.randn(Xs.shape[0], 2)
    yt = np.random.randn(Xt.shape[0], 2)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    p1, p2 = ot.unif(ns), ot.unif(nt)
    n_samples = 3
    p = ot.unif(n_samples)

    ysb = nx.from_numpy(ys)
    ytb = nx.from_numpy(yt)
    C1b = nx.from_numpy(C1)
    C2b = nx.from_numpy(C2)
    p1b = nx.from_numpy(p1)
    p2b = nx.from_numpy(p2)
    pb = nx.from_numpy(p)

    Xb, Cb = ot.gromov.fgw_barycenters(
        n_samples, [ysb, ytb], [C1b, C2b], [p1b, p2b], [.5, .5], 0.5, fixed_structure=False,
        fixed_features=False, p=pb, loss_fun='square_loss', max_iter=100, tol=1e-3, random_state=12345
    )

    xalea = np.random.randn(n_samples, 2)
    init_C = ot.dist(xalea, xalea)
    init_Cb = nx.from_numpy(init_C)

    Xb, Cb = ot.gromov.fgw_barycenters(
        n_samples, [ysb, ytb], [C1b, C2b], ps=[p1b, p2b], lambdas=[.5, .5],
        alpha=0.5, fixed_structure=True, init_C=init_Cb, fixed_features=False,
        p=pb, loss_fun='square_loss', max_iter=100, tol=1e-3
    )
    Xb, Cb = nx.to_numpy(Xb), nx.to_numpy(Cb)
    np.testing.assert_allclose(Cb.shape, (n_samples, n_samples))
    np.testing.assert_allclose(Xb.shape, (n_samples, ys.shape[1]))

    init_X = np.random.randn(n_samples, ys.shape[1])
    init_Xb = nx.from_numpy(init_X)

    Xb, Cb, logb = ot.gromov.fgw_barycenters(
        n_samples, [ysb, ytb], [C1b, C2b], [p1b, p2b], [.5, .5], 0.5,
        fixed_structure=False, fixed_features=True, init_X=init_Xb,
        p=pb, loss_fun='square_loss', max_iter=100, tol=1e-3, log=True, random_state=98765
    )
    Xb, Cb = nx.to_numpy(Xb), nx.to_numpy(Cb)
    np.testing.assert_allclose(Cb.shape, (n_samples, n_samples))
    np.testing.assert_allclose(Xb.shape, (n_samples, ys.shape[1]))
