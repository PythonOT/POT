"""Tests for module gromov  """

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedric.vincent-cuaz@inria.fr>
#
# License: MIT License

import numpy as np
import ot
from ot.backend import NumpyBackend
from ot.backend import torch, tf

import pytest


def test_gromov(nx):
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    G = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', G0=G0, verbose=True)
    Gb = nx.to_numpy(ot.gromov.gromov_wasserstein(C1b, C2b, pb, qb, 'square_loss', G0=G0b, verbose=True))

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
    gwb = nx.to_numpy(gwb)

    gw_val = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'kl_loss', G0=G0, log=False)
    gw_valb = nx.to_numpy(
        ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', G0=G0b, log=False)
    )

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
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0, type_as=tp)

        Gb = ot.gromov.gromov_wasserstein(C1b, C2b, pb, qb, 'square_loss', G0=G0b, verbose=True)
        gw_valb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', G0=G0b, log=False)

        nx.assert_same_dtype_device(C1b, Gb)
        nx.assert_same_dtype_device(C1b, gw_valb)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_gromov_device_tf():
    nx = ot.backend.TensorflowBackend()
    n_samples = 50  # nb samples
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()
    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)
        Gb = ot.gromov.gromov_wasserstein(C1b, C2b, pb, qb, 'square_loss', G0=G0b, verbose=True)
        gw_valb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', G0=G0b, log=False)
        nx.assert_same_dtype_device(C1b, Gb)
        nx.assert_same_dtype_device(C1b, gw_valb)

    if len(tf.config.list_physical_devices('GPU')) > 0:
        # Check that everything happens on the GPU
        C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)
        Gb = ot.gromov.gromov_wasserstein(C1b, C2b, pb, qb, 'square_loss', verbose=True)
        gw_valb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', log=False)
        nx.assert_same_dtype_device(C1b, Gb)
        nx.assert_same_dtype_device(C1b, gw_valb)
        assert nx.dtype_device(Gb)[1].startswith("GPU")


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

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            p1 = torch.tensor(p, requires_grad=True, device=device)
            q1 = torch.tensor(q, requires_grad=True, device=device)
            C11 = torch.tensor(C1, requires_grad=True, device=device)
            C12 = torch.tensor(C2, requires_grad=True, device=device)

            val = ot.gromov_wasserstein2(C11, C12, p1, q1)

            val.backward()

            assert val.device == p1.device
            assert q1.shape == q1.grad.shape
            assert p1.shape == p1.grad.shape
            assert C11.shape == C11.grad.shape
            assert C12.shape == C12.grad.shape


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_gromov(nx):
    n_samples = 10  # nb samples

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

    C1b, C2b, pb, qb = nx.from_numpy(C1, C2, p, q)

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
        C1, C2, p, q, 'kl_loss', max_iter=10, epsilon=1e-2, log=True)
    gwb, logb = ot.gromov.entropic_gromov_wasserstein2(
        C1b, C2b, pb, qb, 'kl_loss', max_iter=10, epsilon=1e-2, log=True)
    gwb = nx.to_numpy(gwb)

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
@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_gromov_dtype_device(nx):
    # setup
    n_samples = 5  # nb samples

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

        C1b, C2b, pb, qb = nx.from_numpy(C1, C2, p, q, type_as=tp)

        Gb = ot.gromov.entropic_gromov_wasserstein(
            C1b, C2b, pb, qb, 'square_loss', epsilon=5e-4, verbose=True
        )
        gw_valb = ot.gromov.entropic_gromov_wasserstein2(
            C1b, C2b, pb, qb, 'square_loss', epsilon=5e-4, verbose=True
        )

        nx.assert_same_dtype_device(C1b, Gb)
        nx.assert_same_dtype_device(C1b, gw_valb)


def test_pointwise_gromov(nx):
    n_samples = 5  # nb samples

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

    C1b, C2b, pb, qb = nx.from_numpy(C1, C2, p, q)

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

    np.testing.assert_allclose(float(logb['gw_dist_estimated']), 0.0, atol=1e-08)
    np.testing.assert_allclose(float(logb['gw_dist_std']), 0.0, atol=1e-08)

    G, log = ot.gromov.pointwise_gromov_wasserstein(
        C1, C2, p, q, loss, max_iter=100, alpha=0.1, log=True, verbose=True, random_state=42)
    G = NumpyBackend().todense(G)
    Gb, logb = ot.gromov.pointwise_gromov_wasserstein(
        C1b, C2b, pb, qb, lossb, max_iter=100, alpha=0.1, log=True, verbose=True, random_state=42)
    Gb = nx.to_numpy(nx.todense(Gb))

    np.testing.assert_allclose(G, Gb, atol=1e-06)


@pytest.skip_backend("tf", reason="test very slow with tf backend")
@pytest.skip_backend("jax", reason="test very slow with jax backend")
def test_sampled_gromov(nx):
    n_samples = 5  # nb samples

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

    C1b, C2b, pb, qb = nx.from_numpy(C1, C2, p, q)

    def loss(x, y):
        return np.abs(x - y)

    def lossb(x, y):
        return nx.abs(x - y)

    G, log = ot.gromov.sampled_gromov_wasserstein(
        C1, C2, p, q, loss, max_iter=20, nb_samples_grad=2, epsilon=1, log=True, verbose=True, random_state=42)
    Gb, logb = ot.gromov.sampled_gromov_wasserstein(
        C1b, C2b, pb, qb, lossb, max_iter=20, nb_samples_grad=2, epsilon=1, log=True, verbose=True, random_state=42)
    Gb = nx.to_numpy(Gb)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


def test_gromov_barycenter(nx):
    ns = 5
    nt = 8

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    p1 = ot.unif(ns)
    p2 = ot.unif(nt)
    n_samples = 3
    p = ot.unif(n_samples)

    C1b, C2b, p1b, p2b, pb = nx.from_numpy(C1, C2, p1, p2, p)

    Cb = ot.gromov.gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'square_loss', max_iter=100, tol=1e-3, verbose=False, random_state=42
    )
    Cbb = nx.to_numpy(ot.gromov.gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'square_loss', max_iter=100, tol=1e-3, verbose=False, random_state=42
    ))
    np.testing.assert_allclose(Cb, Cbb, atol=1e-06)
    np.testing.assert_allclose(Cbb.shape, (n_samples, n_samples))

    # test of gromov_barycenters with `log` on
    Cb_, err_ = ot.gromov.gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'square_loss', max_iter=100, tol=1e-3, verbose=False, random_state=42, log=True
    )
    Cbb_, errb_ = ot.gromov.gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'square_loss', max_iter=100, tol=1e-3, verbose=False, random_state=42, log=True
    )
    Cbb_ = nx.to_numpy(Cbb_)
    np.testing.assert_allclose(Cb_, Cbb_, atol=1e-06)
    np.testing.assert_array_almost_equal(err_['err'], nx.to_numpy(*errb_['err']))
    np.testing.assert_allclose(Cbb_.shape, (n_samples, n_samples))

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

    # test of gromov_barycenters with `log` on
    Cb2_, err2_ = ot.gromov.gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'kl_loss', max_iter=100, tol=1e-3, verbose=False, random_state=42, log=True
    )
    Cb2b_, err2b_ = ot.gromov.gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'kl_loss', max_iter=100, tol=1e-3, verbose=False, random_state=42, log=True
    )
    Cb2b_ = nx.to_numpy(Cb2b_)
    np.testing.assert_allclose(Cb2_, Cb2b_, atol=1e-06)
    np.testing.assert_array_almost_equal(err2_['err'], nx.to_numpy(*err2b_['err']))
    np.testing.assert_allclose(Cb2b_.shape, (n_samples, n_samples))


@pytest.mark.filterwarnings("ignore:divide")
def test_gromov_entropic_barycenter(nx):
    ns = 5
    nt = 10

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    p1 = ot.unif(ns)
    p2 = ot.unif(nt)
    n_samples = 2
    p = ot.unif(n_samples)

    C1b, C2b, p1b, p2b, pb = nx.from_numpy(C1, C2, p1, p2, p)

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

    # test of entropic_gromov_barycenters with `log` on
    Cb_, err_ = ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'square_loss', 1e-3, max_iter=100, tol=1e-3, verbose=True, random_state=42, log=True
    )
    Cbb_, errb_ = ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'square_loss', 1e-3, max_iter=100, tol=1e-3, verbose=True, random_state=42, log=True
    )
    Cbb_ = nx.to_numpy(Cbb_)
    np.testing.assert_allclose(Cb_, Cbb_, atol=1e-06)
    np.testing.assert_array_almost_equal(err_['err'], nx.to_numpy(*errb_['err']))
    np.testing.assert_allclose(Cbb_.shape, (n_samples, n_samples))

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

    # test of entropic_gromov_barycenters with `log` on
    Cb2_, err2_ = ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'kl_loss', 1e-3, max_iter=100, tol=1e-3, verbose=True, random_state=42, log=True
    )
    Cb2b_, err2b_ = ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'kl_loss', 1e-3, max_iter=100, tol=1e-3, verbose=True, random_state=42, log=True
    )
    Cb2b_ = nx.to_numpy(Cb2b_)
    np.testing.assert_allclose(Cb2_, Cb2b_, atol=1e-06)
    np.testing.assert_array_almost_equal(err2_['err'], nx.to_numpy(*err2b_['err']))
    np.testing.assert_allclose(Cb2b_.shape, (n_samples, n_samples))


def test_fgw(nx):
    n_samples = 20  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    ys = np.random.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)
    M /= M.max()

    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)

    G, log = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, 'square_loss', alpha=0.5, G0=G0, log=True)
    Gb, logb = ot.gromov.fused_gromov_wasserstein(Mb, C1b, C2b, pb, qb, 'square_loss', alpha=0.5, G0=G0b, log=True)
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

    fgw, log = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, p, q, 'square_loss', G0=None, alpha=0.5, log=True)
    fgwb, logb = ot.gromov.fused_gromov_wasserstein2(Mb, C1b, C2b, pb, qb, 'square_loss', G0=G0b, alpha=0.5, log=True)
    fgwb = nx.to_numpy(fgwb)

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
    n_samples = 20  # nb samples

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

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            p1 = torch.tensor(p, requires_grad=True, device=device)
            q1 = torch.tensor(q, requires_grad=True, device=device)
            C11 = torch.tensor(C1, requires_grad=True, device=device)
            C12 = torch.tensor(C2, requires_grad=True, device=device)
            M1 = torch.tensor(M, requires_grad=True, device=device)

            val = ot.fused_gromov_wasserstein2(M1, C11, C12, p1, q1)

            val.backward()

            assert val.device == p1.device
            assert q1.shape == q1.grad.shape
            assert p1.shape == p1.grad.shape
            assert C11.shape == C11.grad.shape
            assert C12.shape == C12.grad.shape
            assert M1.shape == M1.grad.shape


def test_fgw_barycenter(nx):
    np.random.seed(42)

    ns = 10
    nt = 20

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    ys = np.random.randn(Xs.shape[0], 2)
    yt = np.random.randn(Xt.shape[0], 2)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    p1, p2 = ot.unif(ns), ot.unif(nt)
    n_samples = 3
    p = ot.unif(n_samples)

    ysb, ytb, C1b, C2b, p1b, p2b, pb = nx.from_numpy(ys, yt, C1, C2, p1, p2, p)

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


def test_gromov_wasserstein_linear_unmixing(nx):
    n = 4

    X1, y1 = ot.datasets.make_data_classif('3gauss', n, random_state=42)
    X2, y2 = ot.datasets.make_data_classif('3gauss2', n, random_state=42)

    C1 = ot.dist(X1)
    C2 = ot.dist(X2)
    Cdict = np.stack([C1, C2])
    p = ot.unif(n)

    C1b, C2b, Cdictb, pb = nx.from_numpy(C1, C2, Cdict, p)

    tol = 10**(-5)
    # Tests without regularization
    reg = 0.
    unmixing1, C1_emb, OT, reconstruction1 = ot.gromov.gromov_wasserstein_linear_unmixing(
        C1, Cdict, reg=reg, p=p, q=p,
        tol_outer=tol, tol_inner=tol, max_iter_outer=20, max_iter_inner=200
    )

    unmixing1b, C1b_emb, OTb, reconstruction1b = ot.gromov.gromov_wasserstein_linear_unmixing(
        C1b, Cdictb, reg=reg, p=None, q=None,
        tol_outer=tol, tol_inner=tol, max_iter_outer=20, max_iter_inner=200
    )

    unmixing2, C2_emb, OT, reconstruction2 = ot.gromov.gromov_wasserstein_linear_unmixing(
        C2, Cdict, reg=reg, p=None, q=None,
        tol_outer=tol, tol_inner=tol, max_iter_outer=20, max_iter_inner=200
    )

    unmixing2b, C2b_emb, OTb, reconstruction2b = ot.gromov.gromov_wasserstein_linear_unmixing(
        C2b, Cdictb, reg=reg, p=pb, q=pb,
        tol_outer=tol, tol_inner=tol, max_iter_outer=20, max_iter_inner=200
    )

    np.testing.assert_allclose(unmixing1, nx.to_numpy(unmixing1b), atol=5e-06)
    np.testing.assert_allclose(unmixing1, [1., 0.], atol=5e-01)
    np.testing.assert_allclose(unmixing2, nx.to_numpy(unmixing2b), atol=5e-06)
    np.testing.assert_allclose(unmixing2, [0., 1.], atol=5e-01)
    np.testing.assert_allclose(C1_emb, nx.to_numpy(C1b_emb), atol=1e-06)
    np.testing.assert_allclose(C2_emb, nx.to_numpy(C2b_emb), atol=1e-06)
    np.testing.assert_allclose(reconstruction1, nx.to_numpy(reconstruction1b), atol=1e-06)
    np.testing.assert_allclose(reconstruction2, nx.to_numpy(reconstruction2b), atol=1e-06)
    np.testing.assert_allclose(C1b_emb.shape, (n, n))
    np.testing.assert_allclose(C2b_emb.shape, (n, n))

    # Tests with regularization

    reg = 0.001
    unmixing1, C1_emb, OT, reconstruction1 = ot.gromov.gromov_wasserstein_linear_unmixing(
        C1, Cdict, reg=reg, p=p, q=p,
        tol_outer=tol, tol_inner=tol, max_iter_outer=20, max_iter_inner=200
    )

    unmixing1b, C1b_emb, OTb, reconstruction1b = ot.gromov.gromov_wasserstein_linear_unmixing(
        C1b, Cdictb, reg=reg, p=None, q=None,
        tol_outer=tol, tol_inner=tol, max_iter_outer=20, max_iter_inner=200
    )

    unmixing2, C2_emb, OT, reconstruction2 = ot.gromov.gromov_wasserstein_linear_unmixing(
        C2, Cdict, reg=reg, p=None, q=None,
        tol_outer=tol, tol_inner=tol, max_iter_outer=20, max_iter_inner=200
    )

    unmixing2b, C2b_emb, OTb, reconstruction2b = ot.gromov.gromov_wasserstein_linear_unmixing(
        C2b, Cdictb, reg=reg, p=pb, q=pb,
        tol_outer=tol, tol_inner=tol, max_iter_outer=20, max_iter_inner=200
    )

    np.testing.assert_allclose(unmixing1, nx.to_numpy(unmixing1b), atol=1e-06)
    np.testing.assert_allclose(unmixing1, [1., 0.], atol=1e-01)
    np.testing.assert_allclose(unmixing2, nx.to_numpy(unmixing2b), atol=1e-06)
    np.testing.assert_allclose(unmixing2, [0., 1.], atol=1e-01)
    np.testing.assert_allclose(C1_emb, nx.to_numpy(C1b_emb), atol=1e-06)
    np.testing.assert_allclose(C2_emb, nx.to_numpy(C2b_emb), atol=1e-06)
    np.testing.assert_allclose(reconstruction1, nx.to_numpy(reconstruction1b), atol=1e-06)
    np.testing.assert_allclose(reconstruction2, nx.to_numpy(reconstruction2b), atol=1e-06)
    np.testing.assert_allclose(C1b_emb.shape, (n, n))
    np.testing.assert_allclose(C2b_emb.shape, (n, n))


def test_gromov_wasserstein_dictionary_learning(nx):

    # create dataset composed from 2 structures which are repeated 5 times
    shape = 4
    n_samples = 2
    n_atoms = 2
    projection = 'nonnegative_symmetric'
    X1, y1 = ot.datasets.make_data_classif('3gauss', shape, random_state=42)
    X2, y2 = ot.datasets.make_data_classif('3gauss2', shape, random_state=42)
    C1 = ot.dist(X1)
    C2 = ot.dist(X2)
    Cs = [C1.copy() for _ in range(n_samples // 2)] + [C2.copy() for _ in range(n_samples // 2)]
    ps = [ot.unif(shape) for _ in range(n_samples)]
    q = ot.unif(shape)

    # Provide initialization for the graph dictionary of shape (n_atoms, shape, shape)
    # following the same procedure than implemented in gromov_wasserstein_dictionary_learning.
    dataset_means = [C.mean() for C in Cs]
    np.random.seed(0)
    Cdict_init = np.random.normal(loc=np.mean(dataset_means), scale=np.std(dataset_means), size=(n_atoms, shape, shape))

    if projection == 'nonnegative_symmetric':
        Cdict_init = 0.5 * (Cdict_init + Cdict_init.transpose((0, 2, 1)))
        Cdict_init[Cdict_init < 0.] = 0.

    Csb = nx.from_numpy(*Cs)
    psb = nx.from_numpy(*ps)
    qb, Cdict_initb = nx.from_numpy(q, Cdict_init)

    # Test: compare reconstruction error using initial dictionary and dictionary learned using this initialization
    # > Compute initial reconstruction of samples on this random dictionary without backend
    use_adam_optimizer = True
    verbose = False
    tol = 10**(-5)
    epochs = 1

    initial_total_reconstruction = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Cs[i], Cdict_init, p=ps[i], q=q, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        initial_total_reconstruction += reconstruction

    # > Learn the dictionary using this init
    Cdict, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Cs, D=n_atoms, nt=shape, ps=ps, q=q, Cdict_init=Cdict_init,
        epochs=epochs, batch_size=2 * n_samples, learning_rate=1., reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )
    # > Compute reconstruction of samples on learned dictionary without backend
    total_reconstruction = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Cs[i], Cdict, p=None, q=None, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction += reconstruction

    np.testing.assert_array_less(total_reconstruction, initial_total_reconstruction)

    # Test: Perform same experiments after going through backend

    Cdictb, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Csb, D=n_atoms, nt=shape, ps=None, q=None, Cdict_init=Cdict_initb,
        epochs=epochs, batch_size=n_samples, learning_rate=1., reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )
    # Compute reconstruction of samples on learned dictionary
    total_reconstruction_b = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Csb[i], Cdictb, p=psb[i], q=qb, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction_b += reconstruction

    total_reconstruction_b = nx.to_numpy(total_reconstruction_b)
    np.testing.assert_array_less(total_reconstruction_b, initial_total_reconstruction)
    np.testing.assert_allclose(total_reconstruction_b, total_reconstruction, atol=1e-05)
    np.testing.assert_allclose(total_reconstruction_b, total_reconstruction, atol=1e-05)
    np.testing.assert_allclose(Cdict, nx.to_numpy(Cdictb), atol=1e-03)

    # Test: Perform same comparison without providing the initial dictionary being an optional input
    #       knowing than the initialization scheme is the same than implemented to set the benchmarked initialization.
    np.random.seed(0)
    Cdict_bis, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Cs, D=n_atoms, nt=shape, ps=None, q=None, Cdict_init=None,
        epochs=epochs, batch_size=n_samples, learning_rate=1., reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_bis = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Cs[i], Cdict_bis, p=ps[i], q=q, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction_bis += reconstruction

    np.testing.assert_allclose(total_reconstruction_bis, total_reconstruction, atol=1e-05)

    # Test: Same after going through backend
    np.random.seed(0)
    Cdictb_bis, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Csb, D=n_atoms, nt=shape, ps=psb, q=qb, Cdict_init=None,
        epochs=epochs, batch_size=n_samples, learning_rate=1., reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_b_bis = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Csb[i], Cdictb_bis, p=None, q=None, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction_b_bis += reconstruction

    total_reconstruction_b_bis = nx.to_numpy(total_reconstruction_b_bis)
    np.testing.assert_allclose(total_reconstruction_b_bis, total_reconstruction_b, atol=1e-05)
    np.testing.assert_allclose(Cdict_bis, nx.to_numpy(Cdictb_bis), atol=1e-03)

    # Test: Perform same comparison without providing the initial dictionary being an optional input
    #       and testing other optimization settings untested until now.
    #       We pass previously estimated dictionaries to speed up the process.
    use_adam_optimizer = False
    verbose = True
    use_log = True

    np.random.seed(0)
    Cdict_bis2, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Cs, D=n_atoms, nt=shape, ps=ps, q=q, Cdict_init=Cdict,
        epochs=epochs, batch_size=n_samples, learning_rate=10., reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=use_log, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_bis2 = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Cs[i], Cdict_bis2, p=ps[i], q=q, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction_bis2 += reconstruction

    np.testing.assert_array_less(total_reconstruction_bis2, total_reconstruction)

    # Test: Same after going through backend
    np.random.seed(0)
    Cdictb_bis2, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Csb, D=n_atoms, nt=shape, ps=psb, q=qb, Cdict_init=Cdictb,
        epochs=epochs, batch_size=n_samples, learning_rate=10., reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=use_log, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_b_bis2 = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Csb[i], Cdictb_bis2, p=psb[i], q=qb, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction_b_bis2 += reconstruction

    total_reconstruction_b_bis2 = nx.to_numpy(total_reconstruction_b_bis2)
    np.testing.assert_allclose(total_reconstruction_b_bis2, total_reconstruction_bis2, atol=1e-05)


def test_fused_gromov_wasserstein_linear_unmixing(nx):

    n = 4
    X1, y1 = ot.datasets.make_data_classif('3gauss', n, random_state=42)
    X2, y2 = ot.datasets.make_data_classif('3gauss2', n, random_state=42)
    F, y = ot.datasets.make_data_classif('3gauss', n, random_state=42)

    C1 = ot.dist(X1)
    C2 = ot.dist(X2)
    Cdict = np.stack([C1, C2])
    Ydict = np.stack([F, F])
    p = ot.unif(n)

    C1b, C2b, Fb, Cdictb, Ydictb, pb = nx.from_numpy(C1, C2, F, Cdict, Ydict, p)

    # Tests without regularization
    reg = 0.

    unmixing1, C1_emb, Y1_emb, OT, reconstruction1 = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
        C1, F, Cdict, Ydict, p=p, q=p, alpha=0.5, reg=reg,
        tol_outer=10**(-6), tol_inner=10**(-6), max_iter_outer=10, max_iter_inner=50
    )

    unmixing1b, C1b_emb, Y1b_emb, OTb, reconstruction1b = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
        C1b, Fb, Cdictb, Ydictb, p=None, q=None, alpha=0.5, reg=reg,
        tol_outer=10**(-6), tol_inner=10**(-6), max_iter_outer=10, max_iter_inner=50
    )

    unmixing2, C2_emb, Y2_emb, OT, reconstruction2 = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
        C2, F, Cdict, Ydict, p=None, q=None, alpha=0.5, reg=reg,
        tol_outer=10**(-6), tol_inner=10**(-6), max_iter_outer=10, max_iter_inner=50
    )

    unmixing2b, C2b_emb, Y2b_emb, OTb, reconstruction2b = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
        C2b, Fb, Cdictb, Ydictb, p=pb, q=pb, alpha=0.5, reg=reg,
        tol_outer=10**(-6), tol_inner=10**(-6), max_iter_outer=10, max_iter_inner=50
    )

    np.testing.assert_allclose(unmixing1, nx.to_numpy(unmixing1b), atol=4e-06)
    np.testing.assert_allclose(unmixing1, [1., 0.], atol=4e-01)
    np.testing.assert_allclose(unmixing2, nx.to_numpy(unmixing2b), atol=4e-06)
    np.testing.assert_allclose(unmixing2, [0., 1.], atol=4e-01)
    np.testing.assert_allclose(C1_emb, nx.to_numpy(C1b_emb), atol=1e-03)
    np.testing.assert_allclose(C2_emb, nx.to_numpy(C2b_emb), atol=1e-03)
    np.testing.assert_allclose(Y1_emb, nx.to_numpy(Y1b_emb), atol=1e-03)
    np.testing.assert_allclose(Y2_emb, nx.to_numpy(Y2b_emb), atol=1e-03)
    np.testing.assert_allclose(reconstruction1, nx.to_numpy(reconstruction1b), atol=1e-06)
    np.testing.assert_allclose(reconstruction2, nx.to_numpy(reconstruction2b), atol=1e-06)
    np.testing.assert_allclose(C1b_emb.shape, (n, n))
    np.testing.assert_allclose(C2b_emb.shape, (n, n))

    # Tests with regularization
    reg = 0.001

    unmixing1, C1_emb, Y1_emb, OT, reconstruction1 = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
        C1, F, Cdict, Ydict, p=p, q=p, alpha=0.5, reg=reg,
        tol_outer=10**(-6), tol_inner=10**(-6), max_iter_outer=10, max_iter_inner=50
    )

    unmixing1b, C1b_emb, Y1b_emb, OTb, reconstruction1b = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
        C1b, Fb, Cdictb, Ydictb, p=None, q=None, alpha=0.5, reg=reg,
        tol_outer=10**(-6), tol_inner=10**(-6), max_iter_outer=10, max_iter_inner=50
    )

    unmixing2, C2_emb, Y2_emb, OT, reconstruction2 = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
        C2, F, Cdict, Ydict, p=None, q=None, alpha=0.5, reg=reg,
        tol_outer=10**(-6), tol_inner=10**(-6), max_iter_outer=10, max_iter_inner=50
    )

    unmixing2b, C2b_emb, Y2b_emb, OTb, reconstruction2b = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
        C2b, Fb, Cdictb, Ydictb, p=pb, q=pb, alpha=0.5, reg=reg,
        tol_outer=10**(-6), tol_inner=10**(-6), max_iter_outer=10, max_iter_inner=50
    )

    np.testing.assert_allclose(unmixing1, nx.to_numpy(unmixing1b), atol=1e-06)
    np.testing.assert_allclose(unmixing1, [1., 0.], atol=1e-01)
    np.testing.assert_allclose(unmixing2, nx.to_numpy(unmixing2b), atol=1e-06)
    np.testing.assert_allclose(unmixing2, [0., 1.], atol=1e-01)
    np.testing.assert_allclose(C1_emb, nx.to_numpy(C1b_emb), atol=1e-03)
    np.testing.assert_allclose(C2_emb, nx.to_numpy(C2b_emb), atol=1e-03)
    np.testing.assert_allclose(Y1_emb, nx.to_numpy(Y1b_emb), atol=1e-03)
    np.testing.assert_allclose(Y2_emb, nx.to_numpy(Y2b_emb), atol=1e-03)
    np.testing.assert_allclose(reconstruction1, nx.to_numpy(reconstruction1b), atol=1e-06)
    np.testing.assert_allclose(reconstruction2, nx.to_numpy(reconstruction2b), atol=1e-06)
    np.testing.assert_allclose(C1b_emb.shape, (n, n))
    np.testing.assert_allclose(C2b_emb.shape, (n, n))


def test_fused_gromov_wasserstein_dictionary_learning(nx):

    # create dataset composed from 2 structures which are repeated 5 times
    shape = 4
    n_samples = 2
    n_atoms = 2
    projection = 'nonnegative_symmetric'
    X1, y1 = ot.datasets.make_data_classif('3gauss', shape, random_state=42)
    X2, y2 = ot.datasets.make_data_classif('3gauss2', shape, random_state=42)
    F, y = ot.datasets.make_data_classif('3gauss', shape, random_state=42)

    C1 = ot.dist(X1)
    C2 = ot.dist(X2)
    Cs = [C1.copy() for _ in range(n_samples // 2)] + [C2.copy() for _ in range(n_samples // 2)]
    Ys = [F.copy() for _ in range(n_samples)]
    ps = [ot.unif(shape) for _ in range(n_samples)]
    q = ot.unif(shape)

    # Provide initialization for the graph dictionary of shape (n_atoms, shape, shape)
    # following the same procedure than implemented in gromov_wasserstein_dictionary_learning.
    dataset_structure_means = [C.mean() for C in Cs]
    np.random.seed(0)
    Cdict_init = np.random.normal(loc=np.mean(dataset_structure_means), scale=np.std(dataset_structure_means), size=(n_atoms, shape, shape))
    if projection == 'nonnegative_symmetric':
        Cdict_init = 0.5 * (Cdict_init + Cdict_init.transpose((0, 2, 1)))
        Cdict_init[Cdict_init < 0.] = 0.
    dataset_feature_means = np.stack([Y.mean(axis=0) for Y in Ys])
    Ydict_init = np.random.normal(loc=dataset_feature_means.mean(axis=0), scale=dataset_feature_means.std(axis=0), size=(n_atoms, shape, 2))

    Csb = nx.from_numpy(*Cs)
    Ysb = nx.from_numpy(*Ys)
    psb = nx.from_numpy(*ps)
    qb, Cdict_initb, Ydict_initb = nx.from_numpy(q, Cdict_init, Ydict_init)

    # Test: Compute initial reconstruction of samples on this random dictionary
    alpha = 0.5
    use_adam_optimizer = True
    verbose = False
    tol = 1e-05
    epochs = 1

    initial_total_reconstruction = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Cs[i], Ys[i], Cdict_init, Ydict_init, p=ps[i], q=q,
            alpha=alpha, reg=0., tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        initial_total_reconstruction += reconstruction

    # > Learn a dictionary using this given initialization and check that the reconstruction loss
    # on the learned dictionary is lower than the one using its initialization.
    Cdict, Ydict, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Cs, Ys, D=n_atoms, nt=shape, ps=ps, q=q, Cdict_init=Cdict_init, Ydict_init=Ydict_init,
        epochs=epochs, batch_size=n_samples, learning_rate_C=1., learning_rate_Y=1., alpha=alpha, reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Cs[i], Ys[i], Cdict, Ydict, p=None, q=None, alpha=alpha, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction += reconstruction
    # Compare both
    np.testing.assert_array_less(total_reconstruction, initial_total_reconstruction)

    # Test: Perform same experiments after going through backend

    Cdictb, Ydictb, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Csb, Ysb, D=n_atoms, nt=shape, ps=None, q=None, Cdict_init=Cdict_initb, Ydict_init=Ydict_initb,
        epochs=epochs, batch_size=2 * n_samples, learning_rate_C=1., learning_rate_Y=1., alpha=alpha, reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_b = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Csb[i], Ysb[i], Cdictb, Ydictb, p=psb[i], q=qb, alpha=alpha, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction_b += reconstruction

    total_reconstruction_b = nx.to_numpy(total_reconstruction_b)
    np.testing.assert_array_less(total_reconstruction_b, initial_total_reconstruction)
    np.testing.assert_allclose(total_reconstruction_b, total_reconstruction, atol=1e-05)
    np.testing.assert_allclose(Cdict, nx.to_numpy(Cdictb), atol=1e-03)
    np.testing.assert_allclose(Ydict, nx.to_numpy(Ydictb), atol=1e-03)

    # Test: Perform similar experiment without providing the initial dictionary being an optional input
    np.random.seed(0)
    Cdict_bis, Ydict_bis, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Cs, Ys, D=n_atoms, nt=shape, ps=None, q=None, Cdict_init=None, Ydict_init=None,
        epochs=epochs, batch_size=n_samples, learning_rate_C=1., learning_rate_Y=1., alpha=alpha, reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_bis = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Cs[i], Ys[i], Cdict_bis, Ydict_bis, p=ps[i], q=q, alpha=alpha, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction_bis += reconstruction

    np.testing.assert_allclose(total_reconstruction_bis, total_reconstruction, atol=1e-05)

    # > Same after going through backend
    np.random.seed(0)
    Cdictb_bis, Ydictb_bis, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Csb, Ysb, D=n_atoms, nt=shape, ps=None, q=None, Cdict_init=None, Ydict_init=None,
        epochs=epochs, batch_size=n_samples, learning_rate_C=1., learning_rate_Y=1., alpha=alpha, reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )

    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_b_bis = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Csb[i], Ysb[i], Cdictb_bis, Ydictb_bis, p=psb[i], q=qb, alpha=alpha, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction_b_bis += reconstruction

    total_reconstruction_b_bis = nx.to_numpy(total_reconstruction_b_bis)
    np.testing.assert_allclose(total_reconstruction_b_bis, total_reconstruction_b, atol=1e-05)

    # Test: without using adam optimizer, with log and verbose set to True
    use_adam_optimizer = False
    verbose = True
    use_log = True

    # > Experiment providing previously estimated dictionary to speed up the test compared to providing initial random init.
    np.random.seed(0)
    Cdict_bis2, Ydict_bis2, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Cs, Ys, D=n_atoms, nt=shape, ps=ps, q=q, Cdict_init=Cdict, Ydict_init=Ydict,
        epochs=epochs, batch_size=n_samples, learning_rate_C=10., learning_rate_Y=10., alpha=alpha, reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=use_log, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_bis2 = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Cs[i], Ys[i], Cdict_bis2, Ydict_bis2, p=ps[i], q=q, alpha=alpha, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction_bis2 += reconstruction

    np.testing.assert_array_less(total_reconstruction_bis2, total_reconstruction)

    # > Same after going through backend
    np.random.seed(0)
    Cdictb_bis2, Ydictb_bis2, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Csb, Ysb, D=n_atoms, nt=shape, ps=None, q=None, Cdict_init=Cdictb, Ydict_init=Ydictb,
        epochs=epochs, batch_size=n_samples, learning_rate_C=10., learning_rate_Y=10., alpha=alpha, reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=use_log, use_adam_optimizer=use_adam_optimizer, verbose=verbose
    )

    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_b_bis2 = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Csb[i], Ysb[i], Cdictb_bis2, Ydictb_bis2, p=None, q=None, alpha=alpha, reg=0.,
            tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50
        )
        total_reconstruction_b_bis2 += reconstruction

    # > Compare results with/without backend
    total_reconstruction_b_bis2 = nx.to_numpy(total_reconstruction_b_bis2)
    np.testing.assert_allclose(total_reconstruction_bis2, total_reconstruction_b_bis2, atol=1e-05)
