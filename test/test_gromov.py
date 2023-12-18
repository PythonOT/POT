"""Tests for module gromov  """

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np
import pytest
import warnings

import ot
from ot.backend import NumpyBackend
from ot.backend import torch, tf


def test_gromov(nx):
    n_samples = 20  # nb samples
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=1)
    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    G = ot.gromov.gromov_wasserstein(
        C1, C2, None, q, 'square_loss', G0=G0, verbose=True,
        alpha_min=0., alpha_max=1.)
    Gb = nx.to_numpy(ot.gromov.gromov_wasserstein(
        C1b, C2b, pb, None, 'square_loss', symmetric=True, G0=G0b, verbose=True))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    Id = (1 / (1.0 * n_samples)) * np.eye(n_samples, n_samples)

    np.testing.assert_allclose(Gb, np.flipud(Id), atol=1e-04)
    for armijo in [False, True]:
        gw, log = ot.gromov.gromov_wasserstein2(C1, C2, None, q, 'kl_loss', armijo=armijo, log=True)
        gwb, logb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, None, 'kl_loss', armijo=armijo, log=True)
        gwb = nx.to_numpy(gwb)

        gw_val = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'kl_loss', armijo=armijo, G0=G0, log=False)
        gw_valb = nx.to_numpy(
            ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', armijo=armijo, G0=G0b, log=False)
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


def test_asymmetric_gromov(nx):
    n_samples = 20  # nb samples
    rng = np.random.RandomState(0)
    C1 = rng.uniform(low=0., high=10, size=(n_samples, n_samples))
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    C2 = C1[idx, :][:, idx]

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    G, log = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', G0=G0, log=True, symmetric=False, verbose=True)
    Gb, logb = ot.gromov.gromov_wasserstein(C1b, C2b, pb, qb, 'square_loss', log=True, symmetric=False, G0=G0b, verbose=True)
    Gb = nx.to_numpy(Gb)
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    np.testing.assert_allclose(log['gw_dist'], 0., atol=1e-04)
    np.testing.assert_allclose(logb['gw_dist'], 0., atol=1e-04)

    gw, log = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'square_loss', G0=G0, log=True, symmetric=False, verbose=True)
    gwb, logb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'square_loss', log=True, symmetric=False, G0=G0b, verbose=True)

    G = log['T']
    Gb = nx.to_numpy(logb['T'])
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    np.testing.assert_allclose(log['gw_dist'], 0., atol=1e-04)
    np.testing.assert_allclose(logb['gw_dist'], 0., atol=1e-04)


def test_gromov_integer_warnings(nx):
    n_samples = 10  # nb samples
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=1)
    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()
    C1 = C1.astype(np.int32)
    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    G = ot.gromov.gromov_wasserstein(
        C1, C2, None, q, 'square_loss', G0=G0, verbose=True,
        alpha_min=0., alpha_max=1.)
    Gb = nx.to_numpy(ot.gromov.gromov_wasserstein(
        C1b, C2b, pb, None, 'square_loss', symmetric=True, G0=G0b, verbose=True))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(G, 0., atol=1e-09)


def test_gromov_dtype_device(nx):
    # setup
    n_samples = 20  # nb samples

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

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            Gb = ot.gromov.gromov_wasserstein(C1b, C2b, pb, qb, 'square_loss', G0=G0b, verbose=True)
            gw_valb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', armijo=True, G0=G0b, log=False)

        nx.assert_same_dtype_device(C1b, Gb)
        nx.assert_same_dtype_device(C1b, gw_valb)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_gromov_device_tf():
    nx = ot.backend.TensorflowBackend()
    n_samples = 20  # nb samples
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
        gw_valb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', armijo=True, G0=G0b, log=False)
        nx.assert_same_dtype_device(C1b, Gb)
        nx.assert_same_dtype_device(C1b, gw_valb)

    if len(tf.config.list_physical_devices('GPU')) > 0:
        # Check that everything happens on the GPU
        C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)
        Gb = ot.gromov.gromov_wasserstein(C1b, C2b, pb, qb, 'square_loss', verbose=True)
        gw_valb = ot.gromov.gromov_wasserstein2(C1b, C2b, pb, qb, 'kl_loss', armijo=True, log=False)
        nx.assert_same_dtype_device(C1b, Gb)
        nx.assert_same_dtype_device(C1b, gw_valb)
        assert nx.dtype_device(Gb)[1].startswith("GPU")


def test_gromov2_gradients():
    n_samples = 20  # nb samples

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

            # classical gradients
            p1 = torch.tensor(p, requires_grad=True, device=device)
            q1 = torch.tensor(q, requires_grad=True, device=device)
            C11 = torch.tensor(C1, requires_grad=True, device=device)
            C12 = torch.tensor(C2, requires_grad=True, device=device)

            # Test with exact line-search
            val = ot.gromov_wasserstein2(C11, C12, p1, q1)

            val.backward()

            assert val.device == p1.device
            assert q1.shape == q1.grad.shape
            assert p1.shape == p1.grad.shape
            assert C11.shape == C11.grad.shape
            assert C12.shape == C12.grad.shape

            # Test with armijo line-search
            # classical gradients
            p1 = torch.tensor(p, requires_grad=True, device=device)
            q1 = torch.tensor(q, requires_grad=True, device=device)
            C11 = torch.tensor(C1, requires_grad=True, device=device)
            C12 = torch.tensor(C2, requires_grad=True, device=device)

            q1.grad = None
            p1.grad = None
            C11.grad = None
            C12.grad = None
            val = ot.gromov_wasserstein2(C11, C12, p1, q1, armijo=True)

            val.backward()

            assert val.device == p1.device
            assert q1.shape == q1.grad.shape
            assert p1.shape == p1.grad.shape
            assert C11.shape == C11.grad.shape
            assert C12.shape == C12.grad.shape


def test_gw_helper_backend(nx):
    n_samples = 10  # nb samples

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=0)
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=1)

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)
    Gb, logb = ot.gromov.gromov_wasserstein(C1b, C2b, pb, qb, 'square_loss', armijo=False, symmetric=True, G0=G0b, log=True)

    # calls with nx=None
    constCb, hC1b, hC2b = ot.gromov.init_matrix(C1b, C2b, pb, qb, loss_fun='square_loss')

    def f(G):
        return ot.gromov.gwloss(constCb, hC1b, hC2b, G, None)

    def df(G):
        return ot.gromov.gwggrad(constCb, hC1b, hC2b, G, None)

    def line_search(cost, G, deltaG, Mi, cost_G):
        return ot.gromov.solve_gromov_linesearch(G, deltaG, cost_G, C1b, C2b, M=0., reg=1., nx=None)
    # feed the precomputed local optimum Gb to cg
    res, log = ot.optim.cg(pb, qb, 0., 1., f, df, Gb, line_search, log=True, numItermax=1e4, stopThr=1e-9, stopThr2=1e-9)
    # check constraints
    np.testing.assert_allclose(res, Gb, atol=1e-06)


@pytest.mark.parametrize('loss_fun', [
    'square_loss',
    'kl_loss',
    pytest.param('unknown_loss', marks=pytest.mark.xfail(raises=ValueError)),
])
def test_gw_helper_validation(loss_fun):
    n_samples = 10  # nb samples
    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=0)
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=1)
    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    ot.gromov.init_matrix(C1, C2, p, q, loss_fun=loss_fun)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
@pytest.mark.parametrize('loss_fun', [
    'square_loss',
    'kl_loss',
    pytest.param('unknown_loss', marks=pytest.mark.xfail(raises=ValueError)),
])
def test_entropic_gromov(nx, loss_fun):
    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    G, log = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, None, q, loss_fun, symmetric=None, G0=G0,
        epsilon=1e-2, max_iter=10, verbose=True, log=True)
    Gb = nx.to_numpy(ot.gromov.entropic_gromov_wasserstein(
        C1b, C2b, pb, None, loss_fun, symmetric=True, G0=None,
        epsilon=1e-2, max_iter=10, verbose=True, log=False
    ))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
@pytest.mark.parametrize('loss_fun', [
    'square_loss',
    'kl_loss',
    pytest.param('unknown_loss', marks=pytest.mark.xfail(raises=ValueError)),
])
def test_entropic_gromov2(nx, loss_fun):
    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    gw, log = ot.gromov.entropic_gromov_wasserstein2(
        C1, C2, p, None, loss_fun, symmetric=True, G0=None,
        max_iter=10, epsilon=1e-2, log=True)
    gwb, logb = ot.gromov.entropic_gromov_wasserstein2(
        C1b, C2b, None, qb, loss_fun, symmetric=None, G0=G0b,
        max_iter=10, epsilon=1e-2, log=True)
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


@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_proximal_gromov(nx):
    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    with pytest.raises(ValueError):
        loss_fun = 'weird_loss_fun'
        G, log = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, None, q, loss_fun, symmetric=None, G0=G0,
            epsilon=1e-1, max_iter=10, solver='PPA', verbose=True, log=True, numItermax=1)

    G, log = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, None, q, 'square_loss', symmetric=None, G0=G0,
        epsilon=1e-1, max_iter=10, solver='PPA', verbose=True, log=True, numItermax=1)
    Gb = nx.to_numpy(ot.gromov.entropic_gromov_wasserstein(
        C1b, C2b, pb, None, 'square_loss', symmetric=True, G0=None,
        epsilon=1e-1, max_iter=10, solver='PPA', verbose=True, log=False, numItermax=1
    ))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-02)  # cf convergence gromov

    gw, log = ot.gromov.entropic_gromov_wasserstein2(
        C1, C2, p, q, 'kl_loss', symmetric=True, G0=None,
        max_iter=10, epsilon=1e-1, solver='PPA', warmstart=True, log=True)
    gwb, logb = ot.gromov.entropic_gromov_wasserstein2(
        C1b, C2b, pb, qb, 'kl_loss', symmetric=None, G0=G0b,
        max_iter=10, epsilon=1e-1, solver='PPA', warmstart=True, log=True)
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
        q, Gb.sum(0), atol=1e-02)  # cf convergence gromov


@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_asymmetric_entropic_gromov(nx):
    n_samples = 10  # nb samples
    rng = np.random.RandomState(0)
    C1 = rng.uniform(low=0., high=10, size=(n_samples, n_samples))
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    C2 = C1[idx, :][:, idx]

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)
    G = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q, 'square_loss', symmetric=None, G0=G0,
        epsilon=1e-1, max_iter=5, verbose=True, log=False)
    Gb = nx.to_numpy(ot.gromov.entropic_gromov_wasserstein(
        C1b, C2b, pb, qb, 'square_loss', symmetric=False, G0=None,
        epsilon=1e-1, max_iter=5, verbose=True, log=False
    ))
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    gw = ot.gromov.entropic_gromov_wasserstein2(
        C1, C2, None, None, 'kl_loss', symmetric=False, G0=None,
        max_iter=5, epsilon=1e-1, log=False)
    gwb = ot.gromov.entropic_gromov_wasserstein2(
        C1b, C2b, pb, qb, 'kl_loss', symmetric=None, G0=G0b,
        max_iter=5, epsilon=1e-1, log=False)
    gwb = nx.to_numpy(gwb)

    np.testing.assert_allclose(gw, gwb, atol=1e-06)
    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)


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

        for solver in ['PGD', 'PPA', 'BAPG']:
            if solver == 'BAPG':
                Gb = ot.gromov.BAPG_gromov_wasserstein(
                    C1b, C2b, pb, qb, max_iter=2, verbose=True)
                gw_valb = ot.gromov.BAPG_gromov_wasserstein2(
                    C1b, C2b, pb, qb, max_iter=2, verbose=True)
            else:
                Gb = ot.gromov.entropic_gromov_wasserstein(
                    C1b, C2b, pb, qb, max_iter=2, solver=solver, verbose=True)
                gw_valb = ot.gromov.entropic_gromov_wasserstein2(
                    C1b, C2b, pb, qb, max_iter=2, solver=solver, verbose=True)

            nx.assert_same_dtype_device(C1b, Gb)
            nx.assert_same_dtype_device(C1b, gw_valb)


def test_BAPG_gromov(nx):
    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    # complete test with marginal loss = True
    marginal_loss = True
    with pytest.raises(ValueError):
        loss_fun = 'weird_loss_fun'
        G, log = ot.gromov.BAPG_gromov_wasserstein(
            C1, C2, None, q, loss_fun, symmetric=None, G0=G0,
            epsilon=1e-1, max_iter=10, marginal_loss=marginal_loss,
            verbose=True, log=True)

    G, log = ot.gromov.BAPG_gromov_wasserstein(
        C1, C2, None, q, 'square_loss', symmetric=None, G0=G0,
        epsilon=1e-1, max_iter=10, marginal_loss=marginal_loss,
        verbose=True, log=True)
    Gb = nx.to_numpy(ot.gromov.BAPG_gromov_wasserstein(
        C1b, C2b, pb, None, 'square_loss', symmetric=True, G0=None,
        epsilon=1e-1, max_iter=10, marginal_loss=marginal_loss, verbose=True,
        log=False
    ))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-02)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-02)  # cf convergence gromov

    with pytest.warns(UserWarning):

        gw = ot.gromov.BAPG_gromov_wasserstein2(
            C1, C2, p, q, 'kl_loss', symmetric=False, G0=None,
            max_iter=10, epsilon=1e-2, marginal_loss=marginal_loss, log=False)

    gw, log = ot.gromov.BAPG_gromov_wasserstein2(
        C1, C2, p, q, 'kl_loss', symmetric=False, G0=None,
        max_iter=10, epsilon=1., marginal_loss=marginal_loss, log=True)
    gwb, logb = ot.gromov.BAPG_gromov_wasserstein2(
        C1b, C2b, pb, qb, 'kl_loss', symmetric=None, G0=G0b,
        max_iter=10, epsilon=1., marginal_loss=marginal_loss, log=True)
    gwb = nx.to_numpy(gwb)

    G = log['T']
    Gb = nx.to_numpy(logb['T'])

    np.testing.assert_allclose(gw, gwb, atol=1e-06)
    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-02)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-02)  # cf convergence gromov

    marginal_loss = False
    G, log = ot.gromov.BAPG_gromov_wasserstein(
        C1, C2, None, q, 'square_loss', symmetric=None, G0=G0,
        epsilon=1e-1, max_iter=10, marginal_loss=marginal_loss,
        verbose=True, log=True)
    Gb = nx.to_numpy(ot.gromov.BAPG_gromov_wasserstein(
        C1b, C2b, pb, None, 'square_loss', symmetric=False, G0=None,
        epsilon=1e-1, max_iter=10, marginal_loss=marginal_loss, verbose=True,
        log=False
    ))


@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_fgw(nx):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    rng = np.random.RandomState(42)
    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)

    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)

    with pytest.raises(ValueError):
        loss_fun = 'weird_loss_fun'
        G, log = ot.gromov.entropic_fused_gromov_wasserstein(
            M, C1, C2, None, None, loss_fun, symmetric=None, G0=G0,
            epsilon=1e-1, max_iter=10, verbose=True, log=True)

    G, log = ot.gromov.entropic_fused_gromov_wasserstein(
        M, C1, C2, None, None, 'square_loss', symmetric=None, G0=G0,
        epsilon=1e-1, max_iter=10, verbose=True, log=True)
    Gb = nx.to_numpy(ot.gromov.entropic_fused_gromov_wasserstein(
        Mb, C1b, C2b, pb, qb, 'square_loss', symmetric=True, G0=None,
        epsilon=1e-1, max_iter=10, verbose=True, log=False
    ))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    fgw, log = ot.gromov.entropic_fused_gromov_wasserstein2(
        M, C1, C2, p, q, 'kl_loss', symmetric=True, G0=None,
        max_iter=10, epsilon=1e-1, log=True)
    fgwb, logb = ot.gromov.entropic_fused_gromov_wasserstein2(
        Mb, C1b, C2b, pb, qb, 'kl_loss', symmetric=None, G0=G0b,
        max_iter=10, epsilon=1e-1, log=True)
    fgwb = nx.to_numpy(fgwb)

    G = log['T']
    Gb = nx.to_numpy(logb['T'])

    np.testing.assert_allclose(fgw, fgwb, atol=1e-06)
    np.testing.assert_allclose(fgw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_proximal_fgw(nx):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    rng = np.random.RandomState(42)
    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)

    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)

    G, log = ot.gromov.entropic_fused_gromov_wasserstein(
        M, C1, C2, p, q, 'square_loss', symmetric=None, G0=G0,
        epsilon=1e-1, max_iter=10, solver='PPA', verbose=True, log=True, numItermax=1)
    Gb = nx.to_numpy(ot.gromov.entropic_fused_gromov_wasserstein(
        Mb, C1b, C2b, pb, qb, 'square_loss', symmetric=True, G0=None,
        epsilon=1e-1, max_iter=10, solver='PPA', verbose=True, log=False, numItermax=1
    ))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    fgw, log = ot.gromov.entropic_fused_gromov_wasserstein2(
        M, C1, C2, p, None, 'kl_loss', symmetric=True, G0=None,
        max_iter=5, epsilon=1e-1, solver='PPA', warmstart=True, log=True)
    fgwb, logb = ot.gromov.entropic_fused_gromov_wasserstein2(
        Mb, C1b, C2b, None, qb, 'kl_loss', symmetric=None, G0=G0b,
        max_iter=5, epsilon=1e-1, solver='PPA', warmstart=True, log=True)
    fgwb = nx.to_numpy(fgwb)

    G = log['T']
    Gb = nx.to_numpy(logb['T'])

    np.testing.assert_allclose(fgw, fgwb, atol=1e-06)
    np.testing.assert_allclose(fgw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


def test_BAPG_fgw(nx):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    rng = np.random.RandomState(42)
    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)

    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)

    with pytest.raises(ValueError):
        loss_fun = 'weird_loss_fun'
        G, log = ot.gromov.BAPG_fused_gromov_wasserstein(
            M, C1, C2, p, q, loss_fun=loss_fun, max_iter=1, log=True)

    # complete test with marginal loss = True
    marginal_loss = True

    G, log = ot.gromov.BAPG_fused_gromov_wasserstein(
        M, C1, C2, p, q, 'square_loss', symmetric=None, G0=G0,
        epsilon=1e-1, max_iter=10, marginal_loss=marginal_loss, log=True)
    Gb = nx.to_numpy(ot.gromov.BAPG_fused_gromov_wasserstein(
        Mb, C1b, C2b, pb, qb, 'square_loss', symmetric=True, G0=None,
        epsilon=1e-1, max_iter=10, marginal_loss=marginal_loss, verbose=True))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-02)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-02)  # cf convergence gromov

    with pytest.warns(UserWarning):

        fgw = ot.gromov.BAPG_fused_gromov_wasserstein2(
            M, C1, C2, p, q, 'kl_loss', symmetric=False, G0=None,
            max_iter=10, epsilon=1e-3, marginal_loss=marginal_loss, log=False)

    fgw, log = ot.gromov.BAPG_fused_gromov_wasserstein2(
        M, C1, C2, p, None, 'kl_loss', symmetric=True, G0=None,
        max_iter=5, epsilon=1, marginal_loss=marginal_loss, log=True)
    fgwb, logb = ot.gromov.BAPG_fused_gromov_wasserstein2(
        Mb, C1b, C2b, None, qb, 'kl_loss', symmetric=None, G0=G0b,
        max_iter=5, epsilon=1, marginal_loss=marginal_loss, log=True)
    fgwb = nx.to_numpy(fgwb)

    G = log['T']
    Gb = nx.to_numpy(logb['T'])

    np.testing.assert_allclose(fgw, fgwb, atol=1e-06)
    np.testing.assert_allclose(fgw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-02)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-02)  # cf convergence gromov

    # Tests with marginal_loss = False
    marginal_loss = False
    G, log = ot.gromov.BAPG_fused_gromov_wasserstein(
        M, C1, C2, p, q, 'square_loss', symmetric=False, G0=G0,
        epsilon=1e-1, max_iter=10, marginal_loss=marginal_loss, log=True)
    Gb = nx.to_numpy(ot.gromov.BAPG_fused_gromov_wasserstein(
        Mb, C1b, C2b, pb, qb, 'square_loss', symmetric=None, G0=None,
        epsilon=1e-1, max_iter=10, marginal_loss=marginal_loss, verbose=True))
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-02)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-02)  # cf convergence gromov


def test_asymmetric_entropic_fgw(nx):
    n_samples = 5  # nb samples
    rng = np.random.RandomState(0)
    C1 = rng.uniform(low=0., high=10, size=(n_samples, n_samples))
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    C2 = C1[idx, :][:, idx]

    ys = rng.randn(n_samples, 2)
    yt = ys[idx, :]
    M = ot.dist(ys, yt)

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)
    G = ot.gromov.entropic_fused_gromov_wasserstein(
        M, C1, C2, p, q, 'square_loss', symmetric=None, G0=G0,
        max_iter=5, epsilon=1e-1, verbose=True, log=False)
    Gb = nx.to_numpy(ot.gromov.entropic_fused_gromov_wasserstein(
        Mb, C1b, C2b, pb, qb, 'square_loss', symmetric=False, G0=None,
        max_iter=5, epsilon=1e-1, verbose=True, log=False
    ))
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    fgw = ot.gromov.entropic_fused_gromov_wasserstein2(
        M, C1, C2, p, q, 'kl_loss', symmetric=False, G0=None,
        max_iter=5, epsilon=1e-1, log=False)
    fgwb = ot.gromov.entropic_fused_gromov_wasserstein2(
        Mb, C1b, C2b, pb, qb, 'kl_loss', symmetric=None, G0=G0b,
        max_iter=5, epsilon=1e-1, log=False)
    fgwb = nx.to_numpy(fgwb)

    np.testing.assert_allclose(fgw, fgwb, atol=1e-06)
    np.testing.assert_allclose(fgw, 0, atol=1e-1, rtol=1e-1)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_fgw_dtype_device(nx):
    # setup
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=rng)

    xt = xs[::-1].copy()

    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)
    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        Mb, C1b, C2b, pb, qb = nx.from_numpy(M, C1, C2, p, q, type_as=tp)

        for solver in ['PGD', 'PPA', 'BAPG']:
            if solver == 'BAPG':
                Gb = ot.gromov.BAPG_fused_gromov_wasserstein(
                    Mb, C1b, C2b, pb, qb, max_iter=2)
                fgw_valb = ot.gromov.BAPG_fused_gromov_wasserstein2(
                    Mb, C1b, C2b, pb, qb, max_iter=2)

            else:
                Gb = ot.gromov.entropic_fused_gromov_wasserstein(
                    Mb, C1b, C2b, pb, qb, max_iter=2, solver=solver)
                fgw_valb = ot.gromov.entropic_fused_gromov_wasserstein2(
                    Mb, C1b, C2b, pb, qb, max_iter=2, solver=solver)

            nx.assert_same_dtype_device(C1b, Gb)
            nx.assert_same_dtype_device(C1b, fgw_valb)


def test_entropic_fgw_barycenter(nx):
    ns = 5
    nt = 10

    rng = np.random.RandomState(42)
    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    ys = rng.randn(Xs.shape[0], 2)
    yt = rng.randn(Xt.shape[0], 2)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    p1 = ot.unif(ns)
    p2 = ot.unif(nt)
    n_samples = 3
    p = ot.unif(n_samples)

    ysb, ytb, C1b, C2b, p1b, p2b, pb = nx.from_numpy(ys, yt, C1, C2, p1, p2, p)

    with pytest.raises(ValueError):
        loss_fun = 'weird_loss_fun'
        X, C, log = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples, [ys, yt], [C1, C2], None, p, [.5, .5], loss_fun, 0.1,
            max_iter=10, tol=1e-3, verbose=True, warmstartT=True, random_state=42,
            solver='PPA', numItermax=10, log=True, symmetric=True,
        )
    with pytest.raises(ValueError):
        stop_criterion = 'unknown stop criterion'
        X, C, log = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples, [ys, yt], [C1, C2], None, p, [.5, .5], 'square_loss',
            0.1, max_iter=10, tol=1e-3, stop_criterion=stop_criterion,
            verbose=True, warmstartT=True, random_state=42,
            solver='PPA', numItermax=10, log=True, symmetric=True,
        )

    for stop_criterion in ['barycenter', 'loss']:
        X, C, log = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples, [ys, yt], [C1, C2], None, p, [.5, .5], 'square_loss',
            epsilon=0.1, max_iter=10, tol=1e-3, stop_criterion=stop_criterion,
            verbose=True, warmstartT=True, random_state=42, solver='PPA',
            numItermax=10, log=True, symmetric=True
        )
        Xb, Cb = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples, [ysb, ytb], [C1b, C2b], [p1b, p2b], None, [.5, .5],
            'square_loss', epsilon=0.1, max_iter=10, tol=1e-3,
            stop_criterion=stop_criterion, verbose=False, warmstartT=True,
            random_state=42, solver='PPA', numItermax=10, log=False, symmetric=True)
        Xb, Cb = nx.to_numpy(Xb, Cb)

        np.testing.assert_allclose(C, Cb, atol=1e-06)
        np.testing.assert_allclose(Cb.shape, (n_samples, n_samples))
        np.testing.assert_allclose(X, Xb, atol=1e-06)
        np.testing.assert_allclose(Xb.shape, (n_samples, ys.shape[1]))

    # test with 'kl_loss' and log=True
    # providing init_C, init_Y
    generator = ot.utils.check_random_state(42)
    xalea = generator.randn(n_samples, 2)
    init_C = ot.utils.dist(xalea, xalea)
    init_C /= init_C.max()
    init_Cb = nx.from_numpy(init_C)

    init_Y = np.zeros((n_samples, ys.shape[1]), dtype=ys.dtype)
    init_Yb = nx.from_numpy(init_Y)

    X, C, log = ot.gromov.entropic_fused_gromov_barycenters(
        n_samples, [ys, yt], [C1, C2], [p1, p2], p, None, 'kl_loss', 0.1, True,
        max_iter=10, tol=1e-3, verbose=False, warmstartT=False, random_state=42,
        solver='PPA', numItermax=1, init_C=init_C, init_Y=init_Y, log=True
    )
    Xb, Cb, logb = ot.gromov.entropic_fused_gromov_barycenters(
        n_samples, [ysb, ytb], [C1b, C2b], [p1b, p2b], pb, [.5, .5], 'kl_loss',
        0.1, True, max_iter=10, tol=1e-3, verbose=False, warmstartT=False,
        random_state=42, solver='PPA', numItermax=1, init_C=init_Cb,
        init_Y=init_Yb, log=True)
    Xb, Cb = nx.to_numpy(Xb, Cb)

    np.testing.assert_allclose(C, Cb, atol=1e-06)
    np.testing.assert_allclose(Cb.shape, (n_samples, n_samples))
    np.testing.assert_allclose(X, Xb, atol=1e-06)
    np.testing.assert_allclose(Xb.shape, (n_samples, ys.shape[1]))
    np.testing.assert_array_almost_equal(log['err_feature'], nx.to_numpy(*logb['err_feature']))
    np.testing.assert_array_almost_equal(log['err_structure'], nx.to_numpy(*logb['err_structure']))

    # add tests with fixed_structures or fixed_features
    init_C = ot.utils.dist(xalea, xalea)
    init_C /= init_C.max()
    init_Cb = nx.from_numpy(init_C)

    init_Y = np.zeros((n_samples, ys.shape[1]), dtype=ys.dtype)
    init_Yb = nx.from_numpy(init_Y)

    fixed_structure, fixed_features = True, False
    with pytest.raises(ot.utils.UndefinedParameter):  # to raise an error when `fixed_structure=True`and `init_C=None`
        Xb, Cb = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples, [ysb, ytb], [C1b, C2b], ps=[p1b, p2b], lambdas=None,
            fixed_structure=fixed_structure, init_C=None,
            fixed_features=fixed_features, p=None, max_iter=10, tol=1e-3
        )

    Xb, Cb = ot.gromov.entropic_fused_gromov_barycenters(
        n_samples, [ysb, ytb], [C1b, C2b], ps=[p1b, p2b], lambdas=None,
        fixed_structure=fixed_structure, init_C=init_Cb,
        fixed_features=fixed_features, max_iter=10, tol=1e-3
    )
    Xb, Cb = nx.to_numpy(Xb), nx.to_numpy(Cb)
    np.testing.assert_allclose(Cb, init_Cb)
    np.testing.assert_allclose(Xb.shape, (n_samples, ys.shape[1]))

    fixed_structure, fixed_features = False, True
    with pytest.raises(ot.utils.UndefinedParameter):  # to raise an error when `fixed_features=True`and `init_X=None`
        Xb, Cb, logb = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples, [ysb, ytb], [C1b, C2b], [p1b, p2b], lambdas=[.5, .5],
            fixed_structure=fixed_structure, fixed_features=fixed_features,
            init_Y=None, p=pb, max_iter=10, tol=1e-3,
            warmstartT=True, log=True, random_state=98765, verbose=True
        )
    Xb, Cb, logb = ot.gromov.entropic_fused_gromov_barycenters(
        n_samples, [ysb, ytb], [C1b, C2b], [p1b, p2b], lambdas=[.5, .5],
        fixed_structure=fixed_structure, fixed_features=fixed_features,
        init_Y=init_Yb, p=pb, max_iter=10, tol=1e-3,
        warmstartT=True, log=True, random_state=98765, verbose=True
    )

    X, C = nx.to_numpy(Xb), nx.to_numpy(Cb)
    np.testing.assert_allclose(C.shape, (n_samples, n_samples))
    np.testing.assert_allclose(Xb, init_Yb)


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
    with pytest.raises(ValueError):
        stop_criterion = 'unknown stop criterion'
        Cb = ot.gromov.gromov_barycenters(
            n_samples, [C1, C2], None, p, [.5, .5], 'square_loss', max_iter=10,
            tol=1e-3, stop_criterion=stop_criterion, verbose=False,
            random_state=42
        )

    for stop_criterion in ['barycenter', 'loss']:
        Cb = ot.gromov.gromov_barycenters(
            n_samples, [C1, C2], None, p, [.5, .5], 'square_loss', max_iter=10,
            tol=1e-3, stop_criterion=stop_criterion, verbose=False,
            random_state=42
        )
        Cbb = nx.to_numpy(ot.gromov.gromov_barycenters(
            n_samples, [C1b, C2b], [p1b, p2b], None, [.5, .5], 'square_loss',
            max_iter=10, tol=1e-3, stop_criterion=stop_criterion,
            verbose=False, random_state=42
        ))
        np.testing.assert_allclose(Cb, Cbb, atol=1e-06)
        np.testing.assert_allclose(Cbb.shape, (n_samples, n_samples))

        # test of gromov_barycenters with `log` on
        Cb_, err_ = ot.gromov.gromov_barycenters(
            n_samples, [C1, C2], [p1, p2], p, None, 'square_loss', max_iter=10,
            tol=1e-3, stop_criterion=stop_criterion, verbose=False,
            warmstartT=True, random_state=42, log=True
        )
        Cbb_, errb_ = ot.gromov.gromov_barycenters(
            n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5], 'square_loss',
            max_iter=10, tol=1e-3, stop_criterion=stop_criterion,
            verbose=False, warmstartT=True, random_state=42, log=True
        )
        Cbb_ = nx.to_numpy(Cbb_)
        np.testing.assert_allclose(Cb_, Cbb_, atol=1e-06)
        np.testing.assert_array_almost_equal(err_['err'], nx.to_numpy(*errb_['err']))
        np.testing.assert_allclose(Cbb_.shape, (n_samples, n_samples))

    Cb2 = ot.gromov.gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'kl_loss', max_iter=10, tol=1e-3, warmstartT=True, random_state=42
    )
    Cb2b = nx.to_numpy(ot.gromov.gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'kl_loss', max_iter=10, tol=1e-3, warmstartT=True, random_state=42
    ))
    np.testing.assert_allclose(Cb2, Cb2b, atol=1e-06)
    np.testing.assert_allclose(Cb2b.shape, (n_samples, n_samples))

    # test of gromov_barycenters with `log` on
    # providing init_C
    generator = ot.utils.check_random_state(42)
    xalea = generator.randn(n_samples, 2)
    init_C = ot.utils.dist(xalea, xalea)
    init_C /= init_C.max()
    init_Cb = nx.from_numpy(init_C)

    Cb2_, err2_ = ot.gromov.gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5], 'kl_loss', max_iter=10,
        tol=1e-3, verbose=False, random_state=42, log=True, init_C=init_C
    )
    Cb2b_, err2b_ = ot.gromov.gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5], 'kl_loss',
        max_iter=10, tol=1e-3, verbose=True, random_state=42,
        init_C=init_Cb, log=True
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

    with pytest.raises(ValueError):
        loss_fun = 'weird_loss_fun'
        Cb = ot.gromov.entropic_gromov_barycenters(
            n_samples, [C1, C2], None, p, [.5, .5], loss_fun, 1e-3,
            max_iter=10, tol=1e-3, verbose=True, warmstartT=True, random_state=42
        )
    with pytest.raises(ValueError):
        stop_criterion = 'unknown stop criterion'
        Cb = ot.gromov.entropic_gromov_barycenters(
            n_samples, [C1, C2], None, p, [.5, .5], 'square_loss', 1e-3,
            max_iter=10, tol=1e-3, stop_criterion=stop_criterion,
            verbose=True, warmstartT=True, random_state=42
        )

    Cb = ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1, C2], None, p, [.5, .5], 'square_loss', 1e-3,
        max_iter=10, tol=1e-3, verbose=True, warmstartT=True, random_state=42
    )
    Cbb = nx.to_numpy(ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], None, [.5, .5], 'square_loss', 1e-3,
        max_iter=10, tol=1e-3, verbose=True, warmstartT=True, random_state=42
    ))
    np.testing.assert_allclose(Cb, Cbb, atol=1e-06)
    np.testing.assert_allclose(Cbb.shape, (n_samples, n_samples))

    # test of entropic_gromov_barycenters with `log` on
    for stop_criterion in ['barycenter', 'loss']:
        Cb_, err_ = ot.gromov.entropic_gromov_barycenters(
            n_samples, [C1, C2], [p1, p2], p, None, 'square_loss', 1e-3,
            max_iter=10, tol=1e-3, stop_criterion=stop_criterion, verbose=True,
            random_state=42, log=True
        )
        Cbb_, errb_ = ot.gromov.entropic_gromov_barycenters(
            n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5], 'square_loss',
            1e-3, max_iter=10, tol=1e-3, stop_criterion=stop_criterion,
            verbose=True, random_state=42, log=True
        )
        Cbb_ = nx.to_numpy(Cbb_)
        np.testing.assert_allclose(Cb_, Cbb_, atol=1e-06)
        np.testing.assert_array_almost_equal(err_['err'], nx.to_numpy(*errb_['err']))
        np.testing.assert_allclose(Cbb_.shape, (n_samples, n_samples))

    Cb2 = ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5],
        'kl_loss', 1e-3, max_iter=10, tol=1e-3, random_state=42
    )
    Cb2b = nx.to_numpy(ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'kl_loss', 1e-3, max_iter=10, tol=1e-3, random_state=42
    ))
    np.testing.assert_allclose(Cb2, Cb2b, atol=1e-06)
    np.testing.assert_allclose(Cb2b.shape, (n_samples, n_samples))

    # test of entropic_gromov_barycenters with `log` on
    # providing init_C
    generator = ot.utils.check_random_state(42)
    xalea = generator.randn(n_samples, 2)
    init_C = ot.utils.dist(xalea, xalea)
    init_C /= init_C.max()
    init_Cb = nx.from_numpy(init_C)

    Cb2_, err2_ = ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1, C2], [p1, p2], p, [.5, .5], 'kl_loss', 1e-3,
        max_iter=10, tol=1e-3, warmstartT=True, verbose=True, random_state=42,
        init_C=init_C, log=True
    )
    Cb2b_, err2b_ = ot.gromov.entropic_gromov_barycenters(
        n_samples, [C1b, C2b], [p1b, p2b], pb, [.5, .5],
        'kl_loss', 1e-3, max_iter=10, tol=1e-3, warmstartT=True, verbose=True,
        random_state=42, init_Cb=init_Cb, log=True
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

    rng = np.random.RandomState(42)
    ys = rng.randn(xs.shape[0], 2)
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

    G, log = ot.gromov.fused_gromov_wasserstein(M, C1, C2, None, q, 'square_loss', alpha=0.5, armijo=True, symmetric=None, G0=G0, log=True)
    Gb, logb = ot.gromov.fused_gromov_wasserstein(Mb, C1b, C2b, pb, None, 'square_loss', alpha=0.5, armijo=True, symmetric=True, G0=G0b, log=True)
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

    fgw, log = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, p, None, 'square_loss', armijo=True, symmetric=True, G0=None, alpha=0.5, log=True)
    fgwb, logb = ot.gromov.fused_gromov_wasserstein2(Mb, C1b, C2b, None, qb, 'square_loss', armijo=True, symmetric=None, G0=G0b, alpha=0.5, log=True)
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


def test_asymmetric_fgw(nx):
    n_samples = 20  # nb samples
    rng = np.random.RandomState(0)
    C1 = rng.uniform(low=0., high=10, size=(n_samples, n_samples))
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    C2 = C1[idx, :][:, idx]

    # add features
    F1 = rng.uniform(low=0., high=10, size=(n_samples, 1))
    F2 = F1[idx, :]
    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    M = ot.dist(F1, F2)
    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)

    G, log = ot.gromov.fused_gromov_wasserstein(
        M, C1, C2, p, q, 'square_loss', alpha=0.5, G0=G0, log=True,
        symmetric=False, verbose=True)
    Gb, logb = ot.gromov.fused_gromov_wasserstein(
        Mb, C1b, C2b, pb, qb, 'square_loss', alpha=0.5, log=True,
        symmetric=None, G0=G0b, verbose=True)
    Gb = nx.to_numpy(Gb)
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    np.testing.assert_allclose(log['fgw_dist'], 0., atol=1e-04)
    np.testing.assert_allclose(logb['fgw_dist'], 0., atol=1e-04)

    fgw, log = ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p, q, 'square_loss', alpha=0.5, G0=G0, log=True,
        symmetric=None, verbose=True)
    fgwb, logb = ot.gromov.fused_gromov_wasserstein2(
        Mb, C1b, C2b, pb, qb, 'square_loss', alpha=0.5, log=True,
        symmetric=False, G0=G0b, verbose=True)

    G = log['T']
    Gb = nx.to_numpy(logb['T'])
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    np.testing.assert_allclose(log['fgw_dist'], 0., atol=1e-04)
    np.testing.assert_allclose(logb['fgw_dist'], 0., atol=1e-04)

    # Tests with kl-loss:
    for armijo in [False, True]:
        G, log = ot.gromov.fused_gromov_wasserstein(
            M, C1, C2, p, q, 'kl_loss', alpha=0.5, armijo=armijo, G0=G0,
            log=True, symmetric=False, verbose=True)
        Gb, logb = ot.gromov.fused_gromov_wasserstein(
            Mb, C1b, C2b, pb, qb, 'kl_loss', alpha=0.5, armijo=armijo,
            log=True, symmetric=None, G0=G0b, verbose=True)
        Gb = nx.to_numpy(Gb)
        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(
            p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose(
            q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

        np.testing.assert_allclose(log['fgw_dist'], 0., atol=1e-04)
        np.testing.assert_allclose(logb['fgw_dist'], 0., atol=1e-04)

        fgw, log = ot.gromov.fused_gromov_wasserstein2(
            M, C1, C2, p, q, 'kl_loss', alpha=0.5, G0=G0, log=True,
            symmetric=None, verbose=True)
        fgwb, logb = ot.gromov.fused_gromov_wasserstein2(
            Mb, C1b, C2b, pb, qb, 'kl_loss', alpha=0.5, log=True,
            symmetric=False, G0=G0b, verbose=True)

        G = log['T']
        Gb = nx.to_numpy(logb['T'])
        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(
            p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose(
            q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

        np.testing.assert_allclose(log['fgw_dist'], 0., atol=1e-04)
        np.testing.assert_allclose(logb['fgw_dist'], 0., atol=1e-04)


def test_fgw_integer_warnings(nx):
    n_samples = 20  # nb samples
    rng = np.random.RandomState(0)
    C1 = rng.uniform(low=0., high=10, size=(n_samples, n_samples))
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    C2 = C1[idx, :][:, idx]

    # add features
    F1 = rng.uniform(low=0., high=10, size=(n_samples, 1))
    F2 = F1[idx, :]
    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    M = ot.dist(F1, F2).astype(np.int32)
    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)

    G, log = ot.gromov.fused_gromov_wasserstein(
        M, C1, C2, p, q, 'square_loss', alpha=0.5, G0=G0, log=True,
        symmetric=False, verbose=True)
    Gb, logb = ot.gromov.fused_gromov_wasserstein(
        Mb, C1b, C2b, pb, qb, 'square_loss', alpha=0.5, log=True,
        symmetric=None, G0=G0b, verbose=True)
    Gb = nx.to_numpy(Gb)
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(G, 0., atol=1e-06)


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

            # full gradients with alpha
            p1 = torch.tensor(p, requires_grad=True, device=device)
            q1 = torch.tensor(q, requires_grad=True, device=device)
            C11 = torch.tensor(C1, requires_grad=True, device=device)
            C12 = torch.tensor(C2, requires_grad=True, device=device)
            M1 = torch.tensor(M, requires_grad=True, device=device)
            alpha = torch.tensor(0.5, requires_grad=True, device=device)

            val = ot.fused_gromov_wasserstein2(M1, C11, C12, p1, q1, alpha=alpha)

            val.backward()

            assert val.device == p1.device
            assert q1.shape == q1.grad.shape
            assert p1.shape == p1.grad.shape
            assert C11.shape == C11.grad.shape
            assert C12.shape == C12.grad.shape
            assert alpha.shape == alpha.grad.shape


def test_fgw_helper_backend(nx):
    n_samples = 20  # nb samples

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=0)
    ys = rng.randn(xs.shape[0], 2)
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=1)
    yt = rng.randn(xt.shape[0], 2)

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
    alpha = 0.5
    Gb, logb = ot.gromov.fused_gromov_wasserstein(Mb, C1b, C2b, pb, qb, 'square_loss', alpha=0.5, armijo=False, symmetric=True, G0=G0b, log=True)

    # calls with nx=None
    constCb, hC1b, hC2b = ot.gromov.init_matrix(C1b, C2b, pb, qb, loss_fun='square_loss')

    def f(G):
        return ot.gromov.gwloss(constCb, hC1b, hC2b, G, None)

    def df(G):
        return ot.gromov.gwggrad(constCb, hC1b, hC2b, G, None)

    def line_search(cost, G, deltaG, Mi, cost_G):
        return ot.gromov.solve_gromov_linesearch(G, deltaG, cost_G, C1b, C2b, M=(1 - alpha) * Mb, reg=alpha, nx=None)
    # feed the precomputed local optimum Gb to cg
    res, log = ot.optim.cg(pb, qb, (1 - alpha) * Mb, alpha, f, df, Gb, line_search, log=True, numItermax=1e4, stopThr=1e-9, stopThr2=1e-9)

    def line_search(cost, G, deltaG, Mi, cost_G):
        return ot.optim.line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=None)
    # feed the precomputed local optimum Gb to cg
    res_armijo, log_armijo = ot.optim.cg(pb, qb, (1 - alpha) * Mb, alpha, f, df, Gb, line_search, log=True, numItermax=1e4, stopThr=1e-9, stopThr2=1e-9)
    # check constraints
    np.testing.assert_allclose(res, Gb, atol=1e-06)
    np.testing.assert_allclose(res_armijo, Gb, atol=1e-06)


def test_fgw_barycenter(nx):
    ns = 10
    nt = 20

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    rng = np.random.RandomState(42)
    ys = rng.randn(Xs.shape[0], 2)
    yt = rng.randn(Xt.shape[0], 2)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    C1 /= C1.max()
    C2 /= C2.max()

    p1, p2 = ot.unif(ns), ot.unif(nt)
    n_samples = 3
    p = ot.unif(n_samples)

    ysb, ytb, C1b, C2b, p1b, p2b, pb = nx.from_numpy(ys, yt, C1, C2, p1, p2, p)
    lambdas = [.5, .5]
    Csb = [C1b, C2b]
    Ysb = [ysb, ytb]
    Xb, Cb, logb = ot.gromov.fgw_barycenters(
        n_samples, Ysb, Csb, None, lambdas, 0.5, fixed_structure=False,
        fixed_features=False, p=pb, loss_fun='square_loss', max_iter=10, tol=1e-3,
        random_state=12345, log=True
    )
    # test correspondance with utils function
    recovered_Cb = ot.gromov.update_square_loss(pb, lambdas, logb['Ts_iter'][-1], Csb)
    recovered_Xb = ot.gromov.update_feature_matrix(lambdas, [y.T for y in Ysb], logb['Ts_iter'][-1], pb).T

    np.testing.assert_allclose(Cb, recovered_Cb)
    np.testing.assert_allclose(Xb, recovered_Xb)

    xalea = rng.randn(n_samples, 2)
    init_C = ot.dist(xalea, xalea)
    init_C /= init_C.max()
    init_Cb = nx.from_numpy(init_C)

    with pytest.raises(ot.utils.UndefinedParameter):  # to raise an error when `fixed_structure=True`and `init_C=None`
        Xb, Cb = ot.gromov.fgw_barycenters(
            n_samples, Ysb, Csb, ps=[p1b, p2b], lambdas=None,
            alpha=0.5, fixed_structure=True, init_C=None, fixed_features=False,
            p=None, loss_fun='square_loss', max_iter=10, tol=1e-3
        )

    Xb, Cb = ot.gromov.fgw_barycenters(
        n_samples, [ysb, ytb], [C1b, C2b], ps=[p1b, p2b], lambdas=None,
        alpha=0.5, fixed_structure=True, init_C=init_Cb, fixed_features=False,
        p=None, loss_fun='square_loss', max_iter=10, tol=1e-3
    )
    Xb, Cb = nx.to_numpy(Xb), nx.to_numpy(Cb)
    np.testing.assert_allclose(Cb.shape, (n_samples, n_samples))
    np.testing.assert_allclose(Xb.shape, (n_samples, ys.shape[1]))

    init_X = rng.randn(n_samples, ys.shape[1])
    init_Xb = nx.from_numpy(init_X)

    with pytest.raises(ot.utils.UndefinedParameter):  # to raise an error when `fixed_features=True`and `init_X=None`
        Xb, Cb, logb = ot.gromov.fgw_barycenters(
            n_samples, [ysb, ytb], [C1b, C2b], [p1b, p2b], [.5, .5], 0.5,
            fixed_structure=False, fixed_features=True, init_X=None,
            p=pb, loss_fun='square_loss', max_iter=10, tol=1e-3,
            warmstartT=True, log=True, random_state=98765, verbose=True
        )
    Xb, Cb, logb = ot.gromov.fgw_barycenters(
        n_samples, [ysb, ytb], [C1b, C2b], [p1b, p2b], [.5, .5], 0.5,
        fixed_structure=False, fixed_features=True, init_X=init_Xb,
        p=pb, loss_fun='square_loss', max_iter=10, tol=1e-3,
        warmstartT=True, log=True, random_state=98765, verbose=True
    )

    X, C = nx.to_numpy(Xb), nx.to_numpy(Cb)
    np.testing.assert_allclose(C.shape, (n_samples, n_samples))
    np.testing.assert_allclose(X.shape, (n_samples, ys.shape[1]))

    # add test with 'kl_loss'
    with pytest.raises(ValueError):
        stop_criterion = 'unknown stop criterion'
        X, C, log = ot.gromov.fgw_barycenters(
            n_samples, [ys, yt], [C1, C2], [p1, p2], [.5, .5], 0.5,
            fixed_structure=False, fixed_features=False, p=p, loss_fun='kl_loss',
            max_iter=100, tol=1e-3, stop_criterion=stop_criterion, init_C=C,
            init_X=X, warmstartT=True, random_state=12345, log=True
        )

    for stop_criterion in ['barycenter', 'loss']:
        X, C, log = ot.gromov.fgw_barycenters(
            n_samples, [ys, yt], [C1, C2], [p1, p2], [.5, .5], 0.5,
            fixed_structure=False, fixed_features=False, p=p, loss_fun='kl_loss',
            max_iter=100, tol=1e-3, stop_criterion=stop_criterion, init_C=C,
            init_X=X, warmstartT=True, random_state=12345, log=True, verbose=True
        )
        np.testing.assert_allclose(C.shape, (n_samples, n_samples))
        np.testing.assert_allclose(X.shape, (n_samples, ys.shape[1]))

    # test correspondance with utils function
    recovered_C = ot.gromov.update_kl_loss(p, lambdas, log['T'], [C1, C2])
    np.testing.assert_allclose(C, recovered_C)


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
    rng = np.random.RandomState(0)
    Cdict_init = rng.normal(loc=np.mean(dataset_means), scale=np.std(dataset_means), size=(n_atoms, shape, shape))

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
    Cdict_bis, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Cs, D=n_atoms, nt=shape, ps=None, q=None, Cdict_init=None,
        epochs=epochs, batch_size=n_samples, learning_rate=1., reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose,
        random_state=0
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
    Cdictb_bis, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Csb, D=n_atoms, nt=shape, ps=psb, q=qb, Cdict_init=None,
        epochs=epochs, batch_size=n_samples, learning_rate=1., reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer,
        verbose=verbose, random_state=0
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

    Cdict_bis2, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Cs, D=n_atoms, nt=shape, ps=ps, q=q, Cdict_init=Cdict,
        epochs=epochs, batch_size=n_samples, learning_rate=10., reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=use_log, use_adam_optimizer=use_adam_optimizer,
        verbose=verbose, random_state=0,
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
    Cdictb_bis2, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Csb, D=n_atoms, nt=shape, ps=psb, q=qb, Cdict_init=Cdictb,
        epochs=epochs, batch_size=n_samples, learning_rate=10., reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=use_log, use_adam_optimizer=use_adam_optimizer,
        verbose=verbose, random_state=0,
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
    rng = np.random.RandomState(0)
    Cdict_init = rng.normal(loc=np.mean(dataset_structure_means), scale=np.std(dataset_structure_means), size=(n_atoms, shape, shape))
    if projection == 'nonnegative_symmetric':
        Cdict_init = 0.5 * (Cdict_init + Cdict_init.transpose((0, 2, 1)))
        Cdict_init[Cdict_init < 0.] = 0.
    dataset_feature_means = np.stack([Y.mean(axis=0) for Y in Ys])
    Ydict_init = rng.normal(loc=dataset_feature_means.mean(axis=0), scale=dataset_feature_means.std(axis=0), size=(n_atoms, shape, 2))

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
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose,
        random_state=0
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
    Cdict_bis, Ydict_bis, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Cs, Ys, D=n_atoms, nt=shape, ps=None, q=None, Cdict_init=None, Ydict_init=None,
        epochs=epochs, batch_size=n_samples, learning_rate_C=1., learning_rate_Y=1., alpha=alpha, reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose,
        random_state=0
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
    Cdictb_bis, Ydictb_bis, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Csb, Ysb, D=n_atoms, nt=shape, ps=None, q=None, Cdict_init=None, Ydict_init=None,
        epochs=epochs, batch_size=n_samples, learning_rate_C=1., learning_rate_Y=1., alpha=alpha, reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=False, use_adam_optimizer=use_adam_optimizer, verbose=verbose,
        random_state=0,
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
    Cdict_bis2, Ydict_bis2, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Cs, Ys, D=n_atoms, nt=shape, ps=ps, q=q, Cdict_init=Cdict, Ydict_init=Ydict,
        epochs=epochs, batch_size=n_samples, learning_rate_C=10., learning_rate_Y=10., alpha=alpha, reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=use_log, use_adam_optimizer=use_adam_optimizer,
        verbose=verbose, random_state=0,
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
    Cdictb_bis2, Ydictb_bis2, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Csb, Ysb, D=n_atoms, nt=shape, ps=None, q=None, Cdict_init=Cdictb, Ydict_init=Ydictb,
        epochs=epochs, batch_size=n_samples, learning_rate_C=10., learning_rate_Y=10., alpha=alpha, reg=0.,
        tol_outer=tol, tol_inner=tol, max_iter_outer=10, max_iter_inner=50,
        projection=projection, use_log=use_log, use_adam_optimizer=use_adam_optimizer, verbose=verbose,
        random_state=0,
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


def test_semirelaxed_gromov(nx):
    rng = np.random.RandomState(0)
    # unbalanced proportions
    list_n = [30, 15]
    nt = 2
    ns = np.sum(list_n)
    # create directed sbm with C2 as connectivity matrix
    C1 = np.zeros((ns, ns), dtype=np.float64)
    C2 = np.array([[0.8, 0.05],
                   [0.05, 1.]], dtype=np.float64)
    for i in range(nt):
        for j in range(nt):
            ni, nj = list_n[i], list_n[j]
            xij = rng.binomial(size=(ni, nj), n=1, p=C2[i, j])
            C1[i * ni: (i + 1) * ni, j * nj: (j + 1) * nj] = xij
    p = ot.unif(ns, type_as=C1)
    q0 = ot.unif(C2.shape[0], type_as=C1)
    G0 = p[:, None] * q0[None, :]
    # asymmetric
    C1b, C2b, pb, q0b, G0b = nx.from_numpy(C1, C2, p, q0, G0)

    for loss_fun in ['square_loss', 'kl_loss']:
        G, log = ot.gromov.semirelaxed_gromov_wasserstein(
            C1, C2, p, loss_fun='square_loss', symmetric=None, log=True, G0=G0)
        Gb, logb = ot.gromov.semirelaxed_gromov_wasserstein(
            C1b, C2b, None, loss_fun='square_loss', symmetric=False, log=True,
            G0=None, alpha_min=0., alpha_max=1.)

        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, nx.sum(Gb, axis=1), atol=1e-04)
        np.testing.assert_allclose(list_n / ns, np.sum(G, axis=0), atol=1e-01)
        np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-01)

        srgw, log2 = ot.gromov.semirelaxed_gromov_wasserstein2(
            C1, C2, None, loss_fun='square_loss', symmetric=False, log=True, G0=G0)
        srgwb, logb2 = ot.gromov.semirelaxed_gromov_wasserstein2(
            C1b, C2b, pb, loss_fun='square_loss', symmetric=None, log=True, G0=None)

        G = log2['T']
        Gb = nx.to_numpy(logb2['T'])
        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose(list_n / ns, Gb.sum(0), atol=1e-04)  # cf convergence gromov

        np.testing.assert_allclose(log2['srgw_dist'], logb['srgw_dist'], atol=1e-07)
        np.testing.assert_allclose(logb2['srgw_dist'], log['srgw_dist'], atol=1e-07)

    # symmetric
    C1 = 0.5 * (C1 + C1.T)
    C1b, C2b, pb, q0b, G0b = nx.from_numpy(C1, C2, p, q0, G0)

    G, log = ot.gromov.semirelaxed_gromov_wasserstein(
        C1, C2, p, loss_fun='square_loss', symmetric=None, log=True, G0=None)
    Gb = ot.gromov.semirelaxed_gromov_wasserstein(
        C1b, C2b, pb, loss_fun='square_loss', symmetric=True, log=False, G0=G0b)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, nx.sum(Gb, axis=1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-02)  # cf convergence gromov

    srgw, log2 = ot.gromov.semirelaxed_gromov_wasserstein2(
        C1, C2, p, loss_fun='square_loss', symmetric=True, log=True, G0=G0)
    srgwb, logb2 = ot.gromov.semirelaxed_gromov_wasserstein2(
        C1b, C2b, pb, loss_fun='square_loss', symmetric=None, log=True, G0=None)

    srgw_ = ot.gromov.semirelaxed_gromov_wasserstein2(C1, C2, p, loss_fun='square_loss', symmetric=True, log=False, G0=G0)

    G = log2['T']
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, nx.sum(Gb, 1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(list_n / ns, np.sum(G, axis=0), atol=1e-01)
    np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-01)

    np.testing.assert_allclose(log2['srgw_dist'], log['srgw_dist'], atol=1e-07)
    np.testing.assert_allclose(logb2['srgw_dist'], log['srgw_dist'], atol=1e-07)
    np.testing.assert_allclose(srgw, srgw_, atol=1e-07)


def test_semirelaxed_gromov2_gradients():
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)

    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=5)

    p = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    if torch:

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for loss_fun in ['square_loss', 'kl_loss']:
                # semirelaxed solvers do not support gradients over masses yet.
                p1 = torch.tensor(p, requires_grad=False, device=device)
                C11 = torch.tensor(C1, requires_grad=True, device=device)
                C12 = torch.tensor(C2, requires_grad=True, device=device)

                val = ot.gromov.semirelaxed_gromov_wasserstein2(C11, C12, p1, loss_fun=loss_fun)

                val.backward()

                assert val.device == p1.device
                assert p1.grad is None
                assert C11.shape == C11.grad.shape
                assert C12.shape == C12.grad.shape


def test_srgw_helper_backend(nx):
    n_samples = 20  # nb samples

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=0)
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=1)

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    for loss_fun in ['square_loss', 'kl_loss']:
        C1b, C2b, pb, qb = nx.from_numpy(C1, C2, p, q)
        Gb, logb = ot.gromov.semirelaxed_gromov_wasserstein(C1b, C2b, pb, loss_fun, armijo=False, symmetric=True, G0=None, log=True)

        # calls with nx=None
        constCb, hC1b, hC2b, fC2tb = ot.gromov.init_matrix_semirelaxed(C1b, C2b, pb, loss_fun)
        ones_pb = nx.ones(pb.shape[0], type_as=pb)

        def f(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_pb, nx.dot(qG, fC2tb))
            return ot.gromov.gwloss(constCb + marginal_product, hC1b, hC2b, G, nx=None)

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_pb, nx.dot(qG, fC2tb))
            return ot.gromov.gwggrad(constCb + marginal_product, hC1b, hC2b, G, nx=None)

        def line_search(cost, G, deltaG, Mi, cost_G):
            return ot.gromov.solve_semirelaxed_gromov_linesearch(
                G, deltaG, cost_G, hC1b, hC2b, ones_pb, 0., 1., fC2t=fC2tb, nx=None)
        # feed the precomputed local optimum Gb to semirelaxed_cg
        res, log = ot.optim.semirelaxed_cg(pb, qb, 0., 1., f, df, Gb, line_search, log=True, numItermax=1e4, stopThr=1e-9, stopThr2=1e-9)
        # check constraints
        np.testing.assert_allclose(res, Gb, atol=1e-06)


@pytest.mark.parametrize('loss_fun', [
    'square_loss', 'kl_loss',
    pytest.param('unknown_loss', marks=pytest.mark.xfail(raises=ValueError)),
])
def test_gw_semirelaxed_helper_validation(loss_fun):
    n_samples = 20  # nb samples
    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=0)
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=1)
    p = ot.unif(n_samples)
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    ot.gromov.init_matrix_semirelaxed(C1, C2, p, loss_fun=loss_fun)


def test_semirelaxed_fgw(nx):
    rng = np.random.RandomState(0)
    list_n = [16, 8]
    nt = 2
    ns = 24
    # create directed sbm with C2 as connectivity matrix
    C1 = np.zeros((ns, ns))
    C2 = np.array([[0.7, 0.05],
                   [0.05, 0.9]])
    for i in range(nt):
        for j in range(nt):
            ni, nj = list_n[i], list_n[j]
            xij = rng.binomial(size=(ni, nj), n=1, p=C2[i, j])
            C1[i * ni: (i + 1) * ni, j * nj: (j + 1) * nj] = xij
    F1 = np.zeros((ns, 1))
    F1[:16] = rng.normal(loc=0., scale=0.01, size=(16, 1))
    F1[16:] = rng.normal(loc=1., scale=0.01, size=(8, 1))
    F2 = np.zeros((2, 1))
    F2[1, :] = 1.
    M = (F1 ** 2).dot(np.ones((1, nt))) + np.ones((ns, 1)).dot((F2 ** 2).T) - 2 * F1.dot(F2.T)

    p = ot.unif(ns)
    q0 = ot.unif(C2.shape[0])
    G0 = p[:, None] * q0[None, :]

    # asymmetric
    Mb, C1b, C2b, pb, q0b, G0b = nx.from_numpy(M, C1, C2, p, q0, G0)
    G, log = ot.gromov.semirelaxed_fused_gromov_wasserstein(M, C1, C2, None, loss_fun='square_loss', alpha=0.5, symmetric=None, log=True, G0=None)
    Gb, logb = ot.gromov.semirelaxed_fused_gromov_wasserstein(Mb, C1b, C2b, pb, loss_fun='square_loss', alpha=0.5, symmetric=False, log=True, G0=G0b)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, nx.sum(Gb, axis=1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose([2 / 3, 1 / 3], nx.sum(Gb, axis=0), atol=1e-02)  # cf convergence gromov

    srgw, log2 = ot.gromov.semirelaxed_fused_gromov_wasserstein2(M, C1, C2, p, loss_fun='square_loss', alpha=0.5, symmetric=False, log=True, G0=G0)
    srgwb, logb2 = ot.gromov.semirelaxed_fused_gromov_wasserstein2(Mb, C1b, C2b, None, loss_fun='square_loss', alpha=0.5, symmetric=None, log=True, G0=None)

    G = log2['T']
    Gb = nx.to_numpy(logb2['T'])
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose([2 / 3, 1 / 3], Gb.sum(0), atol=1e-04)  # cf convergence gromov

    np.testing.assert_allclose(log2['srfgw_dist'], logb['srfgw_dist'], atol=1e-07)
    np.testing.assert_allclose(logb2['srfgw_dist'], log['srfgw_dist'], atol=1e-07)

    # symmetric
    for loss_fun in ['square_loss', 'kl_loss']:
        C1 = 0.5 * (C1 + C1.T)
        Mb, C1b, C2b, pb, q0b, G0b = nx.from_numpy(M, C1, C2, p, q0, G0)

        G, log = ot.gromov.semirelaxed_fused_gromov_wasserstein(M, C1, C2, p, loss_fun=loss_fun, alpha=0.5, symmetric=None, log=True, G0=None)
        Gb = ot.gromov.semirelaxed_fused_gromov_wasserstein(Mb, C1b, C2b, pb, loss_fun=loss_fun, alpha=0.5, symmetric=True, log=False, G0=G0b)

        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, nx.sum(Gb, axis=1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose([2 / 3, 1 / 3], nx.sum(Gb, axis=0), atol=1e-02)  # cf convergence gromov

        srgw, log2 = ot.gromov.semirelaxed_fused_gromov_wasserstein2(M, C1, C2, p, loss_fun=loss_fun, alpha=0.5, symmetric=True, log=True, G0=G0)
        srgwb, logb2 = ot.gromov.semirelaxed_fused_gromov_wasserstein2(Mb, C1b, C2b, pb, loss_fun=loss_fun, alpha=0.5, symmetric=None, log=True, G0=None)

        srgw_ = ot.gromov.semirelaxed_fused_gromov_wasserstein2(M, C1, C2, p, loss_fun=loss_fun, alpha=0.5, symmetric=True, log=False, G0=G0)

        G = log2['T']
        Gb = nx.to_numpy(logb2['T'])
        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose([2 / 3, 1 / 3], Gb.sum(0), atol=1e-04)  # cf convergence gromov

        np.testing.assert_allclose(log2['srfgw_dist'], log['srfgw_dist'], atol=1e-07)
        np.testing.assert_allclose(logb2['srfgw_dist'], log['srfgw_dist'], atol=1e-07)
        np.testing.assert_allclose(srgw, srgw_, atol=1e-07)


def test_semirelaxed_fgw2_gradients():
    n_samples = 20  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)

    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=5)

    p = ot.unif(n_samples)

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
            # semirelaxed solvers do not support gradients over masses yet.
            for loss_fun in ['square_loss', 'kl_loss']:
                p1 = torch.tensor(p, requires_grad=False, device=device)
                C11 = torch.tensor(C1, requires_grad=True, device=device)
                C12 = torch.tensor(C2, requires_grad=True, device=device)
                M1 = torch.tensor(M, requires_grad=True, device=device)

                val = ot.gromov.semirelaxed_fused_gromov_wasserstein2(M1, C11, C12, p1, loss_fun=loss_fun)

                val.backward()

                assert val.device == p1.device
                assert p1.grad is None
                assert C11.shape == C11.grad.shape
                assert C12.shape == C12.grad.shape
                assert M1.shape == M1.grad.shape

                # full gradients with alpha
                p1 = torch.tensor(p, requires_grad=False, device=device)
                C11 = torch.tensor(C1, requires_grad=True, device=device)
                C12 = torch.tensor(C2, requires_grad=True, device=device)
                M1 = torch.tensor(M, requires_grad=True, device=device)
                alpha = torch.tensor(0.5, requires_grad=True, device=device)

                val = ot.gromov.semirelaxed_fused_gromov_wasserstein2(M1, C11, C12, p1, loss_fun=loss_fun, alpha=alpha)

                val.backward()

                assert val.device == p1.device
                assert p1.grad is None
                assert C11.shape == C11.grad.shape
                assert C12.shape == C12.grad.shape
                assert alpha.shape == alpha.grad.shape


def test_srfgw_helper_backend(nx):
    n_samples = 20  # nb samples

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=0)
    ys = rng.randn(xs.shape[0], 2)
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=1)
    yt = rng.randn(xt.shape[0], 2)

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
    alpha = 0.5
    Gb, logb = ot.gromov.semirelaxed_fused_gromov_wasserstein(Mb, C1b, C2b, pb, 'square_loss', alpha=0.5, armijo=False, symmetric=True, G0=G0b, log=True)

    # calls with nx=None
    constCb, hC1b, hC2b, fC2tb = ot.gromov.init_matrix_semirelaxed(C1b, C2b, pb, loss_fun='square_loss')
    ones_pb = nx.ones(pb.shape[0], type_as=pb)

    def f(G):
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_pb, nx.dot(qG, fC2tb))
        return ot.gromov.gwloss(constCb + marginal_product, hC1b, hC2b, G, nx=None)

    def df(G):
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_pb, nx.dot(qG, fC2tb))
        return ot.gromov.gwggrad(constCb + marginal_product, hC1b, hC2b, G, nx=None)

    def line_search(cost, G, deltaG, Mi, cost_G):
        return ot.gromov.solve_semirelaxed_gromov_linesearch(
            G, deltaG, cost_G, C1b, C2b, ones_pb, M=(1 - alpha) * Mb, reg=alpha, nx=None)
    # feed the precomputed local optimum Gb to semirelaxed_cg
    res, log = ot.optim.semirelaxed_cg(pb, qb, (1 - alpha) * Mb, alpha, f, df, Gb, line_search, log=True, numItermax=1e4, stopThr=1e-9, stopThr2=1e-9)
    # check constraints
    np.testing.assert_allclose(res, Gb, atol=1e-06)


def test_entropic_semirelaxed_gromov(nx):
    # unbalanced proportions
    list_n = [30, 15]
    nt = 2
    ns = np.sum(list_n)
    # create directed sbm with C2 as connectivity matrix
    C1 = np.zeros((ns, ns), dtype=np.float64)
    C2 = np.array([[0.8, 0.05],
                   [0.05, 1.]], dtype=np.float64)
    rng = np.random.RandomState(0)
    for i in range(nt):
        for j in range(nt):
            ni, nj = list_n[i], list_n[j]
            xij = rng.binomial(size=(ni, nj), n=1, p=C2[i, j])
            C1[i * ni: (i + 1) * ni, j * nj: (j + 1) * nj] = xij
    p = ot.unif(ns, type_as=C1)
    q0 = ot.unif(C2.shape[0], type_as=C1)
    G0 = p[:, None] * q0[None, :]
    # asymmetric
    C1b, C2b, pb, q0b, G0b = nx.from_numpy(C1, C2, p, q0, G0)
    epsilon = 0.1
    for loss_fun in ['square_loss', 'kl_loss']:
        G, log = ot.gromov.entropic_semirelaxed_gromov_wasserstein(C1, C2, p, loss_fun=loss_fun, epsilon=epsilon, symmetric=None, log=True, G0=G0)
        Gb, logb = ot.gromov.entropic_semirelaxed_gromov_wasserstein(C1b, C2b, None, loss_fun=loss_fun, epsilon=epsilon, symmetric=False, log=True, G0=None)

        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, nx.sum(Gb, axis=1), atol=1e-04)
        np.testing.assert_allclose(list_n / ns, np.sum(G, axis=0), atol=1e-01)
        np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-01)

        srgw, log2 = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(C1, C2, None, loss_fun=loss_fun, epsilon=epsilon, symmetric=False, log=True, G0=G0)
        srgwb, logb2 = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(C1b, C2b, pb, loss_fun=loss_fun, epsilon=epsilon, symmetric=None, log=True, G0=None)

        G = log2['T']
        Gb = nx.to_numpy(logb2['T'])
        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose(list_n / ns, Gb.sum(0), atol=1e-04)  # cf convergence gromov

        np.testing.assert_allclose(log2['srgw_dist'], logb['srgw_dist'], atol=1e-07)
        np.testing.assert_allclose(logb2['srgw_dist'], log['srgw_dist'], atol=1e-07)

    # symmetric
    C1 = 0.5 * (C1 + C1.T)
    C1b, C2b, pb, q0b, G0b = nx.from_numpy(C1, C2, p, q0, G0)

    G, log = ot.gromov.entropic_semirelaxed_gromov_wasserstein(C1, C2, p, loss_fun='square_loss', epsilon=epsilon, symmetric=None, log=True, G0=None)
    Gb = ot.gromov.entropic_semirelaxed_gromov_wasserstein(C1b, C2b, None, loss_fun='square_loss', epsilon=epsilon, symmetric=True, log=False, G0=G0b)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, nx.sum(Gb, axis=1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-02)  # cf convergence gromov

    srgw, log2 = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(C1, C2, p, loss_fun='square_loss', epsilon=epsilon, symmetric=True, log=True, G0=G0)
    srgwb, logb2 = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(C1b, C2b, pb, loss_fun='square_loss', epsilon=epsilon, symmetric=None, log=True, G0=None)

    srgw_ = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(C1, C2, p, loss_fun='square_loss', epsilon=epsilon, symmetric=True, log=False, G0=G0)

    G = log2['T']
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, nx.sum(Gb, 1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(list_n / ns, np.sum(G, axis=0), atol=1e-01)
    np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-01)

    np.testing.assert_allclose(log2['srgw_dist'], log['srgw_dist'], atol=1e-07)
    np.testing.assert_allclose(logb2['srgw_dist'], log['srgw_dist'], atol=1e-07)
    np.testing.assert_allclose(srgw, srgw_, atol=1e-07)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_semirelaxed_gromov_dtype_device(nx):
    # setup
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    for tp in nx.__type_list__:

        print(nx.dtype_device(tp))
        for loss_fun in ['square_loss', 'kl_loss']:
            C1b, C2b, pb = nx.from_numpy(C1, C2, p, type_as=tp)

            Gb = ot.gromov.entropic_semirelaxed_gromov_wasserstein(
                C1b, C2b, pb, loss_fun, epsilon=0.1, verbose=True
            )
            gw_valb = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(
                C1b, C2b, pb, loss_fun, epsilon=0.1, verbose=True
            )

            nx.assert_same_dtype_device(C1b, Gb)
            nx.assert_same_dtype_device(C1b, gw_valb)


def test_entropic_semirelaxed_fgw(nx):
    rng = np.random.RandomState(0)
    list_n = [16, 8]
    nt = 2
    ns = 24
    # create directed sbm with C2 as connectivity matrix
    C1 = np.zeros((ns, ns))
    C2 = np.array([[0.7, 0.05],
                   [0.05, 0.9]])
    for i in range(nt):
        for j in range(nt):
            ni, nj = list_n[i], list_n[j]
            xij = rng.binomial(size=(ni, nj), n=1, p=C2[i, j])
            C1[i * ni: (i + 1) * ni, j * nj: (j + 1) * nj] = xij
    F1 = np.zeros((ns, 1))
    F1[:16] = rng.normal(loc=0., scale=0.01, size=(16, 1))
    F1[16:] = rng.normal(loc=1., scale=0.01, size=(8, 1))
    F2 = np.zeros((2, 1))
    F2[1, :] = 1.
    M = (F1 ** 2).dot(np.ones((1, nt))) + np.ones((ns, 1)).dot((F2 ** 2).T) - 2 * F1.dot(F2.T)

    p = ot.unif(ns)
    q0 = ot.unif(C2.shape[0])
    G0 = p[:, None] * q0[None, :]

    # asymmetric
    Mb, C1b, C2b, pb, q0b, G0b = nx.from_numpy(M, C1, C2, p, q0, G0)

    G, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(M, C1, C2, None, loss_fun='square_loss', epsilon=0.1, alpha=0.5, symmetric=None, log=True, G0=None)
    Gb, logb = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(Mb, C1b, C2b, pb, loss_fun='square_loss', epsilon=0.1, alpha=0.5, symmetric=False, log=True, G0=G0b)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, nx.sum(Gb, axis=1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose([2 / 3, 1 / 3], nx.sum(Gb, axis=0), atol=1e-02)  # cf convergence gromov

    srgw, log2 = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(M, C1, C2, p, loss_fun='square_loss', epsilon=0.1, alpha=0.5, symmetric=False, log=True, G0=G0)
    srgwb, logb2 = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(Mb, C1b, C2b, None, loss_fun='square_loss', epsilon=0.1, alpha=0.5, symmetric=None, log=True, G0=None)

    G = log2['T']
    Gb = nx.to_numpy(logb2['T'])
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose([2 / 3, 1 / 3], Gb.sum(0), atol=1e-04)  # cf convergence gromov

    np.testing.assert_allclose(log2['srfgw_dist'], logb['srfgw_dist'], atol=1e-07)
    np.testing.assert_allclose(logb2['srfgw_dist'], log['srfgw_dist'], atol=1e-07)

    # symmetric
    C1 = 0.5 * (C1 + C1.T)
    Mb, C1b, C2b, pb, q0b, G0b = nx.from_numpy(M, C1, C2, p, q0, G0)

    for loss_fun in ['square_loss', 'kl_loss']:
        G, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(M, C1, C2, p, loss_fun=loss_fun, epsilon=0.1, alpha=0.5, symmetric=None, log=True, G0=None)
        Gb = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(Mb, C1b, C2b, pb, loss_fun=loss_fun, epsilon=0.1, alpha=0.5, symmetric=True, log=False, G0=G0b)

        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, nx.sum(Gb, axis=1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose([2 / 3, 1 / 3], nx.sum(Gb, axis=0), atol=1e-02)  # cf convergence gromov

        srgw, log2 = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(M, C1, C2, p, loss_fun=loss_fun, epsilon=0.1, alpha=0.5, symmetric=True, log=True, G0=G0)
        srgwb, logb2 = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(Mb, C1b, C2b, pb, loss_fun=loss_fun, epsilon=0.1, alpha=0.5, symmetric=None, log=True, G0=None)

        srgw_ = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(M, C1, C2, p, loss_fun=loss_fun, epsilon=0.1, alpha=0.5, symmetric=True, log=False, G0=G0)

        G = log2['T']
        Gb = nx.to_numpy(logb2['T'])
        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose([2 / 3, 1 / 3], Gb.sum(0), atol=1e-04)  # cf convergence gromov

        np.testing.assert_allclose(log2['srfgw_dist'], log['srfgw_dist'], atol=1e-07)
        np.testing.assert_allclose(logb2['srfgw_dist'], log['srfgw_dist'], atol=1e-07)
        np.testing.assert_allclose(srgw, srgw_, atol=1e-07)


@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_semirelaxed_fgw_dtype_device(nx):
    # setup
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    rng = np.random.RandomState(42)
    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)
    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        Mb, C1b, C2b, pb = nx.from_numpy(M, C1, C2, p, type_as=tp)

        for loss_fun in ['square_loss', 'kl_loss']:
            Gb = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
                Mb, C1b, C2b, pb, loss_fun, epsilon=0.1, verbose=True
            )
            fgw_valb = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(
                Mb, C1b, C2b, pb, loss_fun, epsilon=0.1, verbose=True
            )

            nx.assert_same_dtype_device(C1b, Gb)
            nx.assert_same_dtype_device(C1b, fgw_valb)


def test_not_implemented_solver():
    # test sinkhorn
    n_samples = 5  # nb samples
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=rng)
    xt = xs[::-1].copy()
    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()
    M = ot.dist(ys, yt)

    solver = 'not_implemented'
    # entropic gw and fgw
    with pytest.raises(ValueError):
        ot.gromov.entropic_gromov_wasserstein(
            C1, C2, p, q, 'square_loss', epsilon=1e-1, solver=solver)
    with pytest.raises(ValueError):
        ot.gromov.entropic_fused_gromov_wasserstein(
            M, C1, C2, p, q, 'square_loss', epsilon=1e-1, solver=solver)
