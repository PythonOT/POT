""" Tests for gromov._gw.py """

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

    def line_search(cost, G, deltaG, Mi, cost_G, df_G):
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

    # test edge cases for gw barycenters:
    # C1 as list
    with pytest.raises(ValueError):
        C1_list = [list(c) for c in C1]
        _ = ot.gromov.gromov_barycenters(
            n_samples, [C1_list], None, p, None, 'square_loss', max_iter=10,
            tol=1e-3, stop_criterion=stop_criterion, verbose=False,
            random_state=42
        )

    # p1, p2 as lists
    with pytest.raises(ValueError):
        p1_list = list(p1)
        p2_list = list(p2)
        _ = ot.gromov.gromov_barycenters(
            n_samples, [C1, C2], [p1_list, p2_list], p, None, 'square_loss', max_iter=10,
            tol=1e-3, stop_criterion=stop_criterion, verbose=False,
            random_state=42
        )

    # unique input structure
    Cb = ot.gromov.gromov_barycenters(
        n_samples, [C1], None, p, None, 'square_loss', max_iter=10,
        tol=1e-3, stop_criterion=stop_criterion, verbose=False,
        random_state=42
    )
    Cbb = nx.to_numpy(ot.gromov.gromov_barycenters(
        n_samples, [C1b], None, None, [1.], 'square_loss',
        max_iter=10, tol=1e-3, stop_criterion=stop_criterion,
        verbose=False, random_state=42
    ))
    np.testing.assert_allclose(Cb, Cbb, atol=1e-06)
    np.testing.assert_allclose(Cbb.shape, (n_samples, n_samples))


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

    def line_search(cost, G, deltaG, Mi, cost_G, df_G):
        return ot.gromov.solve_gromov_linesearch(G, deltaG, cost_G, C1b, C2b, M=(1 - alpha) * Mb, reg=alpha, nx=None)
    # feed the precomputed local optimum Gb to cg
    res, log = ot.optim.cg(pb, qb, (1 - alpha) * Mb, alpha, f, df, Gb, line_search, log=True, numItermax=1e4, stopThr=1e-9, stopThr2=1e-9)

    def line_search(cost, G, deltaG, Mi, cost_G, df_G):
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
    recovered_Cb = ot.gromov.update_barycenter_structure(
        logb['Ts_iter'][-1], Csb, lambdas, pb, target=False, check_zeros=True)
    recovered_Xb = ot.gromov.update_barycenter_feature(
        logb['Ts_iter'][-1], Ysb, lambdas, pb, target=False, check_zeros=True)

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
            max_iter=10, tol=1e-3, stop_criterion=stop_criterion, init_C=C,
            init_X=X, warmstartT=True, random_state=12345, log=True
        )

    for stop_criterion in ['barycenter', 'loss']:
        X, C, log = ot.gromov.fgw_barycenters(
            n_samples, [ys, yt], [C1, C2], [p1, p2], [.5, .5], 0.5,
            fixed_structure=False, fixed_features=False, p=p, loss_fun='kl_loss',
            max_iter=10, tol=1e-3, stop_criterion=stop_criterion, init_C=C,
            init_X=X, warmstartT=True, random_state=12345, log=True, verbose=True
        )
        np.testing.assert_allclose(C.shape, (n_samples, n_samples))
        np.testing.assert_allclose(X.shape, (n_samples, ys.shape[1]))

    # test correspondance with utils function
    recovered_C = ot.gromov.update_barycenter_structure(
        log['T'], [C1, C2], lambdas, p, loss_fun='kl_loss',
        target=False, check_zeros=False)

    np.testing.assert_allclose(C, recovered_C)

    # test edge cases for fgw barycenters:
    # C1 as list
    with pytest.raises(ValueError):
        C1b_list = [list(c) for c in C1b]
        _, _, _ = ot.gromov.fgw_barycenters(
            n_samples, [ysb], [C1b_list], [p1b], None, 0.5,
            fixed_structure=False, fixed_features=False, p=pb, loss_fun='square_loss',
            max_iter=10, tol=1e-3, stop_criterion=stop_criterion, init_C=Cb,
            init_X=Xb, warmstartT=True, random_state=12345, log=True, verbose=True
        )

    # p1, p2 as lists
    with pytest.raises(ValueError):
        p1_list = list(p1)
        p2_list = list(p2)
        _, _, _ = ot.gromov.fgw_barycenters(
            n_samples, [ysb, ytb], [C1b, C2b], [p1_list, p2_list], None, 0.5,
            fixed_structure=False, fixed_features=False, p=p, loss_fun='kl_loss',
            max_iter=10, tol=1e-3, stop_criterion=stop_criterion, init_C=Cb,
            init_X=Xb, warmstartT=True, random_state=12345, log=True, verbose=True
        )

    # unique input structure
    X, C = ot.gromov.fgw_barycenters(
        n_samples, [ys], [C1], [p1], None, 0.5,
        fixed_structure=False, fixed_features=False, p=p, loss_fun='square_loss',
        max_iter=10, tol=1e-3, stop_criterion=stop_criterion,
        warmstartT=True, random_state=12345, log=False, verbose=False
    )
    Xb, Cb = ot.gromov.fgw_barycenters(
        n_samples, [ysb], [C1b], [p1b], [1.], 0.5,
        fixed_structure=False, fixed_features=False, p=pb, loss_fun='square_loss',
        max_iter=10, tol=1e-3, stop_criterion=stop_criterion,
        warmstartT=True, random_state=12345, log=False, verbose=False
    )

    np.testing.assert_allclose(C, Cb, atol=1e-06)
    np.testing.assert_allclose(X, Xb, atol=1e-06)
