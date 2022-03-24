"""Tests for module bregman on OT with bregman projections """

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

from itertools import product

import numpy as np
import pytest

import ot
from ot.backend import torch, tf


@pytest.mark.parametrize("verbose, warn", product([True, False], [True, False]))
def test_sinkhorn(verbose, warn):
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.sinkhorn(u, u, M, 1, stopThr=1e-10, verbose=verbose, warn=warn)

    # check constraints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn

    with pytest.warns(UserWarning):
        ot.sinkhorn(u, u, M, 1, stopThr=0, numItermax=1)


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized",
                                    "sinkhorn_epsilon_scaling",
                                    "greenkhorn",
                                    "sinkhorn_log"])
def test_convergence_warning(method):
    # test sinkhorn
    n = 100
    a1 = ot.datasets.make_1D_gauss(n, m=30, s=10)
    a2 = ot.datasets.make_1D_gauss(n, m=40, s=10)
    A = np.asarray([a1, a2]).T
    M = ot.utils.dist0(n)

    with pytest.warns(UserWarning):
        ot.sinkhorn(a1, a2, M, 1., method=method, stopThr=0, numItermax=1)

    if method in ["sinkhorn", "sinkhorn_stabilized", "sinkhorn_log"]:
        with pytest.warns(UserWarning):
            ot.barycenter(A, M, 1, method=method, stopThr=0, numItermax=1)
        with pytest.warns(UserWarning):
            ot.sinkhorn2(a1, a2, M, 1, method=method, stopThr=0, numItermax=1)


def test_not_implemented_method():
    # test sinkhorn
    w = 10
    n = w ** 2
    rng = np.random.RandomState(42)
    A_img = rng.rand(2, w, w)
    A_flat = A_img.reshape(n, 2)
    a1, a2 = A_flat.T
    M_flat = ot.utils.dist0(n)
    not_implemented = "new_method"
    reg = 0.01
    with pytest.raises(ValueError):
        ot.sinkhorn(a1, a2, M_flat, reg, method=not_implemented)
    with pytest.raises(ValueError):
        ot.sinkhorn2(a1, a2, M_flat, reg, method=not_implemented)
    with pytest.raises(ValueError):
        ot.barycenter(A_flat, M_flat, reg, method=not_implemented)
    with pytest.raises(ValueError):
        ot.bregman.barycenter_debiased(A_flat, M_flat, reg,
                                       method=not_implemented)
    with pytest.raises(ValueError):
        ot.bregman.convolutional_barycenter2d(A_img, reg,
                                              method=not_implemented)
    with pytest.raises(ValueError):
        ot.bregman.convolutional_barycenter2d_debiased(A_img, reg,
                                                       method=not_implemented)


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized"])
def test_nan_warning(method):
    # test sinkhorn
    n = 100
    a1 = ot.datasets.make_1D_gauss(n, m=30, s=10)
    a2 = ot.datasets.make_1D_gauss(n, m=40, s=10)

    M = ot.utils.dist0(n)
    reg = 0
    with pytest.warns(UserWarning):
        # warn set to False to avoid catching a convergence warning instead
        ot.sinkhorn(a1, a2, M, reg, method=method, warn=False)


def test_sinkhorn_stabilization():
    # test sinkhorn
    n = 100
    a1 = ot.datasets.make_1D_gauss(n, m=30, s=10)
    a2 = ot.datasets.make_1D_gauss(n, m=40, s=10)
    M = ot.utils.dist0(n)
    reg = 1e-5
    loss1 = ot.sinkhorn2(a1, a2, M, reg, method="sinkhorn_log")
    loss2 = ot.sinkhorn2(a1, a2, M, reg, tau=1, method="sinkhorn_stabilized")
    np.testing.assert_allclose(
        loss1, loss2, atol=1e-06)  # cf convergence sinkhorn


@pytest.mark.parametrize("method, verbose, warn",
                         product(["sinkhorn", "sinkhorn_stabilized",
                                  "sinkhorn_log"],
                                 [True, False], [True, False]))
def test_sinkhorn_multi_b(method, verbose, warn):
    # test sinkhorn
    n = 10
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    b = rng.rand(n, 3)
    b = b / np.sum(b, 0, keepdims=True)

    M = ot.dist(x, x)

    loss0, log = ot.sinkhorn(u, b, M, .1, method=method, stopThr=1e-10,
                             log=True)

    loss = [ot.sinkhorn2(u, b[:, k], M, .1, method=method, stopThr=1e-10,
                         verbose=verbose, warn=warn) for k in range(3)]
    # check constraints
    np.testing.assert_allclose(
        loss0, loss, atol=1e-4)  # cf convergence sinkhorn


def test_sinkhorn_backends(nx):
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)

    M = ot.dist(x, y)

    G = ot.sinkhorn(a, a, M, 1)

    ab, M_nx = nx.from_numpy(a, M)

    Gb = ot.sinkhorn(ab, ab, M_nx, 1)

    np.allclose(G, nx.to_numpy(Gb))


def test_sinkhorn2_backends(nx):
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)

    M = ot.dist(x, y)

    G = ot.sinkhorn(a, a, M, 1)

    ab, M_nx = nx.from_numpy(a, M)

    Gb = ot.sinkhorn2(ab, ab, M_nx, 1)

    np.allclose(G, nx.to_numpy(Gb))


def test_sinkhorn2_gradients():
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)

    M = ot.dist(x, y)

    if torch:

        a1 = torch.tensor(a, requires_grad=True)
        b1 = torch.tensor(a, requires_grad=True)
        M1 = torch.tensor(M, requires_grad=True)

        val = ot.sinkhorn2(a1, b1, M1, 1)

        val.backward()

        assert a1.shape == a1.grad.shape
        assert b1.shape == b1.grad.shape
        assert M1.shape == M1.grad.shape


def test_sinkhorn_empty():
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G, log = ot.sinkhorn([], [], M, 1, stopThr=1e-10, method="sinkhorn_log",
                         verbose=True, log=True)
    # check constraints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)

    G, log = ot.sinkhorn([], [], M, 1, stopThr=1e-10, verbose=True, log=True)
    # check constraints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)

    G, log = ot.sinkhorn([], [], M, 1, stopThr=1e-10,
                         method='sinkhorn_stabilized', verbose=True, log=True)
    # check constraints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)

    G, log = ot.sinkhorn(
        [], [], M, 1, stopThr=1e-10, method='sinkhorn_epsilon_scaling',
        verbose=True, log=True)
    # check constraints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)

    # test empty weights greenkhorn
    ot.sinkhorn([], [], M, 1, method='greenkhorn', stopThr=1e-10, log=True)


@pytest.skip_backend('tf')
@pytest.skip_backend("jax")
def test_sinkhorn_variants(nx):
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    ub, M_nx = nx.from_numpy(u, M)

    G = ot.sinkhorn(u, u, M, 1, method='sinkhorn', stopThr=1e-10)
    Gl = nx.to_numpy(ot.sinkhorn(ub, ub, M_nx, 1, method='sinkhorn_log', stopThr=1e-10))
    G0 = nx.to_numpy(ot.sinkhorn(ub, ub, M_nx, 1, method='sinkhorn', stopThr=1e-10))
    Gs = nx.to_numpy(ot.sinkhorn(ub, ub, M_nx, 1, method='sinkhorn_stabilized', stopThr=1e-10))
    Ges = nx.to_numpy(ot.sinkhorn(
        ub, ub, M_nx, 1, method='sinkhorn_epsilon_scaling', stopThr=1e-10))
    G_green = nx.to_numpy(ot.sinkhorn(ub, ub, M_nx, 1, method='greenkhorn', stopThr=1e-10))

    # check values
    np.testing.assert_allclose(G, G0, atol=1e-05)
    np.testing.assert_allclose(G, Gl, atol=1e-05)
    np.testing.assert_allclose(G0, Gs, atol=1e-05)
    np.testing.assert_allclose(G0, Ges, atol=1e-05)
    np.testing.assert_allclose(G0, G_green, atol=1e-5)


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized",
                                    "sinkhorn_epsilon_scaling",
                                    "greenkhorn",
                                    "sinkhorn_log"])
@pytest.skip_arg(("nx", "method"), ("tf", "sinkhorn_epsilon_scaling"), reason="tf does not support sinkhorn_epsilon_scaling", getter=str)
@pytest.skip_arg(("nx", "method"), ("tf", "greenkhorn"), reason="tf does not support greenkhorn", getter=str)
@pytest.skip_arg(("nx", "method"), ("jax", "sinkhorn_epsilon_scaling"), reason="jax does not support sinkhorn_epsilon_scaling", getter=str)
@pytest.skip_arg(("nx", "method"), ("jax", "greenkhorn"), reason="jax does not support greenkhorn", getter=str)
def test_sinkhorn_variants_dtype_device(nx, method):
    n = 100

    x = np.random.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        ub, Mb = nx.from_numpy(u, M, type_as=tp)

        Gb = ot.sinkhorn(ub, ub, Mb, 1, method=method, stopThr=1e-10)

        nx.assert_same_dtype_device(Mb, Gb)


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized", "sinkhorn_log"])
def test_sinkhorn2_variants_dtype_device(nx, method):
    n = 100

    x = np.random.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        ub, Mb = nx.from_numpy(u, M, type_as=tp)

        lossb = ot.sinkhorn2(ub, ub, Mb, 1, method=method, stopThr=1e-10)

        nx.assert_same_dtype_device(Mb, lossb)


@pytest.mark.skipif(not tf, reason="tf not installed")
@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized", "sinkhorn_log"])
def test_sinkhorn2_variants_device_tf(method):
    nx = ot.backend.TensorflowBackend()
    n = 100
    x = np.random.randn(n, 2)
    u = ot.utils.unif(n)
    M = ot.dist(x, x)

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        ub, Mb = nx.from_numpy(u, M)
        Gb = ot.sinkhorn(ub, ub, Mb, 1, method=method, stopThr=1e-10)
        lossb = ot.sinkhorn2(ub, ub, Mb, 1, method=method, stopThr=1e-10)
        nx.assert_same_dtype_device(Mb, Gb)
        nx.assert_same_dtype_device(Mb, lossb)

    if len(tf.config.list_physical_devices('GPU')) > 0:
        # Check that everything happens on the GPU
        ub, Mb = nx.from_numpy(u, M)
        Gb = ot.sinkhorn(ub, ub, Mb, 1, method=method, stopThr=1e-10)
        lossb = ot.sinkhorn2(ub, ub, Mb, 1, method=method, stopThr=1e-10)
        nx.assert_same_dtype_device(Mb, Gb)
        nx.assert_same_dtype_device(Mb, lossb)
        assert nx.dtype_device(Gb)[1].startswith("GPU")


@pytest.skip_backend('tf')
@pytest.skip_backend("jax")
def test_sinkhorn_variants_multi_b(nx):
    # test sinkhorn
    n = 50
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    b = rng.rand(n, 3)
    b = b / np.sum(b, 0, keepdims=True)

    M = ot.dist(x, x)

    ub, bb, M_nx = nx.from_numpy(u, b, M)

    G = ot.sinkhorn(u, b, M, 1, method='sinkhorn', stopThr=1e-10)
    Gl = nx.to_numpy(ot.sinkhorn(ub, bb, M_nx, 1, method='sinkhorn_log', stopThr=1e-10))
    G0 = nx.to_numpy(ot.sinkhorn(ub, bb, M_nx, 1, method='sinkhorn', stopThr=1e-10))
    Gs = nx.to_numpy(ot.sinkhorn(ub, bb, M_nx, 1, method='sinkhorn_stabilized', stopThr=1e-10))

    # check values
    np.testing.assert_allclose(G, G0, atol=1e-05)
    np.testing.assert_allclose(G, Gl, atol=1e-05)
    np.testing.assert_allclose(G0, Gs, atol=1e-05)


@pytest.skip_backend('tf')
@pytest.skip_backend("jax")
def test_sinkhorn2_variants_multi_b(nx):
    # test sinkhorn
    n = 50
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    b = rng.rand(n, 3)
    b = b / np.sum(b, 0, keepdims=True)

    M = ot.dist(x, x)

    ub, bb, M_nx = nx.from_numpy(u, b, M)

    G = ot.sinkhorn2(u, b, M, 1, method='sinkhorn', stopThr=1e-10)
    Gl = nx.to_numpy(ot.sinkhorn2(ub, bb, M_nx, 1, method='sinkhorn_log', stopThr=1e-10))
    G0 = nx.to_numpy(ot.sinkhorn2(ub, bb, M_nx, 1, method='sinkhorn', stopThr=1e-10))
    Gs = nx.to_numpy(ot.sinkhorn2(ub, bb, M_nx, 1, method='sinkhorn_stabilized', stopThr=1e-10))

    # check values
    np.testing.assert_allclose(G, G0, atol=1e-05)
    np.testing.assert_allclose(G, Gl, atol=1e-05)
    np.testing.assert_allclose(G0, Gs, atol=1e-05)


def test_sinkhorn_variants_log():
    # test sinkhorn
    n = 50
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G0, log0 = ot.sinkhorn(u, u, M, 1, method='sinkhorn', stopThr=1e-10, log=True)
    Gl, logl = ot.sinkhorn(u, u, M, 1, method='sinkhorn_log', stopThr=1e-10, log=True)
    Gs, logs = ot.sinkhorn(u, u, M, 1, method='sinkhorn_stabilized', stopThr=1e-10, log=True)
    Ges, loges = ot.sinkhorn(
        u, u, M, 1, method='sinkhorn_epsilon_scaling', stopThr=1e-10, log=True,)
    G_green, loggreen = ot.sinkhorn(u, u, M, 1, method='greenkhorn', stopThr=1e-10, log=True)

    # check values
    np.testing.assert_allclose(G0, Gs, atol=1e-05)
    np.testing.assert_allclose(G0, Gl, atol=1e-05)
    np.testing.assert_allclose(G0, Ges, atol=1e-05)
    np.testing.assert_allclose(G0, G_green, atol=1e-5)


@pytest.mark.parametrize("verbose, warn", product([True, False], [True, False]))
def test_sinkhorn_variants_log_multib(verbose, warn):
    # test sinkhorn
    n = 50
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)
    b = rng.rand(n, 3)
    b = b / np.sum(b, 0, keepdims=True)

    M = ot.dist(x, x)

    G0, log0 = ot.sinkhorn(u, b, M, 1, method='sinkhorn', stopThr=1e-10, log=True)
    Gl, logl = ot.sinkhorn(u, b, M, 1, method='sinkhorn_log', stopThr=1e-10, log=True,
                           verbose=verbose, warn=warn)
    Gs, logs = ot.sinkhorn(u, b, M, 1, method='sinkhorn_stabilized', stopThr=1e-10, log=True,
                           verbose=verbose, warn=warn)

    # check values
    np.testing.assert_allclose(G0, Gs, atol=1e-05)
    np.testing.assert_allclose(G0, Gl, atol=1e-05)


@pytest.mark.parametrize("method, verbose, warn",
                         product(["sinkhorn", "sinkhorn_stabilized", "sinkhorn_log"],
                                 [True, False], [True, False]))
def test_barycenter(nx, method, verbose, warn):
    n_bins = 100  # nb bins

    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n_bins, m=30, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n_bins, m=40, s=10)

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T

    # loss matrix + normalization
    M = ot.utils.dist0(n_bins)
    M /= M.max()

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    A_nx, M_nx, weights_nx = nx.from_numpy(A, M, weights)
    reg = 1e-2

    if nx.__name__ in ("jax", "tf") and method == "sinkhorn_log":
        with pytest.raises(NotImplementedError):
            ot.bregman.barycenter(A_nx, M_nx, reg, weights, method=method)
    else:
        # wasserstein
        bary_wass_np = ot.bregman.barycenter(A, M, reg, weights, method=method, verbose=verbose, warn=warn)
        bary_wass, _ = ot.bregman.barycenter(A_nx, M_nx, reg, weights_nx, method=method, log=True)
        bary_wass = nx.to_numpy(bary_wass)

        np.testing.assert_allclose(1, np.sum(bary_wass))
        np.testing.assert_allclose(bary_wass, bary_wass_np)

        ot.bregman.barycenter(A_nx, M_nx, reg, log=True)


@pytest.mark.parametrize("method, verbose, warn",
                         product(["sinkhorn", "sinkhorn_log"],
                                 [True, False], [True, False]))
def test_barycenter_debiased(nx, method, verbose, warn):
    n_bins = 100  # nb bins

    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n_bins, m=30, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n_bins, m=40, s=10)

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T

    # loss matrix + normalization
    M = ot.utils.dist0(n_bins)
    M /= M.max()

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    A_nx, M_nx, weights_nx = nx.from_numpy(A, M, weights)

    # wasserstein
    reg = 1e-2
    if nx.__name__ in ("jax", "tf") and method == "sinkhorn_log":
        with pytest.raises(NotImplementedError):
            ot.bregman.barycenter_debiased(A_nx, M_nx, reg, weights, method=method)
    else:
        bary_wass_np = ot.bregman.barycenter_debiased(A, M, reg, weights, method=method,
                                                      verbose=verbose, warn=warn)
        bary_wass, _ = ot.bregman.barycenter_debiased(A_nx, M_nx, reg, weights_nx, method=method, log=True)
        bary_wass = nx.to_numpy(bary_wass)

        np.testing.assert_allclose(1, np.sum(bary_wass), atol=1e-3)
        np.testing.assert_allclose(bary_wass, bary_wass_np, atol=1e-5)

        ot.bregman.barycenter_debiased(A_nx, M_nx, reg, log=True, verbose=False)


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_log"])
def test_convergence_warning_barycenters(method):
    w = 10
    n_bins = w ** 2  # nb bins

    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n_bins, m=30, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n_bins, m=40, s=10)

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T
    A_img = A.reshape(2, w, w)
    A_img /= A_img.sum((1, 2))[:, None, None]

    # loss matrix + normalization
    M = ot.utils.dist0(n_bins)
    M /= M.max()

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])
    reg = 0.1
    with pytest.warns(UserWarning):
        ot.bregman.barycenter_debiased(A, M, reg, weights, method=method, numItermax=1)
    with pytest.warns(UserWarning):
        ot.bregman.barycenter(A, M, reg, weights, method=method, numItermax=1)
    with pytest.warns(UserWarning):
        ot.bregman.convolutional_barycenter2d(A_img, reg, weights,
                                              method=method, numItermax=1)
    with pytest.warns(UserWarning):
        ot.bregman.convolutional_barycenter2d_debiased(A_img, reg, weights,
                                                       method=method, numItermax=1)


def test_barycenter_stabilization(nx):
    n_bins = 100  # nb bins

    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n_bins, m=30, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n_bins, m=40, s=10)

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T

    # loss matrix + normalization
    M = ot.utils.dist0(n_bins)
    M /= M.max()

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    A_nx, M_nx, weights_b = nx.from_numpy(A, M, weights)

    # wasserstein
    reg = 1e-2
    bar_np = ot.bregman.barycenter(A, M, reg, weights, method="sinkhorn", stopThr=1e-8, verbose=True)
    bar_stable = nx.to_numpy(ot.bregman.barycenter(
        A_nx, M_nx, reg, weights_b, method="sinkhorn_stabilized",
        stopThr=1e-8, verbose=True
    ))
    bar = nx.to_numpy(ot.bregman.barycenter(
        A_nx, M_nx, reg, weights_b, method="sinkhorn",
        stopThr=1e-8, verbose=True
    ))
    np.testing.assert_allclose(bar, bar_stable)
    np.testing.assert_allclose(bar, bar_np)


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_log"])
def test_wasserstein_bary_2d(nx, method):
    size = 20  # size of a square image
    a1 = np.random.rand(size, size)
    a1 += a1.min()
    a1 = a1 / np.sum(a1)
    a2 = np.random.rand(size, size)
    a2 += a2.min()
    a2 = a2 / np.sum(a2)
    # creating matrix A containing all distributions
    A = np.zeros((2, size, size))
    A[0, :, :] = a1
    A[1, :, :] = a2

    A_nx = nx.from_numpy(A)

    # wasserstein
    reg = 1e-2
    if nx.__name__ in ("jax", "tf") and method == "sinkhorn_log":
        with pytest.raises(NotImplementedError):
            ot.bregman.convolutional_barycenter2d(A_nx, reg, method=method)
    else:
        bary_wass_np, log_np = ot.bregman.convolutional_barycenter2d(A, reg, method=method, verbose=True, log=True)
        bary_wass = nx.to_numpy(ot.bregman.convolutional_barycenter2d(A_nx, reg, method=method))

        np.testing.assert_allclose(1, np.sum(bary_wass), rtol=1e-3)
        np.testing.assert_allclose(bary_wass, bary_wass_np, atol=1e-3)

        # help in checking if log and verbose do not bug the function
        ot.bregman.convolutional_barycenter2d(A, reg, log=True, verbose=True)


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_log"])
def test_wasserstein_bary_2d_debiased(nx, method):
    size = 20  # size of a square image
    a1 = np.random.rand(size, size)
    a1 += a1.min()
    a1 = a1 / np.sum(a1)
    a2 = np.random.rand(size, size)
    a2 += a2.min()
    a2 = a2 / np.sum(a2)
    # creating matrix A containing all distributions
    A = np.zeros((2, size, size))
    A[0, :, :] = a1
    A[1, :, :] = a2

    A_nx = nx.from_numpy(A)

    # wasserstein
    reg = 1e-2
    if nx.__name__ in ("jax", "tf") and method == "sinkhorn_log":
        with pytest.raises(NotImplementedError):
            ot.bregman.convolutional_barycenter2d_debiased(A_nx, reg, method=method)
    else:
        bary_wass_np, log_np = ot.bregman.convolutional_barycenter2d_debiased(A, reg, method=method, verbose=True, log=True)
        bary_wass = nx.to_numpy(ot.bregman.convolutional_barycenter2d_debiased(A_nx, reg, method=method))

        np.testing.assert_allclose(1, np.sum(bary_wass), rtol=1e-3)
        np.testing.assert_allclose(bary_wass, bary_wass_np, atol=1e-3)

        # help in checking if log and verbose do not bug the function
        ot.bregman.convolutional_barycenter2d(A, reg, log=True, verbose=True)


def test_unmix(nx):
    n_bins = 50  # nb bins

    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n_bins, m=20, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n_bins, m=40, s=10)

    a = ot.datasets.make_1D_gauss(n_bins, m=30, s=10)

    # creating matrix A containing all distributions
    D = np.vstack((a1, a2)).T

    # loss matrix + normalization
    M = ot.utils.dist0(n_bins)
    M /= M.max()

    M0 = ot.utils.dist0(2)
    M0 /= M0.max()
    h0 = ot.unif(2)

    ab, Db, M_nx, M0b, h0b = nx.from_numpy(a, D, M, M0, h0)

    # wasserstein
    reg = 1e-3
    um_np = ot.bregman.unmix(a, D, M, M0, h0, reg, 1, alpha=0.01)
    um = nx.to_numpy(ot.bregman.unmix(ab, Db, M_nx, M0b, h0b, reg, 1, alpha=0.01))

    np.testing.assert_allclose(1, np.sum(um), rtol=1e-03, atol=1e-03)
    np.testing.assert_allclose([0.5, 0.5], um, rtol=1e-03, atol=1e-03)
    np.testing.assert_allclose(um, um_np)

    ot.bregman.unmix(ab, Db, M_nx, M0b, h0b, reg,
                     1, alpha=0.01, log=True, verbose=True)


def test_empirical_sinkhorn(nx):
    # test sinkhorn
    n = 10
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(0, n), (n, 1))
    M = ot.dist(X_s, X_t)
    M_m = ot.dist(X_s, X_t, metric='euclidean')

    ab, bb, X_sb, X_tb, M_nx, M_mb = nx.from_numpy(a, b, X_s, X_t, M, M_m)

    G_sqe = nx.to_numpy(ot.bregman.empirical_sinkhorn(X_sb, X_tb, 1))
    sinkhorn_sqe = nx.to_numpy(ot.sinkhorn(ab, bb, M_nx, 1))

    G_log, log_es = ot.bregman.empirical_sinkhorn(X_sb, X_tb, 0.1, log=True)
    G_log = nx.to_numpy(G_log)
    sinkhorn_log, log_s = ot.sinkhorn(ab, bb, M_nx, 0.1, log=True)
    sinkhorn_log = nx.to_numpy(sinkhorn_log)

    G_m = nx.to_numpy(ot.bregman.empirical_sinkhorn(X_sb, X_tb, 1, metric='euclidean'))
    sinkhorn_m = nx.to_numpy(ot.sinkhorn(ab, bb, M_mb, 1))

    loss_emp_sinkhorn = nx.to_numpy(ot.bregman.empirical_sinkhorn2(X_sb, X_tb, 1))
    loss_sinkhorn = nx.to_numpy(ot.sinkhorn2(ab, bb, M_nx, 1))

    # check constraints
    np.testing.assert_allclose(
        sinkhorn_sqe.sum(1), G_sqe.sum(1), atol=1e-05)  # metric sqeuclidian
    np.testing.assert_allclose(
        sinkhorn_sqe.sum(0), G_sqe.sum(0), atol=1e-05)  # metric sqeuclidian
    np.testing.assert_allclose(
        sinkhorn_log.sum(1), G_log.sum(1), atol=1e-05)  # log
    np.testing.assert_allclose(
        sinkhorn_log.sum(0), G_log.sum(0), atol=1e-05)  # log
    np.testing.assert_allclose(
        sinkhorn_m.sum(1), G_m.sum(1), atol=1e-05)  # metric euclidian
    np.testing.assert_allclose(
        sinkhorn_m.sum(0), G_m.sum(0), atol=1e-05)  # metric euclidian
    np.testing.assert_allclose(loss_emp_sinkhorn, loss_sinkhorn, atol=1e-05)


def test_lazy_empirical_sinkhorn(nx):
    # test sinkhorn
    n = 10
    a = ot.unif(n)
    b = ot.unif(n)
    numIterMax = 1000

    X_s = np.reshape(np.arange(n, dtype=np.float64), (n, 1))
    X_t = np.reshape(np.arange(0, n, dtype=np.float64), (n, 1))
    M = ot.dist(X_s, X_t)
    M_m = ot.dist(X_s, X_t, metric='euclidean')

    ab, bb, X_sb, X_tb, M_nx, M_mb = nx.from_numpy(a, b, X_s, X_t, M, M_m)

    f, g = ot.bregman.empirical_sinkhorn(X_sb, X_tb, 1, numIterMax=numIterMax, isLazy=True, batchSize=(1, 3), verbose=True)
    f, g = nx.to_numpy(f), nx.to_numpy(g)
    G_sqe = np.exp(f[:, None] + g[None, :] - M / 1)
    sinkhorn_sqe = nx.to_numpy(ot.sinkhorn(ab, bb, M_nx, 1))

    f, g, log_es = ot.bregman.empirical_sinkhorn(X_sb, X_tb, 0.1, numIterMax=numIterMax, isLazy=True, batchSize=1, log=True)
    f, g = nx.to_numpy(f), nx.to_numpy(g)
    G_log = np.exp(f[:, None] + g[None, :] - M / 0.1)
    sinkhorn_log, log_s = ot.sinkhorn(ab, bb, M_nx, 0.1, log=True)
    sinkhorn_log = nx.to_numpy(sinkhorn_log)

    f, g = ot.bregman.empirical_sinkhorn(X_sb, X_tb, 1, metric='euclidean', numIterMax=numIterMax, isLazy=True, batchSize=1)
    f, g = nx.to_numpy(f), nx.to_numpy(g)
    G_m = np.exp(f[:, None] + g[None, :] - M_m / 1)
    sinkhorn_m = nx.to_numpy(ot.sinkhorn(ab, bb, M_mb, 1))

    loss_emp_sinkhorn, log = ot.bregman.empirical_sinkhorn2(X_sb, X_tb, 1, numIterMax=numIterMax, isLazy=True, batchSize=1, log=True)
    loss_emp_sinkhorn = nx.to_numpy(loss_emp_sinkhorn)
    loss_sinkhorn = nx.to_numpy(ot.sinkhorn2(ab, bb, M_nx, 1))

    # check constraints
    np.testing.assert_allclose(
        sinkhorn_sqe.sum(1), G_sqe.sum(1), atol=1e-05)  # metric sqeuclidian
    np.testing.assert_allclose(
        sinkhorn_sqe.sum(0), G_sqe.sum(0), atol=1e-05)  # metric sqeuclidian
    np.testing.assert_allclose(
        sinkhorn_log.sum(1), G_log.sum(1), atol=1e-05)  # log
    np.testing.assert_allclose(
        sinkhorn_log.sum(0), G_log.sum(0), atol=1e-05)  # log
    np.testing.assert_allclose(
        sinkhorn_m.sum(1), G_m.sum(1), atol=1e-05)  # metric euclidian
    np.testing.assert_allclose(
        sinkhorn_m.sum(0), G_m.sum(0), atol=1e-05)  # metric euclidian
    np.testing.assert_allclose(loss_emp_sinkhorn, loss_sinkhorn, atol=1e-05)


def test_empirical_sinkhorn_divergence(nx):
    # Test sinkhorn divergence
    n = 10
    a = np.linspace(1, n, n)
    a /= a.sum()
    b = ot.unif(n)
    X_s = np.reshape(np.arange(n, dtype=np.float64), (n, 1))
    X_t = np.reshape(np.arange(0, n * 2, 2, dtype=np.float64), (n, 1))
    M = ot.dist(X_s, X_t)
    M_s = ot.dist(X_s, X_s)
    M_t = ot.dist(X_t, X_t)

    ab, bb, X_sb, X_tb, M_nx, M_sb, M_tb = nx.from_numpy(a, b, X_s, X_t, M, M_s, M_t)

    emp_sinkhorn_div = nx.to_numpy(ot.bregman.empirical_sinkhorn_divergence(X_sb, X_tb, 1, a=ab, b=bb))
    sinkhorn_div = nx.to_numpy(
        ot.sinkhorn2(ab, bb, M_nx, 1)
        - 1 / 2 * ot.sinkhorn2(ab, ab, M_sb, 1)
        - 1 / 2 * ot.sinkhorn2(bb, bb, M_tb, 1)
    )
    emp_sinkhorn_div_np = ot.bregman.empirical_sinkhorn_divergence(X_s, X_t, 1, a=a, b=b)

    # check constraints
    np.testing.assert_allclose(emp_sinkhorn_div, emp_sinkhorn_div_np, atol=1e-05)
    np.testing.assert_allclose(
        emp_sinkhorn_div, sinkhorn_div, atol=1e-05)  # cf conv emp sinkhorn

    ot.bregman.empirical_sinkhorn_divergence(X_sb, X_tb, 1, a=ab, b=bb, log=True)


def test_stabilized_vs_sinkhorn_multidim(nx):
    # test if stable version matches sinkhorn
    # for multidimensional inputs
    n = 100

    # Gaussian distributions
    a = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    b1 = ot.datasets.make_1D_gauss(n, m=60, s=8)
    b2 = ot.datasets.make_1D_gauss(n, m=30, s=4)

    # creating matrix A containing all distributions
    b = np.vstack((b1, b2)).T

    M = ot.utils.dist0(n)
    M /= np.median(M)
    epsilon = 0.1

    ab, bb, M_nx = nx.from_numpy(a, b, M)

    G_np, _ = ot.bregman.sinkhorn(a, b, M, reg=epsilon, method="sinkhorn", log=True)
    G, log = ot.bregman.sinkhorn(ab, bb, M_nx, reg=epsilon,
                                 method="sinkhorn_stabilized",
                                 log=True)
    G = nx.to_numpy(G)
    G2, log2 = ot.bregman.sinkhorn(ab, bb, M_nx, epsilon,
                                   method="sinkhorn", log=True)
    G2 = nx.to_numpy(G2)

    np.testing.assert_allclose(G_np, G2)
    np.testing.assert_allclose(G, G2)


def test_implemented_methods():
    IMPLEMENTED_METHODS = ['sinkhorn', 'sinkhorn_stabilized']
    ONLY_1D_methods = ['greenkhorn', 'sinkhorn_epsilon_scaling']
    NOT_VALID_TOKENS = ['foo']
    # test generalized sinkhorn for unbalanced OT barycenter
    n = 3
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = ot.utils.unif(n)
    A = rng.rand(n, 2)
    A /= A.sum(0, keepdims=True)
    M = ot.dist(x, x)
    epsilon = 1.0

    for method in IMPLEMENTED_METHODS:
        ot.bregman.sinkhorn(a, b, M, epsilon, method=method)
        ot.bregman.sinkhorn2(a, b, M, epsilon, method=method)
        ot.bregman.barycenter(A, M, reg=epsilon, method=method)
    with pytest.raises(ValueError):
        for method in set(NOT_VALID_TOKENS):
            ot.bregman.sinkhorn(a, b, M, epsilon, method=method)
            ot.bregman.sinkhorn2(a, b, M, epsilon, method=method)
            ot.bregman.barycenter(A, M, reg=epsilon, method=method)
    for method in ONLY_1D_methods:
        ot.bregman.sinkhorn(a, b, M, epsilon, method=method)
        with pytest.raises(ValueError):
            ot.bregman.sinkhorn2(a, b, M, epsilon, method=method)


@pytest.skip_backend('tf')
@pytest.skip_backend("cupy")
@pytest.skip_backend("jax")
@pytest.mark.filterwarnings("ignore:Bottleneck")
def test_screenkhorn(nx):
    # test screenkhorn
    rng = np.random.RandomState(0)
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    x = rng.randn(n, 2)
    M = ot.dist(x, x)

    ab, bb, M_nx = nx.from_numpy(a, b, M)

    # sinkhorn
    G_sink = nx.to_numpy(ot.sinkhorn(ab, bb, M_nx, 1e-1))
    # screenkhorn
    G_screen = nx.to_numpy(ot.bregman.screenkhorn(ab, bb, M_nx, 1e-1, uniform=True, verbose=True))
    # check marginals
    np.testing.assert_allclose(G_sink.sum(0), G_screen.sum(0), atol=1e-02)
    np.testing.assert_allclose(G_sink.sum(1), G_screen.sum(1), atol=1e-02)


def test_convolutional_barycenter_non_square(nx):
    # test for image with height not equal width
    A = np.ones((2, 2, 3)) / (2 * 3)
    A_nx = nx.from_numpy(A)

    b_np = ot.bregman.convolutional_barycenter2d(A, 1e-03)
    b = nx.to_numpy(ot.bregman.convolutional_barycenter2d(A_nx, 1e-03))

    np.testing.assert_allclose(np.ones((2, 3)) / (2 * 3), b, atol=1e-02)
    np.testing.assert_allclose(np.ones((2, 3)) / (2 * 3), b, atol=1e-02)
    np.testing.assert_allclose(b, b_np)
