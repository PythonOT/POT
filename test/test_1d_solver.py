"""Tests for module 1d Wasserstein solver"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.backend import tf
from ot.lp import wasserstein_1d

from scipy.stats import wasserstein_distance


def test_emd_1d_emd2_1d_with_weights():
    # test emd1d gives similar results as emd
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.randn(n, 1)
    v = rng.randn(m, 1)

    w_u = rng.uniform(0.0, 1.0, n)
    w_u = w_u / w_u.sum()

    w_v = rng.uniform(0.0, 1.0, m)
    w_v = w_v / w_v.sum()

    M = ot.dist(u, v, metric="sqeuclidean")

    G, log = ot.emd(w_u, w_v, M, log=True)
    wass = log["cost"]
    G_1d, log = ot.emd_1d(u, v, w_u, w_v, metric="sqeuclidean", log=True)
    wass1d = log["cost"]
    wass1d_emd2 = ot.emd2_1d(u, v, w_u, w_v, metric="sqeuclidean", log=False)
    wass1d_euc = ot.emd2_1d(u, v, w_u, w_v, metric="euclidean", log=False)

    # check loss is similar
    np.testing.assert_allclose(wass, wass1d)
    np.testing.assert_allclose(wass, wass1d_emd2)

    # check loss is similar to scipy's implementation for Euclidean metric
    wass_sp = wasserstein_distance(u.reshape((-1,)), v.reshape((-1,)), w_u, w_v)
    np.testing.assert_allclose(wass_sp, wass1d_euc)

    # check constraints
    np.testing.assert_allclose(w_u, G.sum(1))
    np.testing.assert_allclose(w_v, G.sum(0))

    # check that an error is raised if the metric is not a Minkowski one
    np.testing.assert_raises(ValueError, ot.emd_1d, u, v, w_u, w_v, metric="cosine")
    np.testing.assert_raises(ValueError, ot.emd2_1d, u, v, w_u, w_v, metric="cosine")


def test_wasserstein_1d(nx):
    rng = np.random.RandomState(0)

    n = 100
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)

    # test 1 : wasserstein_1d should be close to scipy W_1 implementation
    np.testing.assert_almost_equal(
        wasserstein_1d(xb, xb, rho_ub, rho_vb, p=1),
        wasserstein_distance(x, x, rho_u, rho_v),
    )

    # test 2 : wasserstein_1d should be close to one when only translating the support
    np.testing.assert_almost_equal(wasserstein_1d(xb, xb + 1, p=2), 1.0)

    # test 3 : arrays test
    X = np.stack((np.linspace(0, 5, n), np.linspace(0, 5, n) * 10), -1)
    Xb = nx.from_numpy(X)
    res = wasserstein_1d(Xb, Xb, rho_ub, rho_vb, p=2)
    np.testing.assert_almost_equal(100 * res[0], res[1], decimal=4)


def test_wasserstein_1d_type_devices(nx):
    rng = np.random.RandomState(0)

    n = 10
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v, type_as=tp)

        res = wasserstein_1d(xb, xb, rho_ub, rho_vb, p=1)

        nx.assert_same_dtype_device(xb, res)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_wasserstein_1d_device_tf():
    nx = ot.backend.TensorflowBackend()
    rng = np.random.RandomState(0)
    n = 10
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)
        res = wasserstein_1d(xb, xb, rho_ub, rho_vb, p=1)
        nx.assert_same_dtype_device(xb, res)

    if len(tf.config.list_physical_devices("GPU")) > 0:
        # Check that everything happens on the GPU
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)
        res = wasserstein_1d(xb, xb, rho_ub, rho_vb, p=1)
        nx.assert_same_dtype_device(xb, res)
        assert nx.dtype_device(res)[1].startswith("GPU")


def test_emd_1d_emd2_1d():
    # test emd1d gives similar results as emd
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.randn(n, 1)
    v = rng.randn(m, 1)

    M = ot.dist(u, v, metric="sqeuclidean")

    G, log = ot.emd([], [], M, log=True)
    wass = log["cost"]
    G_1d, log = ot.emd_1d(u, v, [], [], metric="sqeuclidean", log=True)
    wass1d = log["cost"]
    wass1d_emd2 = ot.emd2_1d(u, v, [], [], metric="sqeuclidean", log=False)
    wass1d_euc = ot.emd2_1d(u, v, [], [], metric="euclidean", log=False)

    # check loss is similar
    np.testing.assert_allclose(wass, wass1d)
    np.testing.assert_allclose(wass, wass1d_emd2)

    # check loss is similar to scipy's implementation for Euclidean metric
    wass_sp = wasserstein_distance(u.reshape((-1,)), v.reshape((-1,)))
    np.testing.assert_allclose(wass_sp, wass1d_euc)

    # check constraints
    np.testing.assert_allclose(np.ones((n,)) / n, G.sum(1))
    np.testing.assert_allclose(np.ones((m,)) / m, G.sum(0))

    # check G is similar
    np.testing.assert_allclose(G, G_1d, atol=1e-15)

    # check AssertionError is raised if called on non 1d arrays
    u = rng.randn(n, 2)
    v = rng.randn(m, 2)
    with pytest.raises(AssertionError):
        ot.emd_1d(u, v, [], [])


def test_emd1d_type_devices(nx):
    rng = np.random.RandomState(0)

    n = 10
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v, type_as=tp)

        emd = ot.emd_1d(xb, xb, rho_ub, rho_vb)
        emd2 = ot.emd2_1d(xb, xb, rho_ub, rho_vb)

        nx.assert_same_dtype_device(xb, emd)
        nx.assert_same_dtype_device(xb, emd2)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_emd1d_device_tf():
    nx = ot.backend.TensorflowBackend()
    rng = np.random.RandomState(0)
    n = 10
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)
        emd = ot.emd_1d(xb, xb, rho_ub, rho_vb)
        emd2 = ot.emd2_1d(xb, xb, rho_ub, rho_vb)
        nx.assert_same_dtype_device(xb, emd)
        nx.assert_same_dtype_device(xb, emd2)

    if len(tf.config.list_physical_devices("GPU")) > 0:
        # Check that everything happens on the GPU
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)
        emd = ot.emd_1d(xb, xb, rho_ub, rho_vb)
        emd2 = ot.emd2_1d(xb, xb, rho_ub, rho_vb)
        nx.assert_same_dtype_device(xb, emd)
        nx.assert_same_dtype_device(xb, emd2)
        assert nx.dtype_device(emd)[1].startswith("GPU")


def test_wasserstein_1d_circle():
    # test binary_search_circle and wasserstein_circle give similar results as emd
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.rand(
        n,
    )
    v = rng.rand(
        m,
    )

    w_u = rng.uniform(0.0, 1.0, n)
    w_u = w_u / w_u.sum()

    w_v = rng.uniform(0.0, 1.0, m)
    w_v = w_v / w_v.sum()

    M1 = np.minimum(np.abs(u[:, None] - v[None]), 1 - np.abs(u[:, None] - v[None]))

    wass1 = ot.emd2(w_u, w_v, M1)

    wass1_bsc = ot.binary_search_circle(u, v, w_u, w_v, p=1)
    w1_circle = ot.wasserstein_circle(u, v, w_u, w_v, p=1)

    M2 = M1**2
    wass2 = ot.emd2(w_u, w_v, M2)
    wass2_bsc = ot.binary_search_circle(u, v, w_u, w_v, p=2)
    w2_circle = ot.wasserstein_circle(u, v, w_u, w_v, p=2)

    # check loss is similar
    np.testing.assert_allclose(wass1, wass1_bsc)
    np.testing.assert_allclose(wass1, w1_circle, rtol=1e-2)
    np.testing.assert_allclose(wass2, wass2_bsc)
    np.testing.assert_allclose(wass2, w2_circle)


@pytest.skip_backend("tf")
def test_wasserstein1d_circle_devices(nx):
    rng = np.random.RandomState(0)

    n = 10
    x = np.linspace(0, 1, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v, type_as=tp)

        w1 = ot.wasserstein_circle(xb, xb, rho_ub, rho_vb, p=1)
        w2_bsc = ot.wasserstein_circle(xb, xb, rho_ub, rho_vb, p=2)

        nx.assert_same_dtype_device(xb, w1)
        nx.assert_same_dtype_device(xb, w2_bsc)


def test_wasserstein_1d_unif_circle():
    # test semidiscrete_wasserstein2_unif_circle versus wasserstein_circle
    n = 20
    m = 1000

    rng = np.random.RandomState(0)
    u = rng.rand(
        n,
    )
    v = rng.rand(
        m,
    )

    # w_u = rng.uniform(0., 1., n)
    # w_u = w_u / w_u.sum()

    w_u = ot.utils.unif(n)
    w_v = ot.utils.unif(m)

    M1 = np.minimum(np.abs(u[:, None] - v[None]), 1 - np.abs(u[:, None] - v[None]))
    wass2 = ot.emd2(w_u, w_v, M1**2)

    wass2_circle = ot.wasserstein_circle(u, v, w_u, w_v, p=2, eps=1e-15)
    wass2_unif_circle = ot.semidiscrete_wasserstein2_unif_circle(u, w_u)

    # check loss is similar
    np.testing.assert_allclose(wass2, wass2_unif_circle, atol=1e-2)
    np.testing.assert_allclose(wass2_circle, wass2_unif_circle, atol=1e-2)


def test_wasserstein1d_unif_circle_devices(nx):
    rng = np.random.RandomState(0)

    n = 10
    x = np.linspace(0, 1, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        xb, rho_ub = nx.from_numpy(x, rho_u, type_as=tp)

        w2 = ot.semidiscrete_wasserstein2_unif_circle(xb, rho_ub)

        nx.assert_same_dtype_device(xb, w2)


def test_binary_search_circle_log():
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.rand(
        n,
    )
    v = rng.rand(
        m,
    )

    wass2_bsc, log = ot.binary_search_circle(u, v, p=2, log=True)
    optimal_thetas = log["optimal_theta"]

    assert optimal_thetas.shape[0] == 1


def test_wasserstein_circle_bad_shape():
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.rand(n, 2)
    v = rng.rand(m, 1)

    with pytest.raises(ValueError):
        _ = ot.wasserstein_circle(u, v, p=2)

    with pytest.raises(ValueError):
        _ = ot.wasserstein_circle(u, v, p=1)
