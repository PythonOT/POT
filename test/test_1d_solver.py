"""Tests for module 1d Wasserstein solver"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.lp import wasserstein_1d

from ot.backend import get_backend_list, tf
from scipy.stats import wasserstein_distance

backend_list = get_backend_list()


def test_emd_1d_emd2_1d_with_weights():
    # test emd1d gives similar results as emd
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.randn(n, 1)
    v = rng.randn(m, 1)

    w_u = rng.uniform(0., 1., n)
    w_u = w_u / w_u.sum()

    w_v = rng.uniform(0., 1., m)
    w_v = w_v / w_v.sum()

    M = ot.dist(u, v, metric='sqeuclidean')

    G, log = ot.emd(w_u, w_v, M, log=True)
    wass = log["cost"]
    G_1d, log = ot.emd_1d(u, v, w_u, w_v, metric='sqeuclidean', log=True)
    wass1d = log["cost"]
    wass1d_emd2 = ot.emd2_1d(u, v, w_u, w_v, metric='sqeuclidean', log=False)
    wass1d_euc = ot.emd2_1d(u, v, w_u, w_v, metric='euclidean', log=False)

    # check loss is similar
    np.testing.assert_allclose(wass, wass1d)
    np.testing.assert_allclose(wass, wass1d_emd2)

    # check loss is similar to scipy's implementation for Euclidean metric
    wass_sp = wasserstein_distance(u.reshape((-1,)), v.reshape((-1,)), w_u, w_v)
    np.testing.assert_allclose(wass_sp, wass1d_euc)

    # check constraints
    np.testing.assert_allclose(w_u, G.sum(1))
    np.testing.assert_allclose(w_v, G.sum(0))


@pytest.mark.parametrize('nx', backend_list)
def test_wasserstein_1d(nx):
    from scipy.stats import wasserstein_distance

    rng = np.random.RandomState(0)

    n = 100
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)

    # test 1 : wasserstein_1d should be close to scipy W_1 implementation
    np.testing.assert_almost_equal(wasserstein_1d(xb, xb, rho_ub, rho_vb, p=1),
                                   wasserstein_distance(x, x, rho_u, rho_v))

    # test 2 : wasserstein_1d should be close to one when only translating the support
    np.testing.assert_almost_equal(wasserstein_1d(xb, xb + 1, p=2),
                                   1.)

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
    if not tf:
        return
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

    if len(tf.config.list_physical_devices('GPU')) > 0:
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

    M = ot.dist(u, v, metric='sqeuclidean')

    G, log = ot.emd([], [], M, log=True)
    wass = log["cost"]
    G_1d, log = ot.emd_1d(u, v, [], [], metric='sqeuclidean', log=True)
    wass1d = log["cost"]
    wass1d_emd2 = ot.emd2_1d(u, v, [], [], metric='sqeuclidean', log=False)
    wass1d_euc = ot.emd2_1d(u, v, [], [], metric='euclidean', log=False)

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
    u = np.random.randn(n, 2)
    v = np.random.randn(m, 2)
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

    if len(tf.config.list_physical_devices('GPU')) > 0:
        # Check that everything happens on the GPU
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)
        emd = ot.emd_1d(xb, xb, rho_ub, rho_vb)
        emd2 = ot.emd2_1d(xb, xb, rho_ub, rho_vb)
        nx.assert_same_dtype_device(xb, emd)
        nx.assert_same_dtype_device(xb, emd2)
        assert nx.dtype_device(emd)[1].startswith("GPU")
