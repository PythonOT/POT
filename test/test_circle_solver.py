"""Tests for module Circle Wasserstein solver"""

# Author: Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

import numpy as np
import pytest

import ot


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
        # print(nx.dtype_device(tp))

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
        # print(nx.dtype_device(tp))

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


@pytest.skip_backend("tf")
def test_linear_circular_ot_devices(nx):
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

        lcot = ot.linear_circular_ot(xb, xb, rho_ub, rho_vb)

        nx.assert_same_dtype_device(xb, lcot)


def test_linear_circular_ot_bad_shape():
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.rand(n, 2)
    v = rng.rand(m, 1)

    with pytest.raises(ValueError):
        _ = ot.linear_circular_ot(u, v)


def test_linear_circular_ot_same_dist():
    n = 20
    rng = np.random.RandomState(0)
    u = rng.rand(n)

    lcot = ot.linear_circular_ot(u, u)
    np.testing.assert_almost_equal(lcot, 0.0)


def test_linear_circular_ot_different_dist():
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.rand(n)
    v = rng.rand(m)

    lcot = ot.linear_circular_ot(u, v)
    assert lcot > 0.0


def test_linear_circular_embedding_shape():
    n = 20
    rng = np.random.RandomState(0)
    u = rng.rand(n, 2)

    ts = np.linspace(0, 1, 101)[:-1]

    emb = ot.lp.solver_circle.linear_circular_embedding(ts, u)
    assert emb.shape == (100, 2)

    emb = ot.lp.solver_circle.linear_circular_embedding(ts, u[:, 0])
    assert emb.shape == (100, 1)


def test_linear_circular_ot_unif_circle():
    n = 20
    m = 1000

    rng = np.random.RandomState(0)
    u = rng.rand(
        n,
    )
    v = rng.rand(
        m,
    )

    lcot = ot.linear_circular_ot(u, v)
    lcot_unif = ot.linear_circular_ot(u)

    # check loss is similar
    np.testing.assert_allclose(lcot, lcot_unif, atol=1e-2)
