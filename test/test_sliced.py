"""Tests for module sliced"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.sliced import get_random_projections


def test_get_random_projections():
    rng = np.random.RandomState(0)
    projections = get_random_projections(1000, 50, rng)
    np.testing.assert_almost_equal(np.sum(projections ** 2, 0), 1.)


def test_sliced_same_dist():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    res = ot.sliced_wasserstein_distance(x, x, u, u, 10, seed=rng)
    np.testing.assert_almost_equal(res, 0.)


def test_sliced_bad_shapes():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(n, 4)
    u = ot.utils.unif(n)

    with pytest.raises(ValueError):
        _ = ot.sliced_wasserstein_distance(x, y, u, u, 10, seed=rng)


def test_sliced_log():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 4)
    y = rng.randn(n, 4)
    u = ot.utils.unif(n)

    res, log = ot.sliced_wasserstein_distance(x, y, u, u, 10, p=1, seed=rng, log=True)
    assert len(log) == 2
    projections = log["projections"]
    projected_emds = log["projected_emds"]

    assert projections.shape[1] == len(projected_emds) == 10
    for emd in projected_emds:
        assert emd > 0


def test_sliced_different_dists():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)
    y = rng.randn(n, 2)

    res = ot.sliced_wasserstein_distance(x, y, u, u, 10, seed=rng)
    assert res > 0.


def test_1d_sliced_equals_emd():
    n = 100
    m = 120
    rng = np.random.RandomState(0)

    x = rng.randn(n, 1)
    a = rng.uniform(0, 1, n)
    a /= a.sum()
    y = rng.randn(m, 1)
    u = ot.utils.unif(m)
    res = ot.sliced_wasserstein_distance(x, y, a, u, 10, seed=42)
    expected = ot.emd2_1d(x.squeeze(), y.squeeze(), a, u)
    np.testing.assert_almost_equal(res ** 2, expected)


def test_max_sliced_same_dist():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    res = ot.max_sliced_wasserstein_distance(x, x, u, u, 10, seed=rng)
    np.testing.assert_almost_equal(res, 0.)


def test_max_sliced_different_dists():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)
    y = rng.randn(n, 2)

    res, log = ot.max_sliced_wasserstein_distance(x, y, u, u, 10, seed=rng, log=True)
    assert res > 0.


def test_sliced_backend(nx):

    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)

    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    n_projections = 20

    xb = nx.from_numpy(x)
    yb = nx.from_numpy(y)
    Pb = nx.from_numpy(P)

    val0 = ot.sliced_wasserstein_distance(x, y, projections=P)

    val = ot.sliced_wasserstein_distance(xb, yb, n_projections=n_projections, seed=0)
    val2 = ot.sliced_wasserstein_distance(xb, yb, n_projections=n_projections, seed=0)

    assert val > 0
    assert val == val2

    valb = nx.to_numpy(ot.sliced_wasserstein_distance(xb, yb, projections=Pb))

    assert np.allclose(val0, valb)


def test_sliced_backend_type_devices(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)

    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        xb = nx.from_numpy(x, type_as=tp)
        yb = nx.from_numpy(y, type_as=tp)
        Pb = nx.from_numpy(P, type_as=tp)

        valb = ot.sliced_wasserstein_distance(xb, yb, projections=Pb)

        nx.assert_same_dtype_device(xb, valb)


def test_max_sliced_backend(nx):

    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)

    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    n_projections = 20

    xb = nx.from_numpy(x)
    yb = nx.from_numpy(y)
    Pb = nx.from_numpy(P)

    val0 = ot.max_sliced_wasserstein_distance(x, y, projections=P)

    val = ot.max_sliced_wasserstein_distance(xb, yb, n_projections=n_projections, seed=0)
    val2 = ot.max_sliced_wasserstein_distance(xb, yb, n_projections=n_projections, seed=0)

    assert val > 0
    assert val == val2

    valb = nx.to_numpy(ot.max_sliced_wasserstein_distance(xb, yb, projections=Pb))

    assert np.allclose(val0, valb)


def test_max_sliced_backend_type_devices(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)

    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        xb = nx.from_numpy(x, type_as=tp)
        yb = nx.from_numpy(y, type_as=tp)
        Pb = nx.from_numpy(P, type_as=tp)

        valb = ot.max_sliced_wasserstein_distance(xb, yb, projections=Pb)

        nx.assert_same_dtype_device(xb, valb)
