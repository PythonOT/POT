"""Tests for module sliced"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.sliced import get_random_projections
from ot.backend import tf, torch


def test_get_random_projections():
    rng = np.random.RandomState(0)
    projections = get_random_projections(1000, 50, rng)
    np.testing.assert_almost_equal(np.sum(projections**2, 0), 1.0)


def test_sliced_same_dist():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    res = ot.sliced_wasserstein_distance(x, x, u, u, 10, seed=rng)
    np.testing.assert_almost_equal(res, 0.0)


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
    assert res > 0.0


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
    np.testing.assert_almost_equal(res**2, expected)


def test_max_sliced_same_dist():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    res = ot.max_sliced_wasserstein_distance(x, x, u, u, 10, seed=rng)
    np.testing.assert_almost_equal(res, 0.0)


def test_max_sliced_different_dists():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)
    y = rng.randn(n, 2)

    res, log = ot.max_sliced_wasserstein_distance(x, y, u, u, 10, seed=rng, log=True)
    assert res > 0.0


def test_sliced_same_proj():
    n_projections = 10
    seed = 12
    rng = np.random.RandomState(0)
    X = rng.randn(8, 2)
    Y = rng.randn(8, 2)
    cost1, log1 = ot.sliced_wasserstein_distance(
        X, Y, seed=seed, n_projections=n_projections, log=True
    )
    P = get_random_projections(X.shape[1], n_projections=10, seed=seed)
    cost2, log2 = ot.sliced_wasserstein_distance(X, Y, projections=P, log=True)

    assert np.allclose(log1["projections"], log2["projections"])
    assert np.isclose(cost1, cost2)


def test_sliced_backend(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)

    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    n_projections = 20

    xb, yb, Pb = nx.from_numpy(x, y, P)

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

        xb, yb, Pb = nx.from_numpy(x, y, P, type_as=tp)

        valb = ot.sliced_wasserstein_distance(xb, yb, projections=Pb)

        nx.assert_same_dtype_device(xb, valb)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_sliced_backend_device_tf():
    nx = ot.backend.TensorflowBackend()
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)
    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        xb, yb, Pb = nx.from_numpy(x, y, P)
        valb = ot.sliced_wasserstein_distance(xb, yb, projections=Pb)
        nx.assert_same_dtype_device(xb, valb)

    if len(tf.config.list_physical_devices("GPU")) > 0:
        # Check that everything happens on the GPU
        xb, yb, Pb = nx.from_numpy(x, y, P)
        valb = ot.sliced_wasserstein_distance(xb, yb, projections=Pb)
        nx.assert_same_dtype_device(xb, valb)
        assert nx.dtype_device(valb)[1].startswith("GPU")


def test_max_sliced_backend(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)

    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    n_projections = 20

    xb, yb, Pb = nx.from_numpy(x, y, P)

    val0 = ot.max_sliced_wasserstein_distance(x, y, projections=P)

    val = ot.max_sliced_wasserstein_distance(
        xb, yb, n_projections=n_projections, seed=0
    )
    val2 = ot.max_sliced_wasserstein_distance(
        xb, yb, n_projections=n_projections, seed=0
    )

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

        xb, yb, Pb = nx.from_numpy(x, y, P, type_as=tp)

        valb = ot.max_sliced_wasserstein_distance(xb, yb, projections=Pb)

        nx.assert_same_dtype_device(xb, valb)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_max_sliced_backend_device_tf():
    nx = ot.backend.TensorflowBackend()
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)
    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        xb, yb, Pb = nx.from_numpy(x, y, P)
        valb = ot.max_sliced_wasserstein_distance(xb, yb, projections=Pb)
        nx.assert_same_dtype_device(xb, valb)

    if len(tf.config.list_physical_devices("GPU")) > 0:
        # Check that everything happens on the GPU
        xb, yb, Pb = nx.from_numpy(x, y, P)
        valb = ot.max_sliced_wasserstein_distance(xb, yb, projections=Pb)
        nx.assert_same_dtype_device(xb, valb)
        assert nx.dtype_device(valb)[1].startswith("GPU")


def test_projections_stiefel():
    rng = np.random.RandomState(0)

    n_projs = 500
    x = rng.randn(100, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    ssw, log = ot.sliced_wasserstein_sphere(
        x, x, n_projections=n_projs, seed=rng, log=True
    )

    P = log["projections"]
    P_T = np.transpose(P, [0, 2, 1])
    np.testing.assert_almost_equal(
        np.matmul(P_T, P), np.array([np.eye(2) for k in range(n_projs)])
    )


def test_sliced_sphere_same_dist():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))
    u = ot.utils.unif(n)

    res = ot.sliced_wasserstein_sphere(x, x, u, u, 10, seed=rng)
    np.testing.assert_almost_equal(res, 0.0)


def test_sliced_sphere_same_proj():
    n_projections = 10
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    y = rng.randn(n, 3)
    y = y / np.sqrt(np.sum(y**2, -1, keepdims=True))

    seed = 42

    cost1, log1 = ot.sliced_wasserstein_sphere(
        x, y, seed=seed, n_projections=n_projections, log=True
    )
    cost2, log2 = ot.sliced_wasserstein_sphere(
        x, y, seed=seed, n_projections=n_projections, log=True
    )

    assert np.allclose(log1["projections"], log2["projections"])
    assert np.isclose(cost1, cost2)


def test_sliced_sphere_bad_shapes():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    y = rng.randn(n, 4)
    y = y / np.sqrt(np.sum(x**2, -1, keepdims=True))

    u = ot.utils.unif(n)

    with pytest.raises(ValueError):
        _ = ot.sliced_wasserstein_sphere(x, y, u, u, 10, seed=rng)


def test_sliced_sphere_values_on_the_sphere():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    y = rng.randn(n, 4)

    u = ot.utils.unif(n)

    with pytest.raises(ValueError):
        _ = ot.sliced_wasserstein_sphere(x, y, u, u, 10, seed=rng)


def test_sliced_sphere_log():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 4)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))
    y = rng.randn(n, 4)
    y = y / np.sqrt(np.sum(y**2, -1, keepdims=True))
    u = ot.utils.unif(n)

    res, log = ot.sliced_wasserstein_sphere(x, y, u, u, 10, p=1, seed=rng, log=True)
    assert len(log) == 2
    projections = log["projections"]
    projected_emds = log["projected_emds"]

    assert projections.shape[0] == len(projected_emds) == 10
    for emd in projected_emds:
        assert emd > 0


def test_sliced_sphere_different_dists():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    u = ot.utils.unif(n)
    y = rng.randn(n, 3)
    y = y / np.sqrt(np.sum(y**2, -1, keepdims=True))

    res = ot.sliced_wasserstein_sphere(x, y, u, u, 10, seed=rng)
    assert res > 0.0


def test_1d_sliced_sphere_equals_emd():
    n = 100
    m = 120
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))
    x_coords = (np.arctan2(-x[:, 1], -x[:, 0]) + np.pi) / (2 * np.pi)
    a = rng.uniform(0, 1, n)
    a /= a.sum()

    y = rng.randn(m, 2)
    y = y / np.sqrt(np.sum(y**2, -1, keepdims=True))
    y_coords = (np.arctan2(-y[:, 1], -y[:, 0]) + np.pi) / (2 * np.pi)
    u = ot.utils.unif(m)

    res = ot.sliced_wasserstein_sphere(x, y, a, u, 10, seed=42, p=2)
    expected = ot.binary_search_circle(x_coords.T, y_coords.T, a, u, p=2)

    res1 = ot.sliced_wasserstein_sphere(x, y, a, u, 10, seed=42, p=1)
    expected1 = ot.binary_search_circle(x_coords.T, y_coords.T, a, u, p=1)

    np.testing.assert_almost_equal(res**2, expected)
    np.testing.assert_almost_equal(res1, expected1, decimal=3)


@pytest.skip_backend("tf")
def test_sliced_sphere_backend_type_devices(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    y = rng.randn(2 * n, 3)
    y = y / np.sqrt(np.sum(y**2, -1, keepdims=True))

    sw_np, log = ot.sliced_wasserstein_sphere(x, y, log=True)
    P = log["projections"]

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        xb, yb = nx.from_numpy(x, y, type_as=tp)

        valb = ot.sliced_wasserstein_sphere(
            xb, yb, projections=nx.from_numpy(P, type_as=tp)
        )

        nx.assert_same_dtype_device(xb, valb)
        np.testing.assert_almost_equal(sw_np, nx.to_numpy(valb))


def test_sliced_sphere_gradient():
    if torch:
        import torch.nn.functional as F

        X0 = torch.randn((20, 3))
        X0 = F.normalize(X0, p=2, dim=-1)
        X0.requires_grad_(True)

        X1 = torch.randn((20, 3))
        X1 = F.normalize(X1, p=2, dim=-1)

        sw = ot.sliced_wasserstein_sphere(X1, X0, n_projections=100, p=2)
        grad_x0 = torch.autograd.grad(sw, X0)[0]

        assert not torch.any(torch.isnan(grad_x0))


def test_sliced_sphere_unif_values_on_the_sphere():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    u = ot.utils.unif(n)

    with pytest.raises(ValueError):
        _ = ot.sliced_wasserstein_sphere_unif(x, u, 10, seed=rng)


def test_sliced_sphere_unif_log():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 4)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))
    u = ot.utils.unif(n)

    res, log = ot.sliced_wasserstein_sphere_unif(x, u, 10, seed=rng, log=True)
    assert len(log) == 2
    projections = log["projections"]
    projected_emds = log["projected_emds"]

    assert projections.shape[0] == len(projected_emds) == 10
    for emd in projected_emds:
        assert emd > 0


def test_sliced_sphere_unif_backend_type_devices(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        xb = nx.from_numpy(x, type_as=tp)

        valb = ot.sliced_wasserstein_sphere_unif(xb)

        nx.assert_same_dtype_device(xb, valb)
