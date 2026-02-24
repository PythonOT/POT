"""Tests for module sliced"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty <ncourty@irisa.fr>
#         Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.sliced import get_random_projections
from ot.backend import tf, torch
from contextlib import nullcontext


def test_env():
    import sys

    print(sys.executable)


# def test_backend():
#    import os
#    print("DISABLE:", os.environ.get("POT_BACKEND_DISABLE_PYTORCH"))
#    import torch
#    print("TORCH:", torch.__version__)


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


def test_max_sliced_dim_check():
    n = 3
    x = np.zeros((n, 2))
    y = np.zeros((n + 1, 3))
    with pytest.raises(ValueError):
        _ = ot.max_sliced_wasserstein_distance(x, y, n_projections=10)


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

    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, 2 * n)
    b /= b.sum()
    a_b = nx.from_numpy(a)
    b_b = nx.from_numpy(b)
    val = ot.sliced_wasserstein_distance(x, y, a=a, b=b, projections=P)
    val_b = ot.sliced_wasserstein_distance(xb, yb, a=a_b, b=b_b, projections=Pb)
    np.testing.assert_almost_equal(val, nx.to_numpy(val_b))


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

    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, 2 * n)
    b /= b.sum()
    a_b = nx.from_numpy(a)
    b_b = nx.from_numpy(b)
    val = ot.max_sliced_wasserstein_distance(x, y, a=a, b=b, projections=P)
    val_b = ot.max_sliced_wasserstein_distance(xb, yb, a=a_b, b=b_b, projections=Pb)
    np.testing.assert_almost_equal(val, nx.to_numpy(val_b))


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

    rng = np.random.RandomState(0)

    projections = ot.sliced.get_projections_sphere(3, n_projs, seed=rng)
    projections_T = np.transpose(projections, [0, 2, 1])

    np.testing.assert_almost_equal(
        np.matmul(projections_T, projections),
        np.array([np.eye(2) for k in range(n_projs)]),
    )

    # np.testing.assert_almost_equal(projections, P)


def test_projections_sphere_to_circle():
    rng = np.random.RandomState(0)

    n_projs = 500
    x = rng.randn(100, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    x_projs, _ = ot.sliced.projection_sphere_to_circle(x, n_projs)
    assert x_projs.shape == (n_projs, 100)
    assert np.all(x_projs >= 0) and np.all(x_projs < 1)


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

    u = ot.utils.unif(n)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    # dimension problem
    y = rng.randn(n, 4)
    with pytest.raises(ValueError):
        _ = ot.sliced_wasserstein_sphere(x, y, u, u, 10, seed=rng)

    # not on the sphere
    y = rng.randn(n, 3)
    with pytest.raises(ValueError):
        _ = ot.sliced_wasserstein_sphere(x, y, u, u, 10, seed=rng)

    with pytest.raises(ValueError):
        _ = ot.sliced_wasserstein_sphere(y, x, u, u, 10, seed=rng)


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


def test_linear_sliced_sphere_same_dist():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))
    u = ot.utils.unif(n)

    res = ot.linear_sliced_wasserstein_sphere(x, x, u, u, 10, seed=rng)
    np.testing.assert_almost_equal(res, 0.0)


def test_linear_sliced_sphere_same_proj():
    n_projections = 10
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    y = rng.randn(n, 3)
    y = y / np.sqrt(np.sum(y**2, -1, keepdims=True))

    seed = 42

    cost1, log1 = ot.linear_sliced_wasserstein_sphere(
        x, y, seed=seed, n_projections=n_projections, log=True
    )
    cost2, log2 = ot.linear_sliced_wasserstein_sphere(
        x, y, seed=seed, n_projections=n_projections, log=True
    )

    assert np.allclose(log1["projections"], log2["projections"])
    assert np.isclose(cost1, cost2)


def test_linear_sliced_sphere_bad_shapes():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    y = rng.randn(n, 4)
    y = y / np.sqrt(np.sum(x**2, -1, keepdims=True))

    u = ot.utils.unif(n)

    with pytest.raises(ValueError):
        _ = ot.linear_sliced_wasserstein_sphere(x, y, u, u, 10, seed=rng)


def test_linear_sliced_sphere_values_on_the_sphere():
    n = 100
    rng = np.random.RandomState(0)

    u = ot.utils.unif(n)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    # shape problem
    y = rng.randn(n, 4)

    with pytest.raises(ValueError):
        _ = ot.linear_sliced_wasserstein_sphere(x, y, u, u, 10, seed=rng)

    # not on sphere
    y = rng.randn(n, 3)

    with pytest.raises(ValueError):
        _ = ot.linear_sliced_wasserstein_sphere(x, y, u, u, 10, seed=rng)

    with pytest.raises(ValueError):
        _ = ot.linear_sliced_wasserstein_sphere(y, x, u, u, 10, seed=rng)


def test_linear_sliced_sphere_log():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 4)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))
    y = rng.randn(n, 4)
    y = y / np.sqrt(np.sum(y**2, -1, keepdims=True))
    u = ot.utils.unif(n)

    res, log = ot.linear_sliced_wasserstein_sphere(x, y, u, u, 10, seed=rng, log=True)
    assert len(log) == 2
    projections = log["projections"]
    projected_emds = log["projected_emds"]

    assert projections.shape[0] == len(projected_emds) == 10
    for emd in projected_emds:
        assert emd > 0


def test_linear_sliced_sphere_different_dists():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    u = ot.utils.unif(n)
    y = rng.randn(n, 3)
    y = y / np.sqrt(np.sum(y**2, -1, keepdims=True))

    res = ot.linear_sliced_wasserstein_sphere(x, y, u, u, 10, seed=rng)
    assert res > 0.0


def test_1d_linear_sliced_sphere_equals_emd():
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

    res = ot.linear_sliced_wasserstein_sphere(x, y, a, u, 100, seed=42)
    expected = ot.linear_circular_ot(x_coords.T, y_coords.T, a, u)

    np.testing.assert_almost_equal(res**2, expected, decimal=5)


@pytest.skip_backend("tf")
def test_linear_sliced_sphere_backend_type_devices(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 3)
    x = x / np.sqrt(np.sum(x**2, -1, keepdims=True))

    y = rng.randn(2 * n, 3)
    y = y / np.sqrt(np.sum(y**2, -1, keepdims=True))

    sw_np, log = ot.linear_sliced_wasserstein_sphere(x, y, log=True)
    P = log["projections"]

    for tp in nx.__type_list__:
        xb, yb = nx.from_numpy(x, y, type_as=tp)

        valb = ot.linear_sliced_wasserstein_sphere(
            xb, yb, projections=nx.from_numpy(P, type_as=tp)
        )

        nx.assert_same_dtype_device(xb, valb)
        np.testing.assert_almost_equal(sw_np, nx.to_numpy(valb))


def test_sliced_permutations():
    n = 4
    n_proj = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(n, 2)

    thetas = ot.sliced.get_random_projections(d, n_proj, seed=0).T

    # test without provided thetas
    _, _ = ot.sliced.sliced_plans(x, y, n_proj=n_proj)

    # test with invalid shapes
    with pytest.raises(AssertionError):
        ot.sliced.sliced_plans(x[:, 1:], y, thetas=thetas)


def test_sliced_plans():
    x = [1, 2]
    with pytest.raises(AssertionError):
        ot.sliced.min_pivot_sliced(x, x, n_proj=2)

    n = 4
    m = 5
    n_proj = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(m, 2)

    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    thetas = ot.sliced.get_random_projections(d, n_proj, seed=0).T

    # test with a and b not uniform
    ot.sliced.sliced_plans(x, y, a, b, thetas=thetas, dense=True)

    # test with the minkowski metric
    ot.sliced.sliced_plans(x, y, thetas=thetas, metric="minkowski")

    # test with an unsupported metric
    with pytest.raises(AssertionError):
        ot.sliced.sliced_plans(x, y, thetas=thetas, metric="mahalanobis")

    # test with a warm theta
    ot.sliced.sliced_plans(x, y, n_proj=10, warm_theta=thetas[-1])

    # test permutations
    n = 5
    m = 5
    n_proj = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(m, 2)

    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    # test with the minkowski metric
    ot.sliced.sliced_plans(x, y, n_proj=10, metric="minkowski")


def test_min_pivot_sliced():
    x = [1, 2]
    with pytest.raises(AssertionError):
        ot.sliced.min_pivot_sliced(x, x, n_proj=3)

    n = 10
    m = 4
    n_proj = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(m, 2)
    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    thetas = ot.sliced.get_random_projections(d, n_proj, seed=0).T

    # identity of the indiscernibles
    _, min_cost = ot.min_pivot_sliced(x, x, a, a, n_proj=10)
    np.testing.assert_almost_equal(min_cost, 0.0)

    _, min_cost = ot.sliced.min_pivot_sliced(x, y, a, b, thetas=thetas, dense=True)

    # result should be an upper-bound of W2 and relatively close
    w2 = ot.emd2(a, b, ot.dist(x, y))
    print("w2:", w2, "min_cost:", min_cost)
    assert min_cost >= w2
    assert min_cost <= 1.5 * w2

    # test without provided thetas
    ot.sliced.min_pivot_sliced(x, y, a, b, n_proj=n_proj, log=True)

    # test with invalid shapes
    with pytest.raises(AssertionError):
        ot.sliced.min_pivot_sliced(x[:, 1:], y, thetas=thetas)

    # test the logs
    _, min_cost, log = ot.sliced.min_pivot_sliced(
        x, y, a, b, thetas=thetas, dense=False, log=True
    )
    assert len(log) == 5
    costs = log["costs"]
    assert len(costs) == thetas.shape[0]
    assert len(log["min_theta"]) == d
    assert (log["thetas"] == thetas).all()
    for c in costs:
        assert c > 0

    # test with different metrics
    ot.sliced.min_pivot_sliced(x, y, thetas=thetas, metric="minkowski")
    ot.sliced.min_pivot_sliced(x, y, thetas=thetas, metric="euclidean")
    ot.sliced.min_pivot_sliced(x, y, thetas=thetas, metric="cityblock")

    # test with an unsupported metric
    with pytest.raises(AssertionError):
        ot.sliced.min_pivot_sliced(x, y, thetas=thetas, metric="mahalanobis")

    # test with a warm theta
    ot.sliced.min_pivot_sliced(x, y, n_proj=10, warm_theta=thetas[-1])


def test_expected_sliced():
    x = [1, 2]
    with pytest.raises(AssertionError):
        ot.sliced.min_pivot_sliced(x, x, n_proj=2)

    n = 10
    m = 24
    n_proj = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(m, 2)
    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    thetas = ot.sliced.get_random_projections(d, n_proj, seed=0).T

    _, expected_cost = ot.sliced.expected_sliced(x, y, a, b, dense=True, thetas=thetas)
    # result should be a coarse upper-bound of W2
    w2 = ot.emd2(a, b, ot.dist(x, y))
    assert expected_cost >= w2
    assert expected_cost <= 3 * w2

    # test without provided thetas
    ot.sliced.expected_sliced(x, y, n_proj=n_proj, log=True)
    ot.sliced.expected_sliced(x, y, a, b, n_proj=n_proj, log=True)

    # test with invalid shapes
    with pytest.raises(AssertionError):
        ot.sliced.min_pivot_sliced(x[:, 1:], y, thetas=thetas)

    # with a small temperature (i.e. large beta), the cost should be close
    # to min_pivot
    _, expected_cost = ot.sliced.expected_sliced(
        x, y, a, b, thetas=thetas, dense=True, beta=100.0
    )
    _, min_cost = ot.sliced.min_pivot_sliced(x, y, a, b, thetas=thetas, dense=True)
    np.testing.assert_almost_equal(expected_cost, min_cost, decimal=3)

    # test the logs
    _, min_cost, log = ot.sliced.expected_sliced(
        x, y, a, b, thetas=thetas, dense=False, log=True
    )
    assert len(log) == 4
    costs = log["costs"]
    assert len(costs) == thetas.shape[0]
    assert len(log["weights"]) == thetas.shape[0]
    assert (log["thetas"] == thetas).all()
    for c in costs:
        assert c > 0

    # test with the minkowski metric
    ot.sliced.expected_sliced(x, y, thetas=thetas, metric="minkowski")

    # test with an unsupported metric
    with pytest.raises(AssertionError):
        ot.sliced.expected_sliced(x, y, thetas=thetas, metric="mahalanobis")


def test_sliced_plans_backends(nx):
    n = 10
    m = 24
    n_proj = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(m, 2)
    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    x_b, y_b, a_b, b_b = nx.from_numpy(x, y, a, b)

    thetas_b = ot.sliced.get_random_projections(
        d, n_proj, seed=0, backend=nx, type_as=x_b
    ).T
    thetas = nx.to_numpy(thetas_b)

    context = (
        nullcontext()
        if str(nx) not in ["tf", "jax"]
        else pytest.raises(NotImplementedError)
    )

    with context:
        _, expected_cost_b = ot.sliced.expected_sliced(
            x_b, y_b, a_b, b_b, dense=True, thetas=thetas_b
        )
        # result should be the same than numpy version
        _, expected_cost = ot.sliced.expected_sliced(
            x, y, a, b, dense=True, thetas=thetas
        )
        np.testing.assert_almost_equal(expected_cost_b, expected_cost)

    # for min_pivot
    _, min_cost_b = ot.sliced.min_pivot_sliced(
        x_b, y_b, a_b, b_b, dense=True, thetas=thetas_b
    )
    # result should be the same than numpy version
    _, min_cost = ot.sliced.min_pivot_sliced(x, y, a, b, dense=True, thetas=thetas)
    np.testing.assert_almost_equal(min_cost_b, min_cost)

    # for thetas
    thetas_b = ot.sliced.get_random_projections(
        d, n_proj, seed=0, backend=nx, type_as=x_b
    ).T

    # test with the minkowski metric
    ot.sliced.min_pivot_sliced(x_b, y_b, thetas=thetas_b, metric="minkowski")
